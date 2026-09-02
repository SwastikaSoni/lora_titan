"""
AODV-lite — reactive routing for Titan DMS bake-off.

Routes are discovered on-demand via RREQ/RREP flooding.  No HELLO beacons,
no expanding-ring search — kept minimal for LoRa's low throughput.

Route lifecycle:
  1. App wants to send to dst_addr → get_next_hop()
  2. No route? → buffer the request, emit RREQ (broadcast)
  3. Each intermediate node rebroadcasts RREQ (with dedup)
  4. Destination (or node with fresh route) replies with RREP (unicast back)
  5. RREP installs forward route at each hop
  6. Original sender now has a next_hop → FORWARD
  7. Route expires after route_timeout_s of inactivity
  8. Link break detected → RERR flooded toward source

Control frame encoding:
  All control frames use MessageClass.CTRL with a msgpack payload dict:
    {"type": "RREQ"|"RREP"|"RERR", ...fields...}

Config keys (via scenarios.yaml -> config dict):
  default_ttl:        int,   max hops (default 5)
  route_timeout_s:    float, route expiry if unused (default 60.0)
  rreq_retry_s:       float, time before re-issuing RREQ (default 5.0)
  rreq_max_retries:   int,   give up after this many (default 3)
  dedup_cache_size:    int,   (default 256)
  dedup_expiry_s:      float, (default 30.0)
  neighbour_timeout:   float, (default 30.0)
  pending_buf_size:    int,   max frames buffered waiting for route (default 16)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import List, Optional, NamedTuple

from .frame import Frame, MessageClass, BROADCAST_ID, encode_payload, decode_payload
from .routing import (
    IRoutingScheme,
    RoutingAction,
    ActionType,
)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class _RouteEntry(NamedTuple):
    next_hop: int
    hop_count: int
    dst_seq: int       # destination sequence number (freshness)
    last_used: float   # sim-time of last use


class _RreqState:
    """Tracks an outstanding RREQ we originated."""
    __slots__ = ("dst_addr", "rreq_seq", "retries", "last_sent")

    def __init__(self, dst_addr: int, rreq_seq: int, now: float) -> None:
        self.dst_addr = dst_addr
        self.rreq_seq = rreq_seq
        self.retries = 0
        self.last_sent = now


class _DedupEntry:
    __slots__ = ("rx_time",)
    def __init__(self, rx_time: float) -> None:
        self.rx_time = rx_time


class _PendingFrame:
    __slots__ = ("dst_addr", "queued_time")
    def __init__(self, dst_addr: int, queued_time: float) -> None:
        self.dst_addr = dst_addr
        self.queued_time = queued_time


# ---------------------------------------------------------------------------
# AODV-lite routing
# ---------------------------------------------------------------------------

class AodvRouting(IRoutingScheme):

    # Control frame type tags
    RREQ = "RREQ"
    RREP = "RREP"
    RERR = "RERR"

    def __init__(
        self,
        node_addr: int,
        bs_addr: int = 0x0001,
        is_bs: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(node_addr, bs_addr, is_bs, config)

        self._default_ttl: int = self.config.get("default_ttl", 5)
        self._route_timeout: float = self.config.get("route_timeout_s", 60.0)
        self._rreq_retry: float = self.config.get("rreq_retry_s", 5.0)
        self._rreq_max_retries: int = self.config.get("rreq_max_retries", 3)
        self._cache_size: int = self.config.get("dedup_cache_size", 256)
        self._dedup_expiry: float = self.config.get("dedup_expiry_s", 30.0)
        self._nb_timeout: float = self.config.get("neighbour_timeout", 30.0)
        self._pending_buf_size: int = self.config.get("pending_buf_size", 16)

        # Routing table: dst_addr -> _RouteEntry
        self._routes: dict[int, _RouteEntry] = {}

        # RREQ dedup: (src_id, rreq_seq) -> _DedupEntry
        self._dedup: OrderedDict[tuple[int, int], _DedupEntry] = OrderedDict()

        # Data frame dedup (same as flood): (src_id, seq) -> _DedupEntry
        self._data_dedup: OrderedDict[tuple[int, int], _DedupEntry] = OrderedDict()

        # Outstanding RREQs we originated: dst_addr -> _RreqState
        self._pending_rreqs: dict[int, _RreqState] = {}

        # Frames waiting for route discovery: list of (dst_addr, queued_time)
        self._pending_buf: list[_PendingFrame] = []

        # Control frames to emit (filled by on_tick / on_frame_received)
        self._outbox: list[Frame] = []

        # Sequence counters
        self._seq: int = 0         # data frame seq
        self._rreq_seq: int = 0    # RREQ id

    # ------------------------------------------------------------------
    # ABC: on_frame_received
    # ------------------------------------------------------------------

    def on_frame_received(
        self,
        frame: Frame,
        rssi_dbm: float,
        snr_db: float,
        rx_time: float,
    ) -> RoutingAction:
        self.update_neighbour(frame.prev_hop_id, rssi_dbm, snr_db, rx_time)

        # Check if this is a control frame (RREQ/RREP/RERR)
        if frame.message_class == MessageClass.CTRL:
            return self._handle_control(frame, rssi_dbm, snr_db, rx_time)

        # --- Data frame handling ---

        # Data dedup
        data_key = (frame.src_id, frame.seq)
        if data_key in self._data_dedup:
            return RoutingAction(ActionType.DROP, reason="duplicate_data")
        self._cache_insert(self._data_dedup, data_key, rx_time)

        # Learn reverse route to source (for RREP path)
        self._learn_route(
            dst=frame.src_id,
            next_hop=frame.prev_hop_id,
            hop_count=frame.hop_count + 1,
            dst_seq=frame.seq,
            now=rx_time,
        )

        # For us?
        if frame.dst_id == self.node_addr or (
            self.is_bs and frame.dst_id == self.bs_addr
        ):
            return RoutingAction(ActionType.DELIVER, frame=frame)

        # Forward along known route
        route = self._routes.get(frame.dst_id)
        if route is not None:
            forwarded = replace(
                frame,
                ttl=frame.ttl - 1,
                hop_count=frame.hop_count + 1,
                prev_hop_id=self.node_addr,
            )
            if forwarded.ttl <= 0:
                return RoutingAction(ActionType.DROP, reason="ttl_expired")
            self._routes[frame.dst_id] = route._replace(last_used=rx_time)
            return RoutingAction(
                ActionType.FORWARD,
                frame=forwarded,
                next_hop=route.next_hop,
                reason="aodv_forward",
            )

        # No route — drop (we're an intermediate node, not the originator)
        return RoutingAction(ActionType.DROP, reason="no_route_intermediate")

    # ------------------------------------------------------------------
    # ABC: get_next_hop
    # ------------------------------------------------------------------

    def get_next_hop(
        self,
        dst_addr: int,
        tx_time: float,
    ) -> RoutingAction:
        seq = self._next_data_seq()

        # Do we have a route?
        route = self._routes.get(dst_addr)
        if route is not None:
            self._routes[dst_addr] = route._replace(last_used=tx_time)
            return RoutingAction(
                ActionType.FORWARD,
                next_hop=route.next_hop,
                seq=seq,
                ttl=self._default_ttl,
                reason="aodv_route_found",
            )

        # No route — initiate RREQ if not already pending
        if dst_addr not in self._pending_rreqs:
            self._initiate_rreq(dst_addr, tx_time)

        # Buffer the send request
        if len(self._pending_buf) < self._pending_buf_size:
            self._pending_buf.append(_PendingFrame(dst_addr, tx_time))

        return RoutingAction(
            ActionType.DROP,
            seq=seq,
            ttl=self._default_ttl,
            reason="no_route_rreq_sent",
        )

    # ------------------------------------------------------------------
    # ABC: on_tick
    # ------------------------------------------------------------------

    def on_tick(self, now: float) -> None:
        # 1. Expire dedup caches
        self._expire_cache(self._dedup, now)
        self._expire_cache(self._data_dedup, now)

        # 2. Expire stale neighbours
        self.expire_neighbours(now, self._nb_timeout)

        # 3. Expire stale routes
        expired_routes = [
            dst for dst, r in self._routes.items()
            if (now - r.last_used) > self._route_timeout
        ]
        for dst in expired_routes:
            del self._routes[dst]

        # 4. Retry pending RREQs
        for dst, state in list(self._pending_rreqs.items()):
            if (now - state.last_sent) >= self._rreq_retry:
                if state.retries >= self._rreq_max_retries:
                    # Give up — drop pending frames for this dst
                    del self._pending_rreqs[dst]
                    self._pending_buf = [
                        p for p in self._pending_buf if p.dst_addr != dst
                    ]
                else:
                    state.retries += 1
                    state.last_sent = now
                    self._emit_rreq(dst, state.rreq_seq, now)

        # 5. Expire old pending frames (older than rreq_max_retries * rreq_retry)
        max_wait = self._rreq_max_retries * self._rreq_retry
        self._pending_buf = [
            p for p in self._pending_buf
            if (now - p.queued_time) <= max_wait
        ]

    # ------------------------------------------------------------------
    # ABC: get_pending_control_frames
    # ------------------------------------------------------------------

    def get_pending_control_frames(self) -> List[Frame]:
        out = self._outbox[:]
        self._outbox.clear()
        return out

    # ------------------------------------------------------------------
    # Properties for bakeoff metrics
    # ------------------------------------------------------------------

    @property
    def route_count(self) -> int:
        return len(self._routes)

    @property
    def pending_rreq_count(self) -> int:
        return len(self._pending_rreqs)

    @property
    def pending_buf_len(self) -> int:
        return len(self._pending_buf)

    @property
    def current_seq(self) -> int:
        return self._seq

    def has_route_to(self, dst: int) -> bool:
        return dst in self._routes

    def get_route(self, dst: int) -> Optional[_RouteEntry]:
        return self._routes.get(dst)

    # ------------------------------------------------------------------
    # Control frame handling
    # ------------------------------------------------------------------

    def _handle_control(
        self,
        frame: Frame,
        rssi_dbm: float,
        snr_db: float,
        rx_time: float,
    ) -> RoutingAction:
        try:
            msg = decode_payload(frame.payload)
        except Exception:
            return RoutingAction(ActionType.DROP, reason="bad_ctrl_payload")

        msg_type = msg.get("type")
        if msg_type == self.RREQ:
            return self._handle_rreq(frame, msg, rssi_dbm, snr_db, rx_time)
        elif msg_type == self.RREP:
            return self._handle_rrep(frame, msg, rssi_dbm, snr_db, rx_time)
        elif msg_type == self.RERR:
            return self._handle_rerr(frame, msg, rx_time)
        else:
            return RoutingAction(ActionType.DROP, reason="unknown_ctrl_type")

    def _handle_rreq(
        self, frame: Frame, msg: dict,
        rssi_dbm: float, snr_db: float, rx_time: float,
    ) -> RoutingAction:
        rreq_src = msg["src"]
        rreq_dst = msg["dst"]
        rreq_seq = msg["rreq_seq"]
        hop_count = msg["hop_count"]

        # Dedup
        dedup_key = (rreq_src, rreq_seq)
        if dedup_key in self._dedup:
            return RoutingAction(ActionType.DROP, reason="duplicate_rreq")
        self._cache_insert(self._dedup, dedup_key, rx_time)

        # Learn reverse route to RREQ originator
        self._learn_route(
            dst=rreq_src,
            next_hop=frame.prev_hop_id,
            hop_count=hop_count + 1,
            dst_seq=rreq_seq,
            now=rx_time,
        )

        # Are we the destination?
        if rreq_dst == self.node_addr or (self.is_bs and rreq_dst == self.bs_addr):
            self._emit_rrep(
                rrep_dst=rreq_src,
                rrep_src=self.node_addr,
                hop_count=0,
                dst_seq=self._seq,
                now=rx_time,
            )
            return RoutingAction(ActionType.DROP, reason="rreq_answered")

        # Do we have a fresh route to the destination?
        route = self._routes.get(rreq_dst)
        if route is not None:
            self._emit_rrep(
                rrep_dst=rreq_src,
                rrep_src=rreq_dst,
                hop_count=route.hop_count,
                dst_seq=route.dst_seq,
                now=rx_time,
            )
            return RoutingAction(ActionType.DROP, reason="rreq_proxy_answered")

        # Rebroadcast RREQ with incremented hop_count
        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="rreq_ttl_expired")

        fwd_msg = {
            "type": self.RREQ,
            "src": rreq_src,
            "dst": rreq_dst,
            "rreq_seq": rreq_seq,
            "hop_count": hop_count + 1,
        }
        fwd_frame = Frame.new(
            src_id=rreq_src,
            dst_id=BROADCAST_ID,
            message_class=MessageClass.CTRL,
            payload=encode_payload(fwd_msg),
            seq=frame.seq,
            ttl=frame.ttl - 1,
            prev_hop_id=self.node_addr,
            hop_count=frame.hop_count + 1,
        )
        self._outbox.append(fwd_frame)
        return RoutingAction(ActionType.DROP, reason="rreq_forwarded")

    def _handle_rrep(
        self, frame: Frame, msg: dict,
        rssi_dbm: float, snr_db: float, rx_time: float,
    ) -> RoutingAction:
        rrep_src = msg["src"]      # the node the route goes TO
        rrep_dst = msg["dst"]      # who requested the route (RREQ originator)
        hop_count = msg["hop_count"]
        dst_seq = msg["dst_seq"]

        # Install forward route to rrep_src (the destination we wanted)
        self._learn_route(
            dst=rrep_src,
            next_hop=frame.prev_hop_id,
            hop_count=hop_count + 1,
            dst_seq=dst_seq,
            now=rx_time,
        )

        # Is the RREP for us?
        if rrep_dst == self.node_addr:
            # Route discovered — clear pending RREQ
            self._pending_rreqs.pop(rrep_src, None)
            # Note: the frame that originally triggered this RREQ is gone —
            # get_next_hop() returned DROP for it and the caller (stack.send)
            # does not retry. _pending_buf only tracks (dst_addr, queued_time)
            # for tick-driven expiry; it is not replayed once a route lands.
            # The route is still useful: it's picked up the next time the
            # app layer independently calls send() for this dst.
            return RoutingAction(ActionType.DROP, reason="rrep_installed")

        # Forward RREP toward the RREQ originator
        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="rrep_ttl_expired")

        route_to_origin = self._routes.get(rrep_dst)
        if route_to_origin is None:
            return RoutingAction(ActionType.DROP, reason="rrep_no_reverse_route")

        fwd_msg = {
            "type": self.RREP,
            "src": rrep_src,
            "dst": rrep_dst,
            "hop_count": hop_count + 1,
            "dst_seq": dst_seq,
        }
        fwd_frame = Frame.new(
            src_id=rrep_src,
            dst_id=rrep_dst,
            message_class=MessageClass.CTRL,
            payload=encode_payload(fwd_msg),
            seq=frame.seq,
            ttl=frame.ttl - 1,
            prev_hop_id=self.node_addr,
            hop_count=frame.hop_count + 1,
        )
        self._outbox.append(fwd_frame)
        return RoutingAction(ActionType.DROP, reason="rrep_forwarded")

    def _handle_rerr(
        self, frame: Frame, msg: dict, rx_time: float,
    ) -> RoutingAction:
        broken_dst = msg["broken_dst"]

        # Remove route if it goes through the reporter
        route = self._routes.get(broken_dst)
        if route is not None and route.next_hop == frame.prev_hop_id:
            del self._routes[broken_dst]

            # Re-flood RERR if we had a route (we relied on it)
            if frame.ttl > 1:
                rerr_msg = {"type": self.RERR, "broken_dst": broken_dst}
                rerr_frame = Frame.new(
                    src_id=self.node_addr,
                    dst_id=BROADCAST_ID,
                    message_class=MessageClass.CTRL,
                    payload=encode_payload(rerr_msg),
                    seq=self._next_data_seq(),
                    ttl=frame.ttl - 1,
                    prev_hop_id=self.node_addr,
                )
                self._outbox.append(rerr_frame)

        return RoutingAction(ActionType.DROP, reason="rerr_processed")

    # ------------------------------------------------------------------
    # RREQ / RREP emission
    # ------------------------------------------------------------------

    def _initiate_rreq(self, dst_addr: int, now: float) -> None:
        rreq_seq = self._next_rreq_seq()
        self._pending_rreqs[dst_addr] = _RreqState(dst_addr, rreq_seq, now)
        self._emit_rreq(dst_addr, rreq_seq, now)

    def _emit_rreq(self, dst_addr: int, rreq_seq: int, now: float) -> None:
        msg = {
            "type": self.RREQ,
            "src": self.node_addr,
            "dst": dst_addr,
            "rreq_seq": rreq_seq,
            "hop_count": 0,
        }
        frame = Frame.new(
            src_id=self.node_addr,
            dst_id=BROADCAST_ID,
            message_class=MessageClass.CTRL,
            payload=encode_payload(msg),
            seq=self._next_data_seq(),
            ttl=self._default_ttl,
            prev_hop_id=self.node_addr,
            timestamp_ms=int(now * 1000),
        )
        self._outbox.append(frame)

    def _emit_rrep(
        self, rrep_dst: int, rrep_src: int,
        hop_count: int, dst_seq: int, now: float,
    ) -> None:
        msg = {
            "type": self.RREP,
            "src": rrep_src,
            "dst": rrep_dst,
            "hop_count": hop_count,
            "dst_seq": dst_seq,
        }
        # RREP is logically addressed to rrep_dst, but the radio medium is
        # broadcast (VirtualChannel delivers to every node in range
        # regardless of dst_id) — the intended next hop just happens to be
        # the closest node that will accept it in _handle_rrep. There is no
        # separate physical unicast step here.
        frame = Frame.new(
            src_id=rrep_src,
            dst_id=rrep_dst,
            message_class=MessageClass.CTRL,
            payload=encode_payload(msg),
            seq=self._next_data_seq(),
            ttl=self._default_ttl,
            prev_hop_id=self.node_addr,
            timestamp_ms=int(now * 1000),
        )
        self._outbox.append(frame)

    # ------------------------------------------------------------------
    # Route learning
    # ------------------------------------------------------------------

    def _learn_route(
        self, dst: int, next_hop: int,
        hop_count: int, dst_seq: int, now: float,
    ) -> None:
        existing = self._routes.get(dst)
        # Install if: no existing route, or new route is fresher, or
        # same freshness but shorter
        if existing is None or (
            dst_seq > existing.dst_seq
        ) or (
            dst_seq == existing.dst_seq and hop_count < existing.hop_count
        ):
            self._routes[dst] = _RouteEntry(
                next_hop=next_hop,
                hop_count=hop_count,
                dst_seq=dst_seq,
                last_used=now,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_data_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
        return s

    def _next_rreq_seq(self) -> int:
        s = self._rreq_seq
        self._rreq_seq = (self._rreq_seq + 1) & 0xFFFF
        return s

    def _cache_insert(
        self,
        cache: OrderedDict[tuple[int, int], _DedupEntry],
        key: tuple[int, int],
        rx_time: float,
    ) -> None:
        if key in cache:
            cache.move_to_end(key)
            cache[key].rx_time = rx_time
        else:
            cache[key] = _DedupEntry(rx_time)
            while len(cache) > self._cache_size:
                cache.popitem(last=False)

    def _expire_cache(
        self,
        cache: OrderedDict[tuple[int, int], _DedupEntry],
        now: float,
    ) -> None:
        expired = [k for k, v in cache.items() if (now - v.rx_time) > self._dedup_expiry]
        for k in expired:
            del cache[k]