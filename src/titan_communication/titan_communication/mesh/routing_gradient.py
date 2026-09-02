"""
Proactive gradient routing for Titan DMS bake-off.

BS periodically broadcasts a beacon. Each node records its hop-count to
BS and selects the neighbour with the lowest link_metric() as its
"uphill" next-hop. Data frames always flow uphill toward BS.

Downlink (BS -> robot) uses reverse-path: each beacon/data reception
from a node installs a reverse route entry for that node.

Control frame encoding:
  Beacons use MessageClass.CTRL with a msgpack payload dict:
    {"type": "BEACON", "hop_count": int, "bs_seq": int}

Config keys (via scenarios.yaml -> config dict):
  default_ttl:        int,   max hops (default 5)
  beacon_interval_s:  float, how often BS emits a beacon (default 5.0)
  route_timeout_s:    float, gradient entry expiry (default 30.0)
  neighbour_timeout:  float, neighbour expiry (default 30.0)
  dedup_cache_size:   int,   (default 256)
  dedup_expiry_s:     float, (default 30.0)
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

class _GradientEntry(NamedTuple):
    """Per-destination gradient state."""
    next_hop: int
    hop_count: int     # our hop-count to BS (uplink) or to dst (downlink)
    metric: float      # link_metric value of the next_hop
    last_updated: float


class _DedupEntry:
    __slots__ = ("rx_time",)
    def __init__(self, rx_time: float) -> None:
        self.rx_time = rx_time


# ---------------------------------------------------------------------------
# Gradient routing
# ---------------------------------------------------------------------------

class GradientRouting(IRoutingScheme):

    BEACON = "BEACON"

    def __init__(
        self,
        node_addr: int,
        bs_addr: int = 0x0001,
        is_bs: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(node_addr, bs_addr, is_bs, config)

        self._default_ttl: int = self.config.get("default_ttl", 5)
        self._beacon_interval: float = self.config.get("beacon_interval_s", 5.0)
        self._route_timeout: float = self.config.get("route_timeout_s", 30.0)
        self._nb_timeout: float = self.config.get("neighbour_timeout", 30.0)
        self._cache_size: int = self.config.get("dedup_cache_size", 256)
        self._dedup_expiry: float = self.config.get("dedup_expiry_s", 30.0)

        # Uplink gradient: this node's best hop-count to BS and chosen next-hop
        self._uplink_hop_count: int = 0 if is_bs else 0xFF
        self._uplink_next_hop: Optional[int] = None
        self._uplink_metric: float = 0.0 if is_bs else float("inf")
        self._uplink_updated: float = 0.0

        # Downlink routes: dst_addr -> _GradientEntry
        # (for BS sending to specific robots, or intermediate relay)
        self._downlink: dict[int, _GradientEntry] = {}

        # Dedup: beacon (bs_addr, bs_seq), data (src_id, seq)
        self._beacon_dedup: OrderedDict[tuple[int, int], _DedupEntry] = OrderedDict()
        self._data_dedup: OrderedDict[tuple[int, int], _DedupEntry] = OrderedDict()

        # Control frame outbox
        self._outbox: list[Frame] = []

        # Timing
        self._last_beacon_time: float = -self._beacon_interval
        self._seq: int = 0
        self._beacon_seq: int = 0

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

        # Control frame — beacon
        if frame.message_class == MessageClass.CTRL:
            return self._handle_control(frame, rssi_dbm, snr_db, rx_time)

        # --- Data frame ---

        # Data dedup
        data_key = (frame.src_id, frame.seq)
        if data_key in self._data_dedup:
            return RoutingAction(ActionType.DROP, reason="duplicate_data")
        self._cache_insert(self._data_dedup, data_key, rx_time)

        # Learn downlink (reverse) route to the source
        self._learn_downlink(
            dst=frame.src_id,
            next_hop=frame.prev_hop_id,
            hop_count=frame.hop_count + 1,
            now=rx_time,
        )

        # For us?
        if frame.dst_id == self.node_addr or (
            self.is_bs and frame.dst_id == self.bs_addr
        ):
            return RoutingAction(ActionType.DELIVER, frame=frame)

        # Uplink (toward BS)?
        if frame.dst_id == self.bs_addr or frame.dst_id == BROADCAST_ID:
            return self._forward_uplink(frame, rx_time)

        # Downlink (BS/relay -> specific robot)?
        return self._forward_downlink(frame, rx_time)

    # ------------------------------------------------------------------
    # ABC: get_next_hop
    # ------------------------------------------------------------------

    def get_next_hop(
        self,
        dst_addr: int,
        tx_time: float,
    ) -> RoutingAction:
        seq = self._next_data_seq()

        # Uplink to BS?
        if dst_addr == self.bs_addr:
            if self.is_bs:
                # We are the BS — deliver locally
                return RoutingAction(
                    ActionType.DELIVER,
                    seq=seq,
                    ttl=self._default_ttl,
                    reason="gradient_local_bs",
                )
            if self._uplink_next_hop is not None and self._uplink_hop_count < 0xFF:
                return RoutingAction(
                    ActionType.FORWARD,
                    next_hop=self._uplink_next_hop,
                    seq=seq,
                    ttl=self._default_ttl,
                    reason="gradient_uplink",
                )
            return RoutingAction(
                ActionType.DROP,
                seq=seq,
                ttl=self._default_ttl,
                reason="no_gradient_to_bs",
            )

        # Downlink to specific robot
        entry = self._downlink.get(dst_addr)
        if entry is not None:
            return RoutingAction(
                ActionType.FORWARD,
                next_hop=entry.next_hop,
                seq=seq,
                ttl=self._default_ttl,
                reason="gradient_downlink",
            )

        return RoutingAction(
            ActionType.DROP,
            seq=seq,
            ttl=self._default_ttl,
            reason="no_downlink_route",
        )

    # ------------------------------------------------------------------
    # ABC: on_tick
    # ------------------------------------------------------------------

    def on_tick(self, now: float) -> None:
        # 1. BS emits periodic beacon
        if self.is_bs and (now - self._last_beacon_time) >= self._beacon_interval:
            self._emit_beacon(now)
            self._last_beacon_time = now

        # 2. Expire dedup caches
        self._expire_cache(self._beacon_dedup, now)
        self._expire_cache(self._data_dedup, now)

        # 3. Expire neighbours
        self.expire_neighbours(now, self._nb_timeout)

        # 4. Expire uplink gradient if stale
        if not self.is_bs and (now - self._uplink_updated) > self._route_timeout:
            self._uplink_hop_count = 0xFF
            self._uplink_next_hop = None
            self._uplink_metric = float("inf")

        # 5. Expire downlink routes
        expired = [
            dst for dst, e in self._downlink.items()
            if (now - e.last_updated) > self._route_timeout
        ]
        for dst in expired:
            del self._downlink[dst]

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
    def uplink_hop_count(self) -> int:
        return self._uplink_hop_count

    @property
    def uplink_next_hop(self) -> Optional[int]:
        return self._uplink_next_hop

    @property
    def downlink_route_count(self) -> int:
        return len(self._downlink)

    @property
    def current_seq(self) -> int:
        return self._seq

    def has_uplink(self) -> bool:
        return self._uplink_next_hop is not None and self._uplink_hop_count < 0xFF

    def has_downlink_to(self, dst: int) -> bool:
        return dst in self._downlink

    # ------------------------------------------------------------------
    # Beacon handling
    # ------------------------------------------------------------------

    def _handle_control(
        self, frame: Frame,
        rssi_dbm: float, snr_db: float, rx_time: float,
    ) -> RoutingAction:
        try:
            msg = decode_payload(frame.payload)
        except Exception:
            return RoutingAction(ActionType.DROP, reason="bad_ctrl_payload")

        if msg.get("type") != self.BEACON:
            return RoutingAction(ActionType.DROP, reason="unknown_ctrl_type")

        return self._handle_beacon(frame, msg, rssi_dbm, snr_db, rx_time)

    def _handle_beacon(
        self, frame: Frame, msg: dict,
        rssi_dbm: float, snr_db: float, rx_time: float,
    ) -> RoutingAction:
        bs_seq = msg["bs_seq"]
        beacon_hop_count = msg["hop_count"]

        # Dedup
        dedup_key = (frame.src_id, bs_seq)
        if dedup_key in self._beacon_dedup:
            return RoutingAction(ActionType.DROP, reason="duplicate_beacon")
        self._cache_insert(self._beacon_dedup, dedup_key, rx_time)

        # BS doesn't process its own beacons
        if self.is_bs:
            return RoutingAction(ActionType.DROP, reason="bs_own_beacon")

        # Update neighbour's hop_count_to_bs
        self.update_neighbour(
            frame.prev_hop_id, rssi_dbm, snr_db, rx_time,
            hop_count_to_bs=beacon_hop_count,
        )

        # Candidate: my hop_count would be beacon_hop_count + 1
        candidate_hops = beacon_hop_count + 1
        candidate_metric = self.link_metric(frame.prev_hop_id)

        # Accept if: fewer hops, or same hops but better metric
        should_update = False
        if candidate_hops < self._uplink_hop_count:
            should_update = True
        elif candidate_hops == self._uplink_hop_count and candidate_metric < self._uplink_metric:
            should_update = True
        elif frame.prev_hop_id == self._uplink_next_hop:
            # Refresh from current parent even if metric got worse
            # (prevents oscillation — stick with parent until it expires)
            should_update = True

        if should_update:
            self._uplink_hop_count = candidate_hops
            self._uplink_next_hop = frame.prev_hop_id
            self._uplink_metric = candidate_metric
            self._uplink_updated = rx_time

        # Rebroadcast beacon with incremented hop_count
        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="beacon_ttl_expired")

        fwd_msg = {
            "type": self.BEACON,
            "hop_count": candidate_hops,
            "bs_seq": bs_seq,
        }
        fwd_frame = Frame.new(
            src_id=frame.src_id,   # keep original src (BS addr)
            dst_id=BROADCAST_ID,
            message_class=MessageClass.CTRL,
            payload=encode_payload(fwd_msg),
            seq=frame.seq,
            ttl=frame.ttl - 1,
            prev_hop_id=self.node_addr,
            hop_count=frame.hop_count + 1,
        )
        self._outbox.append(fwd_frame)
        return RoutingAction(ActionType.DROP, reason="beacon_forwarded")

    def _emit_beacon(self, now: float) -> None:
        bs_seq = self._beacon_seq
        self._beacon_seq = (self._beacon_seq + 1) & 0xFFFF

        # Dedup own beacon
        dedup_key = (self.node_addr, bs_seq)
        self._cache_insert(self._beacon_dedup, dedup_key, now)

        msg = {
            "type": self.BEACON,
            "hop_count": 0,
            "bs_seq": bs_seq,
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

    # ------------------------------------------------------------------
    # Forwarding
    # ------------------------------------------------------------------

    def _forward_uplink(self, frame: Frame, rx_time: float) -> RoutingAction:
        """Forward a frame toward BS using the gradient."""
        if self._uplink_next_hop is None or self._uplink_hop_count >= 0xFF:
            return RoutingAction(ActionType.DROP, reason="no_gradient_to_bs")

        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="ttl_expired")

        # Don't forward back to the node that sent it to us
        if self._uplink_next_hop == frame.prev_hop_id:
            return RoutingAction(ActionType.DROP, reason="loop_prevention")

        forwarded = replace(
            frame,
            ttl=frame.ttl - 1,
            hop_count=frame.hop_count + 1,
            prev_hop_id=self.node_addr,
        )
        return RoutingAction(
            ActionType.FORWARD,
            frame=forwarded,
            next_hop=self._uplink_next_hop,
            reason="gradient_uplink_fwd",
        )

    def _forward_downlink(self, frame: Frame, rx_time: float) -> RoutingAction:
        """Forward a frame toward a specific robot using downlink table."""
        entry = self._downlink.get(frame.dst_id)
        if entry is None:
            return RoutingAction(ActionType.DROP, reason="no_downlink_route")

        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="ttl_expired")

        if entry.next_hop == frame.prev_hop_id:
            return RoutingAction(ActionType.DROP, reason="loop_prevention")

        forwarded = replace(
            frame,
            ttl=frame.ttl - 1,
            hop_count=frame.hop_count + 1,
            prev_hop_id=self.node_addr,
        )
        return RoutingAction(
            ActionType.FORWARD,
            frame=forwarded,
            next_hop=entry.next_hop,
            reason="gradient_downlink_fwd",
        )

    # ------------------------------------------------------------------
    # Downlink learning
    # ------------------------------------------------------------------

    def _learn_downlink(
        self, dst: int, next_hop: int,
        hop_count: int, now: float,
    ) -> None:
        """Install/update reverse route toward a source node."""
        metric = self.link_metric(next_hop)
        existing = self._downlink.get(dst)

        if existing is None or (
            hop_count < existing.hop_count
        ) or (
            hop_count == existing.hop_count and metric < existing.metric
        ) or (
            next_hop == existing.next_hop  # refresh from same path
        ):
            self._downlink[dst] = _GradientEntry(
                next_hop=next_hop,
                hop_count=hop_count,
                metric=metric,
                last_updated=now,
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_data_seq(self) -> int:
        s = self._seq
        self._seq = (self._seq + 1) & 0xFFFF
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