"""
Flooding with dedup — simplest mesh routing baseline for Titan DMS bake-off.

Every frame is rebroadcast to all neighbours unless:
  - Already in the dedup cache (duplicate)
  - TTL has reached 0
  - This node is the final destination (DELIVER to app)

Config keys (via scenarios.yaml -> config dict):
  default_ttl:       int, max hops for originated frames (default 5)
  dedup_cache_size:   int, max entries before oldest evicted (default 256)
  dedup_expiry_s:     float, seconds before a dedup entry expires (default 30.0)
  neighbour_timeout:  float, seconds before a neighbour is expired (default 30.0)
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import replace
from typing import List, Optional

from .frame import Frame, BROADCAST_ID
from .routing import (
    IRoutingScheme,
    RoutingAction,
    ActionType,
)


class _DedupEntry:
    __slots__ = ("rx_time",)

    def __init__(self, rx_time: float) -> None:
        self.rx_time = rx_time


class FloodRouting(IRoutingScheme):
    """Broadcast-flood with TTL + sequence-number dedup."""

    def __init__(
        self,
        node_addr: int,
        bs_addr: int = 0x0001,
        is_bs: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        super().__init__(node_addr, bs_addr, is_bs, config)

        self._default_ttl: int = self.config.get("default_ttl", 5)
        self._cache_size: int = self.config.get("dedup_cache_size", 256)
        self._dedup_expiry: float = self.config.get("dedup_expiry_s", 30.0)
        self._nb_timeout: float = self.config.get("neighbour_timeout", 30.0)

        # OrderedDict for LRU eviction: key = (src_id, seq)
        self._dedup: OrderedDict[tuple[int, int], _DedupEntry] = OrderedDict()

        # Local sequence counter for frames this node originates
        self._seq: int = 0

    # ------------------------------------------------------------------
    # ABC implementation
    # ------------------------------------------------------------------

    def on_frame_received(
        self,
        frame: Frame,
        rssi_dbm: float,
        snr_db: float,
        rx_time: float,
    ) -> RoutingAction:
        # Always update neighbour table from the previous hop
        self.update_neighbour(frame.prev_hop_id, rssi_dbm, snr_db, rx_time)

        # 1. Dedup check — keyed on (original source, sequence number)
        dedup_key = (frame.src_id, frame.seq)
        if dedup_key in self._dedup:
            return RoutingAction(ActionType.DROP, reason="duplicate")

        # 2. Record in dedup cache
        self._cache_insert(dedup_key, rx_time)

        # 3. Is this frame for us?
        if frame.dst_id == self.node_addr or (
            self.is_bs and frame.dst_id == self.bs_addr
        ):
            return RoutingAction(ActionType.DELIVER, frame=frame)

        # 4. TTL exhausted?
        if frame.ttl <= 1:
            return RoutingAction(ActionType.DROP, reason="ttl_expired")

        # 5. Rebroadcast with decremented TTL, incremented hop_count,
        #    and our addr as prev_hop_id.  Frame is frozen -> use replace().
        forwarded = replace(
            frame,
            ttl=frame.ttl - 1,
            hop_count=frame.hop_count + 1,
            prev_hop_id=self.node_addr,
        )
        return RoutingAction(
            ActionType.BROADCAST,
            frame=forwarded,
            reason="flood_forward",
        )

    def get_next_hop(
        self,
        dst_addr: int,
        tx_time: float,
    ) -> RoutingAction:
        # Flooding doesn't pick a specific next hop — always broadcast.
        # Register in dedup cache so we don't re-flood our own frame.
        seq = self._next_seq()
        dedup_key = (self.node_addr, seq)
        self._cache_insert(dedup_key, tx_time)

        # Return BROADCAST with the seq number the stack should use
        # when building the Frame.  frame=None here because the stack
        # builds the Frame (it knows payload, message_class, etc.).
        return RoutingAction(
            ActionType.BROADCAST,
            next_hop=BROADCAST_ID,
            seq=seq,
            ttl=self._default_ttl,
            reason="flood_originate",
        )

    def on_tick(self, now: float) -> None:
        # Expire old dedup entries
        expired = [
            k for k, v in self._dedup.items()
            if (now - v.rx_time) > self._dedup_expiry
        ]
        for k in expired:
            del self._dedup[k]

        # Expire stale neighbours
        self.expire_neighbours(now, self._nb_timeout)

    def get_pending_control_frames(self) -> List[Frame]:
        # Flooding has no control frames (no beacons, no RREQ).
        return []

    # ------------------------------------------------------------------
    # Properties for bakeoff metrics
    # ------------------------------------------------------------------

    @property
    def dedup_cache_len(self) -> int:
        """Current dedup cache occupancy — for overhead metrics."""
        return len(self._dedup)

    @property
    def current_seq(self) -> int:
        """Current sequence counter (read-only, for test assertions)."""
        return self._seq

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        seq = self._seq
        self._seq = (self._seq + 1) & 0xFFFF  # wrap at 16-bit
        return seq

    def _cache_insert(self, key: tuple[int, int], rx_time: float) -> None:
        if key in self._dedup:
            self._dedup.move_to_end(key)
            self._dedup[key].rx_time = rx_time
        else:
            self._dedup[key] = _DedupEntry(rx_time)
            # LRU eviction if cache full
            while len(self._dedup) > self._cache_size:
                self._dedup.popitem(last=False)