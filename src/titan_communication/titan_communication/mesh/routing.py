"""
IRoutingScheme — abstract base for mesh routing in Titan DMS.

Three implementations will subclass this:
  - routing_flood.py   (flooding with dedup + TTL)
  - routing_aodv.py    (reactive RREQ/RREP/RERR)
  - routing_gradient.py (proactive BS-beacon gradient)

Design constraints:
  - No rclpy imports (Tier 1 SimPy + Tier 2 ROS must share code).
  - No simpy imports here either — the ABC is pure Python.
    SimPy process wrappers live in stack.py.
  - Routing schemes never touch the radio directly. They return
    RoutingAction objects; stack.py interprets them.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, List

from .frame import Frame, BROADCAST_ID


# ---------------------------------------------------------------------------
# Routing actions — what the scheme tells the stack to do
# ---------------------------------------------------------------------------

class ActionType(Enum):
    """What stack.py should do with a routing decision."""
    FORWARD = auto()     # enqueue frame toward next_hop
    BROADCAST = auto()   # enqueue frame to all neighbours (addr=0xFFFF)
    DROP = auto()        # discard silently (duplicate, TTL=0, no route)
    DELIVER = auto()     # frame reached its destination — hand to app layer


@dataclass(frozen=True)
class RoutingAction:
    """Immutable instruction returned by the routing scheme to the stack."""
    action: ActionType
    frame: Optional[Frame] = None  # the Frame to forward/deliver (None for DROP)
    next_hop: int = BROADCAST_ID   # destination node addr (FORWARD only)
    seq: int = 0                   # seq number for originated frames
    ttl: int = 5                   # TTL for originated frames
    reason: str = ""               # human-readable, for logging / CSV


# ---------------------------------------------------------------------------
# Neighbour table entry — shared across schemes
# ---------------------------------------------------------------------------

@dataclass
class NeighbourInfo:
    """What we know about a one-hop neighbour from recent receptions."""
    addr: int
    last_rssi_dbm: float = -999.0
    last_snr_db: float = -99.0
    last_seen_time: float = 0.0    # sim-time seconds
    hop_count_to_bs: int = 0xFF    # only gradient uses this; others ignore


# ---------------------------------------------------------------------------
# Routing scheme ABC
# ---------------------------------------------------------------------------

class IRoutingScheme(abc.ABC):
    """
    Base class for all mesh routing schemes in the Titan DMS bake-off.

    Lifecycle (driven by stack.py):
      1. __init__()                    — called once per node
      2. on_frame_received()           — called every time the radio delivers a frame
      3. get_next_hop()                — called when the app layer wants to SEND a new frame
      4. on_tick()                     — called periodically (e.g. every 1 s sim-time)
      5. get_pending_control_frames()  — called after on_tick() to collect any
                                         control frames the scheme wants to emit
    """

    def __init__(
        self,
        node_addr: int,
        bs_addr: int = 0x0001,
        is_bs: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        self.node_addr = node_addr
        self.bs_addr = bs_addr
        self.is_bs = is_bs
        self.config = config or {}
        self.neighbours: dict[int, NeighbourInfo] = {}

    # ---- required overrides ------------------------------------------------

    @abc.abstractmethod
    def on_frame_received(
        self,
        frame: Frame,
        rssi_dbm: float,
        snr_db: float,
        rx_time: float,
    ) -> RoutingAction:
        """
        Called when a frame arrives from the radio (already CRC-verified
        and unpacked into a Frame object).

        Returns:
            RoutingAction telling the stack what to do with this frame.
        """
        ...

    @abc.abstractmethod
    def get_next_hop(
        self,
        dst_addr: int,
        tx_time: float,
    ) -> RoutingAction:
        """
        Called when this node's app layer wants to SEND a new data frame.

        The stack will build the actual Frame using the returned seq/ttl.
        For uplink (robot -> BS): dst_addr == bs_addr.

        Returns:
            RoutingAction with seq and ttl set for the stack to use.
        """
        ...

    @abc.abstractmethod
    def on_tick(self, now: float) -> None:
        """Periodic maintenance: expire routes, emit beacons, prune caches."""
        ...

    @abc.abstractmethod
    def get_pending_control_frames(self) -> List[Frame]:
        """Return any control Frames the scheme wants to transmit after on_tick()."""
        ...

    # ---- shared helpers (concrete) -----------------------------------------

    def update_neighbour(
        self,
        addr: int,
        rssi_dbm: float,
        snr_db: float,
        rx_time: float,
        hop_count_to_bs: int = 0xFF,
    ) -> None:
        """Update or insert a neighbour table entry."""
        if addr in self.neighbours:
            nb = self.neighbours[addr]
            nb.last_rssi_dbm = rssi_dbm
            nb.last_snr_db = snr_db
            nb.last_seen_time = rx_time
            if hop_count_to_bs != 0xFF:
                nb.hop_count_to_bs = hop_count_to_bs
        else:
            self.neighbours[addr] = NeighbourInfo(
                addr=addr,
                last_rssi_dbm=rssi_dbm,
                last_snr_db=snr_db,
                last_seen_time=rx_time,
                hop_count_to_bs=hop_count_to_bs,
            )

    def expire_neighbours(self, now: float, timeout_s: float = 30.0) -> None:
        """Remove neighbours not heard from within timeout_s."""
        stale = [
            addr for addr, nb in self.neighbours.items()
            if (now - nb.last_seen_time) > timeout_s
        ]
        for addr in stale:
            del self.neighbours[addr]

    def link_metric(self, addr: int) -> float:
        """
        Weighted link cost for next-hop selection. Lower is better.

        Weights (tunable via self.config):
          w_rssi:  -1.0   (more negative RSSI = higher cost)
          w_snr:   -0.5
          w_hops:  10.0   (penalty per hop to BS)

        Returns float('inf') if neighbour unknown.
        """
        nb = self.neighbours.get(addr)
        if nb is None:
            return float("inf")

        w_rssi = self.config.get("w_rssi", -1.0)
        w_snr = self.config.get("w_snr", -0.5)
        w_hops = self.config.get("w_hops", 10.0)

        return (
            w_rssi * nb.last_rssi_dbm
            + w_snr * nb.last_snr_db
            + w_hops * nb.hop_count_to_bs
        )