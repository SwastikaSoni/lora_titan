"""SimPy-based concrete implementations of ILoRaTransport and IChannel.

These are the Tier-1 counterparts to the eventual ESP32 SX1276 hardware
implementations. They live in the ``radio/`` package (not ``mesh/``)
because they operate at the same abstraction level as the paper's
"physical layer of the LoRaWAN protocol" (§III-B): frames go on air,
frames come off air; anything above (mesh routing, priority, dedup)
is somebody else's problem.

Collision model — deliberately simple for step 8:
    * All-or-nothing capture. If two transmissions overlap in time at
      a receiver, both are dropped. Real LoRa has partial capture
      (~6 dB SNR margin lets the stronger signal win); we skip that
      here. Bias is toward pessimism, which is the right side for a
      SAR-critical protocol.
    * SF/BW/CR mismatch = silent drop. A LoRa receiver locked to SF=10
      cannot decode SF=7, just like real hardware.

Position model:
    * Each transport registers a ``PositionGetter`` callable with the
      channel. The channel queries positions at the *start* of each
      broadcast — snapshot semantics. If a robot moves during the
      113 ms it takes to transmit, we still use the start-of-tx position
      for the whole delivery. Fine at typical robot speeds (m/s) where
      113 ms of motion is ~10 cm — well below the shadowing decorrelation
      distance the channel model implicitly assumes.
"""

from __future__ import annotations

import math
from collections.abc import Generator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import simpy

from titan_communication.radio.airtime import LoRaParams, time_on_air_s
from titan_communication.radio.channel import ChannelModel, is_decodable, snr_db
from titan_communication.radio.duty import DutyBucket, DutyViolationError
from titan_communication.radio.transport import (
    IChannel,
    ILoRaTransport,
    PositionGetter,
    ReceiveCallback,
    ReceptionInfo,
    TransportBusyError,
    TxDoneCallback,
)

if TYPE_CHECKING:
    from simpy.events import Event

__all__ = [
    "VirtualChannel",
    "VirtualLoRaTransport",
]


@dataclass
class _Registration:
    transport: VirtualLoRaTransport
    position_getter: PositionGetter


@dataclass
class _OnAirInterval:
    """A transmission currently occupying the medium at some receiver."""

    sender_id: str
    start_s: float
    end_s: float
    rssi_dbm: float


# ---------------------------------------------------------------------------
# VirtualLoRaTransport
# ---------------------------------------------------------------------------
class VirtualLoRaTransport(ILoRaTransport):
    """A single simulated LoRa radio bound to a SimPy environment."""

    def __init__(
        self,
        env: simpy.Environment,
        channel: VirtualChannel,
        node_id: str,
        params: LoRaParams,
        duty: DutyBucket,
        position_getter: PositionGetter,
        tx_power_dbm: float = 15.0,
    ) -> None:
        self._env = env
        self._channel = channel
        self._node_id = node_id
        self._params = params
        self._duty = duty
        self._tx_power = tx_power_dbm
        self._busy = False
        self._recv_cbs: list[ReceiveCallback] = []
        self._tx_cbs: list[TxDoneCallback] = []
        channel.register(self, position_getter)

    # -- ILoRaTransport implementation ------------------------------------

    @property
    def node_id(self) -> str:
        return self._node_id

    @property
    def params(self) -> LoRaParams:
        return self._params

    @property
    def duty(self) -> DutyBucket:
        return self._duty

    @property
    def tx_power_dbm(self) -> float:
        return self._tx_power

    @property
    def is_busy(self) -> bool:
        return self._busy

    def send(self, payload: bytes) -> None:
        if self._busy:
            raise TransportBusyError(f"{self._node_id} is currently transmitting")
        if not payload or len(payload) > 255:
            raise ValueError(f"payload length {len(payload)} out of range 1..255")

        airtime = time_on_air_s(self._params, len(payload))
        now = float(self._env.now)
        if not self._duty.can_transmit(now, airtime):
            raise DutyViolationError(
                f"{self._node_id}: cannot transmit {airtime * 1000:.2f} ms at "
                f"t={now:.3f}s under {self._duty.policy.name}"
            )
        self._duty.register(now, airtime)

        # Kick off two parallel processes: the channel-side delivery to
        # peers, and the transport's own busy timer + tx_done callback.
        self._channel.broadcast(self, payload)
        self._env.process(self._tx_lifecycle(airtime))

    def on_receive(self, callback: ReceiveCallback) -> None:
        self._recv_cbs.append(callback)

    def on_tx_done(self, callback: TxDoneCallback) -> None:
        self._tx_cbs.append(callback)

    # -- Internal --------------------------------------------------------

    def _tx_lifecycle(self, airtime_s: float) -> Generator[Event, object, None]:
        self._busy = True
        yield self._env.timeout(airtime_s)
        self._busy = False
        for cb in self._tx_cbs:
            cb()

    def _deliver(self, payload: bytes, info: ReceptionInfo) -> None:
        """Called by VirtualChannel when a packet successfully arrives."""
        for cb in self._recv_cbs:
            cb(payload, info)


# ---------------------------------------------------------------------------
# VirtualChannel
# ---------------------------------------------------------------------------
class VirtualChannel(IChannel):
    """A shared, deterministic simulated LoRa medium.

    Owns a ChannelModel (pathloss + shadowing) and coordinates all
    transports registered against it. Every broadcast iterates over all
    other registered transports, applies the channel model to compute
    per-link RSSI, and schedules a delivery process if the link is up
    and no collision occurs.
    """

    def __init__(
        self,
        env: simpy.Environment,
        model: ChannelModel,
        rng: np.random.Generator | None = None,
    ) -> None:
        self._env = env
        self._model = model
        self._rng = rng
        self._registry: dict[str, _Registration] = {}
        # Per-receiver on-air intervals for collision detection.
        self._active_by_receiver: dict[str, list[_OnAirInterval]] = {}

    # -- IChannel implementation -----------------------------------------

    def register(
        self,
        transport: ILoRaTransport,
        position_getter: PositionGetter,
    ) -> None:
        if not isinstance(transport, VirtualLoRaTransport):
            raise TypeError(
                f"VirtualChannel only accepts VirtualLoRaTransport, "
                f"got {type(transport).__name__}"
            )
        self._registry[transport.node_id] = _Registration(transport, position_getter)
        self._active_by_receiver.setdefault(transport.node_id, [])

    def unregister(self, transport: ILoRaTransport) -> None:
        self._registry.pop(transport.node_id, None)
        self._active_by_receiver.pop(transport.node_id, None)

    def broadcast(self, sender: ILoRaTransport, payload: bytes) -> None:
        if sender.node_id not in self._registry:
            raise ValueError(
                f"sender {sender.node_id!r} is not registered with this channel"
            )
        sender_reg = self._registry[sender.node_id]
        sender_pos = sender_reg.position_getter()
        airtime = time_on_air_s(sender.params, len(payload))
        start_s = float(self._env.now)
        end_s = start_s + airtime
        self._prune_stale_intervals(start_s)

        for peer_id, peer_reg in self._registry.items():
            if peer_id == sender.node_id:
                continue
            peer = peer_reg.transport

            # SF/BW/CR mismatch = silent drop.
            if not self._params_compatible(sender.params, peer.params):
                continue

            peer_pos = peer_reg.position_getter()
            distance = _euclidean(sender_pos, peer_pos)
            rssi = self._model.rssi_dbm(
                tx_power_dbm=sender.tx_power_dbm,
                distance_m=distance,
                rng=self._rng,
            )
            if not is_decodable(
                rssi_dbm=rssi,
                spreading_factor=peer.params.spreading_factor,
                bandwidth_hz=peer.params.bandwidth_hz,
            ):
                continue

            # Register the on-air interval at the receiver for collision
            # detection. Delivery process checks for overlap at delivery time.
            interval = _OnAirInterval(
                sender_id=sender.node_id,
                start_s=start_s,
                end_s=end_s,
                rssi_dbm=rssi,
            )
            self._active_by_receiver[peer_id].append(interval)
            self._env.process(
                self._deliver_after_airtime(peer, peer_id, interval, payload)
            )

    # -- Internal --------------------------------------------------------

    @staticmethod
    def _params_compatible(a: LoRaParams, b: LoRaParams) -> bool:
        """Two radios can talk iff SF, BW, CR all match."""
        return (
            a.spreading_factor == b.spreading_factor
            and a.bandwidth_hz == b.bandwidth_hz
            and a.coding_rate == b.coding_rate
        )

    def _deliver_after_airtime(
        self,
        peer: VirtualLoRaTransport,
        peer_id: str,
        interval: _OnAirInterval,
        payload: bytes,
    ) -> Generator[Event, object, None]:
        # Wait for the full on-air duration before delivering.
        yield self._env.timeout(interval.end_s - float(self._env.now))

        # Collision check: did any *other* transmission overlap this one
        # at this receiver during [start_s, end_s]?
        # Do NOT remove this interval from the active list here — a sibling
        # delivery process may still need to see it for its own collision
        # check. Pruning happens lazily at the start of each new broadcast.
        collided = self._collision_at(peer_id, interval)

        if collided:
            return

        # Successful reception. Deliver to callbacks with the RSSI/SNR/time
        # snapshot taken at broadcast time.
        info = ReceptionInfo(
            rssi_dbm=interval.rssi_dbm,
            snr_db=snr_db(interval.rssi_dbm, peer.params.bandwidth_hz),
            arrival_time_s=float(self._env.now),
        )
        peer._deliver(payload, info)

    def _collision_at(
        self, peer_id: str, interval: _OnAirInterval
    ) -> bool:
        """True if any other on-air interval at this receiver overlaps ``interval``."""
        for other in self._active_by_receiver[peer_id]:
            if other is interval:
                continue
            # Overlap iff other.start < interval.end AND other.end > interval.start.
            if other.start_s < interval.end_s and other.end_s > interval.start_s:
                return True
        return False


    def _prune_stale_intervals(self, now_s: float) -> None:
        """Drop intervals whose end_s <= now_s from every receiver's active list.

        Called at the start of each new broadcast. An interval that has
        fully cleared the air cannot collide with anything scheduled from
        this point forward, so keeping it wastes memory. We do NOT prune
        inside _deliver_after_airtime because a sibling delivery process
        firing at the same env.now must still be able to see its neighbours.
        """
        for peer_id, active in self._active_by_receiver.items():
            self._active_by_receiver[peer_id] = [
                iv for iv in active if iv.end_s > now_s
            ]


# ---------------------------------------------------------------------------
# Geometry helper
# ---------------------------------------------------------------------------
def _euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
