"""Transport-layer interfaces for the LoRa mesh.

Two interfaces live here:

    ILoRaTransport — a single node's radio. Real hardware (SX1276 on
    ESP32) and the SimPy virtual radio both implement this.

    IChannel — the shared medium. Registers transports with a
    position getter (robots move), delivers a broadcast from one
    transport to all others whose RSSI clears sensitivity.

Behavioural model matches embedded LoRa hardware:

    - send() is non-blocking. It kicks off a transmission and returns
      immediately, just like calling transmit() on an SX1276 driver.
      The SX1276 raises the TX_DONE interrupt when the FIFO is empty;
      our equivalent is the on_tx_done callback.
    - Reception is callback-based, analogous to RX_DONE.
    - The radio is either idle or busy; send() while busy raises
      TransportBusyError. Real hardware would return an error code;
      we raise instead of returning a status so callers can't ignore it.
    - Duty-cycle enforcement lives above the radio (in the mesh queue
      of Weeks 3-4). If send() is called and duty says no, we raise
      DutyViolationError. Real hardware doesn't enforce duty — the MAC
      layer does — so this mirrors reality.

Step 5 defines only the interfaces. Concrete VirtualLoRaTransport +
VirtualChannel arrive in step 8's end-to-end smoke test.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

from titan_communication.radio.airtime import LoRaParams
from titan_communication.radio.duty import DutyBucket

__all__ = [
    "IChannel",
    "ILoRaTransport",
    "PositionGetter",
    "ReceiveCallback",
    "ReceptionInfo",
    "TransportBusyError",
    "TransportError",
    "TxDoneCallback",
]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TransportError(RuntimeError):
    """Base for transport-layer errors."""


class TransportBusyError(TransportError):
    """Raised when send() is called while a prior transmission is still in flight."""


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReceptionInfo:
    """Per-reception metadata delivered to receive callbacks.

    All fields are *at-receiver* values — RSSI and SNR after the
    channel model has been applied for this specific TX/RX link.

    Notably absent: sender identity. Real SX1276 hardware doesn't know
    who sent the bytes — that's in the mesh frame header. Keeping
    ReceptionInfo hardware-realistic prevents mesh code from silently
    depending on info that won't exist on real radios.
    """

    rssi_dbm: float
    snr_db: float
    arrival_time_s: float


# Callback type aliases. Named so mypy errors on a wrong signature point
# here instead of at anonymous types buried in method signatures.
ReceiveCallback = Callable[[bytes, ReceptionInfo], None]
TxDoneCallback = Callable[[], None]
PositionGetter = Callable[[], tuple[float, float]]  # returns (x_m, y_m)


# ---------------------------------------------------------------------------
# ILoRaTransport
# ---------------------------------------------------------------------------
class ILoRaTransport(ABC):
    """A single node's LoRa radio.

    Planned implementations:
        VirtualLoRaTransport — SimPy-based, for Tier 1 bake-off and Tier
                               2 Gazebo scenarios. Built in step 8.
        SX1276Transport      — Real hardware on ESP32, for the leader-only
                               hardware proof-of-concept. Deferred.

    Invariant: send() must not block. On real hardware it enqueues into
    the SX1276 FIFO and returns; in SimPy it kicks off a process and
    returns. Callers that need to know when TX completes register
    on_tx_done or poll is_busy.
    """

    # -- Identity ---------------------------------------------------------

    @property
    @abstractmethod
    def node_id(self) -> str:
        """Unique identifier for this radio.

        Used by IChannel to track registered transports and by the mesh
        layer (via frame header) for source-based deduplication.
        """

    # -- Configuration (fixed at construction) ----------------------------

    @property
    @abstractmethod
    def params(self) -> LoRaParams:
        """LoRa PHY parameters (SF, BW, CR, ...) for airtime + sensitivity."""

    @property
    @abstractmethod
    def duty(self) -> DutyBucket:
        """Per-node duty-cycle bucket. Mesh queue consults this before dequeuing."""

    @property
    @abstractmethod
    def tx_power_dbm(self) -> float:
        """Transmit power in dBm. Paper uses 15 dBm (§III-B, Algorithm 2)."""

    # -- Runtime state ----------------------------------------------------

    @property
    @abstractmethod
    def is_busy(self) -> bool:
        """True if a transmission is currently in progress.

        send() while busy raises TransportBusyError. Callers that want
        to defer instead of erroring should either poll this or register
        on_tx_done.
        """

    # -- Transmission -----------------------------------------------------

    @abstractmethod
    def send(self, payload: bytes) -> None:
        """Transmit ``payload`` bytes. Returns immediately (non-blocking).

        Raises:
            TransportBusyError: radio is currently transmitting.
            DutyViolationError: the duty policy would be violated.
            ValueError: payload is empty or exceeds 255 bytes
                        (LoRa PHY payload limit).
        """

    # -- Subscription -----------------------------------------------------

    @abstractmethod
    def on_receive(self, callback: ReceiveCallback) -> None:
        """Register a callback for successfully-received packets.

        Multiple callbacks may be registered; they fire in registration
        order. Callbacks run in the transport's thread of control
        (SimPy process context in Tier 1; ISR-adjacent context on
        hardware — keep them fast).

        The transport delivers every packet whose RSSI clears
        sensitivity. Filtering by destination address is the mesh
        layer's job, not the radio's.
        """

    @abstractmethod
    def on_tx_done(self, callback: TxDoneCallback) -> None:
        """Register a callback that fires exactly once per completed send().

        Same threading rules as on_receive. Callbacks fire in
        registration order after each transmission's on-air time
        elapses.
        """


# ---------------------------------------------------------------------------
# IChannel
# ---------------------------------------------------------------------------
class IChannel(ABC):
    """The shared radio medium.

    Planned implementation:
        VirtualChannel — SimPy-based, wraps ChannelModel + collision
                         detection. Built in step 8.

    Real hardware has no equivalent object — the atmosphere is the
    channel. This abstraction exists purely for simulation to model
    pathloss, shadowing, and collisions across nodes deterministically.
    """

    @abstractmethod
    def register(
        self,
        transport: ILoRaTransport,
        position_getter: PositionGetter,
    ) -> None:
        """Register a transport with a callable returning its current position.

        The callable is invoked at the start of each transmission to
        compute pathloss to every other registered transport. Using a
        callable rather than a static tuple accommodates moving robots
        without a "channel.update_position(...)" side-channel.
        """

    @abstractmethod
    def unregister(self, transport: ILoRaTransport) -> None:
        """Remove a transport. Idempotent — unregistering an unknown transport is a no-op."""

    @abstractmethod
    def broadcast(self, sender: ILoRaTransport, payload: bytes) -> None:
        """Called by ``sender.send()`` internally. Delivers to in-range peers.

        For each other registered transport, the concrete channel:
            1. Computes distance from sender's position to peer's position.
            2. Applies ChannelModel pathloss + shadowing -> RSSI at peer.
            3. If RSSI >= sensitivity(peer.params), delivers to peer's
               receive callbacks after airtime elapses.
            4. Otherwise silently drops (mimics "signal too weak to decode").

        SF/BW/CR mismatch between sender and peer is treated as a drop
        (LoRa parameters must match for decoding).
        """
