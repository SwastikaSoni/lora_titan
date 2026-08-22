"""Tests for transport-layer interfaces and value types."""

from __future__ import annotations

import dataclasses

import pytest
from titan_communication.radio.airtime import PAPER_DEFAULT, LoRaParams
from titan_communication.radio.duty import DutyBucket, RegionPolicy
from titan_communication.radio.transport import (
    IChannel,
    ILoRaTransport,
    PositionGetter,
    ReceiveCallback,
    ReceptionInfo,
    TransportBusyError,
    TransportError,
    TxDoneCallback,
)


# ---------------------------------------------------------------------------
# Minimal fake implementations — enough to prove the interface is usable.
# The real VirtualLoRaTransport + VirtualChannel arrive in step 8.
# ---------------------------------------------------------------------------
class _FakeTransport(ILoRaTransport):
    """Records calls and stores callbacks. Does not actually transmit."""

    def __init__(
        self,
        node_id: str,
        params: LoRaParams,
        duty: DutyBucket,
        tx_power_dbm: float = 15.0,
    ) -> None:
        self._node_id = node_id
        self._params = params
        self._duty = duty
        self._tx_power = tx_power_dbm
        self._busy = False
        self._sent: list[bytes] = []
        self._recv_cbs: list[ReceiveCallback] = []
        self._tx_cbs: list[TxDoneCallback] = []

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
            raise TransportBusyError(f"{self._node_id} is busy")
        if not payload or len(payload) > 255:
            raise ValueError(f"payload length {len(payload)} out of range 1..255")
        self._sent.append(payload)

    def on_receive(self, callback: ReceiveCallback) -> None:
        self._recv_cbs.append(callback)

    def on_tx_done(self, callback: TxDoneCallback) -> None:
        self._tx_cbs.append(callback)

    # Test helpers (not part of the interface).
    def set_busy(self, busy: bool) -> None:
        self._busy = busy

    def fire_receive(self, payload: bytes, info: ReceptionInfo) -> None:
        for cb in self._recv_cbs:
            cb(payload, info)

    def fire_tx_done(self) -> None:
        for cb in self._tx_cbs:
            cb()


class _FakeChannel(IChannel):
    """Records registrations and broadcasts. Does not actually propagate signal."""

    def __init__(self) -> None:
        self._registry: dict[str, PositionGetter] = {}
        self._broadcasts: list[tuple[str, bytes]] = []

    def register(
        self, transport: ILoRaTransport, position_getter: PositionGetter
    ) -> None:
        self._registry[transport.node_id] = position_getter

    def unregister(self, transport: ILoRaTransport) -> None:
        self._registry.pop(transport.node_id, None)

    def broadcast(self, sender: ILoRaTransport, payload: bytes) -> None:
        self._broadcasts.append((sender.node_id, payload))


@pytest.fixture
def duty_bucket() -> DutyBucket:
    return DutyBucket(RegionPolicy.us915_polite())


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class TestExceptions:
    def test_transport_error_is_runtime_error(self) -> None:
        assert issubclass(TransportError, RuntimeError)

    def test_transport_busy_error_is_transport_error(self) -> None:
        assert issubclass(TransportBusyError, TransportError)

    def test_can_raise_and_catch_as_base(self) -> None:
        with pytest.raises(TransportError):
            raise TransportBusyError("radio busy")

    def test_error_message_preserved(self) -> None:
        with pytest.raises(TransportBusyError, match="node-A busy"):
            raise TransportBusyError("node-A busy")


# ---------------------------------------------------------------------------
# ReceptionInfo
# ---------------------------------------------------------------------------
class TestReceptionInfo:
    def test_valid_construction(self) -> None:
        info = ReceptionInfo(rssi_dbm=-80.0, snr_db=12.5, arrival_time_s=1.234)
        assert info.rssi_dbm == -80.0
        assert info.snr_db == 12.5
        assert info.arrival_time_s == 1.234

    def test_frozen_prevents_mutation(self) -> None:
        info = ReceptionInfo(-80.0, 12.5, 1.234)
        with pytest.raises(dataclasses.FrozenInstanceError):
            info.rssi_dbm = -90.0  # type: ignore[misc]

    def test_equality_by_value(self) -> None:
        a = ReceptionInfo(-80.0, 12.5, 1.234)
        b = ReceptionInfo(-80.0, 12.5, 1.234)
        c = ReceptionInfo(-80.0, 12.5, 1.235)
        assert a == b
        assert a != c

    def test_hashable(self) -> None:
        # frozen dataclasses are hashable by default; useful for dedup caches.
        info = ReceptionInfo(-80.0, 12.5, 1.234)
        assert hash(info) == hash(ReceptionInfo(-80.0, 12.5, 1.234))

    def test_no_sender_id_field(self) -> None:
        """Regression: sender_id was intentionally left out (see module docstring)."""
        assert "sender_id" not in {f.name for f in dataclasses.fields(ReceptionInfo)}


# ---------------------------------------------------------------------------
# ILoRaTransport ABC contract
# ---------------------------------------------------------------------------
class TestILoRaTransportABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            ILoRaTransport()  # type: ignore[abstract]

    def test_partial_subclass_fails_to_instantiate(self) -> None:
        # Missing on_receive and on_tx_done -> should be uninstantiable.
        class Incomplete(ILoRaTransport):
            @property
            def node_id(self) -> str:
                return "x"

            @property
            def params(self) -> LoRaParams:
                return PAPER_DEFAULT

            @property
            def duty(self) -> DutyBucket:
                return DutyBucket(RegionPolicy.us915_polite())

            @property
            def tx_power_dbm(self) -> float:
                return 15.0

            @property
            def is_busy(self) -> bool:
                return False

            def send(self, payload: bytes) -> None:
                pass

        with pytest.raises(TypeError, match="abstract"):
            Incomplete()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self, duty_bucket: DutyBucket) -> None:
        t = _FakeTransport("node-A", PAPER_DEFAULT, duty_bucket)
        assert isinstance(t, ILoRaTransport)


# ---------------------------------------------------------------------------
# IChannel ABC contract
# ---------------------------------------------------------------------------
class TestIChannelABC:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError, match="abstract"):
            IChannel()  # type: ignore[abstract]

    def test_complete_subclass_instantiates(self) -> None:
        assert isinstance(_FakeChannel(), IChannel)


# ---------------------------------------------------------------------------
# Fake transport sanity: prove the interface is actually usable
# ---------------------------------------------------------------------------
class TestFakeTransportSanity:
    def test_properties_return_construction_values(
        self, duty_bucket: DutyBucket
    ) -> None:
        t = _FakeTransport("node-A", PAPER_DEFAULT, duty_bucket, tx_power_dbm=14.0)
        assert t.node_id == "node-A"
        assert t.params is PAPER_DEFAULT
        assert t.duty is duty_bucket
        assert t.tx_power_dbm == 14.0
        assert t.is_busy is False

    def test_send_when_idle_succeeds(self, duty_bucket: DutyBucket) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        t.send(b"hello")
        assert t._sent == [b"hello"]

    def test_send_when_busy_raises(self, duty_bucket: DutyBucket) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        t.set_busy(True)
        with pytest.raises(TransportBusyError):
            t.send(b"hello")

    @pytest.mark.parametrize("payload", [b"", b"x" * 256])
    def test_send_invalid_payload_raises(
        self, duty_bucket: DutyBucket, payload: bytes
    ) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        with pytest.raises(ValueError, match="payload length"):
            t.send(payload)

    def test_on_receive_stores_callback(self, duty_bucket: DutyBucket) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        received: list[tuple[bytes, ReceptionInfo]] = []

        def cb(payload: bytes, info: ReceptionInfo) -> None:
            received.append((payload, info))

        t.on_receive(cb)
        info = ReceptionInfo(-80.0, 12.5, 1.0)
        t.fire_receive(b"hello", info)
        assert received == [(b"hello", info)]

    def test_multiple_receive_callbacks_fire_in_order(
        self, duty_bucket: DutyBucket
    ) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        order: list[str] = []
        t.on_receive(lambda _p, _i: order.append("first"))
        t.on_receive(lambda _p, _i: order.append("second"))
        t.fire_receive(b"x", ReceptionInfo(-80.0, 12.5, 1.0))
        assert order == ["first", "second"]

    def test_on_tx_done_fires_after_send(self, duty_bucket: DutyBucket) -> None:
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        fired: list[bool] = []
        t.on_tx_done(lambda: fired.append(True))
        t.send(b"hello")
        t.fire_tx_done()
        assert fired == [True]


# ---------------------------------------------------------------------------
# Fake channel sanity
# ---------------------------------------------------------------------------
class TestFakeChannelSanity:
    def test_register_and_broadcast(self, duty_bucket: DutyBucket) -> None:
        ch = _FakeChannel()
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        ch.register(t, lambda: (0.0, 0.0))
        ch.broadcast(t, b"hi")
        assert ch._broadcasts == [("A", b"hi")]

    def test_unregister_is_idempotent(self, duty_bucket: DutyBucket) -> None:
        ch = _FakeChannel()
        t = _FakeTransport("A", PAPER_DEFAULT, duty_bucket)
        ch.unregister(t)  # unknown -> no-op
        ch.register(t, lambda: (0.0, 0.0))
        ch.unregister(t)
        ch.unregister(t)  # again -> still no-op
        # Registry should be empty.
        assert "A" not in ch._registry


# ---------------------------------------------------------------------------
# Type aliases exist and can be used in annotations
# ---------------------------------------------------------------------------
class TestTypeAliases:
    def test_receive_callback_annotation(self) -> None:
        def cb(payload: bytes, info: ReceptionInfo) -> None:
            _ = (payload, info)

        annotated: ReceiveCallback = cb
        annotated(b"x", ReceptionInfo(-80.0, 12.5, 1.0))

    def test_tx_done_callback_annotation(self) -> None:
        def cb() -> None:
            pass

        annotated: TxDoneCallback = cb
        annotated()

    def test_position_getter_annotation(self) -> None:
        def cb() -> tuple[float, float]:
            return (1.0, 2.0)

        annotated: PositionGetter = cb
        assert annotated() == (1.0, 2.0)
