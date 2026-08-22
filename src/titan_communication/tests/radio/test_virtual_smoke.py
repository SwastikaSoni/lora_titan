"""End-to-end Tier-1 smoke test — Week-2 milestone.

Three scenarios prove the full stack composes correctly:

    1. Two nodes in range: A sends, B receives with correct RSSI/SNR/
       arrival_time, and A's duty bucket accounts for the airtime.
    2. Two nodes out of range: A sends, B never receives (link below
       sensitivity).
    3. Three nodes, two simultaneous senders: collision at the receiver
       drops both packets.

If all three pass, the plumbing from step 2 (airtime) through step 7
(queue not exercised here — smoke test is deliberately below the queue)
holds together. The queue integration is exercised in Week 3+.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest
import simpy
from titan_communication.mesh.frame import Frame, MessageClass
from titan_communication.radio.airtime import PAPER_DEFAULT, time_on_air_s
from titan_communication.radio.channel import (
    ChannelModel,
    PathLossModel,
    ShadowingModel,
)
from titan_communication.radio.duty import DutyBucket, RegionPolicy
from titan_communication.radio.transport import ReceptionInfo
from titan_communication.radio.virtual import VirtualChannel, VirtualLoRaTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class RxLog:
    """Records packets delivered to a node's on_receive callback."""

    payloads: list[bytes]
    infos: list[ReceptionInfo]

    @classmethod
    def new(cls) -> RxLog:
        return cls(payloads=[], infos=[])

    def append(self, payload: bytes, info: ReceptionInfo) -> None:
        self.payloads.append(payload)
        self.infos.append(info)


def _free_space_channel(env: simpy.Environment) -> VirtualChannel:
    """Deterministic free-space channel (no shadowing) at 915 MHz."""
    return VirtualChannel(
        env=env,
        model=ChannelModel(
            name="fs_915",
            pathloss=PathLossModel(
                path_loss_exponent=2.0,
                reference_distance_m=1.0,
                reference_pathloss_db=31.676,
            ),
            shadowing=ShadowingModel(std_dev_db=0.0),
        ),
        rng=None,  # deterministic (no shadowing samples needed)
    )


def _make_node(
    env: simpy.Environment,
    channel: VirtualChannel,
    node_id: str,
    position: tuple[float, float],
) -> tuple[VirtualLoRaTransport, RxLog]:
    """Build a node at a fixed position; return (transport, rx_log)."""
    duty = DutyBucket(RegionPolicy.us915_polite())
    transport = VirtualLoRaTransport(
        env=env,
        channel=channel,
        node_id=node_id,
        params=PAPER_DEFAULT,
        duty=duty,
        position_getter=lambda: position,
        tx_power_dbm=15.0,
    )
    log = RxLog.new()
    transport.on_receive(log.append)
    return transport, log


def _sos_payload() -> bytes:
    """A realistic SOS frame packed for TX."""
    return Frame.new(
        src_id=0x1001,
        dst_id=0xFFFF,
        message_class=MessageClass.SOS,
        payload=b"lat=42.41,lon=-83.14",
    ).pack()


# ---------------------------------------------------------------------------
# Scenario 1 — In range, single sender, verify delivery + accounting
# ---------------------------------------------------------------------------
class TestScenarioInRangeDelivery:
    def test_frame_arrives_at_receiver(self) -> None:
        env = simpy.Environment()
        channel = _free_space_channel(env)
        # 100 m separation: free-space PL = 31.676 + 40 = 71.676 dB;
        # RSSI = 15 - 71.676 = -56.68 dBm; sensitivity @ SF10/BW500 = -126 dBm.
        # ~70 dB of margin. Cannot fail on link budget.
        a, a_log = _make_node(env, channel, "A", (0.0, 0.0))
        _b, b_log = _make_node(env, channel, "B", (100.0, 0.0))

        payload = _sos_payload()

        def sender_process(env: simpy.Environment) -> object:
            a.send(payload)
            yield env.timeout(1.0)  # let delivery process complete

        env.process(sender_process(env))
        env.run()

        assert len(b_log.payloads) == 1, "B should have received exactly one packet"
        assert b_log.payloads[0] == payload
        assert len(a_log.payloads) == 0, "A shouldn't receive its own broadcast"

    def test_reception_info_is_sane(self) -> None:
        env = simpy.Environment()
        channel = _free_space_channel(env)
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        _b, b_log = _make_node(env, channel, "B", (100.0, 0.0))

        payload = _sos_payload()

        def sender_process(env: simpy.Environment) -> object:
            a.send(payload)
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()

        info = b_log.infos[0]
        # RSSI should be roughly -56.7 dBm (see distance calc above).
        assert info.rssi_dbm == pytest.approx(-56.676, abs=0.1)
        # SNR = RSSI - noise_floor(500 kHz, NF=6) = -56.676 - (-111.01) ~= 54.3
        assert info.snr_db == pytest.approx(54.3, abs=0.5)
        # Arrival time equals airtime (send happened at t=0).
        expected_airtime = time_on_air_s(PAPER_DEFAULT, len(payload))
        assert info.arrival_time_s == pytest.approx(expected_airtime, abs=1e-6)

    def test_duty_bucket_charged_for_transmission(self) -> None:
        env = simpy.Environment()
        channel = _free_space_channel(env)
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        _make_node(env, channel, "B", (100.0, 0.0))

        payload = _sos_payload()
        expected_airtime = time_on_air_s(PAPER_DEFAULT, len(payload))

        def sender_process(env: simpy.Environment) -> object:
            a.send(payload)
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()

        assert a.duty.airtime_used_s(now_s=env.now) == pytest.approx(
            expected_airtime
        )


# ---------------------------------------------------------------------------
# Scenario 2 — Out of range: link budget fails, no delivery
# ---------------------------------------------------------------------------
class TestScenarioOutOfRange:
    def test_far_receiver_gets_nothing(self) -> None:
        env = simpy.Environment()
        # Urban NLOS channel so we can push nodes out of range with a
        # tractable distance. n=3.5, PL0=40, no shadowing.
        channel = VirtualChannel(
            env=env,
            model=ChannelModel(
                name="urban_dry",
                pathloss=PathLossModel(3.5, 1.0, 40.0),
                shadowing=ShadowingModel(0.0),
            ),
        )
        # 3 km separation on n=3.5: PL = 40 + 35*log10(3000) = 40 + 121.9
        # = 161.9 dB. RSSI = 15 - 161.9 = -146.9 dBm. Below sensitivity
        # of -126 dBm at SF10/BW500 by ~20 dB.
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        _b, b_log = _make_node(env, channel, "B", (3000.0, 0.0))

        def sender_process(env: simpy.Environment) -> object:
            a.send(_sos_payload())
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()

        assert len(b_log.payloads) == 0, (
            f"B should not decode packet at 3 km in urban model, got "
            f"{len(b_log.payloads)} deliveries"
        )

    def test_sender_still_busy_and_duty_charged_even_when_no_receiver(self) -> None:
        """The medium doesn't know or care whether anyone was listening."""
        env = simpy.Environment()
        channel = VirtualChannel(
            env=env,
            model=ChannelModel(
                name="urban_dry",
                pathloss=PathLossModel(3.5, 1.0, 40.0),
                shadowing=ShadowingModel(0.0),
            ),
        )
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        _make_node(env, channel, "B", (3000.0, 0.0))  # too far

        payload = _sos_payload()

        def sender_process(env: simpy.Environment) -> object:
            a.send(payload)
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()
        assert a.duty.airtime_used_s(now_s=env.now) > 0.0


# ---------------------------------------------------------------------------
# Scenario 3 — Collision: two simultaneous senders wipe each other out
# ---------------------------------------------------------------------------
class TestScenarioCollision:
    def test_simultaneous_senders_both_lost_at_receiver(self) -> None:
        env = simpy.Environment()
        channel = _free_space_channel(env)
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        c, _ = _make_node(env, channel, "C", (50.0, 0.0))
        _b, b_log = _make_node(env, channel, "B", (100.0, 0.0))

        payload_a = Frame.new(1, 0xFFFF, MessageClass.CTRL, b"from A").pack()
        payload_c = Frame.new(3, 0xFFFF, MessageClass.CTRL, b"from C").pack()

        def sender_process(env: simpy.Environment) -> object:
            # Both fire at t=0 -> overlapping on-air intervals at B.
            a.send(payload_a)
            c.send(payload_c)
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()

        assert len(b_log.payloads) == 0, (
            f"B should have dropped both colliding packets, got "
            f"{len(b_log.payloads)} deliveries"
        )

    def test_staggered_senders_both_arrive(self) -> None:
        """No overlap in on-air time -> no collision -> both delivered."""
        env = simpy.Environment()
        channel = _free_space_channel(env)
        a, _ = _make_node(env, channel, "A", (0.0, 0.0))
        c, _ = _make_node(env, channel, "C", (50.0, 0.0))
        _b, b_log = _make_node(env, channel, "B", (100.0, 0.0))

        payload_a = Frame.new(1, 0xFFFF, MessageClass.CTRL, b"from A").pack()
        payload_c = Frame.new(3, 0xFFFF, MessageClass.CTRL, b"from C").pack()

        airtime = time_on_air_s(PAPER_DEFAULT, len(payload_a))

        def sender_process(env: simpy.Environment) -> object:
            a.send(payload_a)
            # Wait until A's transmission fully clears the air, then C sends.
            yield env.timeout(airtime + 0.001)
            c.send(payload_c)
            yield env.timeout(1.0)

        env.process(sender_process(env))
        env.run()
        assert len(b_log.payloads) == 2, (
            f"B should have received both staggered packets, got "
            f"{len(b_log.payloads)}"
        )
        # Order preserved.
        assert b_log.payloads[0] == payload_a
        assert b_log.payloads[1] == payload_c
