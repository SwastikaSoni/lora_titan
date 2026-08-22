"""Tests for the LoRa duty-cycle / dwell-time bucket."""

from __future__ import annotations

import pytest
from titan_communication.radio.airtime import PAPER_DEFAULT, time_on_air_s
from titan_communication.radio.duty import DutyBucket, DutyViolationError, RegionPolicy


# ---------------------------------------------------------------------------
# RegionPolicy validation
# ---------------------------------------------------------------------------
class TestRegionPolicyValidation:
    def test_valid_construction(self) -> None:
        p = RegionPolicy(
            name="test", duty_cycle_fraction=0.01, duty_window_s=3600.0,
            max_dwell_time_s=0.400,
        )
        assert p.name == "test"

    @pytest.mark.parametrize("frac", [-0.01, 0.0, 1.01, 2.0])
    def test_invalid_duty_fraction_raises(self, frac: float) -> None:
        with pytest.raises(ValueError, match="duty_cycle_fraction"):
            RegionPolicy("bad", frac, 3600.0, None)

    @pytest.mark.parametrize("window", [0.0, -1.0])
    def test_invalid_window_raises(self, window: float) -> None:
        with pytest.raises(ValueError, match="duty_window_s"):
            RegionPolicy("bad", 0.01, window, None)

    @pytest.mark.parametrize("dwell", [0.0, -0.1])
    def test_invalid_dwell_raises(self, dwell: float) -> None:
        with pytest.raises(ValueError, match="max_dwell_time_s"):
            RegionPolicy("bad", None, 3600.0, dwell)


# ---------------------------------------------------------------------------
# RegionPolicy factory methods
# ---------------------------------------------------------------------------
class TestRegionPolicyFactories:
    def test_us915_polite(self) -> None:
        p = RegionPolicy.us915_polite()
        assert p.duty_cycle_fraction == 0.10
        assert p.duty_window_s == 3600.0
        assert p.max_dwell_time_s is None

    def test_us915_fcc_raw_has_no_caps(self) -> None:
        p = RegionPolicy.us915_fcc_raw()
        assert p.duty_cycle_fraction is None
        assert p.max_dwell_time_s is None

    def test_us915_hopping_has_400ms_dwell(self) -> None:
        p = RegionPolicy.us915_hopping()
        assert p.duty_cycle_fraction is None
        assert p.max_dwell_time_s == 0.400

    def test_in865_matches_lora_alliance_1pct(self) -> None:
        p = RegionPolicy.in865()
        assert p.duty_cycle_fraction == 0.01
        assert p.duty_window_s == 3600.0

    def test_eu868_matches_etsi_1pct(self) -> None:
        p = RegionPolicy.eu868()
        assert p.duty_cycle_fraction == 0.01
        assert p.duty_window_s == 3600.0


# ---------------------------------------------------------------------------
# max_airtime_in_window_s
# ---------------------------------------------------------------------------
class TestMaxAirtimeInWindow:
    def test_no_cap_is_inf(self) -> None:
        assert RegionPolicy.us915_fcc_raw().max_airtime_in_window_s() == float("inf")

    def test_1pct_1hr_is_36s(self) -> None:
        # 0.01 * 3600 = 36 seconds
        assert RegionPolicy.in865().max_airtime_in_window_s() == pytest.approx(36.0)

    def test_10pct_1hr_is_360s(self) -> None:
        assert RegionPolicy.us915_polite().max_airtime_in_window_s() == pytest.approx(360.0)


# ---------------------------------------------------------------------------
# DutyBucket basics
# ---------------------------------------------------------------------------
class TestDutyBucketBasics:
    def test_fresh_bucket_can_transmit(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        assert b.can_transmit(now_s=0.0, airtime_s=0.1)

    def test_fresh_bucket_zero_utilization(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        assert b.utilization(now_s=0.0) == 0.0

    def test_fresh_bucket_zero_airtime(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        assert b.airtime_used_s(now_s=0.0) == 0.0

    def test_negative_airtime_raises(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        with pytest.raises(ValueError, match="airtime_s"):
            b.can_transmit(now_s=0.0, airtime_s=-0.1)

    def test_zero_airtime_raises(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        with pytest.raises(ValueError, match="airtime_s"):
            b.can_transmit(now_s=0.0, airtime_s=0.0)


# ---------------------------------------------------------------------------
# Accounting after registration
# ---------------------------------------------------------------------------
class TestDutyBucketAccounting:
    def test_register_increments_airtime(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=10.0, airtime_s=0.5)
        assert b.airtime_used_s(now_s=10.0) == pytest.approx(0.5)

    def test_register_updates_utilization(self) -> None:
        b = DutyBucket(RegionPolicy.in865())  # 36 s budget
        b.register(now_s=10.0, airtime_s=3.6)  # 10% of budget
        assert b.utilization(now_s=10.0) == pytest.approx(0.10)

    def test_multiple_registers_accumulate(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=1.0, airtime_s=0.1)
        b.register(now_s=2.0, airtime_s=0.2)
        b.register(now_s=3.0, airtime_s=0.3)
        assert b.airtime_used_s(now_s=3.0) == pytest.approx(0.6)

    def test_monotonic_timestamp_enforced(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=10.0, airtime_s=0.1)
        with pytest.raises(ValueError, match="non-decreasing"):
            b.register(now_s=5.0, airtime_s=0.1)


# ---------------------------------------------------------------------------
# Cap enforcement
# ---------------------------------------------------------------------------
class TestCapEnforcement:
    def test_register_exactly_at_cap_ok(self) -> None:
        # 36 s budget, register exactly 36 s -> allowed
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=36.0)
        assert b.utilization(now_s=0.0) == pytest.approx(1.0)

    def test_register_over_cap_raises(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=35.9)
        with pytest.raises(DutyViolationError):
            b.register(now_s=1.0, airtime_s=0.2)  # 35.9 + 0.2 > 36.0

    def test_can_transmit_returns_false_at_cap(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=36.0)
        assert not b.can_transmit(now_s=1.0, airtime_s=0.001)

    def test_try_register_returns_false_at_cap(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=36.0)
        assert b.try_register(now_s=1.0, airtime_s=0.001) is False

    def test_try_register_does_not_modify_state_when_blocked(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=36.0)
        before = b.airtime_used_s(now_s=1.0)
        b.try_register(now_s=1.0, airtime_s=0.001)
        after = b.airtime_used_s(now_s=1.0)
        assert before == after

    def test_try_register_returns_true_when_allowed(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        assert b.try_register(now_s=0.0, airtime_s=0.1) is True
        assert b.airtime_used_s(now_s=0.0) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Rolling window eviction
# ---------------------------------------------------------------------------
class TestRollingEviction:
    def test_entry_still_in_before_window(self) -> None:
        b = DutyBucket(RegionPolicy.in865())  # 3600 s window
        b.register(now_s=100.0, airtime_s=5.0)
        # At t = 100 + 3599, still inside window
        assert b.airtime_used_s(now_s=3699.0) == pytest.approx(5.0)

    def test_entry_evicts_exactly_at_window_boundary(self) -> None:
        # Entry recorded at end_time=100, window=3600. At now=3700,
        # cutoff = 100, and our eviction uses <= so it drops out.
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=100.0, airtime_s=5.0)
        assert b.airtime_used_s(now_s=3700.0) == pytest.approx(0.0)

    def test_after_eviction_new_transmission_allowed(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=36.0)  # cap
        assert not b.can_transmit(now_s=1.0, airtime_s=0.001)
        # Advance past the window; the old entry should evict.
        assert b.can_transmit(now_s=3600.0, airtime_s=0.001)

    def test_partial_eviction_frees_partial_budget(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=20.0)
        b.register(now_s=1800.0, airtime_s=15.0)  # total 35 s used
        # At t = 3601, only the first entry has evicted (at cutoff 1); the
        # 1800-timestamp one is still in. Budget: 36 - 15 = 21 s free.
        assert b.can_transmit(now_s=3601.0, airtime_s=20.0)
        assert not b.can_transmit(now_s=3601.0, airtime_s=22.0)


# ---------------------------------------------------------------------------
# Dwell time enforcement
# ---------------------------------------------------------------------------
class TestDwellEnforcement:
    def test_us915_hopping_blocks_450ms(self) -> None:
        b = DutyBucket(RegionPolicy.us915_hopping())
        assert not b.can_transmit(now_s=0.0, airtime_s=0.450)

    def test_us915_hopping_allows_400ms(self) -> None:
        # Exactly at limit — allowed.
        b = DutyBucket(RegionPolicy.us915_hopping())
        assert b.can_transmit(now_s=0.0, airtime_s=0.400)

    def test_us915_hopping_allows_399ms(self) -> None:
        b = DutyBucket(RegionPolicy.us915_hopping())
        assert b.can_transmit(now_s=0.0, airtime_s=0.399)

    def test_register_dwell_violation_raises(self) -> None:
        b = DutyBucket(RegionPolicy.us915_hopping())
        with pytest.raises(DutyViolationError):
            b.register(now_s=0.0, airtime_s=0.500)


# ---------------------------------------------------------------------------
# Uncapped region (US915 raw)
# ---------------------------------------------------------------------------
class TestNoCap:
    def test_us915_raw_always_allowed(self) -> None:
        b = DutyBucket(RegionPolicy.us915_fcc_raw())
        assert b.can_transmit(now_s=0.0, airtime_s=1000.0)
        assert b.can_transmit(now_s=0.0, airtime_s=0.001)

    def test_us915_raw_utilization_always_zero(self) -> None:
        b = DutyBucket(RegionPolicy.us915_fcc_raw())
        b.register(now_s=0.0, airtime_s=100.0)
        assert b.utilization(now_s=0.0) == 0.0

    def test_us915_raw_still_tracks_airtime(self) -> None:
        # Even without a cap, airtime is tracked for the bake-off's
        # "duty utilisation per node" metric.
        b = DutyBucket(RegionPolicy.us915_fcc_raw())
        b.register(now_s=0.0, airtime_s=5.0)
        assert b.airtime_used_s(now_s=0.0) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# time_until_next_available
# ---------------------------------------------------------------------------
class TestTimeUntilNextAvailable:
    def test_zero_when_room(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        assert b.time_until_next_available(now_s=0.0, airtime_s=0.1) == 0.0

    def test_zero_when_no_cap(self) -> None:
        b = DutyBucket(RegionPolicy.us915_fcc_raw())
        assert b.time_until_next_available(now_s=0.0, airtime_s=1000.0) == 0.0

    def test_positive_wait_when_full(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=100.0, airtime_s=36.0)
        # We want another 0.1 s. Oldest entry ends at 100; it exits at
        # 100 + 3600 = 3700. At t = 200, wait = 3500 s.
        wait = b.time_until_next_available(now_s=200.0, airtime_s=0.1)
        assert wait == pytest.approx(3500.0)

    def test_none_when_dwell_violation(self) -> None:
        b = DutyBucket(RegionPolicy.us915_hopping())
        assert b.time_until_next_available(now_s=0.0, airtime_s=0.500) is None

    def test_none_when_single_tx_exceeds_budget(self) -> None:
        # 40 s tx at IN865 (36 s budget) — impossible even in an empty window.
        b = DutyBucket(RegionPolicy.in865())
        assert b.time_until_next_available(now_s=0.0, airtime_s=40.0) is None

    def test_matches_actual_availability(self) -> None:
        # Sanity: waiting the reported duration makes can_transmit True.
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=100.0, airtime_s=36.0)
        wait = b.time_until_next_available(now_s=200.0, airtime_s=0.1)
        assert wait is not None
        assert b.can_transmit(now_s=200.0 + wait, airtime_s=0.1)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------
class TestReset:
    def test_reset_clears_history(self) -> None:
        b = DutyBucket(RegionPolicy.in865())
        b.register(now_s=0.0, airtime_s=10.0)
        b.reset()
        assert b.airtime_used_s(now_s=0.0) == 0.0
        assert b.can_transmit(now_s=0.0, airtime_s=36.0)


# ---------------------------------------------------------------------------
# Integration: use the paper's real packet size against real region policies
# ---------------------------------------------------------------------------
class TestIntegrationWithPaperPacket:
    """These tests connect step 2 (airtime.py) to step 3 (duty.py)."""

    def test_paper_packet_fits_in_us915_polite(self) -> None:
        # Paper packet: SF=10/BW=500/CR=4/5/PL=32 -> 113.152 ms.
        # US915-polite budget: 360 s per hour. Trivially fits.
        airtime = time_on_air_s(PAPER_DEFAULT, 32)
        b = DutyBucket(RegionPolicy.us915_polite())
        assert b.can_transmit(now_s=0.0, airtime_s=airtime)

    def test_paper_packet_never_hits_dwell_cap(self) -> None:
        # Paper packet is 113 ms; well under the 400 ms US915 hopping cap.
        airtime = time_on_air_s(PAPER_DEFAULT, 32)
        b = DutyBucket(RegionPolicy.us915_hopping())
        assert b.can_transmit(now_s=0.0, airtime_s=airtime)

    def test_in865_hourly_packet_budget_for_paper(self) -> None:
        # IN865 1% = 36 s/hour budget. Paper packet is 0.113152 s.
        # Max packets per hour = floor(36 / 0.113152) = 318.
        airtime = time_on_air_s(PAPER_DEFAULT, 32)
        b = DutyBucket(RegionPolicy.in865())
        n_sent = 0
        t = 0.0
        while b.try_register(now_s=t, airtime_s=airtime):
            n_sent += 1
            t += 0.001  # 1 ms between sends; well under 3600 s window
        # 36 / 0.113152 = 318.17..., so 318 fit, 319th blocks.
        assert n_sent == 318
