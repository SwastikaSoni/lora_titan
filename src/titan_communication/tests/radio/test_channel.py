"""Tests for the log-distance + shadowing channel model and sensitivity."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from titan_communication.radio.channel import (
    DEFAULT_NOISE_FIGURE_DB,
    ChannelModel,
    PathLossModel,
    ShadowingModel,
    demod_snr_db,
    is_decodable,
    noise_floor_dbm,
    sensitivity_dbm,
    snr_db,
)

# Physical constants for the free-space hand-check.
SPEED_OF_LIGHT_M_S: float = 299_792_458.0


# ---------------------------------------------------------------------------
# PathLossModel validation
# ---------------------------------------------------------------------------
class TestPathLossModelValidation:
    def test_valid_construction(self) -> None:
        m = PathLossModel(2.0, 1.0, 31.67)
        assert m.path_loss_exponent == 2.0

    @pytest.mark.parametrize("n", [0.0, -0.5, -1.0])
    def test_invalid_exponent_raises(self, n: float) -> None:
        with pytest.raises(ValueError, match="path_loss_exponent"):
            PathLossModel(n, 1.0, 31.67)

    @pytest.mark.parametrize("d0", [0.0, -1.0])
    def test_invalid_reference_distance_raises(self, d0: float) -> None:
        with pytest.raises(ValueError, match="reference_distance_m"):
            PathLossModel(2.0, d0, 31.67)


# ---------------------------------------------------------------------------
# Path loss reference values
# ---------------------------------------------------------------------------
class TestPathLossReferenceValues:
    def test_free_space_at_reference_distance(self) -> None:
        # d = d0 -> log10(1) = 0 -> PL = PL0
        m = PathLossModel(2.0, 1.0, 31.676)
        assert m.pathloss_db(1.0) == pytest.approx(31.676)

    def test_free_space_915mhz_at_100m(self) -> None:
        # Friis at 915 MHz. lambda = c/f, PL0(1m) = 20 log10(4pi/lambda).
        # At d=100m: PL = PL0 + 20*log10(100) = PL0 + 40.
        freq_hz = 915e6
        wavelength = SPEED_OF_LIGHT_M_S / freq_hz
        pl0 = 20.0 * math.log10(4.0 * math.pi / wavelength)
        m = PathLossModel(2.0, 1.0, pl0)
        assert m.pathloss_db(100.0) == pytest.approx(pl0 + 40.0, abs=1e-9)

    def test_free_space_at_1000m(self) -> None:
        # n=2, PL0=31.676 @ 1m -> at 1000m: PL = 31.676 + 60 = 91.676
        m = PathLossModel(2.0, 1.0, 31.676)
        assert m.pathloss_db(1000.0) == pytest.approx(91.676, abs=1e-9)

    def test_higher_exponent_gives_higher_loss(self) -> None:
        # Same PL0, higher n -> more loss at any d > d0.
        low_n = PathLossModel(2.0, 1.0, 40.0)
        high_n = PathLossModel(3.5, 1.0, 40.0)
        assert high_n.pathloss_db(100.0) > low_n.pathloss_db(100.0)

    def test_exponent_2p5_at_100m(self) -> None:
        # n=2.5, PL0=40 @ 1m -> at 100m: PL = 40 + 25*log10(100) = 40 + 50 = 90
        m = PathLossModel(2.5, 1.0, 40.0)
        assert m.pathloss_db(100.0) == pytest.approx(90.0, abs=1e-9)


# ---------------------------------------------------------------------------
# Distance edge cases
# ---------------------------------------------------------------------------
class TestPathLossDistanceEdgeCases:
    def test_distance_below_reference_clamps_to_reference(self) -> None:
        m = PathLossModel(2.0, 1.0, 31.676)
        # Any d < 1m should give PL = PL0 (not less).
        assert m.pathloss_db(0.5) == pytest.approx(31.676)
        assert m.pathloss_db(0.01) == pytest.approx(31.676)

    def test_distance_exactly_reference(self) -> None:
        m = PathLossModel(2.0, 1.0, 31.676)
        assert m.pathloss_db(1.0) == pytest.approx(31.676)

    def test_zero_distance_raises(self) -> None:
        m = PathLossModel(2.0, 1.0, 31.676)
        with pytest.raises(ValueError, match="distance_m"):
            m.pathloss_db(0.0)

    def test_negative_distance_raises(self) -> None:
        m = PathLossModel(2.0, 1.0, 31.676)
        with pytest.raises(ValueError, match="distance_m"):
            m.pathloss_db(-10.0)


# ---------------------------------------------------------------------------
# Shadowing
# ---------------------------------------------------------------------------
class TestShadowingModel:
    def test_negative_std_raises(self) -> None:
        with pytest.raises(ValueError, match="std_dev_db"):
            ShadowingModel(-1.0)

    def test_zero_std_returns_zero(self) -> None:
        s = ShadowingModel(0.0)
        rng = np.random.default_rng(42)
        assert s.sample_db(rng) == 0.0

    def test_reproducible_with_seed(self) -> None:
        s = ShadowingModel(5.0)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        assert s.sample_db(rng1) == s.sample_db(rng2)

    def test_different_seeds_give_different_samples(self) -> None:
        s = ShadowingModel(5.0)
        rng1 = np.random.default_rng(1)
        rng2 = np.random.default_rng(2)
        # Overwhelmingly likely to differ; if this flakes we've bigger problems.
        assert s.sample_db(rng1) != s.sample_db(rng2)

    def test_sample_statistics_over_large_n(self) -> None:
        # Draw 10000 samples of a Gaussian with sigma=5. Expect mean ~ 0
        # (SE = 5/sqrt(10000) = 0.05, so within +/- 0.15 with p ~ 0.997)
        # and std ~ 5 (within ~5%).
        s = ShadowingModel(5.0)
        rng = np.random.default_rng(12345)
        samples = np.array([s.sample_db(rng) for _ in range(10_000)])
        assert abs(samples.mean()) < 0.15
        assert 4.75 < samples.std(ddof=1) < 5.25


# ---------------------------------------------------------------------------
# ChannelModel: pathloss + shadowing combined
# ---------------------------------------------------------------------------
class TestChannelModel:
    def _free_space(self) -> ChannelModel:
        return ChannelModel(
            name="free_space",
            pathloss=PathLossModel(2.0, 1.0, 31.676),
            shadowing=ShadowingModel(0.0),
        )

    def test_deterministic_when_rng_none(self) -> None:
        c = self._free_space()
        # Same call twice, no rng -> identical.
        assert c.pathloss_db(100.0) == c.pathloss_db(100.0)

    def test_shadowing_applied_when_rng_given(self) -> None:
        c = ChannelModel(
            name="noisy",
            pathloss=PathLossModel(2.0, 1.0, 40.0),
            shadowing=ShadowingModel(6.0),
        )
        rng = np.random.default_rng(42)
        pl_shadowed = c.pathloss_db(100.0, rng=rng)
        pl_mean = c.pathloss_db(100.0)  # no rng
        # Shadowing draw is almost certainly nonzero.
        assert pl_shadowed != pl_mean

    def test_shadowing_reproducible(self) -> None:
        c = ChannelModel(
            name="noisy",
            pathloss=PathLossModel(2.0, 1.0, 40.0),
            shadowing=ShadowingModel(6.0),
        )
        rng1 = np.random.default_rng(99)
        rng2 = np.random.default_rng(99)
        assert c.pathloss_db(100.0, rng=rng1) == c.pathloss_db(100.0, rng=rng2)

    def test_rssi_matches_tx_minus_pl(self) -> None:
        c = self._free_space()
        # TX = 15 dBm at 100m, PL = 31.676 + 40 = 71.676
        # RSSI = 15 - 71.676 = -56.676 dBm
        assert c.rssi_dbm(15.0, 100.0) == pytest.approx(-56.676, abs=1e-9)


# ---------------------------------------------------------------------------
# Noise floor
# ---------------------------------------------------------------------------
class TestNoiseFloor:
    def test_bw125khz_nf6(self) -> None:
        # -174 + 10*log10(125000) + 6 = -174 + 50.9691 + 6 = -117.0309
        assert noise_floor_dbm(125_000, 6.0) == pytest.approx(-117.0309, abs=1e-4)

    def test_bw500khz_nf6(self) -> None:
        # -174 + 10*log10(500000) + 6 = -174 + 56.9897 + 6 = -111.0103
        assert noise_floor_dbm(500_000, 6.0) == pytest.approx(-111.0103, abs=1e-4)

    def test_higher_bw_higher_noise(self) -> None:
        assert noise_floor_dbm(500_000) > noise_floor_dbm(125_000)

    def test_higher_nf_higher_noise(self) -> None:
        assert noise_floor_dbm(125_000, 10.0) > noise_floor_dbm(125_000, 6.0)

    @pytest.mark.parametrize("bw", [0, -1000])
    def test_invalid_bw_raises(self, bw: float) -> None:
        with pytest.raises(ValueError, match="bandwidth_hz"):
            noise_floor_dbm(bw, 6.0)

    def test_default_noise_figure_is_6(self) -> None:
        assert DEFAULT_NOISE_FIGURE_DB == 6.0
        assert noise_floor_dbm(125_000) == noise_floor_dbm(125_000, 6.0)


# ---------------------------------------------------------------------------
# Demod SNR floor
# ---------------------------------------------------------------------------
class TestDemodSNR:
    @pytest.mark.parametrize(
        "sf,expected",
        [(6, -5.0), (7, -7.5), (8, -10.0), (9, -12.5),
         (10, -15.0), (11, -17.5), (12, -20.0)],
    )
    def test_datasheet_values(self, sf: int, expected: float) -> None:
        # SX1276 datasheet §5.5.5 Table 13.
        assert demod_snr_db(sf) == expected

    def test_higher_sf_lower_snr_floor(self) -> None:
        # Higher SF processes gain more spread, so demod tolerates lower SNR.
        assert demod_snr_db(12) < demod_snr_db(7)

    @pytest.mark.parametrize("sf", [0, 5, 13, -1, 100])
    def test_invalid_sf_raises(self, sf: int) -> None:
        with pytest.raises(ValueError, match="spreading_factor"):
            demod_snr_db(sf)


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------
class TestSensitivity:
    def test_sf10_bw500_nf6(self) -> None:
        # Noise floor -111.0103 + demod -15 = -126.0103 dBm.
        # Datasheet lists ~-125 dBm for SF=10/BW=500 (worst case),
        # so our formula is within 1 dB — reasonable given NF spread.
        assert sensitivity_dbm(10, 500_000, 6.0) == pytest.approx(
            -126.0103, abs=1e-4
        )

    def test_sf7_bw125_nf6(self) -> None:
        # -117.0309 + -7.5 = -124.5309
        assert sensitivity_dbm(7, 125_000, 6.0) == pytest.approx(
            -124.5309, abs=1e-4
        )

    def test_sf12_bw125_nf6(self) -> None:
        # -117.0309 + -20.0 = -137.0309 — the long-range corner.
        assert sensitivity_dbm(12, 125_000, 6.0) == pytest.approx(
            -137.0309, abs=1e-4
        )

    def test_higher_sf_more_sensitive(self) -> None:
        # Same BW, higher SF -> better (more negative) sensitivity.
        assert sensitivity_dbm(12, 125_000) < sensitivity_dbm(7, 125_000)

    def test_higher_bw_less_sensitive(self) -> None:
        # Same SF, higher BW -> worse (less negative) sensitivity.
        assert sensitivity_dbm(10, 500_000) > sensitivity_dbm(10, 125_000)


# ---------------------------------------------------------------------------
# SNR helper
# ---------------------------------------------------------------------------
class TestSNR:
    def test_snr_is_rssi_minus_noise_floor(self) -> None:
        # RSSI = -100 dBm, noise floor at BW=125k/NF=6 = -117.03
        # SNR = -100 - (-117.03) = 17.03 dB
        assert snr_db(-100.0, 125_000, 6.0) == pytest.approx(17.0309, abs=1e-4)

    def test_snr_negative_when_below_noise(self) -> None:
        # LoRa can decode below noise floor thanks to spread-spectrum gain.
        assert snr_db(-130.0, 125_000, 6.0) < 0


# ---------------------------------------------------------------------------
# is_decodable
# ---------------------------------------------------------------------------
class TestIsDecodable:
    def test_strong_signal_decodable(self) -> None:
        # -80 dBm is way above SF=10/BW=500 sensitivity of -126.
        assert is_decodable(-80.0, 10, 500_000)

    def test_weak_signal_not_decodable(self) -> None:
        # -140 dBm at SF=10/BW=500 (sensitivity -126) -> not decodable.
        assert not is_decodable(-140.0, 10, 500_000)

    def test_exactly_at_sensitivity_decodable(self) -> None:
        # Boundary inclusive.
        s = sensitivity_dbm(10, 500_000, 6.0)
        assert is_decodable(s, 10, 500_000, 6.0)

    def test_just_below_sensitivity_not_decodable(self) -> None:
        s = sensitivity_dbm(10, 500_000, 6.0)
        assert not is_decodable(s - 0.01, 10, 500_000, 6.0)


# ---------------------------------------------------------------------------
# YAML preset loading
# ---------------------------------------------------------------------------
class TestYAMLPresetLoading:
    def test_load_free_space_from_inline_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
models:
  free_space:
    path_loss_exponent: 2.0
    reference_distance_m: 1.0
    reference_pathloss_db: 31.676
    shadowing_std_db: 0.0
"""
        p = tmp_path / "channel.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        c = ChannelModel.from_preset(p, "free_space")
        assert c.name == "free_space"
        assert c.pathloss.path_loss_exponent == 2.0
        assert c.shadowing.std_dev_db == 0.0
        # And the loaded model computes the same PL as an inline one.
        assert c.pathloss_db(100.0) == pytest.approx(71.676, abs=1e-9)

    def test_missing_preset_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "channel.yaml"
        p.write_text("models: {}\n", encoding="utf-8")
        with pytest.raises(KeyError, match="nonexistent"):
            ChannelModel.from_preset(p, "nonexistent")

    def test_load_from_real_configs_file(self) -> None:
        cfg = Path(__file__).resolve().parents[4] / "configs" / "channel.yaml"
        if not cfg.exists():
            pytest.skip(f"configs file not found at {cfg}")
        c = ChannelModel.from_preset(cfg, "free_space_915mhz")
        assert c.pathloss.path_loss_exponent == 2.0
        assert c.shadowing.std_dev_db == 0.0


# ---------------------------------------------------------------------------
# Integration: paper's TX power at plausible mesh distances
# ---------------------------------------------------------------------------
class TestIntegrationWithPaper:
    """Ties channel.py to airtime.py's PAPER_DEFAULT — realistic link budget."""

    def test_paper_tx_at_100m_free_space_decodable(self) -> None:
        # TX = 15 dBm, free space at 100m -> RSSI = 15 - 71.676 = -56.676.
        # Comfortably above SF=10/BW=500 sensitivity of -126 dBm.
        c = ChannelModel(
            name="fs",
            pathloss=PathLossModel(2.0, 1.0, 31.676),
            shadowing=ShadowingModel(0.0),
        )
        rssi = c.rssi_dbm(tx_power_dbm=15.0, distance_m=100.0)
        assert is_decodable(rssi, spreading_factor=10, bandwidth_hz=500_000)

    def test_paper_tx_at_urban_2km_may_not_decode(self) -> None:
        # Urban NLOS n=3.5 at 2km + shadowing headroom -> should be near
        # or below the decoding cliff, demonstrating the model produces
        # sensible link-budget numbers.
        c = ChannelModel(
            name="urban",
            pathloss=PathLossModel(3.5, 1.0, 40.0),
            shadowing=ShadowingModel(0.0),
        )
        rssi = c.rssi_dbm(tx_power_dbm=15.0, distance_m=2000.0)
        # PL at 2km = 40 + 35*log10(2000) = 40 + 115.5 = 155.5 dB
        # RSSI = 15 - 155.5 = -140.5 dBm; sensitivity at SF10/BW500 is -126.
        assert not is_decodable(rssi, spreading_factor=10, bandwidth_hz=500_000)
        # But at SF=12/BW=125 (sensitivity -137), still marginal.
        assert not is_decodable(rssi, spreading_factor=12, bandwidth_hz=125_000)
