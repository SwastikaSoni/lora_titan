"""Datasheet-verified tests for the Semtech time-on-air formula.

Every reference ToA value here is worked out step-by-step in the test's
own comment, and cross-checked against the Semtech LoRa Modem Calculator
Tool (SX1276 mode). The tolerance is 0.001 ms because the formula is
exact integer arithmetic on symbol counts, and only T_sym's
floating-point division introduces any real numerical error.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from titan_communication.radio.airtime import (
    PAPER_DEFAULT,
    CodingRate,
    LoRaParams,
    payload_symbol_count,
    preamble_duration_s,
    time_on_air_ms,
    time_on_air_s,
)

# Tight because the reference values are exact hand calculations.
TOL_MS = 1e-3


# ---------------------------------------------------------------------------
# CodingRate
# ---------------------------------------------------------------------------
class TestCodingRate:
    def test_from_str_valid(self) -> None:
        assert CodingRate.from_str("4/5") == CodingRate.CR_4_5
        assert CodingRate.from_str("4/6") == CodingRate.CR_4_6
        assert CodingRate.from_str("4/7") == CodingRate.CR_4_7
        assert CodingRate.from_str("4/8") == CodingRate.CR_4_8

    def test_from_str_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown coding rate"):
            CodingRate.from_str("4/9")

    def test_integer_value_matches_semtech_cr_parameter(self) -> None:
        # Semtech "CR" parameter in the formula is 1..4; actual rate = 4/(4+CR).
        assert int(CodingRate.CR_4_5) == 1
        assert int(CodingRate.CR_4_8) == 4


# ---------------------------------------------------------------------------
# LoRaParams validation
# ---------------------------------------------------------------------------
class TestLoRaParamsValidation:
    def test_valid_construction(self) -> None:
        p = LoRaParams(
            spreading_factor=10,
            bandwidth_hz=500_000,
            coding_rate=CodingRate.CR_4_5,
        )
        assert p.spreading_factor == 10
        assert p.crc_on is True
        assert p.explicit_header is True

    @pytest.mark.parametrize("sf", [5, 13, 0, -1, 100])
    def test_invalid_sf_raises(self, sf: int) -> None:
        with pytest.raises(ValueError, match="spreading_factor"):
            LoRaParams(sf, 500_000, CodingRate.CR_4_5)

    @pytest.mark.parametrize("bw", [0, -1000])
    def test_invalid_bw_raises(self, bw: int) -> None:
        with pytest.raises(ValueError, match="bandwidth_hz"):
            LoRaParams(10, bw, CodingRate.CR_4_5)

    def test_invalid_preamble_raises(self) -> None:
        with pytest.raises(ValueError, match="preamble_symbols"):
            LoRaParams(10, 500_000, CodingRate.CR_4_5, preamble_symbols=4)


# ---------------------------------------------------------------------------
# Symbol duration
# ---------------------------------------------------------------------------
class TestSymbolDuration:
    def test_paper_default_sf10_bw500(self) -> None:
        # T_sym = 1024/500000 = 2.048 ms
        assert PAPER_DEFAULT.symbol_duration_s() == pytest.approx(2.048e-3)

    def test_sf7_bw125(self) -> None:
        # T_sym = 128/125000 = 1.024 ms
        p = LoRaParams(7, 125_000, CodingRate.CR_4_5)
        assert p.symbol_duration_s() == pytest.approx(1.024e-3)

    def test_sf12_bw125(self) -> None:
        # T_sym = 4096/125000 = 32.768 ms
        p = LoRaParams(12, 125_000, CodingRate.CR_4_5)
        assert p.symbol_duration_s() == pytest.approx(32.768e-3)


# ---------------------------------------------------------------------------
# LDRO auto-enable rule
# ---------------------------------------------------------------------------
class TestLDROAutoRule:
    """Semtech AN1200.13 §2.1.6: LDRO required when T_sym > 16 ms."""

    @pytest.mark.parametrize("sf", [7, 8, 9, 10])
    def test_bw125_low_sf_ldro_off(self, sf: int) -> None:
        assert not LoRaParams(sf, 125_000, CodingRate.CR_4_5).ldro_effective()

    @pytest.mark.parametrize("sf", [11, 12])
    def test_bw125_high_sf_ldro_on(self, sf: int) -> None:
        assert LoRaParams(sf, 125_000, CodingRate.CR_4_5).ldro_effective()

    def test_bw250_sf12_ldro_on(self) -> None:
        # T_sym = 4096/250000 = 16.384 ms > 16 ms
        assert LoRaParams(12, 250_000, CodingRate.CR_4_5).ldro_effective()

    def test_bw250_sf11_ldro_off(self) -> None:
        # T_sym = 2048/250000 = 8.192 ms
        assert not LoRaParams(11, 250_000, CodingRate.CR_4_5).ldro_effective()

    @pytest.mark.parametrize("sf", [7, 8, 9, 10, 11, 12])
    def test_bw500_ldro_never(self, sf: int) -> None:
        # Even SF=12/BW=500kHz gives T_sym = 8.192 ms, still < threshold.
        assert not LoRaParams(sf, 500_000, CodingRate.CR_4_5).ldro_effective()

    def test_manual_override_true(self) -> None:
        p = LoRaParams(7, 500_000, CodingRate.CR_4_5, low_data_rate_optimize=True)
        assert p.ldro_effective()

    def test_manual_override_false(self) -> None:
        p = LoRaParams(12, 125_000, CodingRate.CR_4_5, low_data_rate_optimize=False)
        assert not p.ldro_effective()


# ---------------------------------------------------------------------------
# Reference time-on-air values (hand-computed)
# ---------------------------------------------------------------------------
class TestTimeOnAirReferenceValues:
    """Four independent hand-computed ToA values. Each derivation is
    shown in the test comment so future-me can re-check it."""

    def test_paper_default_payload_32(self) -> None:
        # Manuel et al. baseline. SF=10, BW=500kHz, CR=4/5, PL=32.
        # T_sym       = 1024/500000 = 2.048 ms
        # T_preamble  = 12.25 * 2.048 = 25.088 ms
        # numerator   = 8*32 - 4*10 + 28 + 16*1 - 20*0 = 260
        # denominator = 4*(10-0) = 40
        # body        = max(ceil(260/40), 0) * (1+4) = 7*5 = 35
        # N_pay       = 8 + 35 = 43
        # T_payload   = 43 * 2.048 = 88.064 ms
        # T_packet    = 113.152 ms
        assert time_on_air_ms(PAPER_DEFAULT, 32) == pytest.approx(
            113.152, abs=TOL_MS
        )

    def test_datasheet_sf7_bw125_payload_13(self) -> None:
        # Common LoRaWAN uplink case. SF=7, BW=125kHz, CR=4/5, PL=13.
        # T_sym       = 128/125000 = 1.024 ms
        # T_preamble  = 12.25 * 1.024 = 12.544 ms
        # numerator   = 8*13 - 4*7 + 28 + 16 - 0 = 120
        # denominator = 4*(7-0) = 28
        # body        = max(ceil(120/28), 0) * 5 = 5*5 = 25
        # N_pay       = 8 + 25 = 33
        # T_payload   = 33 * 1.024 = 33.792 ms
        # T_packet    = 46.336 ms
        p = LoRaParams(7, 125_000, CodingRate.CR_4_5)
        assert time_on_air_ms(p, 13) == pytest.approx(46.336, abs=TOL_MS)

    def test_long_range_sf12_bw125_cr48_payload_51(self) -> None:
        # Maximum-range preset. SF=12, BW=125kHz, CR=4/8, PL=51, LDRO auto=ON.
        # T_sym       = 4096/125000 = 32.768 ms
        # T_preamble  = 12.25 * 32.768 = 401.408 ms
        # numerator   = 8*51 - 4*12 + 28 + 16 - 0 = 404
        # denominator = 4*(12-2) = 40   (DE=1 from LDRO)
        # body        = max(ceil(404/40), 0) * (4+4) = 11*8 = 88
        # N_pay       = 8 + 88 = 96
        # T_payload   = 96 * 32.768 = 3145.728 ms
        # T_packet    = 3547.136 ms  (~3.55 seconds)
        p = LoRaParams(12, 125_000, CodingRate.CR_4_8)
        assert time_on_air_ms(p, 51) == pytest.approx(3547.136, abs=TOL_MS)

    def test_paper_sf_at_low_bw_payload_13(self) -> None:
        # SF=10, BW=125kHz, CR=4/5, PL=13 — paper SF at long-range BW.
        # LDRO auto=OFF because T_sym = 8.192 ms < 16 ms threshold.
        # T_sym       = 1024/125000 = 8.192 ms
        # T_preamble  = 12.25 * 8.192 = 100.352 ms
        # numerator   = 8*13 - 4*10 + 28 + 16 - 0 = 108
        # denominator = 4*(10-0) = 40
        # body        = max(ceil(108/40), 0) * 5 = 3*5 = 15
        # N_pay       = 8 + 15 = 23
        # T_payload   = 23 * 8.192 = 188.416 ms
        # T_packet    = 288.768 ms
        p = LoRaParams(10, 125_000, CodingRate.CR_4_5)
        assert time_on_air_ms(p, 13) == pytest.approx(288.768, abs=TOL_MS)


# ---------------------------------------------------------------------------
# Formula-branch coverage
# ---------------------------------------------------------------------------
class TestPayloadFormulaBranches:
    def test_negative_ceil_clamped_to_zero(self) -> None:
        # SF=12, BW=125kHz, CR=4/5, PL=1, LDRO on, no CRC, implicit header:
        # numerator = 8 - 48 + 28 + 0 - 20 = -32
        # ceil(-32/40) = 0 (Python ceil rounds toward +inf), max(..,0) = 0
        # N_pay = 8
        p = LoRaParams(
            spreading_factor=12,
            bandwidth_hz=125_000,
            coding_rate=CodingRate.CR_4_5,
            explicit_header=False,
            crc_on=False,
        )
        assert payload_symbol_count(p, 1) == 8

    def test_implicit_header_never_longer_than_explicit(self) -> None:
        explicit = LoRaParams(10, 500_000, CodingRate.CR_4_5, explicit_header=True)
        implicit = LoRaParams(10, 500_000, CodingRate.CR_4_5, explicit_header=False)
        assert payload_symbol_count(implicit, 32) <= payload_symbol_count(explicit, 32)

    def test_crc_off_never_longer_than_crc_on(self) -> None:
        crc_on = LoRaParams(10, 500_000, CodingRate.CR_4_5, crc_on=True)
        crc_off = LoRaParams(10, 500_000, CodingRate.CR_4_5, crc_on=False)
        assert payload_symbol_count(crc_off, 32) <= payload_symbol_count(crc_on, 32)

    def test_larger_preamble_increases_toa(self) -> None:
        short = LoRaParams(10, 500_000, CodingRate.CR_4_5, preamble_symbols=8)
        long_ = LoRaParams(10, 500_000, CodingRate.CR_4_5, preamble_symbols=16)
        assert time_on_air_s(long_, 32) > time_on_air_s(short, 32)

    def test_preamble_duration_matches_hand_calc(self) -> None:
        # Direct check: preamble=8 at PAPER_DEFAULT -> 12.25 * 2.048 = 25.088 ms
        assert preamble_duration_s(PAPER_DEFAULT) * 1000.0 == pytest.approx(
            25.088, abs=TOL_MS
        )


# ---------------------------------------------------------------------------
# Payload argument validation
# ---------------------------------------------------------------------------
class TestPayloadValidation:
    @pytest.mark.parametrize("pl", [0, -1, 256, 1000])
    def test_invalid_payload_raises(self, pl: int) -> None:
        with pytest.raises(ValueError, match="payload_bytes"):
            time_on_air_s(PAPER_DEFAULT, pl)


# ---------------------------------------------------------------------------
# YAML preset loading
# ---------------------------------------------------------------------------
class TestYAMLPresetLoading:
    def test_load_paper_default_from_yaml(self, tmp_path: Path) -> None:
        yaml_content = """
presets:
  paper_default:
    spreading_factor: 10
    bandwidth_hz: 500000
    coding_rate: "4/5"
    preamble_symbols: 8
    explicit_header: true
    crc_on: true
"""
        p = tmp_path / "lora_params.yaml"
        p.write_text(yaml_content, encoding="utf-8")
        loaded = LoRaParams.from_preset(p, "paper_default")

        # Fields that affect the formula must match PAPER_DEFAULT.
        assert loaded.spreading_factor == PAPER_DEFAULT.spreading_factor
        assert loaded.bandwidth_hz == PAPER_DEFAULT.bandwidth_hz
        assert loaded.coding_rate == PAPER_DEFAULT.coding_rate
        assert loaded.preamble_symbols == PAPER_DEFAULT.preamble_symbols
        assert loaded.explicit_header == PAPER_DEFAULT.explicit_header
        assert loaded.crc_on == PAPER_DEFAULT.crc_on

        # And the resulting ToA is identical.
        assert time_on_air_ms(loaded, 32) == pytest.approx(
            time_on_air_ms(PAPER_DEFAULT, 32)
        )

    def test_missing_preset_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "lora_params.yaml"
        p.write_text("presets: {}\n", encoding="utf-8")
        with pytest.raises(KeyError, match="doesnt_exist"):
            LoRaParams.from_preset(p, "doesnt_exist")

    def test_load_from_real_configs_file(self) -> None:
        # Sanity: the repo's own configs/lora_params.yaml has a working
        # 'paper_default' preset that matches PAPER_DEFAULT.
        cfg = Path(__file__).resolve().parents[4] / "configs" / "lora_params.yaml"
        if not cfg.exists():
            pytest.skip(f"configs file not found at {cfg}")
        loaded = LoRaParams.from_preset(cfg, "paper_default")
        assert loaded.spreading_factor == 10
        assert loaded.bandwidth_hz == 500_000
        assert loaded.coding_rate == CodingRate.CR_4_5
