"""Semtech LoRa time-on-air formula (SX1276/77/78/79).

Reference: SX1276/77/78/79 datasheet §4.1.1.6-7 and Semtech AN1200.13,
"LoRa Modem Designer's Guide". The formula is:

    T_sym    = 2^SF / BW
    T_pream  = (n_preamble + 4.25) * T_sym
    N_pay    = 8 + max(ceil((8*PL - 4*SF + 28 + 16*CRC - 20*H) /
                            (4 * (SF - 2*DE))), 0) * (CR + 4)
    T_pay    = N_pay * T_sym
    T_packet = T_pream + T_pay

where:
    SF  = spreading factor, 6..12
    BW  = bandwidth in Hz
    PL  = payload length in bytes (excludes preamble, header, CRC)
    CRC = 1 if payload CRC enabled, else 0
    H   = 0 if explicit header (default), 1 if implicit
    DE  = 1 if Low Data Rate Optimize enabled, else 0
    CR  = Semtech CR parameter 1..4; actual rate = 4/(4+CR)
          (CR=1 -> 4/5, CR=2 -> 4/6, CR=3 -> 4/7, CR=4 -> 4/8)

LDRO auto-enable rule (AN1200.13 §2.1.6): T_sym > 16 ms. That gives:
    BW=125 kHz -> SF in {11, 12}
    BW=250 kHz -> SF = 12
    BW=500 kHz -> never

The paper (Manuel et al., §III-B, Algorithm 2) uses SF=10, BW=500 kHz,
CR=4/5, TX=15 dBm. TX power belongs on the radio, not the airtime
calc — LoRaParams here holds only the parameters that affect ToA.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from math import ceil
from pathlib import Path

import yaml

__all__ = [
    "PAPER_DEFAULT",
    "CodingRate",
    "LoRaParams",
    "payload_duration_s",
    "payload_symbol_count",
    "preamble_duration_s",
    "time_on_air_ms",
    "time_on_air_s",
]


class CodingRate(IntEnum):
    """LoRa coding rate. Value = Semtech CR parameter; actual rate = 4/(4+value)."""

    CR_4_5 = 1
    CR_4_6 = 2
    CR_4_7 = 3
    CR_4_8 = 4

    @classmethod
    def from_str(cls, s: str) -> CodingRate:
        """Parse '4/5', '4/6', '4/7', '4/8'."""
        mapping = {
            "4/5": cls.CR_4_5,
            "4/6": cls.CR_4_6,
            "4/7": cls.CR_4_7,
            "4/8": cls.CR_4_8,
        }
        if s not in mapping:
            raise ValueError(
                f"Unknown coding rate {s!r}. Expected one of {sorted(mapping)}."
            )
        return mapping[s]


@dataclass(frozen=True)
class LoRaParams:
    """LoRa radio parameters that affect time-on-air.

    Payload length is NOT here — it varies per packet, so it's a runtime
    argument to ``time_on_air_s`` etc.

    ``low_data_rate_optimize=None`` means auto-enable per Semtech rule
    (T_sym > 16 ms). Set explicitly to True or False to override.
    """

    spreading_factor: int
    bandwidth_hz: int
    coding_rate: CodingRate
    preamble_symbols: int = 8
    explicit_header: bool = True
    crc_on: bool = True
    low_data_rate_optimize: bool | None = None

    def __post_init__(self) -> None:
        if not 6 <= self.spreading_factor <= 12:
            raise ValueError(
                f"spreading_factor must be 6..12, got {self.spreading_factor}"
            )
        if self.bandwidth_hz <= 0:
            raise ValueError(f"bandwidth_hz must be positive, got {self.bandwidth_hz}")
        # Semtech minimum preamble length is 6 symbols; the modem adds 4.25.
        if self.preamble_symbols < 6:
            raise ValueError(
                f"preamble_symbols must be >= 6 (Semtech minimum), "
                f"got {self.preamble_symbols}"
            )

    def symbol_duration_s(self) -> float:
        """T_sym = 2^SF / BW, in seconds."""
        return (1 << self.spreading_factor) / self.bandwidth_hz

    def ldro_effective(self) -> bool:
        """Effective LDRO after auto-rule and override are applied.

        Auto-rule (AN1200.13 §2.1.6): LDRO on when T_sym > 16 ms.
        """
        if self.low_data_rate_optimize is not None:
            return self.low_data_rate_optimize
        return self.symbol_duration_s() > 0.016

    @classmethod
    def from_preset(cls, config_path: str | Path, preset_name: str) -> LoRaParams:
        """Load a named preset from a YAML config (e.g. configs/lora_params.yaml)."""
        text = Path(config_path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        presets = (doc or {}).get("presets") or {}
        if preset_name not in presets:
            raise KeyError(
                f"preset {preset_name!r} not in {sorted(presets)} "
                f"(from {config_path})"
            )
        p = presets[preset_name]
        return cls(
            spreading_factor=int(p["spreading_factor"]),
            bandwidth_hz=int(p["bandwidth_hz"]),
            coding_rate=CodingRate.from_str(str(p["coding_rate"])),
            preamble_symbols=int(p.get("preamble_symbols", 8)),
            explicit_header=bool(p.get("explicit_header", True)),
            crc_on=bool(p.get("crc_on", True)),
            low_data_rate_optimize=p.get("low_data_rate_optimize"),
        )


# The paper's canonical setting: Manuel et al. §III-B, Algorithm 2.
# SF=10, BW=500 kHz, CR=4/5. T_sym = 2.048 ms, so LDRO auto-resolves
# to False (well under the 16 ms threshold).
PAPER_DEFAULT: LoRaParams = LoRaParams(
    spreading_factor=10,
    bandwidth_hz=500_000,
    coding_rate=CodingRate.CR_4_5,
    preamble_symbols=8,
    explicit_header=True,
    crc_on=True,
)


def preamble_duration_s(params: LoRaParams) -> float:
    """T_preamble = (n_preamble + 4.25) * T_sym."""
    return (params.preamble_symbols + 4.25) * params.symbol_duration_s()


def payload_symbol_count(params: LoRaParams, payload_bytes: int) -> int:
    """Number of symbols in the payload portion of a LoRa packet.

    This is the ``N_pay`` term from the Semtech formula. Always >= 8
    because of the additive fixed overhead.
    """
    if not 1 <= payload_bytes <= 255:
        raise ValueError(f"payload_bytes must be 1..255, got {payload_bytes}")

    sf = params.spreading_factor
    pl = payload_bytes
    crc = 1 if params.crc_on else 0
    h = 0 if params.explicit_header else 1
    de = 1 if params.ldro_effective() else 0
    cr = int(params.coding_rate)  # 1..4

    numerator = 8 * pl - 4 * sf + 28 + 16 * crc - 20 * h
    # (sf - 2*de) is always >= 4 for valid SF, so denominator > 0.
    denominator = 4 * (sf - 2 * de)

    # ceil(...) can go negative for very small payloads with implicit
    # header + no CRC at high SF; max(...,0) clamps that per the datasheet.
    body_symbols = max(ceil(numerator / denominator), 0) * (cr + 4)
    return 8 + body_symbols


def payload_duration_s(params: LoRaParams, payload_bytes: int) -> float:
    """T_pay = N_pay * T_sym, in seconds."""
    return payload_symbol_count(params, payload_bytes) * params.symbol_duration_s()


def time_on_air_s(params: LoRaParams, payload_bytes: int) -> float:
    """Total on-air time of a LoRa packet, in seconds.

    T_packet = T_preamble + T_payload.
    """
    return preamble_duration_s(params) + payload_duration_s(params, payload_bytes)


def time_on_air_ms(params: LoRaParams, payload_bytes: int) -> float:
    """Same as ``time_on_air_s`` but in milliseconds — convenient for duty math."""
    return time_on_air_s(params, payload_bytes) * 1000.0
