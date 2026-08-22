"""LoRa channel model: log-distance pathloss + log-normal shadowing.

Formula (Rappaport, "Wireless Communications", §4.9):

    PL(d) = PL(d0) + 10 * n * log10(d / d0) + X_sigma

where:
    PL(d0) = reference pathloss at reference distance d0 (dB)
    n      = pathloss exponent (2 = free space, higher = obstructed)
    X_sigma ~ N(0, sigma^2) — log-normal shadowing in dB

Received signal strength (dBm):

    RSSI(d) = TX_power - PL(d)

Sensitivity (SX1276/77/78/79 datasheet §5.5.5, Table 13):

    N_floor(dBm)   = -174 + 10*log10(BW_Hz) + NoiseFigure_dB
    Sensitivity    = N_floor + demod_snr(SF)

A packet is decodable when RSSI >= sensitivity.

Limitations of this step:
    - Shadowing draws are per-packet-independent. Real shadowing is
      spatially/temporally correlated with a ~10 m decorrelation distance;
      good enough for the routing bake-off, worth revisiting for the
      adaptive leader recovery if RSSI-trend noise dominates.
    - No frequency-selective fading (LoRa's chirp spreading is robust to
      it, and we're on flat-fading assumption per Petäjäjärvi 2015).
    - Collisions/interference are transport-layer concerns handled in
      step 5, not here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

__all__ = [
    "DEFAULT_NOISE_FIGURE_DB",
    "ChannelModel",
    "PathLossModel",
    "ShadowingModel",
    "demod_snr_db",
    "is_decodable",
    "noise_floor_dbm",
    "sensitivity_dbm",
    "snr_db",
]


# ---------------------------------------------------------------------------
# Path loss
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PathLossModel:
    """Deterministic log-distance pathloss.

    Distances below ``reference_distance_m`` are clamped to it, so
    pathloss never falls below ``reference_pathloss_db``. This keeps
    the near-field region physical without extra branching in callers.
    """

    path_loss_exponent: float
    reference_distance_m: float
    reference_pathloss_db: float

    def __post_init__(self) -> None:
        if self.path_loss_exponent <= 0:
            raise ValueError(
                f"path_loss_exponent must be positive, got {self.path_loss_exponent}"
            )
        if self.reference_distance_m <= 0:
            raise ValueError(
                f"reference_distance_m must be positive, "
                f"got {self.reference_distance_m}"
            )

    def pathloss_db(self, distance_m: float) -> float:
        """Deterministic pathloss at ``distance_m`` (no shadowing)."""
        if distance_m <= 0:
            raise ValueError(f"distance_m must be positive, got {distance_m}")
        d = max(distance_m, self.reference_distance_m)
        return self.reference_pathloss_db + 10.0 * self.path_loss_exponent * math.log10(
            d / self.reference_distance_m
        )


# ---------------------------------------------------------------------------
# Shadowing
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShadowingModel:
    """Log-normal shadowing: zero-mean Gaussian in dB with std ``std_dev_db``."""

    std_dev_db: float

    def __post_init__(self) -> None:
        if self.std_dev_db < 0:
            raise ValueError(
                f"std_dev_db must be non-negative, got {self.std_dev_db}"
            )

    def sample_db(self, rng: np.random.Generator) -> float:
        """One shadowing draw in dB. Zero if std_dev_db == 0."""
        if self.std_dev_db == 0.0:
            return 0.0
        return float(rng.normal(loc=0.0, scale=self.std_dev_db))


# ---------------------------------------------------------------------------
# Combined channel
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChannelModel:
    """Log-distance pathloss + log-normal shadowing, with an RSSI helper.

    ``rng=None`` on any call means "return the deterministic mean";
    seeded ``rng`` gives reproducible shadowing draws.
    """

    name: str
    pathloss: PathLossModel
    shadowing: ShadowingModel

    def pathloss_db(
        self,
        distance_m: float,
        rng: np.random.Generator | None = None,
    ) -> float:
        """PL(d) with optional shadowing sample."""
        pl = self.pathloss.pathloss_db(distance_m)
        if rng is None:
            return pl
        return pl + self.shadowing.sample_db(rng)

    def rssi_dbm(
        self,
        tx_power_dbm: float,
        distance_m: float,
        rng: np.random.Generator | None = None,
    ) -> float:
        """RSSI(d) = TX_power - PL(d)."""
        return tx_power_dbm - self.pathloss_db(distance_m, rng)

    @classmethod
    def from_preset(cls, config_path: str | Path, preset_name: str) -> ChannelModel:
        """Load a named model from a YAML config (e.g. configs/channel.yaml)."""
        text = Path(config_path).read_text(encoding="utf-8")
        doc = yaml.safe_load(text)
        models = (doc or {}).get("models") or {}
        if preset_name not in models:
            raise KeyError(
                f"model {preset_name!r} not in {sorted(models)} (from {config_path})"
            )
        m = models[preset_name]
        return cls(
            name=preset_name,
            pathloss=PathLossModel(
                path_loss_exponent=float(m["path_loss_exponent"]),
                reference_distance_m=float(m["reference_distance_m"]),
                reference_pathloss_db=float(m["reference_pathloss_db"]),
            ),
            shadowing=ShadowingModel(std_dev_db=float(m["shadowing_std_db"])),
        )


# ---------------------------------------------------------------------------
# Sensitivity
# ---------------------------------------------------------------------------

# SX1276/77/78/79 datasheet §5.5.5, Table 13. Demodulation SNR floor
# for each spreading factor. Below these, the LoRa demod cannot recover
# the chirp. Values in dB.
_DEMOD_SNR_BY_SF: dict[int, float] = {
    6: -5.0,
    7: -7.5,
    8: -10.0,
    9: -12.5,
    10: -15.0,
    11: -17.5,
    12: -20.0,
}

# SX1276 datasheet lists sensitivity numbers implying a noise figure
# in the 6..7 dB range. We use 6 as the nominal value; back out
# 1..2 dB of margin when comparing against real hardware.
DEFAULT_NOISE_FIGURE_DB: float = 6.0


def demod_snr_db(spreading_factor: int) -> float:
    """Minimum SNR (dB) at which the LoRa demod can decode, per SF."""
    if spreading_factor not in _DEMOD_SNR_BY_SF:
        raise ValueError(
            f"spreading_factor must be in {sorted(_DEMOD_SNR_BY_SF)}, "
            f"got {spreading_factor}"
        )
    return _DEMOD_SNR_BY_SF[spreading_factor]


def noise_floor_dbm(
    bandwidth_hz: float,
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
) -> float:
    """Receiver thermal noise floor: -174 + 10 log10(BW) + NF."""
    if bandwidth_hz <= 0:
        raise ValueError(f"bandwidth_hz must be positive, got {bandwidth_hz}")
    return -174.0 + 10.0 * math.log10(bandwidth_hz) + noise_figure_db


def sensitivity_dbm(
    spreading_factor: int,
    bandwidth_hz: float,
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
) -> float:
    """Receiver sensitivity = noise floor + demod SNR floor."""
    return noise_floor_dbm(bandwidth_hz, noise_figure_db) + demod_snr_db(
        spreading_factor
    )


def snr_db(
    rssi_dbm: float,
    bandwidth_hz: float,
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
) -> float:
    """SNR of a received signal, given RSSI and receiver BW."""
    return rssi_dbm - noise_floor_dbm(bandwidth_hz, noise_figure_db)


def is_decodable(
    rssi_dbm: float,
    spreading_factor: int,
    bandwidth_hz: float,
    noise_figure_db: float = DEFAULT_NOISE_FIGURE_DB,
) -> bool:
    """True iff RSSI >= sensitivity(SF, BW, NF)."""
    return rssi_dbm >= sensitivity_dbm(spreading_factor, bandwidth_hz, noise_figure_db)
