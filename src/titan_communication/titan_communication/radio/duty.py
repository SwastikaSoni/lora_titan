"""LoRa duty-cycle and dwell-time accounting.

Regional radio regulations impose two kinds of constraint on how much
time a node may spend transmitting:

    1. Percentage duty cycle over a rolling window — e.g. EU868 and
       IN865 cap airtime at 1% of any rolling 1-hour window per sub-band
       (ETSI EN 300 220-2, LoRa Alliance Regional Parameters).

    2. Per-transmission dwell time — e.g. US915 with frequency hopping
       under FCC §15.247 caps each transmission at 400 ms per channel.
       Non-hopping DTS mode has no dwell cap and no percentage cap, but
       for the bake-off we impose a "polite" 10% ceiling to keep the
       channel usable — see ``RegionPolicy.us915_polite``.

``DutyBucket`` tracks per-node airtime and answers three questions the
mesh layer needs to make deferral decisions:

    - can_transmit(now, airtime)  -> would this violate policy?
    - utilization(now)            -> fraction of cap consumed
    - time_until_next_available   -> how long to wait

The bucket keeps state in seconds; feed it whatever monotonic clock or
``simpy.Environment.now`` you use. It doesn't care.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

__all__ = [
    "DutyBucket",
    "DutyViolationError",
    "RegionPolicy",
]


class DutyViolationError(RuntimeError):
    """Raised by ``DutyBucket.register`` when a transmission would violate policy."""


# ---------------------------------------------------------------------------
# RegionPolicy
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RegionPolicy:
    """Regional airtime rules for a LoRa deployment.

    ``duty_cycle_fraction=None`` means no percentage cap.
    ``max_dwell_time_s=None`` means no per-transmission cap.
    Both can be None (unrestricted); both can be set (both enforced).
    """

    name: str
    duty_cycle_fraction: float | None
    duty_window_s: float
    max_dwell_time_s: float | None

    def __post_init__(self) -> None:
        if self.duty_cycle_fraction is not None and not (
            0.0 < self.duty_cycle_fraction <= 1.0
        ):
            raise ValueError(
                f"duty_cycle_fraction must be in (0, 1], "
                f"got {self.duty_cycle_fraction}"
            )
        if self.duty_window_s <= 0:
            raise ValueError(
                f"duty_window_s must be positive, got {self.duty_window_s}"
            )
        if self.max_dwell_time_s is not None and self.max_dwell_time_s <= 0:
            raise ValueError(
                f"max_dwell_time_s must be positive or None, "
                f"got {self.max_dwell_time_s}"
            )

    def max_airtime_in_window_s(self) -> float:
        """Absolute airtime budget for a full window (``inf`` if uncapped)."""
        if self.duty_cycle_fraction is None:
            return float("inf")
        return self.duty_window_s * self.duty_cycle_fraction

    # -- Convenience constructors ------------------------------------------

    @classmethod
    def us915_polite(cls) -> RegionPolicy:
        """US915 with a self-imposed 10% duty ceiling.

        FCC §15.247 DTS mode imposes no percentage cap and no dwell
        limit on non-hopping LoRa. But letting a single node monopolize
        the channel is bad-neighbour behaviour, and it makes the mesh
        bake-off's "duty-cycle-aware deferral" metric meaningless
        (nothing to defer against). We adopt a 10% ceiling by convention
        — an order of magnitude looser than EU rules, still leaves 90%
        of airtime for other nodes. Override via ``us915_fcc_raw`` if
        you want the pure FCC minimum.
        """
        return cls(
            name="US915-polite",
            duty_cycle_fraction=0.10,
            duty_window_s=3600.0,
            max_dwell_time_s=None,
        )

    @classmethod
    def us915_fcc_raw(cls) -> RegionPolicy:
        """US915 non-hopping DTS mode: no caps. FCC §15.247."""
        return cls(
            name="US915-raw",
            duty_cycle_fraction=None,
            duty_window_s=3600.0,  # unused when duty_cycle_fraction is None
            max_dwell_time_s=None,
        )

    @classmethod
    def us915_hopping(cls) -> RegionPolicy:
        """US915 with FHSS: 400 ms dwell per channel (FCC §15.247)."""
        return cls(
            name="US915-hopping",
            duty_cycle_fraction=None,
            duty_window_s=3600.0,
            max_dwell_time_s=0.400,
        )

    @classmethod
    def in865(cls) -> RegionPolicy:
        """IN865 (India): 1% duty cycle per rolling 1-hour window.

        Reference: LoRa Alliance Regional Parameters, IN865-867 section.
        Relevant if we ever port the mesh to an Indian deployment; the
        paper's setup runs US915.
        """
        return cls(
            name="IN865",
            duty_cycle_fraction=0.01,
            duty_window_s=3600.0,
            max_dwell_time_s=None,
        )

    @classmethod
    def eu868(cls) -> RegionPolicy:
        """EU868 sub-band g1: 1% duty per rolling 1-hour window (ETSI EN 300 220-2)."""
        return cls(
            name="EU868",
            duty_cycle_fraction=0.01,
            duty_window_s=3600.0,
            max_dwell_time_s=None,
        )


# ---------------------------------------------------------------------------
# DutyBucket
# ---------------------------------------------------------------------------
@dataclass
class DutyBucket:
    """Per-node airtime accounting with rolling-window eviction.

    State: a deque of ``(end_time_s, airtime_s)`` in chronological order.
    On any query, entries whose ``end_time`` is at or before ``now -
    window`` are dropped (they exited the rolling window). Amortised
    O(1) per operation.

    The choice of ``end_time`` (vs start_time) for the bookkeeping means
    an entry is "in the window" from the instant it ends until exactly
    ``window`` seconds later. The rounding error vs a duration-overlap
    model is at most one packet's airtime — negligible when packets are
    tens to hundreds of ms and windows are ~1 hour.
    """

    policy: RegionPolicy
    _log: deque[tuple[float, float]] = field(
        default_factory=deque, init=False, repr=False
    )

    # -- Internals ---------------------------------------------------------

    def _evict(self, now_s: float) -> None:
        """Drop entries whose ``end_time`` is at or before ``now - window``."""
        cutoff = now_s - self.policy.duty_window_s
        while self._log and self._log[0][0] <= cutoff:
            self._log.popleft()

    def _append(self, now_s: float, airtime_s: float) -> None:
        """Append an entry, enforcing monotonic timestamps."""
        if self._log and now_s < self._log[-1][0]:
            raise ValueError(
                f"timestamps must be non-decreasing: got now_s={now_s}, "
                f"last was {self._log[-1][0]}"
            )
        self._log.append((now_s, airtime_s))

    # -- Queries -----------------------------------------------------------

    def airtime_used_s(self, now_s: float) -> float:
        """Total airtime still within the rolling window at ``now_s``."""
        self._evict(now_s)
        return sum(a for _, a in self._log)

    def utilization(self, now_s: float) -> float:
        """Fraction of the airtime budget consumed. Returns 0.0 when uncapped."""
        if self.policy.duty_cycle_fraction is None:
            return 0.0
        return self.airtime_used_s(now_s) / self.policy.max_airtime_in_window_s()

    def can_transmit(self, now_s: float, airtime_s: float) -> bool:
        """True if transmitting ``airtime_s`` at ``now_s`` would not violate policy."""
        if airtime_s <= 0:
            raise ValueError(f"airtime_s must be positive, got {airtime_s}")
        # Dwell cap: this single transmission over the per-tx limit?
        if (
            self.policy.max_dwell_time_s is not None
            and airtime_s > self.policy.max_dwell_time_s
        ):
            return False
        # Percentage cap: would we exceed the rolling budget?
        if self.policy.duty_cycle_fraction is not None:
            used = self.airtime_used_s(now_s)
            budget = self.policy.max_airtime_in_window_s()
            if used + airtime_s > budget:
                return False
        return True

    def time_until_next_available(
        self, now_s: float, airtime_s: float
    ) -> float | None:
        """Wait duration from ``now_s`` until ``can_transmit`` becomes True.

        Returns:
            0.0 if transmission is allowed right now.
            A positive float = seconds to wait.
            None if never (dwell exceeded, or airtime > full-window budget).
        """
        if airtime_s <= 0:
            raise ValueError(f"airtime_s must be positive, got {airtime_s}")
        # Dwell violation: permanent — no wait helps.
        if (
            self.policy.max_dwell_time_s is not None
            and airtime_s > self.policy.max_dwell_time_s
        ):
            return None
        # Uncapped percentage: allowed immediately.
        if self.policy.duty_cycle_fraction is None:
            return 0.0
        budget = self.policy.max_airtime_in_window_s()
        # Single transmission bigger than a whole window's budget: never.
        if airtime_s > budget:
            return None
        self._evict(now_s)
        used = sum(a for _, a in self._log)
        # Room right now.
        if used + airtime_s <= budget:
            return 0.0
        # Wait for enough old airtime to fall out. Iterate from oldest;
        # each entry (end_time, tx) exits at end_time + window.
        need_to_shed = used + airtime_s - budget
        shed = 0.0
        for end_time, tx_airtime in self._log:
            shed += tx_airtime
            if shed >= need_to_shed:
                return max(end_time + self.policy.duty_window_s - now_s, 0.0)
        # Unreachable: airtime_s <= budget guarantees the loop reaches
        # a sufficient shed value.
        return None  # pragma: no cover

    # -- Mutations ---------------------------------------------------------

    def register(self, now_s: float, airtime_s: float) -> None:
        """Record a completed transmission. Raises ``DutyViolation`` if blocked."""
        if not self.can_transmit(now_s, airtime_s):
            raise DutyViolationError(
                f"Cannot register {airtime_s * 1000:.2f} ms transmission at "
                f"t={now_s:.3f}s under {self.policy.name}: "
                f"utilization would exceed cap, or dwell exceeded."
            )
        self._append(now_s, airtime_s)

    def try_register(self, now_s: float, airtime_s: float) -> bool:
        """Register if allowed; return False without changing state if blocked."""
        if not self.can_transmit(now_s, airtime_s):
            return False
        self._append(now_s, airtime_s)
        return True

    def reset(self) -> None:
        """Clear all tracked history. Useful for test setup."""
        self._log.clear()
