"""fuzzy_rules.py — Fuzzy Inference System + Vector Field Histogram (FISVFH).

Pure Python + numpy + scikit-fuzzy. No ROS dependency, so the controller
can be unit-tested and tuned with synthetic scans before it is ever run
in Gazebo. ``fisvfh_node.py`` (Week 6, task 2) wires this up to
``/scan`` and ``/odometry/filtered`` and publishes the result on
``/cmd_vel``.

Pipeline, matching the three FISVFH stages:

1. ``VFHHistogram.build`` — bin a LiDAR scan into a polar histogram of
   per-sector minimum clearance ("environment mapping" + "histogram
   generation").
2. ``VFHHistogram.select_direction`` — threshold the histogram into
   free "valleys" and pick the one closest to the goal heading
   ("sector selection").
3. ``FuzzyController.compute`` — take the selected sector's clearance
   and heading error and fuzzy-infer (linear velocity, angular
   velocity) ("fuzzy velocity control").

``FISVFHController`` chains all three steps into the single call a ROS
node needs per control cycle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

__all__ = ["FISVFHConfig", "FISVFHController", "FuzzyController", "VFHHistogram"]


def _wrap_to_pi(angle: float) -> float:
    """Wrap a scalar angle into ``[-pi, pi)``."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _wrap_to_pi_array(angles: np.ndarray) -> np.ndarray:
    """Wrap an array of angles into ``[-pi, pi)``."""
    return (angles + math.pi) % (2 * math.pi) - math.pi


@dataclass
class FISVFHConfig:
    """Tunable parameters shared by the histogram and fuzzy stages.

    Defaults target the Week 6 milestone: reach a goal GPS fix while
    keeping >= 0.5 m clearance from obstacles in
    ``phase1_region2_obstacles.sdf``.
    """

    num_sectors: int = 72
    """Angular resolution of the polar histogram (72 -> 5 deg/sector)."""

    max_range: float = 10.0
    """LiDAR range (m) beyond which obstacles are ignored / sectors with
    no return are treated as clear."""

    safe_distance: float = 0.6
    """Minimum clearance (m) for a sector to count as "free". The Week 6
    requirement is >= 0.5 m; this defaults a bit above that floor for
    margin, combined with comfort_factor preferring real open space
    beyond even that."""

    min_valley_width: int = 3
    """Minimum contiguous free sectors for a valley to be considered
    wide enough to steer straight through its center toward the goal;
    narrower valleys are aimed at their midpoint instead."""

    max_linear_vel: float = 0.6
    """Fuzzy output ceiling for linear velocity (m/s)."""

    max_angular_vel: float = 1.5
    """Fuzzy output ceiling for angular velocity (rad/s)."""

    goal_tolerance: float = 0.8
    """Distance tolerance (m) to consider goal reached."""

    direction_change_penalty: float = 0.6
    """Cost weight (per radian) for switching away from the previously
    selected direction. Without this, two valleys of similar cost (e.g.
    passing an obstacle left vs. right) can flip-flop every cycle as
    the goal heading jitters by a fraction of a degree — the robot
    commits left, then right, then left, and never completes either
    pass. Higher values commit harder to the current side; 0 disables
    hysteresis entirely."""

    comfort_factor: float = 1.5
    """Steering targets need clearance >= safe_distance * comfort_factor,
    not merely >= safe_distance, to be preferred. Without this, clipping
    the target to the goal-nearest edge of a free valley aims the robot
    to graze an obstacle at the bare-minimum legal distance even when
    genuinely open space is a few sectors further into the same valley —
    which also reads as "close" to the fuzzy controller and throttles
    speed to a crawl. 1.0 disables the margin (bare minimum only)."""

    angular_deadzone: float = 0.05
    """Heading errors smaller than this (rad) are treated as "already
    aimed at the goal" — no breakaway floor is applied, so small sector-
    quantization noise doesn't make the robot twitch on an open path."""

    min_effective_angular_vel: float = 1.0
    """Friction/stiction compensation floor (rad/s). Below some command
    magnitude, real robots (this one included — measured directly via
    commanded vs. IMU-reported angular velocity) don't actually rotate;
    the torque gets absorbed by static friction and the wheels just
    scrub in place. Below this measured threshold, a "correct" but weak
    fuzzy output (e.g. 0.3 rad/s for a moderate heading error) can drive
    the robot to creep toward an obstacle it "decided" to avoid, without
    ever actually turning — indistinguishable from not avoiding at all.
    Whenever the heading error exceeds ``angular_deadzone`` (a real turn
    is genuinely needed), the commanded angular velocity is floored to
    at least this magnitude. Set to 0 to disable (pure proportional
    output, appropriate for a robot without this deadband)."""

    angular_slew_rate: float = 2.0
    """Max change in commanded angular_vel per second (rad/s^2). Matches
    this robot's DiffDrive plugin ``max_angular_acceleration``, so the
    controller never requests a faster change than the robot could
    physically achieve. Without this, min_effective_angular_vel snaps
    the command from ~0 to full floor the instant a real turn starts,
    every control cycle — the robot's actual response lags behind
    (ramps up over ~1-2s), so it overshoots the target heading, the
    next cycle commands a hard correction back, and the robot ends up
    tracing a small stationary circle beside the obstacle instead of
    actually clearing it. Ramping the command in over a fraction of a
    second keeps the decision loop and the physical response in sync.
    Set to 0 (or very large) to disable slew limiting."""

    control_period: float = 0.1
    """Control loop period (s) — must match the node's control_frequency
    (1 / control_frequency) for angular_slew_rate to correspond to a
    real rad/s^2. Only used for the slew-rate limit."""


class VFHHistogram:
    """Builds the polar obstacle histogram and selects a steering sector.

    Angles are robot-frame radians in ``[-pi, pi)``, with 0 = straight
    ahead. Sectors outside the LiDAR's field of view (or with no valid
    return) default to ``max_range`` — i.e. assumed clear. For a
    forward-facing LiDAR this means the histogram wraps around at the
    rear of the robot rather than stitching valleys across that seam;
    that is an acceptable simplification since the goal direction is
    normally ahead of the robot, not behind it.
    """

    def __init__(self, config: FISVFHConfig) -> None:
        self.config = config
        self.sector_width = 2 * math.pi / config.num_sectors
        self._last_direction: float | None = None

    def reset(self) -> None:
        """Forget the committed direction (e.g. after the goal changes)."""
        self._last_direction = None

    def build(self, ranges: np.ndarray, angle_min: float, angle_increment: float) -> np.ndarray:
        """Bin raw LiDAR ranges into per-sector minimum clearance.

        Args:
            ranges: 1-D array of LiDAR range readings (as from
                ``sensor_msgs/LaserScan.ranges``).
            angle_min: angle (rad) of ``ranges[0]``, robot frame.
            angle_increment: angular step (rad) between samples.

        Returns:
            Array of length ``num_sectors``: the minimum valid range
            observed in each sector, or ``max_range`` if the sector had
            no valid returns.
        """
        ranges = np.asarray(ranges, dtype=float)
        valid = np.isfinite(ranges) & (ranges > 0.0)
        clamped = np.where(valid, ranges, self.config.max_range)
        clamped = np.clip(clamped, 0.0, self.config.max_range)

        angles = _wrap_to_pi_array(angle_min + angle_increment * np.arange(len(ranges)))
        sector_idx = np.floor((angles + math.pi) / self.sector_width).astype(int)
        sector_idx = np.clip(sector_idx, 0, self.config.num_sectors - 1)

        histogram = np.full(self.config.num_sectors, self.config.max_range)
        np.minimum.at(histogram, sector_idx, clamped)
        return histogram

    def select_direction(self, histogram: np.ndarray, heading_to_goal: float) -> tuple[float, float]:
        """Pick the free direction closest to the goal heading.

        Args:
            histogram: per-sector clearance, as returned by ``build``.
            heading_to_goal: desired heading (rad, robot frame).

        Returns:
            ``(selected_heading, clearance)``: the chosen steering
            direction (robot frame) and the minimum clearance within
            the sectors that produced it.
        """
        free = histogram >= self.config.safe_distance
        if not free.any():
            # Nothing meets the safety threshold — limp toward whatever
            # direction has the most room rather than freezing.
            idx = int(np.argmax(histogram))
            angle = self._index_to_angle(idx)
            self._last_direction = angle
            return angle, float(histogram[idx])

        goal_idx = self._angle_to_index(heading_to_goal)
        comfortable = self.config.safe_distance * self.config.comfort_factor
        best_angle = 0.0
        best_clearance = 0.0
        best_cost = math.inf

        for start, end in self._find_valleys(free):
            width = end - start + 1
            if width >= self.config.min_valley_width:
                target_idx = self._pick_target_in_valley(
                    histogram, start, end, goal_idx, comfortable
                )
            else:
                target_idx = (start + end) // 2

            angle = self._index_to_angle(target_idx)
            # Clearance local to the actual heading we'd steer onto, not
            # the whole valley — a valley can span 100+ degrees, and its
            # global minimum may sit nowhere near where we're pointing.
            local_clearance = self._windowed_clearance(histogram, target_idx, start, end)

            cost = abs(_wrap_to_pi(angle - heading_to_goal))
            if self._last_direction is not None:
                # Penalize switching away from the direction we're already
                # committed to, so two similarly-scored valleys (e.g. pass
                # left vs. right of an obstacle) don't flip-flop every
                # cycle as the goal heading jitters — see
                # FISVFHConfig.direction_change_penalty.
                cost += self.config.direction_change_penalty * abs(
                    _wrap_to_pi(angle - self._last_direction)
                )
            if cost < best_cost:
                best_cost = cost
                best_angle = angle
                best_clearance = local_clearance

        self._last_direction = best_angle
        return best_angle, best_clearance

    @staticmethod
    def _windowed_clearance(histogram: np.ndarray, idx: int, start: int, end: int) -> float:
        """Min clearance over ``idx`` and its immediate neighbors (clamped
        to the valley), so a lone good sample flanked by a worse one
        doesn't get reported as more open than it really is."""
        lo = max(start, idx - 1)
        hi = min(end, idx + 1)
        return float(histogram[lo:hi + 1].min())

    @classmethod
    def _pick_target_in_valley(
        cls, histogram: np.ndarray, start: int, end: int, goal_idx: int, comfortable: float
    ) -> int:
        """Steer as close to the goal as possible without hugging an edge.

        Clips toward ``goal_idx`` first (get as close to the goal heading
        as this valley allows), then — if that point's neighborhood is
        only barely above ``safe_distance`` — searches outward within the
        valley for the nearest sector whose neighborhood clears a
        comfortable margin. Falls back to the valley's single most-open
        sector if none does.
        """
        clipped = int(np.clip(goal_idx, start, end))
        if cls._windowed_clearance(histogram, clipped, start, end) >= comfortable:
            return clipped
        for offset in range(1, end - start + 1):
            for idx in (clipped - offset, clipped + offset):
                if start <= idx <= end and cls._windowed_clearance(
                    histogram, idx, start, end
                ) >= comfortable:
                    return idx
        return start + int(np.argmax(histogram[start:end + 1]))

    @staticmethod
    def _find_valleys(free: np.ndarray) -> list[tuple[int, int]]:
        """Contiguous runs of ``True`` in ``free`` as inclusive (start, end) pairs."""
        valleys = []
        start = None
        for i, is_free in enumerate(free):
            if is_free and start is None:
                start = i
            elif not is_free and start is not None:
                valleys.append((start, i - 1))
                start = None
        if start is not None:
            valleys.append((start, len(free) - 1))
        return valleys

    def _angle_to_index(self, angle: float) -> int:
        wrapped = _wrap_to_pi(angle)
        idx = math.floor((wrapped + math.pi) / self.sector_width)
        return min(idx, self.config.num_sectors - 1)

    def _index_to_angle(self, idx: int) -> float:
        return float(_wrap_to_pi(-math.pi + (idx + 0.5) * self.sector_width))


class FuzzyController:
    """Mamdani fuzzy controller: (clearance, heading error) -> (v, w).

    Inputs:
        distance: clearance (m) of the sector VFH selected.
        heading_error: angle (rad, ``[-pi, pi]``) from current heading
            to the selected sector.

    Outputs:
        linear_vel: forward speed (m/s), always >= 0.
        angular_vel: turn rate (rad/s), positive = left/CCW.

    Rule design: angular velocity tracks heading error directly (five
    rules, one per heading term). Linear velocity is throttled down
    both by proximity to obstacles and by how sharp a turn is needed,
    since a sharp turn at high speed is exactly the failure mode that
    causes oscillation/obstacle-clipping in fuzzy VFH controllers.
    """

    def __init__(self, config: FISVFHConfig) -> None:
        self.config = config
        self._system = self._build_system()
        self._sim = ctrl.ControlSystemSimulation(self._system)
        self._last_angular_vel = 0.0

    def reset(self) -> None:
        """Forget the slew-rate history (e.g. after a long stop)."""
        self._last_angular_vel = 0.0

    def compute(self, distance: float, heading_error: float) -> tuple[float, float]:
        """Run one inference cycle and return ``(linear_vel, angular_vel)``."""
        cfg = self.config
        wrapped_heading_error = float(np.clip(_wrap_to_pi(heading_error), -math.pi, math.pi))
        self._sim.input["distance"] = float(np.clip(distance, 0.0, cfg.max_range))
        self._sim.input["heading_error"] = wrapped_heading_error
        self._sim.compute()
        linear_vel = float(self._sim.output["linear_vel"])
        angular_vel = float(self._sim.output["angular_vel"])

        if abs(wrapped_heading_error) >= cfg.angular_deadzone and (
            abs(angular_vel) < cfg.min_effective_angular_vel
        ):
            # A real turn is needed but the proportional output is weak
            # enough to be absorbed by friction/stiction — floor it to a
            # magnitude that actually moves the robot. See
            # FISVFHConfig.min_effective_angular_vel.
            sign_source = angular_vel if abs(angular_vel) > 1e-9 else wrapped_heading_error
            sign = math.copysign(1.0, sign_source)
            angular_vel = sign * min(cfg.min_effective_angular_vel, cfg.max_angular_vel)

        if cfg.angular_slew_rate > 0.0:
            # Don't request a faster change than the robot's own
            # acceleration limit can track — otherwise the breakaway
            # floor snaps the command from ~0 to full instantly every
            # cycle, the real response lags behind, and the robot
            # overshoots into a stationary circle instead of clearing
            # the obstacle. See FISVFHConfig.angular_slew_rate.
            max_delta = cfg.angular_slew_rate * cfg.control_period
            delta = angular_vel - self._last_angular_vel
            angular_vel = self._last_angular_vel + max(-max_delta, min(max_delta, delta))

        self._last_angular_vel = angular_vel
        return linear_vel, angular_vel

    def _build_system(self) -> ctrl.ControlSystem:
        cfg = self.config
        d_safe = cfg.safe_distance

        distance = ctrl.Antecedent(np.linspace(0.0, cfg.max_range, 201), "distance")
        heading_error = ctrl.Antecedent(np.linspace(-math.pi, math.pi, 181), "heading_error")
        linear_vel = ctrl.Consequent(np.linspace(0.0, cfg.max_linear_vel, 101), "linear_vel")
        angular_vel = ctrl.Consequent(
            np.linspace(-cfg.max_angular_vel, cfg.max_angular_vel, 101), "angular_vel"
        )

        d = distance.universe
        distance["very_close"] = fuzz.trapmf(d, [0.0, 0.0, 0.5 * d_safe, d_safe])
        distance["close"] = fuzz.trimf(d, [0.5 * d_safe, d_safe, 3.0 * d_safe])
        distance["medium"] = fuzz.trimf(d, [d_safe, 3.0 * d_safe, 6.0 * d_safe])
        distance["far"] = fuzz.trapmf(
            d, [3.0 * d_safe, 6.0 * d_safe, cfg.max_range, cfg.max_range]
        )

        h = heading_error.universe
        pi = math.pi
        heading_error["neg_large"] = fuzz.trapmf(h, [-pi, -pi, -pi / 2, -pi / 6])
        heading_error["neg_small"] = fuzz.trimf(h, [-pi / 2, -pi / 6, 0.0])
        heading_error["zero"] = fuzz.trimf(h, [-pi / 6, 0.0, pi / 6])
        heading_error["pos_small"] = fuzz.trimf(h, [0.0, pi / 6, pi / 2])
        heading_error["pos_large"] = fuzz.trapmf(h, [pi / 6, pi / 2, pi, pi])

        lv = linear_vel.universe
        vmax = cfg.max_linear_vel
        linear_vel["stop"] = fuzz.trapmf(lv, [0.0, 0.0, 0.05 * vmax, 0.15 * vmax])
        linear_vel["slow"] = fuzz.trimf(lv, [0.05 * vmax, 0.25 * vmax, 0.5 * vmax])
        linear_vel["medium"] = fuzz.trimf(lv, [0.3 * vmax, 0.55 * vmax, 0.8 * vmax])
        linear_vel["fast"] = fuzz.trapmf(lv, [0.6 * vmax, 0.85 * vmax, vmax, vmax])

        av = angular_vel.universe
        wmax = cfg.max_angular_vel
        angular_vel["sharp_right"] = fuzz.trapmf(av, [-wmax, -wmax, -0.6 * wmax, -0.3 * wmax])
        angular_vel["right"] = fuzz.trimf(av, [-0.6 * wmax, -0.3 * wmax, 0.0])
        angular_vel["straight"] = fuzz.trimf(av, [-0.15 * wmax, 0.0, 0.15 * wmax])
        angular_vel["left"] = fuzz.trimf(av, [0.0, 0.3 * wmax, 0.6 * wmax])
        angular_vel["sharp_left"] = fuzz.trapmf(av, [0.3 * wmax, 0.6 * wmax, wmax, wmax])

        h_tight = heading_error["neg_small"] | heading_error["zero"] | heading_error["pos_small"]
        h_wide = heading_error["neg_large"] | heading_error["pos_large"]

        rules = [
            # Angular velocity tracks heading error directly.
            ctrl.Rule(heading_error["neg_large"], angular_vel["sharp_right"]),
            ctrl.Rule(heading_error["neg_small"], angular_vel["right"]),
            ctrl.Rule(heading_error["zero"], angular_vel["straight"]),
            ctrl.Rule(heading_error["pos_small"], angular_vel["left"]),
            ctrl.Rule(heading_error["pos_large"], angular_vel["sharp_left"]),
            # Linear velocity: throttle down near obstacles and on sharp turns.
            ctrl.Rule(distance["very_close"], linear_vel["stop"]),
            ctrl.Rule(distance["close"] & h_wide, linear_vel["stop"]),
            ctrl.Rule(distance["close"] & h_tight, linear_vel["slow"]),
            ctrl.Rule(distance["medium"] & h_wide, linear_vel["slow"]),
            ctrl.Rule(distance["medium"] & h_tight, linear_vel["medium"]),
            ctrl.Rule(distance["far"] & h_wide, linear_vel["medium"]),
            ctrl.Rule(distance["far"] & h_tight, linear_vel["fast"]),
        ]
        return ctrl.ControlSystem(rules)


class FISVFHController:
    """Ties the VFH histogram and fuzzy controller into one control-cycle call."""

    def __init__(self, config: FISVFHConfig | None = None) -> None:
        self.config = config or FISVFHConfig()
        self.histogram = VFHHistogram(self.config)
        self.fuzzy = FuzzyController(self.config)

    def update(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        heading_to_goal: float,
        distance_to_goal: float = 100.0,
    ) -> tuple[float, float, bool]:
        """Full FISVFH cycle: LiDAR scan + goal heading + distance -> ``(v, w, arrived)``.

        Args:
            ranges: LiDAR ranges (``sensor_msgs/LaserScan.ranges``).
            angle_min: angle (rad) of ``ranges[0]``, robot frame.
            angle_increment: angular step (rad) between samples.
            heading_to_goal: desired heading to the goal (rad, robot frame).
            distance_to_goal: distance to goal (m).

        Returns:
            ``(linear_vel, angular_vel, arrived)`` tuple.
        """
        if distance_to_goal <= self.config.goal_tolerance:
            return 0.0, 0.0, True

        hist = self.histogram.build(ranges, angle_min, angle_increment)
        direction, clearance = self.histogram.select_direction(hist, heading_to_goal)

        if clearance < self.config.safe_distance:
            # VFH found no sector meeting the safety threshold at all — rather
            # than drive the fuzzy controller's compromise output, rotate in
            # place toward the best (least-bad) direction it did find.
            turn_dir = 1.0 if direction >= 0 else -1.0
            return 0.0, turn_dir * self.config.max_angular_vel, False

        lin, ang = self.fuzzy.compute(clearance, direction)
        return lin, ang, False

    def compute(
        self,
        ranges: np.ndarray,
        angle_min: float,
        angle_increment: float,
        heading_to_goal: float,
        distance_to_goal: float = 100.0,
    ) -> tuple[float, float]:
        """Full FISVFH cycle: LiDAR scan + goal heading -> ``(v, w)``."""
        lin, ang, _ = self.update(
            ranges, angle_min, angle_increment, heading_to_goal, distance_to_goal
        )
        return lin, ang


if __name__ == "__main__":
    # Quick manual sanity check — no ROS required. Simulates a 270 deg
    # LiDAR sweep with an obstacle dead ahead and open space elsewhere,
    # with a goal straight ahead, then prints the resulting command.
    controller = FISVFHController()

    n = 270
    demo_angle_min = math.radians(-135)
    demo_angle_increment = math.radians(1)
    demo_ranges = np.full(n, 8.0)
    demo_ranges[125:145] = 0.8  # obstacle dead ahead (still >= safe_distance)

    v, w = controller.compute(
        demo_ranges, demo_angle_min, demo_angle_increment, heading_to_goal=0.0
    )
    print(f"linear_vel={v:.3f} m/s, angular_vel={w:.3f} rad/s")
