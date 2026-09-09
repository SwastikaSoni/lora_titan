

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

from titan_navigation.fuzzy_rules import FISVFHConfig, FISVFHController

# GPS origin baked into worlds/phase1_region2_open.sdf and
# phase1_region2_obstacles.sdf (Detroit Mercy campus, paper Table I).
_DEFAULT_DATUM_LAT = 42.4129540624
_DEFAULT_DATUM_LON = -83.1360653105

_EARTH_RADIUS_M = 6371000.0


def gps_to_local(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    """Equirectangular ENU offset (x=east, y=north), meters, of (lat, lon)
    relative to a datum (lat0, lon0).

    Good enough at course scale (tens to low hundreds of meters — see
    ``phase1_region2_*.sdf``); matches the sim's ENU convention (IMU
    yaw=0 = east per ``navsat.yaml``'s ``yaw_offset: 0.0``), so the
    result lines up directly with the EKF's local (x, y) without any
    extra rotation.
    """
    lat0_rad = math.radians(lat0)
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = dlon * math.cos(lat0_rad) * _EARTH_RADIUS_M
    y = dlat * _EARTH_RADIUS_M
    return x, y


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (rotation about Z, ENU/REP103 convention) from a quaternion."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class FISVFHNode(Node):
    """Drives the robot from its current pose to a goal GPS fix via FISVFH."""

    def __init__(self) -> None:
        super().__init__("fisvfh_node")

        self.declare_parameter("goal_lat", math.nan)
        self.declare_parameter("goal_lon", math.nan)
        self.declare_parameter("datum_lat", _DEFAULT_DATUM_LAT)
        self.declare_parameter("datum_lon", _DEFAULT_DATUM_LON)
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("stale_timeout", 0.5)
        self.declare_parameter("num_sectors", 72)
        self.declare_parameter("max_range", 10.0)
        self.declare_parameter("safe_distance", 0.6)
        self.declare_parameter("goal_tolerance", 0.8)
        self.declare_parameter("max_linear_vel", 0.6)
        self.declare_parameter("max_angular_vel", 1.5)
        self.declare_parameter("direction_change_penalty", 0.6)
        self.declare_parameter("comfort_factor", 1.5)
        self.declare_parameter("angular_deadzone", 0.05)
        self.declare_parameter("min_effective_angular_vel", 1.0)
        self.declare_parameter("angular_slew_rate", 2.0)

        control_frequency = float(self.get_parameter("control_frequency").value)
        goal_lat = float(self.get_parameter("goal_lat").value)
        goal_lon = float(self.get_parameter("goal_lon").value)
        datum_lat = float(self.get_parameter("datum_lat").value)
        datum_lon = float(self.get_parameter("datum_lon").value)
        self._stale_timeout = Duration(
            seconds=float(self.get_parameter("stale_timeout").value)
        )

        self._goal_local: tuple[float, float] | None = None
        if math.isnan(goal_lat) or math.isnan(goal_lon):
            self.get_logger().error(
                "goal_lat/goal_lon not set — node will idle publishing zero "
                "velocity. Launch with -p goal_lat:=<deg> -p goal_lon:=<deg>."
            )
        else:
            self._goal_local = gps_to_local(goal_lat, goal_lon, datum_lat, datum_lon)
            self.get_logger().info(
                f"Goal GPS ({goal_lat:.7f}, {goal_lon:.7f}) -> "
                f"local ({self._goal_local[0]:.2f}, {self._goal_local[1]:.2f}) m"
            )

        config = FISVFHConfig(
            num_sectors=int(self.get_parameter("num_sectors").value),
            max_range=float(self.get_parameter("max_range").value),
            safe_distance=float(self.get_parameter("safe_distance").value),
            goal_tolerance=float(self.get_parameter("goal_tolerance").value),
            max_linear_vel=float(self.get_parameter("max_linear_vel").value),
            max_angular_vel=float(self.get_parameter("max_angular_vel").value),
            direction_change_penalty=float(
                self.get_parameter("direction_change_penalty").value
            ),
            comfort_factor=float(self.get_parameter("comfort_factor").value),
            angular_deadzone=float(self.get_parameter("angular_deadzone").value),
            min_effective_angular_vel=float(
                self.get_parameter("min_effective_angular_vel").value
            ),
            angular_slew_rate=float(self.get_parameter("angular_slew_rate").value),
            control_period=1.0 / control_frequency,
        )
        self._controller = FISVFHController(config)

        self._latest_scan: LaserScan | None = None
        self._latest_odom: Odometry | None = None
        self._last_scan_time = self.get_clock().now()
        self._last_odom_time = self.get_clock().now()
        self._arrived = False

        sensor_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(LaserScan, "/scan", self._scan_cb, sensor_qos)
        self.create_subscription(Odometry, "/odometry/filtered", self._odom_cb, 10)
        self._cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        period = 1.0 / control_frequency
        self._timer = self.create_timer(period, self._control_cycle)

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._last_scan_time = self.get_clock().now()

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._last_odom_time = self.get_clock().now()

    def _control_cycle(self) -> None:
        if self._goal_local is None:
            self._publish(0.0, 0.0)
            return

        if self._latest_scan is None or self._latest_odom is None:
            self.get_logger().info(
                "Waiting for /scan and /odometry/filtered...", throttle_duration_sec=2.0
            )
            self._publish(0.0, 0.0)
            return

        now = self.get_clock().now()
        if now - self._last_scan_time > self._stale_timeout:
            self.get_logger().warning("/scan is stale — stopping.", throttle_duration_sec=1.0)
            self._publish(0.0, 0.0)
            return
        if now - self._last_odom_time > self._stale_timeout:
            self.get_logger().warning(
                "/odometry/filtered is stale — stopping.", throttle_duration_sec=1.0
            )
            self._publish(0.0, 0.0)
            return

        if self._arrived:
            self._publish(0.0, 0.0)
            return

        pose = self._latest_odom.pose.pose
        yaw = yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        gx, gy = self._goal_local
        dx, dy = gx - pose.position.x, gy - pose.position.y
        distance_to_goal = math.hypot(dx, dy)
        heading_to_goal = _wrap_to_pi(math.atan2(dy, dx) - yaw)

        scan = self._latest_scan
        ranges = np.array(scan.ranges, dtype=float)
        linear_vel, angular_vel, arrived = self._controller.update(
            ranges, scan.angle_min, scan.angle_increment, heading_to_goal, distance_to_goal
        )

        if arrived and not self._arrived:
            self._arrived = True
            self.get_logger().info(f"Goal reached (distance={distance_to_goal:.2f} m).")

        self.get_logger().info(
            f"distance_to_goal={distance_to_goal:.2f} m  "
            f"heading_error={math.degrees(heading_to_goal):+.1f} deg  "
            f"cmd=(v={linear_vel:.2f} m/s, w={angular_vel:+.2f} rad/s)",
            throttle_duration_sec=1.0,
        )

        self._publish(linear_vel, angular_vel)

    def _publish(self, linear_vel: float, angular_vel: float) -> None:
        msg = Twist()
        msg.linear.x = linear_vel
        msg.angular.z = angular_vel
        self._cmd_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = FISVFHNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
