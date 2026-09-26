"""
platoon_leader.py — Weeks 7-8, Task 2: leader drives to goal autonomously.

Runs the same FISVFH controller as titan_navigation.fisvfh_node (Week 6)
but namespace-relative, so it can be launched under namespace="leader"
(see platoon_leader.launch.py) and correctly picks up /leader/scan,
/leader/odometry/filtered from the fleet bridge instead of the bare
/scan, /odometry/filtered a single unnamespaced robot would have used.

Also publishes a synthetic fiducial pose on "fiducial_pose" (->
/leader/fiducial_pose) at a fixed rate. This stands in for ArUco/QR
detection (see README §7 — ArUco is skipped in sim as it wouldn't
change the coordination result). Followers (Task 4+) will subscribe to
this to compute a following offset, exactly like they would parse a
detected marker pose from a camera image.
"""

from __future__ import annotations

import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import UInt8MultiArray

from titan_communication.mesh.frame import encode_payload
from titan_navigation.fisvfh_node import (
    _DEFAULT_DATUM_LAT,
    _DEFAULT_DATUM_LON,
    gps_to_local,
    yaw_from_quaternion,
)
from titan_navigation.fuzzy_rules import FISVFHConfig, FISVFHController


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PlatoonLeaderNode(Node):
    """Drives the leader robot to a goal GPS fix via FISVFH; broadcasts
    its own pose as the platoon's fiducial reference."""

    def __init__(self) -> None:
        super().__init__("platoon_leader")

        self.declare_parameter("goal_lat", math.nan)
        self.declare_parameter("goal_lon", math.nan)
        self.declare_parameter("datum_lat", _DEFAULT_DATUM_LAT)
        self.declare_parameter("datum_lon", _DEFAULT_DATUM_LON)
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("fiducial_publish_frequency", 10.0)
        self.declare_parameter("telemetry_frequency", 1.0)
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
        fiducial_frequency = float(self.get_parameter("fiducial_publish_frequency").value)
        telemetry_frequency = float(self.get_parameter("telemetry_frequency").value)
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
                "goal_lat/goal_lon not set — leader will idle publishing zero "
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
        # Relative topic names: launched under namespace="leader", these
        # resolve to /leader/scan, /leader/odometry/filtered, /leader/cmd_vel,
        # /leader/fiducial_pose — see husky_leader/model.sdf for why the
        # Gazebo-side <topic> tags can't rely on this but plain ROS 2 topic
        # names (no leading slash) can.
        self.create_subscription(LaserScan, "scan", self._scan_cb, sensor_qos)
        self.create_subscription(Odometry, "odometry/filtered", self._odom_cb, 10)
        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._fiducial_pub = self.create_publisher(PoseStamped, "fiducial_pose", 10)
        self._mesh_tx_pub = self.create_publisher(UInt8MultiArray, "mesh_tx", 10)

        self._control_timer = self.create_timer(1.0 / control_frequency, self._control_cycle)
        self._fiducial_timer = self.create_timer(
            1.0 / fiducial_frequency, self._publish_fiducial_pose
        )
        self._telemetry_timer = self.create_timer(
            1.0 / telemetry_frequency, self._publish_telemetry
        )

    def _scan_cb(self, msg: LaserScan) -> None:
        self._latest_scan = msg
        self._last_scan_time = self.get_clock().now()

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg
        self._last_odom_time = self.get_clock().now()

    def _publish_fiducial_pose(self) -> None:
        """Broadcast the leader's own pose as the platoon's fiducial
        reference — a stand-in for what ArUco/QR detection would give a
        follower's camera (see module docstring / README §7)."""
        if self._latest_odom is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._latest_odom.header.frame_id
        msg.pose = self._latest_odom.pose.pose
        self._fiducial_pub.publish(msg)

    def _publish_telemetry(self) -> None:
        """Broadcast pose + heading + velocity as a TELEM frame over the
        mesh (Weeks 7-8, Task 3). mesh_bridge_node owns the actual
        MeshStack/VirtualChannel and does the real send() + RSSI calc —
        this just hands it the payload to put on air."""
        if self._latest_odom is None:
            return
        pose = self._latest_odom.pose.pose
        yaw = yaw_from_quaternion(
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
        )
        twist = self._latest_odom.twist.twist
        payload = encode_payload({
            "x": pose.position.x,
            "y": pose.position.y,
            "yaw": yaw,
            "linear_vel": twist.linear.x,
            "angular_vel": twist.angular.z,
        })
        msg = UInt8MultiArray()
        msg.data = list(payload)
        self._mesh_tx_pub.publish(msg)

    def _control_cycle(self) -> None:
        if self._goal_local is None:
            self._publish(0.0, 0.0)
            return

        if self._latest_scan is None or self._latest_odom is None:
            self.get_logger().info(
                "Waiting for scan and odometry/filtered...", throttle_duration_sec=2.0
            )
            self._publish(0.0, 0.0)
            return

        now = self.get_clock().now()
        if now - self._last_scan_time > self._stale_timeout:
            self.get_logger().warning("scan is stale — stopping.", throttle_duration_sec=1.0)
            self._publish(0.0, 0.0)
            return
        if now - self._last_odom_time > self._stale_timeout:
            self.get_logger().warning(
                "odometry/filtered is stale — stopping.", throttle_duration_sec=1.0
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
    node = PlatoonLeaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
