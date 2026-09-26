"""
platoon_follower.py — Weeks 7-8, Task 4: follower tracks a target robot.

Subscribes to /<target_name>/fiducial_pose (the target robot's published
pose), computes a following point (default 0.5 m behind the target along
its heading, matching §IV-A.1), and drives to maintain that offset.

Uses a path-relative error decomposition: longitudinal error (along the
leader's heading) controls linear velocity via feedforward + P, and
lateral error (cross-track) + heading alignment control angular velocity.
This avoids the bearing-angle singularity and oscillation that a naive
point-tracking P controller would have at close range.

Also publishes this follower's own fiducial_pose so the next follower in
a column formation (Task 5) can track it.
"""

from __future__ import annotations

import math

import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node

from titan_navigation.fisvfh_node import yaw_from_quaternion


def _wrap_to_pi(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


class PlatoonFollowerNode(Node):

    def __init__(self) -> None:
        super().__init__("platoon_follower")

        self.declare_parameter("target_name", "leader")
        self.declare_parameter("follow_distance", 0.5)
        self.declare_parameter("kp_longitudinal", 1.5)
        self.declare_parameter("kp_lateral", 2.0)
        self.declare_parameter("kp_heading", 1.5)
        self.declare_parameter("max_linear_vel", 0.8)
        self.declare_parameter("max_angular_vel", 1.5)
        self.declare_parameter("control_frequency", 10.0)
        self.declare_parameter("fiducial_publish_frequency", 10.0)
        self.declare_parameter("leader_timeout", 2.0)
        self.declare_parameter("speed_ema_alpha", 0.4)

        target = str(self.get_parameter("target_name").value)
        self._follow_dist = float(self.get_parameter("follow_distance").value)
        self._kp_lon = float(self.get_parameter("kp_longitudinal").value)
        self._kp_lat = float(self.get_parameter("kp_lateral").value)
        self._kp_hdg = float(self.get_parameter("kp_heading").value)
        self._max_lin = float(self.get_parameter("max_linear_vel").value)
        self._max_ang = float(self.get_parameter("max_angular_vel").value)
        control_freq = float(self.get_parameter("control_frequency").value)
        fiducial_freq = float(self.get_parameter("fiducial_publish_frequency").value)
        self._leader_timeout_sec = float(self.get_parameter("leader_timeout").value)
        self._ema_alpha = float(self.get_parameter("speed_ema_alpha").value)

        self._latest_leader_pose: PoseStamped | None = None
        self._latest_odom: Odometry | None = None
        self._last_leader_stamp: float = 0.0
        self._leader_speed: float = 0.0

        self.create_subscription(
            PoseStamped, f"/{target}/fiducial_pose", self._leader_cb, 10
        )
        self.create_subscription(Odometry, "odometry/filtered", self._odom_cb, 10)

        self._cmd_pub = self.create_publisher(Twist, "cmd_vel", 10)
        self._fiducial_pub = self.create_publisher(PoseStamped, "fiducial_pose", 10)

        self.create_timer(1.0 / control_freq, self._control_cycle)
        self.create_timer(1.0 / fiducial_freq, self._publish_fiducial)

        self.get_logger().info(
            f"Following /{target}/fiducial_pose at {self._follow_dist:.2f} m offset"
        )

    # ── callbacks ──

    def _leader_cb(self, msg: PoseStamped) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._latest_leader_pose is not None and stamp > self._last_leader_stamp + 0.01:
            prev = self._latest_leader_pose.pose
            dx = msg.pose.position.x - prev.position.x
            dy = msg.pose.position.y - prev.position.y
            dt = stamp - self._last_leader_stamp
            speed = math.hypot(dx, dy) / dt
            a = self._ema_alpha
            self._leader_speed = a * speed + (1.0 - a) * self._leader_speed
        self._latest_leader_pose = msg
        self._last_leader_stamp = stamp

    def _odom_cb(self, msg: Odometry) -> None:
        self._latest_odom = msg

    # ── fiducial ──

    def _publish_fiducial(self) -> None:
        if self._latest_odom is None:
            return
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._latest_odom.header.frame_id
        msg.pose = self._latest_odom.pose.pose
        self._fiducial_pub.publish(msg)

    # ── control ──

    def _control_cycle(self) -> None:
        if self._latest_leader_pose is None or self._latest_odom is None:
            self.get_logger().info(
                "Waiting for leader pose and own odometry...",
                throttle_duration_sec=2.0,
            )
            self._publish_cmd(0.0, 0.0)
            return

        now_sec = self.get_clock().now().nanoseconds / 1e9
        age = now_sec - self._last_leader_stamp
        if age > self._leader_timeout_sec:
            self.get_logger().warning(
                f"Leader pose stale ({age:.1f}s) — stopping.",
                throttle_duration_sec=1.0,
            )
            self._publish_cmd(0.0, 0.0)
            return

        # Leader pose and heading
        lp = self._latest_leader_pose.pose
        leader_yaw = yaw_from_quaternion(
            lp.orientation.x, lp.orientation.y, lp.orientation.z, lp.orientation.w
        )
        cos_ly = math.cos(leader_yaw)
        sin_ly = math.sin(leader_yaw)

        # Follow point: follow_distance behind leader along its heading
        target_x = lp.position.x - self._follow_dist * cos_ly
        target_y = lp.position.y - self._follow_dist * sin_ly

        # Follower pose and heading
        fp = self._latest_odom.pose.pose
        follower_yaw = yaw_from_quaternion(
            fp.orientation.x, fp.orientation.y, fp.orientation.z, fp.orientation.w
        )

        # Error in leader-heading-aligned frame
        ex = fp.position.x - target_x
        ey = fp.position.y - target_y
        long_err = ex * cos_ly + ey * sin_ly    # +ahead / -behind
        lat_err = -ex * sin_ly + ey * cos_ly    # +left  / -right

        # Heading alignment: how much follower must turn to face like leader
        heading_err = _wrap_to_pi(leader_yaw - follower_yaw)

        # Linear: feedforward + P correction for longitudinal deficit
        linear = self._leader_speed - self._kp_lon * long_err
        linear = max(0.0, min(linear, self._max_lin))

        # Angular: lateral cross-track correction + heading alignment
        angular = -self._kp_lat * lat_err + self._kp_hdg * heading_err
        angular = max(-self._max_ang, min(self._max_ang, angular))

        dist = math.hypot(ex, ey)
        self.get_logger().info(
            f"dist={dist:.2f}m  long={long_err:+.2f}  lat={lat_err:+.2f}  "
            f"hdg_err={math.degrees(heading_err):+.1f}°  "
            f"cmd=(v={linear:.2f}, w={angular:+.2f})",
            throttle_duration_sec=1.0,
        )

        self._publish_cmd(linear, angular)

    def _publish_cmd(self, linear: float, angular: float) -> None:
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PlatoonFollowerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
