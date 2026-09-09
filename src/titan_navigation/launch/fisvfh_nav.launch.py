"""fisvfh_nav.launch.py — Week 6 FISVFH navigation stack.

Brings up titan_coordination's sim_bringup (Gazebo + ros_gz_bridge +
EKF + navsat_transform) and layers the FISVFH controller node on top,
driving the robot to a configurable goal GPS fix.

Usage:
  ros2 launch titan_navigation fisvfh_nav.launch.py
  ros2 launch titan_navigation fisvfh_nav.launch.py world:=phase1_region2_obstacles
  ros2 launch titan_navigation fisvfh_nav.launch.py \
      goal_lat:=42.4130640624 goal_lon:=-83.1360653105

Default goal is ~65 m due east of the world's GPS origin
(42.4129540624, -83.1360653105) — past the obstacle field in
phase1_region2_obstacles.sdf (obstacles span x=10..55 m) and equally
valid as a straight-line target in phase1_region2_open.sdf.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_coordination = get_package_share_directory("titan_coordination")
    sim_bringup_launch = os.path.join(pkg_coordination, "launch", "sim_bringup.launch.py")

    # ── Launch arguments ──
    world_arg = DeclareLaunchArgument(
        "world",
        default_value="phase1_region2_open",
        description="World file name (without .sdf extension)",
        choices=["phase1_region2_open", "phase1_region2_obstacles"],
    )
    goal_lat_arg = DeclareLaunchArgument(
        "goal_lat",
        default_value="42.4129540624",
        description="Goal latitude (deg). Default: ~65 m east of the world's "
        "GPS origin, clearing the obstacle field.",
    )
    goal_lon_arg = DeclareLaunchArgument(
        "goal_lon",
        default_value="-83.1352735500",
        description="Goal longitude (deg).",
    )
    control_frequency_arg = DeclareLaunchArgument(
        "control_frequency", default_value="10.0", description="FISVFH control loop rate (Hz)."
    )
    num_sectors_arg = DeclareLaunchArgument(
        "num_sectors", default_value="72", description="VFH polar histogram sector count."
    )
    max_range_arg = DeclareLaunchArgument(
        "max_range", default_value="10.0", description="LiDAR range clamp (m)."
    )
    safe_distance_arg = DeclareLaunchArgument(
        "safe_distance",
        default_value="0.6",
        description="Minimum obstacle clearance (m). Week 6 requires >= 0.5 m; "
        "defaults a bit above that floor for margin.",
    )
    goal_tolerance_arg = DeclareLaunchArgument(
        "goal_tolerance", default_value="0.8", description="Arrival tolerance (m)."
    )
    max_linear_vel_arg = DeclareLaunchArgument(
        "max_linear_vel", default_value="0.6", description="Fuzzy output ceiling (m/s)."
    )
    max_angular_vel_arg = DeclareLaunchArgument(
        "max_angular_vel", default_value="1.5", description="Fuzzy output ceiling (rad/s)."
    )
    direction_change_penalty_arg = DeclareLaunchArgument(
        "direction_change_penalty",
        default_value="0.6",
        description="Cost weight for switching the VFH-selected side (left/right of an "
        "obstacle) away from the previously committed one. Raise this if the robot "
        "oscillates between two nearly-equal gaps instead of committing to one.",
    )
    comfort_factor_arg = DeclareLaunchArgument(
        "comfort_factor",
        default_value="1.5",
        description="Steering targets need clearance >= safe_distance * comfort_factor "
        "to be preferred over the bare-minimum edge of a free valley. Raise this if the "
        "robot still crawls/grazes along an obstacle instead of routing around it.",
    )
    angular_deadzone_arg = DeclareLaunchArgument(
        "angular_deadzone",
        default_value="0.05",
        description="Heading errors smaller than this (rad) are treated as already "
        "aimed at the goal -- no breakaway floor applied, so sector-quantization "
        "noise doesn't make the robot twitch on an open path.",
    )
    min_effective_angular_vel_arg = DeclareLaunchArgument(
        "min_effective_angular_vel",
        default_value="1.0",
        description="Friction/stiction compensation floor (rad/s). Below this measured "
        "threshold this robot doesn't actually rotate -- the torque is absorbed by "
        "static friction. Whenever a real turn is needed (heading error past "
        "angular_deadzone), commanded angular velocity is floored to at least this.",
    )
    angular_slew_rate_arg = DeclareLaunchArgument(
        "angular_slew_rate",
        default_value="2.0",
        description="Max change in commanded angular_vel per second (rad/s^2), matching "
        "the robot's DiffDrive max_angular_acceleration. Without this, "
        "min_effective_angular_vel snaps the command to full floor instantly every "
        "cycle, the robot's real response lags behind, and it overshoots into a "
        "small stationary circle beside the obstacle instead of clearing it.",
    )

    # ── 1-5: Gazebo + bridge + TF + navsat + EKF ──
    sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_bringup_launch),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )

    # ── 6: FISVFH controller ──
    fisvfh_node = Node(
        package="titan_navigation",
        executable="fisvfh_node",
        name="fisvfh_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "goal_lat": ParameterValue(LaunchConfiguration("goal_lat"), value_type=float),
                "goal_lon": ParameterValue(LaunchConfiguration("goal_lon"), value_type=float),
                "control_frequency": ParameterValue(
                    LaunchConfiguration("control_frequency"), value_type=float
                ),
                "num_sectors": ParameterValue(
                    LaunchConfiguration("num_sectors"), value_type=int
                ),
                "max_range": ParameterValue(LaunchConfiguration("max_range"), value_type=float),
                "safe_distance": ParameterValue(
                    LaunchConfiguration("safe_distance"), value_type=float
                ),
                "goal_tolerance": ParameterValue(
                    LaunchConfiguration("goal_tolerance"), value_type=float
                ),
                "max_linear_vel": ParameterValue(
                    LaunchConfiguration("max_linear_vel"), value_type=float
                ),
                "max_angular_vel": ParameterValue(
                    LaunchConfiguration("max_angular_vel"), value_type=float
                ),
                "direction_change_penalty": ParameterValue(
                    LaunchConfiguration("direction_change_penalty"), value_type=float
                ),
                "comfort_factor": ParameterValue(
                    LaunchConfiguration("comfort_factor"), value_type=float
                ),
                "angular_deadzone": ParameterValue(
                    LaunchConfiguration("angular_deadzone"), value_type=float
                ),
                "min_effective_angular_vel": ParameterValue(
                    LaunchConfiguration("min_effective_angular_vel"), value_type=float
                ),
                "angular_slew_rate": ParameterValue(
                    LaunchConfiguration("angular_slew_rate"), value_type=float
                ),
            }
        ],
    )

    return LaunchDescription(
        [
            world_arg,
            goal_lat_arg,
            goal_lon_arg,
            control_frequency_arg,
            num_sectors_arg,
            max_range_arg,
            safe_distance_arg,
            goal_tolerance_arg,
            max_linear_vel_arg,
            max_angular_vel_arg,
            direction_change_penalty_arg,
            comfort_factor_arg,
            angular_deadzone_arg,
            min_effective_angular_vel_arg,
            angular_slew_rate_arg,
            sim_bringup,
            fisvfh_node,
        ]
    )
