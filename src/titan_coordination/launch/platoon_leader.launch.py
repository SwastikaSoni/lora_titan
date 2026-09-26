"""
platoon_leader.launch.py — Weeks 7-8, Task 2+3 smoke-test launch.

Brings up the full fleet (sim_bringup: Gazebo + bridge + per-robot TF/
EKF/navsat, all 5 robots), layers platoon_leader on the /leader
namespace (driving it to a configurable goal GPS fix, publishing its
fiducial pose and broadcasting TELEM over the mesh), and starts the
shared mesh_bridge_node + a bs_sink_node to verify the BS receives it.
Followers stay stationary (Task 5+ wires them up).

Usage:
  ros2 launch titan_coordination platoon_leader.launch.py
  ros2 launch titan_coordination platoon_leader.launch.py world:=phase1_region2_obstacles
  ros2 launch titan_coordination platoon_leader.launch.py \
      goal_lat:=42.4130640624 goal_lon:=-83.1360653105

Default goal is ~45 m due east of the world's GPS origin
(42.4129540624, -83.1360653105) — clears phase1_demo_compact.sdf's
obstacle field (x=10..32 m) with margin, and is a valid straight-line
target in phase1_region2_open.sdf too. NOTE: this default is shared
across all `world:=` choices — phase1_region2_obstacles.sdf's field
still spans x=10..55 m, so a default run there now stops short of the
last two obstacles (obstacle_box_6 @ 52, obstacle_cyl_4 @ 55). Pass
goal_lat/goal_lon explicitly if you need the old ~65 m target there.
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
        choices=[
            "phase1_region2_open",
            "phase1_region2_obstacles",
            "phase1_region3_earthquake",
            "phase1_demo_compact",
        ],
    )
    goal_lat_arg = DeclareLaunchArgument(
        "goal_lat",
        default_value="42.4129540624",
        description="Goal latitude (deg). Default: ~45 m east of the world's "
        "GPS origin, clearing phase1_demo_compact's obstacle field.",
    )
    goal_lon_arg = DeclareLaunchArgument(
        "goal_lon",
        default_value="-83.1355171686",
        description="Goal longitude (deg).",
    )
    control_frequency_arg = DeclareLaunchArgument(
        "control_frequency", default_value="10.0", description="FISVFH control loop rate (Hz)."
    )
    fiducial_publish_frequency_arg = DeclareLaunchArgument(
        "fiducial_publish_frequency",
        default_value="10.0",
        description="Rate (Hz) at which the leader broadcasts its fiducial pose.",
    )

    # ── 1-5: Gazebo + bridge + TF + navsat + EKF, all 5 robots ──
    sim_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(sim_bringup_launch),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )

    # ── 6: leader's FISVFH + fiducial pose ──
    platoon_leader = Node(
        package="titan_coordination",
        executable="platoon_leader",
        name="platoon_leader",
        namespace="leader",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                "goal_lat": ParameterValue(LaunchConfiguration("goal_lat"), value_type=float),
                "goal_lon": ParameterValue(LaunchConfiguration("goal_lon"), value_type=float),
                "control_frequency": ParameterValue(
                    LaunchConfiguration("control_frequency"), value_type=float
                ),
                "fiducial_publish_frequency": ParameterValue(
                    LaunchConfiguration("fiducial_publish_frequency"), value_type=float
                ),
            }
        ],
    )

    # ── 7-8: shared mesh bridge (leader + BS) and BS-side verification ──
    mesh_bridge_node = Node(
        package="titan_coordination",
        executable="mesh_bridge_node",
        name="mesh_bridge",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )
    bs_sink_node = Node(
        package="titan_coordination",
        executable="bs_sink_node",
        name="bs_sink",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription(
        [
            world_arg,
            goal_lat_arg,
            goal_lon_arg,
            control_frequency_arg,
            fiducial_publish_frequency_arg,
            sim_bringup,
            platoon_leader,
            mesh_bridge_node,
            bs_sink_node,
        ]
    )
