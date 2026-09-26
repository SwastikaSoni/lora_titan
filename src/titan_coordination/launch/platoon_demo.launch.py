"""
platoon_demo.launch.py — Task 4+: leader + follower(s) in formation.

Includes platoon_leader.launch.py (Gazebo + bridge + TF/EKF/navsat for
all 5 robots, leader FISVFH, mesh bridge, BS sink) and layers follower
control nodes on top.

Usage (Task 4 — single follower):
  ros2 launch titan_coordination platoon_demo.launch.py num_followers:=1

Usage (Task 5 — full column):
  ros2 launch titan_coordination platoon_demo.launch.py num_followers:=4
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

FOLLOWERS = [
    ("follower_1", "leader"),
    ("follower_2", "follower_1"),
    ("follower_3", "follower_2"),
    ("follower_4", "follower_3"),
]


def _spawn_followers(context):
    n = int(LaunchConfiguration("num_followers").perform(context))
    nodes = []
    for i, (fname, target) in enumerate(FOLLOWERS):
        if i >= n:
            break
        nodes.append(
            Node(
                package="titan_coordination",
                executable="platoon_follower",
                name="platoon_follower",
                namespace=fname,
                output="screen",
                parameters=[{
                    "use_sim_time": True,
                    "target_name": target,
                    "follow_distance": 0.5,
                }],
            )
        )
    return nodes


def generate_launch_description():
    pkg = get_package_share_directory("titan_coordination")
    leader_launch = os.path.join(pkg, "launch", "platoon_leader.launch.py")

    world_arg = DeclareLaunchArgument(
        "world", default_value="phase1_region2_open",
        description="World file name (without .sdf)",
    )
    num_followers_arg = DeclareLaunchArgument(
        "num_followers", default_value="1",
        description="Number of followers to activate (1-4)",
    )

    leader_bringup = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(leader_launch),
        launch_arguments={"world": LaunchConfiguration("world")}.items(),
    )

    return LaunchDescription([
        world_arg,
        num_followers_arg,
        leader_bringup,
        OpaqueFunction(function=_spawn_followers),
    ])
