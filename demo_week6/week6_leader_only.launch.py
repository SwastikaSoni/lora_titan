"""
week6_leader_only.launch.py — DEMO ONLY, standalone Week 6 smoke test.

Spawns ONLY the Husky leader (no followers) in week6_leader_only.sdf
and drives it point-to-point through the obstacle field via
titan_navigation's fisvfh_node — the actual Week 6 deliverable,
unmodified. Not part of any package: run by direct file path, no
colcon build/install involved, nothing here touches src/, worlds/, or
the existing build/install trees. Safe to delete this whole
demo_week6/ directory after the demo.

fisvfh_node.py subscribes to the *absolute* topics "/scan",
"/odometry/filtered", "/cmd_vel" (see its own docstring) — the
remappings below rewrite those exact strings to the leader's real
namespaced topics (bridge.yaml only exposes namespaced topics now, no
bare /scan) without editing fisvfh_node.py or its Week 6 launch file.

Usage:
    ros2 launch ~/titan_ws/demo_week6/week6_leader_only.launch.py
"""

import importlib.util
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess, SetEnvironmentVariable
from launch_ros.actions import Node

WS_ROOT = os.path.expanduser("~/titan_ws")
DEMO_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_make_robot_nodes():
    """Import make_robot_nodes() straight from sim_bringup.launch.py's
    source file — reuses the exact TF/navsat/EKF wiring for "leader"
    without duplicating it here or modifying that file."""
    path = os.path.join(
        WS_ROOT, "src/titan_coordination/launch/sim_bringup.launch.py"
    )
    spec = importlib.util.spec_from_file_location("sim_bringup", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.make_robot_nodes


def generate_launch_description():
    make_robot_nodes = _load_make_robot_nodes()

    pkg_coordination_config = os.path.join(WS_ROOT, "src/titan_coordination/config")
    bridge_config = os.path.join(pkg_coordination_config, "bridge.yaml")
    ekf_config = os.path.join(pkg_coordination_config, "ekf.yaml")
    navsat_config = os.path.join(pkg_coordination_config, "navsat.yaml")
    world_path = os.path.join(DEMO_DIR, "week6_leader_only.sdf")
    models_dir = os.path.join(WS_ROOT, "models")

    set_gz_resource_path = SetEnvironmentVariable(
        name="GZ_SIM_RESOURCE_PATH", value=models_dir
    )

    start_gazebo = ExecuteProcess(
        cmd=["gz", "sim", "-r", "-v", "1", world_path],
        output="screen",
    )

    # bridge.yaml lists all 5 robots' topics, but only "leader"'s gz
    # topics actually exist in this world — the follower entries just
    # never have anything to bridge, harmlessly.
    start_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        output="screen",
        parameters=[{"config_file": bridge_config, "use_sim_time": True}],
    )

    robot_nodes = make_robot_nodes("leader", "husky", ekf_config, navsat_config)

    fisvfh_node = Node(
        package="titan_navigation",
        executable="fisvfh_node",
        name="fisvfh_node",
        output="screen",
        parameters=[
            {
                "use_sim_time": True,
                # Same defaults as titan_navigation's fisvfh_nav.launch.py —
                # ~65 m due east, clearing the obstacle field (x=10..55).
                "goal_lat": 42.4129540624,
                "goal_lon": -83.1352735500,
            }
        ],
        remappings=[
            ("/scan", "/leader/scan"),
            ("/odometry/filtered", "/leader/odometry/filtered"),
            ("/cmd_vel", "/leader/cmd_vel"),
        ],
    )

    return LaunchDescription(
        [
            set_gz_resource_path,
            start_gazebo,
            start_bridge,
            *robot_nodes,
            fisvfh_node,
        ]
    )
