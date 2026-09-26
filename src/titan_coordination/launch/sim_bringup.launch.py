"""
sim_bringup.launch.py

Launches the full 5-robot fleet simulation stack:
  1. Gazebo Ionic with a world file (1 Husky leader + 4 Jackal followers)
  2. ros_gz_bridge (Gazebo topics <-> ROS 2 topics, all 5 robots)
  3. Per-robot static TF publishers (sensor frames)
  4. Per-robot robot_localization EKF (fuses odom + IMU + GPS)
  5. Per-robot navsat_transform (converts GPS lat/lon -> local XY)

Every robot gets its own namespace (/leader, /follower_1 .. /follower_4)
matching the topic namespace baked into that robot's model.sdf plugins
(see husky_leader/model.sdf and scripts/gen_follower_models.py — an
explicit gz-sim plugin <topic> is used literally, not auto-scoped by
instance name, so the namespace has to be baked into each model AND
threaded through every ROS node below via `namespace=`).

Usage:
  ros2 launch titan_coordination sim_bringup.launch.py
  ros2 launch titan_coordination sim_bringup.launch.py world:=phase1_region2_obstacles
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    SetEnvironmentVariable,
    ExecuteProcess,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


# ── Fleet definition ──
# name -> robot profile. Profile picks which sensor-mount offsets apply
# (must match the corresponding model.sdf: husky_leader/model.sdf for
# "husky", scripts/gen_follower_models.py's constants for "jackal").
FLEET = [
    ("leader", "husky"),
    ("follower_1", "jackal"),
    ("follower_2", "jackal"),
    ("follower_3", "jackal"),
    ("follower_4", "jackal"),
]

# (lidar_z, camera_x, camera_z) relative to base_link, per profile.
SENSOR_OFFSETS = {
    "husky": {"lidar_z": 0.20, "camera_x": 0.49, "camera_z": 0.15},
    "jackal": {"lidar_z": 0.192, "camera_x": 0.214, "camera_z": 0.02},
}


def make_robot_nodes(name: str, profile: str, ekf_config, navsat_config):
    """Static TFs + navsat_transform + EKF for one robot, all namespaced."""
    off = SENSOR_OFFSETS[profile]

    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        namespace=name,
        arguments=[
            '--x', '0', '--y', '0', '--z', str(off['lidar_z']),
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', f'{name}/base_link',
            '--child-frame-id', f'{name}/lidar_link/lidar_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera',
        namespace=name,
        arguments=[
            '--x', str(off['camera_x']), '--y', '0', '--z', str(off['camera_z']),
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', f'{name}/base_link',
            '--child-frame-id', f'{name}/camera_link/camera_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # child frame confirmed via `ros2 topic echo <ns>/imu/data --field
    # header.frame_id` on two differently-named live instances — see
    # scripts/gen_follower_models.py's docstring.
    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_imu',
        namespace=name,
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', f'{name}/base_link',
            '--child-frame-id', f'{name}/base_link/imu_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # navsat_sensor lives on the same base_link element as imu_sensor, so
    # it follows the identical "<ns>/base_link/<sensor_name>" pattern.
    static_tf_gps = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_gps',
        namespace=name,
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', f'{name}/base_link',
            '--child-frame-id', f'{name}/base_link/navsat_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # navsat_transform: GPS lat/lon -> local XY. Must start before the EKF.
    # All remap targets are RELATIVE — namespace=name prefixes them to
    # /<name>/imu/data etc. Left column is navsat_transform_node's own
    # internal default topic name (unrelated to our namespacing scheme).
    start_navsat = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        namespace=name,
        output='screen',
        parameters=[
            navsat_config,
            {
                'use_sim_time': True,
                'base_link_frame_id': f'{name}/base_link',
                'world_frame_id': f'{name}/odom',
            },
        ],
        remappings=[
            ('imu', 'imu/data'),
            ('gps/fix', 'gps/fix'),
            ('odometry/filtered', 'odometry/filtered'),
            ('odometry/gps', 'odometry/gps'),
            ('gps/filtered', 'gps/filtered'),
        ],
    )

    # EKF: fuses wheel odom + IMU + GPS. Output: <name>/odometry/filtered —
    # the one pose estimate everything downstream (FISVFH, follower
    # controller, LoRa telemetry) reads for this robot.
    start_ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        namespace=name,
        output='screen',
        parameters=[
            ekf_config,
            {
                'use_sim_time': True,
                'odom_frame': f'{name}/odom',
                'base_link_frame': f'{name}/base_link',
                'world_frame': f'{name}/odom',
            },
        ],
    )

    return [
        static_tf_lidar,
        static_tf_camera,
        static_tf_imu,
        static_tf_gps,
        start_navsat,
        start_ekf,
    ]


def generate_launch_description():
    # ── Paths ──
    pkg_coordination = get_package_share_directory('titan_coordination')
    ws_root = os.path.expanduser('~/titan_ws')
    worlds_dir = os.path.join(ws_root, 'worlds')
    models_dir = os.path.join(ws_root, 'models')

    bridge_config = os.path.join(pkg_coordination, 'config', 'bridge.yaml')
    ekf_config = os.path.join(pkg_coordination, 'config', 'ekf.yaml')
    navsat_config = os.path.join(pkg_coordination, 'config', 'navsat.yaml')

    # ── Launch arguments ──
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='phase1_region2_open',
        description='World file name (without .sdf extension)',
        choices=[
            'phase1_region2_open',
            'phase1_region2_obstacles',
            'phase1_region3_earthquake',
            'phase1_demo_compact',
        ],
    )

    # ── Environment ──
    set_gz_resource_path = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=models_dir,
    )

    # ── 1. Gazebo ──
    start_gazebo = ExecuteProcess(
        cmd=[
            'gz', 'sim', '-r', '-v', '1',
            PathJoinSubstitution([
                worlds_dir,
                [LaunchConfiguration('world'), '.sdf'],
            ]),
        ],
        output='screen',
    )

    # ── 2. ros_gz_bridge ──
    # One shared bridge node — bridge.yaml already has all 5 robots'
    # topics fully namespaced (e.g. "/leader/scan"), so no per-robot
    # bridge namespace is needed here.
    start_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        name='ros_gz_bridge',
        output='screen',
        parameters=[{
            'config_file': bridge_config,
            'use_sim_time': True,
        }],
    )

    # ── 3-5. Per-robot TF + navsat_transform + EKF ──
    robot_nodes = []
    for name, profile in FLEET:
        robot_nodes.extend(make_robot_nodes(name, profile, ekf_config, navsat_config))

    # ── Build launch description ──
    return LaunchDescription([
        world_arg,
        set_gz_resource_path,
        start_gazebo,
        start_bridge,
        *robot_nodes,
    ])
