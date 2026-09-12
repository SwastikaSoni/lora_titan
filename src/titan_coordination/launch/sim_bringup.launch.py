"""
sim_bringup.launch.py

Launches the full single-robot simulation stack:
  1. Gazebo Ionic with a world file
  2. ros_gz_bridge (Gazebo topics ↔ ROS 2 topics)
  3. Static TF publishers (sensor frames)
  4. robot_localization EKF (fuses odom + IMU + GPS)
  5. navsat_transform (converts GPS lat/lon → local XY)

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
            'gz', 'sim', '-r', '-v', '4',
            PathJoinSubstitution([
                worlds_dir,
                [LaunchConfiguration('world'), '.sdf'],
            ]),
        ],
        output='screen',
    )

    # ── 2. ros_gz_bridge ──
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

    # ── 3. Static TF publishers ──
    # These tell the TF tree how sensor frames relate to the robot body.
    # Every node that reads sensor data needs to know "where on the robot
    # is this sensor?" — that's what TF provides.
    #
    # Parent is "husky/base_link", not bare "base_link": the world file
    # includes this model as <name>husky</name>, and gz-sim-diff-drive-system
    # (no <frame_id> override in model.sdf) publishes /odom with
    # child_frame_id "husky/base_link" — confirmed via
    # `ros2 topic echo /odom`. See the note in ekf.yaml.
    #
    # static_tf_imu's child frame is the IMU sensor's *actual* published
    # frame_id ("husky/base_link/imu_sensor", confirmed via
    # `ros2 topic echo /imu/data`), not an invented "imu_link" — the EKF
    # (and navsat_transform) tf2-lookup each measurement's own frame_id,
    # so it must match exactly or the measurement is silently dropped.

    static_tf_lidar = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_lidar',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0.20',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'husky/base_link',
            '--child-frame-id', 'lidar_link',
        ],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_camera',
        arguments=[
            '--x', '0.49', '--y', '0', '--z', '0.15',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'husky/base_link',
            '--child-frame-id', 'camera_link',
        ],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_imu = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_imu',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'husky/base_link',
            '--child-frame-id', 'husky/base_link/imu_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    static_tf_gps = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='static_tf_gps',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id', 'husky/base_link',
            # navsat_sensor is on base_link with no <pose> override in the
            # SDF, so its real published frame_id is
            # "husky/base_link/navsat_sensor" (confirmed by
            # navsat_transform's "Could not obtain ... transform" error
            # naming this exact frame) — not an invented "gps_link".
            '--child-frame-id', 'husky/base_link/navsat_sensor',
        ],
        parameters=[{'use_sim_time': True}],
    )

    # ── 4. navsat_transform ──
    # Converts GPS lat/lon → local XY odometry.
    # Must start BEFORE the EKF so the EKF has /odometry/gps to subscribe to.
    # Inputs:  /gps/fix (NavSatFix), /imu/data (Imu), /odometry/filtered (Odom)
    # Output:  /odometry/gps (Odometry in local frame)
    start_navsat = Node(
        package='robot_localization',
        executable='navsat_transform_node',
        name='navsat_transform',
        output='screen',
        parameters=[navsat_config],
        remappings=[
            # Map the generic topic names to our actual topic names
            ('imu', '/imu/data'),            # IMU input
            ('gps/fix', '/gps/fix'),         # GPS input
            ('odometry/filtered', '/odometry/filtered'),  # from EKF
            ('odometry/gps', '/odometry/gps'),            # output
            ('gps/filtered', '/gps/filtered'),            # filtered GPS output
        ],
    )

    # ── 5. EKF ──
    # Fuses wheel odom + IMU + GPS (from navsat_transform).
    # Input:  /odom, /imu/data, /odometry/gps
    # Output: /odometry/filtered (the ONE pose estimate everything else uses)
    #         Also publishes odom → base_link TF transform.
    start_ekf = Node(
        package='robot_localization',
        executable='ekf_node',
        name='ekf_filter_node',
        output='screen',
        parameters=[ekf_config],
    )

    # ── Build launch description ──
    return LaunchDescription([
        # Arguments
        world_arg,

        # Environment
        set_gz_resource_path,

        # Gazebo + bridge
        start_gazebo,
        start_bridge,

        # TF
        static_tf_lidar,
        static_tf_camera,
        static_tf_imu,
        static_tf_gps,

        # Localization
        start_navsat,
        start_ekf,
    ])