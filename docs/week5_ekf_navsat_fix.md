## Week 5 — Gazebo Scaffolding & Sensor Plumbing — COMPLETE

### What was built

Single simulated Husky-like robot spawns in Gazebo Ionic, publishes all
sensors, drives via `/cmd_vel`, and outputs fused `/odometry/filtered`
from an EKF combining wheel odometry + IMU + GPS.

---

### Task 1: ROS 2 Workspace Skeleton

Created `~/titan_ws/` with four packages:

| Package | Type | Purpose |
|---|---|---|
| `titan_msgs` | CMake (ament_cmake) | Custom .msg definitions (Heartbeat.msg placeholder) |
| `titan_communication` | Python (ament_python) | Mesh stack, virtual LoRa radio (radio/, mesh/, nodes/ subdirs) |
| `titan_coordination` | Python (ament_python) | Platoon leader/follower, launch files, configs |
| `titan_navigation` | Python (ament_python) | FISVFH obstacle avoidance |

Supporting directories: `configs/`, `worlds/`, `models/`, `eval/`, `docs/`, `data/`.

Config files:
- `configs/lora_params.yaml` — paper's exact LoRa settings (SF=10, BW=500kHz, CR=4/5, TX=15dBm, 915MHz)
- `configs/channel.yaml` — Petäjäjärvi 2015 outdoor NLOS pathloss model

```bash
# Build and source
cd ~/titan_ws
source /opt/ros/lyrical/setup.bash
colcon build
source install/setup.bash
```

---

### Task 2: Robot SDF Model

Husky-like leader robot at `models/husky_leader/model.sdf`:
- Chassis: 0.99×0.67×0.20m, 50 kg, differential drive (skid-steer)
- 4 wheels: radius 0.17m, separation 0.571m
- Sensors: IMU (100 Hz), GPS/NavSat (5 Hz), 2D lidar (10 Hz, 360°, 30m range), camera (640×480, 30 fps)
- Plugins: DiffDrive (cmd_vel → wheel torques), JointStatePublisher
- Files: `models/husky_leader/model.config`, `models/husky_leader/model.sdf`

---

### Task 3: Gazebo Worlds

Two world files in `worlds/`:

| World | Description |
|---|---|
| `phase1_region2_open.sdf` | Flat 200×200m ground, no obstacles |
| `phase1_region2_obstacles.sdf` | Same ground + 10 obstacles (boxes, cylinders) from x=10m to x=55m |

Both include:
- GPS origin: Detroit Mercy campus (42.4129540624, -83.1360653105) from paper Table I
- System plugins: Physics, Sensors, Imu, NavSat, Contact, UserCommands, SceneBroadcaster
- Sun, ground plane, and the husky_leader robot

```bash
# Test a world directly in Gazebo
export GZ_SIM_RESOURCE_PATH=~/titan_ws/models
gz sim ~/titan_ws/worlds/phase1_region2_obstacles.sdf
```

---

### Task 4: ros_gz_bridge + Launch File

Bridge config at `src/titan_coordination/config/bridge.yaml` mapping:

| Direction | Gazebo Topic | ROS 2 Topic | Purpose |
|---|---|---|---|
| ROS→GZ | /cmd_vel | /cmd_vel | Drive commands |
| GZ→ROS | /odom | /odom | Wheel odometry |
| GZ→ROS | /imu/data | /imu/data | IMU |
| GZ→ROS | /navsat | /gps/fix | GPS |
| GZ→ROS | /scan | /scan | 2D lidar |
| GZ→ROS | /camera/image | /camera/image | Front camera |
| GZ→ROS | /clock | /clock | Sim time sync |
| GZ→ROS | /joint_states | /joint_states | Wheel joint states |

Launch file at `src/titan_coordination/launch/sim_bringup.launch.py` starts:
1. Gazebo with selected world
2. ros_gz_bridge with bridge.yaml
3. Static TF publishers for lidar_link, camera_link, imu_sensor, navsat_sensor

```bash
# Launch (open world, default)
ros2 launch titan_coordination sim_bringup.launch.py

# Launch with obstacles
ros2 launch titan_coordination sim_bringup.launch.py world:=phase1_region2_obstacles
```

---

### Task 5: robot_localization EKF + navsat_transform

Config files:
- `src/titan_coordination/config/ekf.yaml` — EKF fusing odom + IMU + GPS
- `src/titan_coordination/config/navsat.yaml` — GPS lat/lon → local XY conversion

Data flow:

/odom (50 Hz) ──────────┐
/imu/data (100 Hz) ─────┼──→ EKF ──→ /odometry/filtered (50 Hz)
/gps/fix (5 Hz) │
└──→ navsat_transform ──→ /odometry/gps (5 Hz) ──┘


Key fixes applied during debugging:
- Frame names changed to `husky/odom`, `husky/base_link` (Gazebo namespaces model instance)
- `two_d_mode: true` added to EKF (prevents z-axis drift from IMU gravity)
- Static TF child frames matched to actual Gazebo sensor frame IDs (`husky/base_link/imu_sensor`, `husky/base_link/navsat_sensor`)
- GPS noise in SDF changed from meters to degrees (0.0000045° ≈ 0.5m)

Required apt packages:
```bash
sudo apt install -y \
  ros-lyrical-robot-localization \
  ros-lyrical-ros-gz-bridge \
  ros-lyrical-ros-gz-sim \
  ros-lyrical-tf2-ros
```

---

### Task 6: Smoke Test — PASSED

**All topics publishing at expected rates:**

| Topic | Expected | Actual |
|---|---|---|
| /odom | ~50 Hz | 49.9 Hz |
| /imu/data | ~100 Hz | 99.9 Hz |
| /gps/fix | ~5 Hz | 5.0 Hz |
| /scan | ~10 Hz | 9.8 Hz |
| /odometry/filtered | ~50 Hz | 49.9 Hz |
| /odometry/gps | ~5 Hz | 5.0 Hz |
| /clock | high | 990 Hz |

**EKF position tracking verified:**
- x increases steadily while driving forward (~2 m/s → x≈60m after 30s)
- y stays near zero (2.3e-08)
- z stays exactly 0 (two_d_mode)

**TF tree verified:**

husky/odom
├── husky/base_link (EKF, 50 Hz)
│ ├── camera_link (static)
│ ├── lidar_link (static)
│ ├── husky/base_link/imu_sensor (static)
│ └── husky/base_link/navsat_sensor (static)
└── utm (navsat_transform)


**Lidar verified:** Returns `inf` in open world (correct — nothing to hit), finite distances in obstacles world.

**GPS verified:** Reports lat ~42.41, lon ~-83.14 (Detroit Mercy campus coordinates from paper Table I).

**Smoke test commands:**
```bash
# Launch the full stack
ros2 launch titan_coordination sim_bringup.launch.py world:=phase1_region2_obstacles

# Drive the robot
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 2.0}, angular: {z: 0.0}}" --rate 10

# Check EKF output
ros2 topic echo /odometry/filtered --field pose.pose.position --once

# Check GPS coordinates
ros2 topic echo /gps/fix --once

# Check lidar hits (filter out inf)
ros2 topic echo /scan --field ranges --once | tr ',' '\n' | grep -v inf | head -10

# Check all topic rates
for t in /odom /imu/data /gps/fix /scan /odometry/filtered /odometry/gps /clock; do
  echo "=== $t ==="
  timeout 3 ros2 topic hz $t
done

# Generate TF tree diagram
ros2 run tf2_tools view_frames

# RViz2 — set Fixed Frame to husky/odom
rviz2
```

---

### File tree after Week 5

~/titan_ws/
├── .gitignore
├── configs/
│ ├── channel.yaml
│ ├── lora_params.yaml
│ └── scenarios/
├── models/
│ └── husky_leader/
│ ├── model.config
│ └── model.sdf
├── worlds/
│ ├── phase1_region2_open.sdf
│ └── phase1_region2_obstacles.sdf
├── src/
│ ├── titan_msgs/
│ │ ├── CMakeLists.txt
│ │ ├── package.xml
│ │ └── msg/
│ │ └── Heartbeat.msg
│ ├── titan_communication/
│ │ ├── package.xml
│ │ ├── setup.py
│ │ ├── setup.cfg
│ │ └── titan_communication/
│ │ ├── init.py
│ │ ├── radio/
│ │ ├── mesh/
│ │ └── nodes/
│ ├── titan_coordination/
│ │ ├── package.xml
│ │ ├── setup.py
│ │ ├── setup.cfg
│ │ ├── config/
│ │ │ ├── bridge.yaml
│ │ │ ├── ekf.yaml
│ │ │ └── navsat.yaml
│ │ ├── launch/
│ │ │ └── sim_bringup.launch.py
│ │ └── titan_coordination/
│ │ └── init.py
│ └── titan_navigation/
│ ├── package.xml
│ ├── setup.py
│ ├── setup.cfg
│ └── titan_navigation/
│ └── init.py
├── eval/
│ ├── common/
│ ├── phase1/
│ ├── phase2/
│ ├── bakeoff/
│ └── figures/
├── docs/
│ ├── environment/
│ └── thesis/
└── data/