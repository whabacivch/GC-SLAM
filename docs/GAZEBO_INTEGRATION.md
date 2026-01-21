# Gazebo Integration (Impact Project_v1)

This project is designed to run against **real or simulator-provided sensors** (Gazebo recommended). The SLAM core stays invariant-first: the simulator is only a *data source*.

## System snapshot (operational)

**Component summary**
- `sim_world_node`: ground truth + noisy odom (`/sim/ground_truth`, `/odom`).
- `tb3_odom_bridge_node`: absolute → delta odom (`/odom` → `/sim/odom`).
- `frontend_node`: association + ICP + anchor/loop creation (`/scan`, `/camera/*`, `/odom` → `/sim/*`).
- `fl_backend_node`: inference + loop fusion (`/sim/odom`, `/sim/loop_factor`, `/sim/anchor_create` → `/cdwm/*`).

**Key data flow**
Sensors → Frontend (association + ICP) → LoopFactor → Backend (fusion) → State/Trajectory  
Ground Truth Odom → Odom Bridge (abs→delta) ───────────────────────────────┘

## What “working with Gazebo” means here

- **Frontend** consumes `/scan`, `/odom`, and optionally `/camera/*` to produce `/sim/loop_factor` and `/sim/anchor_create`.
- **Odom bridge** converts Gazebo/TB3 absolute `/odom` into delta odom on `/sim/odom` for the backend.
- **Backend** consumes `/sim/odom` + loop/anchor topics and publishes `/cdwm/state`, `/cdwm/trajectory`, markers, and status.

You can tell what’s actually connected/working via:
- `/cdwm/frontend_status` (JSON sensor connectivity + “slam_operational”)
- `/cdwm/backend_status` (JSON odom/loop/anchor activity)

## Prerequisites (local machine / container)

- ROS 2 Jazzy
- `turtlebot3_gazebo` (if using the built-in TurtleBot3 world launch)
- Gazebo (Harmonic, per Jazzy defaults)

If you use camera/depth topics via `cv_bridge`, keep NumPy compatible (common ROS constraint):
- Prefer `numpy<2.0` in the runtime environment when `cv_bridge` is present.

## Launch: TurtleBot3 Gazebo (recommended first step)

From `Impact Project_v1/fl_ws` (after building and sourcing):

```bash
# Phase 2 note: Gazebo launch files are stored under `phase2/` and are not installed by the MVP package by default.
# See: `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3.launch.py`
```

Optional toggles:
- `enable_frontend:=false` to run backend dead-reckoning only
- `enable_foxglove:=true` for Foxglove (if installed)

## Launch: external Gazebo (bring your own world)

Run Gazebo separately, then launch only the SLAM nodes:

```bash
# Phase 2 note: see `phase2/fl_ws/src/fl_slam_poc/launch/poc_tb3.launch.py`
```

## Topic crosswalk (Gazebo → FL-SLAM)

Frontend parameters (defaults shown):
- `scan_topic` → `/scan`
- `odom_topic` → `/odom` (absolute or delta depends on `odom_is_delta`)
- `camera_topic` → `/camera/image_raw`
- `depth_topic` → `/camera/depth/image_raw`
- `camera_info_topic` → `/camera/depth/camera_info`

Odom bridge parameters (defaults shown):
- `input_topic` → `/odom`
- `output_topic` → `/sim/odom`

Backend subscriptions (currently fixed):
- `/sim/odom`, `/sim/loop_factor`, `/sim/anchor_create`

## Debug checklist (fast)

1. Confirm Gazebo is publishing sensors:
   - `ros2 topic list | rg \"^/scan$|^/odom$|^/camera\"`
2. Confirm frontend sees sensors:
   - `ros2 topic echo /cdwm/frontend_status`
3. Confirm backend is receiving delta odom:
   - `ros2 topic echo /sim/odom`
4. Confirm loop factors appear (requires sufficient sensing):
   - `ros2 topic echo /sim/loop_factor`

## Common gotchas

- If **depth topics are missing**, the frontend will run but will report SLAM as not operational and will skip anchor births (see `/cdwm/frontend_status`).
- If timestamps look inconsistent, set `use_sim_time:=true` for all nodes in the launch file (see Phase 2 `poc_tb3.launch.py`).
