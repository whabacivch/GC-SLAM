# Raw Measurements vs Pipeline Usage

This doc captures what the SLAM pipeline **actually uses** from odom and IMU versus what is **available** in the raw messages and what **additional insight** we could derive. The raw dumps and dead-reckon scripts (`docs/raw_sensor_dump/`, `tools/dead_reckon_odom_dump.py`, `tools/compare_accel_odom.py`) show there is a lot of information we are not yet exploiting.

## Odom (nav_msgs/Odometry)

### Available in the message
| Field | Frame / meaning | Used in pipeline? |
|-------|------------------|-------------------|
| **Pose** (position x,y,z, orientation quat) | Parent frame (e.g. odom_combined); T_{parent←child} | **Yes** — pose relative to first odom, covariance → Gaussian evidence (odom_evidence) |
| **Pose covariance** (6×6) | [x,y,z,roll,pitch,yaw] | **Yes** — inverse as information for pose evidence |
| **Twist** (vx, vy, vz, wx, wy, wz) | **Child/body frame** (e.g. base_footprint) | **Yes** — velocity factor + yaw-rate factor + pose–twist consistency |

So: we use **pose + twist**. Twist is fused as:
- **Body velocity factor** (vx, vy, vz) into the velocity block
- **Yaw‑rate factor** (wz) against gyro‑derived yaw rate
- **Pose–twist kinematic consistency** across the scan dt

### Implications
- **Lateral (y)**: Our odom evidence includes **pose + twist**. We still do not explicitly separate “forward motion” vs “lateral motion” beyond the twist factors, nor do we model a dedicated cross‑sensor lateral consistency likelihood.
- **Consistency checks**: We now include yaw‑rate evidence (odom wz vs gyro‑derived yaw rate), but broader cross‑sensor consistency checks (e.g., integrated Δyaw vs odom Δyaw vs LiDAR Δyaw as an explicit likelihood) are still **diagnostics only**.

---

## IMU (sensor_msgs/Imu)

### Available
| Field | Meaning | Used in pipeline? |
|-------|---------|-------------------|
| **Angular velocity** (gyro x,y,z) | rad/s in sensor frame | **Yes** — preintegration, gyro evidence, IW suff stats (after R_base_imu to base) |
| **Linear acceleration** (accel x,y,z) | m/s² in sensor frame | **Yes** — preintegration (with gravity), vMF gravity evidence, IW suff stats |
| **Orientation** (if present) | Often not filled in raw IMU | N/A |

### How we use it
- **Gyro**: Integrated over scan-to-scan window → delta rotation → gyro evidence (rotation block) and preintegration (position/velocity delta).
- **Accel**: (1) vMF on gravity direction (attitude); (2) preintegration with gravity cancellation → delta position/velocity.
- We do **not** use: raw accel/gyro for explicit “lateral vs forward” decomposition, or a full multi‑sensor Δyaw consistency likelihood (gyro vs odom vs LiDAR) beyond diagnostics.

### Implications
- **Lateral (y)**: IMU lateral accel (ay in body) double‑integrates to large y if there is bias; odom **does** provide vx/vy, which now constrains velocity, but we still do not explicitly decompose “forward vs lateral” motion or model cross‑sensor lateral consistency.
- **Frame / bias**: compare_accel_odom showed accel (gravity direction) agrees with odom orientation (dot ~0.99). We don’t feed that agreement or its failure back into the model (e.g. as a soft constraint or a diagnostic that changes weighting).

---

## What we could derive for more insight (not done today)

1. **Odom twist in the loop**  
   Use vx, vy, wz (and optionally vz, wx, wy) to:
   - Form a velocity / yaw-rate prior or observation (body frame), so we’re not only “pose snapshot” but “pose + twist”.
   - Cross-check with IMU: e.g. preintegrated delta vs odom pose change; gyro ∫ vs odom Δyaw; scale consistency.

2. **Forward vs lateral explicitly**  
   - From odom: vx = forward speed, wz = yaw rate → we can interpret pose evolution as “forward + turn” (as in the dead-reckon script).  
   - From IMU: ax_b, ay_b in body → forward vs lateral acceleration.  
   - We could fuse “forward motion from odom” with “forward/lateral from IMU” in a structured way (e.g. separate forward vs lateral observation models) instead of only fusing full pose + full IMU evidence.

3. **Consistency signals**  
   - Gravity: accel vs odom orientation (we checked offline; could be a running consistency score or outlier downweight).  
   - Yaw: dyaw_gyro vs dyaw_odom vs dyaw_wahba (we log them; could feed into fusion scale or IW weighting).  
   - Lateral: if IMU-only lateral (y) is huge vs odom lateral (y), that could indicate gyro/accel bias and e.g. reduce IMU weight or trigger a bias update.

4. **Raw dumps and scripts**  
   The tools we added (dump_raw_imu_odom, dead_reckon_odom_dump, compare_accel_odom, plot_wz_odom) show that:
   - There is a lot of information in the raw measurements.
   - Simple dead reckoning (odom vx+wz vs odom vx+quat vs IMU-only) and gravity-direction comparison already reveal frame semantics, lateral drift, and agreement/disagreement.
   - The pipeline today does not mirror that level of decomposition (twist, forward/lateral, cross-checks); it uses a subset of the fields and may not be using them in the most informative way.

---

## LiDAR (sensor_msgs/PointCloud2, Livox CustomMsg → PointCloud2)

### Available in the message (Livox converter output)

| Field | Source | Used in pipeline? |
|-------|--------|-------------------|
| **x, y, z** | CustomPoint (float32) | **Yes** — points transformed to base, then deskewed, binned, evidence |
| **intensity** | reflectivity (uint8) → PointCloud2 "intensity" | **No** — parser only includes it in dtype for stride; never returned or used as weight/feature |
| **ring** | line (uint8) | **Yes** — bucket index (ring,tag), point_lambda, IW per-bucket |
| **tag** | tag (uint8) | **Yes** — bucket index, point_lambda |
| **timebase_low, timebase_high** | CustomMsg timebase (uint64) | **Yes** — per-point timestamps (timebase_sec + time_offset) |
| **time_offset** | per-point (uint32 ns) | **Yes** — per-point time when present; Livox MID-360 often has all zeros (rosette) |
| **Per-point covariance** | — | **Not in message** — PointCloud2 / Livox CustomPoint have no covariance field |

### What we use

- **Geometry**: x, y, z → base frame via T_base_lidar → deskew (constant twist) → binning → centroid/direction stats → LiDAR evidence (translation block).
- **Time**: timebase + time_offset → per-point timestamps; scan bounds from timebase_sec and header.stamp; dt_scan for IMU window.
- **Weights**: **Range-based sigmoid only** (GC_RANGE_WEIGHT_*). Not intensity.
- **Metadata**: ring, tag → bucket index for per-(line,tag) IW adaptation and point_lambda (reliability).

### What we leave on the table

- **Intensity (reflectivity)**: We have it in the PointCloud2 (converter writes it). The backend parser reads it only so the struct layout is correct; we **never return it** and **never use it** for weighting or filtering. So stronger/weaker returns could inform point quality or outlier downweighting — we do not use that.
- **Per-point uncertainty**: The message carries **no per-point covariance**. We pass **zeros** for `point_covariances` into binning (`deskewed_covs = zeros` in pipeline). Binning then computes centroid covariance as scatter + weighted average of point covariances; with zeros, we get scatter (geometry) only. So we do not model per-point range/angle uncertainty from the sensor; we rely on **global** Sigma_meas and per-bucket IW (LiDAR bucket noise), not point-wise covariance from the raw message (because there is none).

### Implications

- **Intensity**: All points are weighted by range and by point_lambda (bucket reliability), not by reflectivity. Low-reflectivity or saturated returns are treated the same as high-reflectivity once in the same range band.
- **Covariance**: LiDAR evidence uses centroid covariances that are **scatter + 0** (no per-point cov) plus a global/per-bucket measurement noise (IW). So we are not "leaving on the table" a per-point covariance from the message — the format does not provide it. We are leaving on the table **intensity** as a potential quality/weight signal.

---

## Raw covariance: what it is and whether it is fixed or changes

### Odom (nav_msgs/Odometry)

- **What it is**: `msg.pose.covariance` — 36 floats, row-major 6×6 for [x, y, z, roll, pitch, yaw]. This is the **pose** covariance in the odom (or parent) frame.
- **Used?** **Yes.** We read it every message in `on_odom`, reshape to (6,6), and pass it to `odom_evidence` as the measurement covariance (inverse → information).
- **Fixed or changes?** **Per-message.** We use whatever the bag publishes each time. Whether that is **constant** (same 36 values every message) or **time-varying** depends on the bag and the odom source. We do not assume either; we have not historically inspected the bag to report "fixed" vs "varies". Use `tools/inspect_odom_covariance.py` on your bag to see if pose covariance is identical for all sampled messages or not.

### IMU (sensor_msgs/Imu)

- **What it is**: `orientation_covariance` (9), `angular_velocity_covariance` (9), `linear_acceleration_covariance` (9) — each 3×3 row-major. Often -1 for "unknown".
- **Used?** **No.** The normalizer forwards them; the **backend never uses them**. We use **IW-adapted** Sigma_g (gyro) and Sigma_a (accel) from measurement_noise_state, not the message covariances.
- **Fixed or changes?** Irrelevant for the current pipeline — we ignore them. If we ever used them, we would read them per message (they could be fixed or varying depending on driver).

### LiDAR (PointCloud2 / Livox)

- **What it is**: **No covariance in the message.** CustomPoint has x, y, z, reflectivity, tag, line; no per-point covariance. PointCloud2 has no standard covariance field.
- **Used?** N/A. We pass **zeros** for per-point covariances into binning. Measurement noise is **adaptive**: Sigma_meas from IW (index 2) and per-bucket LiDAR IW, not from the raw message.

### Summary table

| Sensor | Raw covariance | Pipeline uses it? | Fixed or changes in bag? |
|--------|----------------|-------------------|---------------------------|
| Odom   | 6×6 pose       | Yes (every message) | Unknown until inspected; use `inspect_odom_covariance.py` |
| IMU    | 3×3 orient, gyro, accel | No (we use IW) | Not used |
| LiDAR  | None           | N/A (we use zeros + IW Sigma_meas) | N/A |

---

## Summary

- **Odom**: Pipeline uses **pose + covariance** only. **Twist (vx, vy, vz, wx, wy, wz) is never used.** So we are not using “how fast and in which direction” from odom, only “where the robot is.” Raw pose covariance is read per message; use `tools/inspect_odom_covariance.py` to see if it is fixed or varies in your bag.
- **IMU**: We use gyro and accel for preintegration, gravity (vMF), and gyro evidence. We do **not** use IMU message covariances (we use IW). We do not explicitly use “forward vs lateral” or full cross‑sensor consistency beyond the yaw‑rate factor.
- **LiDAR**: We use x, y, z, timebase, time_offset, ring, tag, and **range-based weights** (not intensity). We do **not** use intensity for weighting or features. We pass **zeros** for per-point covariance (message has none); measurement noise is IW-adaptive.
- **Overall**: The raw measurements (and the dumps/scripts) show there is more structure (twist, body frame, lateral vs forward, intensity, consistency between sensors) that could be used for better fusion and insight; the current pipeline uses only part of the available information and may not be using it in the most informative way.
