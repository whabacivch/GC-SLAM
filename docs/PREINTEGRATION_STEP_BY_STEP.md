# Preintegration: Every Step, Including Where Gravity Is Subtracted

This doc shows **every step** of the IMU preintegration process with exact formulas and units. The critical point: **gravity is subtracted in one place only** — when we form **a_world = a_world_nog + gravity_W**. That gives **linear acceleration** in world frame; when stationary it is **(0, 0, 0)** m/s².

**Code:** `fl_slam_poc/backend/operators/imu_preintegration.py` — `preintegrate_imu_relative_pose_jax`, inner `step` function (lines 92–108).

---

## 1. Inputs (what preintegration receives)

| Symbol | Meaning | Unit | Source |
|--------|---------|------|--------|
| `imu_stamps` | Per-sample timestamps | s | From imu_buffer (header.stamp) |
| `imu_gyro` | Angular velocity in **base frame** | rad/s | R_base_imu @ gyro_raw (on_imu) |
| `imu_accel` | **Specific force** in **base frame** | m/s² | R_base_imu @ (accel_raw × 9.81); **NOT linear accel** — reaction to gravity when stationary |
| `weights` | Smooth window weights | — | smooth_window_weights(scan_start, scan_end or t_last_scan, t_scan) |
| `rotvec_start_WB` | World→body rotation at scan start | rad | From belief (pose_pred[3:6]) |
| `gyro_bias` | Gyro bias (from state) | rad/s | belief mean_increment[9:12] |
| `accel_bias` | Accel bias (from state) | m/s² | belief mean_increment[12:15] |
| `gravity_W` | Gravity vector in world frame | m/s² | **GC_GRAVITY_W = (0, 0, -9.81)**; Z-up so gravity points **down** (-Z) |

**Important:** `imu_accel` is **specific force** (what the accelerometer measures). We **never** subtract gravity in `on_imu`; the buffer and everything passed into preintegration is **specific force**. When stationary and level, `imu_accel ≈ (0, 0, +9.81)` m/s² in base frame.

---

## 2. Per-sample loop: exact steps (one sample i)

For each IMU sample `i`, the code does the following in order. All formulas are as in the code.

### Step 1 – Time step (s)

```
dt_i = t_{i+1} - t_i   (s), last dt forced to 0
dt_eff = w_i * dt_i    (s)  — weight-scaled
```

Units: s.

---

### Step 2 – Gyro: bias correction and rotation update (no gravity)

```
omega = gyro_i - gyro_bias     (rad/s, body frame)
dR = Exp(omega * dt_eff)       (3×3, rotation)
R_next = R_k @ dR              (3×3, world→body at next time)
```

No gravity here. This only updates orientation.

---

### Step 3 – Accel: bias-corrected specific force (body frame)

```
a_body = accel_i - accel_bias   (m/s², body frame)
```

- **accel_i** = one row of `imu_accel` = **specific force** in base frame (m/s²).
- **a_body** = bias-corrected **specific force** in body frame. Still **not** linear acceleration.
- When stationary and level: **a_body ≈ (0, 0, +9.81)** m/s².

Units: m/s².

---

### Step 4 – Rotate specific force to world frame (still no gravity)

```
a_world_nog = R_k @ a_body   (m/s², world frame)
```

- **R_k** = world→body rotation at current time (updated each step by gyro). Code comment (line 134): "a_world = R_k @ a_body + gravity_W", so **a_world_nog** is the rotated specific force in world frame.
- **a_world_nog** = specific force in world frame. Still **not** linear acceleration.
- When stationary and level (R_k ≈ I, a_body ≈ (0, 0, +9.81)): **a_world_nog ≈ (0, 0, +9.81)** m/s².

Units: m/s².

---

### Step 5 – GRAVITY SUBTRACTION: form linear acceleration in world frame

```
a_world = a_world_nog + gravity_W   (m/s², world frame)
```

- **gravity_W = (0, 0, -9.81)** m/s² (Z-up: gravity points down).
- So **a_world = a_world_nog + (0, 0, -9.81)**.
- **This is the only place gravity is subtracted.** After this step, **a_world** is **linear acceleration** in world frame (what we integrate for velocity and position).
- When stationary and level: **a_world_nog ≈ (0, 0, +9.81)**, so  
  **a_world = (0, 0, +9.81) + (0, 0, -9.81) = (0, 0, 0)** m/s².

**Concrete example (stationary, first sample):**  
If **a_world_nog = (0.02, -0.09, 9.76)** m/s² (from message 5 extrinsic-applied), then  
**a_world = (0.02, -0.09, 9.76) + (0, 0, -9.81) = (0.02, -0.09, -0.05)** m/s².  
So we get **≈ 0** in z (up to sensor noise). If we did **not** add gravity_W, we would integrate **+9.76** in z and get huge z drift.

Units: m/s².

---

### Step 6 – Integrate linear acceleration → velocity

```
v_next = v_k + a_world * dt_eff   (m/s, world frame)
```

- We integrate **a_world** (linear accel), not a_world_nog. So when stationary, a_world ≈ 0 and velocity does not drift from gravity.

Units: m/s.

---

### Step 7 – Integrate velocity (+ second-order term) → position

```
p_next = p_k + v_k * dt_eff + 0.5 * a_world * (dt_eff * dt_eff)   (m, world frame)
```

- Again uses **a_world** (linear accel). So position integration does not get a constant gravity term.

Units: m.

---

### Step 8 – Accumulate for weighted means (for diagnostics / IW)

```
sum_wdt += dt_eff
sum_a_body += a_body * dt_eff
sum_a_world_nog += a_world_nog * dt_eff
sum_a_world += a_world * dt_eff
```

- **a_world_mean = sum_a_world / sum_wdt** → when stationary, **a_world_mean ≈ (0, 0, 0)** m/s² (linear accel).
- **a_world_nog_mean** stays ≈ (0, 0, +9.81) m/s² (specific force in world).

---

## 3. After the scan: relative pose and body-frame outputs

- **delta_R** = R_start^T @ R_end (relative rotation, start body frame).
- **p_body_frame** = R_start^T @ p_end (world displacement transformed to start body frame).
- **v_body_frame** = R_start^T @ v_end (velocity change in start body frame).
- **delta_pose** = [p_body_frame, rotvec_delta] (6,) for SE(3) relative pose.

So **delta_p_int** and **delta_v_int** passed to the preintegration factor are in **start body frame**; they come from integrating **a_world** (linear accel) in world frame and then transforming. So **gravity has been subtracted** before integration; the z component of delta_p and delta_v should not contain a constant gravity drift when stationary.

---

## 4. Summary: where gravity is subtracted

| Step | Formula | What we get | When stationary, level |
|------|---------|-------------|--------------------------|
| 3 | a_body = accel_i - accel_bias | Specific force (body) | ≈ (0, 0, +9.81) m/s² |
| 4 | a_world_nog = R_k @ a_body | Specific force (world) | ≈ (0, 0, +9.81) m/s² |
| **5** | **a_world = a_world_nog + gravity_W** | **Linear accel (world)** | **≈ (0, 0, 0) m/s²** |
| 6 | v_next = v_k + a_world * dt_eff | Velocity (world) | no gravity drift |
| 7 | p_next = p_k + v_k*dt_eff + 0.5*a_world*dt_eff² | Position (world) | no gravity drift |

**Gravity is subtracted in Step 5 only:** we add **gravity_W = (0, 0, -9.81)** to **a_world_nog**, giving **a_world** = linear acceleration. Everything we integrate (velocity and position) uses **a_world**, so we do **not** get a huge z value from gravity in the preintegration output. If you are seeing a huge z in the **estimated trajectory**, that z is coming from **LiDAR translation evidence** and map–scan feedback (see `archive/docs/TRACE_Z_EVIDENCE_AND_TRAJECTORY.md` for M3DGR-era analysis), not from forgetting to subtract gravity in preintegration. The **buffered IMU** (and any CSV of “extrinsic applied” IMU) still holds **specific force** (reaction to gravity), so those will show **≈ +9.8** in z until you subtract gravity (e.g. with `--linear` in `apply_imu_extrinsic_to_csv.py`).
