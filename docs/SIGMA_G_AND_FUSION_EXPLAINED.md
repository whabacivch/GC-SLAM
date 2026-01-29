# Sigma_g Calculation and Fusion Setup Explained

## 1. What is Sigma_g?

`Sigma_g` is the **gyro measurement noise covariance** (3×3 matrix) representing the uncertainty in gyro rate measurements. It's treated as a **continuous-time power spectral density (PSD)** with units `rad²/s`.

### 1.1 Initialization

**From constants:**
```python
GC_IMU_GYRO_NOISE_DENSITY = 8.7e-7  # rad²/s (continuous-time PSD)
```

**Initial IW state:**
- `Psi_gyro = GC_IMU_GYRO_NOISE_DENSITY * I₃ * nu_extra`
- `nu = p + 1 + nu_extra` where `p=3` (dimension) and `nu_extra = GC_IW_NU_WEAK_ADD`

### 1.2 Inverse-Wishart (IW) Adaptation

`Sigma_g` is stored as an **Inverse-Wishart random variable** `Σ ~ InvWishart(Psi, nu)` and is **adapted online** from measurement residuals.

**IW State Structure:**
```python
MeasurementNoiseIWState:
  - nu: (3,) degrees of freedom per sensor [gyro, accel, lidar]
  - Psi_blocks: (3, 3, 3) scale matrices per sensor
  - block_dims: (3,) dimensions [3, 3, 3]
```

**Posterior Mean (plug-in estimate):**
```python
Sigma_g = Psi_gyro / (nu_gyro - p - 1)
```
where `p=3` is the dimension of the gyro block.

**Adaptation from residuals:**
```python
# Per scan, compute residuals:
r_g = (gyro_i - bg) - omega_avg  # (M, 3)

# Build sufficient statistics:
dPsi_gyro = Σ_m w_m * r_m r_m^T * dt_imu  # Map discrete variance → PSD
dnu_gyro = 1.0

# Update IW state (forgetful retention):
Psi_gyro <- rho * Psi_gyro + dPsi_gyro
nu_gyro <- rho * nu_gyro + dnu_gyro
```

**Key point:** The residual outer product is **multiplied by `dt_imu`** to convert discrete-time variance to continuous-time PSD (line 154 in `measurement_noise_iw_jax.py`).

### 1.3 Usage in Evidence

**In `imu_gyro_rotation_evidence`:**
```python
# Discretize continuous-time PSD to discrete covariance:
Sigma_rot = Sigma_g * dt_int  # (3, 3) rad²; dt_int = Σ_i Δt_i (IMU sample intervals in window)

# Convert to information form:
L_rot = Sigma_rot^-1  # (3, 3) information matrix
```
**Note:** The code uses **dt_int** (sum of IMU sample intervals in the scan-to-scan window), not dt_scan (LiDAR scan duration). See `pipeline.py` and `imu_gyro_evidence.py`.

**Problem:** If `Sigma_g` is too small (from IW adaptation), then `L_rot` becomes huge, dominating all other evidence.

---

## 2. Fusion Setup

### 2.1 Evidence Extraction

**Evidence sources per scan** (GC state ordering: **trans 0:3, rot 3:6**):

1. **LiDAR Evidence** (`matrix_fisher_rotation_evidence` + `planar_translation_evidence`):
   - Rotation: Matrix Fisher → **Fills: `L[3:6, 3:6]`**
   - Translation: planarized WLS → **Fills: `L[0:3, 0:3]`** (self‑adaptive z precision)

2. **Odometry Evidence** (`odom_quadratic_evidence`):
   - SE(3) pose factor: `T_err = T_pred^{-1} ∘ T_odom`
   - Fills: `L[0:6, 0:6]` (full pose: trans 0:3, rot 3:6)

3. **Odom Twist Evidence** (`odom_velocity_evidence`, `odom_yawrate_evidence`, `pose_twist_kinematic_consistency`):
   - Velocity factor (body twist → world velocity), yaw‑rate factor, and pose–twist consistency across scan dt

4. **Planar Priors** (`planar_z_prior`, `velocity_z_prior`):
   - Soft constraints on **z ≈ z_ref** and **v_z ≈ 0**

5. **IMU Accel Evidence** (`imu_vmf_gravity_evidence_time_resolved`):
   - vMF directional likelihood on gravity direction (Laplace at δθ=0), **time‑resolved and reliability‑weighted**
   - Fills: `L[3:6, 3:6]` (rotation block only)

6. **IMU Gyro Evidence** (`imu_gyro_rotation_evidence`):
   - Gaussian on preintegrated rotation residual
   - Fills: `L[3:6, 3:6]` (rotation block only)

7. **IMU Preintegration Evidence** (`imu_preintegration_factor`):
   - Gaussian on velocity/position from accel preintegration vs predicted state
   - Fills the relevant tangent block (velocity/position components)

### 2.2 Evidence Combination

**Additive combination** (`pipeline.py`: evidence sum):
```python
L_evidence = L_lidar + L_odom + L_imu + L_gyro + L_imu_preint + L_planar + L_vel + L_wz + L_consistency
h_evidence = h_lidar + h_odom + h_imu + h_gyro + h_imu_preint + h_planar + h_vel + h_wz + h_consistency
```

**Note:** This happens **before** excitation scaling and fusion scaling. All evidence is combined with equal weight in the sum.

### 2.3 Excitation Prior Scaling

**Fisher-derived excitation scaling** (`operators/excitation.py`):
- Scales down prior strength on **dt (index 15)** and **extrinsics (indices 16:22)** when evidence is strong
- `s_dt = e_dt / (e_dt + pi_dt + ε)` where `e_dt = L_evidence[15, 15]`, `pi_dt = L_prior[15, 15]`
- `s_ex = e_ex / (e_ex + pi_ex + ε)` where `e_ex = trace(L_evidence[16:22, 16:22])`, `pi_ex = trace(L_prior[16:22, 16:22])`
- Prior rows/cols and h entries for dt (15) and extrinsic (16:22) are scaled by `(1 - s_dt)` and `(1 - s_ex)`

### 2.4 Fusion Scale Computation

**From certificates (`fusion_scale_from_certificates`):**

The pipeline **overwrites** the combined evidence certificate’s conditioning with the **6×6 pose block** of `L_evidence` (not the full 22×22) before calling `fusion_scale_from_certificates`. So alpha is driven by pose-relevant conditioning, not by null directions (e.g. bias/extrinsic blocks).

```python
# Conditioning used: 6×6 pose block L_evidence[0:6, 0:6] → cond_pose6 = eig_max / eig_min
# Extract quality metrics from (possibly overwritten) cert:
cond_evidence = cert_evidence.conditioning.cond
ess_evidence = cert_evidence.support.ess_total

# Quality metrics:
cond_quality = c0_cond / (cond_evidence + c0_cond)  # Lower cond = better
support_quality = ess_evidence / (ess_evidence + 1.0)  # Higher ESS = better

# Combined quality (geometric mean):
quality = sqrt(cond_quality * support_quality)

# Map to alpha range:
alpha = alpha_min + (alpha_max - alpha_min) * quality  # then clamped
```

**Mitigation:** Using pose-6 conditioning (instead of full 22×22) avoids alpha being pinned at minimum when gyro or other blocks dominate the full matrix condition number.

### 2.5 Additive Fusion

**Information fusion (`info_fusion_additive`):**
```python
L_post = L_pred + alpha * L_evidence
h_post = h_pred + alpha * h_evidence
```

**Always applies PSD projection** to ensure `L_post` is positive semidefinite.

### 2.6 Certificate Aggregation

**When combining evidence certificates** (`common/certificates.py:aggregate_certificates`):
```python
combined_evidence_cert = aggregate_certificates([
    evidence_cert,  # LiDAR
    odom_cert,      # Odometry
    imu_cert,       # IMU accel
    gyro_cert       # IMU gyro
])
```

**Aggregation rules:**
- **Conditioning:** `cond = max(cond_i)` (worst case); **eig_min/eig_max** = min/max across certs
- **Support:** `ess_total = mean(ess_total_i)`, `support_frac = mean(support_frac_i)`
- **Triggers:** Union of all triggers
- **Frobenius:** `any(frobenius_i)`

**Note:** For fusion scale α, the pipeline then **overwrites** `combined_evidence_cert.conditioning` with the 6×6 pose block conditioning before calling `fusion_scale_from_certificates`, so gyro-dominated full-matrix conditioning no longer pins alpha at minimum.

---

## 3. Current Issues

### 3.1 Gyro Evidence Dominance

**Symptoms:**
- Gyro evidence trace: ~2.5e9 (dominates 100% of total evidence)
- Other evidence traces: LiDAR ~1e5, Odom ~2e3, IMU ~1-10

**Root cause:**
- `Sigma_g` from IW state may be too small (over-adapted)
- When `Sigma_g` is small, `L_rot = (Sigma_g * dt)^-1` becomes huge
- This makes gyro evidence dominate all other sources

**Expected behavior:**
- If `Sigma_g = 8.7e-7 rad²/s` and `dt = 1.0s`:
  - `Sigma_rot = 8.7e-7 rad²`
  - `L_rot trace ≈ 3 / 8.7e-7 ≈ 3.4e6` (not 2.5e9!)

**Actual:**
- `trace(Sigma_g) ≈ 2.6e-6` initially, but adapts to `7.6e-5` later
- If `Sigma_g` per-axis is `2.5e-5 rad²/s` and `dt = 1.0s`:
  - `Sigma_rot per-axis = 2.5e-5 rad²`
  - `L_rot per-axis = 1 / 2.5e-5 = 4e4`
  - `L_rot trace ≈ 3 * 4e4 = 1.2e5` (still not 2.5e9!)

**Conclusion:** There's likely a **units mismatch** or **double-scaling** issue. The gyro evidence is being computed incorrectly.

### 3.2 Fusion Alpha Stuck at Minimum

**Symptoms:**
- `alpha = 0.1` (minimum) for all scans
- Evidence is heavily down-weighted

**Root cause:**
- `cond_evidence` is huge (from gyro dominance)
- `cond_quality = c0_cond / (cond_evidence + c0_cond) → 0`
- `quality → 0`, so `alpha → alpha_min`

**Fix needed:**
- Normalize evidence matrices before combining
- Or cap individual evidence contributions
- Or fix gyro evidence scaling to prevent dominance

### 3.3 Missing State Constraints

**Current evidence coverage** (GC ordering: **trans 0:3, rot 3:6**, then vel, biases, dt, extrinsic):
- **Translation (0:3):** ✅ LiDAR, Odom
- **Rotation (3:6):** ✅ LiDAR, Odom, IMU accel, IMU gyro
- **Velocity (6:9):** ⚠️ Constrained by IMU preintegration factor (velocity/position from accel)
- **Gyro bias (9:12):** ❌ **NOT directly constrained** (estimated via state; IW updates use residuals)
- **Accel bias (12:15):** ❌ **NOT directly constrained**
- **Time offset (15):** ⚠️ Only weakly via LiDAR excitation scaling
- **Extrinsics (16:22):** ⚠️ Only weakly via LiDAR excitation scaling

**Problem:** Biases drift without direct measurement constraints; velocity is partially constrained by preintegration.

---

## 4. Recommendations

### 4.1 Fix Gyro Evidence Scaling

**Check:**
1. Verify `Sigma_g` units are correct (should be `rad²/s`)
2. Verify `dt_scan` is correct (should be IMU integration time, not LiDAR scan duration) ✅ Already fixed
3. Check if there's a double-scaling issue (e.g., `Sigma_g` already includes `dt`)

**Potential fix:**
- Cap `L_gyro` trace to prevent dominance
- Or normalize evidence matrices before combining
- Or fix IW adaptation to prevent `Sigma_g` from becoming too small

### 4.2 Fix Fusion Alpha Computation

**Options:**
1. Normalize evidence matrices by trace before combining
2. Use per-evidence-source fusion scales (not just one global alpha)
3. Cap condition numbers in certificate aggregation
4. Use relative evidence strength (not absolute) for quality metrics

### 4.3 Add Missing Evidence

**Extend IMU evidence to constrain:**
- **Velocity:** From IMU preintegration (already computed, just need to add evidence)
- **Biases:** From IMU residuals (already computed in IW updates, need to add evidence)

---

## 5. Code Locations

**Sigma_g calculation:**
- Initialization: `structures/measurement_noise_iw_jax.py:create_datasheet_measurement_noise_state()`
- IW mean: `operators/measurement_noise_iw_jax.py:measurement_noise_mean_jax()`
- IW updates: `operators/measurement_noise_iw_jax.py:imu_gyro_meas_iw_suffstats_from_avg_rate_jax()`
- Usage: `operators/imu_gyro_evidence.py:imu_gyro_rotation_evidence()`

**Fusion setup:**
- Evidence combination: `backend/pipeline.py` (L_evidence / h_evidence sum, including L_imu_preint)
- Excitation scaling: `operators/excitation.py:compute_excitation_scales_jax`, `apply_excitation_prior_scaling_jax`
- Fusion scale: `operators/fusion.py:fusion_scale_from_certificates()` (conditioning from pose 6×6 in pipeline)
- Additive fusion: `operators/fusion.py:info_fusion_additive()`
- Certificate aggregation: `common/certificates.py:aggregate_certificates()`
- Pipeline reference: `docs/IMU_BELIEF_MAP_AND_FUSION.md`