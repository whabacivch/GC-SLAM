"""
Golden Child SLAM v2 constants only.

Legacy constants have been moved to:
  archive/legacy_common/constants_legacy.py

=============================================================================
CONVENTION QUICK REFERENCE (see docs/FRAME_AND_QUATERNION_CONVENTIONS.md)
=============================================================================

STATE VECTOR (22D):
  [trans(0:3), rot(3:6), vel(6:9), bg(9:12), ba(12:15), dt(15:16), ex(16:22)]
  Note: GC state now uses [trans, rot] ordering (SAME as se3_jax and ROS)

SE(3) POSES:
  Internal 6D: [trans(3), rotvec(3)] = [x, y, z, rx, ry, rz]
  No conversion needed - GC state and SE(3) now use same ordering!

GRAVITY:
  World: Z-UP convention, gravity points DOWN = [0, 0, -9.81] m/s²
  Accelerometer measures reaction to gravity (pointing UP when level)
  Expected accel direction (mu0) = R_body^T @ [0, 0, +1]

IMU UNITS:
  Livox raw: acceleration in g's (1g ≈ 9.81 m/s²)
  Internal: all accelerations in m/s² (scaled by GC_IMU_ACCEL_SCALE)
  Gyro: angular velocity in rad/s

EXTRINSICS:
  T_base_sensor = [tx, ty, tz, rx, ry, rz] where rotation is rotvec (radians)
  Transform: p_base = R_base_sensor @ p_sensor + t_base_sensor
=============================================================================
"""

# =============================================================================
# GOLDEN CHILD MANIFEST CONSTANTS (RuntimeManifest)
# Reference: docs/GOLDEN_CHILD_INTERFACE_SPEC.md Section 6
# These are HARD CONSTANTS - do not modify without spec change
# =============================================================================

# Chart convention
GC_CHART_ID = "GC-RIGHT-01"  # Global chart convention for all beliefs/evidence

# State dimensions
GC_D_Z = 22  # Augmented tangent dimension
GC_D_DESKEW = 22  # Deskew tangent dimension (same as D_Z)

# Fixed-cost budgets (compile-time constants)
GC_K_HYP = 4  # Number of hypotheses, always present
GC_HYP_WEIGHT_FLOOR = 0.0025  # 0.01 / K_HYP, minimum hypothesis weight
GC_B_BINS = 48  # Atlas bins (fixed)
GC_N_POINTS_CAP = 8192  # Max LiDAR points per scan per hypothesis (fixed)
# IMU preintegration: slice to integration window and pad to this length (JIT fixed size).
# At 200 Hz, 512 samples ≈ 2.5 s; covers typical scan-to-scan + deskew window.
GC_MAX_IMU_PREINT_LEN = 512

# Epsilon constants (domain stabilization)
GC_EPS_PSD = 1e-12  # Minimum eigenvalue for PSD projection
GC_EPS_LIFT = 1e-9  # Lift for SPD solves (configurable via yaml)
GC_EPS_MASS = 1e-12  # Mass regularization for InvMass
GC_EPS_R = 1e-6  # Clamp epsilon for Rbar in kappa
GC_EPS_DEN = 1e-12  # Denominator regularization in kappa
GC_EXC_EPS = 1e-12  # Domain guard for excitation ratios

# World gravity (m/s^2) in the odom/world frame used by evidence extraction.
# M3DGR uses Z-UP convention (GT Z ≈ 0.86m), so gravity points down: (0, 0, -9.81)
# NOTE: If world frame is Z-DOWN, use (0, 0, +9.81) instead.
GC_GRAVITY_W = (0.0, 0.0, -9.81)  # Z-UP convention: gravity points in -Z direction

# IMU acceleration scale factor (g → m/s² conversion)
# Livox Mid-360 IMU (ICM40609) outputs acceleration in g's, needs scaling to m/s².
# Set to 9.81 for Livox sensors, 1.0 for sensors that output m/s² directly.
GC_IMU_ACCEL_SCALE = 9.81  # g to m/s² conversion

# Trust/fusion constants
GC_ALPHA_MIN = 0.1  # Minimum fusion scale alpha
GC_ALPHA_MAX = 1.0  # Maximum fusion scale alpha
GC_KAPPA_SCALE = 1.0  # Scale for trust computation
GC_C0_COND = 1e6  # Conditioning scale for trust

# vMF κ approximation blending (continuous; no piecewise gates).
GC_KAPPA_BLEND_R0 = 0.8
GC_KAPPA_BLEND_TAU = 0.03

# Excitation coupling constants
GC_C_DT = 1.0  # Time offset coupling constant
GC_C_EX = 1.0  # Extrinsic coupling constant
GC_C_FROB = 1.0  # Frobenius strength blending constant

# Anchor drift parameters (continuous reanchoring)
GC_ANCHOR_DRIFT_M0 = 0.5  # Position drift threshold (meters)
GC_ANCHOR_DRIFT_R0 = 0.2  # Rotation drift threshold (radians)

# Smoothed initial anchor: IMU stability weights (no gates; smooth downweighting).
# w_k ∝ exp(-c_gyro ‖ω_k‖²) · exp(-c_accel (‖a_k‖ - g)²)
GC_INIT_ANCHOR_GYRO_SCALE = 0.5   # c_gyro: downweight high gyro magnitude (rad²/s²)
GC_INIT_ANCHOR_ACCEL_SCALE = 2.0  # c_accel: downweight when ‖a‖ far from g (s⁴/m²)
GRAVITY_MAG = 9.81  # m/s² for stability score

# State slice indices (0-based, per spec Section 1.1)
# NEW CONVENTION: [trans, rot] ordering (same as se3_jax and ROS)
GC_SLICE_TRANS_START = 0
GC_SLICE_TRANS_END = 3
GC_SLICE_SO3_START = 3
GC_SLICE_SO3_END = 6
GC_SLICE_VEL_START = 6
GC_SLICE_VEL_END = 9
GC_SLICE_GYRO_BIAS_START = 9
GC_SLICE_GYRO_BIAS_END = 12
GC_SLICE_ACCEL_BIAS_START = 12
GC_SLICE_ACCEL_BIAS_END = 15
GC_SLICE_TIME_OFFSET_START = 15
GC_SLICE_TIME_OFFSET_END = 16
GC_SLICE_EXTRINSIC_START = 16
GC_SLICE_EXTRINSIC_END = 22

# Soft assign temperature (for BinSoftAssign)
GC_TAU_SOFT_ASSIGN = 0.1  # Default temperature (configurable)

# Time-warp / membership kernel width as a fraction of scan duration.
# Used to avoid hard [t0,t1] boundaries in deskew and time association.
GC_TIME_WARP_SIGMA_FRAC = 0.1

# =============================================================================
# END GOLDEN CHILD MANIFEST CONSTANTS
# =============================================================================

# =============================================================================
# ADAPTIVE NOISE (Inverse-Wishart priors) — GC v2
# =============================================================================
#
# These are *priors/hyperpriors* (not fixed-tuned constants). They must not be
# inlined in operator code; always reference via `constants.py`.
#
# Units note:
# - IMU noise densities below are treated as continuous-time PSD values (per Hz)
#   in their natural units. Mapping into process diffusion Q is done by the
#   declared process model (see plan/spec).
#
# IW weak prior configuration:
# We store total ν, but choose ν so that (ν - p - 1) is a small positive
# pseudocount (fast adaptation) rather than making the IW mean undefined.
GC_IW_NU_WEAK_ADD = 0.5  # ν = p + 1 + GC_IW_NU_WEAK_ADD  (so ν - p - 1 = 0.5)

# =============================================================================
# Sensor noise hyperpriors (continuous-time; used as IW priors for Σ and as
# white-noise PSD proxies when explicitly discretized once).
# =============================================================================
#
# EXPLICIT NOISE MODEL DECLARATIONS (prevents unit bugs):
#
# Each constant declares exactly ONE of:
#   - PSD S (units /Hz, continuous time), OR
#   - per-sample variance at a declared rate
#
# And specifies exactly ONE conversion path:
#   - For integrated factors: PSD → integrated-angle covariance: S · dt_covered
#   - For rate factors: PSD → per-sample rate covariance: S / Δt
#
# =============================================================================
# Gyro Noise (ICM-40609, Livox Mid-360)
# =============================================================================
# NOISE MODEL: Continuous-time PSD
# UNITS: rad²/s (gyro rate noise PSD)
# PROVENANCE: Datasheet value (verify if 8.7e-7 is PSD or per-sample variance)
# CONVERSION PATH: For integrated factors (gyro rotation evidence):
#   Σ_rot = GC_IMU_GYRO_NOISE_DENSITY · dt_int
#   where dt_int = Σ_i Δt_i over actual IMU sample intervals
GC_IMU_GYRO_NOISE_DENSITY = 8.7e-7   # rad²/s (continuous-time PSD)

# =============================================================================
# Accel Noise (ICM-40609, Livox Mid-360)
# =============================================================================
# NOISE MODEL: Continuous-time PSD
# UNITS: m²/s³ (accel noise PSD, since (m/s²)² · s = m²/s³)
# PROVENANCE: Datasheet value (verify if 9.5e-5 is PSD or per-sample variance)
# CONVERSION PATH: For integrated factors (accel velocity evidence):
#   Σ_vel = GC_IMU_ACCEL_NOISE_DENSITY · dt_int
#   where dt_int = Σ_i Δt_i over actual IMU sample intervals
GC_IMU_ACCEL_NOISE_DENSITY = 9.5e-5  # m²/s³ (continuous-time PSD)

# =============================================================================
# LiDAR Translation Measurement Noise
# =============================================================================
# NOISE MODEL: Discrete covariance (not PSD)
# UNITS: m² (isotropic 3×3 covariance scale)
# PROVENANCE: Legacy default (adapted by IW updates from residuals)
# CONVERSION PATH: Direct use (already discrete covariance, no conversion needed)
GC_LIDAR_SIGMA_MEAS = 0.01  # m² (discrete covariance, isotropic 3×3 scale)

# Livox Mid-360 bucketization constants (Phase 3 part 2)
GC_LIDAR_N_LINES = 8
GC_LIDAR_N_TAGS = 3
GC_LIDAR_N_BUCKETS = GC_LIDAR_N_LINES * GC_LIDAR_N_TAGS  # 24

# =============================================================================
# Process diffusion-rate priors (Q is per-second; discretized exactly once as dt*Q)
# =============================================================================
#
# Each value below is a diffusion *rate* in the units of the corresponding state
# coordinate squared per second (z^2 / s), compatible with `cov += dt * Q`.
#
# State block ordering: [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)].
#
# Notes:
# - Rotation diffusion is in rad^2 / s and can be reasonably tied to gyro PSD.
# - Translation diffusion is in m^2 / s (random walk on position in this model).
# - Velocity diffusion is in (m/s)^2 / s = m^2 / s^3.
GC_PROCESS_ROT_DIFFUSION = GC_IMU_GYRO_NOISE_DENSITY  # rad^2 / s
GC_PROCESS_TRANS_DIFFUSION = 1e-4  # m^2 / s (weak prior; IW adapts from innovations)
GC_PROCESS_VEL_DIFFUSION = GC_IMU_ACCEL_NOISE_DENSITY  # m^2 / s^3
GC_PROCESS_BG_DIFFUSION = 1e-8  # (rad/s)^2 / s = rad^2 / s^3
GC_PROCESS_BA_DIFFUSION = 1e-6  # (m/s^2)^2 / s = m^2 / s^5
GC_PROCESS_DT_DIFFUSION = 1e-6  # s^2 / s = s
GC_PROCESS_EXTRINSIC_DIFFUSION = 1e-8  # (se(3))^2 / s (weak; IW adapts)

# =============================================================================
# Ornstein-Uhlenbeck (OU) damping for bounded uncertainty propagation
# =============================================================================
# OU-style mean-reverting diffusion prevents unbounded growth during missing-data gaps.
# The damping rate λ controls how quickly uncertainty saturates: Σ(∞) → Q/(2λ).
# For A = -λI, the closed-form propagation is:
#   Σ(t+Δt) = e^(-2λΔt) Σ(t) + (1 - e^(-2λΔt))/(2λ) Q
# 
# Choose λ so that for typical dt (0.1s), the correction is small, but for large
# gaps (10s+), uncertainty saturates rather than exploding.
GC_OU_DAMPING_LAMBDA = 0.1  # 1/s (damping rate; larger = faster saturation)
# At λ=0.1, saturation time constant is 1/(2λ) = 5s, so Σ(∞) = Q/(2*0.1) = 5*Q
# For dt=0.1s: e^(-2*0.1*0.1) ≈ 0.98, so nearly pure diffusion (correct for small dt)
# For dt=10s: e^(-2*0.1*10) ≈ 0.135, so strong damping (prevents explosion)

# =============================================================================
# Continuous weighting / domain safeguards (no discrete gates)
# =============================================================================
GC_WEIGHT_FLOOR = 1e-12  # strictly positive floor for continuous weights
GC_NONFINITE_SENTINEL = 1e6  # finite sentinel used by domain projection at wrapper boundaries

# Range weighting parameters (continuous; used in PointCloud2 parsing)
GC_RANGE_WEIGHT_SIGMA = 0.25
GC_RANGE_WEIGHT_MIN_R = 0.5
GC_RANGE_WEIGHT_MAX_R = 50.0

# IW retention factors (forgetful prior). Applies deterministically every scan.
GC_IW_RHO_ROT = 0.995
GC_IW_RHO_TRANS = 0.99
GC_IW_RHO_VEL = 0.95
GC_IW_RHO_BG = 0.999
GC_IW_RHO_BA = 0.999
GC_IW_RHO_DT = 0.9999
GC_IW_RHO_EX = 0.9999

# IW updates: applied every scan (no gates). We always add the sufficient statistics we have.
# No prediction at scan 0 -> no process innovation residuals -> zero process suff stats (weight 0).
# "Don't have the info yet" is equivalent to "don't have the suff stats" — so we just contribute
# what we have; no branch on "do we have enough". Process weight = min(1, scan_count); meas/lidar = 1.

# Measurement-noise retention (separate from process noise; deterministic per scan)
GC_IW_RHO_MEAS_GYRO = 0.995
GC_IW_RHO_MEAS_ACCEL = 0.995
GC_IW_RHO_MEAS_LIDAR = 0.99

# =============================================================================
# Planar Robot Constraints (Phase 1: z fix via soft prior)
# =============================================================================
# For ground-hugging robots, these constraints prevent z runaway.
# The z feedback loop (LiDAR z treated same as x,y + map feedback) causes
# drift to -50 to -80m without these constraints.

# Reference z height for M3DGR dataset (meters)
# Ground truth z is approximately 0.86m throughout the trajectory
GC_PLANAR_Z_REF = 0.86

# Soft z constraint std dev (meters)
# Smaller = stronger constraint pulling z toward z_ref
# 0.1m allows some flexibility for uneven terrain
GC_PLANAR_Z_SIGMA = 0.1

# Soft vel_z=0 constraint std dev (m/s)
# Very tight - ground robots don't fly
# 0.01 m/s = 1 cm/s vertical velocity tolerance
GC_PLANAR_VZ_SIGMA = 0.01

# Process diffusion for z coordinate (m^2/s)
# Much smaller than GC_PROCESS_TRANS_DIFFUSION to prevent z random walk
GC_PROCESS_Z_DIFFUSION = 1e-8

# =============================================================================
# Odometry Twist Constants (Phase 2: velocity factors)
# =============================================================================
# These control the strength of odometry twist (velocity) evidence.
# Wheel odometry provides strong kinematic coupling that was previously unused.

# Default velocity covariance scaling for odom twist (m/s)^2
# Typical wheel encoders have ~1-5% velocity error
GC_ODOM_TWIST_VEL_SIGMA = 0.1  # 0.1 m/s std dev

# Default yaw rate covariance from odom twist (rad/s)
# Wheel-derived yaw rate is typically accurate to ~0.01 rad/s
GC_ODOM_TWIST_WZ_SIGMA = 0.01  # rad/s std dev

# Test-only invariants still referenced by active test suite.
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95
