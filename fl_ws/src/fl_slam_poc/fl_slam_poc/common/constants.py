"""
Geometric Compositional SLAM v2 constants only.

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
# GEOMETRIC COMPOSITIONAL MANIFEST CONSTANTS (RuntimeManifest)
# Reference: docs/GC_SLAM.md Section 6
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
# Kimera (and GC) use Z-UP convention; gravity points down: (0, 0, -9.81).
# NOTE: If world frame is Z-DOWN, use (0, 0, +9.81) instead.
GC_GRAVITY_W = (0.0, 0.0, -9.81)  # Z-UP convention: gravity points in -Z direction

# IMU acceleration scale factor (g → m/s² conversion)
# Livox Mid-360 IMU (ICM40609) outputs acceleration in g's, needs scaling to m/s².
# Set to 9.81 for Livox sensors, 1.0 for sensors that output m/s² directly.
GC_IMU_ACCEL_SCALE = 9.81  # g to m/s² conversion

# Trust/fusion constants
# Full strength: do not scale evidence down; use alpha = 1.0 so L_post = L_pred + L_evidence.
GC_ALPHA_MIN = 1.0  # Minimum fusion scale alpha (1.0 = full trust)
GC_ALPHA_MAX = 1.0  # Maximum fusion scale alpha (1.0 = full trust)
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


# Time-warp / membership kernel width as a fraction of scan duration.
# Used to avoid hard [t0,t1] boundaries in deskew and time association.
GC_TIME_WARP_SIGMA_FRAC = 0.1

# =============================================================================
# END GEOMETRIC COMPOSITIONAL MANIFEST CONSTANTS
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

# Reference z height in the GC "body/base" frame (meters).
#
# IMPORTANT (frame contract): state/body frame is configurable (Kimera: acl_jackal2/base).
# World Z of the base origin should be ~0 (ground contact). See docs/FRAME_AND_QUATERNION_CONVENTIONS.md.
GC_PLANAR_Z_REF = 0.0

# Odom z variance prior (m^2): minimum variance for odom pose z component.
# Caps trust in odom z so planar robots with bad/unobserved z don't pollute state.
# Large value (1e6 = sigma_z >= 1000m) effectively means "don't trust odom z".
# Smaller values allow using actual odom z with appropriate uncertainty.
GC_ODOM_Z_VARIANCE_PRIOR = 1e6

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

# =============================================================================
# PRIMITIVE MAP + OT CONSTANTS (Visual-LiDAR Integration)
# Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md
# =============================================================================
#
# These are REQUIRED config fields - no silent defaults. The pipeline will
# fail-fast if they are not explicitly set in config when needed.
#
# Budget names (fixed-cost, branch-free per scan):
#   N_FEAT      - Camera feature/splat budget per scan
#   N_SURFEL    - LiDAR surfel budget per scan
#   K_ASSOC     - Fixed candidate neighborhood size for OT association
#   K_SINKHORN  - Fixed Sinkhorn iteration count (no convergence check)
#   RINGBUF_LEN - Camera frame ring buffer length for soft time association

# Feature extraction budget (camera splats)
GC_N_FEAT = 512  # Fixed camera feature count per scan

# Surfel extraction budget (LiDAR surfels)
GC_N_SURFEL = 1024  # Fixed LiDAR surfel count per scan

# OT association budgets
GC_K_ASSOC = 8  # Fixed candidate neighborhood per measurement primitive
GC_K_SINKHORN = 50  # Fixed Sinkhorn iterations (no convergence check)

# Camera frame ring buffer for soft time association
GC_RINGBUF_LEN = 5  # Number of camera frames to buffer

# =============================================================================
# POSE EVIDENCE BACKEND CONFIGURATION
# =============================================================================
# Single-path enforcement: exactly one pose evidence path active at runtime.
# Selection is explicit via config (no fallback, no silent coexistence).
#
# Valid value:
#   "primitives" - Primitive alignment pose evidence (visual + LiDAR)
GC_POSE_EVIDENCE_BACKEND_PRIMITIVES = "primitives"

# =============================================================================
# MAP BACKEND CONFIGURATION
# =============================================================================
# Single-path enforcement: exactly one map representation at runtime.
#
# Valid value:
#   "primitive_map"  - PrimitiveMap (Gaussian x vMF atlas with stable IDs)
GC_MAP_BACKEND_PRIMITIVE_MAP = "primitive_map"

# =============================================================================
# PRIMITIVE MAP PARAMETERS
# =============================================================================
# PrimitiveMap: persistent atlas of probabilistic primitives
# Each primitive has:
#   - Geometry: Gaussian in info form (Lambda, theta) in 3D
#   - Orientation: vMF natural parameter eta (resultant or B=3)
#   - Stable ID for temporal tracking
#   - Optional: color/descriptor payload

# Maximum primitives in the map (fixed budget for memory)
GC_PRIMITIVE_MAP_MAX_SIZE = 50000

# Forgetting factor for primitive weights (continuous, applied every scan)
GC_PRIMITIVE_FORGETTING_FACTOR = 0.995

# Merge threshold: primitives with Bhattacharyya distance below this are merged
GC_PRIMITIVE_MERGE_THRESHOLD = 0.1

# Cull threshold: primitives with weight below this are removed
GC_PRIMITIVE_CULL_WEIGHT_THRESHOLD = 1e-4

# vMF concentration clamp (prevents numerical issues)
GC_PRIMITIVE_KAPPA_MIN = 1e-3
GC_PRIMITIVE_KAPPA_MAX = 1e4

# vMF appearance/orientation lobes (multi-lobe mixture, fixed budget)
# Contract: vMF is not optional. Use B=3 lobes (default) for appearance/direction modeling.
# When a producer cannot populate multiple lobes yet, it must still populate a meaningful
# resultant (e.g., lobe 0), with remaining lobes set to 0.0 (never omitted).
GC_VMF_N_LOBES = 3

# =============================================================================
# PRIMITIVE MAP FUSE (STREAMING REDUCTION)
# =============================================================================
# Chunk size for streaming scatter-add in primitive_map_fuse. Fixed-cost (compile-time constant).
# Smaller chunks reduce peak memory; larger chunks improve throughput.
GC_FUSE_CHUNK_SIZE = 1024

# Block size for association -> fuse streaming (measurements per block).
GC_ASSOC_BLOCK_SIZE = 256

# Fixed insertion budget per scan (global PrimitiveMap). Atlas tiling will replace this later.
GC_K_INSERT = 64
GC_K_INSERT_TILE = GC_K_INSERT

# =============================================================================
# CAMERA INTRINSICS/EXTRINSICS (required when pose_evidence_backend="primitives")
# =============================================================================
# These are loaded from config at runtime; fail-fast if missing when needed.
# Format: T_base_camera = [x, y, z, rx, ry, rz] (rotvec, radians)
# K = [fx, fy, cx, cy] (pinhole model, no distortion for now)

# Placeholder values - must be overridden in config when camera is used
GC_DEFAULT_CAMERA_K = [500.0, 500.0, 320.0, 240.0]  # fx, fy, cx, cy
GC_DEFAULT_T_BASE_CAMERA = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # identity
