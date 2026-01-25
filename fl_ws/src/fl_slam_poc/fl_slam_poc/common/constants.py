"""
Golden Child SLAM v2 constants only.

Legacy constants have been moved to:
  archive/legacy_common/constants_legacy.py
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

# Epsilon constants (domain stabilization)
GC_EPS_PSD = 1e-12  # Minimum eigenvalue for PSD projection
GC_EPS_LIFT = 1e-9  # Lift for SPD solves (configurable via yaml)
GC_EPS_MASS = 1e-12  # Mass regularization for InvMass
GC_EPS_R = 1e-6  # Clamp epsilon for Rbar in kappa
GC_EPS_DEN = 1e-12  # Denominator regularization in kappa
GC_EXC_EPS = 1e-12  # Domain guard for excitation ratios

# World gravity (m/s^2) in the odom/world frame used by evidence extraction.
GC_GRAVITY_W = (0.0, 0.0, -9.81)

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

# State slice indices (0-based, per spec Section 1.1)
GC_SLICE_SO3_START = 0
GC_SLICE_SO3_END = 3
GC_SLICE_TRANS_START = 3
GC_SLICE_TRANS_END = 6
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
# Units:
# - `GC_IMU_GYRO_NOISE_DENSITY` is treated as continuous-time gyro *rate* noise PSD,
#   with effective units rad^2 / s. When used for an integrated angle residual over
#   duration dt, we discretize once as: Σ_rot ≈ PSD * dt.
# - `GC_IMU_ACCEL_NOISE_DENSITY` is treated as continuous-time accel noise PSD,
#   with effective units m^2 / s^3 (since (m/s^2)^2 * s).
GC_IMU_GYRO_NOISE_DENSITY = 8.7e-7   # rad^2 / s   (gyro rate noise PSD proxy)
GC_IMU_ACCEL_NOISE_DENSITY = 9.5e-5  # m^2 / s^3   (accel noise PSD proxy)

# Default LiDAR translation-measurement covariance used by TranslationWLS (prior; adapted by IW updates).
GC_LIDAR_SIGMA_MEAS = 0.01  # isotropic 3x3 covariance scale (legacy default)

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
# State block ordering: [rot(3), trans(3), vel(3), bg(3), ba(3), dt(1), ex(6)].
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

# Measurement-noise retention (separate from process noise; deterministic per scan)
GC_IW_RHO_MEAS_GYRO = 0.995
GC_IW_RHO_MEAS_ACCEL = 0.995
GC_IW_RHO_MEAS_LIDAR = 0.99

# Test-only invariants still referenced by active test suite.
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95
