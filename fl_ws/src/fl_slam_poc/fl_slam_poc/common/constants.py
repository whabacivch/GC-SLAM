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

# IW update timing: minimum scan count before applying updates
# Rationale: Need real deltas (scan-to-scan differences) for meaningful IW updates.
# Scans 0 and 1 don't have sufficient history for reliable innovation residuals.
GC_IW_UPDATE_MIN_SCAN = 2  # Start IW updates from scan 2 onwards

# Measurement-noise retention (separate from process noise; deterministic per scan)
GC_IW_RHO_MEAS_GYRO = 0.995
GC_IW_RHO_MEAS_ACCEL = 0.995
GC_IW_RHO_MEAS_LIDAR = 0.99

# Test-only invariants still referenced by active test suite.
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95
