"""
FL-SLAM Constants and Configuration Values.

All magic numbers are centralized here with clear documentation.
"""

# =============================================================================
# Depth Processing Constants
# =============================================================================

# Valid depth range for depth camera processing (meters)
DEPTH_MIN_VALID = 0.1  # Minimum valid depth (10cm) - closer is sensor noise
DEPTH_MAX_VALID = 10.0  # Maximum valid depth (10m) - farther is unreliable

# Depth image stride for point cloud generation
DEPTH_STRIDE_DEFAULT = 4  # Process every 4th pixel (balance speed vs density)

# =============================================================================
# Numerical Stability Thresholds
# =============================================================================

# Epsilon for weight/probability comparisons
WEIGHT_EPSILON = 1e-12  # Weights below this are considered zero

# Numerical floor for responsibility normalization (domain projection, not gating)
# When total responsibility mass is below this, use uniform to prevent division by zero.
RESPONSIBILITY_MASS_FLOOR = 1e-12

# Epsilon for covariance regularization
# Standardized to 1e-8 for consistency across JAX kernels and NumPy code
COV_REGULARIZATION_MIN = 1e-8  # Minimum eigenvalue for positive definiteness

# Epsilon for numerical comparisons
NUMERICAL_EPSILON = 1e-6  # General numerical tolerance

# Epsilon for timestamp/temporal comparisons (seconds)
# Used to determine if alignment is significant
TIMESTAMP_EPSILON = 1e-9

# Epsilon for Dirichlet parameter domain projection (alpha > EPS)
# Stricter than general numerical epsilon to ensure interior of simplex
DIRICHLET_INTERIOR_EPS = 1e-9

# =============================================================================
# Sensor Timeout Constants
# =============================================================================

# How long before a sensor is considered stale (seconds)
SENSOR_TIMEOUT_DEFAULT = 5.0

# Grace period after node startup before warning about missing sensors (seconds)
SENSOR_STARTUP_GRACE_PERIOD = 10.0

# =============================================================================
# Buffer and History Constants
# =============================================================================

# Maximum length for feature/sensor data buffers
FEATURE_BUFFER_MAX_LENGTH = 10

# Maximum length for state history buffer (timestamp alignment window)
STATE_BUFFER_MAX_LENGTH = 500

# =============================================================================
# Compute Budget Constants (Declared, not heuristic)
# =============================================================================

# RGB-D association scale prior (meters) for dense module fusion.
# Interpreted as the typical spatial deviation between RGB-D evidence and its
# anchor; used as the Gaussian distance scale in soft association (NOT a gate).
DENSE_ASSOCIATION_RADIUS_DEFAULT = 0.5

# Prior weight for creating a new dense module (new-component responsibility).
# Interpreted as a prior mass (ESS) in the association mixture.
DENSE_NEW_COMPONENT_WEIGHT_PRIOR = 1.0

# Dense module compute budget (max modules retained)
DENSE_MODULE_COMPUTE_BUDGET = 10000

# Dense module culling fraction (budgeted recomposition)
DENSE_MODULE_KEEP_FRACTION = 0.8

# Pending factor buffer budgets (pre-anchor arrival buffering)
LOOP_PENDING_BUFFER_BUDGET = 100
IMU_PENDING_BUFFER_BUDGET = 100

# Base module mass prior (ESS) used for atlas modules.
# Represents the prior count before evidence fusion.
MODULE_MASS_PRIOR = 1.0

# Maximum trajectory path length for visualization
TRAJECTORY_PATH_MAX_LENGTH = 1000

# =============================================================================
# ICP Constants
# =============================================================================

# Minimum degrees of freedom for SE(3) observability
N_MIN_SE3_DOF = 6  # SE(3) has 6 DOF, need at least 6 constraints

# Sigmoid steepness for DOF weight function
K_SIGMOID = 0.5  # Chosen so w(n=6) ≈ 0.5, w(n=12) ≈ 0.95

# Default ICP convergence tolerance
ICP_TOLERANCE_DEFAULT = 1e-4

# Default ICP maximum iterations
ICP_MAX_ITER_DEFAULT = 15

# ICP reference point count for information weighting
ICP_N_REF_DEFAULT = 100.0

# ICP MSE sigma for quality weighting
ICP_SIGMA_MSE_DEFAULT = 0.01

# ICP covariance scales (prior ratios for translation/rotation).
# These encode the prior relative scale of translational vs rotational residuals
# in se(3) tangent space for typical ICP residual statistics.
ICP_COV_TRANS_SCALE = 1.0
ICP_COV_ROT_SCALE = 0.01

# ICP covariance for degenerate cases (no valid points)
# Prior scale used when no valid points are available (domain projection).
ICP_COVARIANCE_PRIOR_DEGENERATE = 1.0

# =============================================================================
# Descriptor Constants
# =============================================================================

# Number of bins for scan descriptors (histogram)
DESCRIPTOR_BINS_DEFAULT = 60

# =============================================================================
# Timestamp Alignment Constants
# =============================================================================

# Prior sigma for timestamp alignment (seconds)
ALIGNMENT_SIGMA_PRIOR = 0.1

# Prior strength for alignment model
ALIGNMENT_PRIOR_STRENGTH = 5.0

# Floor value for alignment sigma (minimum uncertainty)
ALIGNMENT_SIGMA_FLOOR = 0.001

# =============================================================================
# Process Noise Constants
# =============================================================================

# Prior for translational process noise (meters)
PROCESS_NOISE_TRANS_PRIOR = 0.03

# Prior for rotational process noise (radians)
PROCESS_NOISE_ROT_PRIOR = 0.015

# Prior strength for process noise model
PROCESS_NOISE_PRIOR_STRENGTH = 10.0

# =============================================================================
# NIG (Normal-Inverse-Gamma) Prior Constants
# =============================================================================

# Prior parameters for descriptor models
NIG_PRIOR_KAPPA = 1.0   # Prior belief strength
NIG_PRIOR_ALPHA = 2.0   # Shape parameter (must be > 1 for finite variance)
NIG_PRIOR_BETA = 1.0    # Scale parameter

# =============================================================================
# Birth Model Constants
# =============================================================================

# Poisson intensity for new component birth
BIRTH_INTENSITY_DEFAULT = 10.0

# Expected scan period (seconds)
SCAN_PERIOD_DEFAULT = 0.1

# Base weight for new components
BASE_COMPONENT_WEIGHT_DEFAULT = 1.0

# =============================================================================
# Fisher-Rao Distance Constants
# =============================================================================

# Prior scale for Fisher-Rao distance normalization
FR_DISTANCE_SCALE_PRIOR = 1.0

# Prior strength for Fisher-Rao scale
FR_SCALE_PRIOR_STRENGTH = 5.0

# =============================================================================
# Warning Count Limits
# =============================================================================

# Maximum times to warn about missing data before suppressing
MAX_WARNING_COUNT = 3

# =============================================================================
# Covariance Initial Values
# =============================================================================

# Initial position covariance (meters^2)
INIT_POS_COV = 0.2**2

# Initial rotation covariance (radians^2)
INIT_ROT_COV = 0.1**2

# =============================================================================
# Debug/Logging Constants
# =============================================================================

# How often to log periodic status (seconds)
STATUS_CHECK_PERIOD = 5.0

# How many initial scans to log for debugging
INITIAL_SCAN_LOG_COUNT = 10

# How often to log scan count after initial period
SCAN_LOG_FREQUENCY = 20

# =============================================================================
# 3D Point Cloud Processing Constants
# =============================================================================

# Default voxel size for point cloud downsampling (meters)
VOXEL_SIZE_DEFAULT = 0.05  # 5cm voxels balance detail vs performance

# Maximum number of points after filtering (for memory management)
MAX_POINTS_AFTER_FILTER = 50000

# Minimum points required for valid 3D ICP registration
MIN_POINTS_FOR_3D_ICP = 100

# Maximum correspondence distance for 3D ICP (meters)
ICP_MAX_CORRESPONDENCE_DIST_DEFAULT = 0.5

# Normal estimation radius (meters) - for point-to-plane ICP
NORMAL_ESTIMATION_RADIUS = 0.1

# =============================================================================
# GPU Processing Constants
# =============================================================================

# Default CUDA device index
CUDA_DEVICE_INDEX = 0

# GPU memory limit for point cloud operations (bytes)
# RTX 4050 has ~6GB VRAM, reserve some for system
GPU_MEMORY_LIMIT = 4 * 1024 * 1024 * 1024  # 4GB

# Batch size for GPU nearest neighbor queries
GPU_BATCH_SIZE = 10000

# =============================================================================
# 3D Sensor Defaults
# =============================================================================

# Default point cloud topic (generic LiDAR-style naming).
# For RGB-D cameras that publish PointCloud2, override this to `/camera/depth/points`.
POINTCLOUD_TOPIC_DEFAULT = "/lidar/points"

# Point cloud message queue size
POINTCLOUD_QUEUE_SIZE = 2

# Point cloud processing rate limit (Hz) - prevent overload
POINTCLOUD_RATE_LIMIT_HZ = 30.0

# =============================================================================
# QoS (Quality of Service) Constants
# =============================================================================

# QoS depth for high-frequency sensor topics (IMU at 200Hz)
QOS_DEPTH_SENSOR_HIGH_FREQ = 500  # ~2.5s buffer at 200Hz

# QoS depth for medium-frequency topics (odom, LiDAR at ~20Hz)
QOS_DEPTH_SENSOR_MED_FREQ = 100  # ~5s buffer at 20Hz

# QoS depth for low-frequency topics (TF, state)
QOS_DEPTH_LOW_FREQ = 100

# Default QoS reliability for IMU (BEST_EFFORT for rosbag compatibility)
# Options: "reliable", "best_effort", "system_default"
QOS_IMU_RELIABILITY_DEFAULT = "best_effort"

# =============================================================================
# IMU Constants
# =============================================================================

# Default IMU noise parameters.
# Notes:
# - The pipeline uses noise densities (measurement noise) for preintegration.
# - IMU bias evolution is handled via a self-adaptive Wishart model; we intentionally
#   DO NOT use IMU random-walk parameters as user-tunable inputs.
IMU_GYRO_NOISE_DENSITY_DEFAULT = 1.7e-4     # rad/s/sqrt(Hz)
IMU_ACCEL_NOISE_DENSITY_DEFAULT = 1.9e-4    # m/s^2/sqrt(Hz)

# NOTE: Random-walk parameters are not used in the MVP pipeline (adaptive process noise instead).
IMU_GYRO_RANDOM_WALK_DEFAULT = 1.0e-5       # rad/s^2/sqrt(Hz)
IMU_ACCEL_RANDOM_WALK_DEFAULT = 1.0e-4      # m/s^3/sqrt(Hz)

# Bias innovation prior (per sqrt(second)) used to seed the adaptive bias noise model.
# This is a prior, not a dataset calibration input.
IMU_GYRO_BIAS_INNOV_STD_PRIOR = 1.0e-4      # rad/s/sqrt(s)
IMU_ACCEL_BIAS_INNOV_STD_PRIOR = 1.0e-3     # m/s^2/sqrt(s)

# Default IMU topic (M3DGR: Livox MID-360 IMU)
IMU_TOPIC_DEFAULT = "/livox/mid360/imu"

# IMU acceleration scale factor (prior)
# Livox Mid-360 IMU (ICM40609) outputs acceleration in g's, needs scaling to m/s².
# Set to 9.81 for Livox sensors, 1.0 for sensors that output m/s² directly.
IMU_ACCEL_SCALE_DEFAULT = 9.81  # g to m/s² conversion

# Gravity vector (world frame, z-down by default; override via ROS params if needed)
GRAVITY_DEFAULT = (0.0, 0.0, -9.81)

# =============================================================================
# Keyframe Constants
# =============================================================================

# Motion-based keyframe thresholds
KEYFRAME_TRANSLATION_THRESHOLD_DEFAULT = 0.5   # meters
KEYFRAME_ROTATION_THRESHOLD_DEFAULT = 0.26     # radians (~15 degrees)

# =============================================================================
# 15D State Extension Constants (Backend Phase 2)
# =============================================================================
# State ordering: [p(3), R(3), v(3), b_g(3), b_a(3)]

# Initial state prior standard deviations
STATE_PRIOR_POSE_TRANS_STD = 0.2       # meters
STATE_PRIOR_POSE_ROT_STD = 0.1         # radians
STATE_PRIOR_VELOCITY_STD = 1.0         # m/s (high uncertainty at init)
STATE_PRIOR_GYRO_BIAS_STD = 0.01       # rad/s
STATE_PRIOR_ACCEL_BIAS_STD = 0.1       # m/s^2

# Process noise for bias random walk (per-second variance)
PROCESS_NOISE_VELOCITY_STD = 0.1       # m/s per sqrt(s)
PROCESS_NOISE_GYRO_BIAS_STD = 1.0e-5   # rad/s per sqrt(s) (matches IMU_GYRO_RANDOM_WALK_DEFAULT)
PROCESS_NOISE_ACCEL_BIAS_STD = 1.0e-4  # m/s^2 per sqrt(s) (matches IMU_ACCEL_RANDOM_WALK_DEFAULT)

# State dimension (15D only - no 6D pose-only path)
STATE_DIM_VELOCITY = 3  # [vx, vy, vz]
STATE_DIM_BIAS = 6      # [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
STATE_DIM_FULL = 15     # Pose(6) + Velocity(3) + Bias(6)

# IMU preintegration ordering (Contract B)
# Covariance is 9x9 in basis [delta_p(3), delta_v(3), delta_theta(3)]
IMU_FACTOR_COV_DIM = 9

# =============================================================================
# IMU Routing Constants (Dirichlet Prior Parameters)
# =============================================================================

# Prior logit strength for known keyframe-to-anchor mapping.
# Statistical interpretation: log-odds ratio for deterministic mapping vs uniform prior.
# Equivalent to prior effective sample size ≈ exp(5.0)/sum(exp(logits)) ≈ 148 pseudo-observations.
IMU_ROUTING_MAPPED_LOGIT = 5.0

# Spatial scale parameter for distance-based logit prior (meters).
# exp(-λ·d) gives Gaussian-like decay with scale √(1/λ); λ=2.0 → scale ≈ 0.7m.
IMU_ROUTING_DISTANCE_SCALE = 2.0

# =============================================================================
# JAX Kernel Constants
# =============================================================================

# Minimum mixture weight to consider valid (prevents division by zero)
MIN_MIXTURE_WEIGHT = 1e-15

# Hellinger tilt weight for robustness to outliers
HELLINGER_TILT_WEIGHT = 2.0

# =============================================================================
# Debug/Logging Thresholds
# =============================================================================
# These control how many times certain log messages are emitted before throttling

RESPONSIBILITY_DEBUG_LOG_THRESHOLD = 5  # First N responsibility computations logged
BIRTH_DEBUG_LOG_THRESHOLD = 10  # First N birth events logged
LOOP_DEBUG_LOG_THRESHOLD = 5  # First N loop factor events logged
ICP_VALIDATION_LOG_THRESHOLD = 5  # First N ICP validation results logged
LOOP_FACTOR_VALIDATION_LOG_THRESHOLD = 3  # First N loop factor validations logged
LOOP_PUBLISHED_LOG_THRESHOLD = 5  # First N published loop factors logged
SCAN_BUFFER_LOG_FREQUENCY = 10  # Log scan buffer every N scans
IMU_FACTOR_LOG_THRESHOLD = 5  # First N IMU factor events logged
IMU_SEGMENT_BUFFER_LOG_THRESHOLD = 3  # First N IMU segment buffer events logged
LOOP_RECEIVE_LOG_THRESHOLD = 3  # First N loop factor receives logged
LOOP_PROCESS_LOG_THRESHOLD = 5  # First N loop factor processes logged
LOOP_COMPLETION_LOG_THRESHOLD = 3  # First N loop completions logged

# =============================================================================
# Frontend Processing Constants
# =============================================================================

ANCHOR_CREATE_MAX_POINTS = 1000  # Max points to include in anchor message
DEPTH_POINTS_FALLBACK_MAX = 1000  # Max depth points when pointcloud unavailable
RGBD_PUBLISH_RATE_DIVISOR = 5  # Publish RGB-D every N scans
RGBD_SPATIAL_GRID_SIZE = 0.1  # Grid size for spatial subsampling (meters)
RGBD_KAPPA_NORMAL_DEFAULT = 10.0  # vMF concentration for surface normals
RGBD_COLOR_VARIANCE_DEFAULT = 0.01  # Color noise variance (RGB space)
RGBD_ALPHA_MEAN_DEFAULT = 1.0  # Dirichlet alpha mean for occupancy
RGBD_ALPHA_VAR_DEFAULT = 0.1  # Dirichlet alpha variance for occupancy
STATUS_PUBLISH_INTERVAL_SEC = 1.0  # Status message publish interval

# =============================================================================
# Adaptive Parameter Floors
# =============================================================================
# Minimum values to prevent numerical instability

FR_DISTANCE_SCALE_FLOOR = 0.01  # Min Fisher-Rao distance scale
ICP_MAX_ITER_FLOOR = 3.0  # Min ICP iterations
ICP_TOLERANCE_FLOOR = 1e-6  # Min ICP convergence tolerance

# =============================================================================
# IMU Thresholds
# =============================================================================

IMU_MIN_MEASUREMENTS_PUBLISH = 2  # Min IMU samples to publish segment
IMU_MIN_MEASUREMENTS_WARNING = 10  # Warn if fewer than this many samples
IMU_BUFFER_SIZE_WARNING = 100  # Warn if buffer exceeds this size
IMU_COVARIANCE_DEGENERATE = 1e6  # Covariance value indicating degenerate measurement
IMU_CHOLESKY_REGULARIZATION_FALLBACK = 1e-6  # Regularization for Cholesky failures

# =============================================================================
# Sensor I/O Constants
# =============================================================================

POINTCLOUD_RATE_LIMIT_HZ_DEFAULT = 30.0  # Default rate limit for point clouds
RGB_BYTES_PER_PIXEL = 3  # Bytes per pixel for RGB8/BGR8 encoding
RGBA_BYTES_PER_PIXEL = 4  # Bytes per pixel for RGBA8/BGRA8 encoding
MM_TO_M_SCALE = 1e-3  # Millimeter to meter conversion factor
