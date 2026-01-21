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

# Epsilon for covariance regularization
COV_REGULARIZATION_MIN = 1e-9  # Minimum eigenvalue for positive definiteness

# Epsilon for numerical comparisons
NUMERICAL_EPSILON = 1e-6  # General numerical tolerance

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

# Maximum length for state history buffer
STATE_BUFFER_MAX_LENGTH = 500

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
# Loop Closure Constants
# =============================================================================

# Minimum responsibility for creating an anchor
ANCHOR_RESPONSIBILITY_MIN = 0.1

# Minimum responsibility for publishing a loop factor
LOOP_RESPONSIBILITY_MIN = 0.1

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
# IMU Constants
# =============================================================================

# Default IMU noise parameters (typical for consumer-grade IMU like RealSense)
# Reference: Forster et al. (2017) "On-Manifold Preintegration"
IMU_GYRO_NOISE_DENSITY_DEFAULT = 1.0e-3     # rad/s/sqrt(Hz)
IMU_ACCEL_NOISE_DENSITY_DEFAULT = 1.0e-2    # m/s^2/sqrt(Hz)
IMU_GYRO_RANDOM_WALK_DEFAULT = 1.0e-5       # rad/s^2/sqrt(Hz)
IMU_ACCEL_RANDOM_WALK_DEFAULT = 1.0e-4      # m/s^3/sqrt(Hz)

# Default IMU topic (M3DGR dataset uses /camera/imu)
IMU_TOPIC_DEFAULT = "/camera/imu"

# Gravity vector (world frame, z-up convention)
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

# State dimension
STATE_DIM_POSE = 6      # [x, y, z, rx, ry, rz]
STATE_DIM_VELOCITY = 3  # [vx, vy, vz]
STATE_DIM_BIAS = 6      # [bg_x, bg_y, bg_z, ba_x, ba_y, ba_z]
STATE_DIM_FULL = 15     # Pose + Velocity + Bias

# IMU factor message ordering (from IMUFactor.msg)
# Covariance is 9x9 in basis [delta_p(3), delta_v(3), delta_theta(3)]
IMU_FACTOR_COV_DIM = 9
