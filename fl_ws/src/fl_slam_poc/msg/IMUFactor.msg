# IMUFactor: Preintegrated IMU measurement between two keyframes
#
# This message carries the result of IMU preintegration (Forster et al. 2017)
# between keyframe_i and keyframe_j, including:
# - Preintegrated increments (delta_p, delta_v, delta_rotvec)
# - Covariance of preintegration uncertainty
# - Bias reference used during integration
# - Number of IMU measurements integrated

std_msgs/Header header

# Keyframe identifiers
uint64 keyframe_i      # Reference keyframe ID (anchor)
uint64 keyframe_j      # Target keyframe ID (current)

# Integration interval
float64 dt             # Time interval in seconds

# Preintegrated increments (body frame)
float64[3] delta_p     # Position increment [m]
float64[3] delta_v     # Velocity increment [m/s]
float64[3] delta_rotvec # Rotation increment (axis-angle) [rad]

# Bias reference used during preintegration
float64[3] bias_gyro   # Gyroscope bias [rad/s]
float64[3] bias_accel  # Accelerometer bias [m/s²]

# Covariance matrix (9×9 flattened row-major: [p(3), v(3), R(3)])
# Basis: [δp_x, δp_y, δp_z, δv_x, δv_y, δv_z, δω_x, δω_y, δω_z]
float64[81] cov_preint

# Number of IMU measurements integrated
uint32 n_measurements
