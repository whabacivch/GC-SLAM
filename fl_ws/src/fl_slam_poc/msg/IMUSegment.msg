## IMUSegment: Raw IMU segment between keyframes (Contract B)
##
## Contract (binding):
## - Units: gyro rad/s, accel m/s^2
## - Frame: IMU/body frame
## - Delta convention: ΔR = R_i^T R_j
## - Δp, Δv expressed in i-frame
## - Timebase: stamps are in the same clock domain as keyframes
## - Gravity: world frame; use gravity_world if provided, else backend parameter
##
## Required fields:
## - keyframe_i, keyframe_j
## - t_i, t_j
## - accel[k], gyro[k], stamp[k] (monotonic stamps)
## - bias_ref_bg, bias_ref_ba at segment start

std_msgs/Header header

uint64 keyframe_i
uint64 keyframe_j

float64 t_i
float64 t_j

float64[3] bias_ref_bg
float64[3] bias_ref_ba

float64[3] gravity_world

float64[] stamp
float64[] accel
float64[] gyro
