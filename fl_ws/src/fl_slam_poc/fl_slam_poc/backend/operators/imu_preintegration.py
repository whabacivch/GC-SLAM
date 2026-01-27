"""
IMU preintegration primitives for constant-twist deskew and IMU evidence.

Implements a simple, deterministic, fixed-cost discrete-time preintegration:
  - integrate gyro -> delta_R
  - rotate accel to world, add gravity -> a_world
  - integrate a_world -> delta_v, delta_p

This is branch-free w.r.t. data: weights are continuous and can go to ~0.
"""

from __future__ import annotations

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.geometry import se3_jax


@jax.jit
def smooth_window_weights(
    imu_stamps: jnp.ndarray,  # (M,)
    scan_start_time: float,
    scan_end_time: float,
    sigma: float,
) -> jnp.ndarray:
    """
    Smooth membership weights for IMU samples in [scan_start_time, scan_end_time].

    w(t) = sigmoid((t - start)/sigma) * sigmoid((end - t)/sigma)
    """
    t = jnp.asarray(imu_stamps, dtype=jnp.float64)
    start = jnp.array(scan_start_time, dtype=jnp.float64)
    end = jnp.array(scan_end_time, dtype=jnp.float64)
    sig = jnp.maximum(jnp.array(sigma, dtype=jnp.float64), 1e-6)

    a = (t - start) / sig
    b = (end - t) / sig
    w_min = jax.nn.sigmoid(a)
    w_max = jax.nn.sigmoid(b)
    # Strictly-positive continuous floor (no discrete gating / no exact zeros).
    w_raw = w_min * w_max
    wf = jnp.array(constants.GC_WEIGHT_FLOOR, dtype=jnp.float64)
    return w_raw * (1.0 - wf) + wf


@jax.jit
def preintegrate_imu_relative_pose_jax(
    imu_stamps: jnp.ndarray,  # (M,)
    imu_gyro: jnp.ndarray,    # (M,3) rad/s
    imu_accel: jnp.ndarray,   # (M,3) m/s^2
    weights: jnp.ndarray,     # (M,) continuous
    rotvec_start_WB: jnp.ndarray,  # (3,) world->body rotation at scan start (approx)
    gyro_bias: jnp.ndarray,   # (3,)
    accel_bias: jnp.ndarray,  # (3,)
    gravity_W: jnp.ndarray,   # (3,)
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Fixed-cost IMU preintegration over the given stamp array.

    Returns:
      delta_pose_se3: (6,) [trans, rotvec] representing relative motion over the window
      delta_R:        (3,3) relative rotation
      delta_p:        (3,) relative position in world frame
      ess:            scalar effective sample size proxy = sum(weights)
    """
    imu_stamps = jnp.asarray(imu_stamps, dtype=jnp.float64).reshape(-1)
    imu_gyro = jnp.asarray(imu_gyro, dtype=jnp.float64)
    imu_accel = jnp.asarray(imu_accel, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64).reshape(-1)

    w = weights
    ess = jnp.sum(w)

    # dt_i = t_{i+1} - t_i (last dt forced to 0)
    dt = jnp.concatenate([imu_stamps[1:] - imu_stamps[:-1], jnp.zeros((1,), dtype=jnp.float64)], axis=0)
    dt = jnp.maximum(dt, 0.0)

    # Start orientation
    R = se3_jax.so3_exp(rotvec_start_WB)

    def step(carry, inp):
        R_k, v_k, p_k = carry
        gyro_i, accel_i, dt_i, w_i = inp

        # scale dt by continuous weight (so low-weight samples contribute negligibly)
        dt_eff = w_i * dt_i

        omega = gyro_i - gyro_bias
        dR = se3_jax.so3_exp(omega * dt_eff)
        R_next = R_k @ dR

        a_body = accel_i - accel_bias
        a_world = R_k @ a_body + gravity_W

        v_next = v_k + a_world * dt_eff
        p_next = p_k + v_k * dt_eff + 0.5 * a_world * (dt_eff * dt_eff)
        return (R_next, v_next, p_next), None

    carry0 = (R, jnp.zeros((3,), dtype=jnp.float64), jnp.zeros((3,), dtype=jnp.float64))
    (R_end, v_end, p_end), _ = jax.lax.scan(step, carry0, (imu_gyro, imu_accel, dt, w))

    # CRITICAL FIX: Compute RELATIVE rotation delta, not absolute end orientation.
    # R_end = R_start @ dR_0 @ dR_1 @ ... (absolute)
    # delta_R = R_start^T @ R_end = dR_0 @ dR_1 @ ... (relative)
    # Before: returned rotvec_end = Log(R_end) which is absolute orientation.
    # This caused imu_gyro_rotation_evidence to compute R_end_imu = R_start @ Exp(rotvec_end)
    # = R_start @ R_end (WRONG!) instead of just R_end.
    R_start = se3_jax.so3_exp(rotvec_start_WB)
    delta_R = R_start.T @ R_end  # Relative rotation from start to end
    rotvec_delta = se3_jax.so3_log(delta_R)
    delta_pose = jnp.concatenate([p_end, rotvec_delta], axis=0)
    return delta_pose, delta_R, p_end, ess

