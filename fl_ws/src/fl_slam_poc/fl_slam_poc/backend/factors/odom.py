"""Odometry factor processing for backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np

from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.backend.fusion.gaussian_info import (
    ALPHA_DIVERGENCE_DEFAULT,
    MAX_ALPHA_DIVERGENCE_PRIOR,
    make_evidence,
    mean_cov,
    trust_scaled_fusion,
)
from fl_slam_poc.common import constants
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.jax_utils import (
    quat_to_rotvec,
    se3_compose_np,
    se3_cov_compose_np,
    se3_relative_np,
)
from fl_slam_poc.common.utils import stamp_to_sec

if TYPE_CHECKING:
    from nav_msgs.msg import Odometry
    from fl_slam_poc.backend.backend_node import FLBackend


def _spd_solve(A: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
    """Strict SPD solve using Cholesky (raises on failure)."""
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"{name}: expected square matrix, got {A.shape}")
    if b.shape[0] != A.shape[0]:
        raise ValueError(f"{name}: rhs shape {b.shape} incompatible with {A.shape}")
    Lc = np.linalg.cholesky(A)
    y = np.linalg.solve(Lc, b)
    return np.linalg.solve(Lc.T, y)


def process_odom(backend: "FLBackend", msg: "Odometry") -> None:
    """
    Process delta odometry using information-geometric covariance weighting.

    The odom covariance tells us measurement confidence per dimension:
    - High variance (Z: 1e6) → low precision → minimal update influence
    - Low variance (XY: 0.001) → high precision → strong update influence

    This is by-construction correct: no heuristics, just proper fusion.
    """
    # Duplicate detection: skip if we've already processed this exact message
    odom_key = (msg.header.stamp.sec, msg.header.stamp.nanosec)
    if odom_key == backend._last_odom_key:
        backend.get_logger().debug(
            "Odom: duplicate stamp skipped",
            throttle_duration_sec=5.0,
        )
        return
    backend._last_odom_key = odom_key

    backend.odom_count += 1
    backend.last_odom_time = time.time()
    # Store odometry message timestamp for trajectory export (NOT wall clock!)
    backend.last_odom_stamp = stamp_to_sec(msg.header.stamp)

    # Extract delta (6D pose)
    dx = float(msg.pose.pose.position.x)
    dy = float(msg.pose.pose.position.y)
    dz = float(msg.pose.pose.position.z)

    qx = float(msg.pose.pose.orientation.x)
    qy = float(msg.pose.pose.orientation.y)
    qz = float(msg.pose.pose.orientation.z)
    qw = float(msg.pose.pose.orientation.w)
    rotvec_delta = quat_to_rotvec(np.array([qx, qy, qz, qw], dtype=float))

    delta_pose = np.array([dx, dy, dz, rotvec_delta[0], rotvec_delta[1], rotvec_delta[2]], dtype=float)

    # Extract odom measurement covariance (6x6)
    # This encodes per-dimension confidence: Z has ~1e6 variance = garbage
    odom_cov = np.array(msg.pose.covariance, dtype=float).reshape(6, 6)

    # Log first few messages with covariance info
    if backend.odom_count <= 3:
        diag_var = np.diag(odom_cov)
        backend.get_logger().info(
            f"Backend received odom #{backend.odom_count}, "
            f"delta=({dx:.3f}, {dy:.3f}, {dz:.3f}), "
            f"var=[{diag_var[0]:.4f}, {diag_var[1]:.4f}, {diag_var[2]:.1f}]"
        )

    # Get current state (15D - no 6D path)
    mu, cov = mean_cov(backend.L, backend.h)

    # 15D state: Extract pose, compose, then rebuild
    mu_pose = mu[:6]
    mu_vel = mu[6:9]
    mu_bias = mu[9:15]

    linearization_point = mu_pose.copy()

    # Compose pose to get predicted mean (6D)
    mu_pose_pred = se3_compose_np(mu_pose, delta_pose)

    # Get process noise (15D)
    Q_full = backend.process_noise.estimate()
    Q_pose = Q_full[:6, :6]

    # =====================================================================
    # Information-geometric covariance fusion
    # =====================================================================
    cov_pose_prior = cov[:6, :6]
    cov_pose_transported = se3_cov_compose_np(cov_pose_prior, Q_pose, mu_pose)

    reg = np.eye(6) * constants.COV_REGULARIZATION_MIN
    cov_transported_reg = cov_pose_transported + reg
    odom_cov_reg = odom_cov + reg

    # Strict inversion - no fallback to pinv
    L_prior = _spd_solve(cov_transported_reg, np.eye(6), "odom.cov_transported_reg")
    L_meas = _spd_solve(odom_cov_reg, np.eye(6), "odom.odom_cov_reg")

    h_prior = L_prior @ mu_pose  # Prior wants us at old pose
    h_meas = L_meas @ mu_pose_pred  # Measurement wants us at predicted pose

    L_fused, h_fused, trust_diag = trust_scaled_fusion(
        L_prior,
        h_prior,
        L_meas,
        h_meas,
        max_divergence=MAX_ALPHA_DIVERGENCE_PRIOR,
        alpha=ALPHA_DIVERGENCE_DEFAULT,
    )

    # Strict inversion - no fallback to pinv
    cov_pose_fused = _spd_solve(L_fused, np.eye(6), "odom.L_fused")
    mu_pose_fused = _spd_solve(L_fused, h_fused, "odom.L_fused")

    if backend.odom_count <= 3:
        delta_from_pred = mu_pose_fused - mu_pose_pred
        backend.get_logger().info(
            f"Info fusion #{backend.odom_count}: "
            f"shift_from_pred=({delta_from_pred[0]*1000:.1f}, {delta_from_pred[1]*1000:.1f}, "
            f"{delta_from_pred[2]*1000:.1f})mm"
        )

    mu_pred = np.concatenate([mu_pose_fused, mu_vel, mu_bias])

    cov_pred = cov.copy()
    cov_pred[:6, :6] = cov_pose_fused
    cov_pred[6:9, 6:9] += Q_full[6:9, 6:9]
    cov_pred[9:12, 9:12] += Q_full[9:12, 9:12]
    cov_pred[12:15, 12:15] += Q_full[12:15, 12:15]

    residual_vec = np.zeros(constants.STATE_DIM_FULL, dtype=float)
    if backend.prev_mu is not None:
        prev_pose = backend.prev_mu[:6]
        predicted_from_prev = se3_compose_np(prev_pose, delta_pose)
        residual_vec[:6] = se3_relative_np(mu_pose_fused, predicted_from_prev).astype(float)
        backend.process_noise.update(residual_vec)
    backend.prev_mu = mu.copy()

    # Update state
    backend.L, backend.h = make_evidence(mu_pred, cov_pred)
    backend.state_buffer.append((stamp_to_sec(msg.header.stamp), mu_pred.copy(), cov_pred.copy()))

    from fl_slam_poc.backend.diagnostics import publish_state, publish_report
    publish_state(
        backend, "odom", backend.L, backend.h, backend.odom_frame,
        backend.pub_state, backend.pub_path, backend.tf_broadcaster,
        backend.trajectory_poses, backend.max_path_length,
        backend.trajectory_file, backend.last_odom_stamp,
    )

    residual_pose = residual_vec[:6]  # 15D state - extract pose residual
    _, frob_stats = gaussian_frobenius_correction(residual_pose)

    Q_full = backend.process_noise.estimate()

    publish_report(backend, OpReport(
        name="GaussianPredictSE3",
        exact=False,
        approximation_triggers=["Linearization"],
        family_in="Gaussian",
        family_out="Gaussian",
        closed_form=True,
        frobenius_applied=True,
        frobenius_operator="gaussian_identity_third_order",
        frobenius_delta_norm=float(frob_stats["delta_norm"]),
        frobenius_input_stats=dict(frob_stats["input_stats"]),
        frobenius_output_stats=dict(frob_stats["output_stats"]),
        metrics={
            "covariance_transport": "adjoint",
            "state_dim": backend.state_dim,
            "linearization_point": linearization_point.tolist(),
            "process_noise_trace": float(np.trace(Q_full)),
            "process_noise_confidence": backend.process_noise.confidence(),
        },
        notes="Delta-odom composed in SE(3) with adjoint covariance transport.",
    ), backend.pub_report)
