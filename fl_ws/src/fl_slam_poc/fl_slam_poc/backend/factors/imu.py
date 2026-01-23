"""IMU factor processing for backend."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numpy as np
import jax.numpy as jnp

from fl_slam_poc.backend.routing.dirichlet_router import DirichletRoutingModule
from fl_slam_poc.backend.fusion.gaussian_geom import imu_tangent_frobenius_correction
from fl_slam_poc.backend.fusion.gaussian_info import make_evidence, mean_cov
from fl_slam_poc.common.geometry.se3_jax import se3_compose as se3_compose_jax
from fl_slam_poc.common.geometry.se3_jax import se3_exp as se3_exp_jax
from fl_slam_poc.backend.math.imu_kernel import imu_batched_projection_kernel
from fl_slam_poc.common import constants
from fl_slam_poc.common.jax_utils import to_jax, to_numpy
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.validation import (
    ContractViolation,
    validate_timestamp,
    validate_covariance,
    detect_hardcoded_value,
    warn_near_zero_delta,
)

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend
    from fl_slam_poc.msg import IMUSegment


def _se3_compose(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return to_numpy(se3_compose_jax(to_jax(a), to_jax(b)))


def _se3_exp(xi: np.ndarray) -> np.ndarray:
    return to_numpy(se3_exp_jax(to_jax(xi)))


def estimate_preint_covariance(
    dt_total: float,
    n_meas: int,
    gyro_noise_density: float,
    accel_noise_density: float,
) -> np.ndarray:
    """
    Approximate preintegration covariance from noise densities.
    """
    if n_meas < 2:
        return np.eye(9, dtype=float) * 1e6
    dt_avg = dt_total / max(n_meas - 1, 1)
    sigma_g = gyro_noise_density / np.sqrt(max(dt_avg, constants.WEIGHT_EPSILON))
    sigma_a = accel_noise_density / np.sqrt(max(dt_avg, constants.WEIGHT_EPSILON))

    cov_preint = np.zeros((9, 9), dtype=float)
    cov_preint[:3, :3] = np.eye(3) * (sigma_a**2) * dt_total**2
    cov_preint[3:6, 3:6] = np.eye(3) * (sigma_a**2) * dt_total
    cov_preint[6:9, 6:9] = np.eye(3) * (sigma_g**2) * dt_total
    cov_preint += np.eye(9, dtype=float) * constants.COV_REGULARIZATION_MIN
    return cov_preint


def process_imu_segment(backend: "FLBackend", msg: "IMUSegment") -> None:
    """
    Process IMU segment with batched moment matching across anchors.

    This is the core of the Hellinger-Dirichlet IMU integration (Phase 2).
    """
    backend.imu_factor_count += 1
    backend.last_imu_time = time.time()

    # =====================================================================
    # Extract IMU segment data (Contract B)
    # =====================================================================
    keyframe_i = int(msg.keyframe_i)  # Reference keyframe (anchor)
    keyframe_j = int(msg.keyframe_j)  # Current keyframe
    t_i = float(msg.t_i)
    t_j = float(msg.t_j)
    dt_header = max(t_j - t_i, 0.0)

    stamps = np.asarray(msg.stamp, dtype=np.float64).reshape(-1)
    accel_raw = np.asarray(msg.accel, dtype=np.float64).reshape(-1)
    gyro_raw = np.asarray(msg.gyro, dtype=np.float64).reshape(-1)

    if stamps.size == 0 or accel_raw.size % 3 != 0 or gyro_raw.size % 3 != 0:
        backend.get_logger().warn("IMU segment malformed: empty stamps or invalid accel/gyro length")
        return

    accel = accel_raw.reshape((-1, 3))
    gyro = gyro_raw.reshape((-1, 3))
    if accel.shape[0] != stamps.shape[0] or gyro.shape[0] != stamps.shape[0]:
        backend.get_logger().warn("IMU segment malformed: stamps/accel/gyro length mismatch")
        return

    dt_stamps = float(stamps[-1] - stamps[0]) if stamps.size > 0 else 0.0
    dt = max(dt_stamps, 0.0)
    stamp_deltas = np.diff(stamps) if stamps.size > 1 else np.array([], dtype=np.float64)
    non_monotonic_count = int(np.sum(stamp_deltas <= 0)) if stamp_deltas.size > 0 else 0
    stamp_delta_min = float(np.min(stamp_deltas)) if stamp_deltas.size > 0 else None
    stamp_delta_mean = float(np.mean(stamp_deltas)) if stamp_deltas.size > 0 else None
    stamp_delta_max = float(np.max(stamp_deltas)) if stamp_deltas.size > 0 else None
    dt_gap_start = float(stamps[0] - t_i) if stamps.size > 0 else None
    dt_gap_end = float(t_j - stamps[-1]) if stamps.size > 0 else None

    bias_gyro = np.asarray(msg.bias_ref_bg, dtype=np.float64).reshape(3)
    bias_accel = np.asarray(msg.bias_ref_ba, dtype=np.float64).reshape(3)
    bias_ref = np.concatenate([bias_gyro, bias_accel])

    gravity_msg = np.asarray(msg.gravity_world, dtype=np.float64).reshape(-1)
    if gravity_msg.size == 3 and np.linalg.norm(gravity_msg) > 0.0:
        gravity_world = gravity_msg
    else:
        gravity_world = backend.gravity

    n_measurements = int(stamps.shape[0])
    R_imu = estimate_preint_covariance(
        dt_total=dt,
        n_meas=n_measurements,
        gyro_noise_density=backend.imu_gyro_noise_density,
        accel_noise_density=backend.imu_accel_noise_density,
    )
    R_nom = R_imu.copy()

    # =====================================================================
    # Contract Validation (Data Flow Audit)
    # =====================================================================
    try:
        # Validate timestamps
        validate_timestamp(t_i, "imu_segment.t_i")
        validate_timestamp(t_j, "imu_segment.t_j")

        if t_j <= t_i:
            raise ContractViolation(
                f"imu_segment: t_j ({t_j:.6f}) <= t_i ({t_i:.6f})"
            )

        # Validate IMU measurements are finite
        if not np.all(np.isfinite(accel)):
            raise ContractViolation(f"imu_segment.accel: Contains inf/nan")
        if not np.all(np.isfinite(gyro)):
            raise ContractViolation(f"imu_segment.gyro: Contains inf/nan")
        if not np.all(np.isfinite(stamps)):
            raise ContractViolation(f"imu_segment.stamps: Contains inf/nan")

        # Validate covariance
        validate_covariance(R_imu, "imu_segment.R_imu")

        # Detect hardcoded identity covariance (common fake value)
        detect_hardcoded_value(R_imu, np.eye(9), "imu_segment.R_imu")

        # Log actual values for first few IMU segments (data flow audit)
        if backend.imu_factor_count <= 5:
            accel_norm = np.linalg.norm(accel, axis=1).mean()
            gyro_norm = np.linalg.norm(gyro, axis=1).mean()
            backend.get_logger().info(
                f"IMU segment #{backend.imu_factor_count} validation: "
                f"dt={dt:.4f}s, n_meas={n_measurements}, "
                f"accel_mean_norm={accel_norm:.4f}m/s², "
                f"gyro_mean_norm={gyro_norm:.4f}rad/s, "
                f"R_imu_trace={np.trace(R_imu):.3e}"
            )

    except ContractViolation as e:
        backend.get_logger().error(
            f"IMU segment contract violation (kf_i={keyframe_i}, kf_j={keyframe_j}): {e}"
        )
        from fl_slam_poc.backend.diagnostics import publish_report
        publish_report(backend, OpReport(
            name="IMUSegmentContractViolation",
            exact=False,
            family_in="IMU",
            family_out="None",
            closed_form=False,
            domain_projection=False,
            metrics={
                "keyframe_i": keyframe_i,
                "keyframe_j": keyframe_j,
                "error": str(e),
            },
            notes=f"Contract violation prevented IMU segment processing: {e}",
        ), backend.pub_report)
        return  # Skip this segment

    # =====================================================================
    # Validation
    # =====================================================================
    if len(backend.anchors) == 0 or keyframe_i not in backend.anchors:
        if keyframe_i not in backend.pending_imu_factors:
            backend.pending_imu_factors[keyframe_i] = []
        if len(backend.pending_imu_factors[keyframe_i]) >= backend.max_pending_imu_per_anchor:
            pending_count = len(backend.pending_imu_factors[keyframe_i])
            backend.get_logger().warn(
                f"IMU buffer budget exceeded for keyframe {keyframe_i}: "
                f"{pending_count} pending (budget={backend.max_pending_imu_per_anchor})"
            )
            from fl_slam_poc.backend.diagnostics import publish_report
            publish_report(backend, OpReport(
                name="IMUSegmentBufferBudgetExceeded",
                exact=True,
                family_in="IMU",
                family_out="Gaussian",
                closed_form=True,
                domain_projection=False,
                metrics={
                    "keyframe_i": keyframe_i,
                    "pending_count": pending_count,
                    "budget": backend.max_pending_imu_per_anchor,
                },
                notes="IMU segment buffer exceeded configured compute budget; segment retained.",
            ), backend.pub_report)
        backend.pending_imu_factors[keyframe_i].append(msg)
        backend.get_logger().info(
            f"IMU segment buffered: kf_{keyframe_i} (anchors={len(backend.anchors)})",
            throttle_duration_sec=10.0,
        )
        from fl_slam_poc.backend.diagnostics import publish_report
        publish_report(backend, OpReport(
            name="IMUSegmentBuffered",
            exact=True,
            family_in="IMU",
            family_out="Gaussian",
            closed_form=True,
            domain_projection=False,
            metrics={"keyframe_i": keyframe_i, "buffered": True},
            notes="IMU segment arrived before anchor creation - buffered for processing.",
        ), backend.pub_report)
        return

    # =====================================================================
    # Get current state (15D)
    # =====================================================================
    mu_current, cov_current = mean_cov(backend.L, backend.h)
    cov_trace_before = float(np.trace(cov_current))

    # =====================================================================
    # Build anchor data for batched processing
    # Anchors are 6D (pose); embed into 15D with velocity=0 and bias=bias_ref
    # =====================================================================
    anchor_ids = sorted(backend.anchors.keys())
    M = len(anchor_ids)

    anchor_mus = np.zeros((M, 15), dtype=np.float64)
    anchor_covs = np.zeros((M, 15, 15), dtype=np.float64)

    for idx, aid in enumerate(anchor_ids):
        mu_a, cov_a, _, _, _ = backend.anchors[aid]

        # Embed 6D anchor pose into 15D state
        anchor_mus[idx, :6] = mu_a[:6]  # Pose
        anchor_mus[idx, 6:9] = 0.0  # Velocity (unknown at anchor time)
        anchor_mus[idx, 9:15] = bias_ref  # Bias from preintegration reference

        # Embed covariance
        anchor_covs[idx, :6, :6] = cov_a[:6, :6]
        anchor_covs[idx, 6:9, 6:9] = np.eye(3) * constants.STATE_PRIOR_VELOCITY_STD**2
        anchor_covs[idx, 9:12, 9:12] = np.eye(3) * constants.STATE_PRIOR_GYRO_BIAS_STD**2
        anchor_covs[idx, 12:15, 12:15] = np.eye(3) * constants.STATE_PRIOR_ACCEL_BIAS_STD**2

    # =====================================================================
    # Enhanced Anchor Matching: keyframe_to_anchor mapping + Hellinger fallback
    # =====================================================================
    mu_current_pose = mu_current[:6]

    # Compute distances to all anchors (for fallback matching)
    anchor_distances = np.zeros(M, dtype=np.float64)
    for idx, aid in enumerate(anchor_ids):
        mu_a_pose = anchor_mus[idx, :6]
        # Simple Euclidean distance on position
        anchor_distances[idx] = np.linalg.norm(mu_a_pose[:3] - mu_current_pose[:3])

    # =====================================================================
    # Dirichlet routing with enhanced initial logits
    # =====================================================================
    if not hasattr(backend, "_imu_routing_module") or backend._imu_routing_module is None:
        backend._imu_routing_module = DirichletRoutingModule(n_anchors=M)
    elif backend._imu_routing_module.n_anchors != M:
        backend._imu_routing_module.resize(M)

    # Enhanced initial logits: use keyframe mapping if available, else distance
    initial_logits = np.zeros(M, dtype=np.float64)

    if keyframe_i in backend.keyframe_to_anchor:
        mapped_anchor_id = backend.keyframe_to_anchor[keyframe_i]
        if mapped_anchor_id in anchor_ids:
            mapped_idx = anchor_ids.index(mapped_anchor_id)
            initial_logits[mapped_idx] = constants.IMU_ROUTING_MAPPED_LOGIT
            if backend.imu_factor_count <= 3:
                backend.get_logger().info(
                    f"IMU segment: using keyframe mapping kf_{keyframe_i} -> anchor_{mapped_anchor_id}"
                )
    else:
        min_idx = np.argmin(anchor_distances)
        min_dist = anchor_distances[min_idx]
        initial_logits[min_idx] = -constants.IMU_ROUTING_DISTANCE_SCALE * min_dist
        if backend.imu_factor_count <= 3:
            backend.get_logger().info(
                f"IMU segment: using distance nearest-neighbor (kf_{keyframe_i} -> anchor_{anchor_ids[min_idx]}, "
                f"dist={min_dist:.4f}m)"
            )

    routing_weights = backend._imu_routing_module.update(initial_logits)

    # =====================================================================
    # JAX Batched Projection Kernel (Contract B)
    # =====================================================================
    imu_valid = np.ones((n_measurements,), dtype=bool)
    joint_mean_jax, cov_joint_jax, diagnostics = imu_batched_projection_kernel(
        anchor_mus=jnp.array(anchor_mus),
        anchor_covs=jnp.array(anchor_covs),
        current_mu=jnp.array(mu_current),
        current_cov=jnp.array(cov_current),
        routing_weights=jnp.array(routing_weights),
        imu_stamps=jnp.array(stamps),
        imu_accel=jnp.array(accel),
        imu_gyro=jnp.array(gyro),
        imu_valid=jnp.array(imu_valid),
        R_imu=jnp.array(R_imu),
        R_nom=jnp.array(R_nom),
        dt_total=dt,
        gravity=jnp.array(gravity_world),
    )

    joint_mean = np.array(joint_mean_jax)
    cov_joint = np.array(cov_joint_jax)

    # =====================================================================
    # Apply Frobenius (BCH third-order) correction to JAX kernel output
    # =====================================================================
    delta_p_out = joint_mean[0:3]
    delta_v_out = joint_mean[3:6]
    delta_R_out = joint_mean[6:9]

    delta_p, delta_v, delta_rotvec, imu_frob_stats = imu_tangent_frobenius_correction(
        delta_p_out, delta_v_out, delta_R_out, cov_preint=R_imu
    )

    joint_mean[0:3] = delta_p
    joint_mean[3:6] = delta_v
    joint_mean[6:9] = delta_rotvec

    # =====================================================================
    # Post-Frobenius Validation (Detect dead integration)
    # =====================================================================
    if warn_near_zero_delta(delta_p, "imu_delta_p") and warn_near_zero_delta(delta_v, "imu_delta_v"):
        backend.get_logger().warning(
            f"IMU segment #{backend.imu_factor_count}: Near-zero deltas after Frobenius correction! "
            f"delta_p_norm={np.linalg.norm(delta_p):.6e}m, "
            f"delta_v_norm={np.linalg.norm(delta_v):.6e}m/s. "
            f"This may indicate dead integration (no actual IMU data flowing)."
        )

    # Log actual delta values for first few segments (data flow audit)
    if backend.imu_factor_count <= 5:
        backend.get_logger().info(
            f"IMU segment #{backend.imu_factor_count} deltas after Frobenius: "
            f"delta_p_norm={np.linalg.norm(delta_p):.4f}m, "
            f"delta_v_norm={np.linalg.norm(delta_v):.4f}m/s, "
            f"delta_rot_norm={np.linalg.norm(delta_rotvec):.4f}rad, "
            f"frob_correction_norm={imu_frob_stats['delta_norm']:.6e}"
        )

    # =====================================================================
    # Exact marginalization (Schur) on joint Gaussian
    # =====================================================================
    L_joint, h_joint = make_evidence(joint_mean, cov_joint)
    L_ii = L_joint[:15, :15]
    L_ij = L_joint[:15, 15:]
    L_ji = L_joint[15:, :15]
    L_jj = L_joint[15:, 15:]
    h_i = h_joint[:15]
    h_j = h_joint[15:]

    L_ii_reg = L_ii + np.eye(15, dtype=L_ii.dtype) * constants.COV_REGULARIZATION_MIN
    L_ii_chol = np.linalg.cholesky(L_ii_reg)
    L_ii_inv_L_ij = np.linalg.solve(L_ii_chol, L_ij)
    L_ii_inv_h_i = np.linalg.solve(L_ii_chol, h_i)
    L_j = L_jj - L_ji @ L_ii_inv_L_ij
    h_j = h_j - L_ji @ L_ii_inv_h_i

    delta_mu, delta_cov = mean_cov(L_j, h_j)

    # Retract delta onto current state
    pose_delta = delta_mu[:6]
    rest_delta = delta_mu[6:]
    pose_new = _se3_compose(mu_current[:6], _se3_exp(pose_delta))
    rest_new = mu_current[6:] + rest_delta
    mu_new = np.concatenate([pose_new, rest_new])
    cov_new = delta_cov

    # =====================================================================
    # Apply bias random walk noise (Gaussian random walk prior)
    # =====================================================================
    Q_bias = np.zeros((6, 6), dtype=np.float64)
    Q_bias[:3, :3] = np.eye(3) * (backend.imu_gyro_random_walk**2) * dt
    Q_bias[3:6, 3:6] = np.eye(3) * (backend.imu_accel_random_walk**2) * dt
    cov_new[9:15, 9:15] += Q_bias
    cov_trace_after = float(np.trace(cov_new))

    # =====================================================================
    # Update state
    # =====================================================================
    backend.L, backend.h = make_evidence(mu_new, cov_new)
    from fl_slam_poc.backend.diagnostics import publish_state
    publish_state(
        backend, "imu", backend.L, backend.h, backend.odom_frame,
        backend.pub_state, backend.pub_path, backend.tf_broadcaster,
        backend.trajectory_poses, backend.max_path_length,
        backend.trajectory_file, backend.last_odom_stamp,
    )

    # =====================================================================
    # Diagnostics
    # =====================================================================
    routing_diag = backend._imu_routing_module.get_update_diagnostics()
    max_resp = float(np.max(routing_weights)) if routing_weights.size > 0 else 0.0

    if backend.imu_factor_count <= 5:
        v_new = mu_new[6:9]
        backend.get_logger().info(
            f"IMU segment #{backend.imu_factor_count} applied: "
            f"v_norm={np.linalg.norm(v_new):.3f}m/s, "
            f"valid_anchors={diagnostics.get('valid_anchors', 0)}/{M}, "
            f"hellinger_shift={routing_diag['hellinger_shift']:.4f}"
        )

    from fl_slam_poc.backend.diagnostics import publish_report
    publish_report(backend, OpReport(
        name="IMUFactorUpdate",
            exact=False,
            approximation_triggers=["LegendreEProjection"],
            family_in="IMU",
            family_out="Gaussian",
            closed_form=False,
            frobenius_applied=True,
            frobenius_operator="imu_tangent_bch_third_order",
            frobenius_delta_norm=float(imu_frob_stats["delta_norm"]),
            frobenius_input_stats=dict(imu_frob_stats["input_stats"]),
            frobenius_output_stats=dict(imu_frob_stats["output_stats"]),
            metrics={
                "keyframe_i": keyframe_i,
                "keyframe_j": keyframe_j,
                "dt_header": float(dt_header),
                "dt_stamps": float(dt_stamps),
                "dt_gap_start": float(dt_gap_start) if dt_gap_start is not None else 0.0,
                "dt_gap_end": float(dt_gap_end) if dt_gap_end is not None else 0.0,
                "stamp_delta_min": stamp_delta_min,
                "stamp_delta_mean": stamp_delta_mean,
                "stamp_delta_max": stamp_delta_max,
                "non_monotonic_count": non_monotonic_count,
                "dt_sec": dt,
                "n_measurements": n_measurements,
                "delta_p_norm_m": float(np.linalg.norm(delta_p)),
                "delta_v_norm_ms": float(np.linalg.norm(delta_v)),
                "delta_rot_norm_rad": float(np.linalg.norm(delta_rotvec)),
                "bias_in_model": True,
                "factor_scope": "two_state",
                "projection": "e_projection(moment_match)",
                "sigma_scheme": "spherical_radial_cubature",
                "marginalization": "Schur",
                "convention_delta_R": "R_i^T R_j",
                "convention_delta_frame": "i_frame",
                "gravity_world": gravity_world.tolist(),
                "state_dim": backend.state_dim,
                "n_anchors": M,
                "valid_anchors": diagnostics.get("valid_anchors", 0),
                "degenerate_weights": diagnostics.get("degenerate_weights", False),
                "ess": diagnostics.get("ess", None),
                "hellinger_mean": diagnostics.get("hellinger_mean", None),
                "weight_entropy": diagnostics.get("weight_entropy", None),
                "routing_alpha": routing_diag["alpha"].tolist(),
                "routing_w": routing_diag["responsibilities"].tolist(),
                "routing_retention": routing_diag["retention"],
                "routing_hellinger_shift": routing_diag["hellinger_shift"],
                "routing_max_resp": max_resp,
            "benefit_expected": float(diagnostics.get("ess") or 0.0),
            "benefit_realized": cov_trace_before - cov_trace_after,
                "bias_rw_cov_adaptive": False,
                "bias_rw_cov_trace_gyro": 0.0,
                "bias_rw_cov_trace_accel": 0.0,
            },
        notes="IMU two-state factor: joint update + single e-projection + Schur marginalization.",
    ), backend.pub_report)
