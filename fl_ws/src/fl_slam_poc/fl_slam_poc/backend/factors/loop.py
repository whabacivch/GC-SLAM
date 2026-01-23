"""Loop factor processing for backend."""

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
    se3_adjoint_np,
    se3_compose_np,
    se3_cov_compose_np,
    se3_inverse_np,
    se3_relative_np,
)

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend
    from fl_slam_poc.msg import LoopFactor


def process_loop(backend: "FLBackend", msg: "LoopFactor") -> None:
    """Loop closure update via one-shot barycentric recomposition (no Schur complement)."""
    backend.loop_factor_count += 1
    backend.last_loop_time = time.time()

    anchor_id = int(msg.anchor_id)

    if not hasattr(backend, "_loop_recv_count"):
        backend._loop_recv_count = 0
    backend._loop_recv_count += 1
    if backend._loop_recv_count <= 3:
        backend.get_logger().info(f"Backend received loop factor #{backend._loop_recv_count} for anchor {anchor_id}")

    anchor_data = backend.anchors.get(anchor_id)

    if backend.loop_factor_count <= 5:
        backend.get_logger().info(
            f"Loop factor #{backend.loop_factor_count}: anchor {anchor_id} "
            f"{'FOUND' if anchor_data is not None else 'NOT FOUND'}, "
            f"total anchors: {len(backend.anchors)}"
        )

    if anchor_data is None:
        if anchor_id not in backend.pending_loop_factors:
            backend.pending_loop_factors[anchor_id] = []

        if len(backend.pending_loop_factors[anchor_id]) >= backend.max_pending_loops_per_anchor:
            pending_count = len(backend.pending_loop_factors[anchor_id])
            backend.get_logger().warn(
                f"Loop factor buffer budget exceeded for anchor {anchor_id}: "
                f"{pending_count} pending (budget={backend.max_pending_loops_per_anchor})"
            )
            from fl_slam_poc.backend.diagnostics import publish_report
            publish_report(backend, OpReport(
                name="LoopFactorBufferBudgetExceeded",
                exact=True,
                family_in="Gaussian",
                family_out="Gaussian",
                closed_form=True,
                domain_projection=False,
                metrics={
                    "anchor_id": anchor_id,
                    "pending_count": pending_count,
                    "budget": backend.max_pending_loops_per_anchor,
                },
                notes="Loop factor buffer exceeded configured compute budget; factor retained.",
            ), backend.pub_report)
        backend.pending_loop_factors[anchor_id].append(msg)
        backend.get_logger().debug(
            f"Buffering loop factor for unknown anchor {anchor_id} "
            f"({len(backend.pending_loop_factors[anchor_id])} pending)"
        )

        from fl_slam_poc.backend.diagnostics import publish_report
        publish_report(backend, OpReport(
            name="LoopFactorBuffered",
            exact=True,
            family_in="Gaussian",
            family_out="Gaussian",
            closed_form=True,
            domain_projection=False,
            metrics={"anchor_id": anchor_id, "buffered": True},
            notes="Loop factor arrived before anchor creation - buffered for processing.",
        ), backend.pub_report)
        return

    mu_anchor, cov_anchor, L_anchor, h_anchor, _ = anchor_data

    rx = float(msg.rel_pose.position.x)
    ry = float(msg.rel_pose.position.y)
    rz = float(msg.rel_pose.position.z)
    qx = float(msg.rel_pose.orientation.x)
    qy = float(msg.rel_pose.orientation.y)
    qz = float(msg.rel_pose.orientation.z)
    qw = float(msg.rel_pose.orientation.w)
    rotvec_rel = quat_to_rotvec(np.array([qx, qy, qz, qw], dtype=float))
    rel = np.array([rx, ry, rz, rotvec_rel[0], rotvec_rel[1], rotvec_rel[2]], dtype=float)

    cov_rel = np.array(msg.covariance, dtype=float).reshape(6, 6)
    weight = max(float(msg.weight), 0.0)

    mu_full, cov_full = mean_cov(backend.L, backend.h)
    mu_current_pose = mu_full[:6]
    cov_current_pose = cov_full[:6, :6]
    mu_anchor_pose = mu_anchor[:6]
    cov_anchor_pose = cov_anchor[:6, :6]
    cov_trace_curr_before = float(np.trace(cov_current_pose))
    cov_trace_anchor_before = float(np.trace(cov_anchor_pose))

    def _spd_solve(A: np.ndarray, b: np.ndarray, name: str) -> np.ndarray:
        """Strict SPD solve via Cholesky - no fallback."""
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(f"{name}: expected square matrix, got {A.shape}")
        if b.shape[0] != A.shape[0]:
            raise ValueError(f"{name}: rhs shape {b.shape} incompatible with {A.shape}")
        Lc = np.linalg.cholesky(A)
        y = np.linalg.solve(Lc, b)
        return np.linalg.solve(Lc.T, y)

    def _gaussian_product(
        mu_a: np.ndarray, cov_a: np.ndarray, mu_b: np.ndarray, cov_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        cov_a = np.asarray(cov_a, dtype=float)
        cov_b = np.asarray(cov_b, dtype=float)
        mu_a = np.asarray(mu_a, dtype=float).reshape(-1)
        mu_b = np.asarray(mu_b, dtype=float).reshape(-1)

        I_a = _spd_solve(cov_a, np.eye(cov_a.shape[0]), "loop.cov_a")
        I_b = _spd_solve(cov_b, np.eye(cov_b.shape[0]), "loop.cov_b")
        Sigma_inv = I_a + I_b
        Sigma = _spd_solve(Sigma_inv, np.eye(Sigma_inv.shape[0]), "loop.Sigma_inv")
        mu = Sigma @ (I_a @ mu_a + I_b @ mu_b)
        Sigma = 0.5 * (Sigma + Sigma.T)
        return mu, Sigma

    cov_rel_eff = cov_rel / max(weight, constants.WEIGHT_EPSILON)

    mu_curr_pred = se3_compose_np(mu_anchor_pose, rel)
    cov_curr_pred = se3_cov_compose_np(cov_anchor_pose, cov_rel_eff, mu_anchor_pose)

    rel_inv = se3_inverse_np(rel)
    Ad_rel_inv = se3_adjoint_np(rel_inv)
    cov_rel_inv = Ad_rel_inv @ cov_rel_eff @ Ad_rel_inv.T
    mu_anchor_pred = se3_compose_np(mu_current_pose, rel_inv)
    cov_anchor_pred = se3_cov_compose_np(cov_current_pose, cov_rel_inv, mu_current_pose)

    L_curr_prior, h_curr_prior = make_evidence(mu_current_pose, cov_current_pose)
    L_curr_meas, h_curr_meas = make_evidence(mu_curr_pred, cov_curr_pred)
    L_anchor_prior, h_anchor_prior = make_evidence(mu_anchor_pose, cov_anchor_pose)
    L_anchor_meas, h_anchor_meas = make_evidence(mu_anchor_pred, cov_anchor_pred)

    L_curr_fused, h_curr_fused, trust_diag_curr = trust_scaled_fusion(
        L_curr_prior,
        h_curr_prior,
        L_curr_meas,
        h_curr_meas,
        max_divergence=MAX_ALPHA_DIVERGENCE_PRIOR,
        alpha=ALPHA_DIVERGENCE_DEFAULT,
    )
    L_anchor_fused, h_anchor_fused, trust_diag_anchor = trust_scaled_fusion(
        L_anchor_prior,
        h_anchor_prior,
        L_anchor_meas,
        h_anchor_meas,
        max_divergence=MAX_ALPHA_DIVERGENCE_PRIOR,
        alpha=ALPHA_DIVERGENCE_DEFAULT,
    )

    mu_pose_new, cov_pose_new = mean_cov(L_curr_fused, h_curr_fused)
    mu_anchor_new, cov_anchor_new = mean_cov(L_anchor_fused, h_anchor_fused)
    cov_trace_curr_after = float(np.trace(cov_pose_new))
    cov_trace_anchor_after = float(np.trace(cov_anchor_new))

    # 15D state update (no 6D path)
    mu_new = mu_full.copy()
    mu_new[:6] = mu_pose_new

    cov_new = cov_full.copy()
    cov_new[:6, :6] = cov_pose_new
    backend.L, backend.h = make_evidence(mu_new, cov_new)

    anchor_points = anchor_data[4]

    # 15D anchor update (no 6D path)
    mu_anchor_full = mu_anchor.copy()
    mu_anchor_full[:6] = mu_anchor_new

    cov_anchor_full = cov_anchor.copy()
    cov_anchor_full[:6, :6] = cov_anchor_new
    cov_anchor_full[:6, 6:] = 0.0
    cov_anchor_full[6:, :6] = 0.0

    L_anchor_new_full, h_anchor_new_full = make_evidence(mu_anchor_full, cov_anchor_full)
    backend.anchors[anchor_id] = (
        mu_anchor_full.copy(),
        cov_anchor_full.copy(),
        L_anchor_new_full.copy(),
        h_anchor_new_full.reshape(-1).copy(),
        anchor_points,
    )

    mu_updated, cov_updated = mean_cov(backend.L, backend.h)
    cov_current = cov_full

    Z_pred = se3_compose_np(se3_inverse_np(mu_anchor_pose), mu_current_pose)
    innovation = se3_relative_np(rel, Z_pred)

    from fl_slam_poc.backend.diagnostics import publish_map, publish_report
    publish_map(
        backend, backend.anchors, backend.dense_modules, backend.odom_frame,
        backend.pub_map, backend.PointCloud2, backend.PointField
    )

    _, frob_stats = gaussian_frobenius_correction(innovation)

    publish_report(backend,
        OpReport(
            name="LoopFactorRecomposition",
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
                "anchor_id": int(msg.anchor_id),
                "weight": weight,
                "innovation_norm": float(np.linalg.norm(innovation)),
                "cov_rel_trace": float(np.trace(cov_rel)),
                "anchor_cov_trace_after": float(np.trace(cov_anchor_new)),
                "current_cov_trace_after": float(np.trace(cov_updated)),
                "trust_beta_curr": trust_diag_curr["beta"],
                "trust_beta_anchor": trust_diag_anchor["beta"],
                "trust_divergence_full_curr": trust_diag_curr["divergence_full"],
                "trust_divergence_full_anchor": trust_diag_anchor["divergence_full"],
                "trust_quality_curr": trust_diag_curr["trust_quality"],
                "trust_quality_anchor": trust_diag_anchor["trust_quality"],
            "benefit_expected_curr": trust_diag_curr["beta"],
            "benefit_expected_anchor": trust_diag_anchor["beta"],
            "benefit_realized_curr": cov_trace_curr_before - cov_trace_curr_after,
            "benefit_realized_anchor": cov_trace_anchor_before - cov_trace_anchor_after,
            },
            notes="Loop closure via trust-scaled precision fusion (alpha-divergence).",
        ), backend.pub_report)

    from fl_slam_poc.backend.diagnostics import publish_loop_marker, publish_state
    publish_loop_marker(
        backend, int(msg.anchor_id), mu_anchor_new, mu_updated,
        backend.odom_frame, backend.pub_loop_markers
    )
    publish_state(
        backend, "loop", backend.L, backend.h, backend.odom_frame,
        backend.pub_state, backend.pub_path, backend.tf_broadcaster,
        backend.trajectory_poses, backend.max_path_length,
        backend.trajectory_file, backend.last_odom_stamp,
    )

    if backend.loop_factor_count <= 3:
        backend.get_logger().info(
            f"Loop #{backend.loop_factor_count} processed: innovation_norm={np.linalg.norm(innovation):.6f}, "
            f"weight={weight:.6f}, cov_trace_before={float(np.trace(cov_current)):.3f}, "
            f"cov_trace_after={float(np.trace(cov_updated)):.3f}"
        )
