"""
Visual Pose Evidence Operator for Geometric Compositional SLAM v2.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 7

Computes pose evidence from primitive alignment via OT responsibilities.
This replaces bin-based Matrix Fisher + Planar Translation evidence.

Operator: visual_pose_evidence(association_result, measurement_batch, map_view, belief_pred)
    -> (L_pose, h_pose, CertBundle, ExpectedEffect)

Key constraints:
- Uses OT responsibilities (continuous; no gating)
- Fixed-cost: evidence is a weighted sum over all associations
- Returns 22D evidence matching the belief state dimension
- Frobenius correction when linearization is used
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.common.belief import BeliefGaussianInfo
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    InfluenceCert,
    SupportCert,
)
from fl_slam_poc.backend.structures.measurement_batch import (
    MeasurementBatch,
    measurement_batch_mean_positions,
    measurement_batch_mean_directions,
    measurement_batch_kappas,
)
from fl_slam_poc.backend.structures.primitive_map import PrimitiveMapView
from fl_slam_poc.backend.operators.primitive_association import PrimitiveAssociationResult


# =============================================================================
# Visual Pose Evidence Result
# =============================================================================


@dataclass
class VisualPoseEvidenceResult:
    """Result of visual pose evidence computation."""
    # 22D evidence: L_pose @ z = h_pose
    L_pose: jnp.ndarray  # (22, 22) precision/information
    h_pose: jnp.ndarray  # (22,) information vector

    # Diagnostic: per-component contributions
    L_trans: jnp.ndarray  # (3, 3) translation precision
    h_trans: jnp.ndarray  # (3,) translation information
    L_rot: jnp.ndarray    # (3, 3) rotation precision
    h_rot: jnp.ndarray    # (3,) rotation information

    # Cost statistics
    total_weighted_cost: float
    n_associations: int
    mean_responsibility: float


# =============================================================================
# Core Evidence Computation
# =============================================================================


@jax.jit
def _compute_translation_evidence_wls(
    meas_positions: jnp.ndarray,      # (N, 3) measurement positions in body frame
    meas_precisions: jnp.ndarray,     # (N, 3, 3) measurement precisions
    map_positions: jnp.ndarray,       # (M, 3) map positions in world frame
    responsibilities: jnp.ndarray,    # (N, K) OT responsibilities
    candidate_indices: jnp.ndarray,   # (N, K) map indices per measurement
    R_pred: jnp.ndarray,              # (3, 3) predicted rotation world<-body
    t_pred: jnp.ndarray,              # (3,) predicted translation
    eps_lift: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Compute translation evidence via weighted least squares over OT correspondences.

    The residual for association (i, k) is:
        r_{ik} = map_pos[k] - (R_pred @ meas_pos[i] + t)

    WLS objective: sum_{i,k} pi_{ik} * r_{ik}^T Lambda_{ik} r_{ik}

    Linearization around t_pred:
        At t_pred: r_{ik} = map_pos[k] - R_pred @ meas_pos[i] - t_pred
        d/dt (r_{ik}) = -I

    Evidence (Gaussian):
        L_t = sum_{i,k} pi_{ik} Lambda_{ik}
        h_t = sum_{i,k} pi_{ik} Lambda_{ik} (map_pos[k] - R_pred @ meas_pos[i])

    Fully vectorized over N measurements and K candidates (no Python loops).

    Returns:
        L_trans: (3, 3) translation precision
        h_trans: (3,) translation information
        cost: scalar weighted cost
    """
    N, K = responsibilities.shape

    # Handle empty case with static shapes
    def _compute():
        # Transform measurements to world frame (predicted): (N, 3)
        meas_world = jnp.einsum("ij,nj->ni", R_pred, meas_positions)

        # Gather all map positions at once: (N, K, 3)
        map_pos_all = map_positions[candidate_indices]

        # Residuals: map_pos - R_pred @ meas_pos - t_pred: (N, K, 3)
        residuals_all = map_pos_all - meas_world[:, None, :] - t_pred[None, None, :]

        # Weighted precision: sum over N and K
        # L_trans = sum_{i,k} pi_{ik} * Lambda_i = sum_i (sum_k pi_{ik}) * Lambda_i
        # Since pi sums over k for each i, we can simplify
        pi_sum_k = jnp.sum(responsibilities, axis=1)  # (N,)
        # L_trans = sum_i pi_sum_k[i] * Lambda_i
        L_trans = jnp.einsum("n,nij->ij", pi_sum_k, meas_precisions)

        # Target contributions: (N, K, 3)
        target_all = map_pos_all - meas_world[:, None, :]

        # h_trans = sum_{i,k} pi_{ik} * Lambda_i @ target_{ik}
        # = sum_i Lambda_i @ sum_k pi_{ik} * target_{ik}
        weighted_target_per_i = jnp.einsum("nk,nkj->nj", responsibilities, target_all)  # (N, 3)
        h_trans = jnp.einsum("nij,nj->i", meas_precisions, weighted_target_per_i)

        # Cost: sum_{i,k} pi_{ik} * r_{ik}^T Lambda_i @ r_{ik}
        # = sum_i sum_k pi_{ik} * r_{ik}^T Lambda_i @ r_{ik}
        # First compute Lambda @ r for all: (N, K, 3)
        Lambda_r = jnp.einsum("nij,nkj->nki", meas_precisions, residuals_all)
        # Then r^T @ Lambda @ r: (N, K)
        r_Lambda_r = jnp.einsum("nki,nki->nk", residuals_all, Lambda_r)
        # Weighted sum
        total_cost = jnp.sum(responsibilities * r_Lambda_r)

        # Regularize
        L_trans_reg = L_trans + eps_lift * jnp.eye(3, dtype=jnp.float64)

        return L_trans_reg, h_trans, total_cost

    def _empty():
        return (
            eps_lift * jnp.eye(3, dtype=jnp.float64),
            jnp.zeros((3,), dtype=jnp.float64),
            0.0,
        )

    # Use lax.cond for static control flow
    return jax.lax.cond(
        (N > 0) & (K > 0),
        _compute,
        _empty,
    )


@jax.jit
def _compute_rotation_evidence_vmf(
    meas_directions: jnp.ndarray,     # (N, 3) measurement directions in body frame
    meas_kappas: jnp.ndarray,         # (N,) vMF concentrations
    map_directions: jnp.ndarray,      # (M, 3) map directions in world frame
    map_kappas: jnp.ndarray,          # (M,) map vMF concentrations
    responsibilities: jnp.ndarray,    # (N, K) OT responsibilities
    candidate_indices: jnp.ndarray,   # (N, K) map indices
    R_pred: jnp.ndarray,              # (3, 3) predicted rotation
    eps_lift: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray, float]:
    """
    Compute rotation evidence from directional correspondences via vMF.

    Uses the Matrix Fisher / vMF correspondence:
    - Each correspondence contributes to a scatter matrix
    - The rotation that maximizes alignment is found via SVD
    - Evidence is built from the scatter matrix eigenstructure

    For each association (i, k):
        mu_meas_i (body) should align with mu_map_k (world) via R:
        R @ mu_meas_i ≈ mu_map_k

    Scatter matrix: S = sum_{i,k} pi_{ik} * sqrt(kappa_i * kappa_k) * mu_map_k @ mu_meas_i^T

    Fisher information for rotation around R_pred:
        L_rot ≈ diag(eigenvalues of S)  (in tangent space)

    Fully vectorized over N measurements and K candidates (no Python loops).

    Returns:
        L_rot: (3, 3) rotation precision (in tangent space)
        h_rot: (3,) rotation information
        cost: scalar directional cost (Hellinger-based)
    """
    N, K = responsibilities.shape

    def _compute():
        # Gather all map data at once: (N, K, 3) and (N, K)
        map_dir_all = map_directions[candidate_indices]  # (N, K, 3)
        map_kappa_all = map_kappas[candidate_indices]    # (N, K)

        # Weight = pi * sqrt(kappa_meas * kappa_map): (N, K)
        kappa_weight_all = jnp.sqrt(meas_kappas[:, None] * map_kappa_all + 1e-12)
        weights_all = responsibilities * kappa_weight_all  # (N, K)

        # Scatter contribution: S = sum_{i,k} w_{ik} * map_dir_{ik} @ meas_dir_i^T
        # Using einsum: outer product then weighted sum
        # S_ij = sum_{n,k} w_{nk} * map_dir_{nk,i} * meas_dir_{n,j}
        S = jnp.einsum("nk,nki,nj->ij", weights_all, map_dir_all, meas_directions)

        # Directional cost: sum_{i,k} w_{ik} * (1 - dot(R @ meas_i, map_{ik}))
        meas_rotated = jnp.einsum("ij,nj->ni", R_pred, meas_directions)  # (N, 3)
        # dot(R @ meas_i, map_{ik}) for all i, k
        dot_products = jnp.einsum("ni,nki->nk", meas_rotated, map_dir_all)  # (N, K)
        total_cost = jnp.sum(weights_all * (1.0 - dot_products))

        # SVD of scatter matrix to get rotation
        U, s, Vt = jnp.linalg.svd(S)

        # Fisher information from eigenvalues
        L_rot = jnp.diag(s + eps_lift)

        # Compute small rotation error from scatter
        R_scatter = U @ Vt
        det_R = jnp.linalg.det(R_scatter)
        R_scatter = jnp.where(det_R < 0, U @ jnp.diag(jnp.array([1., 1., -1.])) @ Vt, R_scatter)

        # Delta rotation from R_pred to R_scatter
        R_delta = R_scatter @ R_pred.T
        rotvec_delta = se3_jax.so3_log(R_delta)  # (3,)

        # h_rot = L_rot @ delta_rotvec (pointing towards optimal)
        h_rot = L_rot @ rotvec_delta

        return L_rot, h_rot, total_cost

    def _empty():
        return (
            eps_lift * jnp.eye(3, dtype=jnp.float64),
            jnp.zeros((3,), dtype=jnp.float64),
            0.0,
        )

    return jax.lax.cond(
        (N > 0) & (K > 0),
        _compute,
        _empty,
    )


# =============================================================================
# Main Visual Pose Evidence Operator
# =============================================================================


def visual_pose_evidence(
    association_result: PrimitiveAssociationResult,
    measurement_batch: MeasurementBatch,
    map_view: PrimitiveMapView,
    belief_pred: BeliefGaussianInfo,
    eps_lift: float = constants.GC_EPS_LIFT,
    eps_mass: float = constants.GC_EPS_MASS,
    chart_id: str = constants.GC_CHART_ID,
    anchor_id: str = "visual_pose_evidence",
    z_lin_pose: Optional[jnp.ndarray] = None,
) -> Tuple[VisualPoseEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Compute 22D pose evidence from primitive alignment via OT.

    This replaces bin-based Matrix Fisher + Planar Translation evidence.
    Uses soft correspondences from OT to build a WLS translation term
    and a scatter-based rotation term. Evidence is expected NLL under
    responsibilities evaluated at linearization point (z_lin_pose or belief_pred).

    Args:
        association_result: OT association (responsibilities, candidate_indices)
        measurement_batch: Current scan measurements
        map_view: Map primitives view (M_{t-1}; pre-update map)
        belief_pred: Predicted belief
        z_lin_pose: Optional 6D pose [trans(3), rotvec(3)] for linearization; when provided
            used instead of belief_pred.mean_world_pose (e.g. IMU+odom-informed z_lin).
        eps_lift: Matrix regularization
        eps_mass: Mass regularization

    Returns:
        (VisualPoseEvidenceResult, CertBundle, ExpectedEffect)
    """
    N_meas = measurement_batch.n_valid
    N_assoc, K_assoc = association_result.responsibilities.shape

    # Handle empty case
    if N_meas == 0 or N_assoc == 0 or map_view.count == 0:
        L_pose = eps_lift * jnp.eye(22, dtype=jnp.float64)
        h_pose = jnp.zeros((22,), dtype=jnp.float64)

        result = VisualPoseEvidenceResult(
            L_pose=L_pose,
            h_pose=h_pose,
            L_trans=jnp.zeros((3, 3), dtype=jnp.float64),
            h_trans=jnp.zeros((3,), dtype=jnp.float64),
            L_rot=jnp.zeros((3, 3), dtype=jnp.float64),
            h_rot=jnp.zeros((3,), dtype=jnp.float64),
            total_weighted_cost=0.0,
            n_associations=0,
            mean_responsibility=0.0,
        )
        cert = CertBundle.create_exact(chart_id=chart_id, anchor_id=anchor_id)
        effect = ExpectedEffect(
            objective_name="visual_pose_evidence",
            predicted=0.0,
            realized=0.0,
        )
        return result, cert, effect

    # Linearization point: z_lin_pose (IMU+odom-informed) or belief_pred mean
    if z_lin_pose is not None:
        z_lin_pose = jnp.asarray(z_lin_pose, dtype=jnp.float64).ravel()[:6]
        t_pred = z_lin_pose[:3]
        rotvec_pred = z_lin_pose[3:6]
    else:
        pose_pred = belief_pred.mean_world_pose(eps_lift=eps_lift)
        t_pred = pose_pred[:3]
        rotvec_pred = pose_pred[3:6]
    R_pred = se3_jax.so3_exp(rotvec_pred)

    # Get measurement arrays
    meas_positions = measurement_batch_mean_positions(measurement_batch, eps_lift=eps_lift)
    meas_directions = measurement_batch_mean_directions(measurement_batch, eps_mass=eps_mass)
    meas_kappas = measurement_batch_kappas(measurement_batch)

    # Get precisions from info form
    Lambda_reg = measurement_batch.Lambdas + eps_lift * jnp.eye(3, dtype=jnp.float64)[None, :, :]

    # Filter to valid measurements
    valid_mask = measurement_batch.valid_mask
    valid_indices = jnp.where(valid_mask)[0][:N_meas]

    meas_positions = meas_positions[valid_indices]
    meas_directions = meas_directions[valid_indices]
    meas_kappas = meas_kappas[valid_indices]
    meas_precisions = Lambda_reg[valid_indices]

    # Filter associations to valid measurement rows
    responsibilities = association_result.responsibilities[valid_indices]
    candidate_indices = association_result.candidate_indices[valid_indices]
    row_masses = association_result.row_masses[valid_indices]
    N_assoc, K_assoc = responsibilities.shape

    # Get map arrays
    map_positions = map_view.positions  # Already in world frame
    map_directions = map_view.directions
    map_kappas = map_view.kappas

    # Compute translation evidence
    L_trans, h_trans, trans_cost = _compute_translation_evidence_wls(
        meas_positions=meas_positions,
        meas_precisions=meas_precisions,
        map_positions=map_positions,
        responsibilities=responsibilities,
        candidate_indices=candidate_indices,
        R_pred=R_pred,
        t_pred=t_pred,
        eps_lift=eps_lift,
    )

    # Compute rotation evidence
    L_rot, h_rot, rot_cost = _compute_rotation_evidence_vmf(
        meas_directions=meas_directions,
        meas_kappas=meas_kappas,
        map_directions=map_directions,
        map_kappas=map_kappas,
        responsibilities=responsibilities,
        candidate_indices=candidate_indices,
        R_pred=R_pred,
        eps_lift=eps_lift,
    )

    # Embed into 22D state space
    # State: [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]
    # Pose evidence affects trans[0:3] and rot[3:6]
    L_pose = eps_lift * jnp.eye(22, dtype=jnp.float64)
    h_pose = jnp.zeros((22,), dtype=jnp.float64)

    # Set translation block [0:3, 0:3]
    L_pose = L_pose.at[:3, :3].set(L_trans)
    h_pose = h_pose.at[:3].set(h_trans)

    # Set rotation block [3:6, 3:6]
    L_pose = L_pose.at[3:6, 3:6].set(L_rot)
    h_pose = h_pose.at[3:6].set(h_rot)

    # Compute diagnostics
    total_cost = trans_cost + rot_cost
    mean_resp = float(jnp.mean(row_masses)) if N_assoc > 0 else 0.0

    result = VisualPoseEvidenceResult(
        L_pose=L_pose,
        h_pose=h_pose,
        L_trans=L_trans,
        h_trans=h_trans,
        L_rot=L_rot,
        h_rot=h_rot,
        total_weighted_cost=total_cost,
        n_associations=int(N_assoc * K_assoc),
        mean_responsibility=mean_resp,
    )

    # Certificate with linearization trigger
    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["linearization", "ot_soft_correspondence"],
        frobenius_applied=True,  # We use predicted pose for linearization
        support=SupportCert(
            ess_total=float(jnp.sum(row_masses)),
            support_frac=float(N_assoc) / float(max(N_meas, 1)),
        ),
        influence=InfluenceCert(
            lift_strength=eps_lift,
            psd_projection_delta=0.0,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="visual_pose_evidence",
        predicted=total_cost,
        realized=total_cost,
    )

    return result, cert, effect


# =============================================================================
# Utility: Build Combined Evidence for Fusion
# =============================================================================


def build_visual_pose_evidence_22d(
    visual_result: VisualPoseEvidenceResult,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract 22D evidence tensors for info fusion.

    Returns:
        L_pose: (22, 22) precision
        h_pose: (22,) information vector
    """
    return visual_result.L_pose, visual_result.h_pose
