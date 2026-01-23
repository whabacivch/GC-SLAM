"""
RGB-D evidence processing and dense module management.

Handles parsing, association, and fusion of RGB-D evidence into dense 3D modules.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, List

import numpy as np

from fl_slam_poc.backend.fusion.gaussian_info import mean_cov
from fl_slam_poc.backend.state.modules import Dense3DModule
from fl_slam_poc.common import constants
from fl_slam_poc.common.dirichlet_geom import third_order_correct
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.utils import vec_stats

if TYPE_CHECKING:
    from fl_slam_poc.backend.backend_node import FLBackend


def parse_rgbd_evidence(msg_data: str) -> List[dict]:
    """
    Parse RGB-D evidence JSON payload.
    
    Payload schema:
      {"evidence": [ {position_L, position_h, color_L, color_h, normal_theta, alpha_mean, alpha_var}, ... ]}
    
    Args:
        msg_data: JSON string from /sim/rgbd_evidence topic
        
    Returns:
        List of evidence dictionaries with numpy arrays
        
    Raises:
        ValueError: If payload is invalid
    """
    payload = json.loads(msg_data)
    evidence_in = payload.get("evidence", [])
    if not isinstance(evidence_in, list) or len(evidence_in) == 0:
        return []
    
    evidence_list: List[dict] = []
    for ev in evidence_in:
        L = np.asarray(ev["position_L"], dtype=float)
        h = np.asarray(ev["position_h"], dtype=float).reshape(-1)
        out = {"position_L": L, "position_h": h}
        
        if "color_L" in ev and "color_h" in ev:
            out["color_L"] = np.asarray(ev["color_L"], dtype=float)
            out["color_h"] = np.asarray(ev["color_h"], dtype=float).reshape(-1)
        if "normal_theta" in ev:
            out["normal_theta"] = np.asarray(ev["normal_theta"], dtype=float).reshape(-1)
        if "alpha_mean" in ev:
            out["alpha_mean"] = float(ev["alpha_mean"])
        if "alpha_var" in ev:
            out["alpha_var"] = float(ev["alpha_var"])
        
        evidence_list.append(out)
    
    return evidence_list


def _dense_responsibilities(
    backend: "FLBackend",
    mu_obs: np.ndarray,
) -> tuple[dict[int, float], float, bool, str]:
    """
    Compute soft association responsibilities for dense evidence.

    Returns:
        (responsibilities, r_new, domain_projection, projection_reason)
    """
    if len(backend.anchors) == 0:
        return {}, 1.0, False, ""

    sigma = float(backend.dense_association_radius)
    domain_projection = False
    projection_reason = ""
    if sigma <= 0.0:
        sigma = float(constants.DENSE_ASSOCIATION_RADIUS_DEFAULT)
        domain_projection = True
        projection_reason = "invalid_sigma"

    responsibilities: dict[int, float] = {}
    total = float(constants.DENSE_NEW_COMPONENT_WEIGHT_PRIOR)
    sigma2 = sigma * sigma

    for anchor_id, (mu_anchor, _, _, _, _) in backend.anchors.items():
        dist = float(np.linalg.norm(mu_anchor[:3] - mu_obs[:3]))
        likelihood = np.exp(-dist * dist / (2.0 * sigma2))
        if anchor_id in backend.sparse_anchors:
            prior_mass = float(backend.sparse_anchors[anchor_id].mass)
        else:
            prior_mass = float(constants.MODULE_MASS_PRIOR)
        w = prior_mass * likelihood
        responsibilities[anchor_id] = w
        total += w

    if total < constants.RESPONSIBILITY_MASS_FLOOR:
        n = len(responsibilities) + 1
        if n <= 0:
            return {}, 1.0, True, "empty_mixture"
        uniform = 1.0 / n
        return {aid: uniform for aid in responsibilities}, uniform, True, "mass_floor"

    normed = {aid: w / total for aid, w in responsibilities.items()}
    r_new = float(constants.DENSE_NEW_COMPONENT_WEIGHT_PRIOR) / total
    return normed, r_new, domain_projection, projection_reason


def process_rgbd_evidence(
    backend: "FLBackend",
    evidence_list: List[dict],
) -> None:
    """
    Process RGB-D evidence from frontend.
    
    Strategy (ORDER-INVARIANT, SOFT ASSOCIATION):
    - Compute responsibilities over existing anchors + new component
    - Fuse evidence fractionally by responsibility (no hard gates)
    - New component responsibility spawns a dense module
    
    This ensures: Evidence_A + Evidence_B + Anchor_1 + Anchor_2 
               == Evidence_B + Evidence_A + Anchor_2 + Anchor_1
    
    Args:
        backend: Backend node instance
        evidence_list: List of evidence dictionaries
    """
    for evidence in evidence_list:
        mu_obs, _ = mean_cov(evidence["position_L"], evidence["position_h"])
        responsibilities, r_new, domain_projection, proj_reason = _dense_responsibilities(
            backend, mu_obs
        )

        if domain_projection:
            from fl_slam_poc.backend.diagnostics import publish_report
            publish_report(backend, OpReport(
                name="DenseAssociationDomainProjection",
                exact=True,
                family_in="DenseAssociation",
                family_out="DenseAssociation",
                closed_form=True,
                domain_projection=True,
                metrics={
                    "reason": proj_reason,
                    "n_anchors": len(backend.anchors),
                    "sigma_m": float(backend.dense_association_radius),
                },
                notes="Association responsibilities projected to valid domain.",
            ), backend.pub_report)

        # Fuse to sparse anchors with fractional responsibilities
        for anchor_id, resp in responsibilities.items():
            anchor_data = backend.anchors[anchor_id]
            mu_a, cov_a, _, _, points_a = anchor_data

            if anchor_id not in backend.sparse_anchors:
                from fl_slam_poc.backend.state.modules import SparseAnchorModule
                backend.sparse_anchors[anchor_id] = SparseAnchorModule(
                    anchor_id, mu_a, cov_a, points_a
                )

            backend.sparse_anchors[anchor_id].fuse_rgbd_position(
                evidence["position_L"], evidence["position_h"], weight=float(resp)
            )

        # New-component responsibility spawns a dense module
        if r_new > 0.0:
            add_dense_module(backend, evidence, weight=float(r_new))


def add_dense_module(backend: "FLBackend", evidence: dict, weight: float = 1.0) -> int:
    """
    Add a new dense module from RGB-D evidence.
    
    Args:
        backend: Backend node instance
        evidence: Evidence dictionary with position_L, position_h, etc.
        weight: Evidence weight (responsibility)
        
    Returns:
        Module ID of the newly created module
    """
    if len(backend.dense_modules) >= backend.max_dense_modules:
        # Cull modules by budgeted recomposition (posterior mass)
        cull_dense_modules(backend, keep_fraction=backend.dense_module_keep_fraction)
    
    mu, cov = mean_cov(evidence["position_L"], evidence["position_h"])
    mod = Dense3DModule(backend.next_dense_id, mu, cov)
    mod.update_from_evidence(evidence, weight=weight)
    
    backend.dense_modules[backend.next_dense_id] = mod
    backend.next_dense_id += 1
    
    return mod.module_id


def cull_dense_modules(
    backend: "FLBackend",
    keep_fraction: float,
) -> None:
    """
    Budgeted recomposition for dense module atlas by posterior mass.
    
    Args:
        backend: Backend node instance
        keep_fraction: Fraction of modules to keep (default from constants)
    """
    if len(backend.dense_modules) == 0:
        return
    
    if keep_fraction <= 0.0 or keep_fraction >= 1.0:
        from fl_slam_poc.backend.diagnostics import publish_report
        publish_report(backend, OpReport(
            name="DenseModuleKeepFractionInvalid",
            exact=True,
            family_in="DenseModuleAtlas",
            family_out="DenseModuleAtlas",
            closed_form=True,
            domain_projection=True,
            metrics={
                "keep_fraction_in": float(keep_fraction),
                "expected_range": "(0, 1)",
            },
            notes="Invalid keep_fraction for dense module culling; aborting.",
        ), backend.pub_report)
        raise ValueError(
            f"Dense module cull: keep_fraction must be in (0, 1), got {keep_fraction}"
        )

    # Sort by posterior mass (model-intrinsic objective)
    sorted_mods = sorted(
        backend.dense_modules.items(),
        key=lambda x: x[1].mass,
        reverse=True,
    )

    keep_count = max(1, int(len(sorted_mods) * keep_fraction))
    selected = sorted_mods[:keep_count]
    dropped = sorted_mods[keep_count:]
    if not dropped:
        return

    # Frobenius-corrected mass renormalization (Dirichlet truncation)
    ids = [mod_id for mod_id, _ in sorted_mods]
    alpha = np.array([mod.mass for _, mod in sorted_mods], dtype=float)
    total = float(np.sum(alpha))
    if total <= constants.WEIGHT_EPSILON:
        return

    p = alpha / total
    selected_ids = {mod_id for mod_id, _ in selected}
    mask = np.array([1.0 if mod_id in selected_ids else 0.0 for mod_id in ids], dtype=float)
    p_sel = p * mask
    sel_sum = float(np.sum(p_sel))
    if sel_sum <= constants.WEIGHT_EPSILON:
        return

    q = p_sel / sel_sum
    alpha_before = total * p
    alpha_after = total * q
    delta = alpha_after - alpha_before

    delta_corr = third_order_correct(alpha_before, delta)
    alpha_corr = np.maximum(alpha_before + delta_corr, constants.WEIGHT_EPSILON)
    q_corr = alpha_corr / float(np.sum(alpha_corr))

    # Update masses for retained modules
    for idx, mod_id in enumerate(ids):
        if mod_id in selected_ids:
            backend.dense_modules[mod_id].mass = float(alpha_corr[idx])

    remove_ids = [mod_id for mod_id, _ in dropped]
    for mod_id in remove_ids:
        del backend.dense_modules[mod_id]

    from fl_slam_poc.backend.diagnostics import publish_report
    total_before = float(np.sum(alpha_before))
    total_after = float(np.sum(alpha_corr))
    expected_drop_fraction = 1.0 - float(keep_fraction)
    realized_drop_fraction = float(len(remove_ids)) / float(len(sorted_mods))
    publish_report(backend, OpReport(
        name="DenseModuleBudgetedRecomposition",
        exact=False,
        approximation_triggers=["BudgetTruncation"],
        family_in="DenseModuleAtlas",
        family_out="DenseModuleAtlas",
        closed_form=True,
        frobenius_applied=True,
        frobenius_operator="dirichlet_third_order",
        frobenius_delta_norm=float(np.linalg.norm(delta_corr - delta)),
        frobenius_input_stats={"alpha": vec_stats(alpha_before), "delta": vec_stats(delta)},
        frobenius_output_stats={"delta_corr": vec_stats(delta_corr)},
        metrics={
            "kept": keep_count,
            "dropped": len(remove_ids),
            "keep_fraction": keep_fraction,
            "dropped_mass": float(np.sum(alpha[keep_count:])),
            "benefit_expected_drop_fraction": expected_drop_fraction,
            "benefit_realized_drop_fraction": realized_drop_fraction,
            "mass_total_before": total_before,
            "mass_total_after": total_after,
        },
        notes="Budgeted recomposition by posterior mass (approximation with Frobenius correction).",
    ), backend.pub_report)

    backend.get_logger().info(
        f"Culled {len(remove_ids)} dense modules, {len(backend.dense_modules)} remaining"
    )
