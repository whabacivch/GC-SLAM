"""
Anchor Manager - Lifecycle and Budget Management.

Handles:
- Anchor creation via stochastic birth model
- Anchor descriptor updates
- Budget enforcement with Frobenius correction
- Anchor storage and retrieval

ALL mathematical operations call models.birth and operators.third_order_correct.
NO math duplication, NO heuristic thresholds.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from fl_slam_poc.backend import (
    StochasticBirthModel,
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
)
from fl_slam_poc.common.dirichlet_geom import third_order_correct
from fl_slam_poc.common.op_report import OpReport


def _vec_stats(vec: np.ndarray) -> dict:
    """Statistics summary for OpReport."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "norm": float(np.linalg.norm(v)),
    }


@dataclass
class Anchor:
    """Anchor landmark with NIG descriptor model."""
    anchor_id: int
    stamp_sec: float
    pose: np.ndarray  # SE(3) pose in rotation vector representation
    desc_model: NIGModel  # Uses models.nig (exact)
    weight: float
    depth_points: np.ndarray  # 3D points for ICP
    frame_id: str


class AnchorManager:
    """
    Manages anchor landmark lifecycle.
    
    Uses models.birth for stochastic birth (exact Poisson model).
    Uses operators.third_order_correct for budget projections (Frobenius correction).
    """
    
    def __init__(self, 
                 birth_model: StochasticBirthModel,
                 anchor_id_offset: int = 0):
        """
        Args:
            birth_model: Stochastic birth model (from models.birth)
            anchor_id_offset: Offset for anchor IDs
        """
        self.birth_model = birth_model
        self.anchor_id_offset = anchor_id_offset
        self.anchors: List[Anchor] = []
        self.anchor_counter = 0
        self.base_weight = 0.0  # Base component weight
    
    def update_anchors(self,
                       descriptor: np.ndarray,
                       responsibilities: Dict[int, float],
                       r_new: float,
                       obs_weight: float):
        """
        Update anchor beliefs with soft association.
        
        Uses models.nig.update (exact Bayesian update).
        
        Also updates birth model concentration_scale based on responsibility entropy
        (Self-Adaptive Systems compliant: behavior emerges from posterior uncertainty).
        """
        for anchor in self.anchors:
            r = responsibilities.get(anchor.anchor_id, 0.0)
            w = obs_weight * float(r)
            anchor.desc_model.update(descriptor, w)  # models.nig (exact)
            anchor.weight += w
        
        # Update base component weight
        r_new_eff = obs_weight * float(r_new)
        self.base_weight += r_new_eff
        
        # Update birth model concentration from responsibility entropy
        # This makes birth intensity adaptive (Self-Adaptive Systems Guide Section 3)
        if len(self.anchors) > 0:
            # Build full responsibility array including new component
            all_responsibilities = np.array(
                [responsibilities.get(a.anchor_id, 0.0) for a in self.anchors] + [r_new],
                dtype=float
            )
            self.birth_model.update_concentration_from_responsibilities(
                all_responsibilities, 
                len(self.anchors)
            )
        
        return r_new_eff
    
    def should_birth_anchor(self, r_new_eff: float) -> bool:
        """
        Decide whether to create new anchor via stochastic birth.
        
        Uses models.birth.sample_birth (exact Poisson sampling).
        """
        return self.birth_model.sample_birth(r_new_eff)
    
    def get_birth_probability(self, r_new_eff: float) -> float:
        """Get birth probability for logging."""
        return self.birth_model.birth_probability(r_new_eff)
    
    def create_anchor(self,
                    stamp_sec: float,
                    pose: np.ndarray,
                    descriptor: np.ndarray,
                    desc_model: NIGModel,
                    r_new_eff: float,
                    points: np.ndarray,
                    frame_id: str) -> int:
        """
        Create new anchor.
        
        Args:
            stamp_sec: Timestamp in seconds
            pose: SE(3) pose (rotation vector)
            descriptor: Descriptor vector
            desc_model: Initial NIG model (from global model copy)
            r_new_eff: Effective new-component responsibility
            points: 3D points for ICP
            frame_id: Reference frame
        
        Returns:
            anchor_id: ID of created anchor
        """
        anchor_id = self.anchor_id_offset + self.anchor_counter
        
        anchor = Anchor(
            anchor_id=anchor_id,
            stamp_sec=stamp_sec,
            pose=pose.copy(),
            desc_model=desc_model.copy() if desc_model is not None else NIGModel.from_prior(
                mu=descriptor, kappa=NIG_PRIOR_KAPPA, alpha=NIG_PRIOR_ALPHA, beta=NIG_PRIOR_BETA),
            weight=r_new_eff,
            depth_points=points.copy(),
            frame_id=frame_id
        )
        
        # Update descriptor model with initial observation
        anchor.desc_model.update(descriptor, r_new_eff)
        
        self.anchors.append(anchor)
        self.anchor_counter += 1
        
        return anchor_id
    
    def apply_budget(self, budget: int) -> Optional[OpReport]:
        """
        Enforce anchor budget with KL-minimizing projection.
        
        Uses operators.third_order_correct for Frobenius correction (exact).
        Returns OpReport for logging.
        """
        if budget <= 0 or len(self.anchors) <= budget:
            return None
        
        # Collect weights
        weights = {a.anchor_id: float(a.weight) for a in self.anchors}
        total_weight = sum(weights.values())
        
        if total_weight <= 0.0:
            return None
        
        # Select top-budget anchors by weight (model-intrinsic objective)
        sorted_ids = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        selected_ids = [aid for aid, _ in sorted_ids[:budget]]
        dropped_ids = [aid for aid, _ in sorted_ids[budget:]]
        dropped_mass = sum(weights[i] for i in dropped_ids)
        
        # Frobenius-corrected projection (exact third-order)
        ids = list(weights.keys())
        p = np.array([weights[i] for i in ids], dtype=float)
        total = float(np.sum(p))
        p = p / total
        
        mask = np.array([1.0 if i in selected_ids else 0.0 for i in ids], dtype=float)
        p_sel = p * mask
        sel_sum = float(np.sum(p_sel))
        if sel_sum <= 0.0:
            return None
        
        q = p_sel / sel_sum
        tau = total
        alpha_before = tau * p
        alpha_after = tau * q
        delta = alpha_after - alpha_before
        
        # CRITICAL: Use operators.third_order_correct (Frobenius correction)
        delta_corr = third_order_correct(alpha_before, delta)
        
        alpha_corr = np.maximum(alpha_before + delta_corr, 1e-12)
        q_corr = alpha_corr / float(np.sum(alpha_corr))
        
        # Update base weight with dropped mass
        self.base_weight += dropped_mass
        
        # Keep only selected anchors with corrected weights
        id_to_anchor = {a.anchor_id: a for a in self.anchors}
        self.anchors = [id_to_anchor[i] for i in selected_ids]
        
        for anchor in self.anchors:
            idx = ids.index(anchor.anchor_id)
            anchor.weight = total * float(q_corr[idx])
        
        # Create OpReport for logging
        report = OpReport(
            name="AnchorBudgetProjection",
            exact=False,
            approximation_triggers=["BudgetTruncation"],
            family_in="DescriptorMixture",
            family_out="DescriptorMixture",
            closed_form=True,
            frobenius_applied=True,
            frobenius_operator="dirichlet_third_order",
            frobenius_delta_norm=float(np.linalg.norm(delta_corr - delta)),
            frobenius_input_stats={"alpha": _vec_stats(alpha_before), "delta": _vec_stats(delta)},
            frobenius_output_stats={"delta_corr": _vec_stats(delta_corr)},
            metrics={"dropped": len(dropped_ids), "budget": budget, "dropped_mass": dropped_mass},
        )
        
        return report
    
    def get_all_anchors(self) -> List[Anchor]:
        """Get all active anchors."""
        return self.anchors
    
    def get_anchor_count(self) -> int:
        """Get number of active anchors."""
        return len(self.anchors)
    
    def get_base_weight(self) -> float:
        """Get base component weight."""
        return self.base_weight
