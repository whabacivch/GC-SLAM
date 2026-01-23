"""
Loop Processor - Data Association and Loop Factor Generation.

Handles:
- Fisher-Rao distance computation (soft association)
- Responsibility calculation (no heuristic gating)
- ICP registration with covariance
- Loop factor publishing
- Budget enforcement with Frobenius correction

ALL mathematical operations call operators/ directly.
NO math duplication, NO heuristic thresholds, NO approximations without Frobenius.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np

from fl_slam_poc.backend import (
    NIGModel,
    NIG_PRIOR_KAPPA,
    NIG_PRIOR_ALPHA,
    NIG_PRIOR_BETA,
    AdaptiveParameter,
)
from fl_slam_poc.frontend.scan.icp import (
    icp_3d,
    icp_information_weight,
    icp_covariance_tangent,
    transport_covariance_to_frame,
)
from fl_slam_poc.frontend.scan.pointcloud_gpu import GPUPointCloudProcessor, is_gpu_available
from fl_slam_poc.backend.fusion.gaussian_geom import gaussian_frobenius_correction
from fl_slam_poc.common.dirichlet_geom import third_order_correct
from fl_slam_poc.common.op_report import OpReport
from fl_slam_poc.common.geometry.se3_numpy import (
    rotmat_to_quat,
    rotvec_to_rotmat,
    se3_compose,
    se3_inverse,
)
from fl_slam_poc.common.utils import vec_stats
from fl_slam_poc.common import constants



class LoopProcessor:
    """
    Processes loop closures via soft association and ICP.
    
    Uses operators.information_distances for Fisher-Rao (exact).
    Uses operators.icp for registration (exact solver).
    Uses operators.gaussian_frobenius_correction for linearization (exact third-order).
    """
    
    def __init__(self,
                 fr_distance_scale: AdaptiveParameter,
                 icp_max_iter: AdaptiveParameter,
                 icp_tol: AdaptiveParameter,
                 icp_n_ref: float,
                 icp_sigma_mse: float,
                 use_gpu: bool = False,
                 gpu_device_index: int = 0,
                 gpu_fallback_to_cpu: bool = True,
                 voxel_size: float = 0.05,
                 max_correspondence_distance: float = 0.5,
                 use_3d_pointcloud: bool = False):
        """
        Args:
            fr_distance_scale: Adaptive FR distance scale (from models.adaptive)
            icp_max_iter: Adaptive ICP max iterations (from models.adaptive)
            icp_tol: Adaptive ICP tolerance (from models.adaptive)
            icp_n_ref: ICP reference point count for information weighting
            icp_sigma_mse: ICP MSE sigma for information weighting
            use_gpu: Enable GPU acceleration for ICP
            gpu_device_index: CUDA device index
            gpu_fallback_to_cpu: Fall back to CPU if GPU unavailable
            voxel_size: Voxel grid size for 3D point cloud filtering
            max_correspondence_distance: Max distance for ICP correspondences
            use_3d_pointcloud: Running in 3D point cloud mode
        """
        self.fr_distance_scale = fr_distance_scale
        self.icp_max_iter = icp_max_iter
        self.icp_tol = icp_tol
        self.icp_n_ref = icp_n_ref
        self.icp_sigma_mse = icp_sigma_mse
        
        # GPU configuration
        self.use_gpu = use_gpu
        self.use_3d_pointcloud = use_3d_pointcloud
        self.voxel_size = voxel_size
        self.max_correspondence_distance = max_correspondence_distance
        self._gpu_processor = None
        
        # Initialize GPU processor if enabled
        if use_gpu:
            if is_gpu_available():
                self._gpu_processor = GPUPointCloudProcessor(
                    voxel_size=voxel_size,
                    max_correspondence_distance=max_correspondence_distance,
                    device_index=gpu_device_index,
                    fallback_to_cpu=gpu_fallback_to_cpu
                )
            elif not gpu_fallback_to_cpu:
                raise RuntimeError("GPU requested but not available, and fallback disabled")
    
    def compute_responsibilities(self,
                                 descriptor: np.ndarray,
                                 anchors: List,
                                 global_model: Optional[NIGModel],
                                 base_weight: float) -> Tuple[Dict[int, float], float, bool]:
        """
        Compute soft association responsibilities via Fisher-Rao distances.
        
        Uses models.nig.fisher_rao_distance (exact).
        NO heuristic gating - all anchors receive responsibility.
        
        Args:
            descriptor: Query descriptor
            anchors: List of Anchor objects
            global_model: Global NIG model
            base_weight: Base component weight
        
        Returns:
            responsibilities: Dict of anchor responsibilities
            r_new: New-component responsibility
            domain_projection: True if uniform responsibilities were used due to near-zero mass
        """
        if len(anchors) == 0:
            # No anchors yet, everything goes to new component
            return {}, 1.0, False
        
        # Compute Fisher-Rao distances to each anchor (uses models.nig - exact)
        # Create temporary NIG model from current descriptor for comparison
        temp_model = NIGModel.from_prior(
            mu=np.zeros_like(descriptor, dtype=float),
            kappa=float(NIG_PRIOR_KAPPA),
            alpha=float(NIG_PRIOR_ALPHA),
            beta=float(NIG_PRIOR_BETA),
        )
        temp_model.update(descriptor, weight=1.0)
        
        distances = {}
        for anchor in anchors:
            dist = anchor.desc_model.fisher_rao_distance(temp_model)
            distances[anchor.anchor_id] = dist
        
        # Distance to global model (new component)
        if global_model is not None:
            new_comp_dist = global_model.fisher_rao_distance(temp_model)
        else:
            new_comp_dist = 1.0  # Default large distance
        
        # Convert distances to likelihoods via Gaussian kernel
        # p(x|anchor) ∝ exp(-d² / (2σ²))
        scale = self.fr_distance_scale.value()
        
        likelihoods = {}
        for anchor_id, dist in distances.items():
            likelihoods[anchor_id] = np.exp(-dist**2 / (2 * scale**2))
        
        new_comp_likelihood = np.exp(-new_comp_dist**2 / (2 * scale**2))
        
        # Compute responsibilities via mixture weights
        # r_k = (w_k * p(x|k)) / Σ(w_j * p(x|j))
        anchor_weights = {a.anchor_id: a.weight for a in anchors}
        
        weighted_likelihoods = {
            aid: anchor_weights[aid] * likelihoods[aid]
            for aid in likelihoods.keys()
        }
        weighted_new = base_weight * new_comp_likelihood
        
        total = sum(weighted_likelihoods.values()) + weighted_new
        
        domain_projection = False
        if total < constants.RESPONSIBILITY_MASS_FLOOR:
            # All distances very large, uniform responsibilities
            n = len(anchors) + 1
            responsibilities = {aid: 1.0/n for aid in likelihoods.keys()}
            new_comp_resp = 1.0/n
            domain_projection = True
        else:
            responsibilities = {
                aid: wl / total
                for aid, wl in weighted_likelihoods.items()
            }
            new_comp_resp = weighted_new / total
        
        return responsibilities, new_comp_resp, domain_projection
    
    def apply_loop_budget(self,
                         responsibilities: Dict[int, float],
                         budget: int) -> Tuple[Dict[int, float], Optional[OpReport]]:
        """
        Apply budget to loop factors with Frobenius correction.
        
        Uses operators.third_order_correct (exact).
        Returns (truncated_responsibilities, report).
        """
        if budget <= 0 or len(responsibilities) <= budget:
            return responsibilities, None
        
        # Select top-budget by responsibility (model-intrinsic objective)
        sorted_resp = sorted(responsibilities.items(), key=lambda x: x[1], reverse=True)
        selected = sorted_resp[:budget]
        dropped = sorted_resp[budget:]
        
        selected_ids = [aid for aid, _ in selected]
        dropped_mass = sum(r for _, r in dropped)
        
        # Frobenius-corrected renormalization (exact third-order)
        ids = list(responsibilities.keys())
        p = np.array([responsibilities[i] for i in ids], dtype=float)
        total = float(np.sum(p))
        p = p / total
        
        mask = np.array([1.0 if i in selected_ids else 0.0 for i in ids], dtype=float)
        p_sel = p * mask
        sel_sum = float(np.sum(p_sel))
        if sel_sum <= 0.0:
            return {}, None
        
        q = p_sel / sel_sum
        alpha_before = total * p
        alpha_after = total * q
        delta = alpha_after - alpha_before
        
        # CRITICAL: Use operators.third_order_correct (Frobenius correction)
        delta_corr = third_order_correct(alpha_before, delta)
        
        alpha_corr = np.maximum(alpha_before + delta_corr, constants.WEIGHT_EPSILON)
        q_corr = alpha_corr / float(np.sum(alpha_corr))
        
        # Reconstruct responsibilities
        truncated = {}
        for i, anchor_id in enumerate(ids):
            if anchor_id in selected_ids:
                truncated[anchor_id] = float(q_corr[i])
        
        # Create OpReport
        report = OpReport(
            name="LoopBudgetProjection",
            exact=False,
            approximation_triggers=["BudgetTruncation"],
            family_in="ResponsibilityMixture",
            family_out="ResponsibilityMixture",
            closed_form=True,
            frobenius_applied=True,
            frobenius_operator="dirichlet_third_order",
            frobenius_delta_norm=float(np.linalg.norm(delta_corr - delta)),
            frobenius_input_stats={"alpha": vec_stats(alpha_before), "delta": vec_stats(delta)},
            frobenius_output_stats={"delta_corr": vec_stats(delta_corr)},
            metrics={"dropped": len(dropped), "budget": budget, "dropped_mass": dropped_mass},
        )
        
        return truncated, report
    
    def preprocess_pointcloud(self, points: np.ndarray) -> np.ndarray:
        """
        Preprocess point cloud with voxel filtering (if in 3D mode).
        
        Uses GPU-accelerated filtering when available.
        
        Args:
            points: Input point cloud (N, 3)
        
        Returns:
            Filtered point cloud (M, 3) where M <= N
        """
        if not self.use_3d_pointcloud:
            return points
        
        if self._gpu_processor is not None:
            return self._gpu_processor.voxel_filter(points, self.voxel_size)
        else:
            # Simple CPU voxel filtering fallback
            if points.shape[0] == 0:
                return points
            voxel_indices = np.floor(points / self.voxel_size).astype(np.int32)
            _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
            return points[unique_idx]
    
    def run_icp(self,
                source_points: np.ndarray,
                target_points: np.ndarray,
                preprocess: bool = True) -> Optional[object]:
        """
        Run ICP registration.
        
        Uses GPU-accelerated ICP when available, falls back to operators.icp_3d.
        
        Args:
            source_points: Source point cloud
            target_points: Target point cloud
            preprocess: Apply voxel filtering before ICP (for 3D mode)
        
        Returns:
            icp_result: Object with transform, mse, iterations, converged, etc.
        """
        max_iter = int(self.icp_max_iter.value())
        tol = self.icp_tol.value()
        
        try:
            # Preprocess point clouds in 3D mode
            if preprocess:
                source_points = self.preprocess_pointcloud(source_points)
                target_points = self.preprocess_pointcloud(target_points)
            
            # Check for sufficient points
            if source_points.shape[0] < 3 or target_points.shape[0] < 3:
                return None
            
            # Initial transform guess (identity)
            init_transform = np.zeros(6, dtype=float)

            # Use GPU processor if available, otherwise CPU
            if self._gpu_processor is not None:
                icp_result = self._gpu_processor.icp(
                    source_points,
                    target_points,
                    init=init_transform,
                    max_iter=max_iter,
                    tol=tol,
                    max_correspondence_distance=self.max_correspondence_distance
                )
            else:
                # CRITICAL: Use operators.icp_3d (exact solver)
                icp_result = icp_3d(
                    source_points,
                    target_points,
                    init=init_transform,
                    max_iter=max_iter,
                    tol=tol
                )
            
            return icp_result
        
        except Exception as e:
            # ICP failure is domain constraint, not approximation
            return None
    
    def is_using_gpu(self) -> bool:
        """Check if GPU acceleration is active."""
        return self._gpu_processor is not None and self._gpu_processor.use_gpu
    
    def compute_loop_factor(self,
                           icp_result: object,
                           anchor_pose: np.ndarray,
                           obs_weight: float,
                           responsibility: float) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Compute loop factor from ICP result.
        
        Uses operators.icp_information_weight (exact).
        Uses operators.icp_covariance_tangent (exact).
        Uses operators.transport_covariance_to_frame (exact Adjoint transport).
        
        Args:
            icp_result: ICP result object
            anchor_pose: Anchor SE(3) pose
            obs_weight: Observation weight
            responsibility: Association responsibility
        
        Returns:
            (rel_pose, cov_transported, final_weight, info_weight)
        """
        # Compute information weight (exact formula)
        info_weight = icp_information_weight(
            icp_result.n_source,
            icp_result.n_target,
            icp_result.mse,
            n_ref=self.icp_n_ref,
            sigma_mse=self.icp_sigma_mse
        )
        
        final_weight = obs_weight * float(responsibility) * info_weight
        
        # Compute covariance in tangent space (exact)
        cov = icp_covariance_tangent(icp_result.src_transformed, icp_result.mse)
        
        # Transport covariance via Adjoint (exact SE(3) operation)
        cov_transported = transport_covariance_to_frame(cov, anchor_pose)
        
        return icp_result.transform, cov_transported, final_weight, info_weight
    
    def apply_frobenius_correction(self, transform: np.ndarray) -> Tuple[np.ndarray, dict]:
        """
        Apply Frobenius correction for ICP linearization.
        
        Uses operators.gaussian_frobenius_correction (exact third-order).
        
        Returns:
            (corrected_transform, frob_stats)
        """
        # CRITICAL: Use operators.gaussian_frobenius_correction
        corrected, frob_stats = gaussian_frobenius_correction(transform)
        return corrected, frob_stats
    
    def update_adaptive_params(self, icp_result: object):
        """Update adaptive ICP parameters."""
        self.icp_max_iter.update(float(icp_result.iterations))
        self.icp_tol.update(icp_result.mse)
    
    def compute_relative_transform(self, T_anchor: np.ndarray, T_current: np.ndarray) -> np.ndarray:
        """
        Compute relative transform: Z = T_anchor^{-1} ∘ T_current.
        
        Uses geometry.se3 (exact composition).
        """
        return se3_compose(se3_inverse(T_anchor), T_current)

    @staticmethod
    def rotvec_to_quaternion(rotvec: np.ndarray) -> tuple[float, float, float, float]:
        """Convert rotation vector to quaternion (x, y, z, w)."""
        rotvec = np.asarray(rotvec, dtype=float).reshape(-1)
        if rotvec.shape[0] != 3:
            raise ValueError(f"rotvec_to_quaternion: expected (3,), got {rotvec.shape}")
        R = rotvec_to_rotmat(rotvec)
        return rotmat_to_quat(R)
