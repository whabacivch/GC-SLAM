"""
Belief representation for Geometric Compositional SLAM v2.

The BeliefGaussianInfo dataclass represents a Gaussian belief
in information form on the augmented tangent space.

Uses JAX for all math operations.

=============================================================================
ORDERING CONVENTIONS (UNIFIED - all use [trans, rot] ordering!)
=============================================================================

1. SE(3) POSE (se3_jax format):
   6D vector: [trans(3), rot(3)] = [x, y, z, rx, ry, rz]
   Used by: se3_compose, se3_inverse, X_anchor, mean_world_pose()

2. GC STATE VECTOR (tangent space):
   22D vector: [trans(3), rot(3), vel(3), bg(3), ba(3), dt(1), ex(6)]
   Pose slice: [0:3] = translation, [3:6] = rotation
   Used by: L, h, z_lin, all evidence operators

3. ROS ODOM COVARIANCE:
   [x, y, z, roll, pitch, yaw] = [trans(0:3), rot(3:6)]
   No permutation needed - matches GC ordering!

CONVERSION FUNCTIONS (now identity - kept for backwards compatibility):
   pose_se3_to_z_delta(): returns input unchanged
   pose_z_to_se3_delta(): returns input unchanged

Reference: docs/FRAME_AND_QUATERNION_CONVENTIONS.md
=============================================================================
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import copy

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common.certificates import (
    CertBundle,
    ConditioningCert,
    InfluenceCert,
)
from fl_slam_poc.common.primitives import (
    domain_projection_psd,
    spd_cholesky_solve_lifted,
    spd_cholesky_inverse_lifted,
)
from fl_slam_poc.common.geometry import se3_jax


# =============================================================================
# Constants (from spec)
# =============================================================================

# Chart convention
CHART_ID_GC_RIGHT_01 = "GC-RIGHT-01"

# State dimensions
D_Z = 22  # Augmented tangent dimension
D_DESKEW = 22  # Deskew tangent dimension

# State slice indices (0-based)
# Using Python slice objects for clean indexing
# NEW CONVENTION: [trans, rot] ordering (same as se3_jax and ROS)
SLICE_TRANS = slice(0, 3)  # δt (translation)
SLICE_SO3 = slice(3, 6)  # δθ (rotation)
SLICE_VEL = slice(6, 9)  # δv (velocity)
SLICE_GYRO_BIAS = slice(9, 12)  # δbg (gyro bias)
SLICE_ACCEL_BIAS = slice(12, 15)  # δba (accel bias)
SLICE_TIME_OFFSET = slice(15, 16)  # δΔt (time offset)
SLICE_EXTRINSIC = slice(16, 22)  # δξLI (LiDAR-IMU extrinsic)

# Pose slice (translation + rotation) in the unified GC tangent ordering
# Per new convention: [δt, δθ] = [translation, rotation].
#
# This matches se3_jax and ROS conventions - no conversion needed!
SLICE_POSE = slice(0, 6)


# =============================================================================
# SE(3) Utilities using JAX (6D representation: [trans, rotvec])
# =============================================================================


def se3_identity() -> jnp.ndarray:
    """Return 6D identity pose [0,0,0,0,0,0]."""
    return jnp.zeros(6, dtype=jnp.float64)


def se3_from_rotvec_trans(rotvec: jnp.ndarray, trans: jnp.ndarray) -> jnp.ndarray:
    """
    Construct 6D pose from rotation vector and translation.
    
    Args:
        rotvec: Rotation vector (3,)
        trans: Translation vector (3,)
        
    Returns:
        6D pose [trans, rotvec]
    """
    rotvec = jnp.asarray(rotvec, dtype=jnp.float64)
    trans = jnp.asarray(trans, dtype=jnp.float64)
    return jnp.concatenate([trans, rotvec])


def se3_to_rotvec_trans(pose: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extract rotation vector and translation from 6D pose.
    
    Args:
        pose: 6D pose [trans, rotvec]
        
    Returns:
        Tuple of (rotvec, trans)
    """
    pose = jnp.asarray(pose, dtype=jnp.float64)
    trans = pose[:3]
    rotvec = pose[3:6]
    return rotvec, trans


def pose_z_to_se3_delta(delta_pose_z: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a GC-ordered pose increment to se3_jax ordering.

    IDENTITY FUNCTION (kept for backwards compatibility).
    Both GC and se3_jax now use [trans(3), rot(3)] ordering.
    """
    delta_pose_z = jnp.asarray(delta_pose_z, dtype=jnp.float64).reshape(-1)
    if delta_pose_z.shape[0] != 6:
        raise ValueError(f"delta_pose_z must be (6,), got {delta_pose_z.shape}")
    return delta_pose_z  # Identity - orderings now match


def pose_se3_to_z_delta(delta_pose_se3: jnp.ndarray) -> jnp.ndarray:
    """
    Convert a se3_jax-ordered pose increment to GC tangent ordering.

    IDENTITY FUNCTION (kept for backwards compatibility).
    Both se3_jax and GC now use [trans(3), rot(3)] ordering.
    """
    delta_pose_se3 = jnp.asarray(delta_pose_se3, dtype=jnp.float64).reshape(-1)
    if delta_pose_se3.shape[0] != 6:
        raise ValueError(f"delta_pose_se3 must be (6,), got {delta_pose_se3.shape}")
    return delta_pose_se3  # Identity - orderings now match


@jax.jit
def se3_compose(pose1: jnp.ndarray, pose2: jnp.ndarray) -> jnp.ndarray:
    """Compose two 6D SE(3) poses: pose1 @ pose2."""
    return se3_jax.se3_compose(pose1, pose2)


@jax.jit
def se3_inverse(pose: jnp.ndarray) -> jnp.ndarray:
    """Inverse of 6D SE(3) pose."""
    return se3_jax.se3_inverse(pose)


@jax.jit
def se3_exp(xi: jnp.ndarray) -> jnp.ndarray:
    """
    Exponential map from se(3) to SE(3).
    
    Args:
        xi: 6D tangent vector [trans, rotvec]
        
    Returns:
        6D SE(3) pose
    """
    return se3_jax.se3_exp(xi)


@jax.jit
def se3_log(pose: jnp.ndarray) -> jnp.ndarray:
    """
    Logarithm map from SE(3) to se(3).
    
    Args:
        pose: 6D SE(3) pose
        
    Returns:
        6D tangent vector
    """
    # Full SE(3) Log: xi = [rho;phi] with rho = V(phi)^{-1} t (no stub).
    return se3_jax.se3_log(pose)


# =============================================================================
# Belief Dataclass
# =============================================================================


@dataclass
class BeliefGaussianInfo:
    """
    Gaussian belief on augmented tangent space in information form.
    
    Represents a Gaussian distribution:
        p(δz) ∝ exp(-0.5 * (δz - δz*)^T L (δz - δz*))
        
    where δz* = solve(L, h) is the MAP increment.
    
    All operations use the declared solve:
        δz* = (L + eps_lift * I)^{-1} h
        
    Reference: docs/GEOMETRIC_COMPOSITIONAL_INTERFACE_SPEC.md Section 2.1
    
    Attributes:
        chart_id: Must be "GC-RIGHT-01"
        anchor_id: Local chart instance id (stable within hypothesis)
        X_anchor: SE3 anchor pose as 6D vector [trans, rotvec]
        stamp_sec: Timestamp in seconds
        z_lin: Linearization point in chart coordinates (D_Z,)
        L: Information matrix (D_Z, D_Z) symmetric PSD
        h: Information vector (D_Z,)
        cert: Certificate bundle
    """
    chart_id: str
    anchor_id: str
    X_anchor: jnp.ndarray  # (6,) SE3 as [trans, rotvec]
    stamp_sec: float
    z_lin: jnp.ndarray  # (D_Z,)
    L: jnp.ndarray  # (D_Z, D_Z)
    h: jnp.ndarray  # (D_Z,)
    cert: CertBundle

    def __post_init__(self):
        """Validate belief dimensions and chart."""
        # Validate chart_id
        if self.chart_id != CHART_ID_GC_RIGHT_01:
            raise ValueError(
                f"Invalid chart_id: {self.chart_id}, expected {CHART_ID_GC_RIGHT_01}"
            )
        
        # Convert to JAX arrays
        self.z_lin = jnp.asarray(self.z_lin, dtype=jnp.float64)
        self.L = jnp.asarray(self.L, dtype=jnp.float64)
        self.h = jnp.asarray(self.h, dtype=jnp.float64)
        self.X_anchor = jnp.asarray(self.X_anchor, dtype=jnp.float64)
        
        # Validate dimensions
        if self.z_lin.shape != (D_Z,):
            raise ValueError(f"z_lin must be ({D_Z},), got {self.z_lin.shape}")
        if self.L.shape != (D_Z, D_Z):
            raise ValueError(f"L must be ({D_Z}, {D_Z}), got {self.L.shape}")
        if self.h.shape != (D_Z,):
            raise ValueError(f"h must be ({D_Z},), got {self.h.shape}")
        if self.X_anchor.shape != (6,):
            raise ValueError(f"X_anchor must be (6,), got {self.X_anchor.shape}")

    @classmethod
    def create_prior(
        cls,
        anchor_id: str,
        X_anchor: jnp.ndarray,
        stamp_sec: float,
        mean: jnp.ndarray,
        cov: jnp.ndarray,
        eps_psd: float = 1e-12,
        eps_lift: float = 1e-9,
    ) -> "BeliefGaussianInfo":
        """
        Create belief from mean and covariance (moment form).
        
        Converts to information form with domain projections.
        
        Args:
            anchor_id: Local chart instance id
            X_anchor: SE3 anchor pose (6D)
            stamp_sec: Timestamp
            mean: Mean vector (D_Z,)
            cov: Covariance matrix (D_Z, D_Z)
            eps_psd: PSD projection epsilon
            eps_lift: Solve lift epsilon
            
        Returns:
            BeliefGaussianInfo in information form
        """
        mean = jnp.asarray(mean, dtype=jnp.float64)
        cov = jnp.asarray(cov, dtype=jnp.float64)
        
        # Project covariance to PSD
        cov_psd_result = domain_projection_psd(cov, eps_psd)
        cov_psd = cov_psd_result.M_psd
        
        # Invert to get information matrix
        L, lift_strength = spd_cholesky_inverse_lifted(cov_psd, eps_lift)
        
        # Project L to PSD (should already be, but for consistency)
        L_psd_result = domain_projection_psd(L, eps_psd)
        L_psd = L_psd_result.M_psd
        
        # Compute information vector
        h = L_psd @ mean
        
        # Create certificate
        cert = CertBundle.create_approx(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id=anchor_id,
            triggers=["MomentToInfo"],
            conditioning=ConditioningCert(
                eig_min=L_psd_result.conditioning.eig_min,
                eig_max=L_psd_result.conditioning.eig_max,
                cond=L_psd_result.conditioning.cond,
                near_null_count=L_psd_result.conditioning.near_null_count,
            ),
            influence=InfluenceCert(
                lift_strength=lift_strength,
                psd_projection_delta=cov_psd_result.projection_delta + L_psd_result.projection_delta,
                mass_epsilon_ratio=0.0,
                anchor_drift_rho=0.0,
                dt_scale=1.0,
                extrinsic_scale=1.0,
                trust_alpha=1.0,
            ),
        )
        
        return cls(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id=anchor_id,
            X_anchor=jnp.asarray(X_anchor, dtype=jnp.float64),
            stamp_sec=stamp_sec,
            z_lin=mean,  # Linearization point is the mean
            L=L_psd,
            h=h,
            cert=cert,
        )

    @classmethod
    def create_identity_prior(
        cls,
        anchor_id: str,
        stamp_sec: float,
        prior_precision: float = 1e-6,
    ) -> "BeliefGaussianInfo":
        """
        Create uninformative prior (near-zero precision).
        
        Args:
            anchor_id: Local chart instance id
            stamp_sec: Timestamp
            prior_precision: Small precision value
            
        Returns:
            BeliefGaussianInfo with identity anchor and small L
        """
        X_anchor = se3_identity()
        z_lin = jnp.zeros(D_Z, dtype=jnp.float64)
        L = prior_precision * jnp.eye(D_Z, dtype=jnp.float64)
        h = jnp.zeros(D_Z, dtype=jnp.float64)
        
        cert = CertBundle.create_exact(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id=anchor_id,
            conditioning=ConditioningCert(
                eig_min=prior_precision,
                eig_max=prior_precision,
                cond=1.0,
                near_null_count=D_Z,
            ),
        )
        
        return cls(
            chart_id=CHART_ID_GC_RIGHT_01,
            anchor_id=anchor_id,
            X_anchor=X_anchor,
            stamp_sec=stamp_sec,
            z_lin=z_lin,
            L=L,
            h=h,
            cert=cert,
        )

    def mean_increment(self, eps_lift: float = 1e-9) -> jnp.ndarray:
        """
        Compute MAP increment δz* = (L + eps_lift * I)^{-1} h.
        
        This is the declared solve per spec Section 2.1.
        
        Args:
            eps_lift: Lift epsilon
            
        Returns:
            Mean increment (D_Z,)
        """
        result = spd_cholesky_solve_lifted(self.L, self.h, eps_lift)
        return result.x

    def to_moments(self, eps_lift: float = 1e-9) -> tuple[jnp.ndarray, jnp.ndarray, float]:
        """
        Convert to moment form (mean, covariance).
        
        Always applies lifted solve.
        
        Args:
            eps_lift: Lift epsilon
            
        Returns:
            Tuple of (mean, covariance, lift_strength)
        """
        # Compute mean
        mean = self.mean_increment(eps_lift)
        
        # Compute covariance as inverse of (L + eps_lift * I)
        cov, lift_strength = spd_cholesky_inverse_lifted(self.L, eps_lift)
        
        return mean, cov, lift_strength

    def world_pose(self, eps_lift: float = 1e-9) -> jnp.ndarray:
        """
        Compute world pose from anchor and mean increment.
        
        X_world = X_anchor @ Exp(δξ_pose)
        
        where δξ_pose is the pose slice of the mean increment.
        
        Args:
            eps_lift: Lift epsilon
            
        Returns:
            6D world pose
        """
        delta_z = self.mean_increment(eps_lift)
        delta_pose_z = delta_z[SLICE_POSE]
        delta_pose_se3 = pose_z_to_se3_delta(delta_pose_z)
        return se3_compose(self.X_anchor, se3_exp(delta_pose_se3))

    def mean_world_pose(self, eps_lift: float = 1e-9) -> jnp.ndarray:
        """
        Backwards-compatible alias for world pose.

        Returns:
            6D world pose in se3_jax ordering: [trans, rotvec]
        """
        return self.world_pose(eps_lift=eps_lift)

    def copy(self) -> "BeliefGaussianInfo":
        """Create a deep copy of this belief."""
        return BeliefGaussianInfo(
            chart_id=self.chart_id,
            anchor_id=self.anchor_id,
            X_anchor=jnp.array(self.X_anchor),
            stamp_sec=self.stamp_sec,
            z_lin=jnp.array(self.z_lin),
            L=jnp.array(self.L),
            h=jnp.array(self.h),
            cert=copy.deepcopy(self.cert),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "chart_id": self.chart_id,
            "anchor_id": self.anchor_id,
            "X_anchor": self.X_anchor.tolist(),
            "stamp_sec": self.stamp_sec,
            "z_lin": self.z_lin.tolist(),
            "L_trace": float(jnp.trace(self.L)),
            "h_norm": float(jnp.linalg.norm(self.h)),
            "cert": self.cert.to_dict(),
        }


# =============================================================================
# Hypothesis Container
# =============================================================================


@dataclass
class HypothesisSet:
    """
    Set of K_HYP hypotheses with weights.
    
    Maintains fixed number of hypotheses with weight floor enforcement.
    """
    hypotheses: list[BeliefGaussianInfo]
    weights: jnp.ndarray  # (K_HYP,)
    
    K_HYP: int = 4
    HYP_WEIGHT_FLOOR: float = 0.0025  # 0.01 / K_HYP

    def __post_init__(self):
        """Validate hypothesis set."""
        if len(self.hypotheses) != self.K_HYP:
            raise ValueError(
                f"Must have exactly {self.K_HYP} hypotheses, got {len(self.hypotheses)}"
            )
        
        self.weights = jnp.asarray(self.weights, dtype=jnp.float64)
        if self.weights.shape != (self.K_HYP,):
            raise ValueError(
                f"Weights must be ({self.K_HYP},), got {self.weights.shape}"
            )
        
        # Enforce weight floor (continuous, no branching)
        self._enforce_weight_floor()

    def _enforce_weight_floor(self) -> float:
        """
        Enforce weight floor and renormalize.
        
        Returns the total floor adjustment magnitude.
        """
        # Clamp weights to floor using JAX ops (branchless)
        weights_floored = jnp.maximum(self.weights, self.HYP_WEIGHT_FLOOR)
        floor_adjustment = float(jnp.sum(jnp.abs(weights_floored - self.weights)))
        
        # Renormalize
        self.weights = weights_floored / jnp.sum(weights_floored)
        
        return floor_adjustment

    @classmethod
    def create_uniform(
        cls,
        template_belief: BeliefGaussianInfo,
    ) -> "HypothesisSet":
        """
        Create hypothesis set with uniform weights from template.
        
        Args:
            template_belief: Template belief to copy for all hypotheses
            
        Returns:
            HypothesisSet with uniform weights
        """
        K_HYP = 4  # Use class default
        hypotheses = [template_belief.copy() for _ in range(K_HYP)]
        weights = jnp.ones(K_HYP, dtype=jnp.float64) / K_HYP
        return cls(hypotheses=hypotheses, weights=weights)
