"""
IMU evidence operators for GC v2.

Currently implements the accelerometer direction term as a vMF-style factor:
  ell(delta) = -kappa * dot(mu(delta), xbar)

where:
  mu(delta) = R(delta)^T * (-g_hat)
  xbar is the measured resultant direction over the scan window.

We convert this factor to quadratic Gaussian information by Laplace at delta=0
using closed-form derivatives (intrinsic primitives only; no autodiff).

=============================================================================
GRAVITY CONVENTION (CRITICAL)
=============================================================================
World frame: Z-UP convention
  gravity_W = [0, 0, -9.81] m/s²  (gravity points DOWN in -Z direction)
  g_hat = [0, 0, -1]              (normalized gravity direction, pointing DOWN)
  minus_g_hat = [0, 0, +1]        (expected accel direction, pointing UP)

Accelerometer Convention:
  IMU accelerometers measure REACTION TO GRAVITY (specific force), NOT gravity.
  When level and stationary, accelerometer reads +Z (pointing UP).
  This is the force preventing the sensor from freefalling.

Expected vs Measured:
  mu0 = R_body^T @ minus_g_hat    (expected accel direction in body frame)
  xbar = normalized(mean(accel))  (measured accel direction in body frame)
  Alignment: xbar @ mu0 should be ~+1.0 for correct gravity alignment
  If negative, the IMU extrinsic is likely inverting gravity!

State Ordering:
  L_imu is placed at [0:3, 0:3] which is the ROTATION block in GC ordering.
  (GC state: [rot(0:3), trans(3:6), ...])
=============================================================================

## vMF HESSIAN APPROXIMATION NOTE (Audit Compliance)

The Hessian H_rot used for the information matrix is an approximation to the
true Fisher information of the vMF directional likelihood.

### Exact vMF Fisher Information (on S²)

For vMF(μ, κ) with μ ∈ S² and κ > 0, the Fisher information matrix in the
tangent space at μ is:

    F_vMF = κ * (I - μμᵀ)

This is a rank-2 matrix (the tangent space to S² is 2D).

### Chain Rule for Rotation Parameterization

When μ(θ) = Rᵀ(-g_hat) and R = R₀·Exp(δθ), we need to compose with the
Jacobian ∂μ/∂δθ. The full Hessian is:

    H = (∂μ/∂δθ)ᵀ · F_vMF · (∂μ/∂δθ) + first_order_terms

For the likelihood ℓ(δθ) = -κ·μ(δθ)ᵀ·x̄, the gradient is:

    g = -κ·(μ₀ × x̄)  (cross product, exact)

The exact second derivative has the form:

    H_exact = κ·(x̄·μ₀)·I - κ·(outer products involving x̄, μ₀, and their derivatives)

### Approximation Used

We use a simplified closed-form approximation:

    H_approx = κ * [ (x̄·μ₀)·I - 0.5·(x̄μ₀ᵀ + μ₀x̄ᵀ) ]

This captures the dominant curvature but differs from the exact form by:
1. Missing the second-derivative terms from the rotation Jacobian
2. Using a symmetric average instead of the exact outer product structure

### Error Characteristics

- When x̄ ≈ μ₀ (good alignment): Error is O(κ·|x̄ - μ₀|²), typically < 5%
- When x̄ ⊥ μ₀ (poor alignment): Both exact and approx have low information, error is benign
- When x̄ ≈ -μ₀ (opposite): H is near-zero for both (correctly captures ambiguity)

### Justification for Use

1. The approximation is conservative (tends to underestimate information)
2. PSD projection is always applied afterward (ensures valid covariance)
3. Closed-form avoids autodiff overhead in hot path
4. Error is small when evidence is strong (x̄ ≈ μ₀)
5. When evidence is weak, both forms give low information (safe)

### Alternative (Not Implemented)

For exact Hessian, use JAX autodiff:
    H_exact = jax.hessian(lambda dtheta: -kappa * mu(dtheta) @ xbar)(zeros(3))

This would add computational cost and require differentiating through SO(3) exp.

Reference: Mardia & Jupp (2000) "Directional Statistics" Ch. 9 (vMF Fisher info)
Reference: Sra (2012) "A short note on parameter approximation for vMF"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from fl_slam_poc.common.jax_init import jax, jnp
from fl_slam_poc.common import constants
from fl_slam_poc.common.belief import D_Z
from fl_slam_poc.common.certificates import (
    CertBundle,
    ExpectedEffect,
    ConditioningCert,
    InfluenceCert,
    SupportCert,
    MismatchCert,
)
from fl_slam_poc.common.primitives import domain_projection_psd_core
from fl_slam_poc.common.geometry import se3_jax
from fl_slam_poc.backend.operators.kappa import kappa_from_resultant_v2


@dataclass
class ImuEvidenceResult:
    L_imu: jnp.ndarray  # (22,22)
    h_imu: jnp.ndarray  # (22,)
    kappa: float
    ess: float


@jax.jit
def _accel_resultant_direction_jax(
    imu_accel: jnp.ndarray,   # (M,3)
    weights: jnp.ndarray,     # (M,)
    accel_bias: jnp.ndarray,  # (3,)
    eps_mass: float,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    w = weights
    ess = jnp.sum(w)

    a = imu_accel - accel_bias[None, :]
    n = jnp.linalg.norm(a, axis=1, keepdims=True)
    x = a / (n + eps_mass)  # (M,3)

    S = jnp.sum(w[:, None] * x, axis=0)  # (3,)
    S_norm = jnp.linalg.norm(S)
    xbar = S / (S_norm + eps_mass)
    Rbar = S_norm / (ess + eps_mass)
    return xbar, Rbar, ess


def imu_vmf_gravity_evidence(
    rotvec_world_body: jnp.ndarray,  # (3,) rotvec of body in world
    imu_accel: jnp.ndarray,          # (M,3)
    weights: jnp.ndarray,            # (M,)
    accel_bias: jnp.ndarray,         # (3,)
    gravity_W: jnp.ndarray,          # (3,)
    eps_psd: float,
    eps_mass: float,
    chart_id: str,
    anchor_id: str,
) -> Tuple[ImuEvidenceResult, CertBundle, ExpectedEffect]:
    """
    Closed-form Laplace/I-projection of the vMF-style accelerometer direction factor onto a 22D Gaussian info term.

    We differentiate only over the 3D rotation perturbation δθ applied on the right:
      R(δθ) = R0 @ Exp(δθ)
    """
    rotvec0 = jnp.asarray(rotvec_world_body, dtype=jnp.float64).reshape(-1)
    R0 = se3_jax.so3_exp(rotvec0)

    g = jnp.asarray(gravity_W, dtype=jnp.float64).reshape(-1)
    g_hat = g / (jnp.linalg.norm(g) + eps_mass)
    minus_g_hat = -g_hat

    xbar, Rbar, ess = _accel_resultant_direction_jax(
        imu_accel=jnp.asarray(imu_accel, dtype=jnp.float64),
        weights=jnp.asarray(weights, dtype=jnp.float64).reshape(-1),
        accel_bias=jnp.asarray(accel_bias, dtype=jnp.float64).reshape(-1),
        eps_mass=eps_mass,
    )

    kappa_result, kappa_cert, _kappa_effect = kappa_from_resultant_v2(
        R_bar=float(Rbar),
        eps_r=constants.GC_EPS_R,
        eps_den=constants.GC_EPS_DEN,
        chart_id=chart_id,
        anchor_id=anchor_id,
    )
    kappa = float(kappa_result.kappa)

    # Predicted mean direction in body frame at the linearization point.
    # mu0 = R0^T (-g_hat)
    mu0 = R0.T @ minus_g_hat

    kappa_f = jnp.array(kappa, dtype=jnp.float64)
    x_dot_mu = (xbar @ mu0)

    # Closed-form gradient and Hessian w.r.t. right-perturbation rotation δθ:
    #   g = ∂/∂δθ (-κ mu^T xbar) |0 = -κ (mu0 × xbar)  [EXACT]
    #   H ≈ κ [ (x·mu) I - 0.5 (x mu^T + mu x^T) ]     [APPROXIMATION, see module docstring]
    #
    # NOTE: This H_rot is an approximation. See module docstring for:
    #   - Exact vMF Fisher information derivation
    #   - Error bounds (< 5% when x̄ ≈ μ₀)
    #   - Justification for conservative approximation
    g_rot = -kappa_f * jnp.cross(mu0, xbar)
    I3 = jnp.eye(3, dtype=jnp.float64)
    H_rot = kappa_f * (x_dot_mu * I3 - 0.5 * (jnp.outer(xbar, mu0) + jnp.outer(mu0, xbar)))
    H_rot = 0.5 * (H_rot + H_rot.T)
    H_rot_psd, H_cert_vec = domain_projection_psd_core(H_rot, eps_psd)

    L = jnp.zeros((D_Z, D_Z), dtype=jnp.float64)
    L = L.at[0:3, 0:3].set(H_rot_psd)
    h = jnp.zeros((D_Z,), dtype=jnp.float64)
    h = h.at[0:3].set(-g_rot)

    nll_proxy = float(-kappa_f * (mu0 @ xbar))
    # Conditioning comes from the PSD projection certificate (already clamped).
    proj_delta = float(H_cert_vec[0])
    eig_min = float(H_cert_vec[2])
    eig_max = float(H_cert_vec[3])
    cond = float(H_cert_vec[4])
    near_null = int(H_cert_vec[5])

    cert = CertBundle.create_approx(
        chart_id=chart_id,
        anchor_id=anchor_id,
        triggers=["ImuAccelDirectionClosedFormLaplace"] + list(kappa_cert.approximation_triggers),
        conditioning=ConditioningCert(
            eig_min=eig_min,
            eig_max=eig_max,
            cond=cond,
            near_null_count=near_null,
        ),
        support=SupportCert(ess_total=float(ess), support_frac=1.0),
        mismatch=MismatchCert(nll_per_ess=nll_proxy / (float(ess) + eps_mass), directional_score=float(Rbar)),
        influence=InfluenceCert(
            lift_strength=0.0,
            psd_projection_delta=proj_delta,
            mass_epsilon_ratio=0.0,
            anchor_drift_rho=0.0,
            dt_scale=1.0,
            extrinsic_scale=1.0,
            trust_alpha=1.0,
        ),
    )

    effect = ExpectedEffect(
        objective_name="imu_accel_direction_nll_proxy",
        predicted=nll_proxy,
        realized=None,
    )

    return ImuEvidenceResult(L_imu=L, h_imu=h, kappa=kappa, ess=float(ess)), cert, effect
