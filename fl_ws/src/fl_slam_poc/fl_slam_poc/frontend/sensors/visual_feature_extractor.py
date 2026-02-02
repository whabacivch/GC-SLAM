"""
Visual feature extractor for FL-SLAM POC — closed-form + info-geometry friendly.

Key upgrades vs. the original:
- Explicit color ordering (RGB/BGR) so grayscale conversion is correct.
- Depth sampling can be robust (median window) and returns a local depth variance estimate.
- Invalid depth -> *both* (a) tiny weight and (b) huge covariance (prevents "false confident 3D").
- Backprojection uncertainty uses an *improved closed-form second-moment* covariance for X,Y
  under independent Gaussian (u,v,z): it adds the missing σ_u^2 σ_z^2 / f^2 term (linearization misses this).
- Outputs both covariance and precision (information) for each 3D feature, plus logdet for IG / MHT scoring.
- Contracts are explicit: weight is for prioritization/budgeting; precision encodes metric uncertainty.
  (If you want weight to scale precision downstream, do it *once* in the evidence operator.)
- Per-feature meta exposes depth for LiDAR–camera fusion: depth_m (z_c), depth_sigma_c_sq (σ_c²),
  depth_Lambda_c, depth_theta_c (scalar natural params for 1D depth along ray). Invalid depth:
  Lambda_c=0, theta_c=0 so fusion can use LiDAR-only on that ray.

No global iterative optimization; per-frame operator only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
import base64
import io
import time
import math
import numpy as np

try:
    import cv2  # Optional dependency
except Exception:
    cv2 = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False
    _plt = None


# ----------------------------
# Mathematical constants (documented; not tunables)
# ----------------------------
# MAD scale for Gaussian: median(|x - med|) * MAD_NORMAL_SCALE estimates sigma
MAD_NORMAL_SCALE = 1.4826
# vMF log normalizer: A(kappa) = log(4*pi*sinh(kappa)/kappa)
VMF_LOG_NORM_4PI = math.log(4.0 * math.pi)
# Gaussian log-partition constant
LOG_2PI = math.log(2.0 * math.pi)


# ----------------------------
# vMF and Fisher-Rao helpers (items 3, 4, 15)
# ----------------------------

def _log_sinh_stable(k: float, eps: float = 1e-12) -> float:
    """Stable log(sinh(k)) for k >= 0. Large k: k - log(2) + log1p(-exp(-2k)); small k: Taylor."""
    k = max(float(k), eps)
    if k > 20.0:
        return k - math.log(2.0) + math.log1p(-math.exp(-2.0 * k))
    if k >= 1e-2:
        return math.log(math.sinh(k))
    return math.log(k + (k ** 3) / 6.0)


def A_vmf(k: float, eps: float = 1e-12) -> float:
    """vMF log normalizer: A(kappa) = log(4*pi*sinh(kappa)/kappa)."""
    k = max(float(k), eps)
    return VMF_LOG_NORM_4PI + _log_sinh_stable(k, eps) - math.log(k)


def hellinger2_vmf(
    mu1: np.ndarray,
    k1: float,
    mu2: np.ndarray,
    k2: float,
    eps: float = 1e-12,
) -> float:
    """Squared Hellinger between two vMF on S^2 via BC = exp(A(k_m) - (A(k1)+A(k2))/2), H^2 = 1 - BC."""
    eta1 = k1 * np.asarray(mu1, dtype=np.float64).ravel()[:3]
    eta2 = k2 * np.asarray(mu2, dtype=np.float64).ravel()[:3]
    eta_sum = eta1 + eta2
    km = 0.5 * float(np.linalg.norm(eta_sum))
    km = max(km, eps)
    k1 = max(float(k1), eps)
    k2 = max(float(k2), eps)
    bc = math.exp(A_vmf(km, eps) - 0.5 * (A_vmf(k1, eps) + A_vmf(k2, eps)))
    return max(0.0, 1.0 - bc)


# BC LUT (13): grid (kappa_1, kappa_2, cos(theta)); |eta1+eta2|/2 = 0.5*sqrt(k1^2+k2^2+2*k1*k2*c)
_LUT_GRID_K = 32
_LUT_GRID_C = 32
_BC_LUT: Optional[np.ndarray] = None


def _build_bc_lut() -> np.ndarray:
    """Precompute BC = exp(A(km) - 0.5*(A(k1)+A(k2))) on grid (k1, k2, c)."""
    k_min, k_max = 0.1, 100.0
    k1 = np.logspace(math.log10(k_min), math.log10(k_max), _LUT_GRID_K)
    k2 = k1.copy()
    c = np.linspace(-1.0, 1.0, _LUT_GRID_C)
    lut = np.zeros((_LUT_GRID_K, _LUT_GRID_K, _LUT_GRID_C), dtype=np.float64)
    eps = 1e-12
    for i1, k1v in enumerate(k1):
        for i2, k2v in enumerate(k2):
            for i3, cv in enumerate(c):
                km = 0.5 * math.sqrt(k1v * k1v + k2v * k2v + 2.0 * k1v * k2v * cv)
                km = max(km, eps)
                bc = math.exp(A_vmf(km, eps) - 0.5 * (A_vmf(k1v, eps) + A_vmf(k2v, eps)))
                lut[i1, i2, i3] = max(0.0, min(1.0, bc))
    return lut


def hellinger2_vmf_lut(
    mu1: np.ndarray,
    k1: float,
    mu2: np.ndarray,
    k2: float,
    k_min: float = 0.1,
    k_max: float = 100.0,
) -> float:
    """Hellinger^2 vMF via precomputed BC LUT (trilinear lookup). Falls back to exact if out of range."""
    global _BC_LUT
    if _BC_LUT is None:
        _BC_LUT = _build_bc_lut()
    c = float(np.dot(np.asarray(mu1, dtype=np.float64).ravel()[:3], np.asarray(mu2, dtype=np.float64).ravel()[:3]))
    c = max(-1.0, min(1.0, c))
    k1 = max(k_min, min(k_max, float(k1)))
    k2 = max(k_min, min(k_max, float(k2)))
    log_k_min, log_k_max = math.log10(k_min), math.log10(k_max)
    i1 = (math.log10(k1) - log_k_min) / (log_k_max - log_k_min) * (_LUT_GRID_K - 1)
    i2 = (math.log10(k2) - log_k_min) / (log_k_max - log_k_min) * (_LUT_GRID_K - 1)
    i3 = (c + 1.0) / 2.0 * (_LUT_GRID_C - 1)
    i1 = max(0, min(_LUT_GRID_K - 1.001, i1))
    i2 = max(0, min(_LUT_GRID_K - 1.001, i2))
    i3 = max(0, min(_LUT_GRID_C - 1.001, i3))
    i1l, i1u = int(i1), min(int(i1) + 1, _LUT_GRID_K - 1)
    i2l, i2u = int(i2), min(int(i2) + 1, _LUT_GRID_K - 1)
    i3l, i3u = int(i3), min(int(i3) + 1, _LUT_GRID_C - 1)
    w1 = i1 - i1l
    w2 = i2 - i2l
    w3 = i3 - i3l
    bc = (
        (1 - w1) * (1 - w2) * (1 - w3) * _BC_LUT[i1l, i2l, i3l]
        + (1 - w1) * (1 - w2) * w3 * _BC_LUT[i1l, i2l, i3u]
        + (1 - w1) * w2 * (1 - w3) * _BC_LUT[i1l, i2u, i3l]
        + (1 - w1) * w2 * w3 * _BC_LUT[i1l, i2u, i3u]
        + w1 * (1 - w2) * (1 - w3) * _BC_LUT[i1u, i2l, i3l]
        + w1 * (1 - w2) * w3 * _BC_LUT[i1u, i2l, i3u]
        + w1 * w2 * (1 - w3) * _BC_LUT[i1u, i2u, i3l]
        + w1 * w2 * w3 * _BC_LUT[i1u, i2u, i3u]
    )
    return max(0.0, 1.0 - float(bc))


def _inv_sqrt_spd(S: np.ndarray) -> np.ndarray:
    """Symmetric positive definite: return S^{-1/2} via eigendecomposition."""
    S = np.asarray(S, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ (np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T)


def _logm_spd(S: np.ndarray) -> np.ndarray:
    """Matrix logarithm of SPD S."""
    S = np.asarray(S, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(S)
    eigvals = np.maximum(eigvals, 1e-12)
    return eigvecs @ (np.diag(np.log(eigvals)) @ eigvecs.T)


def fr_cov(S1: np.ndarray, S2: np.ndarray) -> float:
    """Affine-invariant (Fisher-Rao) distance between SPD covariances: 0.5 * |log(S1^{-1/2} S2 S1^{-1/2})|_F^2."""
    R = _inv_sqrt_spd(S1)
    M = R @ S2 @ R.T
    L = _logm_spd(M)
    return 0.5 * float(np.sum(L * L))


def fr_gauss(
    mu1: np.ndarray,
    S1: np.ndarray,
    mu2: np.ndarray,
    S2: np.ndarray,
    alpha: float = 1.0,
) -> float:
    """Combined Fisher-Rao Gaussian distance: sqrt(d_mu^2 + alpha * d_Sigma^2)."""
    mu1 = np.asarray(mu1, dtype=np.float64).ravel()[:3]
    mu2 = np.asarray(mu2, dtype=np.float64).ravel()[:3]
    S1 = np.asarray(S1, dtype=np.float64).reshape(3, 3)
    S2 = np.asarray(S2, dtype=np.float64).reshape(3, 3)
    Sm = 0.5 * (S1 + S2)
    try:
        Sm_inv = np.linalg.inv(Sm + 1e-9 * np.eye(3))
    except np.linalg.LinAlgError:
        return float("inf")
    dmu2 = float((mu1 - mu2) @ Sm_inv @ (mu1 - mu2))
    dS2 = fr_cov(S1, S2)
    return math.sqrt(max(0.0, dmu2 + alpha * dS2))


def A_vmf_coth(k: float, eps: float = 1e-12) -> float:
    """coth(kappa) - 1/kappa for vMF moment matching (resultant)."""
    k = max(float(k), eps)
    if k >= 0.01:
        return 1.0 / math.tanh(k) - 1.0 / k
    return k / 3.0


def mixture_vmf_to_single(
    mus: List[np.ndarray],
    kappas: List[float],
    weights: Optional[List[float]] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, float]:
    """
    Moment-match mixture of vMF to single vMF (14): r = sum_j pi_j A(kappa_j) mu_j,
    mu = r/|r|, kappa approx A^{-1}(|r|). Weights default to uniform.
    """
    B = len(mus)
    if B == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    if weights is None:
        weights = [1.0 / B] * B
    r = np.zeros(3, dtype=np.float64)
    for j in range(B):
        mu_j = np.asarray(mus[j], dtype=np.float64).ravel()[:3]
        mu_j = mu_j / (np.linalg.norm(mu_j) + eps)
        k_j = max(float(kappas[j]), eps)
        r += weights[j] * A_vmf_coth(k_j, eps) * mu_j
    r_norm = float(np.linalg.norm(r))
    if r_norm < eps:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64), 0.0
    mu = r / r_norm
    kappa = r_norm
    if kappa > 100.0:
        kappa = 100.0
    return mu, kappa


def orthogonal_score(
    mu1: np.ndarray,
    S1: np.ndarray,
    mu1_app: Optional[np.ndarray],
    k1_app: float,
    mu2: np.ndarray,
    S2: np.ndarray,
    mu2_app: Optional[np.ndarray],
    k2_app: float,
    beta: float = 1.0,
    alpha: float = 1.0,
) -> float:
    """Product-manifold score: d_pos + beta * d_app (position: FR-Gaussian; appearance: Hellinger vMF)."""
    d_pos = fr_gauss(mu1, S1, mu2, S2, alpha=alpha)
    if mu1_app is not None and mu2_app is not None and k1_app > 0 and k2_app > 0:
        d_app = math.sqrt(hellinger2_vmf(mu1_app, k1_app, mu2_app, k2_app))
        return d_pos + beta * d_app
    return d_pos


def _build_hellinger_heatmap_base64(
    kappas: Optional[np.ndarray] = None,
    n_theta: int = 32,
) -> Optional[str]:
    """Hellinger heatmap for vMF: grid over (kappa_1, theta). Returns base64 PNG or None."""
    if not _HAS_MPL or _plt is None:
        return None
    kappas = kappas if kappas is not None else np.logspace(-1, 2, 24)
    thetas = np.linspace(0, np.pi, n_theta)
    mu1 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    k1 = 1.0
    H2 = np.zeros((len(kappas), len(thetas)))
    for ik, k2 in enumerate(kappas):
        for it, th in enumerate(thetas):
            c = math.cos(th)
            mu2 = np.array([math.sqrt(1 - c * c), 0.0, c], dtype=np.float64)
            if np.isfinite(mu2[0]):
                H2[ik, it] = hellinger2_vmf(mu1, k1, mu2, float(k2))
            else:
                mu2 = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                H2[ik, it] = hellinger2_vmf(mu1, k1, mu2, float(k2))
    fig, ax = _plt.subplots()
    ax.imshow(H2, aspect="auto", origin="lower", extent=[0, np.pi, float(kappas[0]), float(kappas[-1])])
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("kappa_2")
    ax.set_title("Hellinger^2 vMF")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    _plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _build_fr_heatmap_base64(
    n_ratio: int = 20,
    n_offset: int = 16,
) -> Optional[str]:
    """FR heatmap for Gaussians: grid over eigenvalue ratio and mean offset proxy."""
    if not _HAS_MPL or _plt is None:
        return None
    ratios = np.linspace(0.1, 10.0, n_ratio)
    offsets = np.linspace(0.0, 2.0, n_offset)
    S1 = np.eye(3, dtype=np.float64)
    mu1 = np.zeros(3, dtype=np.float64)
    D = np.zeros((len(ratios), len(offsets)))
    for ir, r in enumerate(ratios):
        S2 = np.diag([1.0, r, r * r]).astype(np.float64)
        for i_off, o in enumerate(offsets):
            mu2 = np.array([o, 0.0, 0.0], dtype=np.float64)
            D[ir, i_off] = fr_gauss(mu1, S1, mu2, S2, alpha=1.0)
    fig, ax = _plt.subplots()
    ax.imshow(D, aspect="auto", origin="lower", extent=[float(offsets[0]), float(offsets[-1]), float(ratios[0]), float(ratios[-1])])
    ax.set_xlabel("mean offset")
    ax.set_ylabel("eigenvalue ratio")
    ax.set_title("Fisher-Rao Gaussian")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80)
    _plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Rotation matrix to quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

@dataclass(frozen=True)
class VisualFeatureExtractorConfig:
    """Named configuration for VisualFeatureExtractor parameters."""

    max_features: int = 800
    orb_nlevels: int = 8
    orb_scale_factor: float = 1.2
    depth_scale: float = 1.0
    pixel_sigma: float = 1.0
    depth_model: DepthModel = "linear"
    depth_sigma0: float = 0.01
    depth_sigma_slope: float = 0.01
    depth_sample_mode: DepthSampleMode = "median3"
    min_depth_m: float = 0.05
    max_depth_m: float = 80.0
    cov_reg_eps: float = 1e-9
    invalid_cov_inflate: float = 1e6
    response_soft_scale: float = 50.0
    depth_validity_slope: float = 5.0
    orb_enabled: bool = True
    # Item 1: hex depth sampling
    hex_radius: int = 2
    # Item 2: quadratic surface fit
    quad_fit_radius: int = 2
    quad_fit_min_points: int = 6
    quad_fit_lstsq_eps: float = 1e-8
    # Item 5: Wishart prior regularization
    prior_nu: float = 0.0
    prior_psi_scale: float = 1.0
    # Item 10: Student-t depth residual
    student_t_nu: float = 3.0
    student_t_w_min: float = 0.1
    # Item 11: MA convexity
    ma_tau: float = 10.0
    ma_delta_inflate: float = 1e-4
    # Item 12: kappa scheduling
    kappa0: float = 1.0
    kappa_alpha: float = 10.0
    kappa_max: float = 100.0
    kappa_min: float = 0.1
    # Item 4 / 15: Fisher-Rao and orthogonal score
    fr_alpha: float = 1.0
    score_beta: float = 1.0
    # Item 7: 3DGS export
    gs_opacity_gamma: float = 1.0
    gs_logdet0: float = 0.0
    gs_scale_min: float = 1e-6
    gs_scale_max: float = 1.0
    # Item 16: diagnostics
    debug_plots: bool = False


@dataclass
class Feature3D:
    """A single visual feature lifted to 3D (camera frame) with uncertainty and soft weight."""
    # Pixel location
    u: float
    v: float

    # 3D point in camera frame
    xyz: np.ndarray  # (3,)

    # 3x3 covariance (camera frame)
    cov_xyz: np.ndarray  # (3,3)

    # 3x3 precision (information) = inv(cov_xyz), regularized
    info_xyz: np.ndarray  # (3,3)

    # log(det(cov_xyz)) for IG/MHT scoring (regularized)
    logdet_cov: float

    # Canonical natural parameters (θ = Λ μ)
    canonical_theta: np.ndarray  # (3,)
    canonical_log_partition: float

    # Descriptor (e.g., ORB binary)
    desc: np.ndarray  # (D,) dtype=uint8 or float32

    # Continuous measurement weight in [0,1] (for budgeting/prioritization)
    weight: float

    # Metadata for debugging/audit
    meta: Dict[str, Any]

    # Optional appearance (vMF): view-direction or normal; used for Hellinger / orthogonal score
    mu_app: Optional[np.ndarray] = None  # (3,) unit vector
    kappa_app: float = 0.0

    # Optional RGB from image at (u,v); [0,1]; used for map/splat coloring
    color: Optional[np.ndarray] = None  # (3,) RGB


@dataclass
class QuadFitResult:
    """Result of local quadratic surface fit z(u,v)=a*u^2+b*u*v+c*v^2+d*u+e*v+f."""
    grad_z: np.ndarray  # (2,) zu, zv
    H: np.ndarray       # (2,2) Hessian
    normal: np.ndarray  # (3,) in camera frame, normalized
    K: float            # Gaussian curvature
    ma: float           # det(H), MA proxy
    lam_min: float      # minimum eigenvalue of H


@dataclass
class OpReportEvent:
    """Minimal OpReport-style event; adapt to your common/op_report.py schema."""
    op: str
    exact: bool
    approx_reason: str
    metrics: Dict[str, Any]


@dataclass
class ExtractionResult:
    features: List[Feature3D]
    op_report: List[OpReportEvent]
    timestamp_ns: int


# ----------------------------
# Extractor
# ----------------------------

DepthModel = Literal["linear", "quadratic"]
DepthSampleMode = Literal["nearest", "median3", "median5", "hex"]
ColorOrder = Literal["BGR", "RGB"]


class VisualFeatureExtractor:
    """
    Extract keypoints+descriptors from RGB and lift them to 3D using depth, with analytic uncertainty.

    Backend:
    - ORB if OpenCV available
    - Else deterministic grid sampler with patch descriptors

    Output is designed to be used by downstream evidence operators that:
    - build residuals on the correct manifold,
    - apply Student-t scale mixture / reliability weighting there (not here),
    - and fuse in information form.
    """

    def __init__(
        self,
        intrinsics: PinholeIntrinsics,
        *,
        config: Optional[VisualFeatureExtractorConfig] = None,
    ) -> None:
        self.K = intrinsics
        self.config = config or VisualFeatureExtractorConfig()

        self._orb = self._build_orb()

    # ----------------------------
    # Public API
    # ----------------------------

    def _build_orb(self):
        """Create ORB descriptor if OpenCV is available and enabled."""
        if not self.config.orb_enabled or cv2 is None:
            return None
        return cv2.ORB_create(
            nfeatures=self.config.max_features,
            scaleFactor=float(self.config.orb_scale_factor),
            nlevels=int(self.config.orb_nlevels),
            edgeThreshold=31,
            patchSize=31,
            fastThreshold=20,
        )

    def extract(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        *,
        color_order: ColorOrder = "BGR",
        timestamp_ns: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Args:
            rgb: HxWx3 uint8 (RGB or BGR; specify via color_order)
            depth: HxW depth image. uint16 (mm) or float32 (m); scaled by depth_scale
            color_order: "BGR" or "RGB"
            timestamp_ns: optional; if None uses time.time_ns()

        Returns:
            ExtractionResult(features, op_report, timestamp_ns)
        """
        ts = int(timestamp_ns) if timestamp_ns is not None else time.time_ns()
        op_report: List[OpReportEvent] = []

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(f"rgb must be HxWx3, got {rgb.shape}")
        if depth.ndim != 2:
            raise ValueError(f"depth must be HxW, got {depth.shape}")
        if depth.shape[:2] != rgb.shape[:2]:
            raise ValueError("rgb and depth must have same H,W")

        gray = self._to_gray(rgb, color_order=color_order)

        if self._orb is not None:
            kps, desc = self._orb.detectAndCompute(gray, None)
            if kps is None:
                kps = []
            if desc is None:
                desc = np.zeros((0, 32), dtype=np.uint8)

            # Deterministic trim by response
            if len(kps) > self.config.max_features:
                idx = np.argsort([-kp.response for kp in kps])[: self.config.max_features]
                kps = [kps[i] for i in idx]
                desc = desc[idx]
        else:
            kps, desc = self._grid_features(gray)

        features: List[Feature3D] = []
        num_total = len(kps)

        invalid_depth = 0
        depth_holes = 0

        # Per-frame quality metrics and diagnostics (16)
        weights: List[float] = []
        traces: List[float] = []
        logdets: List[float] = []
        trace_infos: List[float] = []
        rel_depth_noises: List[float] = []
        lam_min_H_list: List[float] = []

        for i, kp in enumerate(kps):
            u = float(kp.pt[0]) if hasattr(kp, "pt") else float(kp[0])
            v = float(kp.pt[1]) if hasattr(kp, "pt") else float(kp[1])

            z_m, z_var_m2, z_valid, zs = self._depth_sample(depth, u, v)
            if not z_valid:
                depth_holes += 1

            w_depth = self._depth_weight(z_m) if z_valid else 0.0

            # Keypoint response-based soft weight (no gating)
            resp = float(getattr(kp, "response", 1.0)) if hasattr(kp, "response") else 1.0
            w_resp = self._response_weight(resp)

            # Weight contract: budgeting/prioritization only (do not also scale precision downstream unless intentional)
            weight = float(np.clip(w_depth * w_resp, 0.0, 1.0))

            # Optional quadratic fit (2) when depth valid
            quad_fit: Optional[QuadFitResult] = None
            if z_valid and np.isfinite(z_m) and z_m > 0.0:
                quad_fit = self._quadratic_fit(depth, u, v, z_m)

            # Student-t effective variance (10)
            var_z_eff = (
                self._student_t_effective_var(z_m, max(z_var_m2, self._depth_sigma(z_m) ** 2), zs)
                if z_valid and np.isfinite(z_var_m2)
                else (self._depth_sigma(z_m) ** 2 if z_valid else float("nan"))
            )
            if not np.isfinite(var_z_eff) and z_valid:
                var_z_eff = self._depth_sigma(z_m) ** 2

            # If depth invalid, produce a benign feature (tiny weight, huge covariance).
            if (not z_valid) or (not np.isfinite(z_m)) or (z_m <= 0.0):
                invalid_depth += 1
                xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                cov_xyz = np.eye(3, dtype=np.float64) * self.config.invalid_cov_inflate
                info_xyz, logdet_cov = self._precision_and_logdet(cov_xyz)
                canonical_theta = np.zeros(3, dtype=np.float64)
                canonical_log_partition = 0.0
            else:
                xyz = self._backproject(u, v, z_m)
                cov_xyz, approx_ev = self._backprojection_cov_closed_form(
                    u=u,
                    v=v,
                    z=z_m,
                    var_u=self.config.pixel_sigma ** 2,
                    var_v=self.config.pixel_sigma ** 2,
                    var_z=max(float(var_z_eff), self._depth_sigma(z_m) ** 2),
                )
                if approx_ev is not None:
                    op_report.append(approx_ev)

                # MA convexity inflation (11)
                if quad_fit is not None:
                    w_ma = self._ma_convexity_weight(quad_fit.lam_min)
                    delta = float(self.config.ma_delta_inflate)
                    cov_xyz = cov_xyz + (1.0 - w_ma) * delta * np.eye(3, dtype=np.float64)

                info_xyz, logdet_cov = self._precision_and_logdet(cov_xyz)
                canonical_theta = info_xyz @ xyz
                # Gaussian log-partition in natural form: A(θ,Λ) = 0.5 μ'Λμ + 0.5 log det Σ + (d/2) log(2π)
                canonical_log_partition = (
                    0.5 * float(xyz @ info_xyz @ xyz)
                    + 0.5 * float(logdet_cov)
                    + 1.5 * LOG_2PI
                )

            # Kappa scheduling (12) and appearance from quad fit
            mu_app: Optional[np.ndarray] = None
            kappa_app = 0.0
            w_ma = 1.0
            if quad_fit is not None:
                w_ma = self._ma_convexity_weight(quad_fit.lam_min)
                mu_app = quad_fit.normal.copy()
                eps = 1e-9
                rel_noise = float(var_z_eff) ** 0.5 / (z_m + eps) if z_valid and np.isfinite(z_m) else 1.0
                rho = 1.0 / (rel_noise + eps)
                mean_curv_mag = abs(quad_fit.K) ** 0.5
                kappa_app = self.config.kappa0 + self.config.kappa_alpha * mean_curv_mag * rho
                kappa_app = max(
                    self.config.kappa_min,
                    min(self.config.kappa_max, kappa_app),
                )
                kappa_app = kappa_app * w_ma

            d = desc[i].copy()

            # Depth natural params for LiDAR–camera fusion (plan: lidar-camera_splat_fusion_and_bev_ot).
            # z_c = depth along ray (m), sigma_c_sq = effective variance used for 3D cov (m²).
            # Lambda_c, theta_c = scalar natural params for 1D depth; PoE: Λf = Λc + Λ_ell, θf = θc + θ_ell (lidar_depth_evidence returns Λ_ell, θ_ell).
            # Invalid depth: Lambda_c=0, theta_c=0 so fusion uses LiDAR-only on that ray.
            if z_valid and np.isfinite(z_m) and z_m > 0 and np.isfinite(var_z_eff) and var_z_eff > 0:
                depth_sigma_c_sq = float(var_z_eff)
                depth_Lambda_c = 1.0 / depth_sigma_c_sq
                depth_theta_c = depth_Lambda_c * z_m
            else:
                depth_sigma_c_sq = float("nan")
                depth_Lambda_c = 0.0
                depth_theta_c = 0.0

            meta = {
                "response": resp,
                "w_resp": float(w_resp),
                "depth_m": float(z_m) if np.isfinite(z_m) else float("nan"),
                "depth_valid": bool(z_valid),
                "depth_var_m2": float(z_var_m2) if np.isfinite(z_var_m2) else float("nan"),
                "w_depth": float(w_depth),
                "weight": float(weight),
                # For LiDAR–camera depth fusion (same ray, natural-param add on z).
                "depth_sigma_c_sq": depth_sigma_c_sq,
                "depth_Lambda_c": depth_Lambda_c,
                "depth_theta_c": depth_theta_c,
            }

            # Sample RGB at (u,v) for map/splat coloring (nearest pixel, bounds-clamped)
            h, w = rgb.shape[0], rgb.shape[1]
            ix = int(round(np.clip(u, 0, w - 1)))
            iy = int(round(np.clip(v, 0, h - 1)))
            rgb_sample = np.asarray(rgb[iy, ix], dtype=np.float64).ravel()[:3] / 255.0

            features.append(
                Feature3D(
                    u=u,
                    v=v,
                    xyz=xyz,
                    cov_xyz=cov_xyz,
                    info_xyz=info_xyz,
                    logdet_cov=float(logdet_cov),
                    canonical_theta=canonical_theta,
                    canonical_log_partition=float(canonical_log_partition),
                    desc=d,
                    weight=weight,
                    meta=meta,
                    mu_app=mu_app,
                    kappa_app=float(kappa_app),
                    color=rgb_sample,
                )
            )

            weights.append(weight)
            traces.append(float(np.trace(cov_xyz)))
            logdets.append(float(logdet_cov))
            trace_infos.append(float(np.trace(info_xyz)))
            if z_valid and np.isfinite(z_m) and z_m > 0 and np.isfinite(var_z_eff):
                rel_depth_noises.append(float((var_z_eff ** 0.5) / (z_m + 1e-9)))
            if quad_fit is not None:
                lam_min_H_list.append(float(quad_fit.lam_min))

        metrics_base: Dict[str, Any] = {
            "num_features": len(features),
            "num_input_kps": num_total,
            "num_invalid_depth": int(invalid_depth),
            "num_depth_holes": int(depth_holes),
            "depth_sample_mode": self.config.depth_sample_mode,
            "method": "ORB" if self._orb is not None else "GRID_PATCH",
            "weight_mean": float(np.mean(weights)) if weights else 0.0,
            "weight_p10": float(np.percentile(weights, 10)) if weights else 0.0,
            "weight_p50": float(np.percentile(weights, 50)) if weights else 0.0,
            "weight_p90": float(np.percentile(weights, 90)) if weights else 0.0,
            "trace_cov_mean": float(np.mean(traces)) if traces else 0.0,
            "logdet_cov_mean": float(np.mean(logdets)) if logdets else 0.0,
            "trace_info_mean": float(np.mean(trace_infos)) if trace_infos else 0.0,
            "logdet_cov_p10": float(np.percentile(logdets, 10)) if logdets else 0.0,
            "logdet_cov_p90": float(np.percentile(logdets, 90)) if logdets else 0.0,
            "rel_depth_noise_mean": float(np.mean(rel_depth_noises)) if rel_depth_noises else float("nan"),
            "lam_min_H_mean": float(np.mean(lam_min_H_list)) if lam_min_H_list else float("nan"),
        }
        if self.config.debug_plots and _HAS_MPL and _plt is not None:
            hm_hellinger = _build_hellinger_heatmap_base64()
            if hm_hellinger is not None:
                metrics_base["hellinger_vmf_heatmap_base64"] = hm_hellinger
            hm_fr = _build_fr_heatmap_base64()
            if hm_fr is not None:
                metrics_base["fr_gaussian_heatmap_base64"] = hm_fr
        op_report.append(
            OpReportEvent(
                op="visual_feature_extraction",
                exact=(self._orb is not None),
                approx_reason="" if self._orb is not None else "opencv_missing_grid_fallback",
                metrics=metrics_base,
            )
        )

        return ExtractionResult(features=features, op_report=op_report, timestamp_ns=ts)

    # ----------------------------
    # Internals
    # ----------------------------

    def _to_gray(self, rgb: np.ndarray, *, color_order: ColorOrder) -> np.ndarray:
        if rgb.dtype != np.uint8:
            # Fallback: simple average
            return rgb.mean(axis=2).astype(np.uint8)

        if cv2 is None:
            return rgb.mean(axis=2).astype(np.uint8)

        if color_order == "BGR":
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        if color_order == "RGB":
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        raise ValueError(f"Unknown color_order={color_order}")

    def _depth_sample(
        self, depth: np.ndarray, u: float, v: float
    ) -> Tuple[float, float, bool, Optional[List[float]]]:
        """
        Depth sampling that can be robust (median window) and returns a local variance estimate.

        Returns:
            z_m: depth in meters
            z_var_m2: local depth variance estimate (meters^2), NaN if unavailable
            valid: whether a plausible depth was obtained
            zs: optional list of sample depths used (for Student-t); None if unavailable
        """
        x = int(round(u))
        y = int(round(v))
        H, W = depth.shape[:2]
        if x < 0 or y < 0 or x >= W or y >= H:
            return float("nan"), float("nan"), False, None

        if self.config.depth_sample_mode == "nearest":
            z = self._depth_to_meters(depth[y, x])
            z_m, z_var, valid = self._validate_depth(z)
            return z_m, z_var, valid, None

        if self.config.depth_sample_mode == "hex":
            z_hat, sigma_z2, valid, zs = self._depth_sample_hex(depth, u, v, x, y, H, W)
            return z_hat, sigma_z2, valid, zs

        # Windowed robust sampling (square)
        if self.config.depth_sample_mode == "median3":
            r = 1
        elif self.config.depth_sample_mode == "median5":
            r = 2
        else:
            raise ValueError(f"Unknown depth_sample_mode={self.config.depth_sample_mode}")

        x0 = max(0, x - r)
        x1 = min(W - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(H - 1, y + r)

        window = depth[y0 : y1 + 1, x0 : x1 + 1].reshape(-1)
        zs = np.array([self._depth_to_meters(vv) for vv in window], dtype=np.float64)
        zs = zs[np.isfinite(zs)]
        if zs.size == 0:
            return float("nan"), float("nan"), False, None

        # Reject non-positive depths
        zs = zs[zs > 0.0]
        if zs.size == 0:
            return float("nan"), float("nan"), False, None

        z_med = float(np.median(zs))
        z_var = float(np.var(zs)) if zs.size >= 4 else float("nan")

        z_med, z_var, valid = self._validate_depth(z_med, var_override=z_var)
        return z_med, z_var, valid, zs.tolist()

    def _depth_sample_hex(
        self, depth: np.ndarray, u: float, v: float, x: int, y: int, H: int, W: int
    ) -> Tuple[float, float, bool, Optional[List[float]]]:
        """Hexagonal 7-sample stencil: center + 6 at angle k*pi/3. Robust depth = median; scale = MAD_NORMAL_SCALE * MAD."""
        r = max(1, int(self.config.hex_radius))
        offsets = [(0, 0)] + [
            (round(r * math.cos(k * math.pi / 3)), round(r * math.sin(k * math.pi / 3)))
            for k in range(6)
        ]
        zs: List[float] = []
        for dx, dy in offsets:
            xi = x + dx
            yi = y + dy
            if 0 <= xi < W and 0 <= yi < H:
                z = self._depth_to_meters(depth[yi, xi])
                if np.isfinite(z) and z > 0.0:
                    zs.append(z)
        if len(zs) < 4:
            return float("nan"), float("nan"), False, None
        zs_arr = np.array(zs, dtype=np.float64)
        z_hat = float(np.median(zs_arr))
        mad = float(np.median(np.abs(zs_arr - z_hat)))
        sigma_z = MAD_NORMAL_SCALE * mad
        sigma_z2 = sigma_z * sigma_z
        z_hat, _, valid = self._validate_depth(z_hat, var_override=sigma_z2)
        return z_hat, sigma_z2, valid, zs

    def _student_t_effective_var(
        self,
        z_hat: float,
        sigma_z2: float,
        zs: Optional[List[float]],
    ) -> float:
        """
        Single-weight t-like inflation: one scalar w from average residual energy over the stencil,
        sigma_eff^2 = sigma^2 / max(w, w_min). Not equivalent to per-sample M-estimator weighting
        (which would use w_i = (nu+1)/(nu + r_i^2/sigma^2) per residual); document for downstream.
        """
        if zs is None or len(zs) < 2 or not np.isfinite(sigma_z2) or sigma_z2 <= 0:
            return sigma_z2
        nu = float(self.config.student_t_nu)
        w_min = float(self.config.student_t_w_min)
        eps = 1e-12
        sigma2 = max(sigma_z2, eps)
        residuals_sq = [(zi - z_hat) ** 2 for zi in zs]
        q = sum(residuals_sq) / (len(residuals_sq) * sigma2 + eps)
        w = (nu + 1.0) / (nu + q)
        w = max(w, w_min)
        return sigma_z2 / w

    def _ma_convexity_weight(self, lam_min: float) -> float:
        """Continuous convexity weight omega_MA = sigmoid(tau * lam_min)."""
        x = float(self.config.ma_tau) * lam_min
        # Numerically stable sigmoid to avoid exp overflow on large |x|.
        if x >= 0.0:
            return 1.0 / (1.0 + math.exp(-x))
        ex = math.exp(x)
        return ex / (1.0 + ex)

    def _quadratic_fit(
        self,
        depth: np.ndarray,
        u: float,
        v: float,
        z_hat: float,
    ) -> Optional[QuadFitResult]:
        """
        Fit local Monge patch z(u~,v~)=a*u~^2+b*u~*v~+c*v~^2+d*u~+e*v~+f in centered pixel coords
        u~=ui-u0, v~=vi-v0 so A'A is well-conditioned. Gradient/Hessian evaluated at (0,0).
        Derivatives then rescaled to camera-plane metric: ∂/∂x=(fx/z)∂/∂u~, ∂/∂y=(fy/z)∂/∂v~
        so curvature K and normal are in consistent 3D geometry (not resolution-dependent).
        """
        u0, v0 = float(u), float(v)
        x0 = int(round(u0))
        y0 = int(round(v0))
        H_img, W_img = depth.shape[:2]
        r = max(1, int(self.config.quad_fit_radius))
        pts: List[Tuple[float, float, float]] = []
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                xi = x0 + dx
                yi = y0 + dy
                if 0 <= xi < W_img and 0 <= yi < H_img:
                    zi = self._depth_to_meters(depth[yi, xi])
                    if np.isfinite(zi) and zi > 0.0:
                        pts.append((float(xi), float(yi), zi))
        if len(pts) < self.config.quad_fit_min_points:
            return None
        # Local (centered) coordinates for well-conditioned normal equations
        u_tilde = np.array([ui - u0 for ui, _, _ in pts], dtype=np.float64)
        v_tilde = np.array([vi - v0 for _, vi, _ in pts], dtype=np.float64)
        z_vals = np.array([zi for _, _, zi in pts], dtype=np.float64)
        A = np.column_stack(
            (
                u_tilde * u_tilde,
                u_tilde * v_tilde,
                v_tilde * v_tilde,
                u_tilde,
                v_tilde,
                np.ones(len(pts), dtype=np.float64),
            )
        )
        eps_lstsq = float(self.config.quad_fit_lstsq_eps)
        try:
            beta, _, _, _ = np.linalg.lstsq(
                A.T @ A + eps_lstsq * np.eye(6, dtype=np.float64), A.T @ z_vals, rcond=None
            )
        except np.linalg.LinAlgError:
            return None
        a, b_coef, c, d, e, f_coef = beta
        # At center (0,0): zu = d, zv = e; Hessian [[2a, b],[b, 2c]]
        zu_pix = float(d)
        zv_pix = float(e)
        H_pix = np.array([[2.0 * a, b_coef], [b_coef, 2.0 * c]], dtype=np.float64)
        # Rescale to camera-plane metric: x=(u-cx)z/fx, y=(v-cy)z/fy => ∂/∂x=(fx/z)∂/∂u, ∂/∂y=(fy/z)∂/∂v
        fx, fy, cx, cy = self.K.fx, self.K.fy, self.K.cx, self.K.cy
        z = max(float(z_hat), 1e-6)
        sx = fx / z
        sy = fy / z
        zu = sx * zu_pix
        zv = sy * zv_pix
        H_2x2 = np.array(
            [
                [sx * sx * H_pix[0, 0], sx * sy * H_pix[0, 1]],
                [sx * sy * H_pix[1, 0], sy * sy * H_pix[1, 1]],
            ],
            dtype=np.float64,
        )
        det_H = float(np.linalg.det(H_2x2))
        grad_sq = zu * zu + zv * zv
        K = det_H / ((1.0 + grad_sq) ** 2) if (1.0 + grad_sq) > 0 else 0.0
        eigvals = np.linalg.eigvalsh(H_2x2)
        lam_min = float(np.min(eigvals))
        n = np.array([-zu, -zv, 1.0], dtype=np.float64)
        nnorm = float(np.linalg.norm(n))
        if nnorm > 0:
            n = n / nnorm
        return QuadFitResult(
            grad_z=np.array([zu, zv], dtype=np.float64),
            H=H_2x2,
            normal=n,
            K=float(K),
            ma=float(det_H),
            lam_min=lam_min,
        )

    def _depth_to_meters(self, z_raw: Any) -> float:
        if z_raw is None:
            return float("nan")
        # Works for uint16, float32, etc.
        try:
            return float(z_raw) * self.config.depth_scale
        except Exception:
            return float("nan")

    def _validate_depth(self, z_m: float, var_override: Optional[float] = None) -> Tuple[float, float, bool]:
        if (not np.isfinite(z_m)) or (z_m <= 0.0):
            return float("nan"), float("nan"), False

        if z_m < self.config.min_depth_m or z_m > self.config.max_depth_m:
            # Still "valid" in the mathematical sense, but we treat it as low reliability.
            # Keep valid=True so downstream can decide (we will downweight via _depth_weight).
            var = float(var_override) if (var_override is not None and np.isfinite(var_override)) else float("nan")
            return z_m, var, True

        var = float(var_override) if (var_override is not None and np.isfinite(var_override)) else float("nan")
        return z_m, var, True

    def _depth_weight(self, z: float) -> float:
        """
        Continuous weight based on depth plausibility. No hard gate.
        Smoothly goes to ~0 outside [min_depth, max_depth].
        """
        if not np.isfinite(z):
            return 0.0
        a = self.config.depth_validity_slope
        w_min = 1.0 / (1.0 + math.exp(-a * (z - self.config.min_depth_m)))
        w_max = 1.0 / (1.0 + math.exp(+a * (z - self.config.max_depth_m)))
        return float(np.clip(w_min * w_max, 0.0, 1.0))

    def _response_weight(self, response: float) -> float:
        """Continuous weight from keypoint response (no gating)."""
        s = self.config.response_soft_scale
        return float(response / (response + s)) if response > 0 else 0.0

    def _depth_sigma(self, z_m: float) -> float:
        """
        Depth noise model (meters). Closed-form and selectable.
        """
        z = abs(float(z_m))
        if self.config.depth_model == "linear":
            return self.config.depth_sigma0 + self.config.depth_sigma_slope * z
        if self.config.depth_model == "quadratic":
            return self.config.depth_sigma0 + self.config.depth_sigma_slope * (z ** 2)
        raise ValueError(f"Unknown depth_model={self.config.depth_model}")

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        return np.array([x, y, z], dtype=np.float64)

    def _precision_and_logdet(self, cov_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute precision and logdet with a tiny diagonal regularization for stability.
        Optional Wishart prior (5): Lambda' = Lambda + nu Psi^{-1}, Psi = prior_psi_scale * I.
        Returned precision Lambda is used for both info_xyz and canonical_theta = Lambda @ mu (same Λ).
        """
        cov = cov_xyz.astype(np.float64, copy=False)
        cov_reg = cov + (self.config.cov_reg_eps * np.eye(3, dtype=np.float64))
        try:
            sign, logdet = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                cov_reg = cov + (1e-6 * np.eye(3, dtype=np.float64))
                sign, logdet = np.linalg.slogdet(cov_reg)
            Lambda = np.linalg.inv(cov_reg)
            if self.config.prior_nu > 0 and self.config.prior_psi_scale > 0:
                psi_inv_scale = 1.0 / float(self.config.prior_psi_scale)
                Lambda = Lambda + (float(self.config.prior_nu) * psi_inv_scale) * np.eye(
                    3, dtype=np.float64
                )
                cov_reg = np.linalg.inv(Lambda)
                sign, logdet = np.linalg.slogdet(cov_reg)
            return Lambda, float(logdet)
        except np.linalg.LinAlgError:
            cov_fallback = cov + (1e-3 * np.eye(3, dtype=np.float64))
            info = np.linalg.inv(cov_fallback)
            sign, logdet = np.linalg.slogdet(cov_fallback)
            return info, float(logdet)

    def _backprojection_cov_closed_form(
        self,
        *,
        u: float,
        v: float,
        z: float,
        var_u: float,
        var_v: float,
        var_z: float,
    ) -> Tuple[np.ndarray, Optional[OpReportEvent]]:
        """
        Closed-form covariance for pinhole backprojection with independent Gaussian (u,v,z).

        X = (u-cx) z / fx
        Y = (v-cy) z / fy
        Z = z

        Linear propagation (J Σ J^T) misses the σ_u^2 σ_z^2 / f^2 term because X and Y are bilinear in (u,z).
        Here we compute a better closed-form second-moment covariance:

        Assumes Cov(u,z)=0, Cov(v,z)=0 (independent u,v,z). If depth and keypoint localization are
        correlated (e.g. structured light / stereo), systematic anisotropy may appear; this is a limitation.

        Var(X) = ( z^2 Var(u) + (u-cx)^2 Var(z) + Var(u)Var(z) ) / fx^2
        Var(Y) = ( z^2 Var(v) + (v-cy)^2 Var(z) + Var(v)Var(z) ) / fy^2
        Cov(X,Y) = ((u-cx)(v-cy) Var(z)) / (fx fy)           [assuming Cov(u,v)=0]
        Cov(X,Z) = ((u-cx) Var(z)) / fx
        Cov(Y,Z) = ((v-cy) Var(z)) / fy
        Var(Z) = Var(z)

        This remains analytic/closed-form, is PSD for nonnegative variances, and is strictly better than linearization.
        """
        fx, fy, cx, cy = self.K.fx, self.K.fy, self.K.cx, self.K.cy

        du = float(u - cx)
        dv = float(v - cy)
        z = float(z)

        vu = float(max(var_u, 0.0))
        vv = float(max(var_v, 0.0))
        vz = float(max(var_z, 0.0))

        # Variances
        var_x = (z * z * vu + du * du * vz + vu * vz) / (fx * fx)
        var_y = (z * z * vv + dv * dv * vz + vv * vz) / (fy * fy)
        var_z = vz

        # Covariances
        cov_xy = (du * dv * vz) / (fx * fy)
        cov_xz = (du * vz) / fx
        cov_yz = (dv * vz) / fy

        cov_xyz = np.array(
            [
                [var_x,  cov_xy, cov_xz],
                [cov_xy, var_y,  cov_yz],
                [cov_xz, cov_yz, var_z],
            ],
            dtype=np.float64,
        )

        approx_event = OpReportEvent(
            op="rgbd_backprojection_covariance",
            exact=True,  # this is closed-form under stated assumptions
            approx_reason="independent_uvz_gaussian_closed_form_second_moment",
            metrics={
                "pixel_sigma": float(math.sqrt(vu)),
                "depth_sigma_eff": float(math.sqrt(vz)),
                "z_m": float(z),
                "trace_cov": float(np.trace(cov_xyz)),
                "var_u": float(vu),
                "var_v": float(vv),
                "var_z": float(vz),
            },
        )
        return cov_xyz, approx_event

    def _grid_features(self, gray: np.ndarray) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Deterministic fallback when cv2 is not available:
        - Sample a grid of points
        - Use a small normalized patch as descriptor
        """
        H, W = gray.shape[:2]
        step = max(8, int(min(H, W) / 40))
        pts: List[Tuple[float, float]] = []
        descs: List[np.ndarray] = []
        patch = 9
        r = patch // 2

        for y in range(r, H - r, step):
            for x in range(r, W - r, step):
                pts.append((float(x), float(y)))
                p = gray[y - r : y + r + 1, x - r : x + r + 1].astype(np.float32)
                p = (p - p.mean()) / (p.std() + 1e-6)
                descs.append(p.flatten())

                if len(pts) >= self.config.max_features:
                    break
            if len(pts) >= self.config.max_features:
                break

        desc = (
            np.stack(descs, axis=0).astype(np.float32)
            if descs
            else np.zeros((0, patch * patch), np.float32)
        )
        return pts, desc

    def feature_to_splat_3dgs(
        self,
        feat: Feature3D,
        color: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        3DGS-compatible splat export (7): mean, scale, rot (quat), opacity, color.
        Sigma = cov_xyz; eigendecompose; scales = sqrt(clamp(eigvals)); opacity from logdet.
        """
        Sigma = np.asarray(feat.cov_xyz, dtype=np.float64)
        eigvals, R = np.linalg.eigh(Sigma)
        eigvals = np.maximum(eigvals, 1e-12)
        smin = float(self.config.gs_scale_min)
        smax = float(self.config.gs_scale_max)
        scales = np.sqrt(np.clip(eigvals, smin, smax))
        quat = _rotmat_to_quat(R)
        logdet = float(np.sum(np.log(np.maximum(eigvals, 1e-12))))
        opacity = 1.0 / (1.0 + math.exp(-self.config.gs_opacity_gamma * (self.config.gs_logdet0 - logdet)))
        rgb = color
        if rgb is None:
            rgb = np.array([0.5, 0.5, 0.5], dtype=np.float64)
        else:
            rgb = np.asarray(rgb, dtype=np.float64).ravel()[:3]
        return {
            "mean": feat.xyz.copy(),
            "scale": scales,
            "rot": quat,
            "opacity": float(opacity),
            "color": rgb,
        }
