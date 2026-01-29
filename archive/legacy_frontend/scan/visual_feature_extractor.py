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

No global iterative optimization; per-frame operator only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Literal
import time
import math
import numpy as np

try:
    import cv2  # Optional dependency
except Exception:
    cv2 = None


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


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

    # Descriptor (e.g., ORB binary)
    desc: np.ndarray  # (D,) dtype=uint8 or float32

    # Continuous measurement weight in [0,1] (for budgeting/prioritization)
    weight: float

    # Metadata for debugging/audit
    meta: Dict[str, Any]


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
DepthSampleMode = Literal["nearest", "median3", "median5"]
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
        max_features: int = 800,
        orb_nlevels: int = 8,
        orb_scale_factor: float = 1.2,
        # Depth scale: e.g. uint16 mm -> meters => depth_scale=0.001
        depth_scale: float = 1.0,
        # Pixel measurement noise (pixels) for keypoint localization
        pixel_sigma: float = 1.0,
        # Depth noise model parameters (meters) — intended as priors; calibrate later.
        depth_model: DepthModel = "linear",
        depth_sigma0: float = 0.01,
        depth_sigma_slope: float = 0.01,
        # Depth sampling
        depth_sample_mode: DepthSampleMode = "median3",
        # Soft weighting priors (no hard gates)
        min_depth_m: float = 0.05,
        max_depth_m: float = 80.0,
        # Numerical stability for precision inversion
        cov_reg_eps: float = 1e-9,
        invalid_cov_inflate: float = 1e6,  # meters^2 scale for invalid depth points
        # Response soft-weight prior
        response_soft_scale: float = 50.0,
        # Depth validity soft slope (logistic)
        depth_validity_slope: float = 5.0,
    ) -> None:
        self.K = intrinsics
        self.max_features = int(max_features)

        self.depth_scale = float(depth_scale)
        self.pixel_sigma = float(pixel_sigma)

        self.depth_model = depth_model
        self.depth_sigma0 = float(depth_sigma0)
        self.depth_sigma_slope = float(depth_sigma_slope)

        self.depth_sample_mode = depth_sample_mode
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

        self.cov_reg_eps = float(cov_reg_eps)
        self.invalid_cov_inflate = float(invalid_cov_inflate)

        self.response_soft_scale = float(response_soft_scale)
        self.depth_validity_slope = float(depth_validity_slope)

        self._orb = None
        if cv2 is not None:
            self._orb = cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=float(orb_scale_factor),
                nlevels=int(orb_nlevels),
                edgeThreshold=31,
                patchSize=31,
                fastThreshold=20,
            )

    # ----------------------------
    # Public API
    # ----------------------------

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
            if len(kps) > self.max_features:
                idx = np.argsort([-kp.response for kp in kps])[: self.max_features]
                kps = [kps[i] for i in idx]
                desc = desc[idx]
        else:
            kps, desc = self._grid_features(gray)

        features: List[Feature3D] = []
        num_total = len(kps)

        invalid_depth = 0
        depth_holes = 0

        # Per-frame quality metrics
        weights = []
        traces = []
        logdets = []

        for i, kp in enumerate(kps):
            u = float(kp.pt[0]) if hasattr(kp, "pt") else float(kp[0])
            v = float(kp.pt[1]) if hasattr(kp, "pt") else float(kp[1])

            z_m, z_var_m2, z_valid = self._depth_sample(depth, u, v)
            if not z_valid:
                depth_holes += 1

            w_depth = self._depth_weight(z_m) if z_valid else 0.0

            # Keypoint response-based soft weight (no gating)
            resp = float(getattr(kp, "response", 1.0)) if hasattr(kp, "response") else 1.0
            w_resp = self._response_weight(resp)

            # Weight contract: budgeting/prioritization only (do not also scale precision downstream unless intentional)
            weight = float(np.clip(w_depth * w_resp, 0.0, 1.0))

            # If depth invalid, produce a benign feature (tiny weight, huge covariance).
            if (not z_valid) or (not np.isfinite(z_m)) or (z_m <= 0.0):
                invalid_depth += 1
                xyz = np.array([0.0, 0.0, 0.0], dtype=np.float64)
                cov_xyz = np.eye(3, dtype=np.float64) * self.invalid_cov_inflate
                info_xyz, logdet_cov = self._precision_and_logdet(cov_xyz)
            else:
                xyz = self._backproject(u, v, z_m)
                cov_xyz, approx_ev = self._backprojection_cov_closed_form(
                    u=u,
                    v=v,
                    z=z_m,
                    var_u=self.pixel_sigma ** 2,
                    var_v=self.pixel_sigma ** 2,
                    var_z=max(z_var_m2, self._depth_sigma(z_m) ** 2),
                )
                if approx_ev is not None:
                    op_report.append(approx_ev)

                info_xyz, logdet_cov = self._precision_and_logdet(cov_xyz)

            d = desc[i].copy()

            meta = {
                "response": resp,
                "w_resp": float(w_resp),
                "depth_m": float(z_m) if np.isfinite(z_m) else float("nan"),
                "depth_valid": bool(z_valid),
                "depth_var_m2": float(z_var_m2) if np.isfinite(z_var_m2) else float("nan"),
                "w_depth": float(w_depth),
                "weight": float(weight),
            }

            features.append(
                Feature3D(
                    u=u,
                    v=v,
                    xyz=xyz,
                    cov_xyz=cov_xyz,
                    info_xyz=info_xyz,
                    logdet_cov=float(logdet_cov),
                    desc=d,
                    weight=weight,
                    meta=meta,
                )
            )

            weights.append(weight)
            traces.append(float(np.trace(cov_xyz)))
            logdets.append(float(logdet_cov))

        op_report.append(
            OpReportEvent(
                op="visual_feature_extraction",
                exact=True if self._orb is not None else False,
                approx_reason="" if self._orb is not None else "opencv_missing_grid_fallback",
                metrics={
                    "num_features": len(features),
                    "num_input_kps": num_total,
                    "num_invalid_depth": int(invalid_depth),
                    "num_depth_holes": int(depth_holes),
                    "depth_sample_mode": self.depth_sample_mode,
                    "method": "ORB" if self._orb is not None else "GRID_PATCH",
                    "weight_mean": float(np.mean(weights)) if weights else 0.0,
                    "weight_p10": float(np.percentile(weights, 10)) if weights else 0.0,
                    "weight_p50": float(np.percentile(weights, 50)) if weights else 0.0,
                    "weight_p90": float(np.percentile(weights, 90)) if weights else 0.0,
                    "trace_cov_mean": float(np.mean(traces)) if traces else 0.0,
                    "logdet_cov_mean": float(np.mean(logdets)) if logdets else 0.0,
                },
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

    def _depth_sample(self, depth: np.ndarray, u: float, v: float) -> Tuple[float, float, bool]:
        """
        Depth sampling that can be robust (median window) and returns a local variance estimate.

        Returns:
            z_m: depth in meters
            z_var_m2: local depth variance estimate (meters^2), NaN if unavailable
            valid: whether a plausible depth was obtained
        """
        x = int(round(u))
        y = int(round(v))
        H, W = depth.shape[:2]
        if x < 0 or y < 0 or x >= W or y >= H:
            return float("nan"), float("nan"), False

        if self.depth_sample_mode == "nearest":
            z = self._depth_to_meters(depth[y, x])
            return self._validate_depth(z)

        # Windowed robust sampling
        if self.depth_sample_mode == "median3":
            r = 1
        elif self.depth_sample_mode == "median5":
            r = 2
        else:
            raise ValueError(f"Unknown depth_sample_mode={self.depth_sample_mode}")

        x0 = max(0, x - r)
        x1 = min(W - 1, x + r)
        y0 = max(0, y - r)
        y1 = min(H - 1, y + r)

        window = depth[y0 : y1 + 1, x0 : x1 + 1].reshape(-1)
        zs = np.array([self._depth_to_meters(vv) for vv in window], dtype=np.float64)
        zs = zs[np.isfinite(zs)]
        if zs.size == 0:
            return float("nan"), float("nan"), False

        # Reject non-positive depths
        zs = zs[zs > 0.0]
        if zs.size == 0:
            return float("nan"), float("nan"), False

        z_med = float(np.median(zs))
        z_var = float(np.var(zs)) if zs.size >= 4 else float("nan")

        z_med, z_var, valid = self._validate_depth(z_med, var_override=z_var)
        return z_med, z_var, valid

    def _depth_to_meters(self, z_raw: Any) -> float:
        if z_raw is None:
            return float("nan")
        # Works for uint16, float32, etc.
        try:
            return float(z_raw) * self.depth_scale
        except Exception:
            return float("nan")

    def _validate_depth(self, z_m: float, var_override: Optional[float] = None) -> Tuple[float, float, bool]:
        if (not np.isfinite(z_m)) or (z_m <= 0.0):
            return float("nan"), float("nan"), False

        if z_m < self.min_depth_m or z_m > self.max_depth_m:
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
        a = self.depth_validity_slope
        w_min = 1.0 / (1.0 + math.exp(-a * (z - self.min_depth_m)))
        w_max = 1.0 / (1.0 + math.exp(+a * (z - self.max_depth_m)))
        return float(np.clip(w_min * w_max, 0.0, 1.0))

    def _response_weight(self, response: float) -> float:
        """Continuous weight from keypoint response (no gating)."""
        s = self.response_soft_scale
        return float(response / (response + s)) if response > 0 else 0.0

    def _depth_sigma(self, z_m: float) -> float:
        """
        Depth noise model (meters). Closed-form and selectable.
        """
        z = abs(float(z_m))
        if self.depth_model == "linear":
            return self.depth_sigma0 + self.depth_sigma_slope * z
        if self.depth_model == "quadratic":
            return self.depth_sigma0 + self.depth_sigma_slope * (z ** 2)
        raise ValueError(f"Unknown depth_model={self.depth_model}")

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        return np.array([x, y, z], dtype=np.float64)

    def _precision_and_logdet(self, cov_xyz: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute precision and logdet with a tiny diagonal regularization for stability.
        """
        cov = cov_xyz.astype(np.float64, copy=False)
        cov_reg = cov + (self.cov_reg_eps * np.eye(3, dtype=np.float64))
        try:
            sign, logdet = np.linalg.slogdet(cov_reg)
            if sign <= 0:
                # Force PSD-ish by additional reg if needed
                cov_reg = cov + (1e-6 * np.eye(3, dtype=np.float64))
                sign, logdet = np.linalg.slogdet(cov_reg)
            info = np.linalg.inv(cov_reg)
            return info, float(logdet)
        except np.linalg.LinAlgError:
            # Worst-case fallback
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

                if len(pts) >= self.max_features:
                    break
            if len(pts) >= self.max_features:
                break

        desc = (
            np.stack(descs, axis=0).astype(np.float32)
            if descs
            else np.zeros((0, patch * patch), np.float32)
        )
        return pts, desc
