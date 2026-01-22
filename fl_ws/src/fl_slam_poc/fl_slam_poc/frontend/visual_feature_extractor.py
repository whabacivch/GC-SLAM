"""
Visual feature extractor for FL-SLAM POC.

Design commitments:
- No hard gating: never "reject" measurements; always return continuous weights/uncertainty.
- Closed-form-first: feature detection/description is deterministic; uncertainty propagation is analytic.
- Approximation auditing: any approximate uncertainty model emits OpReport entries and optional Frobenius correction hooks.
- No global iterative re-optimization: per-frame operator only.

Assumptions:
- RGB and depth are time-aligned upstream (or passed with timestamps and aligned before calling extract()).
- Depth is metric (meters) OR scaled by depth_scale.
- Camera model: pinhole with intrinsics (fx, fy, cx, cy).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import time
import math
import numpy as np

try:
    import cv2  # Optional dependency
except Exception:
    cv2 = None


@dataclass(frozen=True)
class PinholeIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Feature3D:
    """A single visual feature lifted to 3D with uncertainty and a soft weight."""
    # Pixel location
    u: float
    v: float

    # 3D point in camera frame
    xyz: np.ndarray  # shape (3,)

    # 3x3 covariance in camera frame (approx)
    cov_xyz: np.ndarray  # shape (3,3)

    # Descriptor (e.g., ORB binary)
    desc: np.ndarray  # shape (D,) dtype=uint8 or float32

    # Continuous measurement weight in [0, 1]
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


class VisualFeatureExtractor:
    """
    Extracts keypoints+descriptors from RGB and lifts them to 3D using depth.

    Default backend:
    - ORB (fast, deterministic) if OpenCV available
    - If OpenCV missing, falls back to a simple grid-sampler with patch descriptors
      (still deterministic, but weaker).
    """

    def __init__(
        self,
        intrinsics: PinholeIntrinsics,
        max_features: int = 800,
        orb_nlevels: int = 8,
        orb_scale_factor: float = 1.2,
        depth_scale: float = 1.0,
        # Depth noise model (meters). Keep as parameters; treat as priors in your guide.
        depth_sigma0: float = 0.01,   # base depth noise at near range
        depth_sigma_slope: float = 0.01,  # extra noise that grows with depth
        # Pixel measurement noise (pixels) for keypoint localization
        pixel_sigma: float = 1.0,
        # Soft weighting parameters (do not hard gate)
        min_depth_m: float = 0.05,
        max_depth_m: float = 80.0,
    ) -> None:
        self.K = intrinsics
        self.max_features = int(max_features)
        self.depth_scale = float(depth_scale)
        self.depth_sigma0 = float(depth_sigma0)
        self.depth_sigma_slope = float(depth_sigma_slope)
        self.pixel_sigma = float(pixel_sigma)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)

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

    def extract(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        timestamp_ns: Optional[int] = None,
    ) -> ExtractionResult:
        """
        Args:
            rgb: HxWx3 uint8 (BGR or RGB accepted; we convert to gray)
            depth: HxW depth image. uint16 (mm) or float32 (m); scaled by depth_scale.
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
        if depth.shape[0] != rgb.shape[0] or depth.shape[1] != rgb.shape[1]:
            raise ValueError("rgb and depth must have same H,W")

        gray = self._to_gray(rgb)

        if self._orb is not None:
            kps, desc = self._orb.detectAndCompute(gray, None)
            if kps is None:
                kps = []
            if desc is None:
                desc = np.zeros((0, 32), dtype=np.uint8)
            # Trim deterministically by response
            if len(kps) > self.max_features:
                idx = np.argsort([-kp.response for kp in kps])[: self.max_features]
                kps = [kps[i] for i in idx]
                desc = desc[idx]
        else:
            # Fallback: deterministic grid sampler + small patch descriptor
            kps, desc = self._grid_features(gray)

        features: List[Feature3D] = []
        num_invalid_depth = 0
        num_total = len(kps)

        for i, kp in enumerate(kps):
            u = float(kp.pt[0]) if hasattr(kp, "pt") else float(kp[0])
            v = float(kp.pt[1]) if hasattr(kp, "pt") else float(kp[1])

            z = self._depth_at(depth, u, v)  # meters
            # Continuous weight from depth validity (NO gating)
            w_depth = self._depth_weight(z)

            # If depth is nonsense, we still return a feature with huge covariance and tiny weight.
            if not np.isfinite(z) or z <= 0.0:
                num_invalid_depth += 1
                z = 1.0  # placeholder to keep math finite

            xyz = self._backproject(u, v, z)
            cov_xyz, approx_ev = self._propagate_covariance(u, v, z)

            if approx_ev is not None:
                op_report.append(approx_ev)

            # Descriptor row
            d = desc[i].copy()
            # Keypoint response-based soft weight (also no gating)
            resp = float(getattr(kp, "response", 1.0)) if hasattr(kp, "response") else 1.0
            w_resp = self._response_weight(resp)

            weight = float(np.clip(w_depth * w_resp, 0.0, 1.0))

            features.append(
                Feature3D(
                    u=u,
                    v=v,
                    xyz=xyz,
                    cov_xyz=cov_xyz,
                    desc=d,
                    weight=weight,
                    meta={
                        "response": resp,
                        "depth_m": float(z),
                        "w_depth": float(w_depth),
                        "w_resp": float(w_resp),
                    },
                )
            )

        op_report.append(
            OpReportEvent(
                op="visual_feature_extraction",
                exact=True if self._orb is not None else False,
                approx_reason="" if self._orb is not None else "opencv_missing_grid_fallback",
                metrics={
                    "num_features": len(features),
                    "num_input_kps": num_total,
                    "num_invalid_depth": num_invalid_depth,
                    "method": "ORB" if self._orb is not None else "GRID_PATCH",
                },
            )
        )

        return ExtractionResult(features=features, op_report=op_report, timestamp_ns=ts)

    # ----------------------------
    # Internals
    # ----------------------------

    def _to_gray(self, rgb: np.ndarray) -> np.ndarray:
        if cv2 is not None:
            # Handle either RGB or BGR; converting via cvtColor is robust enough if format consistent upstream.
            return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if rgb.dtype == np.uint8 else rgb.mean(axis=2).astype(np.uint8)
        return rgb.mean(axis=2).astype(np.uint8)

    def _depth_at(self, depth: np.ndarray, u: float, v: float) -> float:
        """Nearest-neighbor depth lookup with scale. No filtering (closed-form-first)."""
        x = int(round(u))
        y = int(round(v))
        if x < 0 or y < 0 or y >= depth.shape[0] or x >= depth.shape[1]:
            return float("nan")

        z_raw = depth[y, x]
        if z_raw is None:
            return float("nan")

        z = float(z_raw) * self.depth_scale
        # If depth is uint16 mm and depth_scale not set, caller should set depth_scale=0.001
        return z

    def _depth_weight(self, z: float) -> float:
        """
        Continuous weight based on depth plausibility.
        - No gating; returns a smooth downweight.
        """
        if not np.isfinite(z):
            return 0.0
        # Softly downweight outside [min_depth, max_depth]
        # Use a smooth logistic-like curve (still "a parameter", but interpret as a prior on validity).
        a = 5.0  # slope; in your spec, treat as a prior strength / risk budget parameter
        w_min = 1.0 / (1.0 + math.exp(-a * (z - self.min_depth_m)))
        w_max = 1.0 / (1.0 + math.exp(+a * (z - self.max_depth_m)))
        return float(np.clip(w_min * w_max, 0.0, 1.0))

    def _response_weight(self, response: float) -> float:
        """Continuous weight from keypoint response (no gating)."""
        # Monotone mapping; normalize by a soft scale.
        # Treat scale as a prior hyperparameter (learnable later).
        s = 50.0
        return float(response / (response + s)) if response > 0 else 0.0

    def _backproject(self, u: float, v: float, z: float) -> np.ndarray:
        x = (u - self.K.cx) * z / self.K.fx
        y = (v - self.K.cy) * z / self.K.fy
        return np.array([x, y, z], dtype=np.float64)

    def _propagate_covariance(
        self, u: float, v: float, z: float
    ) -> Tuple[np.ndarray, Optional[OpReportEvent]]:
        """
        Analytic covariance propagation for backprojection:
            X = (u-cx) z / fx
            Y = (v-cy) z / fy
            Z = z

        We approximate (u,v,z) as independent with variances:
            Var(u)=Var(v)=pixel_sigma^2
            Var(z)=sigma_z(z)^2

        This is an approximation (ignores correlation), so we emit an OpReportEvent.
        """
        fx, fy, cx, cy = self.K.fx, self.K.fy, self.K.cx, self.K.cy

        sigma_u2 = float(self.pixel_sigma ** 2)
        sigma_v2 = float(self.pixel_sigma ** 2)
        sigma_z = self.depth_sigma0 + self.depth_sigma_slope * abs(z)
        sigma_z2 = float(sigma_z ** 2)

        du = (u - cx)
        dv = (v - cy)

        # Jacobian J = d[X,Y,Z]/d[u,v,z]
        # X_u = z/fx, X_v = 0,    X_z = du/fx
        # Y_u = 0,    Y_v = z/fy, Y_z = dv/fy
        # Z_u = 0,    Z_v = 0,    Z_z = 1
        J = np.array(
            [
                [z / fx, 0.0, du / fx],
                [0.0, z / fy, dv / fy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

        Sigma_uvz = np.diag([sigma_u2, sigma_v2, sigma_z2]).astype(np.float64)
        cov_xyz = J @ Sigma_uvz @ J.T

        # Hook for Frobenius correction: if you have a 3rd-order tensor correction
        # for the local chart (camera->SE3 lift), apply it here. For now we only report.
        approx_event = OpReportEvent(
            op="rgbd_backprojection_covariance",
            exact=False,
            approx_reason="independent_uvz_gaussian_linear_cov_propagation",
            metrics={
                "pixel_sigma": self.pixel_sigma,
                "depth_sigma": sigma_z,
                "z_m": float(z),
                "trace_cov": float(np.trace(cov_xyz)),
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

        desc = np.stack(descs, axis=0).astype(np.float32) if descs else np.zeros((0, patch * patch), np.float32)
        # In this fallback, kp is just (u,v)
        return pts, desc
