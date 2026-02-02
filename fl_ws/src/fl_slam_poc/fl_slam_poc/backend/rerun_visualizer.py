"""
Rerun visualization: map points and trajectory (Wayland-friendly; replaces RViz).

Logs PrimitiveMap as Points3D and trajectory as LineStrips3D. Optional: spawn
viewer or save to .rrd file and open with `rerun recording.rrd`.

Reference: .cursor/plans/visual_lidar_rendering_integration_*.plan.md Section 9
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def _ensure_rerun():
    """Lazy import so rerun is optional when use_rerun=False."""
    try:
        import rerun as rr
        return rr
    except ImportError:
        return None


def _set_rerun_time(rr, time_sec: float) -> None:
    """Set current time on the 'time' timeline across rerun API versions."""
    if hasattr(rr, "set_time_seconds"):
        rr.set_time_seconds("time", time_sec)
    else:
        rr.set_time("time", timestamp=time_sec)


class RerunVisualizer:
    """
    Log GC map and trajectory to Rerun (Wayland-friendly viewer).

    Call init() once when use_rerun is True; then log_map() and log_trajectory()
    from the backend when publishing state.
    """

    def __init__(
        self,
        application_id: str = "fl_slam_poc",
        spawn: bool = False,
        recording_path: Optional[str] = None,
    ):
        self._application_id = application_id
        self._spawn = spawn
        self._recording_path = recording_path
        self._initialized = False
        self._rr = None

    def init(self) -> bool:
        """Initialize Rerun (spawn viewer and/or record to file). Returns True if active."""
        if self._initialized:
            return self._rr is not None
        rr = _ensure_rerun()
        if rr is None:
            return False
        self._rr = rr
        rr.init(
            application_id=self._application_id,
            default_enabled=True,
            spawn=self._spawn,
        )
        # If recording to file: must call save() before any log (Rerun API).
        if self._recording_path and not self._spawn:
            rr.save(self._recording_path)
        self._initialized = True
        return True

    def log_map(
        self,
        positions: np.ndarray,
        weights: np.ndarray,
        colors: Optional[np.ndarray],
        time_sec: float,
    ) -> None:
        """
        Log primitive map as Points3D.

        positions: (N, 3), weights: (N,), colors: (N, 3) or None.
        """
        if self._rr is None:
            return
        rr = self._rr
        _set_rerun_time(rr, time_sec)
        n = positions.shape[0]
        if n == 0:
            rr.log("gc/map/points", rr.Points3D(positions=np.zeros((0, 3))))
            return
        # Radii from weight (clamp for display)
        radii = np.clip(weights.astype(np.float32), 0.01, 0.5)
        pos_f = positions.astype(np.float32)
        if colors is not None and colors.shape[0] == n:
            c = np.clip(colors, 0.0, 1.0).astype(np.float32)
            rr.log("gc/map/points", rr.Points3D(positions=pos_f, colors=c, radii=radii))
        else:
            rr.log("gc/map/points", rr.Points3D(positions=pos_f, radii=radii))

    def log_rgbd(
        self,
        rgb: Optional[np.ndarray],
        depth: Optional[np.ndarray],
        time_sec: float,
    ) -> None:
        """
        Log RGB + depth images to Rerun.

        rgb: (H, W, 3) uint8 or float; depth: (H, W) float meters.
        """
        if self._rr is None:
            return
        rr = self._rr
        _set_rerun_time(rr, time_sec)
        if rgb is not None:
            rgb_u8 = np.asarray(rgb)
            if rgb_u8.dtype != np.uint8:
                rgb_u8 = np.clip(rgb_u8, 0, 255).astype(np.uint8)
            rr.log("gc/camera/rgb", rr.Image(rgb_u8))
        if depth is not None:
            depth_f = np.asarray(depth, dtype=np.float32)
            if hasattr(rr, "DepthImage"):
                rr.log("gc/camera/depth", rr.DepthImage(depth_f))
            else:
                rr.log("gc/camera/depth", rr.Image(depth_f))

    def log_lidar(self, points: np.ndarray, time_sec: float) -> None:
        """
        Log LiDAR points as Points3D.

        points: (N, 3) in base/world frame.
        """
        if self._rr is None:
            return
        rr = self._rr
        _set_rerun_time(rr, time_sec)
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        rr.log("gc/lidar/points", rr.Points3D(positions=pts))

    def log_trajectory(self, positions_xyz: np.ndarray, time_sec: float) -> None:
        """
        Log trajectory as LineStrips3D (single strip).

        positions_xyz: (N, 3) path points in order.
        """
        if self._rr is None:
            return
        rr = self._rr
        _set_rerun_time(rr, time_sec)
        if positions_xyz.shape[0] == 0:
            rr.log("gc/trajectory", rr.LineStrips3D([np.zeros((0, 3))]))
            return
        rr.log(
            "gc/trajectory",
            rr.LineStrips3D([positions_xyz.astype(np.float32)]),
        )

    def flush(self) -> None:
        """Flush recording (call at shutdown if desired)."""
        if self._rr is not None:
            try:
                rec = self._rr.get_global_data_recording()
                if rec is not None:
                    rec.flush()
            except Exception:
                pass
