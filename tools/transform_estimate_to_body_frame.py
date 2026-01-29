#!/usr/bin/env python3
"""
Transform estimated trajectory (T_world<-wheel) to body frame (T_world<-body).

Use when M3DGR ground truth is in body (camera_imu) frame and our estimate
is in wheel (base_footprint) frame. Applies: T_world_body = T_world_wheel @ inv(body_T_wheel).

Usage:
  transform_estimate_to_body_frame.py <estimate.tum> <output.tum> [--calib <calib>]

Calib can be:
  - config/m3dgr_body_T_wheel.yaml (default; plain YAML with body_T_wheel_row_major_4x4)
  - M3DGR calibration.md (markdown with body_T_wheel !!opencv-matrix block; no OpenCV needed)
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

# Project root (parent of tools/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_calibration_md(body: str) -> np.ndarray:
    """Parse body_T_wheel from M3DGR calibration.md (body_T_wheel !!opencv-matrix data block)."""
    idx = body.find("body_T_wheel")
    if idx < 0:
        raise ValueError("calibration.md: body_T_wheel not found")
    # From body_T_wheel onward, find the first data: [ ... ] block
    rest = body[idx:]
    match = re.search(r"data:\s*\[\s*(.*?)\s*\]", rest, re.DOTALL)
    if not match:
        raise ValueError("calibration.md: data: [ ... ] block not found after body_T_wheel")
    raw = match.group(1)
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    arr = np.array([float(t) for t in tokens], dtype=np.float64)
    if arr.size != 16:
        raise ValueError(f"Expected 16 elements in data, got {arr.size}")
    return arr.reshape(4, 4)


def load_body_T_wheel(calib_path: Path) -> np.ndarray:
    """Load body_T_wheel 4x4. Supports: (1) calibration.md (!!opencv-matrix block), (2) m3dgr_body_T_wheel.yaml."""
    with open(calib_path, "r", encoding="utf-8") as f:
        body = f.read()
    # M3DGR calibration.md: markdown with body_T_wheel !!opencv-matrix
    if "!!opencv-matrix" in body or (calib_path.suffix.lower() == ".md" and "body_T_wheel" in body):
        return _parse_calibration_md(body)
    # Our plain YAML
    data = yaml.safe_load(body)
    if "body_T_wheel_row_major_4x4" in data:
        arr = np.array(data["body_T_wheel_row_major_4x4"], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported calib format: need body_T_wheel_row_major_4x4 or calibration.md body_T_wheel block")
    if arr.size != 16:
        raise ValueError(f"Expected 16 elements, got {arr.size}")
    return arr.reshape(4, 4)


def tum_line_to_se3(line: list) -> np.ndarray:
    """Build 4x4 SE(3) from TUM line [ts, x, y, z, qx, qy, qz, qw]. Returns T_world<-body."""
    x, y, z = line[1], line[2], line[3]
    qx, qy, qz, qw = line[4], line[5], line[6], line[7]
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T


def se3_to_tum_line(timestamp: float, T: np.ndarray) -> str:
    """Convert 4x4 SE(3) to TUM line (xyzw quaternion)."""
    R = T[:3, :3]
    t = T[:3, 3]
    quat = Rotation.from_matrix(R).as_quat()  # xyzw
    return f"{timestamp:.9f} {t[0]:.6f} {t[1]:.6f} {t[2]:.6f} {quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}\n"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Transform estimate TUM (wheel frame) to body frame for M3DGR GT comparison."
    )
    ap.add_argument("estimate_tum", help="Input TUM trajectory (T_world<-wheel)")
    ap.add_argument("output_tum", help="Output TUM trajectory (T_world<-body)")
    ap.add_argument(
        "--calib",
        type=Path,
        default=_PROJECT_ROOT / "config" / "m3dgr_body_T_wheel.yaml",
        help="Calib file: config/m3dgr_body_T_wheel.yaml or M3DGR calibration.md (body_T_wheel !!opencv-matrix)",
    )
    args = ap.parse_args()

    calib_path = args.calib if args.calib.is_absolute() else _PROJECT_ROOT / args.calib
    if not calib_path.exists():
        print(f"ERROR: Calibration file not found: {calib_path}", file=sys.stderr)
        return 1

    body_T_wheel = load_body_T_wheel(calib_path)
    wheel_T_body = np.linalg.inv(body_T_wheel)

    lines = []
    with open(args.estimate_tum, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split()
            if len(parts) >= 8:
                lines.append([float(p) for p in parts[:8]])

    if not lines:
        print("ERROR: No pose lines in estimate TUM", file=sys.stderr)
        return 1

    with open(args.output_tum, "w", encoding="utf-8") as f:
        f.write("# timestamp x y z qx qy qz qw (T_world<-body, from wheel-frame estimate)\n")
        for line in lines:
            T_world_wheel = tum_line_to_se3(line)
            T_world_body = T_world_wheel @ wheel_T_body
            f.write(se3_to_tum_line(line[0], T_world_body))

    print(f"Transformed {len(lines)} poses to body frame: {args.output_tum}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
