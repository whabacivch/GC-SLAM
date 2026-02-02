#!/usr/bin/env python3
"""
Convert Kimera_Data calibration extrinsics to GC format and optionally write to config.

Dataset convention (rosbags/Kimera_Data/calibration/README.md):
  T_a_b maps coordinates in frame b into frame a → p_a = T_a_b @ p_b.
So T_baselink_lidar = p_baselink = T_baselink_lidar @ p_lidar → GC T_base_lidar (base <- lidar).
T_cameralink_gyro = gyro pose in cameralink; T_baselink_cameralink is identity so T_base_gyro = T_cameralink_gyro.
Bag uses forward_imu_optical_frame (may differ from gyro orientation); we use dataset for translation
and optionally keep bag-estimated rotation for IMU (see --imu-rotation).

Usage:
  python tools/kimera_calibration_to_gc.py rosbags/Kimera_Data/calibration/robots/acl_jackal/extrinsics.yaml
  python tools/kimera_calibration_to_gc.py ... --apply --config fl_ws/src/fl_slam_poc/config/gc_unified.yaml
  python tools/kimera_calibration_to_gc.py ... --imu-rotation -1.602693 0.002604 0  # use bag-estimated IMU rotation
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import numpy as np
from scipy.spatial.transform import Rotation

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _parse_4x4(rows: list) -> np.ndarray:
    """Parse 4x4 matrix from YAML list of rows (each row list of 4 numbers)."""
    M = np.array(rows, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got shape {M.shape}")
    return M


def _find_transform(data: dict, name: str) -> np.ndarray:
    """Get 4x4 transform matrix by name from Kimera extrinsics.yaml."""
    for t in data.get("transforms", []):
        if t.get("name") == name:
            return _parse_4x4(t["T"])
    raise KeyError(f"Transform {name!r} not found in calibration")


def _matrix_to_gc_6d(T: np.ndarray) -> list[float]:
    """Convert 4x4 T (p_a = T @ p_b) to GC format [x, y, z, rx, ry, rz] (translation m, rotvec rad)."""
    R = T[:3, :3]
    t = T[:3, 3]
    rotvec = Rotation.from_matrix(R).as_rotvec()
    return [float(t[0]), float(t[1]), float(t[2]), float(rotvec[0]), float(rotvec[1]), float(rotvec[2])]


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert Kimera calibration to GC extrinsics format")
    ap.add_argument("extrinsics_yaml", help="Path to robots/<robot>/extrinsics.yaml")
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Update gc_unified.yaml with T_base_lidar and T_base_imu (line replace).",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Path to gc_unified.yaml (default: fl_ws/src/fl_slam_poc/config/gc_unified.yaml)",
    )
    ap.add_argument(
        "--imu-rotation",
        nargs=3,
        type=float,
        default=None,
        metavar=("RX", "RY", "RZ"),
        help="Use this rotvec (rad) for IMU rotation instead of dataset (e.g. bag-estimated). Translation still from T_cameralink_gyro.",
    )
    args = ap.parse_args()

    path = Path(args.extrinsics_yaml)
    if not path.is_file():
        print(f"ERROR: File not found: {path}", file=sys.stderr)
        return 1

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # T_baselink_lidar = T_{base<-lidar} → use directly as GC T_base_lidar
    T_baselink_lidar = _find_transform(data, "T_baselink_lidar")
    T_base_lidar = _matrix_to_gc_6d(T_baselink_lidar)

    # T_cameralink_gyro = gyro in cameralink; baselink = cameralink (identity) so this is T_base_gyro.
    # Use for IMU translation. Rotation: dataset has identity; bag showed ~90° for forward_imu_optical_frame.
    T_cameralink_gyro = _find_transform(data, "T_cameralink_gyro")
    t_imu = T_cameralink_gyro[:3, 3]
    R_imu = T_cameralink_gyro[:3, :3]
    if args.imu_rotation is not None:
        rx, ry, rz = args.imu_rotation
        T_base_imu = [float(t_imu[0]), float(t_imu[1]), float(t_imu[2]), rx, ry, rz]
    else:
        T_base_imu = _matrix_to_gc_6d(T_cameralink_gyro)

    print("GC format (T_base_sensor: [x, y, z, rx, ry, rz] m, rad)")
    print(f"  T_base_lidar: {T_base_lidar}")
    print(f"  T_base_imu:   {T_base_imu}")

    if not args.apply:
        return 0

    config_path = Path(args.config) if args.config else (_PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc" / "config" / "gc_unified.yaml")
    if not config_path.is_file():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    lidar_line = f"    T_base_lidar: [{T_base_lidar[0]:.6f}, {T_base_lidar[1]:.6f}, {T_base_lidar[2]:.6f}, {T_base_lidar[3]:.6f}, {T_base_lidar[4]:.6f}, {T_base_lidar[5]:.6f}]\n"
    imu_line = f"    T_base_imu:   [{T_base_imu[0]:.6f}, {T_base_imu[1]:.6f}, {T_base_imu[2]:.6f}, {T_base_imu[3]:.6f}, {T_base_imu[4]:.6f}, {T_base_imu[5]:.6f}]\n"

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    import re
    new_lines = []
    for line in lines:
        if re.match(r"\s*T_base_lidar\s*:", line):
            new_lines.append(lidar_line)
        elif re.match(r"\s*T_base_imu\s*:", line):
            new_lines.append(imu_line)
        else:
            new_lines.append(line)

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Updated {config_path} (from Kimera calibration + {'bag-estimated IMU rotation' if args.imu_rotation else 'dataset IMU'}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
