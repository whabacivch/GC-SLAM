#!/usr/bin/env python3
"""
Print extrinsics (T_base_lidar, T_base_imu) and related frame names from config.

Usage:
  python tools/check_extrinsics.py <config_path>
  python tools/check_extrinsics.py <config_path> --backend-key gc_backend

Loads the backend section from the given YAML; prints T_base_lidar, T_base_imu
(6D [x,y,z,rx,ry,rz] rotvec rad) and which frames they relate:
  base_frame, and note that lidar_frame / imu_frame are the PointCloud2 and IMU
  header.frame_id from the bag (see docs/KIMERA_DATASET_AND_PIPELINE.md).
"""

import argparse
import os
import sys

import yaml

# Project root
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def load_backend_params(config_path: str, backend_key: str = "gc_backend") -> dict:
    """Load backend section from YAML. Fail if file missing or section missing."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not data or backend_key not in data:
        raise ValueError(f"Config missing key {backend_key!r}: {config_path}")
    section = data[backend_key]
    params = section.get("ros__parameters", section) if isinstance(section, dict) else {}
    return dict(params)


def load_6d_from_file(path: str, key: str) -> list:
    """Load 6D [x,y,z,rx,ry,rz] from YAML file (list or dict with key)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Extrinsics file missing: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        raise ValueError(f"Extrinsics file empty or invalid: {path}")
    if isinstance(data, list):
        raw = data
    elif isinstance(data, dict) and key in data:
        raw = data[key]
    else:
        raise ValueError(f"File must be a 6D list or dict with key {key!r}: {path}")
    v = list(raw)
    if len(v) != 6:
        raise ValueError(f"Expected 6D [x,y,z,rx,ry,rz], got length {len(v)} in {path}")
    return v


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Print extrinsics and frame names from GC backend config"
    )
    ap.add_argument("config_path", help="Path to GC config YAML (e.g. gc_kimera.yaml)")
    ap.add_argument(
        "--backend-key",
        default="gc_backend",
        help="YAML key for backend params (default: gc_backend)",
    )
    args = ap.parse_args()

    params = load_backend_params(args.config_path, args.backend_key)
    base_frame = params.get("base_frame", "base_frame")
    odom_frame = params.get("odom_frame", "odom_frame")
    extrinsics_source = str(params.get("extrinsics_source", "inline")).strip().lower()

    if extrinsics_source == "file":
        lidar_file = str(params.get("T_base_lidar_file", "")).strip()
        imu_file = str(params.get("T_base_imu_file", "")).strip()
        if not lidar_file or not imu_file:
            print("ERROR: extrinsics_source=file but T_base_lidar_file or T_base_imu_file empty", file=sys.stderr)
            return 1
        # Resolve paths relative to config dir
        config_dir = os.path.dirname(os.path.abspath(args.config_path))
        lidar_path = lidar_file if os.path.isabs(lidar_file) else os.path.join(config_dir, lidar_file)
        imu_path = imu_file if os.path.isabs(imu_file) else os.path.join(config_dir, imu_file)
        T_base_lidar = load_6d_from_file(lidar_path, "T_base_lidar")
        T_base_imu = load_6d_from_file(imu_path, "T_base_imu")
    else:
        T_base_lidar = list(params.get("T_base_lidar", [0.0] * 6))
        T_base_imu = list(params.get("T_base_imu", [0.0] * 6))
        if len(T_base_lidar) != 6 or len(T_base_imu) != 6:
            print("ERROR: T_base_lidar or T_base_imu must be 6D in config", file=sys.stderr)
            return 1

    print("=" * 60)
    print("EXTRINSICS (T_{base<-sensor}) from config")
    print("=" * 60)
    print(f"Config: {args.config_path}")
    print(f"base_frame:  {base_frame}")
    print(f"odom_frame:  {odom_frame}")
    print()
    print("T_base_lidar [x, y, z, rx, ry, rz] (m, rad):")
    print(f"  {T_base_lidar}")
    print("  → relates: base_frame <- pointcloud_frame_id (LiDAR header.frame_id from bag)")
    print()
    print("T_base_imu [x, y, z, rx, ry, rz] (m, rad):")
    print(f"  {T_base_imu}")
    print("  → relates: base_frame <- imu_frame_id (IMU header.frame_id from bag)")
    print()
    print("See docs/KIMERA_DATASET_AND_PIPELINE.md for Kimera frame names and hardware.")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
