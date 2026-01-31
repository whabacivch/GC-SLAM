#!/usr/bin/env python3
"""
Run full Kimera bag inspection and optionally apply estimated extrinsics to gc_kimera.yaml.

Runs: first_n_messages_summary, diagnose_coordinate_frames,
      estimate_lidar_base_extrinsic_rotation_from_ground, estimate_imu_base_extrinsic_rotation.
Parses suggested T_base_lidar and T_base_imu (rotation only; translation 0,0,0) and,
with --apply, updates config so extrinsics are no longer placeholders.

Requires: ROS 2 sourced (e.g. source /opt/ros/jazzy/setup.bash) so rclpy/deserialize_message work.

Usage:
  source /opt/ros/jazzy/setup.bash
  python tools/inspect_kimera_bag.py /path/to/Kimera_Data/ros2/10_14_acl_jackal-005_ros2
  python tools/inspect_kimera_bag.py /path/to/bag_dir --apply
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], cwd: Path | None = None, capture: bool = True) -> tuple[int, str]:
    r = subprocess.run(
        cmd,
        cwd=cwd or _PROJECT_ROOT,
        capture_output=capture,
        text=True,
    )
    out = (r.stdout or "") + (r.stderr or "")
    return r.returncode, out


def _parse_6d_from_line(line: str, key: str) -> list[float] | None:
    """Parse 'T_base_imu: [0, 0, 0, rx, ry, rz]' or 'T_base_lidar: [x, y, z, rx, ry, rz]' (x,y,z literal → use 0,0,0)."""
    m = re.search(rf"{re.escape(key)}\s*:\s*\[([^\]]+)\]", line)
    if not m:
        return None
    parts = [x.strip() for x in m.group(1).split(",")]
    if len(parts) != 6:
        return None
    try:
        return [float(x) for x in parts]
    except ValueError:
        pass
    # LiDAR tool prints literal "x", "y", "z" for translation; last 3 are rotation.
    try:
        r0, r1, r2 = float(parts[3]), float(parts[4]), float(parts[5])
        return [0.0, 0.0, 0.0, r0, r1, r2]
    except (ValueError, IndexError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Inspect Kimera bag and optionally apply estimated extrinsics to config"
    )
    ap.add_argument(
        "bag_path",
        nargs="?",
        default=None,
        help="Bag directory (or set BAG_PATH env).",
    )
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Update gc_kimera.yaml with estimated T_base_lidar and T_base_imu (rotation only; t=0,0,0).",
    )
    ap.add_argument(
        "--config",
        default=None,
        help="Path to backend config YAML to update (default: fl_ws/src/fl_slam_poc/config/gc_kimera.yaml).",
    )
    ap.add_argument(
        "--lidar-topic",
        default="/acl_jackal/lidar_points",
        help="LiDAR PointCloud2 topic.",
    )
    ap.add_argument(
        "--imu-topic",
        default="/acl_jackal/forward/imu",
        help="IMU topic.",
    )
    ap.add_argument(
        "--odom-topic",
        default="/acl_jackal/jackal_velocity_controller/odom",
        help="Odometry topic.",
    )
    args = ap.parse_args()

    bag_path = args.bag_path or __import__("os").environ.get("BAG_PATH")
    if not bag_path or not Path(bag_path).exists():
        print("ERROR: Bag path missing or not found. Set BAG_PATH or pass bag_path.", file=sys.stderr)
        return 1

    bag_path = str(Path(bag_path).resolve())
    py = sys.executable
    tools = _PROJECT_ROOT / "tools"

    print("=" * 60)
    print("KIMERA BAG INSPECTION")
    print("=" * 60)
    print(f"Bag: {bag_path}")
    print()

    # 1. First-N messages summary
    print("--- 1. First-N messages summary ---")
    code, out = _run([py, str(tools / "first_n_messages_summary.py"), bag_path, "--n", "25"])
    if code != 0:
        print(f"WARN: first_n_messages_summary exited {code}")
    else:
        print(out[:2000] + ("..." if len(out) > 2000 else ""))
    print()

    # 2. Diagnose coordinate frames (Z-up/Z-down, odom ordering)
    print("--- 2. Diagnose coordinate frames ---")
    code, out = _run([
        py, str(tools / "diagnose_coordinate_frames.py"), bag_path,
        "--lidar-topic", args.lidar_topic,
        "--imu-topic", args.imu_topic,
        "--odom-topic", args.odom_topic,
        "--n-scans", "20",
    ])
    if code != 0:
        print(f"WARN: diagnose_coordinate_frames exited {code}")
    print(out)
    print()

    # 3. Estimate LiDAR->base rotation (ground plane)
    print("--- 3. Estimate T_base_lidar rotation (ground plane) ---")
    code_lidar, out_lidar = _run([
        py, str(tools / "estimate_lidar_base_extrinsic_rotation_from_ground.py"), bag_path,
        "--lidar-topic", args.lidar_topic,
        "--skip-scans", "20",
        "--max-scans", "30",
        "--expected-height", "0.35",
        "--height-sigma", "0.25",
    ])
    if code_lidar != 0:
        print(f"WARN: estimate_lidar_base_extrinsic_rotation_from_ground exited {code_lidar}")
    print(out_lidar)
    print()

    # 4. Estimate IMU->base rotation (gravity)
    print("--- 4. Estimate T_base_imu rotation (gravity) ---")
    code_imu, out_imu = _run([
        py, str(tools / "estimate_imu_base_extrinsic_rotation.py"), bag_path,
        "--imu-topic", args.imu_topic,
        "--max-msgs", "5000",
        "--skip-msgs", "500",
    ])
    if code_imu != 0:
        print(f"WARN: estimate_imu_base_extrinsic_rotation exited {code_imu}")
    print(out_imu)
    print()

    # Parse suggested extrinsics (rotation only; translation 0,0,0)
    T_base_lidar = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    T_base_imu = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    for line in out_lidar.splitlines():
        parsed = _parse_6d_from_line(line, "T_base_lidar")
        if parsed is not None:
            T_base_lidar = parsed
            break
    for line in out_imu.splitlines():
        parsed = _parse_6d_from_line(line, "T_base_imu")
        if parsed is not None:
            T_base_imu = parsed
            break

    print("=" * 60)
    print("PARSED EXTRINSICS (rotation from estimators; translation 0,0,0)")
    print("=" * 60)
    print(f"T_base_lidar: {T_base_lidar}")
    print(f"T_base_imu:   {T_base_imu}")
    print()

    if not args.apply:
        print("Run with --apply to write these to gc_kimera.yaml (no changes made).")
        return 0

    # Update config (line-by-line to preserve comments and structure)
    config_path = Path(args.config) if args.config else (_PROJECT_ROOT / "fl_ws" / "src" / "fl_slam_poc" / "config" / "gc_kimera.yaml")
    if not config_path.is_file():
        print(f"ERROR: Config not found: {config_path}", file=sys.stderr)
        return 1

    lidar_line = f"    T_base_lidar: [{T_base_lidar[0]:.6f}, {T_base_lidar[1]:.6f}, {T_base_lidar[2]:.6f}, {T_base_lidar[3]:.6f}, {T_base_lidar[4]:.6f}, {T_base_lidar[5]:.6f}]"
    imu_line = f"    T_base_imu:   [{T_base_imu[0]:.6f}, {T_base_imu[1]:.6f}, {T_base_imu[2]:.6f}, {T_base_imu[3]:.6f}, {T_base_imu[4]:.6f}, {T_base_imu[5]:.6f}]"

    with open(config_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if re.match(r"\s*T_base_lidar\s*:", line):
            new_lines.append(lidar_line + "\n")
        elif re.match(r"\s*T_base_imu\s*:", line):
            new_lines.append(imu_line + "\n")
        else:
            new_lines.append(line)

    with open(config_path, "w", encoding="utf-8") as f:
        f.writelines(new_lines)

    print(f"Updated {config_path} with T_base_lidar and T_base_imu (no longer placeholders).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
