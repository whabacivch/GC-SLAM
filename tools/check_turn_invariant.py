#!/usr/bin/env python3
"""
Check the left-turn invariant:
  For a left (CCW) turn on flat ground:
    mean gyro_base_z > 0
    mean accel_base_y > 0
If gyro_base_z < 0 while accel_base_y > 0, gyro Z is flipped vs accel.

This tool reads rosbag2 sqlite directly (no rosbag2_py) and uses ROS
message deserialization via rclpy.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from rosbag_sqlite_utils import resolve_db3_path, topic_id, topic_type
from fl_slam_poc.common import constants


@dataclass
class TurnInvariantReport:
    bag_path: str
    db_path: str
    imu_topic: str
    odom_topic: str
    t_start_sec: float
    t_end_sec: float
    n_odom: int
    n_imu: int
    yaw_rate_mean: float
    yaw_rate_min: float
    yaw_rate_max: float
    gyro_base_z_mean: float
    accel_base_y_mean: float
    gyro_base_z_sign: int
    accel_base_y_sign: int
    mismatch_flag: bool
    R_base_imu_rotvec: List[float]
    accel_scale: float
    yaw_rate_threshold: float


def _rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _load_odom(cur, odom_topic: str) -> Tuple[np.ndarray, np.ndarray]:
    oid = topic_id(cur, odom_topic)
    if oid is None:
        raise RuntimeError(f"topic not found: {odom_topic}")
    otype = topic_type(cur, odom_topic)
    msg_cls = get_message(otype)

    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp ASC",
        (oid,),
    )
    stamps = []
    yaws = []
    for ts, data in cur:
        msg = deserialize_message(data, msg_cls)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        q = msg.pose.pose.orientation
        yaw = _yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        stamps.append(float(stamp))
        yaws.append(float(yaw))
    return np.array(stamps, dtype=np.float64), np.array(yaws, dtype=np.float64)


def _find_left_turn_segment(t: np.ndarray, yaw: np.ndarray, yaw_rate_threshold: float) -> Tuple[int, int, np.ndarray]:
    # unwrap yaw then compute yaw rate
    yaw_u = np.unwrap(yaw)
    dyaw = np.diff(yaw_u)
    dt = np.diff(t)
    yaw_rate = np.zeros_like(dyaw)
    valid = dt > 1e-6
    yaw_rate[valid] = dyaw[valid] / dt[valid]

    # find longest contiguous segment with yaw_rate > threshold
    segments = []
    start = None
    for i in range(yaw_rate.size):
        if yaw_rate[i] > yaw_rate_threshold:
            if start is None:
                start = i
        else:
            if start is not None:
                segments.append((start, i - 1))
                start = None
    if start is not None:
        segments.append((start, yaw_rate.size - 1))

    if not segments:
        # fallback: use max positive yaw_rate index as single-sample segment
        idx = int(np.argmax(yaw_rate)) if yaw_rate.size else 0
        return idx, idx, yaw_rate

    # choose longest segment
    seg = max(segments, key=lambda s: (s[1] - s[0], s[0]))
    return seg[0], seg[1], yaw_rate


def _load_imu_window(cur, imu_topic: str, t0: float, t1: float) -> Tuple[np.ndarray, np.ndarray]:
    iid = topic_id(cur, imu_topic)
    if iid is None:
        raise RuntimeError(f"topic not found: {imu_topic}")
    itype = topic_type(cur, imu_topic)
    msg_cls = get_message(itype)

    t0_ns = int(t0 * 1e9)
    t1_ns = int(t1 * 1e9)
    cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp ASC",
        (iid, t0_ns, t1_ns),
    )
    gyro = []
    accel = []
    for ts, data in cur:
        msg = deserialize_message(data, msg_cls)
        g = msg.angular_velocity
        a = msg.linear_acceleration
        gyro.append([g.x, g.y, g.z])
        accel.append([a.x, a.y, a.z])
    return np.asarray(gyro, dtype=np.float64), np.asarray(accel, dtype=np.float64)


def _load_rotvec_from_yaml(path: str) -> List[float]:
    cfg = yaml.safe_load(Path(path).read_text())
    vals = None
    # Try common node keys first.
    for node_key in ("gc_backend", "fl_slam_poc"):
        if isinstance(cfg, dict) and node_key in cfg:
            params = cfg.get(node_key, {}).get("ros__parameters", {})
            if isinstance(params, dict) and "T_base_imu" in params:
                vals = params["T_base_imu"]
                break
    # Fallback: search any top-level node for T_base_imu
    if vals is None and isinstance(cfg, dict):
        for node_key, node_val in cfg.items():
            params = node_val.get("ros__parameters", {}) if isinstance(node_val, dict) else {}
            if isinstance(params, dict) and "T_base_imu" in params:
                vals = params["T_base_imu"]
                break
    if vals is None:
        raise RuntimeError(f"failed to read T_base_imu from {path}: key not found")
    if not isinstance(vals, list) or len(vals) != 6:
        raise RuntimeError(f"T_base_imu must be 6D list, got: {vals}")
    return [float(vals[3]), float(vals[4]), float(vals[5])]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory or .db3")
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--yaw-rate-threshold", type=float, default=0.05, help="rad/s")
    ap.add_argument("--config", default="fl_ws/src/fl_slam_poc/config/gc_unified.yaml")
    ap.add_argument("--R-base-imu-rotvec", type=float, nargs=3, default=None)
    ap.add_argument("--accel-scale", type=float, default=None)
    args = ap.parse_args()

    db_path = resolve_db3_path(args.bag)
    if not db_path:
        raise SystemExit(f"could not locate *.db3 under {args.bag}")

    if args.R_base_imu_rotvec is None:
        rotvec = _load_rotvec_from_yaml(args.config)
    else:
        rotvec = [float(v) for v in args.R_base_imu_rotvec]

    accel_scale = float(args.accel_scale) if args.accel_scale is not None else float(constants.GC_IMU_ACCEL_SCALE)

    R_base_imu = _rotvec_to_R(np.array(rotvec, dtype=np.float64))

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    t_odom, yaw = _load_odom(cur, args.odom_topic)
    if t_odom.size < 3:
        raise SystemExit("not enough odom samples")

    seg_start, seg_end, yaw_rate = _find_left_turn_segment(t_odom, yaw, args.yaw_rate_threshold)
    # yaw_rate is between t[i] and t[i+1], so use [t[i], t[i+1]]
    t_start = float(t_odom[seg_start])
    t_end = float(t_odom[min(seg_end + 1, t_odom.size - 1)])

    gyro_imu, accel_imu = _load_imu_window(cur, args.imu_topic, t_start, t_end)
    conn.close()

    if gyro_imu.size == 0:
        raise SystemExit("no IMU samples in selected window")

    accel_imu = accel_imu * accel_scale
    gyro_base = (R_base_imu @ gyro_imu.T).T
    accel_base = (R_base_imu @ accel_imu.T).T

    gyro_z_mean = float(np.mean(gyro_base[:, 2]))
    accel_y_mean = float(np.mean(accel_base[:, 1]))

    # yaw_rate stats over segment
    yr_seg = yaw_rate[seg_start : seg_end + 1]
    yr_mean = float(np.mean(yr_seg)) if yr_seg.size else 0.0
    yr_min = float(np.min(yr_seg)) if yr_seg.size else 0.0
    yr_max = float(np.max(yr_seg)) if yr_seg.size else 0.0

    gz_sign = int(np.sign(gyro_z_mean))
    ay_sign = int(np.sign(accel_y_mean))

    mismatch = (gz_sign < 0 and ay_sign > 0)

    report = TurnInvariantReport(
        bag_path=str(Path(args.bag).resolve()),
        db_path=str(Path(db_path).resolve()),
        imu_topic=args.imu_topic,
        odom_topic=args.odom_topic,
        t_start_sec=t_start,
        t_end_sec=t_end,
        n_odom=int(t_odom.size),
        n_imu=int(gyro_imu.shape[0]),
        yaw_rate_mean=yr_mean,
        yaw_rate_min=yr_min,
        yaw_rate_max=yr_max,
        gyro_base_z_mean=gyro_z_mean,
        accel_base_y_mean=accel_y_mean,
        gyro_base_z_sign=gz_sign,
        accel_base_y_sign=ay_sign,
        mismatch_flag=bool(mismatch),
        R_base_imu_rotvec=[float(v) for v in rotvec],
        accel_scale=accel_scale,
        yaw_rate_threshold=float(args.yaw_rate_threshold),
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
