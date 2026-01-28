#!/usr/bin/env python3
"""
Validate IMU gyro axis/sign conventions against odometry orientation changes.

Goal: distinguish between
  (A) native IMU gyro axes/sign differ from what we assume (even if accel looks fine), vs
  (B) a rotation-convention mismatch inside the estimator (right/left perturbation, composition direction).

Method (dataset-driven, deterministic):
  1) Compute odom yaw-rate from consecutive /odom quaternions (finite difference).
  2) Map raw IMU gyro to base using the configured R_base_imu (rotvec).
  3) Compare omega_base.z vs odom yaw-rate:
       - correlation
       - best-fit scale (yaw_rate ~= s * omega_base.z)
       - sign agreement
  4) Exhaustively test the 8 diagonal sign-flip variants applied to gyro_imu BEFORE R_base_imu:
       gyro_base = R_base_imu @ (S @ gyro_imu), S = diag(sx,sy,sz), sx,sy,sz in {+1,-1}
     and report the best variant (highest |corr|, then highest sign agreement).

This script does NOT change runtime behavior; it produces an audit report.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]], dtype=np.float64)
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    if float(np.std(x)) < 1e-12 or float(np.std(y)) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    # Least squares y ~= s * x (no intercept).
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.dot(x, x)) + 1e-12
    return float(np.dot(x, y) / denom)


def _sign_agreement(x: np.ndarray, y: np.ndarray) -> float:
    sx = np.sign(x)
    sy = np.sign(y)
    return float(np.mean((sx * sy) > 0.0)) if x.size else 0.0


@dataclass
class VariantScore:
    sx: int
    sy: int
    sz: int
    corr: float
    corr_abs: float
    sign_agree: float
    slope: float


@dataclass
class GyroFrameReport:
    bag_uri: str
    odom_topic: str
    imu_topic: str
    n_odom: int
    n_imu: int
    R_base_imu_rotvec: List[float]
    base_mapping_corr: float
    base_mapping_sign_agree: float
    base_mapping_slope: float
    best_variant: VariantScore
    all_variants: List[VariantScore]


def _load_msgs(bag_uri: str, odom_topic: str, imu_topic: str, max_odom: int, max_imu: int):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    odom_cls = get_message(types[odom_topic])
    imu_cls = get_message(types[imu_topic])

    odom = []
    imu = []
    while reader.has_next() and (len(odom) < max_odom or len(imu) < max_imu):
        tpc, data, _ = reader.read_next()
        if tpc == odom_topic and len(odom) < max_odom:
            msg = deserialize_message(data, odom_cls)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            odom.append((float(stamp), msg))
        elif tpc == imu_topic and len(imu) < max_imu:
            msg = deserialize_message(data, imu_cls)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            imu.append((float(stamp), msg))
    odom.sort(key=lambda x: x[0])
    imu.sort(key=lambda x: x[0])
    return odom, imu


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--max-odom", type=int, default=3504)
    ap.add_argument("--max-imu", type=int, default=35032)
    ap.add_argument("--R-base-imu-rotvec", type=float, nargs=3, default=[-0.015586, 0.489293, 0.0])
    args = ap.parse_args()

    bag_uri = str(Path(args.bag).resolve())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    odom_msgs, imu_msgs = _load_msgs(bag_uri, args.odom_topic, args.imu_topic, args.max_odom, args.max_imu)

    # Odom yaw-rate (finite difference).
    odom_t = np.array([s for s, _ in odom_msgs], dtype=np.float64)
    yaw = np.array([_yaw_from_quat_xyzw(m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w) for _, m in odom_msgs])
    yaw_u = np.unwrap(yaw)
    dyaw = np.diff(yaw_u)
    dt = np.diff(odom_t)
    valid = dt > 1e-6
    yaw_rate = np.zeros_like(dyaw)
    yaw_rate[valid] = dyaw[valid] / dt[valid]
    odom_t_mid = odom_t[:-1]

    # IMU gyro
    imu_t = np.array([s for s, _ in imu_msgs], dtype=np.float64)
    gyro_imu = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in imu_msgs], dtype=np.float64)

    # map yaw_rate to IMU timestamps (nearest-left odom interval)
    idx = np.searchsorted(odom_t_mid, imu_t, side="right") - 1
    idx = np.clip(idx, 0, yaw_rate.size - 1)
    yr_i = yaw_rate[idx]

    R_base_imu = _rotvec_to_R(np.array(args.R_base_imu_rotvec, dtype=np.float64))

    def score_variant(sx: int, sy: int, sz: int) -> VariantScore:
        S = np.diag([sx, sy, sz]).astype(np.float64)
        gb = (R_base_imu @ (S @ gyro_imu.T)).T
        oz = gb[:, 2]
        c = _corr(oz, yr_i)
        return VariantScore(
            sx=sx,
            sy=sy,
            sz=sz,
            corr=float(c),
            corr_abs=float(abs(c)),
            sign_agree=float(_sign_agreement(oz, yr_i)),
            slope=float(_slope(oz, yr_i)),
        )

    variants = [score_variant(sx, sy, sz) for sx, sy, sz in itertools.product([1, -1], repeat=3)]
    variants_sorted = sorted(variants, key=lambda v: (v.corr_abs, v.sign_agree), reverse=True)
    best = variants_sorted[0]

    # baseline (no sign flips)
    base = next(v for v in variants if (v.sx, v.sy, v.sz) == (1, 1, 1))

    rep = GyroFrameReport(
        bag_uri=bag_uri,
        odom_topic=args.odom_topic,
        imu_topic=args.imu_topic,
        n_odom=len(odom_msgs),
        n_imu=len(imu_msgs),
        R_base_imu_rotvec=[float(x) for x in args.R_base_imu_rotvec],
        base_mapping_corr=float(base.corr),
        base_mapping_sign_agree=float(base.sign_agree),
        base_mapping_slope=float(base.slope),
        best_variant=best,
        all_variants=variants_sorted,
    )

    out_path.write_text(json.dumps(asdict(rep), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(rep), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

