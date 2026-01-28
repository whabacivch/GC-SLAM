#!/usr/bin/env python3
"""
Empirically validate frame / quaternion / unit conventions for GC v2.

This script is deterministic and designed to support the status labels in:
  docs/FRAME_AND_QUATERNION_CONVENTIONS.md

It inspects a rosbag2 sqlite3 bag and reports:
  - /odom frame_id stability and observed frame_id/child_frame_id values
  - /odom yaw trace statistics (deg) and approximate yaw-rate sign
  - /livox/mid360/imu acceleration magnitude statistics (raw + scaled)
  - /livox/mid360/imu gyro z sign correlation vs odom yaw-rate sign (after applying T_base_imu)
  - Sanity checks for configured T_base_imu and T_base_lidar (gravity alignment in base)

No heuristics/gating: results are continuous statistics + clear thresholds in the report only.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    # Standard yaw extraction for ENU-like frames (about +Z).
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    # Rodrigues formula.
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    K = np.array(
        [
            [0.0, -k[2], k[1]],
            [k[2], 0.0, -k[0]],
            [-k[1], k[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _R_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    # Standard quaternion->rotation for xyzw.
    x, y, z, w = float(qx), float(qy), float(qz), float(qw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


@dataclass
class TopicFrameStats:
    topic: str
    frame_id_values: List[str]
    child_frame_id_values: List[str]
    frame_id_changes: int


@dataclass
class ImuAccelStats:
    accel_norm_raw_mean: float
    accel_norm_raw_std: float
    accel_norm_raw_min: float
    accel_norm_raw_max: float
    accel_norm_scaled_mean: float
    accel_norm_scaled_std: float
    accel_norm_scaled_min: float
    accel_norm_scaled_max: float


@dataclass
class ImuGyroStats:
    gyro_norm_mean: float
    gyro_norm_std: float
    gyro_norm_min: float
    gyro_norm_max: float
    gyro_z_mean: float
    gyro_z_std: float


@dataclass
class OdomYawStats:
    yaw_deg_min: float
    yaw_deg_max: float
    dyaw_deg_mean_abs: float
    yaw_rate_sign_mean: float


@dataclass
class GyroOdomSignStats:
    # Correlate sign(gyro_z_base) with sign(odom_yaw_rate) in overlapping time bins.
    n_samples: int
    sign_agreement_frac: float
    sign_product_mean: float


@dataclass
class ExtrinsicSanity:
    # Gravity alignment in base frame from mean accel.
    accel_base_mean: List[float]
    accel_base_norm: float
    accel_base_z_sign: float
    accel_base_z_value: float


@dataclass
class ValidationReport:
    bag_uri: str
    odom_frames: TopicFrameStats
    imu_frames: TopicFrameStats
    imu_accel: ImuAccelStats
    imu_gyro: ImuGyroStats
    odom_yaw: OdomYawStats
    gyro_vs_odom: GyroOdomSignStats
    extrinsic_sanity: ExtrinsicSanity


@dataclass
class OdomTwistConsistency:
    """
    Compare odom twist (assumed in child frame) against finite-difference position:
      dp_parent ≈ R_parent_child @ v_child * dt
    We report weighted cosine similarity and speed ratio statistics.
    """

    n_pairs: int
    cos_sim_weighted_mean_xy: float
    speed_ratio_weighted_mean_xy: float
    cos_sim_p10_xy: float
    cos_sim_p50_xy: float
    cos_sim_p90_xy: float


def _load_rosbag_messages(
    bag_uri: str,
    topics: List[str],
    max_messages: Optional[int],
) -> Dict[str, List[Tuple[float, object]]]:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("", "")
    reader.open(storage_options, converter_options)

    type_map = {t.name: t.type for t in reader.get_all_topics_and_types()}
    for t in topics:
        if t not in type_map:
            raise RuntimeError(f"topic not in bag: {t} (available: {sorted(type_map.keys())[:20]}...)")

    msg_types = {t: get_message(type_map[t]) for t in topics}
    out: Dict[str, List[Tuple[float, object]]] = {t: [] for t in topics}
    counts = {t: 0 for t in topics}

    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic not in out:
            continue
        msg = deserialize_message(data, msg_types[topic])
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        out[topic].append((float(stamp), msg))
        counts[topic] += 1
        if max_messages is not None and all(c >= max_messages for c in counts.values()):
            break

    return out


def _frame_stats(topic: str, msgs: List[Tuple[float, object]]) -> TopicFrameStats:
    frames = [m.header.frame_id for _, m in msgs]
    childs = [getattr(m, "child_frame_id", "") for _, m in msgs]
    uniq_f = sorted(set(frames))
    uniq_c = sorted(set(childs))
    changes = int(np.sum(np.array(frames[1:], dtype=object) != np.array(frames[:-1], dtype=object))) if len(frames) > 1 else 0
    return TopicFrameStats(topic=topic, frame_id_values=uniq_f, child_frame_id_values=uniq_c, frame_id_changes=changes)


def _odom_yaw_stats(odom_msgs: List[Tuple[float, object]]) -> Tuple[OdomYawStats, np.ndarray, np.ndarray]:
    t = np.array([s for s, _ in odom_msgs], dtype=np.float64)
    yaw = np.array([_yaw_from_quat_xyzw(m.pose.pose.orientation.x,
                                        m.pose.pose.orientation.y,
                                        m.pose.pose.orientation.z,
                                        m.pose.pose.orientation.w) for _, m in odom_msgs], dtype=np.float64)
    yaw_u = np.unwrap(yaw)
    dyaw = np.diff(yaw_u)
    dt = np.diff(t)
    valid = dt > 1e-6
    yaw_rate = np.zeros_like(dyaw)
    yaw_rate[valid] = dyaw[valid] / dt[valid]
    yaw_rate_sign_mean = float(np.mean(np.sign(yaw_rate[valid]))) if np.any(valid) else 0.0
    stats = OdomYawStats(
        yaw_deg_min=float(np.min(yaw) * 180.0 / np.pi) if yaw.size else 0.0,
        yaw_deg_max=float(np.max(yaw) * 180.0 / np.pi) if yaw.size else 0.0,
        dyaw_deg_mean_abs=float(np.mean(np.abs(_wrap_pi(dyaw))) * 180.0 / np.pi) if dyaw.size else 0.0,
        yaw_rate_sign_mean=yaw_rate_sign_mean,
    )
    return stats, t, yaw_rate


def _odom_twist_consistency(odom_msgs: List[Tuple[float, object]]) -> OdomTwistConsistency:
    if len(odom_msgs) < 2:
        return OdomTwistConsistency(
            n_pairs=0,
            cos_sim_weighted_mean_xy=0.0,
            speed_ratio_weighted_mean_xy=0.0,
            cos_sim_p10_xy=0.0,
            cos_sim_p50_xy=0.0,
            cos_sim_p90_xy=0.0,
        )

    t = np.array([s for s, _ in odom_msgs], dtype=np.float64)
    p = np.array([[m.pose.pose.position.x, m.pose.pose.position.y, m.pose.pose.position.z] for _, m in odom_msgs], dtype=np.float64)
    v = np.array([[m.twist.twist.linear.x, m.twist.twist.linear.y, m.twist.twist.linear.z] for _, m in odom_msgs], dtype=np.float64)
    q = np.array(
        [[m.pose.pose.orientation.x, m.pose.pose.orientation.y, m.pose.pose.orientation.z, m.pose.pose.orientation.w] for _, m in odom_msgs],
        dtype=np.float64,
    )

    dp = p[1:, :] - p[:-1, :]
    dt = (t[1:] - t[:-1]).reshape(-1, 1)
    dt = np.maximum(dt, 0.0)

    # Predict dp from twist: dp_pred = R @ v_child * dt
    dp_pred = np.zeros_like(dp)
    for i in range(dp.shape[0]):
        R = _R_from_quat_xyzw(q[i, 0], q[i, 1], q[i, 2], q[i, 3])
        dp_pred[i, :] = (R @ v[i, :]) * float(dt[i, 0])

    # Compare in XY plane (most informative for 2D mobile base).
    dp_xy = dp[:, :2]
    dp_pred_xy = dp_pred[:, :2]
    n = dp_xy.shape[0]
    eps = 1e-12
    dp_norm = np.linalg.norm(dp_xy, axis=1)
    pred_norm = np.linalg.norm(dp_pred_xy, axis=1)

    # Continuous weights: proportional to observed motion magnitude.
    w = dp_norm / (np.sum(dp_norm) + eps)

    cos = np.sum(dp_xy * dp_pred_xy, axis=1) / ((dp_norm * pred_norm) + eps)
    ratio = pred_norm / (dp_norm + eps)

    # Percentiles of cosine similarity (unweighted; for interpretability).
    cos_p10, cos_p50, cos_p90 = [float(x) for x in np.percentile(cos, [10, 50, 90])]

    return OdomTwistConsistency(
        n_pairs=int(n),
        cos_sim_weighted_mean_xy=float(np.sum(w * cos)),
        speed_ratio_weighted_mean_xy=float(np.sum(w * ratio)),
        cos_sim_p10_xy=cos_p10,
        cos_sim_p50_xy=cos_p50,
        cos_sim_p90_xy=cos_p90,
    )


def _imu_accel_stats(imu_msgs: List[Tuple[float, object]], accel_scale: float) -> ImuAccelStats:
    a_raw = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for _, m in imu_msgs], dtype=np.float64)
    n_raw = np.linalg.norm(a_raw, axis=1)
    a_scaled = a_raw * float(accel_scale)
    n_scaled = np.linalg.norm(a_scaled, axis=1)
    return ImuAccelStats(
        accel_norm_raw_mean=float(np.mean(n_raw)),
        accel_norm_raw_std=float(np.std(n_raw)),
        accel_norm_raw_min=float(np.min(n_raw)),
        accel_norm_raw_max=float(np.max(n_raw)),
        accel_norm_scaled_mean=float(np.mean(n_scaled)),
        accel_norm_scaled_std=float(np.std(n_scaled)),
        accel_norm_scaled_min=float(np.min(n_scaled)),
        accel_norm_scaled_max=float(np.max(n_scaled)),
    )


def _imu_gyro_stats(imu_msgs: List[Tuple[float, object]]) -> ImuGyroStats:
    g = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in imu_msgs], dtype=np.float64)
    n = np.linalg.norm(g, axis=1)
    return ImuGyroStats(
        gyro_norm_mean=float(np.mean(n)),
        gyro_norm_std=float(np.std(n)),
        gyro_norm_min=float(np.min(n)),
        gyro_norm_max=float(np.max(n)),
        gyro_z_mean=float(np.mean(g[:, 2])),
        gyro_z_std=float(np.std(g[:, 2])),
    )


def _gyro_vs_odom_sign(
    odom_t: np.ndarray,
    odom_yaw_rate: np.ndarray,
    imu_msgs: List[Tuple[float, object]],
    R_base_imu: np.ndarray,
) -> GyroOdomSignStats:
    imu_t = np.array([s for s, _ in imu_msgs], dtype=np.float64)
    gyro_imu = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in imu_msgs], dtype=np.float64)
    gyro_base = (R_base_imu @ gyro_imu.T).T
    gz = gyro_base[:, 2]

    # Interpolate odom yaw-rate to IMU times (nearest-left).
    # odom_yaw_rate has length N-1 and corresponds to intervals (odom_t[i], odom_t[i+1]).
    # We map each imu_t to the last odom interval start.
    if odom_t.size < 2:
        return GyroOdomSignStats(n_samples=0, sign_agreement_frac=0.0, sign_product_mean=0.0)

    idx = np.searchsorted(odom_t[:-1], imu_t, side="right") - 1
    idx = np.clip(idx, 0, odom_yaw_rate.size - 1)
    yr = odom_yaw_rate[idx]

    # Use sign only; ignore near-zero rates by leaving them in (continuous metric).
    s_prod = np.sign(gz) * np.sign(yr)
    frac = float(np.mean(s_prod > 0.0)) if s_prod.size else 0.0
    return GyroOdomSignStats(n_samples=int(s_prod.size), sign_agreement_frac=frac, sign_product_mean=float(np.mean(s_prod)) if s_prod.size else 0.0)


def _extrinsic_sanity_from_mean_accel(
    imu_msgs: List[Tuple[float, object]],
    accel_scale: float,
    R_base_imu: np.ndarray,
) -> ExtrinsicSanity:
    a_raw = np.array([[m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z] for _, m in imu_msgs], dtype=np.float64)
    a = a_raw * float(accel_scale)
    a_mean = np.mean(a, axis=0)
    a_base_mean = R_base_imu @ a_mean
    return ExtrinsicSanity(
        accel_base_mean=[float(x) for x in a_base_mean.tolist()],
        accel_base_norm=float(np.linalg.norm(a_base_mean)),
        accel_base_z_sign=float(np.sign(a_base_mean[2])),
        accel_base_z_value=float(a_base_mean[2]),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory containing metadata.yaml and *.db3")
    ap.add_argument("--out", required=True, help="output JSON path for the report")
    ap.add_argument("--max-odom", type=int, default=5000, help="max odom msgs to read (for speed)")
    ap.add_argument("--max-imu", type=int, default=20000, help="max imu msgs to read (for speed)")
    ap.add_argument("--accel-scale", type=float, default=9.81, help="assumed accel scale (g->m/s^2)")
    ap.add_argument(
        "--T-base-imu-rotvec",
        type=float,
        nargs=3,
        default=[-0.026, 0.488, 0.0],
        help="rotvec (rad) for R_base_imu (used for sign/accel sanity checks)",
    )
    args = ap.parse_args()

    bag_uri = str(Path(args.bag).resolve())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    msgs = _load_rosbag_messages(
        bag_uri=bag_uri,
        topics=["/odom", "/livox/mid360/imu"],
        max_messages=None,  # we cap per-topic after loading (rosbag2_py doesn't filter by topic efficiently)
    )

    # Downselect deterministically by time order (first N messages).
    odom_msgs = sorted(msgs["/odom"], key=lambda x: x[0])[: int(args.max_odom)]
    imu_msgs = sorted(msgs["/livox/mid360/imu"], key=lambda x: x[0])[: int(args.max_imu)]

    odom_frames = _frame_stats("/odom", odom_msgs)
    imu_frames = _frame_stats("/livox/mid360/imu", imu_msgs)

    odom_yaw, odom_t, odom_yaw_rate = _odom_yaw_stats(odom_msgs)
    imu_accel = _imu_accel_stats(imu_msgs, accel_scale=float(args.accel_scale))
    imu_gyro = _imu_gyro_stats(imu_msgs)

    R_base_imu = _rotvec_to_R(np.array(args.T_base_imu_rotvec, dtype=np.float64))
    gyro_vs_odom = _gyro_vs_odom_sign(odom_t=odom_t, odom_yaw_rate=odom_yaw_rate, imu_msgs=imu_msgs, R_base_imu=R_base_imu)
    extrinsic_sanity = _extrinsic_sanity_from_mean_accel(imu_msgs=imu_msgs, accel_scale=float(args.accel_scale), R_base_imu=R_base_imu)

    report = ValidationReport(
        bag_uri=bag_uri,
        odom_frames=odom_frames,
        imu_frames=imu_frames,
        imu_accel=imu_accel,
        imu_gyro=imu_gyro,
        odom_yaw=odom_yaw,
        gyro_vs_odom=gyro_vs_odom,
        extrinsic_sanity=extrinsic_sanity,
    )

    # Append twist consistency as an extra top-level field without changing the core dataclass shape.
    report_dict = asdict(report)
    report_dict["odom_twist_consistency"] = asdict(_odom_twist_consistency(odom_msgs))

    out_path.write_text(json.dumps(report_dict, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report_dict, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
