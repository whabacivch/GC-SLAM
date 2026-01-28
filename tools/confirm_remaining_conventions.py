#!/usr/bin/env python3
"""
Confirm the last remaining convention items for Dynamic01_ros2:

1) T_base_lidar translation sanity (primarily z height):
   - Estimate the ground-plane height relative to the LiDAR origin directly from raw
     livox_ros_driver2/CustomMsg point clouds in the bag.
   - This is a measurement-based check to corroborate t_base_lidar.z ~= sensor height
     above ground (when base_footprint is at ground contact and rotations are identity).

2) Native IMU axis/sign convention sanity (pre-extrinsic):
   - Use a regression-style check that raw IMU gyro components, when mapped to base
     via R_base_imu, produce omega_base.z that correlates with odom yaw-rate.
   - Also estimate the best-fit linear combination a·gyro_imu ~= yaw_rate_odom and
     compare it to the expected mapping row (e_z^T R_base_imu).

Outputs a JSON report for audit and doc promotion.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def _rotvec_to_R(rotvec: np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=np.float64).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-12:
        return np.eye(3, dtype=np.float64)
    k = rv / theta
    K = np.array(
        [[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]],
        dtype=np.float64,
    )
    return np.eye(3, dtype=np.float64) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def _yaw_from_quat_xyzw(qx: float, qy: float, qz: float, qw: float) -> float:
    return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))


def _wrap_pi(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def _plane_normal_pca(points: np.ndarray) -> np.ndarray:
    # PCA plane normal: eigenvector of smallest eigenvalue of covariance.
    c = np.mean(points, axis=0)
    X = points - c
    cov = np.cov(X.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    n = eigvecs[:, np.argmin(eigvals)]
    # Normalize.
    n = n / (np.linalg.norm(n) + 1e-12)
    # Enforce "up-ish" convention for stability (deterministic): make n.z >= 0
    if n[2] < 0.0:
        n = -n
    return n


def _estimate_ground_height(points: np.ndarray, n: np.ndarray, q: float) -> float:
    """
    Estimate ground plane offset along normal direction using a low quantile of s = n·p.
    For a Z-up LiDAR at height h above ground, ground points tend to have smaller s.

    Returns height magnitude (positive).
    """
    s = points @ n
    s_q = float(np.quantile(s, q))
    return abs(s_q)


def _ransac_ground_plane_height(
    points: np.ndarray,
    *,
    n_up: np.ndarray = np.array([0.0, 0.0, 1.0], dtype=np.float64),
    n_dot_min: float = 0.8,
    dist_thresh: float = 0.05,
    n_iters: int = 250,
    max_points: int = 6000,
    seed: int = 0,
) -> tuple[float, float]:
    """
    Deterministic RANSAC plane fit to estimate ground plane height from LiDAR origin.

    Returns:
      (height_m, inlier_frac)

    Model: plane n·x + d = 0, with ||n||=1 and n·n_up >= n_dot_min.
    Height from origin is |d| when origin is at sensor.
    """
    pts = np.asarray(points, dtype=np.float64)
    N = pts.shape[0]
    if N < 3:
        return 0.0, 0.0

    rng = np.random.RandomState(seed)
    if N > max_points:
        idx = rng.choice(N, size=max_points, replace=False)
        pts = pts[idx, :]
        N = pts.shape[0]

    best_inliers = -1
    best_d = 0.0
    best_frac = 0.0
    up = np.asarray(n_up, dtype=np.float64).reshape(3)
    up = up / (np.linalg.norm(up) + 1e-12)

    for _ in range(int(n_iters)):
        i, j, k = rng.randint(0, N, size=3)
        p1, p2, p3 = pts[i], pts[j], pts[k]
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        nn = float(np.linalg.norm(n))
        if nn < 1e-9:
            continue
        n = n / nn
        # Enforce "up" normal direction.
        if float(np.dot(n, up)) < 0.0:
            n = -n
        if float(np.dot(n, up)) < n_dot_min:
            continue
        d = -float(np.dot(n, p1))
        dist = np.abs(pts @ n + d)
        inliers = int(np.sum(dist < dist_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_d = d
            best_frac = float(inliers / N)

    if best_inliers <= 0:
        return 0.0, 0.0
    return abs(best_d), best_frac

@dataclass
class LidarHeightEstimate:
    n_scans: int
    n_points_total: int
    normal_mean: List[float]
    normal_std: List[float]
    # Height estimates using PCA normal (often dominated by non-ground structure).
    height_q01_mean: float
    height_q01_std: float
    height_q05_mean: float
    height_q05_std: float
    # Height estimates using raw Z quantiles (more robust when LiDAR Z-up is known).
    z_q01_mean: float
    z_q01_std: float
    z_q05_mean: float
    z_q05_std: float
    height_from_z_q01_mean: float
    height_from_z_q01_std: float
    height_from_z_q05_mean: float
    height_from_z_q05_std: float
    # Ground plane height from deterministic RANSAC (preferred).
    height_ransac_mean: float
    height_ransac_std: float
    inlier_frac_ransac_mean: float
    inlier_frac_ransac_std: float


@dataclass
class ImuGyroAxisConfirmation:
    n_pairs: int
    corr_omega_z_vs_yaw_rate: float
    slope_omega_z_vs_yaw_rate: float
    corr_raw_bestfit_vs_yaw_rate: float
    bestfit_a_imu: List[float]
    expected_a_imu: List[float]
    angle_deg_bestfit_vs_expected: float
    sign_agreement_bestfit_vs_expected: float
    corr_raw_bestfit_weighted_vs_yaw_rate: float
    bestfit_a_imu_weighted: List[float]
    angle_deg_bestfit_weighted_vs_expected: float
    sign_agreement_bestfit_weighted_vs_expected: float


@dataclass
class RemainingConventionsReport:
    bag_uri: str
    t_base_lidar: List[float]
    R_base_imu_rotvec: List[float]
    lidar_height: LidarHeightEstimate
    imu_gyro_axis: ImuGyroAxisConfirmation


def _load_rosbag(bag_uri: str):
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_uri, storage_id="sqlite3"),
        rosbag2_py.ConverterOptions("", ""),
    )
    return reader


def _read_livox_scans(reader, topic: str, n_scans: int) -> List[np.ndarray]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    msg_cls = get_message(types[topic])
    scans: List[np.ndarray] = []
    while reader.has_next() and len(scans) < n_scans:
        tpc, data, _ = reader.read_next()
        if tpc != topic:
            continue
        msg = deserialize_message(data, msg_cls)
        # livox_ros_driver2/CustomMsg has field "points" (array of CustomPoint).
        pts = np.array([[p.x, p.y, p.z] for p in msg.points], dtype=np.float64)
        if pts.shape[0] >= 100:
            scans.append(pts)
    return scans


def _read_odom_and_imu(reader, odom_topic: str, imu_topic: str, max_msgs: int) -> Tuple[List[Tuple[float, object]], List[Tuple[float, object]]]:
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    odom_cls = get_message(types[odom_topic])
    imu_cls = get_message(types[imu_topic])

    odom: List[Tuple[float, object]] = []
    imu: List[Tuple[float, object]] = []
    while reader.has_next() and (len(odom) < max_msgs or len(imu) < max_msgs):
        tpc, data, _ = reader.read_next()
        if tpc == odom_topic and len(odom) < max_msgs:
            msg = deserialize_message(data, odom_cls)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            odom.append((float(stamp), msg))
        elif tpc == imu_topic and len(imu) < max_msgs:
            msg = deserialize_message(data, imu_cls)
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            imu.append((float(stamp), msg))
    return odom, imu


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    if x.size == 0 or y.size == 0:
        return 0.0
    xs = float(np.std(x))
    ys = float(np.std(y))
    if xs < 1e-12 or ys < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    # Least squares slope y ≈ a x (no intercept).
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = float(np.dot(x, x)) + 1e-12
    return float(np.dot(x, y) / denom)


def _weighted_solve_a(G: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    # Solve min ||W^(1/2)(G a - y)||^2 with small ridge.
    G = np.asarray(G, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1, 1)
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    w = w / (np.mean(w) + 1e-12)
    W = w.reshape(-1, 1)
    Gw = G * np.sqrt(W)
    yw = y * np.sqrt(W)
    lam = 1e-9
    A = Gw.T @ Gw + lam * np.eye(3)
    b = Gw.T @ yw
    return np.linalg.solve(A, b).reshape(3)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bag", required=True, help="rosbag2 directory")
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--imu-topic", default="/livox/mid360/imu")
    ap.add_argument("--n-lidar-scans", type=int, default=40)
    ap.add_argument("--max-odom", type=int, default=3504)
    ap.add_argument("--max-imu", type=int, default=35032)
    ap.add_argument("--accel-scale", type=float, default=9.81)
    ap.add_argument("--t-base-lidar", type=float, nargs=3, default=[-0.011, 0.0, 0.778])
    ap.add_argument("--R-base-imu-rotvec", type=float, nargs=3, default=[-0.015586, 0.489293, 0.0])
    ap.add_argument("--ground-quantile-1", type=float, default=0.01)
    ap.add_argument("--ground-quantile-2", type=float, default=0.05)
    args = ap.parse_args()

    bag_uri = str(Path(args.bag).resolve())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # --- LiDAR ground height ---
    reader = _load_rosbag(bag_uri)
    scans = _read_livox_scans(reader, args.lidar_topic, n_scans=int(args.n_lidar_scans))
    if not scans:
        raise RuntimeError(f"no lidar scans read from {args.lidar_topic}")

    normals = []
    h1 = []
    h2 = []
    z1 = []
    z2 = []
    hr = []
    fr = []
    n_points_total = 0
    for pts in scans:
        n_points_total += pts.shape[0]
        n = _plane_normal_pca(pts)
        normals.append(n)
        h1.append(_estimate_ground_height(pts, n, float(args.ground_quantile_1)))
        h2.append(_estimate_ground_height(pts, n, float(args.ground_quantile_2)))
        z = pts[:, 2]
        z1.append(float(np.quantile(z, float(args.ground_quantile_1))))
        z2.append(float(np.quantile(z, float(args.ground_quantile_2))))
        h_r, f_r = _ransac_ground_plane_height(pts)
        hr.append(float(h_r))
        fr.append(float(f_r))

    normals = np.array(normals, dtype=np.float64)
    h1 = np.array(h1, dtype=np.float64)
    h2 = np.array(h2, dtype=np.float64)
    z1 = np.array(z1, dtype=np.float64)
    z2 = np.array(z2, dtype=np.float64)
    hr = np.array(hr, dtype=np.float64)
    fr = np.array(fr, dtype=np.float64)
    # If LiDAR is Z-up, ground tends to have negative Z. Height is -z_q when z_q < 0.
    # Use abs() to keep a magnitude; the sign itself is recorded in z_q*.
    hz1 = np.abs(z1)
    hz2 = np.abs(z2)

    lidar_height = LidarHeightEstimate(
        n_scans=len(scans),
        n_points_total=int(n_points_total),
        normal_mean=[float(x) for x in np.mean(normals, axis=0).tolist()],
        normal_std=[float(x) for x in np.std(normals, axis=0).tolist()],
        height_q01_mean=float(np.mean(h1)),
        height_q01_std=float(np.std(h1)),
        height_q05_mean=float(np.mean(h2)),
        height_q05_std=float(np.std(h2)),
        z_q01_mean=float(np.mean(z1)),
        z_q01_std=float(np.std(z1)),
        z_q05_mean=float(np.mean(z2)),
        z_q05_std=float(np.std(z2)),
        height_from_z_q01_mean=float(np.mean(hz1)),
        height_from_z_q01_std=float(np.std(hz1)),
        height_from_z_q05_mean=float(np.mean(hz2)),
        height_from_z_q05_std=float(np.std(hz2)),
        height_ransac_mean=float(np.mean(hr)),
        height_ransac_std=float(np.std(hr)),
        inlier_frac_ransac_mean=float(np.mean(fr)),
        inlier_frac_ransac_std=float(np.std(fr)),
    )

    # --- IMU native-axis confirmation via odom yaw-rate ---
    reader2 = _load_rosbag(bag_uri)
    odom_msgs, imu_msgs = _read_odom_and_imu(reader2, args.odom_topic, args.imu_topic, max_msgs=max(int(args.max_odom), int(args.max_imu)))
    odom_msgs = sorted(odom_msgs, key=lambda x: x[0])[: int(args.max_odom)]
    imu_msgs = sorted(imu_msgs, key=lambda x: x[0])[: int(args.max_imu)]

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

    # IMU gyro in imu frame and mapped to base.
    imu_t = np.array([s for s, _ in imu_msgs], dtype=np.float64)
    gyro_imu = np.array([[m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z] for _, m in imu_msgs], dtype=np.float64)
    R_base_imu = _rotvec_to_R(np.array(args.R_base_imu_rotvec, dtype=np.float64))
    gyro_base = (R_base_imu @ gyro_imu.T).T
    omega_z = gyro_base[:, 2]

    # Interpolate odom yaw_rate to imu timestamps (nearest-left interval).
    idx = np.searchsorted(odom_t_mid, imu_t, side="right") - 1
    idx = np.clip(idx, 0, yaw_rate.size - 1)
    yr_i = yaw_rate[idx]

    # Correlation and slope omega_z ~ yaw_rate.
    corr_oz = _corr(omega_z, yr_i)
    slope_oz = _slope(omega_z, yr_i)

    # Best-fit linear combination a·gyro_imu ~ yaw_rate (no intercept).
    # Solve min ||G a - y||^2.
    G = gyro_imu
    y = yr_i.reshape(-1, 1)
    # Regularize slightly for numerical stability (deterministic).
    lam = 1e-9
    A = G.T @ G + lam * np.eye(3)
    b = G.T @ y
    a_hat = np.linalg.solve(A, b).reshape(3)
    y_hat = G @ a_hat
    corr_best = _corr(y_hat, yr_i)

    # Weighted best-fit emphasizing turning (continuous): w = |yaw_rate|
    w = np.abs(yr_i) + 1e-6
    a_hat_w = _weighted_solve_a(G, yr_i, w)
    y_hat_w = G @ a_hat_w
    corr_best_w = _corr(y_hat_w, yr_i)

    # Expected mapping is e_z^T R_base_imu (row 3 of R_base_imu) because omega_base.z = row3 · gyro_imu.
    a_exp = R_base_imu[2, :]
    # Angle between a_hat and a_exp (ignore magnitude).
    ah_n = a_hat / (np.linalg.norm(a_hat) + 1e-12)
    ae_n = a_exp / (np.linalg.norm(a_exp) + 1e-12)
    dot = float(np.clip(np.dot(ah_n, ae_n), -1.0, 1.0))
    angle = float(math.degrees(math.acos(dot)))
    sign_agree = float(np.sign(np.dot(a_hat, a_exp)))

    ahw_n = a_hat_w / (np.linalg.norm(a_hat_w) + 1e-12)
    dot_w = float(np.clip(np.dot(ahw_n, ae_n), -1.0, 1.0))
    angle_w = float(math.degrees(math.acos(dot_w)))
    sign_agree_w = float(np.sign(np.dot(a_hat_w, a_exp)))

    imu_gyro_axis = ImuGyroAxisConfirmation(
        n_pairs=int(imu_t.size),
        corr_omega_z_vs_yaw_rate=float(corr_oz),
        slope_omega_z_vs_yaw_rate=float(slope_oz),
        corr_raw_bestfit_vs_yaw_rate=float(corr_best),
        bestfit_a_imu=[float(x) for x in a_hat.tolist()],
        expected_a_imu=[float(x) for x in a_exp.tolist()],
        angle_deg_bestfit_vs_expected=float(angle),
        sign_agreement_bestfit_vs_expected=float(sign_agree),
        corr_raw_bestfit_weighted_vs_yaw_rate=float(corr_best_w),
        bestfit_a_imu_weighted=[float(x) for x in a_hat_w.tolist()],
        angle_deg_bestfit_weighted_vs_expected=float(angle_w),
        sign_agreement_bestfit_weighted_vs_expected=float(sign_agree_w),
    )

    report = RemainingConventionsReport(
        bag_uri=bag_uri,
        t_base_lidar=[float(x) for x in args.t_base_lidar],
        R_base_imu_rotvec=[float(x) for x in args.R_base_imu_rotvec],
        lidar_height=lidar_height,
        imu_gyro_axis=imu_gyro_axis,
    )

    out_path.write_text(json.dumps(asdict(report), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(asdict(report), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
