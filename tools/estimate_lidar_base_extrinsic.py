#!/usr/bin/env python3
"""
Estimate static LiDAR mounting extrinsic T_base_lidar for no-TF bags (M3DGR).

We allow using ground truth (VRPN) to estimate the *mounting* parameter, because
in real deployments this would be known from CAD/measurement. We do NOT use GT
for evaluation of the SLAM trajectory itself.

Method: hand–eye calibration from relative motions
  A_i X = X B_i
where:
  - A_i = T_base(t_i)^-1 * T_base(t_{i+1}) from /odom or /vrpn_client_node/UGV/pose
  - B_i = T_lidar(t_i)^-1 * T_lidar(t_{i+1}) estimated by ICP between consecutive point clouds
  - X   = T_base_lidar (what we want)

Outputs:
  - Candidate X_odom and/or X_vrpn
  - Suggested launch arg: lidar_base_extrinsic:="[x,y,z,rx,ry,rz]"
"""

from __future__ import annotations

import argparse
import bisect
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

# Ensure we can import the package without requiring it to be installed system-wide.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_ROOT = os.path.join(_PROJECT_ROOT, "fl_ws", "src", "fl_slam_poc")
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

import rclpy
from rclpy.serialization import deserialize_message

from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry

from fl_slam_poc.common.se3 import quat_to_rotvec, rotvec_to_rotmat, rotmat_to_rotvec, se3_compose, se3_inverse
from fl_slam_poc.frontend.icp import icp_3d


def _resolve_db3_path(bag_path: str) -> str:
    if os.path.isfile(bag_path) and bag_path.endswith(".db3"):
        return bag_path
    if not os.path.isdir(bag_path):
        return ""
    for name in sorted(os.listdir(bag_path)):
        if name.endswith(".db3"):
            return os.path.join(bag_path, name)
    return ""


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _topic_id(cur: sqlite3.Cursor, name: str) -> Optional[int]:
    cur.execute("SELECT id FROM topics WHERE name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    return int(row[0]) if row else None


def _topic_type(cur: sqlite3.Cursor, name: str) -> Optional[str]:
    cur.execute("SELECT type FROM topics WHERE name = ? LIMIT 1", (name,))
    row = cur.fetchone()
    return row[0] if row else None


def _iter_msgs(cur: sqlite3.Cursor, topic_id: int) -> Iterable[tuple[int, bytes]]:
    for ts, data in cur.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id = ? ORDER BY timestamp",
        (topic_id,),
    ):
        yield int(ts), data


def _ns_to_sec(ns: int) -> float:
    return float(ns) * 1e-9


def _se3_from_pose_msg(pose) -> np.ndarray:
    t = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
    rv = quat_to_rotvec(pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w)
    return np.array([t[0], t[1], t[2], rv[0], rv[1], rv[2]], dtype=float)


def _se3_to_R_t(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=float).reshape(-1)
    R = rotvec_to_rotmat(T[3:6])
    t = T[:3].copy()
    return R, t


def _solve_hand_eye(A_list: list[np.ndarray], B_list: list[np.ndarray]) -> np.ndarray:
    """
    Solve A_i X = X B_i for X (SE3) using a Kronecker/SVD rotation solve + LS translation.

    Returns: X as 6D se3 vector [x,y,z,rx,ry,rz] representing T_base_lidar.
    """
    assert len(A_list) == len(B_list) and len(A_list) >= 3

    # --- Rotation: solve (I ⊗ R_A - R_B^T ⊗ I) vec(R_X) = 0 ---
    M_blocks = []
    for A, B in zip(A_list, B_list):
        R_A, _ = _se3_to_R_t(A)
        R_B, _ = _se3_to_R_t(B)
        M_blocks.append(np.kron(np.eye(3), R_A) - np.kron(R_B.T, np.eye(3)))
    M = np.concatenate(M_blocks, axis=0)
    _, _, Vt = np.linalg.svd(M)
    rvec = Vt[-1, :]
    R_X = rvec.reshape(3, 3)
    # Project to SO(3)
    U, _, Vt2 = np.linalg.svd(R_X)
    R_X = U @ Vt2
    if np.linalg.det(R_X) < 0:
        U[:, -1] *= -1
        R_X = U @ Vt2

    # --- Translation: (R_A - I) t_X = R_X t_B - t_A ---
    LHS = []
    RHS = []
    for A, B in zip(A_list, B_list):
        R_A, t_A = _se3_to_R_t(A)
        R_B, t_B = _se3_to_R_t(B)
        LHS.append(R_A - np.eye(3))
        RHS.append(R_X @ t_B - t_A)
    LHSm = np.concatenate(LHS, axis=0)
    RHSm = np.concatenate(RHS, axis=0)
    t_X, *_ = np.linalg.lstsq(LHSm, RHSm, rcond=None)

    rv_X = rotmat_to_rotvec(R_X)
    return np.array([t_X[0], t_X[1], t_X[2], rv_X[0], rv_X[1], rv_X[2]], dtype=float)


def _pairwise_relative(se3_list: list[np.ndarray]) -> list[np.ndarray]:
    """Return relative transforms T_i^-1 * T_{i+1} for a list of absolute poses."""
    out: list[np.ndarray] = []
    for i in range(len(se3_list) - 1):
        out.append(se3_compose(se3_inverse(se3_list[i]), se3_list[i + 1]))
    return out


def _nearest_pose(times: list[float], poses: list[np.ndarray], t: float) -> Optional[np.ndarray]:
    if not times:
        return None
    j = bisect.bisect_left(times, t)
    cand = []
    if j < len(times):
        cand.append(j)
    if j > 0:
        cand.append(j - 1)
    best = min(cand, key=lambda k: abs(times[k] - t))
    return poses[best]


def _try_import_livox_custommsg(type_str: str):
    if type_str == "livox_ros_driver2/msg/CustomMsg":
        from livox_ros_driver2.msg import CustomMsg  # type: ignore

        return CustomMsg
    if type_str == "livox_ros_driver/msg/CustomMsg":
        from livox_ros_driver.msg import CustomMsg  # type: ignore

        return CustomMsg
    raise RuntimeError(f"Unsupported Livox type: {type_str}")


@dataclass
class _Cloud:
    t: float
    xyz: np.ndarray  # (N,3) float32


def _downsample_xyz(xyz: np.ndarray, max_points: int, rng: np.random.Generator) -> np.ndarray:
    if xyz.shape[0] <= max_points:
        return xyz
    idx = rng.choice(xyz.shape[0], size=max_points, replace=False)
    return xyz[idx]


def _load_livox_clouds(
    cur: sqlite3.Cursor,
    topic: str,
    msg_type: str,
    step_sec: float,
    max_clouds: int,
    max_points: int,
    range_max: float,
    seed: int,
) -> list[_Cloud]:
    tid = _topic_id(cur, topic)
    if tid is None:
        raise RuntimeError(f"Topic not found: {topic}")
    MsgT = _try_import_livox_custommsg(msg_type)

    rng = np.random.default_rng(seed)
    out: list[_Cloud] = []
    last_t = None
    for ts_ns, data in _iter_msgs(cur, tid):
        msg = deserialize_message(data, MsgT)
        t = _stamp_to_sec(msg.header.stamp)
        if last_t is not None and (t - last_t) < step_sec:
            continue
        pts = list(getattr(msg, "points", []))
        if not pts:
            continue
        xyz = np.array([(p.x, p.y, p.z) for p in pts], dtype=np.float32)
        ok = np.isfinite(xyz).all(axis=1)
        if not np.any(ok):
            continue
        xyz = xyz[ok]
        # range filter (helps ICP)
        r = np.linalg.norm(xyz, axis=1)
        xyz = xyz[(r > 0.1) & (r < range_max)]
        if xyz.shape[0] < 300:
            continue
        xyz = _downsample_xyz(xyz, max_points=max_points, rng=rng)
        out.append(_Cloud(t=t, xyz=xyz))
        last_t = t
        if len(out) >= max_clouds:
            break
    return out


def _load_pose_stream_vrpn(cur: sqlite3.Cursor, topic: str) -> tuple[list[float], list[np.ndarray]]:
    tid = _topic_id(cur, topic)
    if tid is None:
        return [], []
    times: list[float] = []
    poses: list[np.ndarray] = []
    for _, data in _iter_msgs(cur, tid):
        msg = deserialize_message(data, PoseStamped)
        t = _stamp_to_sec(msg.header.stamp)
        times.append(t)
        poses.append(_se3_from_pose_msg(msg.pose))
    return times, poses


def _load_pose_stream_odom(cur: sqlite3.Cursor, topic: str) -> tuple[list[float], list[np.ndarray]]:
    tid = _topic_id(cur, topic)
    if tid is None:
        return [], []
    times: list[float] = []
    poses: list[np.ndarray] = []
    for _, data in _iter_msgs(cur, tid):
        msg = deserialize_message(data, Odometry)
        t = _stamp_to_sec(msg.header.stamp)
        times.append(t)
        poses.append(_se3_from_pose_msg(msg.pose.pose))
    return times, poses


def _estimate_from_base_source(
    base_name: str,
    base_times: list[float],
    base_poses: list[np.ndarray],
    clouds: list[_Cloud],
    icp_max_iter: int,
    icp_tol: float,
    icp_max_mse: float,
) -> Optional[dict]:
    if not base_times or not base_poses:
        return None
    if len(clouds) < 5:
        return None

    A_list: list[np.ndarray] = []
    B_list: list[np.ndarray] = []
    used_pairs = 0
    dropped_pairs = 0

    for i in range(len(clouds) - 1):
        c0 = clouds[i]
        c1 = clouds[i + 1]
        b0 = _nearest_pose(base_times, base_poses, c0.t)
        b1 = _nearest_pose(base_times, base_poses, c1.t)
        if b0 is None or b1 is None:
            dropped_pairs += 1
            continue

        A = se3_compose(se3_inverse(b0), b1)

        # ICP: estimate lidar delta from cloud0 -> cloud1
        res = icp_3d(c0.xyz, c1.xyz, init=np.zeros(6), max_iter=icp_max_iter, tol=icp_tol)
        if not np.isfinite(res.mse) or res.mse > icp_max_mse or res.n_source < 50 or res.n_target < 50:
            dropped_pairs += 1
            continue

        B = res.transform
        A_list.append(A)
        B_list.append(B)
        used_pairs += 1

    if used_pairs < 6:
        return {
            "base": base_name,
            "ok": False,
            "reason": f"too few usable pairs: {used_pairs} (dropped {dropped_pairs})",
        }

    X = _solve_hand_eye(A_list, B_list)

    # Residual summary (rotation + translation)
    rot_err = []
    trans_err = []
    R_X, t_X = _se3_to_R_t(X)
    for A, B in zip(A_list, B_list):
        R_A, t_A = _se3_to_R_t(A)
        R_B, t_B = _se3_to_R_t(B)
        left_R = R_A @ R_X
        right_R = R_X @ R_B
        dR = left_R.T @ right_R
        ang = np.linalg.norm(rotmat_to_rotvec(dR))
        rot_err.append(float(ang))

        left_t = R_A @ t_X + t_A
        right_t = R_X @ t_B + t_X
        trans_err.append(float(np.linalg.norm(left_t - right_t)))

    return {
        "base": base_name,
        "ok": True,
        "pairs_used": used_pairs,
        "pairs_dropped": dropped_pairs,
        "X": X.tolist(),
        "lidar_base_extrinsic_launch_arg": f'lidar_base_extrinsic:="{json.dumps([float(x) for x in X.tolist()])}"',
        "residuals": {
            "rot_err_rad_mean": float(np.mean(rot_err)),
            "rot_err_rad_p90": float(np.quantile(rot_err, 0.9)),
            "trans_err_m_mean": float(np.mean(trans_err)),
            "trans_err_m_p90": float(np.quantile(trans_err, 0.9)),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Estimate static T_base_lidar for no-TF M3DGR using GT (VRPN) and/or odom.")
    ap.add_argument("bag_path", help="Bag directory containing *.db3 (or direct .db3 file).")
    ap.add_argument("--lidar-topic", default="/livox/mid360/lidar")
    ap.add_argument("--vrpn-topic", default="/vrpn_client_node/UGV/pose")
    ap.add_argument("--odom-topic", default="/odom")
    ap.add_argument("--livox-msg-type", default="livox_ros_driver2/msg/CustomMsg")
    ap.add_argument("--step-sec", type=float, default=1.0, help="LiDAR sampling period (sec) for ICP pairs.")
    ap.add_argument("--max-clouds", type=int, default=220, help="Max LiDAR clouds to sample.")
    ap.add_argument("--max-points", type=int, default=6000, help="Max points per cloud for ICP.")
    ap.add_argument("--range-max", type=float, default=50.0, help="Max range (m) to keep points for ICP.")
    ap.add_argument("--icp-max-iter", type=int, default=15)
    ap.add_argument("--icp-tol", type=float, default=1e-4)
    ap.add_argument("--icp-max-mse", type=float, default=0.05, help="Reject ICP pairs with MSE above this.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default="", help="Optional JSON output path.")
    args = ap.parse_args()

    db_path = _resolve_db3_path(args.bag_path)
    if not db_path or not os.path.exists(db_path):
        raise SystemExit(f"Could not locate *.db3 under '{args.bag_path}'")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    print("=== Estimate LiDAR mounting extrinsic (no-TF) ===")
    print(f"bag_path: {args.bag_path}")
    print(f"db_path:   {db_path}")
    print()

    # Sanity: confirm topics exist
    for tname in [args.lidar_topic, args.odom_topic, args.vrpn_topic]:
        ttype = _topic_type(cur, tname)
        print(f"- topic {tname}: {'present' if ttype else 'MISSING'}" + (f", type={ttype}" if ttype else ""))
    print()

    rclpy.init()
    try:
        clouds = _load_livox_clouds(
            cur,
            topic=args.lidar_topic,
            msg_type=args.livox_msg_type,
            step_sec=float(args.step_sec),
            max_clouds=int(args.max_clouds),
            max_points=int(args.max_points),
            range_max=float(args.range_max),
            seed=int(args.seed),
        )
        print(f"Loaded {len(clouds)} LiDAR clouds for ICP (step={args.step_sec}s, max_points={args.max_points}).")

        odom_times, odom_poses = _load_pose_stream_odom(cur, args.odom_topic)
        vrpn_times, vrpn_poses = _load_pose_stream_vrpn(cur, args.vrpn_topic)
        print(f"Loaded {len(odom_times)} odom poses and {len(vrpn_times)} vrpn poses.")
        print()

        results: dict[str, object] = {
            "bag_path": args.bag_path,
            "db_path": db_path,
            "lidar_topic": args.lidar_topic,
            "odom_topic": args.odom_topic,
            "vrpn_topic": args.vrpn_topic,
            "step_sec": args.step_sec,
            "max_points": args.max_points,
        }

        out_list = []
        for base_name, times, poses in [
            ("odom", odom_times, odom_poses),
            ("vrpn", vrpn_times, vrpn_poses),
        ]:
            est = _estimate_from_base_source(
                base_name=base_name,
                base_times=times,
                base_poses=poses,
                clouds=clouds,
                icp_max_iter=int(args.icp_max_iter),
                icp_tol=float(args.icp_tol),
                icp_max_mse=float(args.icp_max_mse),
            )
            if est is None:
                print(f"[{base_name}] insufficient data (topic missing or too few clouds).")
                continue
            out_list.append(est)
            if est.get("ok"):
                X = np.array(est["X"], dtype=float)
                print(f"[{base_name}] OK: pairs_used={est['pairs_used']}, dropped={est['pairs_dropped']}")
                print(f"  X = {X.tolist()}")
                print(f"  launch: {est['lidar_base_extrinsic_launch_arg']}")
                print(f"  residuals: {est['residuals']}")
            else:
                print(f"[{base_name}] FAIL: {est.get('reason')}")
            print()

        results["solutions"] = out_list

        if args.json_out:
            out_path = os.path.abspath(os.path.expanduser(args.json_out))
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, sort_keys=True)
            print(f"Wrote JSON report: {out_path}")

    finally:
        try:
            rclpy.shutdown()
        except Exception:
            pass
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

