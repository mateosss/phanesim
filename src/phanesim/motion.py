# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.spatial.transform import Rotation

from phanesim.types import Positions, Quaternion, Quaternions, Timestamp, Timestamps, Transform


def _slerp(q0: Quaternion, q1: Quaternion, t: float) -> Quaternion:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:  # take shortest arc
        q1 = -q1
        dot = -dot
    dot = min(dot, 1.0)
    if dot > 0.9995:  # nearly identical — fall back to normalised lerp
        result = q0 + t * (q1 - q0)
        return np.array(result / np.linalg.norm(result), dtype=np.float32)
    theta_0 = float(np.arccos(dot))
    theta = theta_0 * t
    sin_theta_0 = np.sin(theta_0)
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return np.array(s0 * q0 + s1 * q1, dtype=np.float32)


def _interp_transform(
    ts: Timestamps,
    xyz: Positions,
    quats: Quaternions,
    timestamp: Timestamp,
) -> Transform:
    idx = int(np.searchsorted(ts, timestamp))
    if idx == 0:
        return Transform.from_components(xyz[0], Rotation.from_quat(quats[0]))
    if idx >= len(ts):
        return Transform.from_components(xyz[-1], Rotation.from_quat(quats[-1]))
    alpha = float(timestamp - ts[idx - 1]) / float(ts[idx] - ts[idx - 1])
    pos = xyz[idx - 1] + alpha * (xyz[idx] - xyz[idx - 1])
    quat = _slerp(quats[idx - 1], quats[idx], alpha)
    return Transform.from_components(pos, Rotation.from_quat(quat))


@dataclass
class CameraMotion:
    """Camera trajectory.

    CSV columns: timestamp, px, py, pz, qx, qy, qz, qw
    Timestamps are int64 nanoseconds; quaternion is xyzw scalar-last.
    """

    source: Path
    ts: Timestamps  # (N,)
    xyz: Positions  # (N, 3)
    quats: Quaternions  # (N, 4) xyzw

    @classmethod
    def from_path(cls, path: Path) -> CameraMotion:
        df = pd.read_csv(path, engine="pyarrow")
        ts = df["timestamp"].to_numpy(dtype=np.int64)
        xyz = df[["px", "py", "pz"]].to_numpy(dtype=np.float32)
        quats = df[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        return cls(source=path, ts=ts, xyz=xyz, quats=quats)

    def get_pose(self, timestamp: Timestamp) -> Transform:
        return _interp_transform(self.ts, self.xyz, self.quats, timestamp)


@dataclass
class HandMotion:
    """Hand joint trajectories.

    CSV columns: timestamp, {joint}_x, {joint}_y, {joint}_z,
                 {joint}_qx, {joint}_qy, {joint}_qz, {joint}_qw, ...
    One group of 7 columns per joint, repeated for each joint in order.
    Timestamps are int64 nanoseconds; quaternion is xyzw scalar-last.
    """

    source: Path
    joint_names: list[str]  # length J, in CSV column order
    ts: Timestamps  # (N,)
    joints_xyz: npt.NDArray[np.float32]  # (N, J, 3)
    joints_quats: npt.NDArray[np.float32]  # (N, J, 4) xyzw

    @classmethod
    def from_path(cls, path: Path) -> HandMotion:
        df = pd.read_csv(path, engine="pyarrow")
        ts = df["timestamp"].to_numpy(dtype=np.int64)

        pose_cols = [c for c in df.columns if c != "timestamp"]
        # each joint occupies exactly 7 columns: _x _y _z _qx _qy _qz _qw
        joint_names = [pose_cols[i].removesuffix("_x") for i in range(0, len(pose_cols), 7)]
        J, N = len(joint_names), len(ts)

        joints_xyz = np.empty((N, J, 3), dtype=np.float32)
        joints_quats = np.empty((N, J, 4), dtype=np.float32)
        for j, name in enumerate(joint_names):
            joints_xyz[:, j, :] = df[[f"{name}_x", f"{name}_y", f"{name}_z"]].to_numpy(dtype=np.float32)
            joints_quats[:, j, :] = df[[f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"]].to_numpy(
                dtype=np.float32
            )

        return cls(source=path, joint_names=joint_names, ts=ts, joints_xyz=joints_xyz, joints_quats=joints_quats)

    def get_joint_poses(self, timestamp: Timestamp) -> list[Transform]:
        """Return one interpolated Transform per joint at the given timestamp."""
        idx = int(np.searchsorted(self.ts, timestamp))
        clamp_low = idx == 0
        clamp_high = idx >= len(self.ts)

        poses: list[Transform] = []
        for j in range(len(self.joint_names)):
            if clamp_low:
                poses.append(Transform.from_components(self.joints_xyz[0, j], Rotation.from_quat(self.joints_quats[0, j])))
            elif clamp_high:
                poses.append(Transform.from_components(self.joints_xyz[-1, j], Rotation.from_quat(self.joints_quats[-1, j])))
            else:
                alpha = float(timestamp - self.ts[idx - 1]) / float(self.ts[idx] - self.ts[idx - 1])
                pos = self.joints_xyz[idx - 1, j] + alpha * (self.joints_xyz[idx, j] - self.joints_xyz[idx - 1, j])
                quat = _slerp(self.joints_quats[idx - 1, j], self.joints_quats[idx, j], alpha)
                poses.append(Transform.from_components(pos, Rotation.from_quat(quat)))
        return poses
