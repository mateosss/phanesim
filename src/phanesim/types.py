# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import overload

import numpy as np
import numpy.typing as npt

type Scalar = np.float32
type Timestamp = np.int64
type Duration = np.int64
type Vector3 = npt.NDArray[np.float32]  # shape (3,) xyz
type Vector4 = npt.NDArray[np.float32]  # shape (4,) xyzw
type Quaternion = Vector4  # xyzw, scalar last
type Matrix3x3 = npt.NDArray[np.float32]  # shape (3, 3)
type Matrix4x4 = npt.NDArray[np.float32]  # shape (4, 4)
type Color = npt.NDArray[np.float32]  # shape (3,) rgb
type Timestamps = npt.NDArray[np.int64]  # shape (N,) nanoseconds
type Positions = npt.NDArray[np.float32]  # shape (N, 3)
type Quaternions = npt.NDArray[np.float32]  # shape (N, 4) xyzw


def _rotmat_to_quat(R: Matrix3x3) -> Quaternion:
    """Convert a 3x3 rotation matrix to a unit quaternion (xyzw)."""
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


@dataclass(eq=False)
class Transform:
    """Rigid body transform T_a_b: maps points from frame b to frame a (p_a = T_a_b * p_b)."""

    pos: Vector3  # xyz translation
    quat: Quaternion  # xyzw unit quaternion

    @property
    def rotmat(self) -> Matrix3x3:
        x, y, z, w = self.quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @property
    def mat(self) -> Matrix4x4:
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = self.rotmat
        T[:3, 3] = self.pos
        return T

    @property
    def inv(self) -> Transform:
        R_inv = self.rotmat.T
        x, y, z, w = self.quat
        norm_sq = float(x * x + y * y + z * z + w * w)
        return Transform(
            pos=np.array(-(R_inv @ self.pos), dtype=np.float32),
            quat=np.array([-x, -y, -z, w], dtype=np.float32) / norm_sq,
        )

    @overload
    def __mul__(self, other: Transform) -> Transform: ...

    @overload
    def __mul__(self, other: Vector3) -> Vector3: ...

    def __mul__(self, other: Transform | Vector3) -> Transform | Vector3:
        if isinstance(other, Transform):
            m = self.mat @ other.mat
            return Transform(
                pos=np.array(m[:3, 3], dtype=np.float32),
                quat=_rotmat_to_quat(m[:3, :3]),
            )
        return np.array(self.rotmat @ other + self.pos, dtype=np.float32)


@dataclass
class Vignette:
    """Radial light-falloff calibration for a camera.

    Vignetting is the darkening of image corners/edges relative to the center
    caused by the lens geometry.  It is represented here as a grayscale PNG
    whose dimensions match the camera resolution.  Each pixel stores the
    multiplicative attenuation at that screen position:

        0.0 (black) = fully attenuated (no light reaches the sensor)
        1.0 (white) = no attenuation (full brightness)

    Typical vignette PNGs are brightest at the optical centre and fade
    radially outward.  During rendering, Blender's compositor multiplies the
    rendered image by this mask to simulate the falloff.

    The path is resolved relative to the JSON config file that references it.
    """

    path: Path  # absolute path after resolution from the config file's directory


@dataclass
class CameraModel:
    name: str  # e.g. kb4, rt8, pinhole
    parameters: dict[str, float]  # e.g. for kb4: fx, fy, cx, cy, k1, k2, k3, k4


class Shutter(enum.Enum):
    GLOBAL = "global"
    ROLLING = "rolling"


@dataclass
class Camera:
    name: str | None
    T_b_c: Transform  # transform from camera frame to body/rig frame
    intrinsics: CameraModel
    resolution: tuple[int, int]  # (w, h)
    frequency: float  # fps in Hz
    pixel_format: str  # e.g. GRAY8, GRAY12, RGB24, YUYV422
    shutter: Shutter
    exposure: int  # nanoseconds; 0=none, -1=auto
    gain: int  # 0=none, -1=auto
    vignette: Vignette | None = None
    lens_flare: bool = False
    chromatic_aberration: bool = False
    motion_blur: bool = False


@dataclass
class Hand:
    model: Path  # path to mesh and rig file
    name: str | None = None
    scale_length: float = 1.0
    scale_breadth: float = 1.0
    scale_thickness: float = 1.0
    texture: Path | None = None
    color_multiply: Color | None = None

