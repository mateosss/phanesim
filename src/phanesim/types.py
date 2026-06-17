# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import RigidTransform as Transform  # noqa: F401

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
    noise_std: float = 0.0  # Gaussian sensor noise std-dev in 8-bit DN units (0 = disabled)


@dataclass
class Hand:
    model: Path  # path to mesh and rig file
    name: str | None = None
    scale_length: float = 1.0
    scale_breadth: float = 1.0
    scale_thickness: float = 1.0
    texture: Path | None = None
    color_multiply: Color | None = None
