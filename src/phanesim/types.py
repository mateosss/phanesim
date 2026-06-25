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
    motion_blur: bool = False
    # Compositor node parameters — values match Blender node inputs directly
    ca_factor: float = 0.30  # Chromatic Aberration → Factor
    distortion: float = 0.387  # Lens Distortion → Distortion
    dispersion: float = 0.0  # Lens Distortion → Dispersion
    lens_scale: float = 1.2  # Transform → Scale (crops black borders after distortion)
    vignette_factor: float = 0.533  # Vignette → Factor
    vignette_feather: float = 0.4  # Vignette → Feather
    noise_std: float = 0.0  # Sensor Noise → Luminance Noise (0 = node disabled)
    chroma_noise: float = 0.0  # Sensor Noise → Chroma Noise


@dataclass
class Hand:
    model: Path  # path to mesh and rig file
    name: str | None = None
    scale_length: float = 1.0
    scale_breadth: float = 1.0
    scale_thickness: float = 1.0
    texture: Path | None = None
    color_multiply: Color | None = None
