# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy.spatial.transform import RigidTransform

# Re-exported under the project's own name so the rest of the package depends on
# phanesim.types rather than on scipy directly.
Transform = RigidTransform

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
    intrinsics: CameraModel
    resolution: tuple[int, int]  # (w, h)
    pixel_format: str  # e.g. GRAY8, GRAY12, RGB24, YUYV422
    shutter: Shutter  # only GLOBAL is implemented; ROLLING raises at render time
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
class Body:
    """A full-body mesh + armature whose pose assets drive the animation."""

    model: Path  # path to the .blend holding the mesh, rig and pose assets
    armature: str = "rig"  # name of the armature object inside the .blend
    name: str | None = None
    hands: tuple[str, ...] = ("right", "left")  # sides exported to joints_2d.csv


@dataclass
class HeadCamera:
    """A camera rigidly bolted to the head, as on a real headset.

    Both the position and the direction are fixed relative to the skull: the
    configured rest-pose values are carried through the head bone's
    rest-to-pose delta every frame, so the camera moves only when the head does.
    It never turns to follow the hands.

    That is deliberate.  Headset cameras (MEgATrack, UmeTrack) are fixed to the
    device, and hands enter and leave the field of view on their own.  A camera
    that aimed at the hands would centre them in nearly every frame, which both
    removes the off-centre and partially-cropped views a tracker has to cope
    with and correlates the camera pose with the very quantity being predicted.

    rest_position must sit outside the head mesh — in front of the nose tip
    rather than inside it.  For model1 the nose reaches y = -0.174 at eye
    height, so (0, -0.21, 1.715) clears the face.

    rest_forward points where the camera looks in the rest pose.  Hands sit
    below eye level, so it is normally tilted down: (0, -1, -0.268) is 15
    degrees below horizontal, which centres the interaction volume.
    """

    anchor_bone: str = "ORG-spine.006"  # the head bone of a Rigify rig
    rest_position: Vector3 | None = None  # world-space camera point in rest pose
    rest_forward: Vector3 | None = None  # world-space look direction in rest pose
