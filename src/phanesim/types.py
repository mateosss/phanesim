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


@dataclass
class Body:
    """A full-body mesh + armature whose pose assets drive the animation."""

    model: Path  # path to the .blend holding the mesh, rig and pose assets
    armature: str = "rig"  # name of the armature object inside the .blend
    name: str | None = None
    hands: tuple[str, ...] = ("right", "left")  # sides exported to joints_2d.csv


@dataclass
class HeadCamera:
    """A forward-facing AR-headset camera rigidly attached to the head.

    Position: the camera sits at *rest_position*, an absolute world-space point
    valid while the body is in its rest pose.  Each frame that point is carried
    through the head bone's rest-to-pose delta, so the camera follows the head
    exactly as a headset would.  rest_position must sit outside the head mesh —
    in front of the nose tip rather than inside it.  For cmale1.blend the nose
    reaches y = -0.174 at eye height, so (0, -0.21, 1.715) clears the face.

    Orientation: the camera looks along *rest_forward* (also carried through the
    head delta), then turns toward the tracked hands by at most
    *max_deviation_deg*.  This is the headset compromise — the view stays
    fundamentally front-facing, as a device mounted on someone's face must, but
    still yields enough to keep the hands framed and the subject of the shot.
    A rigid forward camera would let low or wide poses drift out of frame; an
    unclamped look-at would swing the head to physically implausible angles.

    Tracking: *track_hands* lists the sides that count as the subject.  With
    both listed the camera aims at their midpoint, so a left-hand pose, a
    right-hand pose or a two-handed pose all stay framed without the camera
    favouring one side.  Hands behind the head are ignored when they would drag
    the aim backwards.
    """

    anchor_bone: str = "ORG-spine.006"  # the head bone of a Rigify rig
    rest_position: Vector3 | None = None  # world-space camera point in rest pose
    rest_forward: Vector3 | None = None  # world-space look direction in rest pose
    track_hands: tuple[str, ...] = ("right", "left")  # sides treated as the subject
    track_landmark: str = "MiddleProximal"  # landmark aimed at within each hand
    max_deviation_deg: float = 35.0  # how far the gaze may leave straight ahead
