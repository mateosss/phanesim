# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import enum
import math
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
class CameraSweep:
    """A steady turn of the camera away from where the head is looking.

    The head-mounted camera is rigid: it points wherever the skull points and
    nothing else. That is right for a headset, but it means a sequence whose
    head never moves sees exactly one view of the room. A sweep turns the camera
    a little further in one direction as the clip runs, so the first frame is
    the rest view and the last is *degrees* away from it.

    It is a deliberate, repeatable movement rather than a random one, so a run
    can be checked by eye: turn it 30 degrees right and the background should
    slide left across the frames.

    The turn is in the camera's own frame and is added on top of whatever the
    head is doing, so head poses and a sweep compose rather than conflict.
    """

    direction: str  # "left", "right", "up" or "down"
    degrees: float

    #: Rotation applied per direction, as (pitch, yaw) multipliers in the
    #: camera's own axes: X is its right axis, Y its down axis.
    _AXES = {"right": (0.0, 1.0), "left": (0.0, -1.0), "up": (1.0, 0.0), "down": (-1.0, 0.0)}

    def __post_init__(self) -> None:
        if self.direction not in self._AXES:
            raise ValueError(f"direction must be one of {sorted(self._AXES)}, got {self.direction!r}")

    def euler_at(self, progress: float) -> tuple[float, float, float]:
        """Rotation in the camera's own frame at *progress* through the clip.

        Args:
            progress: 0.0 on the first frame, 1.0 on the last.

        Returns:
            (pitch, yaw, roll) in radians, for a rotation applied in camera axes.
        """
        pitch, yaw = self._AXES[self.direction]
        angle = math.radians(self.degrees) * progress
        return (pitch * angle, yaw * angle, 0.0)


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
    height, and clip_start is 0.01, so a camera behind that line does not clip
    the face away: it renders the inside of it, filling the frame with skin.
    (0, -0.181, 1.723) clears the nose by 7 mm, and is about as far back as
    model1 goes.

    Closer to the face is better as long as it clears, because the hands work
    close to the body: pulling the camera back from (0, -0.21, 1.715) widens
    what the near field covers, and those 3 cm alone took a tenth of the frames
    from holding no hand to holding one.  It is also where a headset's cameras
    actually sit — the visor, not 10 cm past the nose.

    rest_forward points where the camera looks in the rest pose.  Hands sit
    below eye level and close in, so it is tilted well down: (0, -1, -0.601) is
    31 degrees below horizontal.  Measured on rendered clips, hand centres sat a
    median of 26 degrees below a 15-degree axis -- past halfway to the bottom
    edge -- and steepening it to 31 both centres them (median 14 degrees off the
    axis afterwards) and took the frames holding two hands from 41% to 69%, and
    those holding none from 5% to under 1%.  HEAD_VIEW in posemotion.py is
    calibrated against this angle, so the two have to move together.
    """

    anchor_bone: str = "ORG-spine.006"  # the head bone of a Rigify rig
    rest_position: Vector3 | None = None  # world-space camera point in rest pose
    rest_forward: Vector3 | None = None  # world-space look direction in rest pose
