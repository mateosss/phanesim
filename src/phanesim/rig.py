# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phanesim.motion import CameraMotion, HandMotion
from phanesim.types import Camera, CameraModel, Color, Hand, Shutter, Transform, Vignette


def _transform_from_dict(data: dict) -> Transform:
    return Transform(
        pos=np.array(data["pos"], dtype=np.float32),
        quat=np.array(data["quat"], dtype=np.float32),
    )


def _camera_from_dict(data: dict, base_dir: Path) -> Camera:
    vignette: Vignette | None = None
    if data.get("vignette") is not None:
        vignette = Vignette(path=base_dir / data["vignette"]["path"])
    w, h = data["resolution"]
    return Camera(
        name=data.get("name"),
        T_b_c=_transform_from_dict(data["T_b_c"]),
        intrinsics=CameraModel(
            name=data["intrinsics"]["name"],
            parameters=dict(data["intrinsics"]["parameters"]),
        ),
        resolution=(int(w), int(h)),
        frequency=float(data["frequency"]),
        pixel_format=str(data["pixel_format"]),
        shutter=Shutter(data["shutter"]),
        exposure=int(data["exposure"]),
        gain=int(data["gain"]),
        vignette=vignette,
        lens_flare=bool(data.get("lens_flare", False)),
        chromatic_aberration=bool(data.get("chromatic_aberration", False)),
        motion_blur=bool(data.get("motion_blur", False)),
    )


def _hand_from_dict(data: dict, base_dir: Path) -> Hand:
    color_multiply: Color | None = None
    if data.get("color_multiply") is not None:
        color_multiply = np.array(data["color_multiply"], dtype=np.float32)
    return Hand(
        model=base_dir / data["model"],
        name=data.get("name"),
        scale_length=float(data.get("scale_length", 1.0)),
        scale_breadth=float(data.get("scale_breadth", 1.0)),
        scale_thickness=float(data.get("scale_thickness", 1.0)),
        texture=base_dir / data["texture"] if data.get("texture") else None,
        color_multiply=color_multiply,
    )


@dataclass
class CameraHandRig:
    """A set of cameras and hands with known relative transforms.

    T_c_h[c][h] converts hand-h joint positions from the hand frame into camera-c frame.

    JSON schema:
      {
        "cameras": [<camera>, ...],
        "hands":   [<hand>, ...],
        "T_c_h":   [ [{"pos": [x,y,z], "quat": [x,y,z,w]}, ...], ... ]
      }
    """

    cameras: list[Camera]  # C cameras
    hands: list[Hand]  # H hands
    T_c_h: list[list[Transform]]  # [C][H]

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> CameraHandRig:
        cameras = [_camera_from_dict(c, base_dir) for c in data["cameras"]]
        hands = [_hand_from_dict(h, base_dir) for h in data["hands"]]
        T_c_h = [[_transform_from_dict(t) for t in row] for row in data["T_c_h"]]
        return cls(cameras=cameras, hands=hands, T_c_h=T_c_h)

    @classmethod
    def from_path(cls, path: Path) -> CameraHandRig:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)


@dataclass
class Sequence:
    """One rendering sequence: a rig plus one motion trajectory per camera and per hand.

    JSON schema:
      {
        "name":        "<str>",
        "output_path": "<relative-or-absolute path>",
        "camhand_rig": <CameraHandRig dict>,
        "cam_motions": ["<path-to-csv>", ...],   -- one per camera, in rig order
        "hand_motions": ["<path-to-csv>", ...]   -- one per hand, in rig order
      }
    """

    name: str
    output_path: Path  # relative to the project output_path when inside a Project
    camhand_rig: CameraHandRig
    cam_motions: list[CameraMotion]  # len == len(camhand_rig.cameras)
    hand_motions: list[HandMotion]  # len == len(camhand_rig.hands)

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> Sequence:
        camhand_rig = CameraHandRig.from_dict(data["camhand_rig"], base_dir)
        cam_motions = [CameraMotion.from_path(base_dir / p) for p in data["cam_motions"]]
        hand_motions = [HandMotion.from_path(base_dir / p) for p in data["hand_motions"]]
        return cls(
            name=str(data["name"]),
            output_path=Path(data["output_path"]),
            camhand_rig=camhand_rig,
            cam_motions=cam_motions,
            hand_motions=hand_motions,
        )

    @classmethod
    def from_path(cls, path: Path) -> Sequence:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)


@dataclass
class Project:
    """A named collection of sequences sharing a common output directory.

    JSON schema:
      {
        "name":        "<str>",
        "output_path": "<path>",
        "sequences":   [<Sequence dict>, ...]
      }
    """

    name: str
    output_path: Path
    sequences: list[Sequence]

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> Project:
        sequences = [Sequence.from_dict(s, base_dir) for s in data["sequences"]]
        return cls(
            name=str(data["name"]),
            output_path=Path(data["output_path"]),
            sequences=sequences,
        )

    @classmethod
    def from_path(cls, path: Path) -> Project:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)
