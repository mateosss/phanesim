# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from phanesim.motion import CameraMotion, HandMotion
from phanesim.posemotion import PoseMotion
from phanesim.types import Body, Camera, CameraModel, Color, Hand, HeadCamera, Shutter, Transform


def _transform_from_dict(data: dict) -> Transform:
    return Transform.from_components(
        translation=np.array(data["pos"], dtype=np.float64),
        rotation=Rotation.from_quat(data["quat"]),  # xyzw scalar-last
    )


def _camera_from_dict(data: dict, base_dir: Path) -> Camera:
    w, h = data["resolution"]
    return Camera(
        name=data.get("name"),
        T_b_c=_transform_from_dict(data["T_b_c"]),
        intrinsics=CameraModel(
            name=data["intrinsics"]["name"],
            parameters=dict(data["intrinsics"]["parameters"]),
        ),
        resolution=(int(w), int(h)),
        frequency=float(data.get("frequency", 10.0)),
        pixel_format=str(data["pixel_format"]),
        shutter=Shutter(data["shutter"]),
        exposure=int(data["exposure"]),
        gain=int(data["gain"]),
        motion_blur=bool(data.get("motion_blur", False)),
        ca_factor=float(data.get("ca_factor", 0.30)),
        distortion=float(data.get("distortion", 0.387)),
        dispersion=float(data.get("dispersion", 0.0)),
        lens_scale=float(data.get("lens_scale", 1.2)),
        vignette_factor=float(data.get("vignette_factor", 0.533)),
        vignette_feather=float(data.get("vignette_feather", 0.4)),
        noise_std=float(data.get("noise_std", 0.0)),
        chroma_noise=float(data.get("chroma_noise", 0.0)),
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
    hdri: Path | None = None  # absolute path to HDR/EXR environment map, or None

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> Sequence:
        camhand_rig = CameraHandRig.from_dict(data["camhand_rig"], base_dir)
        cam_motions = [CameraMotion.from_path(base_dir / p) for p in data["cam_motions"]]
        hand_motions = [HandMotion.from_path(base_dir / p) for p in data["hand_motions"]]
        hdri = (base_dir / data["hdri"]) if data.get("hdri") else None
        return cls(
            name=str(data["name"]),
            output_path=Path(data["output_path"]),
            camhand_rig=camhand_rig,
            cam_motions=cam_motions,
            hand_motions=hand_motions,
            hdri=hdri,
        )

    @classmethod
    def from_path(cls, path: Path) -> Sequence:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)


def _body_from_dict(data: dict, base_dir: Path) -> Body:
    return Body(
        model=base_dir / data["model"],
        armature=str(data.get("armature", "rig")),
        name=data.get("name"),
        hands=tuple(data.get("hands", ("right", "left"))),
    )


def _head_camera_from_dict(data: dict) -> HeadCamera:
    def _vec(key: str, default: list[float] | None) -> np.ndarray | None:
        value = data.get(key, default)
        return None if value is None else np.array(value, dtype=np.float32)

    return HeadCamera(
        anchor_bone=str(data.get("anchor_bone", "ORG-spine.006")),
        rest_position=_vec("rest_position", None),
        # The body faces -Y in rest pose; the downward component aims at the
        # interaction volume, where the hands are.
        rest_forward=_vec("rest_forward", [0.0, -1.0, -0.268]),
    )


@dataclass
class BodyRig:
    """A set of head-mounted cameras attached to one full-body model.

    Unlike CameraHandRig there is no T_c_h: the body's joints are already in
    world space, and each camera's pose is derived from the head bone every
    frame rather than read from a trajectory file.

    JSON schema:
      {
        "cameras":     [<camera>, ...],
        "body":        <body>,
        "head_camera": <head_camera>
      }
    """

    cameras: list[Camera]
    body: Body
    head_camera: HeadCamera

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> BodyRig:
        return cls(
            cameras=[_camera_from_dict(c, base_dir) for c in data["cameras"]],
            body=_body_from_dict(data["body"], base_dir),
            head_camera=_head_camera_from_dict(data.get("head_camera", {})),
        )

    @classmethod
    def from_path(cls, path: Path) -> BodyRig:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)


@dataclass
class BodySequence:
    """One rendering sequence driven by pose assets instead of motion CSVs.

    The hand_motions entries are paths to pose *motion description* JSONs (see
    phanesim.posemotion), not CSVs: they name which pose asset is reached when,
    and the bone values are resolved from the model .blend at render time.

    JSON schema:
      {
        "name":         "<str>",
        "output_path":  "<relative-or-absolute path>",
        "body_rig":     <BodyRig dict>,
        "hand_motions": ["<path-to-animation-json>", ...],
        "frames":       <int>,             -- optional
        "hdri":         "<path>"           -- optional
      }
    """

    name: str
    output_path: Path
    body_rig: BodyRig
    hand_motions: list[PoseMotion]
    frames: int | None = None  # frames per motion; None derives them from the camera rate
    hdri: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> BodySequence:
        return cls(
            name=str(data["name"]),
            output_path=Path(data["output_path"]),
            body_rig=BodyRig.from_dict(data["body_rig"], base_dir),
            hand_motions=[PoseMotion.from_path(base_dir / p) for p in data["hand_motions"]],
            frames=int(data["frames"]) if data.get("frames") is not None else None,
            hdri=(base_dir / data["hdri"]) if data.get("hdri") else None,
        )

    @classmethod
    def from_path(cls, path: Path) -> BodySequence:
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
