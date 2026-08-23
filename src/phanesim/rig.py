# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from phanesim.posemotion import PoseMotion
from phanesim.types import Body, Camera, CameraModel, HeadCamera, Shutter


def _camera_from_dict(data: dict, base_dir: Path) -> Camera:
    w, h = data["resolution"]
    return Camera(
        name=data.get("name"),
        intrinsics=CameraModel(
            name=data["intrinsics"]["name"],
            parameters=dict(data["intrinsics"]["parameters"]),
        ),
        resolution=(int(w), int(h)),
        pixel_format=str(data["pixel_format"]),
        shutter=Shutter(data["shutter"]),
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
        "body_rig":     <BodyRig dict>,
        "hand_motions": ["<path-to-animation-json>", ...],
        "frames":       <int>,
        "hdri":         "<path>"           -- optional
      }
    """

    name: str
    body_rig: BodyRig
    hand_motions: list[PoseMotion]
    frames: int  # frames per motion, spread evenly across the timeline
    hdri: Path | None = None

    @classmethod
    def from_dict(cls, data: dict, base_dir: Path) -> BodySequence:
        return cls(
            name=str(data["name"]),
            body_rig=BodyRig.from_dict(data["body_rig"], base_dir),
            hand_motions=[PoseMotion.from_path(base_dir / p) for p in data["hand_motions"]],
            frames=int(data["frames"]),
            hdri=(base_dir / data["hdri"]) if data.get("hdri") else None,
        )

    @classmethod
    def from_path(cls, path: Path) -> BodySequence:
        return cls.from_dict(json.loads(path.read_text()), base_dir=path.parent)
