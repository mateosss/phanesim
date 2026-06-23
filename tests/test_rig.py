# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rig.py dataclass loading — no Blender required."""

from __future__ import annotations

import pytest

from phanesim.rig import Sequence, _camera_from_dict
from phanesim.types import Shutter

_BASE_CAMERA = {
    "T_b_c": {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]},
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 240.0, "fy": 240.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "frequency": 30.0,
    "pixel_format": "MONO8",
    "shutter": "global",
    "exposure": 0,
    "gain": 0,
}


# ---------------------------------------------------------------------------
# Camera — noise_std
# ---------------------------------------------------------------------------


def test_camera_noise_std_loaded(tmp_path):
    cam = _camera_from_dict({**_BASE_CAMERA, "noise_std": 5.0}, tmp_path)
    assert cam.noise_std == pytest.approx(5.0)


def test_camera_noise_std_defaults_to_zero(tmp_path):
    cam = _camera_from_dict(_BASE_CAMERA, tmp_path)
    assert cam.noise_std == pytest.approx(0.0)


def test_camera_noise_std_zero_explicit(tmp_path):
    cam = _camera_from_dict({**_BASE_CAMERA, "noise_std": 0.0}, tmp_path)
    assert cam.noise_std == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Camera — pixel_format and shutter
# ---------------------------------------------------------------------------


def test_camera_pixel_format_mono8(tmp_path):
    cam = _camera_from_dict({**_BASE_CAMERA, "pixel_format": "MONO8"}, tmp_path)
    assert cam.pixel_format == "MONO8"


def test_camera_pixel_format_rgb24(tmp_path):
    cam = _camera_from_dict({**_BASE_CAMERA, "pixel_format": "RGB24"}, tmp_path)
    assert cam.pixel_format == "RGB24"


def test_camera_shutter_is_enum(tmp_path):
    cam = _camera_from_dict(_BASE_CAMERA, tmp_path)
    assert cam.shutter is Shutter.GLOBAL


# ---------------------------------------------------------------------------
# Sequence — hdri path resolution
# ---------------------------------------------------------------------------


def _write_csv(path, content):
    path.write_text(content)
    return path


def _minimal_seq_dict(tmp_path, hdri=None):
    _write_csv(
        tmp_path / "cam.csv",
        "timestamp,px,py,pz,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n1000000000,0,0,0,0,0,0,1\n",
    )
    _write_csv(
        tmp_path / "hand.csv",
        "timestamp,Wrist_x,Wrist_y,Wrist_z,Wrist_qx,Wrist_qy,Wrist_qz,Wrist_qw\n"
        "0,0,0,0,0,0,0,1\n1000000000,0,0,0,0,0,0,1\n",
    )
    data = {
        "name": "test",
        "output_path": "out",
        "camhand_rig": {
            "cameras": [_BASE_CAMERA],
            "hands": [{"model": "hand.blend"}],
            "T_c_h": [[{"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}]],
        },
        "cam_motions": ["cam.csv"],
        "hand_motions": ["hand.csv"],
    }
    if hdri is not None:
        data["hdri"] = hdri
    return data


def test_sequence_hdri_resolved_to_absolute_path(tmp_path):
    seq = Sequence.from_dict(_minimal_seq_dict(tmp_path, hdri="scene.exr"), tmp_path)
    assert seq.hdri == tmp_path / "scene.exr"


def test_sequence_no_hdri_is_none(tmp_path):
    seq = Sequence.from_dict(_minimal_seq_dict(tmp_path), tmp_path)
    assert seq.hdri is None
