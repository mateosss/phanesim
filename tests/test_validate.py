# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import jsonschema
import pytest

from phanesim.validate import (
    validate_camera,
    validate_camera_motion,
    validate_camhand_rig,
    validate_hand,
    validate_hand_motion,
    validate_project,
    validate_sequence,
)

# ---------------------------------------------------------------------------
# Shared valid fixtures
# ---------------------------------------------------------------------------

_TRANSFORM = {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}

_CAMERA = {
    "T_b_c": _TRANSFORM,
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "frequency": 30.0,
    "pixel_format": "RGB24",
    "shutter": "global",
    "exposure": 0,
    "gain": 0,
}

_HAND = {"model": "hand.blend"}

_RIG = {
    "cameras": [_CAMERA],
    "hands": [_HAND],
    "T_c_h": [[_TRANSFORM]],
}

_SEQUENCE = {
    "name": "seq",
    "output_path": "out",
    "camhand_rig": _RIG,
    "cam_motions": ["cam.csv"],
    "hand_motions": ["hand.csv"],
}

_PROJECT = {
    "name": "proj",
    "output_path": "out",
    "sequences": [_SEQUENCE],
}


def _write(tmp_path, data: dict, name: str = "data.json"):
    p = tmp_path / name
    p.write_text(json.dumps(data))
    return p


# ---------------------------------------------------------------------------
# camera
# ---------------------------------------------------------------------------


def test_validate_camera_valid(tmp_path):
    validate_camera(_write(tmp_path, _CAMERA))


def test_validate_camera_missing_required(tmp_path):
    bad = {k: v for k, v in _CAMERA.items() if k != "frequency"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_wrong_shutter(tmp_path):
    bad = {**_CAMERA, "shutter": "electronic"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_negative_resolution(tmp_path):
    bad = {**_CAMERA, "resolution": [0, 480]}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_extra_field_rejected(tmp_path):
    bad = {**_CAMERA, "unknown_field": 42}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_compositor_params(tmp_path):
    cam = {
        **_CAMERA,
        "ca_factor": 0.30,
        "distortion": 0.387,
        "dispersion": 0.0,
        "lens_scale": 1.2,
        "vignette_factor": 0.533,
        "vignette_feather": 0.4,
    }
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_lens_scale_must_be_positive(tmp_path):
    bad = {**_CAMERA, "lens_scale": 0.0}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_distortion_can_be_negative(tmp_path):
    cam = {**_CAMERA, "distortion": -0.3}
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_pixel_format_mono8(tmp_path):
    cam = {**_CAMERA, "pixel_format": "MONO8"}
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_invalid_pixel_format(tmp_path):
    bad = {**_CAMERA, "pixel_format": "GRAY8"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_rolling_shutter_rejected(tmp_path):
    # Schema was restricted to ["global"] only; rolling is no longer accepted.
    bad = {**_CAMERA, "shutter": "rolling"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_noise_std_valid(tmp_path):
    cam = {**_CAMERA, "noise_std": 5.0}
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_noise_std_zero_valid(tmp_path):
    cam = {**_CAMERA, "noise_std": 0.0}
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_noise_std_negative_rejected(tmp_path):
    bad = {**_CAMERA, "noise_std": -1.0}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_chroma_noise_valid(tmp_path):
    cam = {**_CAMERA, "chroma_noise": 0.05}
    validate_camera(_write(tmp_path, cam))


def test_validate_camera_chroma_noise_negative_rejected(tmp_path):
    bad = {**_CAMERA, "chroma_noise": -0.1}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# hand
# ---------------------------------------------------------------------------


def test_validate_hand_valid(tmp_path):
    validate_hand(_write(tmp_path, _HAND))


def test_validate_hand_missing_model(tmp_path):
    with pytest.raises(jsonschema.ValidationError):
        validate_hand(_write(tmp_path, {}))


def test_validate_hand_full(tmp_path):
    hand = {
        "model": "hand.blend",
        "name": "right",
        "scale_length": 1.1,
        "scale_breadth": 0.9,
        "scale_thickness": 1.0,
        "texture": "skin.png",
        "color_multiply": [0.9, 0.8, 0.7],
    }
    validate_hand(_write(tmp_path, hand))


def test_validate_hand_color_out_of_range(tmp_path):
    hand = {**_HAND, "color_multiply": [1.5, 0.5, 0.5]}
    with pytest.raises(jsonschema.ValidationError):
        validate_hand(_write(tmp_path, hand))


# ---------------------------------------------------------------------------
# camhand_rig
# ---------------------------------------------------------------------------


def test_validate_rig_valid(tmp_path):
    validate_camhand_rig(_write(tmp_path, _RIG))


def test_validate_rig_missing_T_c_h(tmp_path):
    bad = {k: v for k, v in _RIG.items() if k != "T_c_h"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camhand_rig(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# sequence
# ---------------------------------------------------------------------------


def test_validate_sequence_valid(tmp_path):
    validate_sequence(_write(tmp_path, _SEQUENCE))


def test_validate_sequence_missing_name(tmp_path):
    bad = {k: v for k, v in _SEQUENCE.items() if k != "name"}
    with pytest.raises(jsonschema.ValidationError):
        validate_sequence(_write(tmp_path, bad))


def test_validate_sequence_with_hdri(tmp_path):
    seq = {**_SEQUENCE, "hdri": "scene.exr"}
    validate_sequence(_write(tmp_path, seq))


# ---------------------------------------------------------------------------
# project
# ---------------------------------------------------------------------------


def test_validate_project_valid(tmp_path):
    validate_project(_write(tmp_path, _PROJECT))


def test_validate_project_missing_sequences(tmp_path):
    bad = {k: v for k, v in _PROJECT.items() if k != "sequences"}
    with pytest.raises(jsonschema.ValidationError):
        validate_project(_write(tmp_path, bad))


# ---------------------------------------------------------------------------
# camera_motion CSV
# ---------------------------------------------------------------------------


def _cam_csv(tmp_path, header: str, row: str = "0,0,0,0,0,0,0,1"):
    p = tmp_path / "cam.csv"
    p.write_text(f"{header}\n{row}\n")
    return p


def test_validate_camera_motion_valid(tmp_path):
    validate_camera_motion(_cam_csv(tmp_path, "timestamp,px,py,pz,qx,qy,qz,qw"))


def test_validate_camera_motion_missing_column(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        validate_camera_motion(_cam_csv(tmp_path, "timestamp,px,py,pz,qx,qy,qz"))


def test_validate_camera_motion_extra_column_allowed(tmp_path):
    # Extra columns are fine — only required ones are checked.
    validate_camera_motion(_cam_csv(tmp_path, "timestamp,px,py,pz,qx,qy,qz,qw,extra"))


# ---------------------------------------------------------------------------
# hand_motion CSV
# ---------------------------------------------------------------------------


_HAND_HEADER = "timestamp,wrist_x,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz,wrist_qw"
_HAND_ROW = "0,0,0,0,0,0,0,1"


def _hand_csv(tmp_path, header: str, row: str = _HAND_ROW):
    p = tmp_path / "hand.csv"
    p.write_text(f"{header}\n{row}\n")
    return p


def test_validate_hand_motion_valid(tmp_path):
    validate_hand_motion(_hand_csv(tmp_path, _HAND_HEADER))


def test_validate_hand_motion_missing_timestamp(tmp_path):
    header = "wrist_x,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz,wrist_qw"
    with pytest.raises(ValueError, match="timestamp"):
        validate_hand_motion(_hand_csv(tmp_path, header, row="0,0,0,0,0,0,1"))


def test_validate_hand_motion_wrong_column_count(tmp_path):
    # 6 pose columns instead of a multiple of 7
    header = "timestamp,wrist_x,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz"
    with pytest.raises(ValueError, match="7 columns per joint"):
        validate_hand_motion(_hand_csv(tmp_path, header, row="0,0,0,0,0,0,0"))


def test_validate_hand_motion_wrong_column_naming(tmp_path):
    # First column after timestamp should end in _x, but naming is wrong
    header = "timestamp,wrist_a,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz,wrist_qw"
    with pytest.raises(ValueError, match="expected columns"):
        validate_hand_motion(_hand_csv(tmp_path, header))


def test_validate_hand_motion_two_joints(tmp_path):
    header = (
        "timestamp,"
        "wrist_x,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz,wrist_qw,"
        "index_x,index_y,index_z,index_qx,index_qy,index_qz,index_qw"
    )
    validate_hand_motion(_hand_csv(tmp_path, header, row="0,0,0,0,0,0,0,1,0,0,0,0,0,0,1"))
