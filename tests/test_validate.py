# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import jsonschema
import pytest

from phanesim.validate import validate_camera

# ---------------------------------------------------------------------------
# Shared valid fixtures
# ---------------------------------------------------------------------------

_TRANSFORM = {"pos": [0.0, 0.0, 0.0], "quat": [0.0, 0.0, 0.0, 1.0]}

_CAMERA = {
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "pixel_format": "RGB24",
    "shutter": "global",
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
    bad = {k: v for k, v in _CAMERA.items() if k != "resolution"}
    with pytest.raises(jsonschema.ValidationError):
        validate_camera(_write(tmp_path, bad))


def test_validate_camera_has_no_rate_or_pose_fields(tmp_path):
    # The frame count lives on the sequence and the pose comes from the head
    # bone, so a camera that still declares them is a stale config.
    for dead in ("frequency", "T_b_c", "exposure", "gain"):
        bad = {**_CAMERA, dead: 1}
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
