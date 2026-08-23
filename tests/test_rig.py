# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for rig.py dataclass loading — no Blender required."""

from __future__ import annotations

import pytest

from phanesim.rig import _camera_from_dict
from phanesim.types import Shutter

_BASE_CAMERA = {
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 240.0, "fy": 240.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "pixel_format": "MONO8",
    "shutter": "global",
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


def test_camera_chroma_noise_loaded(tmp_path):
    cam = _camera_from_dict({**_BASE_CAMERA, "chroma_noise": 0.05}, tmp_path)
    assert cam.chroma_noise == pytest.approx(0.05)


def test_camera_chroma_noise_defaults_to_zero(tmp_path):
    cam = _camera_from_dict(_BASE_CAMERA, tmp_path)
    assert cam.chroma_noise == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Camera — compositor node parameters
# ---------------------------------------------------------------------------


def test_camera_compositor_params_loaded(tmp_path):
    data = {
        **_BASE_CAMERA,
        "ca_factor": 0.5,
        "distortion": 0.2,
        "dispersion": 0.1,
        "lens_scale": 1.3,
        "vignette_factor": 0.8,
        "vignette_feather": 0.6,
    }
    cam = _camera_from_dict(data, tmp_path)
    assert cam.ca_factor == pytest.approx(0.5)
    assert cam.distortion == pytest.approx(0.2)
    assert cam.dispersion == pytest.approx(0.1)
    assert cam.lens_scale == pytest.approx(1.3)
    assert cam.vignette_factor == pytest.approx(0.8)
    assert cam.vignette_feather == pytest.approx(0.6)


def test_camera_compositor_params_defaults(tmp_path):
    cam = _camera_from_dict(_BASE_CAMERA, tmp_path)
    assert cam.ca_factor == pytest.approx(0.30)
    assert cam.distortion == pytest.approx(0.387)
    assert cam.dispersion == pytest.approx(0.0)
    assert cam.lens_scale == pytest.approx(1.2)
    assert cam.vignette_factor == pytest.approx(0.533)
    assert cam.vignette_feather == pytest.approx(0.4)


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
