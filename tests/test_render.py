# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for pure-math functions in render.py (no Blender required).
bpy and mathutils are mocked in conftest.py before this module is imported.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from phanesim.render import _apply_sensor_noise, project_point
from phanesim.types import CameraModel

# ---------------------------------------------------------------------------
# Pinhole projection
# ---------------------------------------------------------------------------

_PINHOLE = CameraModel("pinhole", {"fx": 500.0, "fy": 400.0, "cx": 320.0, "cy": 240.0})


def test_pinhole_on_axis():
    p = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _PINHOLE)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(240.0)


def test_pinhole_positive_x():
    p = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _PINHOLE)
    assert u == pytest.approx(820.0)  # cx + fx * x/z = 320 + 500*1
    assert v == pytest.approx(240.0)


def test_pinhole_positive_y():
    p = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _PINHOLE)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(640.0)  # cy + fy * y/z = 240 + 400*1


def test_pinhole_scales_with_depth():
    p_near = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    p_far = np.array([1.0, 0.0, 2.0], dtype=np.float32)
    u_near, _ = project_point(p_near, _PINHOLE)
    u_far, _ = project_point(p_far, _PINHOLE)
    # Point at twice the depth should be half as far from principal point.
    assert (u_near - 320.0) == pytest.approx(2 * (u_far - 320.0))


# ---------------------------------------------------------------------------
# KB4 (Kannala-Brandt) projection
# ---------------------------------------------------------------------------

_KB4_ZERO_DIST = CameraModel(
    "kb4", {"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0, "k1": 0.0, "k2": 0.0, "k3": 0.0, "k4": 0.0}
)


def test_kb4_on_axis():
    p = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _KB4_ZERO_DIST)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(240.0)


def test_kb4_zero_distortion_matches_pinhole_on_axis():
    """With no distortion, KB4 should agree with pinhole for small angles."""
    # At 45°, theta = atan2(1, 1) = pi/4, d = theta (no distortion), r = 1.
    # u = fx * d * x/r + cx = fx * (pi/4) * 1/1 + cx
    p = np.array([1.0, 0.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _KB4_ZERO_DIST)
    theta = math.atan2(1.0, 1.0)  # 45°
    assert u == pytest.approx(500.0 * theta + 320.0, abs=1e-4)
    assert v == pytest.approx(240.0)


def test_kb4_near_zero_r_returns_principal_point():
    # Very small r (nearly on-axis) should return close to principal point.
    p = np.array([1e-12, 0.0, 1.0], dtype=np.float32)
    u, v = project_point(p, _KB4_ZERO_DIST)
    assert u == pytest.approx(320.0, abs=1e-3)
    assert v == pytest.approx(240.0, abs=1e-3)


# ---------------------------------------------------------------------------
# Unknown model
# ---------------------------------------------------------------------------


def test_unknown_model_raises():
    model = CameraModel("rt8", {})
    p = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported"):
        project_point(p, model)


# ---------------------------------------------------------------------------
# Sensor noise (_apply_sensor_noise)
# ---------------------------------------------------------------------------


class _FakeImage:
    """Minimal stand-in for a Blender Image object."""

    def __init__(self, w: int, h: int, fill: float = 0.5):
        self.size = (w, h)
        self.pixels = [fill, fill, fill, 1.0] * (w * h)
        self.filepath_raw = ""
        self.file_format = ""

    def save(self) -> None:
        pass


def test_apply_sensor_noise_alpha_unchanged(tmp_path):
    img = _FakeImage(4, 4, fill=0.5)
    with (
        patch("phanesim.render.bpy.data.images.load", return_value=img),
        patch("phanesim.render.bpy.data.images.remove"),
    ):
        _apply_sensor_noise(tmp_path, 1, noise_std=30.0)
    result = np.array(img.pixels, dtype=np.float32).reshape(4, 4, 4)
    np.testing.assert_allclose(result[:, :, 3], 1.0)


def test_apply_sensor_noise_clamped_to_unit_range(tmp_path):
    img = _FakeImage(8, 8, fill=0.5)
    with (
        patch("phanesim.render.bpy.data.images.load", return_value=img),
        patch("phanesim.render.bpy.data.images.remove"),
    ):
        _apply_sensor_noise(tmp_path, 1, noise_std=255.0)  # maximum noise
    result = np.array(img.pixels, dtype=np.float32).reshape(8, 8, 4)
    assert float(result[:, :, :3].min()) >= 0.0
    assert float(result[:, :, :3].max()) <= 1.0


def test_apply_sensor_noise_zero_std_is_noop(tmp_path):
    img = _FakeImage(4, 4, fill=0.7)
    original = list(img.pixels)
    with (
        patch("phanesim.render.bpy.data.images.load", return_value=img),
        patch("phanesim.render.bpy.data.images.remove"),
    ):
        _apply_sensor_noise(tmp_path, 1, noise_std=0.0)
    np.testing.assert_allclose(img.pixels, original, atol=1e-6)


def test_apply_sensor_noise_processes_n_frames(tmp_path):
    load_mock = MagicMock(side_effect=[_FakeImage(2, 2) for _ in range(3)])
    with patch("phanesim.render.bpy.data.images.load", load_mock), patch("phanesim.render.bpy.data.images.remove"):
        _apply_sensor_noise(tmp_path, 3, noise_std=5.0)
    assert load_mock.call_count == 3
