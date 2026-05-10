# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for pure-math functions in render.py (no Blender required).
bpy and mathutils are mocked in conftest.py before this module is imported.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phanesim.render import project_point
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
