# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the AR-headset gaze clamp.

conftest mocks bpy/mathutils, so only the numpy-based geometry is exercised here.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from phanesim.render import clamp_direction

FORWARD = np.array([0.0, -1.0, 0.0])  # the direction the body faces in rest pose


def angle_between(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return math.degrees(math.acos(float(np.clip(cos, -1.0, 1.0))))


class TestClampDirection:
    def test_within_limit_is_returned_unchanged(self):
        desired = np.array([0.2, -1.0, -0.1])
        out = clamp_direction(desired, FORWARD, math.radians(35))
        assert angle_between(out, desired) == pytest.approx(0.0, abs=1e-6)

    def test_result_is_a_unit_vector(self):
        for desired in ([1.0, -1.0, 0.0], [0.0, -1.0, -3.0], [5.0, 0.0, 0.0]):
            out = clamp_direction(np.array(desired), FORWARD, math.radians(35))
            assert np.linalg.norm(out) == pytest.approx(1.0, abs=1e-9)

    def test_beyond_limit_is_pulled_back_to_exactly_the_limit(self):
        # Straight down is 90 deg off forward; a 35 deg budget must stop at 35.
        out = clamp_direction(np.array([0.0, 0.0, -1.0]), FORWARD, math.radians(35))
        assert angle_between(out, FORWARD) == pytest.approx(35.0, abs=1e-6)

    def test_clamped_result_stays_in_the_plane_of_the_two_directions(self):
        # Turning the short way round means no sideways roll is introduced.
        desired = np.array([1.0, 0.0, 0.0])
        out = clamp_direction(desired, FORWARD, math.radians(30))
        normal = np.cross(FORWARD, desired)
        assert np.dot(out, normal) == pytest.approx(0.0, abs=1e-9)

    def test_clamped_result_turns_toward_the_target(self):
        # The clamped gaze must be closer to the target than straight ahead was.
        desired = np.array([0.0, 0.0, -1.0])
        out = clamp_direction(desired, FORWARD, math.radians(35))
        assert angle_between(out, desired) < angle_between(FORWARD, desired)

    def test_zero_budget_gives_a_rigid_forward_camera(self):
        out = clamp_direction(np.array([1.0, 0.0, 0.0]), FORWARD, 0.0)
        assert angle_between(out, FORWARD) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_direction_keeps_looking_ahead(self):
        # Directly behind has no unique short way round; the gaze must not spin.
        out = clamp_direction(-FORWARD, FORWARD, math.radians(35))
        assert angle_between(out, FORWARD) == pytest.approx(0.0, abs=1e-6)

    def test_degenerate_desired_falls_back_to_base(self):
        out = clamp_direction(np.zeros(3), FORWARD, math.radians(35))
        assert angle_between(out, FORWARD) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("limit_deg", [0.0, 10.0, 35.0, 60.0, 90.0])
    def test_never_exceeds_the_budget_from_any_direction(self, limit_deg):
        rng = np.random.default_rng(0)
        for _ in range(200):
            desired = rng.normal(size=3)
            if np.linalg.norm(desired) < 1e-6:
                continue
            out = clamp_direction(desired, FORWARD, math.radians(limit_deg))
            assert angle_between(out, FORWARD) <= limit_deg + 1e-6
