# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the compositor's forward lens-distortion map.

The reference values come from measuring Blender's CPU compositor directly: a
grid of known points was rendered through the Lens Distortion node and the
resulting positions recovered, giving agreement to ~0.1 px.  These tests pin
that behaviour so a change to the map is caught rather than silently shifting
every ground-truth coordinate.
"""

from __future__ import annotations

import math

import pytest

from phanesim.render import distort_pixel

W, H = 640, 480


def radius(x: float, y: float) -> float:
    """Distance from the image centre in the compositor's normalised units."""
    return math.hypot((x + 0.5 - W / 2) / (W / 2), (y + 0.5 - H / 2) / (H / 2))


class TestNoDistortion:
    def test_zero_distortion_is_the_identity(self):
        for x, y in ((0, 0), (60, 45), (320, 240), (639, 479)):
            px, py = distort_pixel(x, y, W, H, 0.0)
            assert px == pytest.approx(x, abs=1e-9)
            assert py == pytest.approx(y, abs=1e-9)

    def test_image_centre_is_a_fixed_point(self):
        # The centre has zero radius, so no radial map can move it.
        cx, cy = W / 2 - 0.5, H / 2 - 0.5
        for k in (0.0, 0.1, 0.387, 0.9):
            px, py = distort_pixel(cx, cy, W, H, k)
            assert px == pytest.approx(cx, abs=1e-9)
            assert py == pytest.approx(cy, abs=1e-9)


class TestRadialMap:
    @pytest.mark.parametrize("k", [0.1, 0.2, 0.387, 0.6])
    def test_matches_the_measured_relation(self, k):
        # r_out = r_in * (1 + k) / (1 + k * r_in^2), fitted to Blender output.
        for x, y in ((100, 90), (200, 150), (480, 360), (600, 60)):
            r_in = radius(x, y)
            expected = r_in * (1.0 + k) / (1.0 + k * r_in * r_in)
            px, py = distort_pixel(x, y, W, H, k)
            assert radius(px, py) == pytest.approx(expected, rel=1e-9)

    def test_direction_from_centre_is_preserved(self):
        # A radial map may move a point in or out, never sideways.
        for x, y in ((100, 90), (500, 400), (620, 45)):
            ax = math.atan2((y + 0.5 - H / 2) / (H / 2), (x + 0.5 - W / 2) / (W / 2))
            px, py = distort_pixel(x, y, W, H, 0.387)
            bx = math.atan2((py + 0.5 - H / 2) / (H / 2), (px + 0.5 - W / 2) / (W / 2))
            assert bx == pytest.approx(ax, abs=1e-9)

    def test_small_radius_magnifies_by_one_plus_k(self):
        # The measured ratio tends to (1 + k) as the radius goes to zero.
        k = 0.387
        x, y = W / 2 - 0.5 + 0.4, H / 2 - 0.5
        px, _ = distort_pixel(x, y, W, H, k)
        moved = (px + 0.5 - W / 2) / (x + 0.5 - W / 2)
        assert moved == pytest.approx(1.0 + k, rel=1e-3)

    def test_anisotropic_normalisation(self):
        # x is normalised by width/2 and y by height/2, so on a non-square image
        # equal pixel offsets in x and y are not equally distorted.
        k = 0.387
        dx, _ = distort_pixel(W / 2 - 0.5 + 100, H / 2 - 0.5, W, H, k)
        _, dy = distort_pixel(W / 2 - 0.5, H / 2 - 0.5 + 100, W, H, k)
        assert (dx - (W / 2 - 0.5)) != pytest.approx(dy - (H / 2 - 0.5), rel=1e-3)


class TestLensScale:
    def test_scale_multiplies_the_offset_from_centre(self):
        k, s = 0.387, 1.2
        cx, cy = W / 2 - 0.5, H / 2 - 0.5
        ux, uy = distort_pixel(200, 150, W, H, k, 1.0)
        sx, sy = distort_pixel(200, 150, W, H, k, s)
        assert sx - cx == pytest.approx((ux - cx) * s, rel=1e-9)
        assert sy - cy == pytest.approx((uy - cy) * s, rel=1e-9)

    def test_scale_of_one_is_the_default(self):
        assert distort_pixel(200, 150, W, H, 0.387) == distort_pixel(200, 150, W, H, 0.387, 1.0)

    def test_scale_alone_with_no_distortion(self):
        cx = W / 2 - 0.5
        px, _ = distort_pixel(cx + 100, H / 2 - 0.5, W, H, 0.0, 1.2)
        assert px - cx == pytest.approx(120.0, rel=1e-9)


class TestOutsideTheValidRadius:
    """The map turns over at r = 1/sqrt(k) and folds outer points back inward.

    Blender writes a transparent pixel past that radius, so a landmark out there
    has no position in the rendered image and must be reported as NaN rather
    than as a plausible-looking coordinate inside the frame.
    """

    def test_far_outside_is_nan_not_folded_inward(self):
        # Without the guard this lands near the centre of the image.
        px, py = distort_pixel(20000, 20000, W, H, 0.387)
        assert math.isnan(px) and math.isnan(py)

    def test_the_cutoff_is_where_blender_puts_it(self):
        k = 0.387
        limit = 1.0 / math.sqrt(k)  # k * r^2 == 1
        for scale_r, expect_nan in ((0.98, False), (1.02, True)):
            # place a point along +x at the requested fraction of the limit
            x = (W / 2 - 0.5) + limit * scale_r * (W / 2)
            px, _ = distort_pixel(x, H / 2 - 0.5, W, H, k)
            assert math.isnan(px) is expect_nan

    def test_no_distortion_never_rejects(self):
        # k = 0 makes the guard vacuous; the identity must hold everywhere.
        px, py = distort_pixel(50000, -50000, W, H, 0.0)
        assert px == pytest.approx(50000)
        assert py == pytest.approx(-50000)

    def test_map_is_monotonic_inside_the_valid_radius(self):
        k = 0.387
        limit = 1.0 / math.sqrt(k)
        prev = -1.0
        for frac in [i / 50 for i in range(1, 50)]:
            x = (W / 2 - 0.5) + limit * frac * (W / 2)
            px, _ = distort_pixel(x, H / 2 - 0.5, W, H, k)
            assert not math.isnan(px)
            assert px > prev
            prev = px
