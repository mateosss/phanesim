# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import pytest

from phanesim.types import Transform, _rotmat_to_quat


def _identity() -> Transform:
    return Transform(
        pos=np.zeros(3, dtype=np.float32),
        quat=np.array([0, 0, 0, 1], dtype=np.float32),
    )


def _rot_z(angle_rad: float) -> Transform:
    """Pure rotation around Z axis."""
    half = angle_rad / 2
    return Transform(
        pos=np.zeros(3, dtype=np.float32),
        quat=np.array([0, 0, math.sin(half), math.cos(half)], dtype=np.float32),
    )


# ---------------------------------------------------------------------------
# rotmat
# ---------------------------------------------------------------------------


def test_identity_rotmat():
    t = _identity()
    np.testing.assert_allclose(t.rotmat, np.eye(3, dtype=np.float32), atol=1e-6)


def test_rot_z_90_rotmat():
    t = _rot_z(math.pi / 2)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float32)
    np.testing.assert_allclose(t.rotmat, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# mat (4x4)
# ---------------------------------------------------------------------------


def test_mat_is_4x4():
    t = Transform(pos=np.array([1, 2, 3], dtype=np.float32), quat=np.array([0, 0, 0, 1], dtype=np.float32))
    assert t.mat.shape == (4, 4)


def test_mat_translation():
    t = Transform(pos=np.array([1, 2, 3], dtype=np.float32), quat=np.array([0, 0, 0, 1], dtype=np.float32))
    np.testing.assert_allclose(t.mat[:3, 3], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(t.mat[:3, :3], np.eye(3), atol=1e-6)
    assert t.mat[3, 3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------


def test_identity_inv_is_identity():
    t = _identity()
    inv = t.inv
    np.testing.assert_allclose(inv.pos, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.abs(inv.quat), [0, 0, 0, 1], atol=1e-6)


def test_inv_compose_is_identity():
    t = Transform(
        pos=np.array([1, 2, 3], dtype=np.float32),
        quat=np.array([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)], dtype=np.float32),
    )
    composed = t * t.inv
    # Applying the composed transform to any point should return that point.
    p = np.array([5, 6, 7], dtype=np.float32)
    np.testing.assert_allclose(composed * p, p, atol=1e-5)


def test_inv_then_forward_is_identity():
    t = Transform(
        pos=np.array([3, -1, 2], dtype=np.float32),
        quat=np.array([0, math.sin(math.pi / 6), 0, math.cos(math.pi / 6)], dtype=np.float32),
    )
    p = np.array([1, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(t * (t.inv * p), p, atol=1e-5)


# ---------------------------------------------------------------------------
# __mul__ with Transform
# ---------------------------------------------------------------------------


def test_mul_two_rot_z_90():
    # Two 90° Z rotations should give a 180° Z rotation.
    t = _rot_z(math.pi / 2)
    composed = t * t
    p = np.array([1, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(composed * p, [-1, 0, 0], atol=1e-5)


def test_mul_translation_accumulates():
    t1 = Transform(pos=np.array([1, 0, 0], dtype=np.float32), quat=np.array([0, 0, 0, 1], dtype=np.float32))
    t2 = Transform(pos=np.array([0, 2, 0], dtype=np.float32), quat=np.array([0, 0, 0, 1], dtype=np.float32))
    composed = t1 * t2
    np.testing.assert_allclose(composed.pos, [1, 2, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# __mul__ with Vector3
# ---------------------------------------------------------------------------


def test_apply_identity_to_point():
    t = _identity()
    p = np.array([3, 4, 5], dtype=np.float32)
    np.testing.assert_allclose(t * p, p, atol=1e-6)


def test_apply_translation_to_point():
    t = Transform(pos=np.array([1, 2, 3], dtype=np.float32), quat=np.array([0, 0, 0, 1], dtype=np.float32))
    p = np.array([0, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(t * p, [1, 2, 3], atol=1e-6)


def test_apply_rot_z_90_to_x_axis():
    t = _rot_z(math.pi / 2)
    p = np.array([1, 0, 0], dtype=np.float32)
    np.testing.assert_allclose(t * p, [0, 1, 0], atol=1e-6)


def test_apply_rot_z_90_to_y_axis():
    t = _rot_z(math.pi / 2)
    p = np.array([0, 1, 0], dtype=np.float32)
    np.testing.assert_allclose(t * p, [-1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# _rotmat_to_quat roundtrip
# ---------------------------------------------------------------------------


def test_rotmat_to_quat_identity():
    q = _rotmat_to_quat(np.eye(3, dtype=np.float32))
    # Both [0,0,0,1] and [0,0,0,-1] are valid; check the rotation is identity.
    t = Transform(pos=np.zeros(3, dtype=np.float32), quat=q)
    np.testing.assert_allclose(t.rotmat, np.eye(3), atol=1e-6)


def test_rotmat_to_quat_roundtrip():
    original = _rot_z(math.pi / 3)
    recovered_q = _rotmat_to_quat(original.rotmat)
    recovered_t = Transform(pos=np.zeros(3, dtype=np.float32), quat=recovered_q)
    np.testing.assert_allclose(recovered_t.rotmat, original.rotmat, atol=1e-5)
