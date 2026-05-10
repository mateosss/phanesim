# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from phanesim.types import Transform


def _identity() -> Transform:
    return Transform.from_components([0, 0, 0], Rotation.identity())


def _rot_z(angle_rad: float) -> Transform:
    """Pure rotation around Z axis."""
    return Transform.from_components([0, 0, 0], Rotation.from_euler("z", angle_rad))


# ---------------------------------------------------------------------------
# rotation matrix
# ---------------------------------------------------------------------------


def test_identity_rotmat():
    t = _identity()
    np.testing.assert_allclose(t.rotation.as_matrix(), np.eye(3), atol=1e-6)


def test_rot_z_90_rotmat():
    t = _rot_z(math.pi / 2)
    expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    np.testing.assert_allclose(t.rotation.as_matrix(), expected, atol=1e-6)


# ---------------------------------------------------------------------------
# as_matrix (4x4)
# ---------------------------------------------------------------------------


def test_mat_is_4x4():
    t = Transform.from_components([1, 2, 3], Rotation.identity())
    assert t.as_matrix().shape == (4, 4)


def test_mat_translation():
    t = Transform.from_components([1, 2, 3], Rotation.identity())
    np.testing.assert_allclose(t.as_matrix()[:3, 3], [1, 2, 3], atol=1e-6)
    np.testing.assert_allclose(t.as_matrix()[:3, :3], np.eye(3), atol=1e-6)
    assert t.as_matrix()[3, 3] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# inv
# ---------------------------------------------------------------------------


def test_identity_inv_is_identity():
    t = _identity()
    inv = t.inv()
    np.testing.assert_allclose(inv.translation, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.abs(inv.rotation.as_quat()), [0, 0, 0, 1], atol=1e-6)


def test_inv_compose_is_identity():
    t = Transform.from_components([1, 2, 3], Rotation.from_euler("z", math.pi / 2))
    composed = t * t.inv()
    p = np.array([5, 6, 7], dtype=np.float64)
    np.testing.assert_allclose(composed.apply(p), p, atol=1e-5)


def test_inv_then_forward_is_identity():
    t = Transform.from_components([3, -1, 2], Rotation.from_euler("y", math.pi / 3))
    p = np.array([1, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(t.apply(t.inv().apply(p)), p, atol=1e-5)


# ---------------------------------------------------------------------------
# composition (*)
# ---------------------------------------------------------------------------


def test_mul_two_rot_z_90():
    t = _rot_z(math.pi / 2)
    composed = t * t
    p = np.array([1, 0, 0], dtype=np.float64)
    np.testing.assert_allclose(composed.apply(p), [-1, 0, 0], atol=1e-5)


def test_mul_translation_accumulates():
    t1 = Transform.from_components([1, 0, 0], Rotation.identity())
    t2 = Transform.from_components([0, 2, 0], Rotation.identity())
    composed = t1 * t2
    np.testing.assert_allclose(composed.translation, [1, 2, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# apply (point transformation)
# ---------------------------------------------------------------------------


def test_apply_identity_to_point():
    t = _identity()
    p = np.array([3, 4, 5], dtype=np.float64)
    np.testing.assert_allclose(t.apply(p), p, atol=1e-6)


def test_apply_translation_to_point():
    t = Transform.from_components([1, 2, 3], Rotation.identity())
    np.testing.assert_allclose(t.apply([0, 0, 0]), [1, 2, 3], atol=1e-6)


def test_apply_rot_z_90_to_x_axis():
    t = _rot_z(math.pi / 2)
    np.testing.assert_allclose(t.apply([1, 0, 0]), [0, 1, 0], atol=1e-6)


def test_apply_rot_z_90_to_y_axis():
    t = _rot_z(math.pi / 2)
    np.testing.assert_allclose(t.apply([0, 1, 0]), [-1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# Rotation.from_matrix roundtrip (replaces _rotmat_to_quat tests)
# ---------------------------------------------------------------------------


def test_rotmat_to_quat_identity():
    from scipy.spatial.transform import Rotation

    q = Rotation.from_matrix(np.eye(3)).as_quat()
    np.testing.assert_allclose(np.abs(q), [0, 0, 0, 1], atol=1e-6)


def test_rotmat_to_quat_roundtrip():
    original = _rot_z(math.pi / 3)
    recovered = Rotation.from_matrix(original.rotation.as_matrix())
    np.testing.assert_allclose(recovered.as_matrix(), original.rotation.as_matrix(), atol=1e-5)
