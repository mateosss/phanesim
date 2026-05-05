# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math

import numpy as np
import pytest

from phanesim.motion import CameraMotion, HandMotion, _slerp

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def cam_csv(tmp_path):
    """Two-row camera motion: identity at t=0, +1 on X at t=1s."""
    p = tmp_path / "cam.csv"
    p.write_text("timestamp,px,py,pz,qx,qy,qz,qw\n0,0,0,0,0,0,0,1\n1000000000,1,0,0,0,0,0,1\n")
    return p


@pytest.fixture()
def hand_csv(tmp_path):
    """Two joints (wrist, index), two rows."""
    p = tmp_path / "hand.csv"
    p.write_text(
        "timestamp,"
        "wrist_x,wrist_y,wrist_z,wrist_qx,wrist_qy,wrist_qz,wrist_qw,"
        "index_x,index_y,index_z,index_qx,index_qy,index_qz,index_qw\n"
        "0,0,0,0,0,0,0,1,0.1,0,0,0,0,0,1\n"
        "1000000000,1,0,0,0,0,0,1,1.1,0,0,0,0,0,1\n"
    )
    return p


# ---------------------------------------------------------------------------
# _slerp
# ---------------------------------------------------------------------------


def test_slerp_at_zero_returns_q0():
    q0 = np.array([0, 0, 0, 1], dtype=np.float32)
    q1 = np.array([0, 0, 1, 0], dtype=np.float32)
    result = _slerp(q0, q1, 0.0)
    np.testing.assert_allclose(result, q0, atol=1e-6)


def test_slerp_at_one_returns_q1():
    q0 = np.array([0, 0, 0, 1], dtype=np.float32)
    q1 = np.array([0, 0, 1, 0], dtype=np.float32)
    result = _slerp(q0, q1, 1.0)
    np.testing.assert_allclose(np.abs(result), np.abs(q1), atol=1e-6)


def test_slerp_midpoint_is_normalized():
    q0 = np.array([0, 0, 0, 1], dtype=np.float32)
    half = math.pi / 4  # 45° → midpoint of 0° to 90°
    q1 = np.array([0, 0, math.sin(half * 2), math.cos(half * 2)], dtype=np.float32)
    result = _slerp(q0, q1, 0.5)
    assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-6)


def test_slerp_midpoint_rot_z_90():
    """Midpoint between identity and 90° Z rotation should be 45° Z rotation."""
    q0 = np.array([0, 0, 0, 1], dtype=np.float32)
    q1 = np.array([0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4)], dtype=np.float32)
    result = _slerp(q0, q1, 0.5)
    expected = np.array([0, 0, math.sin(math.pi / 8), math.cos(math.pi / 8)], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_slerp_antipodal_takes_shortest_arc():
    """SLERP of q and -q should not flip direction."""
    q0 = np.array([0, 0, 0, 1], dtype=np.float32)
    q1 = np.array([0, 0, 0, -1], dtype=np.float32)  # same rotation, opposite sign
    result = _slerp(q0, q1, 0.5)
    # Should stay near identity (w close to ±1, xyz close to 0)
    assert abs(float(result[3])) == pytest.approx(1.0, abs=1e-5)


def test_slerp_identical_quaternions():
    """Nearly identical quaternions fall back to normalised lerp."""
    q = np.array([0, 0, 0, 1], dtype=np.float32)
    result = _slerp(q, q, 0.5)
    np.testing.assert_allclose(result, q, atol=1e-6)


# ---------------------------------------------------------------------------
# CameraMotion
# ---------------------------------------------------------------------------


def test_camera_motion_loads(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    assert cm.ts.shape == (2,)
    assert cm.xyz.shape == (2, 3)
    assert cm.quats.shape == (2, 4)
    assert cm.source == cam_csv


def test_camera_motion_at_start(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    pose = cm.get_pose(np.int64(0))
    np.testing.assert_allclose(pose.pos, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(np.abs(pose.quat), [0, 0, 0, 1], atol=1e-6)


def test_camera_motion_at_end(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    pose = cm.get_pose(np.int64(1_000_000_000))
    np.testing.assert_allclose(pose.pos, [1, 0, 0], atol=1e-6)


def test_camera_motion_midpoint_lerp(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    pose = cm.get_pose(np.int64(500_000_000))
    np.testing.assert_allclose(pose.pos, [0.5, 0, 0], atol=1e-6)


def test_camera_motion_before_start_clamps(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    pose = cm.get_pose(np.int64(-1))
    np.testing.assert_allclose(pose.pos, [0, 0, 0], atol=1e-6)


def test_camera_motion_after_end_clamps(cam_csv):
    cm = CameraMotion.from_path(cam_csv)
    pose = cm.get_pose(np.int64(2_000_000_000))
    np.testing.assert_allclose(pose.pos, [1, 0, 0], atol=1e-6)


# ---------------------------------------------------------------------------
# HandMotion
# ---------------------------------------------------------------------------


def test_hand_motion_loads(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    assert hm.joint_names == ["wrist", "index"]
    assert hm.ts.shape == (2,)
    assert hm.joints_xyz.shape == (2, 2, 3)
    assert hm.joints_quats.shape == (2, 2, 4)


def test_hand_motion_at_start(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    poses = hm.get_joint_poses(np.int64(0))
    assert len(poses) == 2
    np.testing.assert_allclose(poses[0].pos, [0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(poses[1].pos, [0.1, 0, 0], atol=1e-6)


def test_hand_motion_at_end(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    poses = hm.get_joint_poses(np.int64(1_000_000_000))
    np.testing.assert_allclose(poses[0].pos, [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(poses[1].pos, [1.1, 0, 0], atol=1e-6)


def test_hand_motion_midpoint_lerp(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    poses = hm.get_joint_poses(np.int64(500_000_000))
    np.testing.assert_allclose(poses[0].pos, [0.5, 0, 0], atol=1e-6)
    np.testing.assert_allclose(poses[1].pos, [0.6, 0, 0], atol=1e-5)


def test_hand_motion_before_start_clamps(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    poses = hm.get_joint_poses(np.int64(-1))
    np.testing.assert_allclose(poses[0].pos, [0, 0, 0], atol=1e-6)


def test_hand_motion_after_end_clamps(hand_csv):
    hm = HandMotion.from_path(hand_csv)
    poses = hm.get_joint_poses(np.int64(2_000_000_000))
    np.testing.assert_allclose(poses[0].pos, [1, 0, 0], atol=1e-6)
