# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import jsonschema
import pytest

from phanesim.validate import (
    validate_body_rig,
    validate_body_sequence,
    validate_pose_motion,
)

_CAMERA = {
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 240.0, "fy": 240.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "pixel_format": "MONO8",
    "shutter": "global",
}

_BODY_RIG = {
    "cameras": [_CAMERA],
    "body": {"model": "../../models/model1/model1.blend", "armature": "rig", "hands": ["right", "left"]},
    "head_camera": {
        "anchor_bone": "ORG-spine.006",
        "rest_position": [0.0, -0.21, 1.715],
        "rest_forward": [0.0, -1.0, -0.268],
    },
}

_BODY_SEQUENCE = {
    "name": "poisson",
    "frames": 21,
    "body_rig": _BODY_RIG,
    "hand_motions": ["animation01.json"],
}

_POSE_MOTION = {
    "name": "animation01",
    "duration_ns": 8_000_000_000,
    "events": [
        {"t_ns": 0, "poses": [{"asset": "Arm_left_x1_y1", "alpha": 1.0}]},
        {"t_ns": 1_500_000_000, "poses": [{"asset": "Arm_left_x2_y1", "alpha": 0.7}, {"asset": "Index_left_y1"}]},
    ],
}


def _write(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


class TestBodyRig:
    def test_valid(self, tmp_path):
        validate_body_rig(_write(tmp_path, "r.json", _BODY_RIG))

    def test_head_camera_is_optional(self, tmp_path):
        data = {k: v for k, v in _BODY_RIG.items() if k != "head_camera"}
        validate_body_rig(_write(tmp_path, "r.json", data))

    def test_body_is_required(self, tmp_path):
        data = {k: v for k, v in _BODY_RIG.items() if k != "body"}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_rig(_write(tmp_path, "r.json", data))

    def test_unknown_hand_side_rejected(self, tmp_path):
        data = {**_BODY_RIG, "body": {**_BODY_RIG["body"], "hands": ["middle"]}}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_rig(_write(tmp_path, "r.json", data))

    def test_duplicate_hands_rejected(self, tmp_path):
        data = {**_BODY_RIG, "body": {**_BODY_RIG["body"], "hands": ["right", "right"]}}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_rig(_write(tmp_path, "r.json", data))

    def test_aiming_fields_are_rejected(self, tmp_path):
        # The camera is rigid; a config that still asks it to follow the hands
        # should fail loudly rather than be silently ignored.
        for dead in ("track_hands", "track_landmark", "max_deviation_deg"):
            head = {**_BODY_RIG["head_camera"], dead: ["left"] if dead == "track_hands" else 1}
            with pytest.raises(jsonschema.ValidationError):
                validate_body_rig(_write(tmp_path, "r.json", {**_BODY_RIG, "head_camera": head}))

    def test_rest_position_must_be_3d(self, tmp_path):
        head = {**_BODY_RIG["head_camera"], "rest_position": [0.0, -0.21]}
        data = {**_BODY_RIG, "head_camera": head}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_rig(_write(tmp_path, "r.json", data))


class TestBodySequence:
    def test_valid(self, tmp_path):
        validate_body_sequence(_write(tmp_path, "s.json", _BODY_SEQUENCE))

    def test_frames_is_required(self, tmp_path):
        # The frame count is the only way to say how much to render.
        data = {k: v for k, v in _BODY_SEQUENCE.items() if k != "frames"}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_sequence(_write(tmp_path, "s.json", data))

    def test_hand_motions_required(self, tmp_path):
        data = {k: v for k, v in _BODY_SEQUENCE.items() if k != "hand_motions"}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_sequence(_write(tmp_path, "s.json", data))

    def test_cam_motions_not_allowed(self, tmp_path):
        # A body sequence derives its camera from the head; a camera motion file
        # would silently do nothing, so the schema rejects it outright.
        data = {**_BODY_SEQUENCE, "cam_motions": ["cam.csv"]}
        with pytest.raises(jsonschema.ValidationError):
            validate_body_sequence(_write(tmp_path, "s.json", data))


class TestPoseMotion:
    def test_valid(self, tmp_path):
        validate_pose_motion(_write(tmp_path, "a.json", _POSE_MOTION))

    def test_events_must_be_sorted(self, tmp_path):
        data = {**_POSE_MOTION, "events": list(reversed(_POSE_MOTION["events"]))}
        with pytest.raises(ValueError, match="sorted by t_ns"):
            validate_pose_motion(_write(tmp_path, "a.json", data))

    def test_events_must_fit_in_duration(self, tmp_path):
        data = {**_POSE_MOTION, "duration_ns": 1_000_000_000}
        with pytest.raises(ValueError, match="start after duration_ns"):
            validate_pose_motion(_write(tmp_path, "a.json", data))

    def test_empty_events_rejected(self, tmp_path):
        data = {**_POSE_MOTION, "events": []}
        with pytest.raises(jsonschema.ValidationError):
            validate_pose_motion(_write(tmp_path, "a.json", data))

    def test_kind_field_rejected(self, tmp_path):
        # Multi-frame playback is gone, so "kind" no longer means anything. A
        # timeline still carrying it is from the old format and should fail
        # loudly rather than be read with the field quietly ignored.
        for dead in ("kind", "duration_ns", "source_frames"):
            data = {**_POSE_MOTION, "events": [{"t_ns": 0, "poses": [{"asset": "x"}], dead: 1}]}
            with pytest.raises(jsonschema.ValidationError):
                validate_pose_motion(_write(tmp_path, "a.json", data))

    def test_negative_duration_rejected(self, tmp_path):
        data = {**_POSE_MOTION, "duration_ns": -1}
        with pytest.raises(jsonschema.ValidationError):
            validate_pose_motion(_write(tmp_path, "a.json", data))

    def test_generated_file_validates(self, tmp_path):
        # The generator and the schema must agree; this is the contract between
        # `generate-motion` and `generate`.
        from phanesim.posemotion import NS_PER_SECOND, PoseAsset, sample_pose_motion

        assets = [PoseAsset(f"P{i}", 1, group="right", part="arm") for i in range(4)] + [
            PoseAsset("L0", 1, group="left", part="arm")
        ]
        motion = sample_pose_motion(assets, 30 * NS_PER_SECOND, seed=3)
        path = tmp_path / "generated.json"
        motion.write(path)
        validate_pose_motion(path)
