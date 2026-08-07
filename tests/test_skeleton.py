# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from phanesim.skeleton import (
    HAND_CONNECTIONS,
    HAND_LANDMARKS,
    LANDMARK_COLORS,
    rigify_hand_landmarks,
)

OPENXR_NAMES = [
    "Wrist",
    "ThumbMetacarpal",
    "ThumbProximal",
    "ThumbDistal",
    "ThumbTip",
    "IndexProximal",
    "IndexIntermediate",
    "IndexDistal",
    "IndexTip",
    "MiddleProximal",
    "MiddleIntermediate",
    "MiddleDistal",
    "MiddleTip",
    "RingProximal",
    "RingIntermediate",
    "RingDistal",
    "RingTip",
    "LittleProximal",
    "LittleIntermediate",
    "LittleDistal",
    "LittleTip",
]


class TestLandmarkTables:
    def test_hand_rig_uses_openxr_names(self):
        assert [name for name, _, _ in HAND_LANDMARKS] == OPENXR_NAMES

    def test_tables_agree_on_length(self):
        assert len(HAND_LANDMARKS) == len(LANDMARK_COLORS) == 21

    def test_connections_are_in_range(self):
        for a, b in HAND_CONNECTIONS:
            assert 0 <= a < 21
            assert 0 <= b < 21


class TestRigifyHandLandmarks:
    @pytest.mark.parametrize("side", ["left", "right", "L", "R", "Right", "LEFT"])
    def test_accepts_side_spellings(self, side):
        assert len(rigify_hand_landmarks(side)) == 21

    def test_rejects_unknown_side(self):
        with pytest.raises(ValueError, match="side must be"):
            rigify_hand_landmarks("middle")

    def test_uses_the_same_openxr_names_as_the_hand_rig(self):
        # Both rigs must produce identical CSV column names so downstream code
        # does not care which model a dataset came from.
        assert [name for name, _, _ in rigify_hand_landmarks("right")] == OPENXR_NAMES

    def test_suffix_matches_side(self):
        assert all(bone.endswith(".R") for _, _, bone in rigify_hand_landmarks("right"))
        assert all(bone.endswith(".L") for _, _, bone in rigify_hand_landmarks("left"))

    def test_only_fingertips_read_the_bone_tail(self):
        tails = [name for name, src, _ in rigify_hand_landmarks("right") if src == "arm_tail"]
        assert tails == ["ThumbTip", "IndexTip", "MiddleTip", "RingTip", "LittleTip"]

    def test_thumb_chain_maps_onto_rigify_numbering(self):
        bones = dict((name, bone) for name, _, bone in rigify_hand_landmarks("right"))
        assert bones["ThumbMetacarpal"] == "ORG-thumb.01.R"
        assert bones["ThumbProximal"] == "ORG-thumb.02.R"
        assert bones["ThumbDistal"] == "ORG-thumb.03.R"
        assert bones["ThumbTip"] == "ORG-thumb.03.R"

    def test_little_finger_maps_to_rigify_pinky(self):
        bones = dict((name, bone) for name, _, bone in rigify_hand_landmarks("right"))
        assert bones["LittleProximal"] == "ORG-f_pinky.01.R"
        assert bones["LittleTip"] == "ORG-f_pinky.03.R"

    def test_distal_and_tip_share_a_bone(self):
        bones = dict((name, bone) for name, _, bone in rigify_hand_landmarks("left"))
        for finger in ("Thumb", "Index", "Middle", "Ring", "Little"):
            assert bones[f"{finger}Distal"] == bones[f"{finger}Tip"]

    def test_bone_names_are_unique_per_landmark_role(self):
        landmarks = rigify_hand_landmarks("right")
        # (bone, source) pairs must be distinct: two landmarks reading the same
        # end of the same bone would be duplicate columns.
        pairs = [(bone, src) for _, src, bone in landmarks]
        assert len(set(pairs)) == len(pairs)
