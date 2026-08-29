# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the --accessories, --camera, --rotate and lane-building CLI logic.

No Blender required.

The parse helpers build a Python expression that is evaluated inside the Blender
subprocess, so what is checked here is that the expression says what the flag
meant.
"""

from __future__ import annotations

import click
import pytest

from phanesim.cli import (
    ACCESSORY_NAMES,
    SWEEP_DIRECTIONS,
    _build_lanes,
    _parse_accessories,
    _parse_camera,
)


class TestParseAccessories:
    def test_default_wears_nothing(self):
        # The renderer starts from bare hands, so a frame only ever shows what
        # was explicitly asked for rather than whatever the .blend was saved with.
        assert eval(_parse_accessories("none")) == set()

    def test_none_removes_everything(self):
        assert eval(_parse_accessories("none")) == set()

    def test_all_selects_every_accessory(self):
        assert eval(_parse_accessories("all")) == set(ACCESSORY_NAMES)

    def test_single_name(self):
        assert eval(_parse_accessories("watch1")) == {"watch1"}

    def test_list_is_split_and_trimmed(self):
        assert eval(_parse_accessories(" ring1 , band1 ")) == {"ring1", "band1"}

    def test_duplicates_collapse(self):
        assert eval(_parse_accessories("ring1,ring1")) == {"ring1"}

    def test_empty_string_is_empty_set(self):
        assert eval(_parse_accessories("")) == set()

    def test_unknown_name_is_rejected(self):
        with pytest.raises(click.BadParameter, match="unknown accessory"):
            _parse_accessories("ring3")

    def test_unknown_name_alongside_valid_is_still_rejected(self):
        with pytest.raises(click.BadParameter, match="ring9"):
            _parse_accessories("watch1,ring9")

    def test_expression_is_stable_between_runs(self):
        # The expression is embedded in a subprocess command line; an unordered
        # set repr would make otherwise identical runs differ.
        assert _parse_accessories("watch1,ring1") == _parse_accessories("ring1,watch1")

    @pytest.mark.parametrize("name", ACCESSORY_NAMES)
    def test_every_advertised_name_is_accepted(self, name):
        assert eval(_parse_accessories(name)) == {name}


class TestParseCamera:
    def test_omitted_means_the_camera_only_follows_the_head(self):
        assert _parse_camera(None) == "None"

    def test_direction_and_degrees_are_parsed(self):
        assert _parse_camera("right,30") == "CameraSweep('right', 30.0)"

    def test_direction_is_case_insensitive_and_trimmed(self):
        assert _parse_camera(" Right , 30 ") == "CameraSweep('right', 30.0)"

    @pytest.mark.parametrize("direction", SWEEP_DIRECTIONS)
    def test_every_advertised_direction_is_accepted(self, direction):
        assert direction in _parse_camera(f"{direction},15")

    def test_negative_degrees_are_allowed(self):
        # right,-30 is just left,30; there is no reason to reject it.
        assert _parse_camera("right,-30") == "CameraSweep('right', -30.0)"

    def test_unknown_direction_is_rejected(self):
        with pytest.raises(click.BadParameter, match="direction must be one of"):
            _parse_camera("sideways,30")

    def test_missing_degrees_is_rejected(self):
        with pytest.raises(click.BadParameter, match="DIRECTION,DEGREES"):
            _parse_camera("right")

    def test_non_numeric_degrees_is_rejected(self):
        with pytest.raises(click.BadParameter, match="DIRECTION,DEGREES"):
            _parse_camera("right,lots")


class TestBuildLanes:
    """--events, --hand and --head decide which timelines get sampled."""

    def test_default_is_one_shared_timeline(self):
        # --events N means N poses in the clip, each of them a left, right or
        # whole-body pose -- not N per body part.
        lanes, _, _ = _build_lanes(hand=None, head=False)
        assert lanes == (("left", "right", "body"),)

    def test_hand_splits_the_hands_onto_their_own_timelines(self):
        lanes, _, _ = _build_lanes(hand=4, head=False)
        assert lanes == (("left",), ("right",))

    def test_whole_body_poses_only_appear_in_the_shared_timeline(self):
        # With one timeline per hand there is nowhere to put a pose that keys
        # both, so those assets sit out.
        lanes, _, _ = _build_lanes(hand=4, head=False)
        assert "body" not in {g for lane in lanes for g in lane}

    def test_head_adds_a_timeline_of_its_own(self):
        lanes, _, _ = _build_lanes(hand=None, head=True)
        assert lanes == (("left", "right", "body"), ("head",))

    def test_head_combines_with_hand(self):
        lanes, _, _ = _build_lanes(hand=2, head=True)
        assert lanes == (("left",), ("right",), ("head",))

    def test_head_is_absent_unless_asked_for(self):
        for hand in (None, 3):
            lanes, _, _ = _build_lanes(hand=hand, head=False)
            assert all("head" not in lane for lane in lanes)

    def test_head_is_never_in_a_shared_lane(self):
        # It has to stay separate, otherwise a head pose would be one of the
        # things a hand event could turn out to be.
        for hand in (None, 3):
            lanes, _, _ = _build_lanes(hand=hand, head=True)
            assert ("head",) in lanes
            assert all("head" not in lane for lane in lanes if lane != ("head",))
