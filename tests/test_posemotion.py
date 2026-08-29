# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import json

import pytest

from phanesim.posemotion import (
    NS_PER_SECOND,
    PoseAsset,
    PoseEvent,
    PoseMotion,
    sample_pose_motion,
)

# Single-group library: the sampler then behaves as one flat timeline.
POSES = [PoseAsset(name=f"Pose_{i}", frame=1, group="right") for i in range(5)]

# Multi-group library: 3 left x 4 right x 2 head, all touching disjoint bones.
# One timeline per body part, the arrangement --hand --head asks for.
SPLIT_LANES = (("left",), ("right",), ("head",))
# One shared timeline, the default: every event is whichever pose was drawn.
SHARED_LANE = (("left", "right", "body"),)

LAYERED = (
    [PoseAsset(name=f"L{i}", frame=1, group="left") for i in range(3)]
    + [PoseAsset(name=f"R{i}", frame=1, group="right") for i in range(4)]
    + [PoseAsset(name=f"H{i}", frame=1, group="head") for i in range(2)]
)


class TestPoseAsset:
    def test_roundtrip(self):
        asset = PoseAsset(name="Right_ok", frame=1, group="right")
        assert PoseAsset.from_dict(asset.to_dict()) == asset

    def test_group_defaults_to_body(self):
        assert PoseAsset(name="Right_ok", frame=1).group == "body"


class TestPoseEvent:
    def test_roundtrip(self):
        event = PoseEvent(t_ns=1000, asset="Right_ok", group="right")
        assert PoseEvent.from_dict(event.to_dict()) == event

    def test_an_event_is_only_a_time_a_name_and_a_group(self):
        # Nothing about playback: every asset is a single-frame pose.
        assert set(PoseEvent(t_ns=1000, asset="Right_ok").to_dict()) == {"t_ns", "asset", "group"}


class TestSamplePoseMotion:
    def test_seed_is_reproducible(self):
        a = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=7)
        b = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=7)
        assert a.to_dict() == b.to_dict()

    def test_different_seeds_differ(self):
        # The whole point of the generator: each run is a different timeline.
        a = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=1)
        b = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=2)
        assert [e.asset for e in a.events] != [e.asset for e in b.events] or [e.t_ns for e in a.events] != [
            e.t_ns for e in b.events
        ]

    def test_events_are_sorted_and_within_duration(self):
        motion = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=3)
        times = [e.t_ns for e in motion.events]
        assert times == sorted(times)
        assert all(t <= motion.duration_ns for t in times)

    def test_starts_at_zero(self):
        # Without a key at t=0 the first frames would open mid-interpolation.
        motion = sample_pose_motion(POSES, 30 * NS_PER_SECOND, seed=5)
        assert motion.events[0].t_ns == 0

    @pytest.mark.parametrize("n", [1, 2, 4, 10, 50])
    def test_event_count_is_exact_per_group(self, n):
        # With one group the count is exactly n; the guarantee is per group.
        motion = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=11, event_count=n)
        assert motion.event_count == n

    @pytest.mark.parametrize("n", [1, 2, 6])
    def test_layered_library_yields_one_timeline_per_group(self, n):
        motion = sample_pose_motion(LAYERED, 20 * NS_PER_SECOND, seed=11, event_count=n, lanes=SPLIT_LANES)
        assert motion.event_count == 3 * n
        for group in ("left", "right", "head"):
            assert sum(1 for e in motion.events if e.group == group) == n

    def test_the_head_stays_still_unless_asked_for(self):
        # A frozen head is the default: the camera is bolted to it, so moving it
        # is a deliberate choice rather than something that happens by accident.
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=1, event_count=4)
        assert "head" not in {e.group for e in motion.events}

    def test_a_lane_can_be_restricted_to_one_group(self):
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=1, event_count=4, lanes=(("right",),))
        assert {e.group for e in motion.events} == {"right"}

    def test_a_shared_lane_draws_from_every_group_in_it(self):
        # The default arrangement: one timeline, each event whichever pose was
        # drawn, so the count is the count and not a per-part count.
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=3, event_count=20, lanes=SHARED_LANE)
        assert motion.event_count == 20
        assert {e.group for e in motion.events} == {"left", "right"}

    def test_lanes_with_no_assets_are_skipped(self):
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=1, event_count=3, lanes=(("left",), ("body",)))
        assert {e.group for e in motion.events} == {"left"}

    def test_unknown_lane_rejected(self):
        with pytest.raises(ValueError, match="no assets in lanes"):
            sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, lanes=(("nose",),))

    def test_hand_and_head_timelines_are_independent(self):
        # The point of the split: the head can turn at a moment when neither
        # hand changes, and vice versa.
        motion = sample_pose_motion(LAYERED, 9 * NS_PER_SECOND, seed=4, event_count=6, lanes=SPLIT_LANES)
        head_times = {e.t_ns for e in motion.events if e.group == "head"}
        hand_times = {e.t_ns for e in motion.events if e.group in ("left", "right")}
        assert head_times - hand_times

    def test_four_poses_in_one_second(self):
        # The case that motivated the redesign; no minimum spacing stands in the way.
        motion = sample_pose_motion(POSES, NS_PER_SECOND, seed=1, event_count=4)
        assert motion.event_count == 4
        assert all(0 <= e.t_ns <= NS_PER_SECOND for e in motion.events)

    def test_event_count_below_one_rejected(self):
        with pytest.raises(ValueError, match="event_count must be at least 1"):
            sample_pose_motion(POSES, 10 * NS_PER_SECOND, event_count=0)

    def test_unknown_rest_asset_rejected(self):
        with pytest.raises(ValueError, match="rest_asset"):
            sample_pose_motion(POSES, 10 * NS_PER_SECOND, rest_asset="Nope")

    def test_rest_asset_selects_opening_pose(self):
        motion = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=2, rest_asset="Pose_3")
        assert motion.events[0].asset == "Pose_3"

    def test_empty_assets_rejected(self):
        with pytest.raises(ValueError, match="no pose assets"):
            sample_pose_motion([], 10 * NS_PER_SECOND)

    def test_non_positive_duration_rejected(self):
        with pytest.raises(ValueError, match="duration_ns must be positive"):
            sample_pose_motion(POSES, 0)

    def test_seed_is_recorded_when_not_given(self):
        motion = sample_pose_motion(POSES, 10 * NS_PER_SECOND)
        assert motion.seed is not None

    def test_omitting_the_seed_gives_a_different_timeline_each_time(self):
        # Otherwise re-running the same command to build up a dataset would just
        # write the same motion again.
        runs = [sample_pose_motion(POSES, 10 * NS_PER_SECOND) for _ in range(5)]
        assert len({m.seed for m in runs}) == 5

    def test_a_recorded_seed_reproduces_its_timeline(self):
        # This is what makes a random default safe: every file says how to
        # regenerate itself.
        original = sample_pose_motion(POSES, 10 * NS_PER_SECOND, event_count=6)
        again = sample_pose_motion(POSES, 10 * NS_PER_SECOND, event_count=6, seed=original.seed)
        assert again.to_dict() == original.to_dict()


class TestPoseMotionIO:
    def test_json_roundtrip(self, tmp_path):
        motion = sample_pose_motion(POSES, 40 * NS_PER_SECOND, seed=13, name="animation01")
        path = tmp_path / "animation01.json"
        motion.write(path)
        loaded = PoseMotion.from_path(path)
        assert loaded.to_dict() == motion.to_dict()
        assert loaded.source == path

    def test_from_dict_sorts_events(self):
        motion = PoseMotion.from_dict(
            {
                "name": "x",
                "duration_ns": 10 * NS_PER_SECOND,
                "events": [
                    {"t_ns": 5000, "asset": "b"},
                    {"t_ns": 1000, "asset": "a"},
                ],
            }
        )
        assert [e.t_ns for e in motion.events] == [1000, 5000]

    def test_assets_are_unique_in_first_use_order(self):
        motion = PoseMotion.from_dict(
            {
                "name": "x",
                "duration_ns": 10 * NS_PER_SECOND,
                "events": [
                    {"t_ns": 0, "asset": "a"},
                    {"t_ns": 1, "asset": "b"},
                    {"t_ns": 2, "asset": "a"},
                ],
            }
        )
        assert motion.assets() == ["a", "b"]

    def test_t_end_is_the_declared_duration(self):
        motion = PoseMotion.from_dict({"name": "x", "duration_ns": 1000, "events": [{"t_ns": 900, "asset": "w"}]})
        assert motion.t_end_ns == 1000

    def test_written_file_is_valid_json(self, tmp_path):
        motion = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=1)
        path = tmp_path / "a.json"
        motion.write(path)
        assert json.loads(path.read_text())["name"] == motion.name

    def test_summary_lists_every_event(self):
        motion = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=8)
        assert len(motion.summary().splitlines()) == len(motion.events)


class TestContinuousMotion:
    """No frame should ever be identical to its neighbour.

    Two sampler properties guarantee it: the timeline ends on an event rather
    than part way through, and no pose is ever drawn twice in a row.
    """

    @pytest.mark.parametrize("n", [2, 3, 8, 30])
    def test_last_event_lands_on_the_end(self, n):
        # Otherwise the final pose freezes for the remainder of the clip.
        motion = sample_pose_motion(POSES, 2 * NS_PER_SECOND, seed=n, event_count=n)
        assert motion.events[-1].t_ns == motion.duration_ns

    @pytest.mark.parametrize("n", [2, 5])
    def test_every_group_reaches_the_end(self, n):
        # Each region has its own timeline and each must span the whole clip.
        motion = sample_pose_motion(LAYERED, 3 * NS_PER_SECOND, seed=n, event_count=n, lanes=SPLIT_LANES)
        for group in ("left", "right", "head"):
            times = [e.t_ns for e in motion.events if e.group == group]
            assert min(times) == 0
            assert max(times) == motion.duration_ns

    def test_single_event_has_no_tail_to_pin(self):
        motion = sample_pose_motion(POSES, 2 * NS_PER_SECOND, seed=1, event_count=1)
        assert [e.t_ns for e in motion.events] == [0]

    @pytest.mark.parametrize("seed", range(25))
    def test_no_pose_repeats_back_to_back_within_a_group(self, seed):
        # Across groups a repeat is meaningless: they drive different bones.
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=seed, event_count=12, lanes=SPLIT_LANES)
        for group in ("left", "right", "head"):
            assets = [e.asset for e in motion.events if e.group == group]
            assert all(a != b for a, b in itertools.pairwise(assets)), (group, assets)

    def test_a_single_available_asset_still_terminates(self):
        # With nothing else to pick, repeats are unavoidable and must be allowed.
        only = [PoseAsset(name="Solo", frame=1, group="right")]
        motion = sample_pose_motion(only, NS_PER_SECOND, seed=1, event_count=4)
        assert motion.event_count == 4
        assert {e.asset for e in motion.events} == {"Solo"}

    def test_event_count_is_still_exact_with_both_rules(self):
        for n in (1, 2, 5, 40):
            m = sample_pose_motion(POSES, 3 * NS_PER_SECOND, seed=n, event_count=n)
            assert m.event_count == n

    def test_events_stay_sorted_after_merging_groups(self):
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=3, event_count=6, lanes=SPLIT_LANES)
        times = [e.t_ns for e in motion.events]
        assert times == sorted(times)
