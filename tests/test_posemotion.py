# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import itertools
import json
from collections import Counter

import pytest

from phanesim.posemotion import (
    HAND_POSE_SHARE,
    NS_PER_SECOND,
    POSE_ALPHA,
    TIME_JITTER,
    PoseAsset,
    PoseEvent,
    PoseMotion,
    PosePick,
    _draw_weight,
    sample_pose_motion,
)


def names(event) -> list[str]:
    """Asset names in one event's bundle."""
    return [p.asset for p in event.poses]


# Single-group library: the sampler then behaves as one flat timeline.
POSES = [PoseAsset(name=f"Pose_{i}", frame=1, group="right", part="arm") for i in range(5)]

# Multi-group library: 3 left x 4 right x 2 head, all touching disjoint bones.
# One timeline per body part, the arrangement --hand --head asks for.
SPLIT_LANES = (("left",), ("right",), ("head",))
# One shared timeline, the default: every event is whichever pose was drawn.
SHARED_LANE = (("left", "right", "body"),)

LAYERED = (
    [PoseAsset(name=f"L{i}", frame=1, group="left", part="arm") for i in range(3)]
    + [PoseAsset(name=f"R{i}", frame=1, group="right", part="arm") for i in range(4)]
    + [PoseAsset(name=f"H{i}", frame=1, group="head") for i in range(2)]
    + [PoseAsset(name=f"B{i}", frame=1, group="body") for i in range(2)]
)


class TestPoseAsset:
    def test_roundtrip(self):
        asset = PoseAsset(name="Right_ok", frame=1, group="right")
        assert PoseAsset.from_dict(asset.to_dict()) == asset

    def test_group_defaults_to_body(self):
        assert PoseAsset(name="Right_ok", frame=1).group == "body"


class TestPoseEvent:
    def test_roundtrip(self):
        event = PoseEvent(t_ns=1000, poses=[PosePick("Right_ok", 0.7)], group="right")
        assert PoseEvent.from_dict(event.to_dict()) == event

    def test_an_event_is_only_a_time_a_name_and_a_group(self):
        # An event is a bundle of picks, each with how far it is blended in.
        assert set(PoseEvent(t_ns=1000, poses=[PosePick("Right_ok")]).to_dict()) == {"t_ns", "group", "poses"}


class TestSamplePoseMotion:
    def test_seed_is_reproducible(self):
        a = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=7)
        b = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=7)
        assert a.to_dict() == b.to_dict()

    def test_different_seeds_differ(self):
        # The whole point of the generator: each run is a different timeline.
        a = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=1)
        b = sample_pose_motion(POSES, 60 * NS_PER_SECOND, seed=2)
        assert [names(e) for e in a.events] != [names(e) for e in b.events] or [e.t_ns for e in a.events] != [
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
        assert {e.group for e in motion.events} == {"left", "right", "body"}

    def test_lanes_with_no_assets_are_skipped(self):
        motion = sample_pose_motion(LAYERED, 5 * NS_PER_SECOND, seed=1, event_count=3, lanes=(("left",), ("nose",)))
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
        assert "Pose_3" in names(motion.events[0])

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
                    {"t_ns": 5000, "poses": [{"asset": "b"}]},
                    {"t_ns": 1000, "poses": [{"asset": "a"}]},
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
                    {"t_ns": 0, "poses": [{"asset": "a"}]},
                    {"t_ns": 1, "poses": [{"asset": "b"}]},
                    {"t_ns": 2, "poses": [{"asset": "a"}]},
                ],
            }
        )
        assert motion.assets() == ["a", "b"]

    def test_t_end_is_the_declared_duration(self):
        motion = PoseMotion.from_dict(
            {"name": "x", "duration_ns": 1000, "events": [{"t_ns": 900, "poses": [{"asset": "w"}]}]}
        )
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
            picked = [names(e) for e in motion.events if e.group == group]
            assert all(a != b for a, b in itertools.pairwise(picked)), (group, picked)

    def test_a_single_available_asset_still_terminates(self):
        # With nothing else to pick, repeats are unavoidable and must be allowed.
        only = [PoseAsset(name="Solo", frame=1, group="right", part="arm")]
        motion = sample_pose_motion(only, NS_PER_SECOND, seed=1, event_count=4)
        assert motion.event_count == 4
        assert {n for e in motion.events for n in names(e)} == {"Solo"}

    def test_event_count_is_still_exact_with_both_rules(self):
        for n in (1, 2, 5, 40):
            m = sample_pose_motion(POSES, 3 * NS_PER_SECOND, seed=n, event_count=n)
            assert m.event_count == n

    def test_events_stay_sorted_after_merging_groups(self):
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=3, event_count=6, lanes=SPLIT_LANES)
        times = [e.t_ns for e in motion.events]
        assert times == sorted(times)


class TestSharedLaneKeepsEveryLimbMoving:
    """A lane shared by both hands must still satisfy the rules per hand.

    Satisfying them only per lane leaves one hand frozen: two events can name the
    same left-hand pose with a right-hand event between them, and only one group
    can own the event pinned to the end of the clip.
    """

    SHARED = (("left", "right", "body"),)

    def _events(self, group, motion):
        return [e for e in motion.events if e.group == group]

    @pytest.mark.parametrize("seed", range(30))
    def test_no_limb_repeats_a_pose_back_to_back(self, seed):
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=seed, event_count=12, lanes=self.SHARED)
        for group in ("left", "right", "body"):
            picked = [names(e) for e in self._events(group, motion)]
            assert all(a != b for a, b in itertools.pairwise(picked)), (group, picked)

    @pytest.mark.parametrize("seed", range(30))
    def test_every_hand_reaches_the_end_of_the_clip(self, seed):
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=seed, event_count=12, lanes=self.SHARED)
        for group in ("left", "right"):
            events = self._events(group, motion)
            # A side whose only event is the opening one is left alone: moving it
            # would start the clip from rest with nothing posed.
            if len(events) > 1:
                assert events[-1].t_ns == motion.duration_ns, group

    def test_a_lone_opening_pose_is_left_at_zero(self):
        # Moving it would leave the clip starting from rest with nothing posed.
        only_left = [a for a in LAYERED if a.group == "left"][:1]
        motion = sample_pose_motion(only_left, 4 * NS_PER_SECOND, seed=1, event_count=1, lanes=(("left",),))
        assert [e.t_ns for e in motion.events] == [0]

    @pytest.mark.parametrize("n", [1, 2, 5, 12, 30])
    def test_event_count_is_unchanged_by_the_pinning(self, n):
        # The fix reassigns times; it must never add or drop an event.
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=n, event_count=n, lanes=self.SHARED)
        assert motion.event_count == n

    @pytest.mark.parametrize("seed", range(30))
    def test_pinning_never_moves_an_event_backwards(self, seed):
        # Times are reassigned, so the timeline must still run forwards and stay
        # inside the clip.
        motion = sample_pose_motion(LAYERED, 4 * NS_PER_SECOND, seed=seed, event_count=12, lanes=self.SHARED)
        times = [e.t_ns for e in motion.events]
        assert times == sorted(times)
        assert all(0 <= t <= motion.duration_ns for t in times)


class TestTimesAreJittered:
    """Gaps are bounded so no stretch of a short clip looks frozen."""

    POOL = [PoseAsset(name=f"P{i}", frame=1, group="right", part="arm") for i in range(12)]

    def _gaps(self, n, seed):
        m = sample_pose_motion(self.POOL, 10 * NS_PER_SECOND, seed=seed, event_count=n)
        t = [e.t_ns for e in m.events]
        return [(b - a) / m.duration_ns for a, b in itertools.pairwise(t)]

    @pytest.mark.parametrize("n", [4, 6, 10, 20])
    def test_no_gap_exceeds_the_jitter_bound(self, n):
        # An evenly spaced grid nudged by +-TIME_JITTER can stretch a gap to at
        # most (1 + 2*TIME_JITTER) spacings.
        bound = (1 + 2 * TIME_JITTER) / (n - 1) + 1e-6
        for seed in range(40):
            assert max(self._gaps(n, seed)) <= bound, (n, seed)

    @pytest.mark.parametrize("n", [4, 6, 10])
    def test_times_still_differ_between_seeds(self, n):
        # Bounded, not fixed: the grid is jittered, not snapped to.
        runs = {
            tuple(e.t_ns for e in sample_pose_motion(self.POOL, 10 * NS_PER_SECOND, seed=s, event_count=n).events)
            for s in range(20)
        }
        assert len(runs) == 20

    def test_event_count_is_unchanged(self):
        for n in (1, 2, 5, 30):
            assert sample_pose_motion(self.POOL, 10 * NS_PER_SECOND, seed=n, event_count=n).event_count == n

    def test_endpoints_are_still_pinned(self):
        for seed in range(20):
            m = sample_pose_motion(self.POOL, 10 * NS_PER_SECOND, seed=seed, event_count=6)
            assert m.events[0].t_ns == 0
            assert m.events[-1].t_ns == m.duration_ns


# A full per-joint library for one hand, as model1 now has.
LIMB = (
    [PoseAsset(name=f"Arm{i}", frame=1, group="left", part="arm") for i in range(4)]
    + [PoseAsset(name=f"Fore{i}", frame=1, group="left", part="forearm") for i in range(3)]
    + [PoseAsset(name=f"Wri{i}", frame=1, group="left", part="wrist") for i in range(3)]
    + [
        PoseAsset(name=f"{f}{i}", frame=1, group="left", part=f"finger:{f}")
        for f in ("thumb", "index", "middle", "ring", "pinky")
        for i in range(3)
    ]
)


class TestBundles:
    """One event sets a whole limb, because one joint alone barely moves it."""

    LANE = (("left",),)

    def _events(self, n=8, seed=0):
        m = sample_pose_motion(LIMB, 5 * NS_PER_SECOND, seed=seed, event_count=n, lanes=self.LANE)
        return m.events

    def _parts(self, event):
        by_name = {a.name: a for a in LIMB}
        return [by_name[p.asset].part for p in event.poses]

    def test_event_count_still_means_events(self):
        # A bundle is one event, however many poses it holds.
        for n in (1, 4, 8, 20):
            assert len(self._events(n)) == n

    def test_every_event_sets_arm_forearm_and_wrist(self):
        # The arm dominates what the camera sees; an event that left it alone
        # would barely change the picture.
        for e in self._events(30):
            parts = self._parts(e)
            assert {"arm", "forearm", "wrist"} <= set(parts)

    def test_every_finger_count_from_one_to_five_occurs(self):
        counts = Counter()
        for seed in range(60):
            for e in self._events(10, seed=seed):
                counts[sum(1 for p in self._parts(e) if p.startswith("finger:"))] += 1
        assert set(counts) == {1, 2, 3, 4, 5}

    def test_two_and_three_fingers_are_the_common_case(self):
        counts = Counter()
        for seed in range(60):
            for e in self._events(10, seed=seed):
                counts[sum(1 for p in self._parts(e) if p.startswith("finger:"))] += 1
        total = sum(counts.values())
        for rare in (1, 4, 5):
            assert counts[2] / total > counts[rare] / total
            assert counts[3] / total > counts[rare] / total

    def test_a_part_is_never_set_twice_in_one_event(self):
        for e in self._events(30):
            parts = self._parts(e)
            assert len(parts) == len(set(parts))

    def test_alphas_stay_in_range(self):
        lo, hi = POSE_ALPHA
        for e in self._events(30):
            assert all(lo <= p.alpha <= hi for p in e.poses)

    def test_a_part_never_repeats_its_pose_back_to_back(self):
        # A part keyed to the pose it already holds is a change that changes
        # nothing, which freezes it until its next event.
        by_name = {a.name: a for a in LIMB}
        last: dict[str, str] = {}
        for e in self._events(40, seed=3):
            for p in e.poses:
                part = by_name[p.asset].part
                assert last.get(part) != p.asset, part
                last[part] = p.asset

    def test_a_lane_without_limb_parts_draws_one_pose(self):
        # The head has no arm to bundle with, so it stays one pose per event.
        heads = [PoseAsset(name=f"H{i}", frame=1, group="head") for i in range(4)]
        m = sample_pose_motion(heads, 5 * NS_PER_SECOND, seed=1, event_count=6, lanes=(("head",),))
        assert all(len(e.poses) == 1 for e in m.events)


class TestArmWeighting:
    """The side raise is drawn less: it puts the hand outside the camera's view."""

    ARMS = [
        PoseAsset(name=f"Arm_left_x{x}_y{y}", frame=1, group="left", part="arm")
        for x in (1, 2, 3, 4)
        for y in (1, 2, 3)
    ]

    def test_weight_follows_the_name_token(self):
        for name, want in (("Arm_left_x1_y2", 0.5), ("Arm_left_x2_y1", 1.5), ("Arm_left_x4_y3", 1.0)):
            assert _draw_weight(PoseAsset(name, 1, "left", "arm")) == want

    def test_only_arms_are_weighted(self):
        # A wrist pose named x1 is a wrist angle, not the side raise.
        assert _draw_weight(PoseAsset("Wrist_left_x1", 1, "left", "wrist")) == 1.0

    def test_an_unknown_name_gets_the_default(self):
        assert _draw_weight(PoseAsset("Arm_left_reach", 1, "left", "arm")) == 1.0

    def test_x2_and_x3_are_drawn_more_than_x1(self):
        seen = Counter()
        for seed in range(80):
            m = sample_pose_motion(self.ARMS, 5 * NS_PER_SECOND, seed=seed, event_count=10, lanes=(("left",),))
            for e in m.events:
                for p in e.poses:
                    seen[p.asset.split("_")[2][:2]] += 1
        assert seen["x2"] > seen["x1"]
        assert seen["x3"] > seen["x1"]
        # Three x-columns share each weight, so the ratio should track it.
        assert seen["x2"] / seen["x1"] == pytest.approx(1.5 / 0.5, rel=0.25)


class TestHeadKeepsPace:
    def test_head_gets_as_many_events_as_each_hand(self):
        # --hand N --head must move the head N times too, not fewer.
        pool = LIMB + [PoseAsset(name=f"H{i}", frame=1, group="head") for i in range(6)]
        lanes = (("left",), ("head",))
        m = sample_pose_motion(pool, 5 * NS_PER_SECOND, seed=1, event_count=10, lanes=lanes)
        assert sum(1 for e in m.events if e.group == "left") == 10
        assert sum(1 for e in m.events if e.group == "head") == 10


class TestHandSlot:
    """The hand is set either by single fingers or by one whole-hand pose."""

    WITH_HAND = LIMB + [PoseAsset(name=f"Fist{i}", frame=1, group="left", part="hand") for i in range(4)]

    def _events(self, pool, n=10, seed=0):
        m = sample_pose_motion(pool, 5 * NS_PER_SECOND, seed=seed, event_count=n, lanes=(("left",),))
        return m.events

    def _parts(self, pool, event):
        by_name = {a.name: a for a in pool}
        return [by_name[p.asset].part for p in event.poses]

    def test_never_both_in_one_event(self):
        # They fill the same slot: a whole-hand pose already sets every finger.
        for seed in range(40):
            for e in self._events(self.WITH_HAND, seed=seed):
                parts = self._parts(self.WITH_HAND, e)
                assert not ("hand" in parts and any(p.startswith("finger:") for p in parts))

    def test_both_kinds_are_drawn_about_equally(self):
        hands = fingers = 0
        for seed in range(60):
            for e in self._events(self.WITH_HAND, seed=seed):
                parts = self._parts(self.WITH_HAND, e)
                hands += "hand" in parts
                fingers += any(p.startswith("finger:") for p in parts)
        assert hands / (hands + fingers) == pytest.approx(HAND_POSE_SHARE, abs=0.06)

    def test_a_library_without_whole_hand_poses_always_uses_fingers(self):
        for e in self._events(LIMB, seed=1):
            assert any(p.startswith("finger:") for p in self._parts(LIMB, e))

    def test_a_library_with_only_whole_hand_poses_always_uses_them(self):
        pool = [a for a in self.WITH_HAND if not a.part.startswith("finger:")]
        for e in self._events(pool, seed=1):
            assert "hand" in self._parts(pool, e)

    def test_the_arm_is_still_always_set(self):
        for seed in range(20):
            for e in self._events(self.WITH_HAND, seed=seed):
                assert "arm" in self._parts(self.WITH_HAND, e)
