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

POSES = [PoseAsset(name=f"Pose_{i}", frame_start=1, frame_end=1) for i in range(5)]
WAVE = PoseAsset(name="Wave_Animation", frame_start=1, frame_end=41, fps=24)


class TestPoseAsset:
    def test_single_frame_is_a_pose(self):
        asset = PoseAsset(name="Right_ok", frame_start=1, frame_end=1)
        assert not asset.is_action
        assert asset.kind == "pose"
        assert asset.duration_ns == 0

    def test_multi_frame_is_an_action(self):
        assert WAVE.is_action
        assert WAVE.kind == "action"
        # 40 frames at 24 fps
        assert WAVE.duration_ns == pytest.approx(40 / 24 * NS_PER_SECOND, rel=1e-6)

    def test_roundtrip(self):
        assert PoseAsset.from_dict(WAVE.to_dict()) == WAVE


class TestPoseEvent:
    def test_pose_event_omits_action_fields(self):
        data = PoseEvent(t_ns=1000, asset="Right_ok").to_dict()
        assert "duration_ns" not in data
        assert "source_frames" not in data

    def test_action_event_keeps_action_fields(self):
        event = PoseEvent(t_ns=1000, asset="Wave_Animation", kind="action", duration_ns=500, source_frames=(1, 41))
        data = event.to_dict()
        assert data["duration_ns"] == 500
        assert data["source_frames"] == [1, 41]
        assert PoseEvent.from_dict(data) == event

    def test_end_ns_includes_playback(self):
        assert PoseEvent(t_ns=100, asset="a", kind="action", duration_ns=50).end_ns == 150
        assert PoseEvent(t_ns=100, asset="a").end_ns == 100


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
        motion = sample_pose_motion(POSES + [WAVE], 60 * NS_PER_SECOND, seed=3)
        times = [e.t_ns for e in motion.events]
        assert times == sorted(times)
        assert all(t <= motion.duration_ns for t in times)

    def test_starts_at_zero(self):
        # Without a key at t=0 the first frames would open mid-interpolation.
        motion = sample_pose_motion(POSES, 30 * NS_PER_SECOND, seed=5)
        assert motion.events[0].t_ns == 0
        assert motion.events[0].blend_ns == 0

    def test_min_gap_is_respected(self):
        min_gap = 2 * NS_PER_SECOND
        motion = sample_pose_motion(POSES, 120 * NS_PER_SECOND, seed=11, min_gap_ns=min_gap, events_per_second=50.0)
        # A high rate would otherwise pack events far tighter than min_gap.
        gaps = [b.t_ns - a.end_ns for a, b in itertools.pairwise(motion.events)]
        assert all(g >= min_gap for g in gaps), gaps

    def test_higher_rate_gives_more_events(self):
        slow = sample_pose_motion(POSES, 300 * NS_PER_SECOND, seed=9, events_per_second=0.1)
        fast = sample_pose_motion(POSES, 300 * NS_PER_SECOND, seed=9, events_per_second=0.6)
        assert len(fast.events) > len(slow.events)

    def test_action_duration_occupies_timeline(self):
        motion = sample_pose_motion([WAVE], 60 * NS_PER_SECOND, seed=4, min_gap_ns=NS_PER_SECOND)
        for prev, nxt in itertools.pairwise(motion.events):
            assert nxt.t_ns >= prev.end_ns

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


class TestPoseMotionIO:
    def test_json_roundtrip(self, tmp_path):
        motion = sample_pose_motion(POSES + [WAVE], 40 * NS_PER_SECOND, seed=13, name="animation01")
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

    def test_t_end_covers_trailing_action(self):
        motion = PoseMotion.from_dict(
            {
                "name": "x",
                "duration_ns": 1000,
                "events": [{"t_ns": 900, "asset": "w", "kind": "action", "duration_ns": 500}],
            }
        )
        # The action runs past duration_ns, so the timeline must extend to cover it.
        assert motion.t_end_ns == 1400

    def test_written_file_is_valid_json(self, tmp_path):
        motion = sample_pose_motion(POSES, 20 * NS_PER_SECOND, seed=1)
        path = tmp_path / "a.json"
        motion.write(path)
        assert json.loads(path.read_text())["name"] == motion.name

    def test_summary_lists_every_event(self):
        motion = sample_pose_motion(POSES + [WAVE], 60 * NS_PER_SECOND, seed=8)
        assert len(motion.summary().splitlines()) == len(motion.events)
