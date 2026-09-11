# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for clips.py — planning and resume bookkeeping. No Blender required."""

from __future__ import annotations

import json
import random
from collections import Counter

import pytest

from phanesim.clips import (
    ACCESSORY_COUNT_WEIGHTS,
    ANIMATION_FILE,
    CALIBRATED_DISTORTION,
    CALIBRATED_LENS_SCALE,
    CAMERA_RANGES,
    HDRI_SPIN_STEP_DEG,
    SEQUENCE_FILE,
    choose_accessories,
    choose_hdri_spin,
    clip_dirs,
    fingerprint,
    is_done,
    lens_scale_for,
    mark_done,
    next_clip_index,
    randomize_camera,
    stray_clip_dirs,
)

CAMERA = {
    "name": "head0",
    "intrinsics": {"name": "pinhole", "parameters": {"fx": 240.0, "fy": 240.0, "cx": 320.0, "cy": 240.0}},
    "resolution": [640, 480],
    "pixel_format": "MONO8",
    "shutter": "global",
    "ca_factor": 0.3,
    "distortion": 0.387,
    "lens_scale": 1.2,
    "vignette_factor": 0.533,
    "vignette_feather": 0.4,
    "noise_std": 0.2,
    "chroma_noise": 0.0,
}


def make_clip(tmp_path, name="clip_00000", seq="{}", anim="{}"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / SEQUENCE_FILE).write_text(seq)
    (d / ANIMATION_FILE).write_text(anim)
    return d


class TestLensScale:
    def test_anchored_to_the_verified_pair(self):
        # 0.387 with 1.2 is the one combination that was checked by eye.
        assert lens_scale_for(CALIBRATED_DISTORTION) == pytest.approx(CALIBRATED_LENS_SCALE)

    def test_no_distortion_needs_no_crop(self):
        assert lens_scale_for(0.0) == pytest.approx(1.0)

    def test_more_distortion_needs_more_crop(self):
        lo, hi = CAMERA_RANGES["distortion"]
        assert lens_scale_for(lo) < lens_scale_for(hi)


class TestRandomizeCamera:
    def _draws(self, n=200):
        rng = random.Random(0)
        return [randomize_camera(CAMERA, rng) for _ in range(n)]

    def test_input_is_not_mutated(self):
        randomize_camera(CAMERA, random.Random(0))
        assert CAMERA["noise_std"] == 0.2
        assert CAMERA["intrinsics"]["parameters"]["fx"] == 240.0

    @pytest.mark.parametrize("field", ["noise_std", "vignette_factor", "distortion"])
    def test_stays_inside_its_range(self, field):
        lo, hi = CAMERA_RANGES[field]
        assert all(lo <= c[field] <= hi for c in self._draws())

    def test_fx_stays_inside_its_range(self):
        lo, hi = CAMERA_RANGES["fx"]
        assert all(lo <= c["intrinsics"]["parameters"]["fx"] <= hi for c in self._draws())

    def test_fx_and_fy_always_agree(self):
        # Only fx reaches the Blender lens, so an fy that differed would make the
        # projected ground truth disagree with the pixels.
        for c in self._draws():
            p = c["intrinsics"]["parameters"]
            assert p["fx"] == p["fy"]

    def test_principal_point_never_moves(self):
        # Blender renders with it at the image centre and applies no shift.
        for c in self._draws():
            p = c["intrinsics"]["parameters"]
            assert (p["cx"], p["cy"]) == (320.0, 240.0)

    def test_lens_scale_follows_distortion(self):
        for c in self._draws():
            assert c["lens_scale"] == pytest.approx(lens_scale_for(c["distortion"]), abs=1e-4)

    def test_resolution_and_format_are_untouched(self):
        for c in self._draws(20):
            assert c["resolution"] == [640, 480]
            assert c["pixel_format"] == "MONO8"

    def test_same_seed_gives_the_same_camera(self):
        a = randomize_camera(CAMERA, random.Random(7))
        b = randomize_camera(CAMERA, random.Random(7))
        assert a == b

    def test_different_draws_differ(self):
        rng = random.Random(3)
        assert randomize_camera(CAMERA, rng) != randomize_camera(CAMERA, rng)


class TestResumeBookkeeping:
    def test_a_fresh_clip_is_not_done(self, tmp_path):
        assert not is_done(make_clip(tmp_path))

    def test_marked_clip_is_done(self, tmp_path):
        d = make_clip(tmp_path)
        mark_done(d, 50)
        assert is_done(d)

    def test_editing_the_sequence_undoes_it(self, tmp_path):
        # The point of the fingerprint: "already rendered" must not be confused
        # with "rendered, but from different settings than are asked for now".
        d = make_clip(tmp_path)
        mark_done(d, 50)
        (d / SEQUENCE_FILE).write_text('{"changed": true}')
        assert not is_done(d)

    def test_editing_the_animation_undoes_it(self, tmp_path):
        d = make_clip(tmp_path)
        mark_done(d, 50)
        (d / ANIMATION_FILE).write_text('{"changed": true}')
        assert not is_done(d)

    def test_a_corrupt_done_file_is_treated_as_not_done(self, tmp_path):
        # Half-written by a process that was killed; redoing the clip is right.
        d = make_clip(tmp_path)
        mark_done(d, 50)
        (d / "_done.json").write_text("{not json")
        assert not is_done(d)

    def test_done_file_records_the_frame_count(self, tmp_path):
        d = make_clip(tmp_path)
        mark_done(d, 81)
        assert json.loads((d / "_done.json").read_text())["frames"] == 81

    def test_fingerprint_covers_both_files(self, tmp_path):
        a = make_clip(tmp_path, "a", seq='{"x": 1}', anim='{"y": 2}')
        b = make_clip(tmp_path, "b", seq='{"x": 1}', anim='{"y": 3}')
        assert fingerprint(a) != fingerprint(b)

    def test_identical_clips_share_a_fingerprint(self, tmp_path):
        a = make_clip(tmp_path, "a", seq='{"x": 1}', anim='{"y": 2}')
        b = make_clip(tmp_path, "b", seq='{"x": 1}', anim='{"y": 2}')
        assert fingerprint(a) == fingerprint(b)


class TestClipDirs:
    def test_lists_clips_in_order(self, tmp_path):
        for name in ("clip_00002", "clip_00000", "clip_00001"):
            make_clip(tmp_path, name)
        assert [d.name for d in clip_dirs(tmp_path)] == ["clip_00000", "clip_00001", "clip_00002"]

    def test_ignores_directories_without_a_sequence(self, tmp_path):
        make_clip(tmp_path, "clip_00000")
        (tmp_path / "notes").mkdir()
        assert [d.name for d in clip_dirs(tmp_path)] == ["clip_00000"]

    def test_empty_dataset_is_empty(self, tmp_path):
        assert clip_dirs(tmp_path) == []


class TestChooseAccessories:
    ALL = ["band1", "ring1", "ring2", "watch1"]

    def _draws(self, available, n=20000, seed=0):
        rng = random.Random(seed)
        return [choose_accessories(available, rng) for _ in range(n)]

    def test_a_model_without_accessories_wears_nothing(self):
        # model2 carries none, so a plan must not claim it wears a watch.
        assert choose_accessories([], random.Random(0)) == []

    def test_count_distribution_follows_the_weights(self):
        counts = Counter(len(d) for d in self._draws(self.ALL))
        for n, want in ACCESSORY_COUNT_WEIGHTS.items():
            assert counts[n] / 20000 == pytest.approx(want, abs=0.02)

    def test_bare_hands_are_at_least_half(self):
        # A network should learn the hand, not the jewellery.
        counts = Counter(len(d) for d in self._draws(self.ALL))
        assert counts[0] / 20000 >= 0.48

    def test_never_draws_more_than_the_model_has(self):
        for available in ([], ["watch1"], ["watch1", "ring1"]):
            assert all(len(d) <= len(available) for d in self._draws(available, n=500))

    def test_never_repeats_an_accessory(self):
        assert all(len(d) == len(set(d)) for d in self._draws(self.ALL, n=2000))

    def test_only_draws_what_is_available(self):
        for d in self._draws(["watch1", "ring1"], n=2000):
            assert set(d) <= {"watch1", "ring1"}

    def test_result_is_sorted(self):
        assert all(d == sorted(d) for d in self._draws(self.ALL, n=2000))

    def test_same_seed_gives_the_same_draw(self):
        assert self._draws(self.ALL, n=50, seed=4) == self._draws(self.ALL, n=50, seed=4)


class TestNextClipIndex:
    def test_empty_dataset_starts_at_zero(self, tmp_path):
        assert next_clip_index(tmp_path) == 0

    def test_continues_past_the_highest(self, tmp_path):
        for n in ("clip_00000", "clip_00001", "clip_00002"):
            make_clip(tmp_path, n)
        assert next_clip_index(tmp_path) == 3

    def test_uses_the_highest_not_the_count(self, tmp_path):
        # A dataset with a gap must not reuse a number that is already taken.
        make_clip(tmp_path, "clip_00000")
        make_clip(tmp_path, "clip_00007")
        assert next_clip_index(tmp_path) == 8

    def test_ignores_directories_that_are_not_clips(self, tmp_path):
        make_clip(tmp_path, "clip_00000")
        (tmp_path / "qa").mkdir()
        assert next_clip_index(tmp_path) == 1


class TestStrayClipDirs:
    def test_finds_a_clip_planned_one_level_too_deep(self, tmp_path):
        # The trap: --output dataset/clip_00002 buries clips where clip_dirs
        # cannot see them, and they would be skipped in silence.
        nested = tmp_path / "clip_00002"
        make_clip(nested, "clip_00000")
        assert [d.name for d in stray_clip_dirs(tmp_path)] == ["clip_00002"]

    def test_a_plain_directory_is_not_a_stray(self, tmp_path):
        make_clip(tmp_path, "clip_00000")
        (tmp_path / "notes").mkdir()
        assert stray_clip_dirs(tmp_path) == []

    def test_a_real_clip_is_not_a_stray(self, tmp_path):
        make_clip(tmp_path, "clip_00000")
        assert stray_clip_dirs(tmp_path) == []


class TestChooseHdriSpin:
    def _draws(self, n=5000, seed=0):
        rng = random.Random(seed)
        return [choose_hdri_spin(rng) for _ in range(n)]

    def test_start_covers_the_whole_panorama(self):
        # Turning about the vertical axis is a yaw, so every angle is upright and
        # there is no reason to leave part of the panorama unreachable.
        starts = [a for a, _ in self._draws()]
        assert min(starts) < 5.0
        assert max(starts) > 355.0
        assert all(0.0 <= a <= 360.0 for a in starts)

    def test_step_magnitude_stays_in_range(self):
        lo, hi = HDRI_SPIN_STEP_DEG
        assert all(lo <= abs(st) <= hi for _, st in self._draws())

    def test_step_goes_both_ways(self):
        signs = {st > 0 for _, st in self._draws(200)}
        assert signs == {True, False}

    def test_step_is_never_zero(self):
        # A zero step would leave the whole clip on one background slice.
        assert all(st != 0.0 for _, st in self._draws())

    def test_same_seed_reproduces(self):
        assert self._draws(20, seed=5) == self._draws(20, seed=5)
