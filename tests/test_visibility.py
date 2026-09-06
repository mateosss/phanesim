# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import json

import pytest

from phanesim.visibility import (
    DEFAULT_RESOLUTION,
    LANDMARKS_PER_HAND,
    THRESHOLDS,
    measure,
    report,
    resolution_for,
)

JOINTS = [
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


def write_clip(root, frames, resolution=(640, 480)):
    """Write a clip whose joints_2d.csv holds *frames*.

    Each frame is {side: [(u, v), ...]}; a landmark may be None for NaN, which is
    what the renderer writes for a point behind the camera.
    """
    clip = root / "clip_00000"
    cam = clip / "cam_head0"
    cam.mkdir(parents=True)
    (clip / "sequence.json").write_text(json.dumps({"body_rig": {"cameras": [{"resolution": list(resolution)}]}}))
    columns = ["timestamp"] + [f"{s}_{j}_{a}" for s in ("right", "left") for j in JOINTS for a in ("u", "v")]
    lines = [",".join(columns)]
    for i, frame in enumerate(frames):
        row = [str(i)]
        for side in ("right", "left"):
            for point in frame[side]:
                row += ["", ""] if point is None else [str(point[0]), str(point[1])]
        lines.append(",".join(row))
    (cam / "joints_2d.csv").write_text("\n".join(lines) + "\n")
    return clip


def hand(u=100.0, v=100.0, n=LANDMARKS_PER_HAND, missing=None):
    """One hand: *n* landmarks stacked at (u, v), the rest placed at *missing*."""
    return [(u, v)] * n + [missing] * (LANDMARKS_PER_HAND - n)


class TestOutOfFrameIsNotNaN:
    """The subtlety the whole module exists for."""

    def test_a_hand_outside_the_image_is_counted_out(self, tmp_path):
        # joints_2d writes NaN only for a landmark behind the camera or past the
        # lens fold.  A hand merely off to the side still carries real, large
        # pixel coordinates, and counting non-NaN as visible would call it seen.
        write_clip(tmp_path, [{"right": hand(u=5000.0), "left": hand(u=-900.0)}])
        tallies, _ = measure(tmp_path)
        assert tallies["any"].share(0) == 1.0

    def test_nan_landmarks_are_counted_out_too(self, tmp_path):
        write_clip(tmp_path, [{"right": hand(n=0), "left": hand(n=0)}])
        tallies, _ = measure(tmp_path)
        assert tallies["any"].share(0) == 1.0

    def test_a_hand_inside_the_image_is_counted_in(self, tmp_path):
        write_clip(tmp_path, [{"right": hand(), "left": hand()}])
        tallies, _ = measure(tmp_path)
        assert tallies["whole"].share(2) == 1.0


class TestThresholdsDisagree:
    """Reporting one number hid that most 'visible' hands were cut by the edge."""

    def test_a_hand_at_the_frame_edge_counts_loosely_but_not_strictly(self, tmp_path):
        # Six landmarks in frame, the other fifteen off the right-hand side.
        clipped = hand(n=6)[:6] + [(5000.0, 100.0)] * (LANDMARKS_PER_HAND - 6)
        write_clip(tmp_path, [{"right": clipped, "left": hand(n=0)}])
        tallies, _ = measure(tmp_path)
        assert tallies["any"].share(1) == 1.0
        assert tallies["half"].share(0) == 1.0
        assert tallies["whole"].share(0) == 1.0

    def test_every_threshold_sees_the_same_frames(self, tmp_path):
        write_clip(tmp_path, [{"right": hand(), "left": hand(n=0)} for _ in range(4)])
        tallies, _ = measure(tmp_path)
        assert {t.total for t in tallies.values()} == {4}

    def test_stricter_thresholds_never_report_more_hands(self, tmp_path):
        frames = [{"right": hand(n=n), "left": hand(n=LANDMARKS_PER_HAND - n)} for n in range(0, 22, 3)]
        write_clip(tmp_path, frames)
        tallies, _ = measure(tmp_path)
        per_frame = [tallies[label].hands_per_frame for label, _ in THRESHOLDS]
        assert per_frame == sorted(per_frame, reverse=True)


class TestResolution:
    def test_read_from_the_clips_sequence_json(self, tmp_path):
        clip = write_clip(tmp_path, [{"right": hand(), "left": hand()}], resolution=(320, 240))
        assert resolution_for(clip / "cam_head0" / "joints_2d.csv") == (320, 240)

    def test_the_frame_size_decides_what_is_out(self, tmp_path):
        # The same landmarks, one resolution that holds them and one that does not.
        big = write_clip(tmp_path / "big", [{"right": hand(u=300.0), "left": hand(n=0)}], (640, 480))
        small = write_clip(tmp_path / "small", [{"right": hand(u=300.0), "left": hand(n=0)}], (160, 120))
        assert measure(big.parent)[0]["whole"].share(1) == 1.0
        assert measure(small.parent)[0]["any"].share(0) == 1.0

    def test_falls_back_when_the_csv_was_moved_from_its_clip(self, tmp_path):
        loose = tmp_path / "cam_head0"
        loose.mkdir(parents=True)
        (loose / "joints_2d.csv").write_text("timestamp\n")
        assert resolution_for(loose / "joints_2d.csv") == DEFAULT_RESOLUTION


class TestMeasure:
    def test_no_csv_anywhere_is_an_error_not_an_empty_report(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="no joints_2d.csv"):
            measure(tmp_path)

    def test_a_whole_dataset_is_summed_over_its_clips(self, tmp_path):
        for i in range(3):
            root = tmp_path / f"d{i}"
            write_clip(root, [{"right": hand(), "left": hand()} for _ in range(2)])
        tallies, per_clip = measure(tmp_path)
        assert tallies["any"].total == 6
        assert len(per_clip) == 3

    def test_report_names_the_emptiest_clips(self, tmp_path):
        write_clip(tmp_path / "good", [{"right": hand(), "left": hand()}])
        write_clip(tmp_path / "bad", [{"right": hand(n=0), "left": hand(n=0)}])
        text = report(tmp_path)
        assert "emptiest clips" in text
        assert "1/1 frames" in text
