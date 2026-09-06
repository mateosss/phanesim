# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import math

from test_visibility import JOINTS, write_clip

from phanesim.debug import (
    DEFAULT_BOX_MARGIN,
    HAND_RECT_FILE,
    LANDMARKS_PER_HAND,
    PRESENCE_MIN_LANDMARKS,
    SIDES,
    hand_rect,
    hand_rect_columns,
    read_hand_rects,
    read_joints_2d,
    resolution_for,
    write_hand_rects,
)


def hand(points):
    """Pad *points* out to a full 21-landmark hand with NaN (None) landmarks."""
    return list(points) + [None] * (LANDMARKS_PER_HAND - len(points))


def in_view(n, x=100.0, y=100.0):
    """*n* landmarks inside a 640x480 frame, spread so their hull has real size."""
    return [(x + i, y + i) for i in range(n)]


def square(x0, y0, x1, y1, n=LANDMARKS_PER_HAND):
    """*n* landmarks whose hull is exactly the box (x0, y0)..(x1, y1)."""
    return hand([(x0, y0), (x1, y1)] + [((x0 + x1) / 2, (y0 + y1) / 2)] * (n - 2))


def rects_for(tmp_path, frames, resolution=(640, 480), margin=0.0):
    """Write a clip, derive hand_rect.csv from it, read it back."""
    clip = write_clip(tmp_path, frames, resolution=resolution)
    cam = clip / "cam_head0"
    write_hand_rects(cam, resolution=resolution, margin=margin)
    return read_hand_rects(cam / HAND_RECT_FILE)


class TestBoundingBox:
    def test_the_box_spans_the_landmarks(self):
        assert hand_rect(square(100, 50, 300, 250), 640, 480, margin=0.0) == (100.0, 50.0, 200.0, 200.0)

    def test_the_margin_grows_the_box_on_every_side(self):
        # 10% of the 200-long side of a 200x100 hull is 20, on all four edges.
        assert hand_rect(square(100, 100, 300, 200), 640, 480, margin=0.1) == (80.0, 80.0, 240.0, 140.0)

    def test_the_pad_is_the_same_on_both_axes(self):
        # The per-axis version this replaced padded a lopsided hull 200/50 = 4x
        # more across the fingers than along them, and bent-down hands spilled
        # out of the bottom of their own box.
        box = hand_rect(square(100, 100, 300, 150), 640, 480, margin=0.1)
        assert box is not None
        assert box[2] - 200.0 == box[3] - 50.0 == 40.0

    def test_it_is_never_tighter_than_padding_each_axis_by_itself(self):
        for hull in ((100, 100, 300, 150), (100, 100, 150, 400), (100, 100, 300, 300)):
            box = hand_rect(square(*hull), 640, 480, margin=0.1)
            assert box is not None
            assert box[2] >= (hull[2] - hull[0]) * 1.2
            assert box[3] >= (hull[3] - hull[1]) * 1.2

    def test_nan_landmarks_do_not_drag_the_box(self):
        # A landmark the renderer could not project is absent, not at (0, 0).
        points = hand([(100, 100), (300, 200)] + [None] * 4 + [(200, 150)] * 15)
        assert hand_rect(points, 640, 480, margin=0.0) == (100.0, 100.0, 200.0, 100.0)


class TestClippedToTheImage:
    """A hand cut by the frame edge is what the detector has to find anyway."""

    def test_the_box_stops_at_the_image_border(self):
        # The hull runs off three sides; the box is what is actually visible.
        box = hand_rect(square(-100, -50, 700, 300), 640, 480, margin=0.0)
        assert box == (0.0, 0.0, 640.0, 300.0)

    def test_landmarks_off_the_edge_still_set_the_extent(self):
        # 5 landmarks in frame, the rest off to the right: present, and the box
        # runs to the right border rather than stopping at the last visible dot.
        points = hand([(600, 100), (600, 300)] + [(600, 200)] * 3 + [(900, 200)] * 16)
        box = hand_rect(points, 640, 480, margin=0.0)
        assert box is not None and box[0] + box[2] == 640.0

    def test_a_hand_wholly_outside_the_image_has_no_box(self):
        assert hand_rect(square(700, 100, 900, 300), 640, 480, margin=0.0) is None

    def test_a_box_with_no_area_left_is_no_box(self):
        # Every landmark on one row -- impossible for a real hand, but a zero-height
        # rectangle is not something a detector could be trained or scored against.
        assert hand_rect(hand([(100, 100)] * LANDMARKS_PER_HAND), 640, 480) is None


class TestPresence:
    def test_a_hand_with_too_few_landmarks_in_frame_is_absent(self):
        in_frame = PRESENCE_MIN_LANDMARKS - 1
        points = hand(in_view(in_frame) + [(5000, 200)] * (LANDMARKS_PER_HAND - in_frame))
        assert hand_rect(points, 640, 480, margin=0.0) is None

    def test_the_threshold_itself_counts_as_present(self):
        points = hand(in_view(PRESENCE_MIN_LANDMARKS) + [(5000, 200)] * 16)
        assert hand_rect(points, 640, 480, margin=0.0) is not None

    def test_a_hand_with_no_landmarks_at_all_is_absent(self):
        assert hand_rect(hand([]), 640, 480, margin=0.0) is None


class TestHandRectCsv:
    def test_both_sides_get_five_columns_each(self):
        assert hand_rect_columns() == [f"{s}_{n}" for s in SIDES for n in ("present", "x", "y", "w", "h")]

    def test_a_present_hand_is_written_with_its_box(self, tmp_path):
        frames = [{"right": square(100, 50, 300, 250), "left": square(400, 50, 500, 250)}]
        rects = rects_for(tmp_path, frames)
        assert rects[0]["right"] == (100.0, 50.0, 200.0, 200.0)
        assert rects[0]["left"] == (400.0, 50.0, 100.0, 200.0)

    def test_an_absent_hand_is_written_as_nan_not_zero(self, tmp_path):
        # Four zeros would be a valid-looking box in the corner, so a consumer
        # that forgot to read the present flag would train on it in silence.
        clip = write_clip(tmp_path, [{"right": square(100, 50, 300, 250), "left": hand([])}])
        write_hand_rects(clip / "cam_head0", resolution=(640, 480))
        with (clip / "cam_head0" / HAND_RECT_FILE).open(newline="") as f:
            row = next(iter(csv.DictReader(f)))
        assert row["left_present"] == "0"
        assert all(math.isnan(float(row[f"left_{a}"])) for a in "xywh")

    def test_one_row_per_frame_keeping_the_timestamps(self, tmp_path):
        frames = [{"right": square(100, 50, 300, 250), "left": hand([])} for _ in range(4)]
        clip = write_clip(tmp_path, frames)
        write_hand_rects(clip / "cam_head0", resolution=(640, 480))
        with (clip / "cam_head0" / HAND_RECT_FILE).open(newline="") as f:
            rows = list(csv.DictReader(f))
        assert [r["timestamp"] for r in rows] == ["0", "1", "2", "3"]

    def test_the_default_margin_is_applied_when_none_is_asked_for(self, tmp_path):
        clip = write_clip(tmp_path, [{"right": square(100, 100, 300, 300), "left": hand([])}])
        write_hand_rects(clip / "cam_head0", resolution=(640, 480))
        rects = read_hand_rects(clip / "cam_head0" / HAND_RECT_FILE)
        box = rects[0]["right"]
        assert box is not None and box[2] == 200.0 * (1 + 2 * DEFAULT_BOX_MARGIN)

    def test_nothing_is_written_where_there_is_no_joints_2d(self, tmp_path):
        assert write_hand_rects(tmp_path) is None
        assert not (tmp_path / HAND_RECT_FILE).exists()


class TestReadJoints2d:
    def test_sides_are_grouped_by_their_column_prefix(self, tmp_path):
        clip = write_clip(tmp_path, [{"right": square(100, 50, 300, 250), "left": hand([])}])
        _, landmarks = read_joints_2d(clip / "cam_head0" / "joints_2d.csv")
        assert set(landmarks) == {"left", "right"}
        assert len(landmarks["right"][0]) == LANDMARKS_PER_HAND

    def test_nan_becomes_none(self, tmp_path):
        clip = write_clip(tmp_path, [{"right": hand([(1.0, 2.0)]), "left": hand([])}])
        _, landmarks = read_joints_2d(clip / "cam_head0" / "joints_2d.csv")
        assert landmarks["right"][0][0] == (1.0, 2.0)
        assert landmarks["right"][0][1] is None
        assert landmarks["left"][0] == [None] * LANDMARKS_PER_HAND

    def test_a_header_only_file_reads_as_no_frames(self, tmp_path):
        columns = ["timestamp"] + [f"{s}_{j}_{a}" for s in SIDES for j in JOINTS for a in ("u", "v")]
        (tmp_path / "joints_2d.csv").write_text(",".join(columns) + "\n")
        assert read_joints_2d(tmp_path / "joints_2d.csv") == ([], {})


class TestResolution:
    def test_falls_back_to_the_clips_sequence_json(self, tmp_path):
        # No frames rendered yet, so the declared resolution is all there is.
        clip = write_clip(tmp_path, [{"right": hand([]), "left": hand([])}], resolution=(320, 240))
        assert resolution_for(clip / "cam_head0") == (320, 240)

    def test_a_rendered_frame_wins_over_the_declaration(self, tmp_path):
        from PIL import Image

        clip = write_clip(tmp_path, [{"right": hand([]), "left": hand([])}], resolution=(320, 240))
        Image.new("RGB", (640, 480)).save(clip / "cam_head0" / "frame_000000.png")
        assert resolution_for(clip / "cam_head0") == (640, 480)

    def test_the_box_is_clipped_against_the_resolution_that_was_found(self, tmp_path):
        clip = tmp_path / "clip_00000"
        cam = clip / "cam_head0"
        cam.mkdir(parents=True)
        (clip / "sequence.json").write_text(json.dumps({"body_rig": {"cameras": [{"resolution": [320, 240]}]}}))
        columns = ["timestamp"] + [f"{s}_{j}_{a}" for s in ("right", "left") for j in JOINTS for a in ("u", "v")]
        row = ["0"] + [str(v) for _ in range(2) for i, _ in enumerate(JOINTS) for v in (100.0 + i, 50.0 + i)]
        (cam / "joints_2d.csv").write_text(",".join(columns) + "\n" + ",".join(row) + "\n")
        write_hand_rects(cam, margin=0.0)
        assert read_hand_rects(cam / HAND_RECT_FILE)[0]["right"] is not None
