# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Hand bounding boxes, and the debug pictures that show them.

Two things live here, because the second exists to check the first.

``hand_rect.csv`` is ground truth for a hand *detector*: per frame, per side,
whether the hand is in the picture and the axis-aligned rectangle it occupies.
The usual pipeline detects a hand and its box first and only then runs keypoint
estimation on the crop, so the box has to be a plain upright rectangle even
though the lens distortion means the hand's real outline is curved.  It is
derived from ``joints_2d.csv`` -- the landmarks are already distorted, so their
hull is already in rendered-pixel space -- which is why nothing here needs
Blender.

The debug overlay then draws those rectangles, plus the landmarks themselves,
onto copies of the rendered frames.  It is only ever a way to look at the CSVs
we export; the drawings are never part of a dataset, because keypoints painted
onto the pixels would be learned as features.

Kept import-light on purpose: ``render.py`` calls into this from inside
Blender, whose interpreter has no Pillow, so the drawing half imports PIL
lazily and everything at module level is stdlib.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from pathlib import Path

from phanesim.skeleton import HAND_CONNECTIONS, LANDMARK_COLORS

JOINTS_2D_FILE = "joints_2d.csv"
HAND_RECT_FILE = "hand_rect.csv"

# The two sides hand_rect.csv always carries, in the order its columns run.
# Fixed rather than taken from joints_2d.csv so every dataset's file has the
# same columns in the same places, whatever order a rig lists its hands in.
SIDES: tuple[str, ...] = ("left", "right")

LANDMARKS_PER_HAND = 21

# How far the box is grown past the landmark hull, as a fraction of the hull's
# *longer* side, added equally to all four edges.  The 21 landmarks are joint
# centres, so their hull runs *inside* the hand: the palm's outer edge and the
# flesh around the fingertips fall outside it, and a detector is normally
# annotated against the silhouette instead.
#
# The one thing worth knowing here is why the padding is not a fraction of each
# axis separately, which is the obvious way to write it.  What has to be covered
# is roughly the thickness of the flesh, and that is the same number of pixels
# whichever way the hand is turned -- but a hand seen edge-on gives a hull as
# lopsided as 3.75:1 (measured on dataset_test), so a per-axis fraction pads the
# long side generously and the short side barely at all.  On a 218x85 hull, 10%
# per axis is 22 px of headroom across the fingers and 8 px along them, and the
# bent-down poses spill out of the bottom of their own box.  Sizing the pad off
# the longer side instead is one number for both axes, and is never smaller than
# the per-axis version on either.
DEFAULT_BOX_MARGIN = 0.12

DEFAULT_RESOLUTION = (640, 480)

# Box colors for the overlay, per side.  Deliberately not any of
# LANDMARK_COLORS, so the rectangle never reads as another finger.
BOX_COLORS: dict[str, tuple[int, int, int]] = {
    "left": (0, 255, 255),  # cyan
    "right": (255, 255, 0),  # yellow
}

# Pixels are rounded to this many decimals in the CSV.  Sub-hundredth-pixel
# precision on a box grown by a 10% margin would be false precision.
_PIXEL_DECIMALS = 2

type Landmark = tuple[float, float] | None
type Rect = tuple[float, float, float, float]  # x, y, w, h -- top-left and size


def hand_rect_columns() -> list[str]:
    """Header of hand_rect.csv, without the leading timestamp column."""
    return [f"{side}_{name}" for side in SIDES for name in ("present", "x", "y", "w", "h")]


def read_joints_2d(csv_path: Path) -> tuple[list[str], dict[str, list[list[Landmark]]]]:
    """Read one joints_2d.csv into per-side landmark pixels.

    Args:
        csv_path: Path to a joints_2d.csv.

    Returns:
        The timestamp column as written, and a mapping side -> one list of 21
        landmarks per frame, each either an (u, v) pair or None where the CSV
        holds NaN (a landmark behind the camera, or past the fold of the lens
        distortion).  Sides absent from the file are absent from the mapping.
    """
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], {}

    # Columns run timestamp then {side}_{landmark}_u, {side}_{landmark}_v, in
    # the rig's own hand order; group them by the side prefix rather than
    # assuming one, since which hands a body exports is configurable.
    columns = [c for c in rows[0] if c != "timestamp"]
    per_side: dict[str, list[str]] = {}
    for column in columns:
        if not column.endswith("_u"):
            continue
        per_side.setdefault(column.split("_", 1)[0], []).append(column)

    timestamps = [row["timestamp"] for row in rows]
    landmarks: dict[str, list[list[Landmark]]] = {}
    for side, u_columns in per_side.items():
        frames: list[list[Landmark]] = []
        for row in rows:
            frame: list[Landmark] = []
            for u_column in u_columns:
                try:
                    u, v = float(row[u_column]), float(row[f"{u_column[:-2]}_v"])
                except (TypeError, ValueError, KeyError):
                    frame.append(None)
                    continue
                frame.append(None if math.isnan(u) or math.isnan(v) else (u, v))
            frames.append(frame)
        landmarks[side] = frames
    return timestamps, landmarks


def hand_rect(
    landmarks: Sequence[Landmark],
    width: int,
    height: int,
    margin: float = DEFAULT_BOX_MARGIN,
) -> Rect | None:
    """The bounding box of one hand in one frame, or None if it is not in shot.

    The hull is taken over every landmark that has pixel coordinates at all,
    including ones off the edge, because those still say how far the hand
    reaches; it is grown by the margin, and only then clipped to the image.

    Presence falls out of that geometry: the hand is in shot when the grown box
    still has area after clipping.  It is deliberately not a count of landmarks
    inside the frame, which is the test this used to apply and which dropped
    hands the picture plainly showed -- a hand entering at a corner puts a
    couple of fingertips in frame and its remaining joints outside, so any
    threshold above about two threw away a box a detector has to predict.  On
    dataset_test1 and dataset_test2 that was 29 of 280 hand-frames.  Counting
    landmarks is still the right question for "how much hand does this render
    show", and visibility.py goes on asking it.

    The margin is applied before the clip, so it can bring a hand just off the
    edge into shot.  That is intended: the box is the annotation, and if the
    annotation overlaps the image the hand is in the picture.  It does mean
    *margin* moves the present flag a little and not only the box size.

    Args:
        landmarks: The 21 landmarks of one hand, None where unprojectable.
            NaN in joints_2d.csv, and so None here, already covers the two ways
            a landmark can be nowhere: behind the camera, or past the fold of
            the lens distortion.  So no landmark that reaches this function is
            reporting a position it does not really have, and the hull cannot
            be dragged somewhere absurd by one.
        width, height: Image size in pixels.
        margin: Fraction of the hull's longer side to grow by, on all four
            sides.  See DEFAULT_BOX_MARGIN for why it is not per-axis.

    Returns:
        (x, y, w, h) with x, y the top-left corner, clipped to the image, or
        None when the box and the image do not overlap in any area.
    """
    seen = [p for p in landmarks if p is not None]
    if not seen:
        return None

    us = [u for u, _ in seen]
    vs = [v for _, v in seen]
    x0, x1 = min(us), max(us)
    y0, y1 = min(vs), max(vs)
    pad = max(x1 - x0, y1 - y0) * margin
    x0, x1 = x0 - pad, x1 + pad
    y0, y1 = y0 - pad, y1 + pad

    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(width), x1), min(float(height), y1)
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        round(x0, _PIXEL_DECIMALS),
        round(y0, _PIXEL_DECIMALS),
        round(x1 - x0, _PIXEL_DECIMALS),
        round(y1 - y0, _PIXEL_DECIMALS),
    )


def hand_rect_rows(
    landmarks: dict[str, list[list[Landmark]]],
    timestamps: list[str],
    width: int,
    height: int,
    margin: float = DEFAULT_BOX_MARGIN,
) -> list[list[object]]:
    """Every row of hand_rect.csv, in the order hand_rect_columns() names.

    An absent hand is written as present=0 and four NaNs rather than four
    zeros: a consumer that forgets to check the flag then fails loudly instead
    of quietly training on a box in the top-left corner.
    """
    nan = float("nan")
    rows: list[list[object]] = []
    for frame_idx, timestamp in enumerate(timestamps):
        row: list[object] = [timestamp]
        for side in SIDES:
            frames = landmarks.get(side)
            rect = None if frames is None else hand_rect(frames[frame_idx], width, height, margin)
            row += [0, nan, nan, nan, nan] if rect is None else [1, *rect]
        rows.append(row)
    return rows


def write_hand_rects(
    cam_dir: Path,
    resolution: tuple[int, int] | None = None,
    margin: float = DEFAULT_BOX_MARGIN,
) -> Path | None:
    """Derive hand_rect.csv from the joints_2d.csv in *cam_dir*.

    Args:
        cam_dir: A cam_<name>/ directory holding joints_2d.csv.
        resolution: (width, height) of the frames.  Worked out from the
            directory when not given.
        margin: Passed through to hand_rect().

    Returns:
        The path written, or None if there was no joints_2d.csv to read.
    """
    csv_path = cam_dir / JOINTS_2D_FILE
    if not csv_path.exists():
        return None
    timestamps, landmarks = read_joints_2d(csv_path)
    width, height = resolution if resolution is not None else resolution_for(cam_dir)

    out_path = cam_dir / HAND_RECT_FILE
    with out_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp"] + hand_rect_columns())
        writer.writerows(hand_rect_rows(landmarks, timestamps, width, height, margin))
    return out_path


def read_hand_rects(csv_path: Path) -> list[dict[str, Rect | None]]:
    """Read hand_rect.csv back, one {side: rect or None} per frame."""
    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    frames: list[dict[str, Rect | None]] = []
    for row in rows:
        frame: dict[str, Rect | None] = {}
        for side in SIDES:
            if row.get(f"{side}_present", "0").strip() not in ("1", "1.0", "True"):
                frame[side] = None
                continue
            frame[side] = (
                float(row[f"{side}_x"]),
                float(row[f"{side}_y"]),
                float(row[f"{side}_w"]),
                float(row[f"{side}_h"]),
            )
        frames.append(frame)
    return frames


def resolution_for(cam_dir: Path) -> tuple[int, int]:
    """The frame size of the render in *cam_dir*.

    The rendered PNGs are asked first, because they are the pixels the boxes
    have to be clipped against; a clip's sequence.json is the fallback for a
    directory whose frames have been moved away, and 640x480 the last resort.
    """
    for frame in sorted(cam_dir.glob("frame_??????.png"))[:1]:
        from PIL import Image  # noqa: PLC0415 -- see module docstring: not available inside Blender

        with Image.open(frame) as img:
            return img.size
    for parent in cam_dir.parents:
        sequence = parent / "sequence.json"
        if sequence.exists():
            camera = json.loads(sequence.read_text())["body_rig"]["cameras"][0]
            width, height = camera["resolution"]
            return int(width), int(height)
    return DEFAULT_RESOLUTION


_DOT_RADIUS = 3
_LINE_WIDTH = 1
_BOX_WIDTH = 2


def overlay_frames(cam_dir: Path, margin: float = DEFAULT_BOX_MARGIN) -> list[Path]:
    """Draw the exported annotations onto every frame in *cam_dir*.

    Per hand: the skeleton lines and per-finger colored dots from
    joints_2d.csv, and the bounding box from hand_rect.csv -- read back from
    the file rather than recomputed, so what the picture shows is what the
    dataset actually carries.  hand_rect.csv is written first if missing.

    Returns:
        The frame_XXXXXX_debug.png paths written.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415 -- see module docstring

    csv_path = cam_dir / JOINTS_2D_FILE
    if not csv_path.exists():
        return []
    _, landmarks = read_joints_2d(csv_path)
    if not landmarks:
        return []

    rect_path = cam_dir / HAND_RECT_FILE
    if not rect_path.exists():
        write_hand_rects(cam_dir, margin=margin)
    rects = read_hand_rects(rect_path) if rect_path.exists() else []

    written: list[Path] = []
    annotated_frames = min(len(f) for f in landmarks.values())
    for frame_idx, frame_path in enumerate(sorted(cam_dir.glob("frame_??????.png"))):
        if frame_idx >= annotated_frames:
            break
        img = Image.open(frame_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for side, frames in landmarks.items():
            hand = frames[frame_idx]

            # Lines first, so the dots sit on top of them.
            for a, b in HAND_CONNECTIONS:
                if a >= len(hand) or b >= len(hand):
                    continue
                start, end = hand[a], hand[b]
                if start is not None and end is not None:
                    draw.line([start, end], fill=LANDMARK_COLORS[a], width=_LINE_WIDTH)
            for k, point in enumerate(hand):
                if point is None:
                    continue
                u, v = point
                r = _DOT_RADIUS
                draw.ellipse([u - r - 1, v - r - 1, u + r + 1, v + r + 1], fill=(0, 0, 0))
                draw.ellipse([u - r, v - r, u + r, v + r], fill=LANDMARK_COLORS[k])

            rect = rects[frame_idx].get(side) if frame_idx < len(rects) else None
            if rect is not None:
                x, y, w, h = rect
                # -1 on the far edge: PIL draws the rectangle inclusive of it,
                # and a box flush with the image would otherwise fall outside.
                draw.rectangle(
                    [x, y, x + w - 1, y + h - 1],
                    outline=BOX_COLORS.get(side, (255, 255, 255)),
                    width=_BOX_WIDTH,
                )

        debug_path = frame_path.with_stem(frame_path.stem + "_debug")
        img.save(debug_path)
        written.append(debug_path)
    return written


def annotate(output_path: Path, overlay: bool = True, margin: float = DEFAULT_BOX_MARGIN) -> list[Path]:
    """Write hand_rect.csv, and optionally the overlays, for every render under *output_path*.

    Walks to each cam_<name>/ by the joints_2d.csv in it, so one render's
    output folder, one clip and a whole dataset all work.

    Returns:
        The cam directories annotated.
    """
    touched: list[Path] = []
    for csv_path in sorted(output_path.rglob(JOINTS_2D_FILE)):
        cam_dir = csv_path.parent
        write_hand_rects(cam_dir, margin=margin)
        if overlay:
            overlay_frames(cam_dir, margin=margin)
        touched.append(cam_dir)
    return touched
