# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""How much of the hand a rendered clip actually shows.

The generator can only be tuned against what came out of it, and "is there a
hand in this frame" turns out to need saying carefully.  ``joints_2d.csv`` writes
NaN only for a landmark behind the camera or past the point where the lens
distortion folds, so a hand that is merely outside the frame still carries real
pixel coordinates and has to be bounded against the resolution to be counted out.

A landmark count also has to come with a threshold, and the threshold changes the
answer a lot: on dataset_test4, counting a hand as present when any 5 of its 21
landmarks are in frame put a hand in 68% of frames, where counting only whole
hands put one in 39%.  The difference is hands sliced by the frame edge.  So this
reports three thresholds side by side rather than picking one, because a detector
and a keypoint regressor do not want the same one.

This module is bpy-free: it reads what a render already wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# Landmarks per hand, in the order joints_2d.csv writes them.
LANDMARKS_PER_HAND = 21

# The thresholds reported, as (label, how many of the 21 landmarks must be in
# frame).  "any" is what a detector can still draw a box around, "half" is a hand
# a person would call visible, and "whole" is one no frame edge has cut.
THRESHOLDS: tuple[tuple[str, int], ...] = (("any", 5), ("half", 11), ("whole", 21))

DEFAULT_RESOLUTION = (640, 480)


def resolution_for(csv_path: Path) -> tuple[int, int]:
    """The camera resolution the clip holding *csv_path* was rendered at.

    Read from the nearest enclosing sequence.json rather than assumed, because a
    hand is counted out by being outside the frame and that test is the frame's
    size.  Falls back to 640x480 when the file was moved away from its clip.
    """
    for parent in csv_path.parents:
        sequence = parent / "sequence.json"
        if sequence.exists():
            camera = json.loads(sequence.read_text())["body_rig"]["cameras"][0]
            width, height = camera["resolution"]
            return int(width), int(height)
    return DEFAULT_RESOLUTION


def hand_pixels(frame_data: pd.DataFrame, side: str) -> np.ndarray:
    """One side's landmark pixels from a joints_2d table, shaped (frames, 21, 2)."""
    u_cols = [c for c in frame_data.columns if c.startswith(f"{side}_") and c.endswith("_u")]
    v_cols = [f"{c[:-2]}_v" for c in u_cols]
    us = frame_data[u_cols].to_numpy(dtype=float)
    vs = frame_data[v_cols].to_numpy(dtype=float)
    return np.stack([us, vs], axis=-1)


def landmarks_in_frame(uv: np.ndarray, width: int, height: int) -> np.ndarray:
    """How many landmarks land inside the image, per frame.

    NaN compares false on every side of the box, so a landmark behind the camera
    is counted out by the same test that counts out one off the edge.
    """
    u, v = uv[..., 0], uv[..., 1]
    inside = (u >= 0) & (u < width) & (v >= 0) & (v < height)
    return inside.sum(axis=-1)


@dataclass
class Tally:
    """Frame counts for one threshold: how many frames showed 0, 1 or 2 hands."""

    label: str
    needed: int
    frames: list[int] = field(default_factory=list)  # hands visible, one entry per frame

    @property
    def total(self) -> int:
        return len(self.frames)

    def share(self, hands: int) -> float:
        return self.frames.count(hands) / self.total if self.total else 0.0

    @property
    def hands_per_frame(self) -> float:
        return sum(self.frames) / self.total if self.total else 0.0


def measure(path: Path) -> tuple[dict[str, Tally], dict[str, list[int]]]:
    """Count visible hands under every threshold, over one render or a dataset.

    Args:
        path: A directory holding joints_2d.csv anywhere beneath it -- one
            render's output folder, one clip, or a whole planned dataset.

    Returns:
        The tallies keyed by threshold label, and the per-clip frame counts for
        the strictest threshold, so the worst clips can be named.

    Raises:
        FileNotFoundError: If no joints_2d.csv is found under *path*.
    """
    csvs = sorted(path.rglob("joints_2d.csv"))
    if not csvs:
        raise FileNotFoundError(f"no joints_2d.csv under {path}")

    tallies = {label: Tally(label, needed) for label, needed in THRESHOLDS}
    per_clip: dict[str, list[int]] = {}

    for csv_path in csvs:
        width, height = resolution_for(csv_path)
        frame_data = pd.read_csv(csv_path)
        # A clip directory holds cam_<name>/joints_2d.csv, so the clip is two up.
        # Keyed by its path rather than its name: clip_00000 is a name two
        # datasets under one root both use, and keying by name would silently
        # merge them into one line of the report.
        clip = str(csv_path.parent.parent.relative_to(path)) if csv_path.parent.parent != path else str(path)
        seen = np.stack(
            [landmarks_in_frame(hand_pixels(frame_data, side), width, height) for side in ("left", "right")]
        )
        for label, needed in THRESHOLDS:
            hands = (seen >= needed).sum(axis=0)
            tallies[label].frames.extend(int(n) for n in hands)
            if label == THRESHOLDS[0][0]:
                per_clip.setdefault(clip, []).extend(int(n) for n in hands)
    return tallies, per_clip


def report(path: Path, worst: int = 5) -> str:
    """A human-readable summary of what *path* shows, for the CLI."""
    tallies, per_clip = measure(path)
    lines = [f"{len(per_clip)} clip(s), {next(iter(tallies.values())).total} frames", ""]
    lines.append(f"{'hand counted as visible when':<32}{'both':>8}{'one':>8}{'none':>8}{'hands/frame':>14}")
    for label, needed in THRESHOLDS:
        t = tallies[label]
        lines.append(
            f"{f'{label} ({needed}+ of 21 landmarks)':<32}"
            f"{t.share(2):>7.1%} {t.share(1):>7.1%} {t.share(0):>7.1%}{t.hands_per_frame:>14.2f}"
        )

    if len(per_clip) > 1:
        ranked = sorted(per_clip.items(), key=lambda kv: -kv[1].count(0) / len(kv[1]))[:worst]
        lines += ["", "  emptiest clips (no hand at all, loosest threshold):"]
        lines += [f"    {name:<16} {c.count(0)}/{len(c)} frames" for name, c in ranked]
    return "\n".join(lines)


def hand_pixel_sizes(path: Path) -> np.ndarray:
    """Bounding-box diagonals in pixels for every hand that is fully in frame.

    Widening the lens brings more hands into frame but makes each one smaller, so
    the two have to be read together: this is the other half of that trade.
    """
    sizes: list[float] = []
    for csv_path in sorted(path.rglob("joints_2d.csv")):
        width, height = resolution_for(csv_path)
        frame_data = pd.read_csv(csv_path)
        for side in ("left", "right"):
            uv = hand_pixels(frame_data, side)
            whole = landmarks_in_frame(uv, width, height) == LANDMARKS_PER_HAND
            for frame in uv[whole]:
                spread = frame.max(axis=0) - frame.min(axis=0)
                sizes.append(float(np.hypot(*spread)))
    return np.array(sizes)
