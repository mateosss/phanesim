# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Planning and bookkeeping for dataset clips.

A *clip* is one continuous stretch of rendered frames kept in its own directory:
its ``sequence.json``, its ``animation.json`` and the frames rendered from them.
It is the unit of generation, of resuming, and of the train/val/test split.

Everything that varies *within* a clip is the motion — hand poses, head angle.
Everything else is fixed for its whole length: background, body, camera.  So a
clip's frames are many samples of one pose path but only *one* sample of the
background and camera, which is why the split has to be made per clip and never
per frame.

This module is bpy-free: planning runs without Blender.
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import UTC, datetime
from pathlib import Path

DONE_FILE = "_done.json"
SEQUENCE_FILE = "sequence.json"
ANIMATION_FILE = "animation.json"

# The distortion the forward map in render.distort_pixel was fitted against.
# lens_scale is anchored to this pair: 0.387 needs 1.2 to crop the black corners
# the distortion opens up, and the crop scales with it.
CALIBRATED_DISTORTION = 0.387
CALIBRATED_LENS_SCALE = 1.2

# Per-clip camera variation.  Only these four move; see fixed_camera_fields()
# for what deliberately does not and why.
CAMERA_RANGES: dict[str, tuple[float, float]] = {
    # Sensor gain differs between devices and exposure settings.
    "noise_std": (0.05, 0.40),
    # How much the lens mount darkens the corners.
    "vignette_factor": (0.30, 0.70),
    # Kept inside the range distort_pixel was verified over.  Beyond ~0.4 the
    # fitted forward map drifts from Blender's own, which would silently put the
    # 2D ground truth in the wrong place.
    "distortion": (0.25, 0.40),
    # Focal length in pixels, i.e. field of view: 210 to 270 spans about 100 to
    # 113 degrees horizontally on a 640-wide sensor.
    "fx": (210.0, 270.0),
}


# How many accessories a clip wears.  Bare hands are weighted highest because
# they are the common case and because a network should learn the hand, not the
# jewellery; the tail above two is kept small since wearing three or four at
# once is rare.  A model that carries fewer than four (model2 carries none) can
# only reach the counts it has, so the realised split is always at least this
# bare -- plan-clips prints what it actually drew.
ACCESSORY_COUNT_WEIGHTS: dict[int, float] = {0: 0.50, 1: 0.30, 2: 0.15, 3: 0.04, 4: 0.01}


def choose_accessories(available: list[str], rng: random.Random) -> list[str]:
    """Draw what one clip wears, from the accessories its model actually has.

    Args:
        available: Short names the model carries; empty gives bare hands.
        rng:       Source of randomness.

    Returns:
        Sorted names, so a clip's sequence.json is stable to compare.
    """
    if not available:
        return []
    counts = [n for n in ACCESSORY_COUNT_WEIGHTS if n <= len(available)]
    weights = [ACCESSORY_COUNT_WEIGHTS[n] for n in counts]
    how_many = rng.choices(counts, weights=weights, k=1)[0]
    return sorted(rng.sample(available, how_many))


def fixed_camera_fields() -> dict[str, str]:
    """Camera fields that must not be randomised, and the reason for each."""
    return {
        "cy/cx": (
            "Blender renders with the principal point at the image centre and no "
            "shift is applied, so moving cx/cy would move the projected ground "
            "truth away from where the pixels actually are."
        ),
        "fy": "Only fx drives the Blender lens, so fy has to equal it.",
        "resolution": "One resolution for the whole dataset, as agreed.",
        "ca_factor/chroma_noise": "The output is MONO8; chroma effects do not survive it.",
    }


def lens_scale_for(distortion: float) -> float:
    """Crop factor that hides the black corners *distortion* opens up.

    Anchored to the one pair that was checked by eye, and scaled linearly from
    there: more distortion pulls the corners further in and needs more crop.
    """
    return 1.0 + (distortion / CALIBRATED_DISTORTION) * (CALIBRATED_LENS_SCALE - 1.0)


def randomize_camera(camera: dict, rng: random.Random) -> dict:
    """Return *camera* with the per-clip fields redrawn.

    The input is left untouched; the caller gets a new dict to write out.
    """
    out = json.loads(json.dumps(camera))  # deep copy of plain JSON data
    out["noise_std"] = round(rng.uniform(*CAMERA_RANGES["noise_std"]), 4)
    out["vignette_factor"] = round(rng.uniform(*CAMERA_RANGES["vignette_factor"]), 4)

    distortion = round(rng.uniform(*CAMERA_RANGES["distortion"]), 4)
    out["distortion"] = distortion
    out["lens_scale"] = round(lens_scale_for(distortion), 4)

    fx = round(rng.uniform(*CAMERA_RANGES["fx"]), 2)
    params = out.setdefault("intrinsics", {}).setdefault("parameters", {})
    params["fx"] = fx
    params["fy"] = fx  # only fx reaches the Blender lens, so they must agree
    return out


def fingerprint(clip_dir: Path) -> str:
    """Hash of the two files that fully describe what a clip should render to.

    Recorded when a clip finishes so that a later run can tell "already rendered"
    from "rendered, but with different settings than are being asked for now".
    """
    h = hashlib.sha256()
    for name in (SEQUENCE_FILE, ANIMATION_FILE):
        h.update((clip_dir / name).read_bytes())
    return h.hexdigest()[:16]


def is_done(clip_dir: Path) -> bool:
    """True if this clip was rendered from exactly its current settings."""
    done = clip_dir / DONE_FILE
    if not done.exists():
        return False
    try:
        return json.loads(done.read_text()).get("fingerprint") == fingerprint(clip_dir)
    except (OSError, ValueError):
        return False


def mark_done(clip_dir: Path, frames: int) -> None:
    """Record that this clip finished, and what it finished from."""
    (clip_dir / DONE_FILE).write_text(
        json.dumps(
            {
                "frames": int(frames),
                "fingerprint": fingerprint(clip_dir),
                "finished_at": datetime.now(UTC).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )


def clip_dirs(dataset_dir: Path) -> list[Path]:
    """Every clip directory under *dataset_dir*, in name order."""
    return sorted(d for d in dataset_dir.iterdir() if d.is_dir() and (d / SEQUENCE_FILE).exists())


def stray_clip_dirs(dataset_dir: Path) -> list[Path]:
    """Directories holding a clip one level too deep to be found.

    A clip is recognised by a sequence.json directly inside it, so planning into
    ``dataset/clip_00002`` instead of ``dataset`` buries the clips where nothing
    looks for them and they are skipped in silence.  Reporting them is cheap and
    beats a dataset that is quietly short.
    """
    return sorted(
        d
        for d in dataset_dir.iterdir()
        if d.is_dir() and not (d / SEQUENCE_FILE).exists() and any(d.rglob(SEQUENCE_FILE))
    )


def next_clip_index(dataset_dir: Path) -> int:
    """One past the highest clip number present, or 0 for an empty dataset.

    Numbering continues rather than restarting so a second plan can extend a
    dataset instead of overwriting the first one -- which would not merely lose
    the plan but invalidate every _done.json and force a re-render.
    """
    highest = -1
    for d in clip_dirs(dataset_dir):
        suffix = d.name.rsplit("_", 1)[-1]
        if suffix.isdigit():
            highest = max(highest, int(suffix))
    return highest + 1
