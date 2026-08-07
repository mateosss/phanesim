# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose-asset motion descriptions and their Poisson-process sampler.

A *motion description* is the timeline half of an animation: it says **which**
pose asset is reached at **what** time, and nothing about bone values.  The bone
values live in the pose assets inside the model .blend and are only resolved at
render time.  Splitting the two means a description is a small, readable,
diff-able JSON that can be generated, inspected and version-controlled without
Blender.

Timeline semantics
------------------
Each event names the asset that is *fully reached* at ``t_ns``.  The transition
into it starts ``blend_ns`` earlier, so the armature holds the previous pose
until ``t_ns - blend_ns`` and then interpolates::

    ... hold prev ...|<-- blend_ns -->|<-- asset reached at t_ns
                  t_ns-blend_ns      t_ns

For ``kind == "action"`` events the asset is a multi-frame animation: after it is
reached at ``t_ns`` its source frames are played back over ``duration_ns``.

This module is bpy-free so it can be imported outside Blender.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

NS_PER_SECOND = 1_000_000_000

# Sampling defaults, matching tools/blender/poisson_animation.py.
DEFAULT_EVENTS_PER_SECOND = 0.4
DEFAULT_BLEND_SECONDS = 0.5
DEFAULT_MIN_GAP_SECONDS = 1.5


@dataclass
class PoseEvent:
    """A single pose asset reached at a point in time.

    Attributes:
        t_ns:         Time at which the asset is fully reached, in nanoseconds.
        asset:        Name of the pose asset (an action inside the model .blend).
        kind:         "pose" for single-frame assets, "action" for multi-frame ones.
        blend_ns:     Duration of the transition that ends at t_ns.
        duration_ns:  For "action" events, playback length after t_ns; 0 for poses.
        source_frames: For "action" events, the asset's own [first, last] frame range.
    """

    t_ns: int
    asset: str
    kind: str = "pose"
    blend_ns: int = int(DEFAULT_BLEND_SECONDS * NS_PER_SECOND)
    duration_ns: int = 0
    source_frames: tuple[int, int] | None = None

    @property
    def end_ns(self) -> int:
        """Time at which this event is finished, including any action playback."""
        return self.t_ns + self.duration_ns

    def to_dict(self) -> dict:
        data: dict = {
            "t_ns": int(self.t_ns),
            "asset": self.asset,
            "kind": self.kind,
            "blend_ns": int(self.blend_ns),
        }
        if self.kind == "action":
            data["duration_ns"] = int(self.duration_ns)
            if self.source_frames is not None:
                data["source_frames"] = [int(self.source_frames[0]), int(self.source_frames[1])]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> PoseEvent:
        frames = data.get("source_frames")
        return cls(
            t_ns=int(data["t_ns"]),
            asset=str(data["asset"]),
            kind=str(data.get("kind", "pose")),
            blend_ns=int(data.get("blend_ns", DEFAULT_BLEND_SECONDS * NS_PER_SECOND)),
            duration_ns=int(data.get("duration_ns", 0)),
            source_frames=(int(frames[0]), int(frames[1])) if frames else None,
        )


@dataclass
class PoseMotion:
    """An ordered timeline of pose events plus the parameters that produced it.

    JSON schema:
      {
        "name":              "<str>",
        "model":             "<path to the .blend holding the pose assets>",
        "seed":              <int>,
        "events_per_second": <float>,
        "duration_ns":       <int>,
        "events":            [<PoseEvent dict>, ...]
      }
    """

    name: str
    events: list[PoseEvent]
    duration_ns: int
    model: str | None = None
    seed: int | None = None
    events_per_second: float = DEFAULT_EVENTS_PER_SECOND
    source: Path | None = field(default=None, compare=False)

    @property
    def t_start_ns(self) -> int:
        return self.events[0].t_ns if self.events else 0

    @property
    def t_end_ns(self) -> int:
        """End of the timeline: the later of the declared duration and the last event."""
        last = max((e.end_ns for e in self.events), default=0)
        return max(int(self.duration_ns), last)

    def assets(self) -> list[str]:
        """Unique asset names referenced by this timeline, in first-use order."""
        seen: list[str] = []
        for e in self.events:
            if e.asset not in seen:
                seen.append(e.asset)
        return seen

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model": self.model,
            "seed": self.seed,
            "events_per_second": self.events_per_second,
            "duration_ns": int(self.duration_ns),
            "events": [e.to_dict() for e in self.events],
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n")

    @classmethod
    def from_dict(cls, data: dict, source: Path | None = None) -> PoseMotion:
        events = [PoseEvent.from_dict(e) for e in data["events"]]
        events.sort(key=lambda e: e.t_ns)
        return cls(
            name=str(data["name"]),
            events=events,
            duration_ns=int(data["duration_ns"]),
            model=data.get("model"),
            seed=data.get("seed"),
            events_per_second=float(data.get("events_per_second", DEFAULT_EVENTS_PER_SECOND)),
            source=source,
        )

    @classmethod
    def from_path(cls, path: Path) -> PoseMotion:
        return cls.from_dict(json.loads(path.read_text()), source=path)

    def summary(self) -> str:
        """Human-readable one-line-per-event timeline, e.g. for CLI output."""
        lines = []
        for e in self.events:
            t = e.t_ns / NS_PER_SECOND
            if e.kind == "action":
                lines.append(f"  {t:6.2f}s  play  {e.asset} ({e.duration_ns / NS_PER_SECOND:.2f}s)")
            else:
                lines.append(f"  {t:6.2f}s  pose  {e.asset}")
        return "\n".join(lines)


@dataclass
class PoseAsset:
    """A pose asset discovered inside a model .blend.

    A single-frame asset is a static pose; a multi-frame one is an animation that
    is played back in full.
    """

    name: str
    frame_start: int
    frame_end: int
    fps: int = 24

    @property
    def is_action(self) -> bool:
        return self.frame_end > self.frame_start

    @property
    def kind(self) -> str:
        return "action" if self.is_action else "pose"

    @property
    def duration_ns(self) -> int:
        if not self.is_action:
            return 0
        return int((self.frame_end - self.frame_start) * NS_PER_SECOND / self.fps)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frame_start": int(self.frame_start),
            "frame_end": int(self.frame_end),
            "fps": int(self.fps),
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseAsset:
        return cls(
            name=str(data["name"]),
            frame_start=int(data["frame_start"]),
            frame_end=int(data["frame_end"]),
            fps=int(data.get("fps", 24)),
        )


def sample_pose_motion(
    assets: list[PoseAsset],
    duration_ns: int,
    *,
    name: str = "animation",
    events_per_second: float = DEFAULT_EVENTS_PER_SECOND,
    blend_ns: int = int(DEFAULT_BLEND_SECONDS * NS_PER_SECOND),
    min_gap_ns: int = int(DEFAULT_MIN_GAP_SECONDS * NS_PER_SECOND),
    seed: int | None = None,
    model: str | None = None,
    rest_asset: str | None = None,
) -> PoseMotion:
    """Draw a random pose timeline from a Poisson process.

    Gesture events arrive as a Poisson process of rate *events_per_second*, so the
    gaps between them are exponentially distributed and memoryless — the same
    statistics as spontaneous, unscheduled hand movement.  Each gap is floored at
    *min_gap_ns* so poses stay legible instead of stacking on top of each other.

    Args:
        assets:            Pose assets available to draw from.
        duration_ns:       Length of the timeline in nanoseconds.
        name:              Name recorded in the description.
        events_per_second: Poisson rate; higher means busier motion.
        blend_ns:          Transition duration into each pose.
        min_gap_ns:        Lower bound applied to every sampled gap.
        seed:              Seed for reproducibility; None draws a random one.
        model:             Path to the .blend the assets came from, recorded for reference.
        rest_asset:        Asset to key at t=0; defaults to the first static pose.

    Returns:
        A PoseMotion whose events are sorted by time.

    Raises:
        ValueError: If *assets* is empty or *duration_ns* is not positive.
    """
    if not assets:
        raise ValueError("no pose assets to sample from")
    if duration_ns <= 0:
        raise ValueError(f"duration_ns must be positive, got {duration_ns}")

    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    events: list[PoseEvent] = []

    # Key a starting pose at t=0 so the timeline does not open mid-interpolation.
    static = [a for a in assets if not a.is_action]
    start_name = rest_asset or (static[0].name if static else assets[0].name)
    start = next((a for a in assets if a.name == start_name), assets[0])
    events.append(PoseEvent(t_ns=0, asset=start.name, kind=start.kind, blend_ns=0))

    t_ns = 0
    while True:
        gap_ns = max(int(rng.expovariate(events_per_second) * NS_PER_SECOND), min_gap_ns)
        t_ns += gap_ns
        if t_ns > duration_ns:
            break

        asset = rng.choice(assets)
        events.append(
            PoseEvent(
                t_ns=t_ns,
                asset=asset.name,
                kind=asset.kind,
                blend_ns=blend_ns,
                duration_ns=asset.duration_ns,
                source_frames=(asset.frame_start, asset.frame_end) if asset.is_action else None,
            )
        )
        # A multi-frame asset occupies the timeline while it plays.
        t_ns += asset.duration_ns

    return PoseMotion(
        name=name,
        events=events,
        duration_ns=duration_ns,
        model=model,
        seed=seed,
        events_per_second=events_per_second,
    )
