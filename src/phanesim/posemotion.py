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
Each event names the asset that is *fully reached* at ``t_ns``.  Motion is
continuous: the armature interpolates from one pose straight into the next over
the whole gap between them, so no frame is ever a repeat of the one before it::

    pose A reached      interpolating       pose B reached
         t0 |---------------------------------| t1

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

# Sampling defaults, used when the caller asks for neither a count nor a length.
DEFAULT_EVENT_COUNT = 8
DEFAULT_DURATION_SECONDS = 20.0
# Generation is reproducible by default: the same command yields the same
# timeline unless a different seed is asked for.
DEFAULT_SEED = 42


@dataclass
class PoseEvent:
    """A single pose asset reached at a point in time.

    Attributes:
        t_ns:         Time at which the asset is fully reached, in nanoseconds.
        asset:        Name of the pose asset (an action inside the model .blend).
        kind:         "pose" for single-frame assets, "action" for multi-frame ones.
        duration_ns:  For "action" events, playback length after t_ns; 0 for poses.
        source_frames: For "action" events, the asset's own [first, last] frame range.
    """

    t_ns: int
    asset: str
    kind: str = "pose"
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
            duration_ns=int(data.get("duration_ns", 0)),
            source_frames=(int(frames[0]), int(frames[1])) if frames else None,
        )


@dataclass
class PoseMotion:
    """An ordered timeline of pose events plus the parameters that produced it.

    JSON schema:
      {
        "name":        "<str>",
        "model":       "<path to the .blend holding the pose assets>",
        "seed":        <int>,
        "duration_ns": <int>,     -- nanoseconds, the canonical timeline length
        "event_count": <int>,     -- derived, equals len(events)
        "events":      [<PoseEvent dict>, ...]
      }
    """

    name: str
    events: list[PoseEvent]
    duration_ns: int
    model: str | None = None
    seed: int | None = None
    source: Path | None = field(default=None, compare=False)

    @property
    def event_count(self) -> int:
        return len(self.events)

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
        # event_count is derived from the events below; it is written so the file
        # states its own size without counting, and is recomputed on every write
        # rather than trusted on read.
        return {
            "name": self.name,
            "model": self.model,
            "seed": self.seed,
            "duration_ns": int(self.duration_ns),
            "event_count": self.event_count,
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
    event_count: int = DEFAULT_EVENT_COUNT,
    seed: int | None = None,
    model: str | None = None,
    rest_asset: str | None = None,
) -> PoseMotion:
    """Place exactly *event_count* poses at random times within *duration_ns*.

    The count and the length are both given, so a request reads directly: four
    poses in one second, ten poses in twenty seconds.  Only *which* poses and
    *when* they land are random.

    This is still a Poisson process, conditioned on its event count.  A Poisson
    process observed to produce n arrivals in [0, T] has those arrivals
    distributed exactly as the order statistics of n independent uniform draws on
    [0, T] — so drawing uniforms and sorting them is not an approximation of the
    process, it is the process with its count fixed.

    The first pose is keyed at t=0 and the last at duration_ns, so motion spans
    the whole clip and no frames are left frozen at either end; the remaining
    ones fall anywhere in between.  No pose is ever drawn twice in a row, since
    an event that changes nothing would freeze every frame until the next one.
    There is no minimum spacing: poses may land arbitrarily close together, and
    the armature simply moves faster to reach the next one in time.

    Args:
        assets:      Pose assets available to draw from.
        duration_ns: Length of the timeline in nanoseconds.
        name:        Name recorded in the description.
        event_count: Exact number of poses, including the one at t=0.
        seed:        Seed for reproducibility; None draws a random one.
        model:       Path to the .blend the assets came from, recorded for reference.
        rest_asset:  Asset to key at t=0; defaults to a random one.

    Returns:
        A PoseMotion with exactly *event_count* events, sorted by time.

    Raises:
        ValueError: If *assets* is empty, *duration_ns* is not positive, or
            *event_count* is less than one.
    """
    if not assets:
        raise ValueError("no pose assets to sample from")
    if duration_ns <= 0:
        raise ValueError(f"duration_ns must be positive, got {duration_ns}")
    if event_count < 1:
        raise ValueError(f"event_count must be at least 1, got {event_count}")

    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    def make(t_ns: int, asset: PoseAsset) -> PoseEvent:
        return PoseEvent(
            t_ns=t_ns,
            asset=asset.name,
            kind=asset.kind,
            duration_ns=asset.duration_ns,
            source_frames=(asset.frame_start, asset.frame_end) if asset.is_action else None,
        )

    # The opening pose is keyed at t=0 with no transition into it.
    start = next((a for a in assets if a.name == rest_asset), None) if rest_asset else rng.choice(assets)
    if start is None:
        raise ValueError(f"rest_asset {rest_asset!r} is not among the available assets")
    events = [make(0, start)]

    # Interior times are the order statistics of uniform draws — the arrival times
    # of the conditioned process.  The final event is pinned to the end of the
    # timeline instead of being drawn: uniform placement leaves the clip ending
    # part way through, and whatever pose was last reached then freezes for the
    # remainder.  Sampled freely, that dead tail averages a quarter of a
    # four-event clip.
    times = sorted(rng.randrange(1, max(2, duration_ns)) for _ in range(max(0, event_count - 2)))
    if event_count > 1:
        times.append(duration_ns)

    previous = start.name
    for t_ns in times:
        # Never draw the pose that is already showing: an event that changes
        # nothing freezes every frame until the next one.
        pool = [a for a in assets if a.name != previous] or assets
        asset = rng.choice(pool)
        events.append(make(t_ns, asset))
        previous = asset.name

    # A multi-frame asset cannot play for longer than the gap it was given, or
    # its playback would run past the pose that follows it.
    for i, event in enumerate(events):
        if event.kind != "action":
            continue
        following = events[i + 1].t_ns if i + 1 < len(events) else duration_ns
        event.duration_ns = min(event.duration_ns, max(0, following - event.t_ns))

    return PoseMotion(
        name=name,
        events=events,
        duration_ns=duration_ns,
        model=model,
        seed=seed,
    )
