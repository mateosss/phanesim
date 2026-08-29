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

Every asset is a single-frame pose.  Multi-frame playback was removed once the
pose library covered the hand shapes on its own: a played-back clip pinned the
whole body to one authored motion, which is the opposite of what a frame-diverse
dataset wants.

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


@dataclass
class PoseEvent:
    """A single pose asset reached at a point in time.

    Attributes:
        t_ns:  Time at which the asset is fully reached, in nanoseconds.
        asset: Name of the pose asset (an action inside the model .blend).
        group: Body region the asset drives ("left", "right", "head", "body").
               Events in different groups touch disjoint bones and are applied
               independently, so they layer rather than replace.
    """

    t_ns: int
    asset: str
    group: str = "body"

    def to_dict(self) -> dict:
        return {
            "t_ns": int(self.t_ns),
            "asset": self.asset,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseEvent:
        return cls(
            t_ns=int(data["t_ns"]),
            asset=str(data["asset"]),
            group=str(data.get("group", "body")),
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
        """End of the timeline.

        Every event is instantaneous and the sampler pins the last one to
        duration_ns, so the declared length is the end.
        """
        return int(self.duration_ns)

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
        return "\n".join(f"  {e.t_ns / NS_PER_SECOND:6.2f}s  {e.group:<5} pose  {e.asset}" for e in self.events)


@dataclass
class PoseAsset:
    """A single-frame pose asset discovered inside a model .blend.

    *frame* is the frame its action holds the pose on, which is what the renderer
    evaluates the action at to read the bone values out.

    *group* records which part of the body the asset drives, derived from the
    bones it animates rather than from its name.  Assets in different groups
    touch disjoint bones and can therefore be combined freely, which is what
    turns a library of N poses into a much larger space of configurations.
    """

    name: str
    frame: int
    group: str = "body"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frame": int(self.frame),
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseAsset:
        return cls(
            name=str(data["name"]),
            frame=int(data["frame"]),
            group=str(data.get("group", "body")),
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
    lanes: tuple[tuple[str, ...], ...] = (("left", "right", "body"),),
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

    Both rules apply per *lane*.  A lane is one timeline; giving each hand its
    own lane lets them change at different moments, while a single lane covering
    both draws one pose at a time and lets whichever hand it belongs to move.
    Because a pose only ever keys its own bones, poses from different lanes are
    held simultaneously rather than replacing one another.

    Args:
        assets:      Pose assets available to draw from.
        duration_ns: Length of the timeline in nanoseconds.
        name:        Name recorded in the description.
        event_count: Exact number of poses, including the one at t=0.
        seed:        Seed for reproducibility; None draws a random one.  Either
                     way the value used is recorded on the returned PoseMotion,
                     so any timeline can be regenerated exactly.
        model:       Path to the .blend the assets came from, recorded for reference.
        rest_asset:  Asset to key at t=0 for the group it belongs to; the other
                     groups open on a random pose.  Defaults to random throughout.
        lanes:       Independent timelines to sample, each named by the groups
                     it draws from.  One lane over ("left", "right", "body") is
                     a single shared timeline where every event is whichever
                     pose was drawn; two lanes ("left",) and ("right",) give
                     each hand its own schedule.  Every lane gets event_count
                     events, so the total is event_count x len(lanes).

    Returns:
        A PoseMotion whose events are sorted by time.

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
    if rest_asset is not None and not any(a.name == rest_asset for a in assets):
        raise ValueError(f"rest_asset {rest_asset!r} is not among the available assets")

    if seed is None:
        seed = random.randrange(2**31)
    rng = random.Random(seed)

    def make(t_ns: int, asset: PoseAsset) -> PoseEvent:
        return PoseEvent(t_ns=t_ns, asset=asset.name, group=asset.group)

    def sample_lane(pool: list[PoseAsset]) -> list[PoseEvent]:
        """Draw one independent timeline from *pool*."""
        # The opening pose is keyed at t=0 so the clip does not start mid-motion.
        start = next((a for a in pool if a.name == rest_asset), None) if rest_asset else rng.choice(pool)
        if start is None:
            start = rng.choice(pool)
        out = [make(0, start)]

        # Interior times are the order statistics of uniform draws — the arrival
        # times of the conditioned process.  The final event is pinned to the end
        # of the timeline instead of being drawn: uniform placement leaves the
        # clip ending part way through, and whatever pose was last reached then
        # freezes for the remainder.  Sampled freely, that dead tail averages a
        # quarter of a four-event clip.
        times = sorted(rng.randrange(1, max(2, duration_ns)) for _ in range(max(0, event_count - 2)))
        if event_count > 1:
            times.append(duration_ns)

        previous = start.name
        for t_ns in times:
            # Never draw the pose that is already showing: an event that changes
            # nothing freezes every frame until the next one.
            candidates = [a for a in pool if a.name != previous] or pool
            asset = rng.choice(candidates)
            out.append(make(t_ns, asset))
            previous = asset.name
        return out

    # Each lane is sampled independently and the timelines are merged.  Where the
    # lanes drive disjoint bones -- the two hands, or a hand and the head -- a
    # library of L left and R right poses spans L x R configurations rather than
    # L + R, because both are held at once.
    pools = [[a for a in assets if a.group in lane] for lane in lanes]
    if not any(pools):
        available = sorted({a.group for a in assets})
        raise ValueError(f"no assets in lanes {[list(lane) for lane in lanes]}; found groups {available}")

    events: list[PoseEvent] = []
    for pool in pools:
        if pool:
            events.extend(sample_lane(pool))
    events.sort(key=lambda e: e.t_ns)

    return PoseMotion(
        name=name,
        events=events,
        duration_ns=duration_ns,
        model=model,
        seed=seed,
    )
