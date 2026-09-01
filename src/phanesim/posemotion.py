# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Pose-asset motion descriptions and their Poisson-process sampler.

A *motion description* is the timeline half of an animation: which pose asset is
reached at what time, and nothing about bone values.  Those live in the pose
assets inside the model .blend and are resolved at render time, which keeps a
description small, readable and diff-able.

Each event names the asset *fully reached* at ``t_ns``; the armature interpolates
from one pose into the next across the whole gap, so consecutive frames always
differ.  Every asset is a single-frame pose.

This module is bpy-free so it can be imported outside Blender.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path

NS_PER_SECOND = 1_000_000_000

# Defaults for generate-motion's --events and --duration.
DEFAULT_EVENT_COUNT = 8
DEFAULT_DURATION_SECONDS = 20.0


@dataclass
class PoseEvent:
    """A pose asset reached at a point in time.

    *group* is the body region the asset drives: "left", "right", "head" or
    "body".  Groups touch disjoint bones, so events from different groups layer
    rather than replace one another.
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

    Serialises to JSON as name, model, seed, duration_ns, event_count and events.
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
        """End of the timeline; the sampler pins the last event to duration_ns."""
        return int(self.duration_ns)

    def assets(self) -> list[str]:
        """Unique asset names referenced by this timeline, in first-use order."""
        seen: list[str] = []
        for e in self.events:
            if e.asset not in seen:
                seen.append(e.asset)
        return seen

    def to_dict(self) -> dict:
        # event_count is derived: written so the file states its own size,
        # recomputed on every write rather than trusted on read.
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

    *frame* is the frame its action holds the pose on, which is where the
    renderer evaluates it.  *group* is derived from the bones the asset animates
    rather than from its name, so assets in different groups combine freely —
    what turns a library of N poses into a much larger space of configurations.
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

    This is a Poisson process conditioned on its event count: a process with n
    arrivals in [0, T] has them distributed exactly as the order statistics of n
    uniform draws on [0, T], so drawing uniforms and sorting them *is* the
    process rather than an approximation of it.

    Two rules keep every limb moving, both enforced per body group rather than
    per lane: the first pose is keyed at t=0 and the last at duration_ns, and no
    pose is drawn twice running for the same group.  Per lane they would not be
    enough — a shared lane can repeat one hand's pose around the other hand's
    event, and only one group can own the event pinned to the end.  There is no
    minimum spacing; poses may land arbitrarily close together.

    Args:
        assets:      Pose assets available to draw from.
        duration_ns: Length of the timeline in nanoseconds.
        name:        Name recorded in the description.
        event_count: Exact number of poses per lane, including the one at t=0.
        seed:        None draws a random one.  Either way it is recorded on the
                     result, so any timeline can be regenerated exactly.
        model:       Path to the .blend the assets came from, for reference.
        rest_asset:  Asset to key at t=0 for its own group; other groups open on
                     a random pose.
        lanes:       Independent timelines, each named by the groups it draws
                     from.  One lane over ("left", "right", "body") shares a
                     single schedule; ("left",) and ("right",) give each hand its
                     own.  Total events is event_count x len(lanes).

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
        # Keyed at t=0 so the clip does not start mid-motion.
        start = next((a for a in pool if a.name == rest_asset), None) if rest_asset else rng.choice(pool)
        if start is None:
            start = rng.choice(pool)
        out = [make(0, start)]

        # Interior times are the order statistics of uniform draws.  The last
        # event is pinned rather than drawn: placed uniformly it leaves the clip
        # ending part way through, freezing the final pose for the remainder —
        # a dead tail averaging a quarter of a four-event clip.
        times = sorted(rng.randrange(1, max(2, duration_ns)) for _ in range(max(0, event_count - 2)))
        if event_count > 1:
            times.append(duration_ns)

        # Tracked per group: two events in a shared lane can name the same
        # left-hand pose with a right-hand event between them, which freezes that
        # hand across the span even though the lane never repeats itself.
        previous: dict[str, str] = {start.group: start.name}
        for t_ns in times:
            candidates = [a for a in pool if a.name != previous.get(a.group)] or pool
            asset = rng.choice(candidates)
            out.append(make(t_ns, asset))
            previous[asset.group] = asset.name

        # Pinning only the lane's last event saves whichever group owned it; the
        # rest hold their final pose to the end.  Whole-body poses are left where
        # they fall, since they key both hands and would fight the hands' own
        # closing poses.
        for group in {e.group for e in out} - {"body"}:
            latest = [e for e in out if e.group == group][-1]
            # Unless it is that group's opening event: moving it would leave the
            # clip starting from rest with nothing posed.
            if latest.t_ns != 0:
                latest.t_ns = duration_ns
        return out

    # Lanes are sampled independently and merged.  Because they drive disjoint
    # bones, L left and R right poses span L x R configurations, not L + R.
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
