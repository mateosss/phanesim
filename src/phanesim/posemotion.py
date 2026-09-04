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

# How far an event may sit from its evenly spaced slot, as a fraction of the
# spacing.  Freely drawn times leave gaps of wildly varying size -- a gap over
# half the clip is common -- and at ten frames a long gap reads as the motion
# having stopped.  Jittering around even spacing keeps the times random while
# capping how large a gap can get.
TIME_JITTER = 0.30

# How many fingers one event moves.  Every count is reachable -- a single finger
# curling is as real as a whole hand closing -- with two and three weighted a
# little higher because those are the common ones.
FINGER_COUNT_WEIGHTS: dict[int, float] = {1: 0.15, 2: 0.25, 3: 0.25, 4: 0.20, 5: 0.15}

# How far each pose in an event is blended in, drawn per pose.  Below about a
# half the pose barely registers; at 1.0 the joint lands exactly on an authored
# pose, so drawing in between is what puts the keyframes at interior points of
# the pose space rather than only on its corners.
POSE_ALPHA = (0.5, 1.0)

# The parts every event sets.  The arm dominates what the camera sees -- moving
# it shifts the hand by ~78 mm against ~10 mm for all five fingers -- so an
# event that left it alone would barely change the picture.
BUNDLE_PARTS = ("arm", "forearm", "wrist")

# Arm poses are not equally worth drawing.  x1 is the side raise, which puts the
# hand outside the headset's view most of the time; x2 and x3 sit in front of the
# body, where the camera is actually looking.  Keyed by the token in the asset
# name because nothing in the bones says where the hand ends up -- so this is
# tuning tied to the current library's naming, and a pose whose name matches
# nothing here simply gets the default weight of one.
ARM_POSE_WEIGHTS: dict[str, float] = {"x1": 0.5, "x2": 1.5, "x3": 1.5, "x4": 1.0}

# How often the hand is set by one whole-hand pose instead of a few single-finger
# ones.  Both sit at the same level of the bundle -- either fills the hand slot,
# never both -- and an even split keeps the hand-authored shapes, which always
# look natural, as common as the independent finger draws, which cover far more
# of the pose space but can stack into shapes no real hand makes.
HAND_POSE_SHARE = 0.5


def _draw_weight(asset: PoseAsset) -> float:
    """How likely an asset is to be drawn, against its siblings on the same part."""
    if asset.part != "arm":
        return 1.0
    return next((w for token, w in ARM_POSE_WEIGHTS.items() if f"_{token}_" in asset.name), 1.0)


@dataclass
class PosePick:
    """One pose asset and how far it is blended in.

    *alpha* is a fraction, not a weight: applying a pick moves the bones it
    drives that fraction of the way from wherever they already are to the pose.
    1.0 lands exactly on the pose, 0.5 stops halfway.
    """

    asset: str
    alpha: float = 1.0

    def to_dict(self) -> dict:
        return {"asset": self.asset, "alpha": round(float(self.alpha), 3)}

    @classmethod
    def from_dict(cls, data: dict) -> PosePick:
        return cls(asset=str(data["asset"]), alpha=float(data.get("alpha", 1.0)))


@dataclass
class PoseEvent:
    """One configuration the body reaches at a point in time.

    An event is a *bundle*: the arm, the forearm, the wrist and some fingers all
    set at once, because each of those assets drives only its own joint and one
    alone barely changes the picture.  The picks are applied in order, each
    blended in by its own alpha.

    *group* is the body region the bundle drives: "left", "right", "head" or
    "body".  Groups touch disjoint bones, so events from different groups layer
    rather than replace one another.
    """

    t_ns: int
    poses: list[PosePick]
    group: str = "body"

    def to_dict(self) -> dict:
        return {
            "t_ns": int(self.t_ns),
            "group": self.group,
            "poses": [p.to_dict() for p in self.poses],
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseEvent:
        return cls(
            t_ns=int(data["t_ns"]),
            poses=[PosePick.from_dict(p) for p in data["poses"]],
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
            for p in e.poses:
                if p.asset not in seen:
                    seen.append(p.asset)
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
        lines = []
        for e in self.events:
            picks = "  ".join(f"{p.asset}@{p.alpha:.2f}" for p in e.poses)
            lines.append(f"  {e.t_ns / NS_PER_SECOND:6.2f}s  {e.group:<5} {picks}")
        return "\n".join(lines)


@dataclass
class PoseAsset:
    """A single-frame pose asset discovered inside a model .blend.

    *frame* is the frame its action holds the pose on, which is where the
    renderer evaluates it.

    *group* and *part* are both derived from the bones the asset animates rather
    than from its name, so renaming or refiling a pose keeps it working.  group
    is the side ("left", "right", "head", "body"); part is the joint it drives
    ("arm", "forearm", "wrist", "finger:index", ...).  Assets on different parts
    touch disjoint bones and combine freely, which is what turns a library of N
    poses into a much larger space of configurations.
    """

    name: str
    frame: int
    group: str = "body"
    part: str = "other"

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "frame": int(self.frame),
            "group": self.group,
            "part": self.part,
        }

    @classmethod
    def from_dict(cls, data: dict) -> PoseAsset:
        return cls(
            name=str(data["name"]),
            frame=int(data["frame"]),
            group=str(data.get("group", "body")),
            part=str(data.get("part", "other")),
        )


def _group_of(pool: list[PoseAsset], name: str) -> str | None:
    """Which side an asset belongs to, or None if it is not in *pool*."""
    return next((a.group for a in pool if a.name == name), None)


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

    Times were once the order statistics of uniform draws — a Poisson process
    conditioned on its event count.  That is the right model for arrivals, but
    its gaps vary wildly, and a gap covering half a ten-frame clip reads as the
    motion having stopped.  Events now sit on an even grid jittered by
    TIME_JITTER instead: still random, but with a bound on how large a gap gets.

    Two rules keep every limb moving, both enforced per body group rather than
    per lane: the first pose is keyed at t=0 and the last at duration_ns, and no
    pose is drawn twice running for the same group.  Per lane they would not be
    enough — a shared lane can repeat one hand's pose around the other hand's
    event, and only one group can own the event pinned to the end.  Times are
    jittered around even spacing rather than drawn freely, so no gap grows large
    enough to look like the motion stopped.

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

    def sample_lane(pool: list[PoseAsset]) -> list[PoseEvent]:
        """Draw one independent timeline of bundles from *pool*."""
        # An event sets one asset per part, so the assets are indexed that way.
        # The side is part of the key: a lane covering both hands must not build
        # a bundle out of a left arm and a right wrist.
        by_key: dict[tuple[str, str], list[PoseAsset]] = {}
        for a in pool:
            by_key.setdefault((a.group, a.part), []).append(a)
        groups = sorted({a.group for a in pool})

        previous: dict[tuple[str, str], str] = {}

        def pick(key: tuple[str, str]) -> PosePick:
            # Never draw the pose that part is already holding: it would key a
            # change that changes nothing.
            options = [a for a in by_key[key] if a.name != previous.get(key)] or by_key[key]
            chosen = rng.choices(options, weights=[_draw_weight(a) for a in options], k=1)[0]
            previous[key] = chosen.name
            return PosePick(asset=chosen.name, alpha=rng.uniform(*POSE_ALPHA))

        def bundle(group: str, force: str | None = None) -> list[PosePick]:
            keys = [(group, part) for part in BUNDLE_PARTS if (group, part) in by_key]

            # The hand slot: either a handful of single fingers, or one authored
            # whole-hand shape.  Whichever is available; a coin toss when both are.
            fingers = sorted(k for k in by_key if k[0] == group and k[1].startswith("finger:"))
            whole = (group, "hand") if (group, "hand") in by_key else None
            if whole is not None and (not fingers or rng.random() < HAND_POSE_SHARE):
                keys.append(whole)
            elif fingers:
                counts = [n for n in FINGER_COUNT_WEIGHTS if n <= len(fingers)]
                weights = [FINGER_COUNT_WEIGHTS[n] for n in counts]
                keys += rng.sample(fingers, rng.choices(counts, weights=weights, k=1)[0])
            if not keys:
                # A lane with no limb parts -- the head, or a whole-body pose --
                # is a single pose, as it was before parts existed.
                keys = sorted(k for k in by_key if k[0] == group)
            picks = [pick(k) for k in keys]
            if force is not None:
                picks = [PosePick(asset=force, alpha=1.0)] + [p for p in picks if p.asset != force]
            return picks

        # One event moves one side.  A lane covering a single group -- what
        # --hand gives each hand -- therefore yields exactly event_count events,
        # and a shared lane spreads that same count across the sides it covers.
        opening = rest_asset if any(a.name == rest_asset for a in pool) else None
        first = str(_group_of(pool, opening)) if opening else rng.choice(groups)
        out = [PoseEvent(t_ns=0, poses=bundle(first, force=opening), group=first)]

        # Interior events sit on an even grid, each nudged by up to TIME_JITTER
        # of one spacing.  The last is pinned rather than drawn: placed freely it
        # leaves the clip ending part way through, freezing the final pose for
        # the remainder — a dead tail averaging a quarter of a four-event clip.
        times: list[int] = []
        if event_count > 1:
            step = duration_ns / (event_count - 1)
            for i in range(1, event_count - 1):
                slot = (i + rng.uniform(-TIME_JITTER, TIME_JITTER)) * step
                times.append(int(round(min(duration_ns - 1, max(1, slot)))))
            times.sort()
            times.append(duration_ns)

        for t_ns in times:
            g = rng.choice(groups)
            out.append(PoseEvent(t_ns=t_ns, poses=bundle(g), group=g))

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
