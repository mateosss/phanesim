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
from collections.abc import Callable
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
# body, where the camera is actually looking.  x4 reaches across the midline, and
# measured over rendered clips it costs about a fifth of the visibility of x2/x3
# at the same height (60% and 67% at y1 and y2, against 88-100%), so it sits
# between them and the side raise.  Keyed by the token in the asset name because
# nothing in the bones says where the hand ends up -- so this is tuning tied to
# the current library's naming, and a pose whose name matches nothing here simply
# gets the default weight of one.
ARM_POSE_WEIGHTS: dict[str, float] = {"x1": 0.5, "x2": 1.5, "x3": 1.5, "x4": 0.7}

# How often the hand is set by one whole-hand pose instead of a few single-finger
# ones.  Both sit at the same level of the bundle -- either fills the hand slot,
# never both -- and an even split keeps the hand-authored shapes, which always
# look natural, as common as the independent finger draws, which cover far more
# of the pose space but can stack into shapes no real hand makes.
HAND_POSE_SHARE = 0.5

# Which arm configurations each head pose can see, as (upper-arm x tokens,
# effective y tokens) allowed on the side the head turned toward ("same") and on
# the other side ("opp"); "both" for the poses that face straight ahead.
#
# The camera is bolted to the head bone, so the head *is* the camera: a head that
# looks away from where the arms are yields a frame with no hand in it, however
# good the hand pose was.  Drawn independently that is what happened -- 41% of
# dataset_test3's frames hold no hand at all, and a clip's outcome was decided by
# whether its head draws happened to match its arm draws.  Weighing head poses
# against this table is what closes that gap.
#
# Like ARM_POSE_WEIGHTS this is tuning tied to the current library's naming, and
# it is deliberately coarse: it answers "could that hand be in frame at all",
# not where in the frame it lands.  A head pose named nothing it knows is treated
# as looking ahead.
_HEAD_VIEW_RAW: dict[str, dict[str, tuple[str, str]]] = {
    # Facing ahead, at eye level, with the camera aimed 31 degrees down.  That
    # aim is what makes this row as wide as it is: at the 15 degrees it used to
    # be, a level head reached a hanging arm only 40% of the time, so y3 was not
    # in this row at all.  Re-measured by re-projecting recorded clips through
    # the steeper camera, y3 comes to 0.58-0.72 and x1 to 0.62-0.77, which is the
    # same band as the rest -- the hands sit low and close, so pointing at them
    # brings nearly the whole reach into view.  Retune this row if that aim moves
    # again; the two are not independent.  Re-checked after the lens was narrowed
    # back to fx 240: every cell here drops a little (x2 to 0.86/0.85/0.67) but
    # none falls out, so the row stands at either focal length.
    #
    # Only the two frontal reaches are listed even though x1 and x4 measure not
    # far behind them (0.62-0.81 against 0.58-0.88).  The table is read as a
    # yes/no and its whole use is to tell head poses apart; listing every arm
    # here makes a level head the answer to everything, which measurably stops
    # the head following the hands at all and stops the two hands being drawn
    # towards each other.  So the frontal reaches sit here, and the two that need
    # more help to be seen are what a turned head is worth drawing for.
    "default": {"both": ("x2 x3", "y1 y2 y3")},
    "lean_left": {"both": ("x2 x3", "y1 y2 y3")},  # a roll, so it only tilts the picture
    "lean_right": {"both": ("x2 x3", "y1 y2 y3")},
    # Pitched down, which is what does reach the hanging arm.
    "down": {"both": ("x2 x3 x4", "y2 y3")},
    "moredown": {"both": ("x2 x3 x4", "y3")},
    # Pitched up, so only a raised arm reaches the view.
    "up": {"both": ("x2 x3 x4", "y1")},
    "farup": {"both": ("x2 x3 x4", "y1")},
    # Turned a little: the near side additionally picks up the arm reaching
    # across the body, and the far one keeps the frontal reaches.
    "lessleft": {"same": ("x2 x3 x4", "y1 y2 y3"), "opp": ("x2 x3", "y2 y3")},
    "lessright": {"same": ("x2 x3 x4", "y1 y2 y3"), "opp": ("x2 x3", "y2 y3")},
    # Turned ~40 degrees: the side raise on the near side has swung in, and only
    # an arm crossing the body still reaches across to the far one.
    "left": {"same": ("x1 x2 x3 x4", "y2 y3"), "opp": ("x4", "y2")},
    "right": {"same": ("x1 x2 x3 x4", "y2 y3"), "opp": ("x4", "y2")},
    # Past the half-angle of the lens: nothing in front of the body is left, and
    # only the side raise on that same side has swung into view.
    "farleft": {"same": ("x1", "y1 y2")},
    "farright": {"same": ("x1", "y1 y2")},
    "leftup": {"same": ("x2 x3", "y1")},
    "rightup": {"same": ("x2 x3", "y1")},
    "left_down": {"same": ("x2 x3", "y3")},
    "right_down": {"same": ("x2 x3", "y3")},
}

HEAD_VIEW: dict[str, dict[str, tuple[frozenset[str], frozenset[str]]]] = {
    key: {rel: (frozenset(xs.split()), frozenset(ys.split())) for rel, (xs, ys) in rules.items()}
    for key, rules in _HEAD_VIEW_RAW.items()
}

# How much a head pose is worth drawing before the hands are considered.  The
# mild poses carry the timeline because they are the ones a hand in front of the
# body is visible under at all; the extremes keep a weight of one so they stay
# reachable, since a hand entering at the edge of the frame is exactly what a
# detector has to learn.
HEAD_BASE_WEIGHTS: dict[str, float] = {
    "default": 2.0,
    "lessleft": 1.8,
    "lessright": 1.8,
    "left": 1.5,
    "right": 1.5,
    "lean_left": 1.8,
    "lean_right": 1.8,
}
DEFAULT_HEAD_WEIGHT = 1.0

# What a head pose's weight is multiplied by, keyed by how many hands it would
# leave in frame.  The zero entry sets the empty-frame rate: not zero, because a
# detector needs frames with nothing to find, but small, because every empty
# frame is a rendered frame the keypoint stage learns nothing from.
HEAD_VIEW_WEIGHTS: dict[int, float] = {0: 0.06, 1: 2.0, 2: 3.0}

# How many points across the stretch a head pose is held are scored against the
# hands.  One point -- the instant the pose is keyed -- lets the hands wander off
# again before the head next moves, since the lanes keep separate schedules.
HEAD_LOOKAHEAD = 8

# How far after the hand move the head is keyed, as a fraction of the average gap
# between hand moves.  Zero would have the head arrive exactly as the hand does,
# which reads as the head knowing where the hand is going; a small lag is what
# following it looks like, and still lands well inside the same frame or the next.
HEAD_FOLLOW_LAG = 0.15

# Letting the head redraw the pose it already holds was tried here and dropped:
# it gained 0.1 points of empty-frame rate and cost the invariant that no group
# ever keys the same pose twice running, which is what stops a stretch of frames
# from being identical.

# How often the second hand is drawn somewhere the first one can be seen from
# too, and by how much such a place is favoured when it is.  A single head pose
# holding both hands is the whole of what makes a two-hand frame.
#
# Kept at a half rather than raised: independent draws already land somewhere one
# head pose covers 71% of the time, and this takes it to 78%.  It was worth much
# more when the view table was narrower (40% to 55%), and most of that job has
# since moved into the table and the camera's aim.  What is left to protect is
# the other end -- frames where one hand is alone are the common case in real
# headset footage, and a detector needs them or it learns that hands come in
# pairs.
ARM_COUPLE_SHARE = 0.5
ARM_COUPLE_BOOST = 6.0

# What an "up" forearm is worth against the nine that leave it straighter, on a
# lowered upper arm and on any other.
#
# Folding the forearm vertical is the raise-your-hand pose only from a lowered
# upper arm; on a level or raised one it puts the hand above anything the camera
# can see, so there it is all but barred.  On a lowered one it is the opposite --
# it is what rescues the pose.  A y3 arm left straight was seen 29% of the time
# in rendered clips; folded up, 75%.  So it is favoured rather than merely
# allowed: two "up" assets at this weight against nine others take about half the
# draws, which leaves the hanging-arm pose reachable without it being the norm.
FOREARM_UP_ON_LOWERED_ARM = 4.0
FOREARM_UP_MISMATCH = 0.02


def _draw_weight(asset: PoseAsset) -> float:
    """How likely an asset is to be drawn, against its siblings on the same part."""
    if asset.part != "arm":
        return 1.0
    return next((w for token, w in ARM_POSE_WEIGHTS.items() if f"_{token}_" in asset.name), 1.0)


def _head_key(name: str) -> str:
    """The part of a head asset's name that HEAD_VIEW is keyed by."""
    _, _, suffix = name.partition("_")
    return (suffix or name).lower()


def _arm_tokens(name: str) -> tuple[str, str] | None:
    """The (x, y) tokens of an *arm* asset's name, or None if it has neither.

    Only meaningful on assets whose part is "arm": finger assets carry x and y
    tokens of their own that mean something else entirely.
    """
    parts = name.lower().split("_")
    x = next((p for p in parts if len(p) == 2 and p[0] == "x" and p[1].isdigit()), None)
    y = next((p for p in parts if len(p) == 2 and p[0] == "y" and p[1].isdigit()), None)
    return (x, y) if x is not None and y is not None else None


def _is_forearm_up(name: str) -> bool:
    """Whether a forearm asset is one of the folded-vertical ones."""
    return "_up_" in name.lower()


def _effective_y(y: str, forearm_up: bool) -> str | None:
    """The height the hand actually ends up at, once the forearm is accounted for.

    Folding the forearm vertical lifts the hand back up: from a lowered upper arm
    (y3) it lands at about mid height, which is the raise-your-hand pose.  From a
    level or already raised arm it goes above the view entirely, which is what
    None means.
    """
    if not forearm_up:
        return y
    return "y2" if y == "y3" else None


def _view_sees(key: str, side: str, x: str, y: str) -> bool:
    """Whether *side*'s hand at (x, y) is in frame under the HEAD_VIEW entry *key*.

    Takes the table's own key, not an asset name: _head_key would read the key
    "lean_left" as "left" and answer for a pose that is not the one asked about.
    """
    rules = HEAD_VIEW.get(key)
    if rules is None:
        rules, key = HEAD_VIEW["default"], "default"
    if "both" in rules:
        allowed = rules["both"]
    else:
        turned = "left" if "left" in key else "right"
        allowed = rules.get("same" if turned == side else "opp")
        if allowed is None:
            return False
    xs, ys = allowed
    return x in xs and y in ys


def _head_sees(head_name: str, side: str, x: str, y: str) -> bool:
    """Whether *side*'s hand at (x, y) is in frame while the head holds a pose."""
    return _view_sees(_head_key(head_name), side, x, y)


def _reachable_heights(y: str) -> tuple[str, ...]:
    """The heights an arm drawn at *y* can end up at, once the forearm is drawn.

    A lowered arm is followed by a folded-up forearm about half the time, which
    lifts the hand to mid height, so a y3 draw is a coin toss between the two.
    """
    return ("y3", "y2") if y == "y3" else (y,)


def _seen_together(side: str, arm: tuple[str, str], other_side: str, other: tuple[str, str]) -> bool:
    """Whether one head pose exists that holds both hands in frame at once.

    Both arms in the same head's view is the whole of what makes a two-hand
    frame, and drawn independently the two rarely land somewhere one pose covers.
    """
    if arm[0] == "x4" and other[0] == "x4":
        # Both arms reaching across the body's midline would pass through each
        # other.  Rare enough on independent draws to leave alone; not something
        # to go out of the way to produce.
        return False
    return any(
        any(_view_sees(key, side, arm[0], y) for y in _reachable_heights(arm[1]))
        and _view_sees(key, other_side, *other)
        for key in HEAD_VIEW
    )


def _arm_state(events: list[PoseEvent], group: str, t_ns: int, part_of: dict[str, str]) -> tuple[str, str] | None:
    """Where *group*'s hand is at *t_ns*, as (x token, effective y token).

    Read out of events already sampled on another lane: the pose reached most
    recently at or before *t_ns*, or that group's opening pose if *t_ns* falls
    before it.  Blending between two events is ignored -- the table is coarse
    enough that the pose being held is the right approximation.

    None when the group has no events, keys no arm, or holds a combination that
    puts the hand out of every view.
    """
    same = sorted((e for e in events if e.group == group), key=lambda e: e.t_ns)
    if not same:
        return None
    reached = [e for e in same if e.t_ns <= t_ns]
    event = reached[-1] if reached else same[0]

    tokens: tuple[str, str] | None = None
    forearm_up = False
    for p in event.poses:
        part = part_of.get(p.asset)
        if part == "arm":
            tokens = _arm_tokens(p.asset) or tokens
        elif part == "forearm" and _is_forearm_up(p.asset):
            forearm_up = True
    if tokens is None:
        return None
    y = _effective_y(tokens[1], forearm_up)
    return None if y is None else (tokens[0], y)


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

    part_of = {a.name: a.part for a in assets}

    def sample_lane(
        pool: list[PoseAsset], prior: list[PoseEvent], count: int, follow: list[int] | None = None
    ) -> list[PoseEvent]:
        """Draw one independent timeline of bundles from *pool*.

        *prior* holds the events sampled on the lanes that ran before this one.
        Lanes still get their own schedules, but two draws read it: the head, to
        weigh each pose by how many hands it would leave in frame, and the second
        hand, to sometimes match the height the first is already at.  Both need
        the hands placed first, which is why the head lane is sampled last.
        """
        # An event sets one asset per part, so the assets are indexed that way.
        # The side is part of the key: a lane covering both hands must not build
        # a bundle out of a left arm and a right wrist.
        by_key: dict[tuple[str, str], list[PoseAsset]] = {}
        for a in pool:
            by_key.setdefault((a.group, a.part), []).append(a)
        groups = sorted({a.group for a in pool})

        previous: dict[tuple[str, str], str] = {}

        def pick(key: tuple[str, str], weight: Callable[[PoseAsset], float] | None = None) -> PosePick:
            # Never draw the pose that part is already holding: it would key a
            # change that changes nothing.
            options = [a for a in by_key[key] if a.name != previous.get(key)] or by_key[key]
            weights = [(weight or _draw_weight)(a) for a in options]
            if not any(w > 0.0 for w in weights):
                # Every option was gated out; fall back to an even draw rather
                # than failing, since a lane must still produce an event.
                weights = [1.0] * len(options)
            chosen = rng.choices(options, weights=weights, k=1)[0]
            previous[key] = chosen.name
            return PosePick(asset=chosen.name, alpha=rng.uniform(*POSE_ALPHA))

        def head_weight(t_ns: int, until_ns: int) -> Callable[[PoseAsset], float]:
            """Weigh head poses by how many hands each leaves in frame while held.

            Scored across the whole stretch the pose is held rather than at the
            instant it is keyed.  The hands are on their own schedule and move
            during that stretch, so a head aimed only at where they were when it
            was drawn loses them again a frame or two later -- which is where
            most of the remaining empty frames came from.
            """
            span = [t_ns + (until_ns - t_ns) * i // (HEAD_LOOKAHEAD - 1) for i in range(HEAD_LOOKAHEAD)]
            arms = [{side: _arm_state(prior, side, t, part_of) for side in ("left", "right")} for t in span]

            def weight(asset: PoseAsset) -> float:
                base = HEAD_BASE_WEIGHTS.get(_head_key(asset.name), DEFAULT_HEAD_WEIGHT)
                seen = [
                    sum(1 for side, st in at.items() if st is not None and _head_sees(asset.name, side, *st))
                    for at in arms
                ]
                mean = sum(seen) / len(seen)
                # Interpolate between the whole-hand-count weights, since a pose
                # that holds one hand for half the stretch sits between them.
                low, high = int(mean), min(2, int(mean) + 1)
                view = HEAD_VIEW_WEIGHTS[low] + (HEAD_VIEW_WEIGHTS[high] - HEAD_VIEW_WEIGHTS[low]) * (mean - low)
                return base * view

            return weight

        def arm_weight(group: str, t_ns: int) -> Callable[[PoseAsset], float] | None:
            """Favour somewhere the other hand can be seen from too, some of the time.

            Matching the other hand's height alone was not enough: two arms at one
            height still land on opposite sides of the body, where no single head
            pose holds both.  Asking the table directly -- is there a head pose
            that sees this candidate *and* the other hand -- couples the direction
            and the height together, and is the whole of what makes a frame with
            two hands in it.
            """
            if rng.random() >= ARM_COUPLE_SHARE:
                return None
            other_side = "right" if group == "left" else "left"
            other = _arm_state(prior, other_side, t_ns, part_of)
            if other is None:
                return None

            def weight(asset: PoseAsset) -> float:
                base = _draw_weight(asset)
                tokens = _arm_tokens(asset.name)
                if tokens is None:
                    return base
                return base * ARM_COUPLE_BOOST if _seen_together(group, tokens, other_side, other) else base

            return weight

        def bundle(group: str, t_ns: int, until_ns: int, force: str | None = None) -> list[PosePick]:
            limbs = [(group, part) for part in BUNDLE_PARTS if (group, part) in by_key]
            picks: list[PosePick] = []

            # The limb chain is drawn shoulder outwards so the forearm can be
            # weighed against the arm the same bundle just landed on: folding it
            # vertical only reads as a pose from a lowered upper arm.
            arm_y: str | None = None
            for key in limbs:
                if key[1] == "arm":
                    p = pick(key, arm_weight(group, t_ns))
                    tokens = _arm_tokens(p.asset)
                    arm_y = tokens[1] if tokens else None
                elif key[1] == "forearm":
                    held = arm_y

                    def forearm(asset: PoseAsset, held: str | None = held) -> float:
                        if not _is_forearm_up(asset.name):
                            return 1.0
                        return FOREARM_UP_ON_LOWERED_ARM if held == "y3" else FOREARM_UP_MISMATCH

                    p = pick(key, forearm)
                else:
                    p = pick(key)
                picks.append(p)

            # The hand slot: either a handful of single fingers, or one authored
            # whole-hand shape.  Whichever is available; a coin toss when both are.
            keys: list[tuple[str, str]] = []
            fingers = sorted(k for k in by_key if k[0] == group and k[1].startswith("finger:"))
            whole = (group, "hand") if (group, "hand") in by_key else None
            if whole is not None and (not fingers or rng.random() < HAND_POSE_SHARE):
                keys.append(whole)
            elif fingers:
                counts = [n for n in FINGER_COUNT_WEIGHTS if n <= len(fingers)]
                weights = [FINGER_COUNT_WEIGHTS[n] for n in counts]
                keys += rng.sample(fingers, rng.choices(counts, weights=weights, k=1)[0])
            if not limbs and not keys:
                # A lane with no limb parts -- the head, or a whole-body pose --
                # is a single pose, as it was before parts existed.
                weight = head_weight(t_ns, until_ns) if group == "head" else None
                picks = [pick(k, weight) for k in sorted(k for k in by_key if k[0] == group)]
            picks += [pick(k) for k in keys]
            if force is not None:
                picks = [PosePick(asset=force, alpha=1.0)] + [p for p in picks if p.asset != force]
            return picks

        # Interior events sit on an even grid, each nudged by up to TIME_JITTER
        # of one spacing.  The last is pinned rather than drawn: placed freely it
        # leaves the clip ending part way through, freezing the final pose for
        # the remainder — a dead tail averaging a quarter of a four-event clip.
        # Drawn before any event is built, because a pose is scored against the
        # stretch it will be held, which runs to the event after it.
        times: list[int] = []
        if follow:
            # The head is keyed just after each hand move rather than on a grid
            # of its own.  On an independent grid a hand can change the instant
            # after the head is keyed and go unseen until the head next moves,
            # which is the floor the weighting alone could not get under.  A
            # person following their own hands re-aims the same way.
            gap = duration_ns / max(1, len(follow))
            lagged = {int(t + HEAD_FOLLOW_LAG * gap) for t in follow if t > 0}
            # Anything the lag carries to the end is dropped rather than clamped:
            # the pinned final event already covers that stretch, and a second key
            # a nanosecond before it would snap the head inside one frame.
            times = sorted(t for t in lagged if 0 < t < duration_ns)
            times.append(duration_ns)
        elif count > 1:
            step = duration_ns / (count - 1)
            for i in range(1, count - 1):
                slot = (i + rng.uniform(-TIME_JITTER, TIME_JITTER)) * step
                times.append(int(round(min(duration_ns - 1, max(1, slot)))))
            times.sort()
            times.append(duration_ns)
        held_until = [*times, duration_ns]

        # One event moves one side.  A lane covering a single group -- what
        # --hand gives each hand -- therefore yields exactly event_count events,
        # and a shared lane spreads that same count across the sides it covers.
        opening = rest_asset if any(a.name == rest_asset for a in pool) else None
        first = str(_group_of(pool, opening)) if opening else rng.choice(groups)
        out = [PoseEvent(t_ns=0, poses=bundle(first, 0, held_until[0], force=opening), group=first)]

        for i, t_ns in enumerate(times):
            g = rng.choice(groups)
            out.append(PoseEvent(t_ns=t_ns, poses=bundle(g, t_ns, held_until[i + 1]), group=g))

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

    # Hands first, head last.  The head lane weighs every pose by how many hands
    # it would leave in frame, and the second hand sometimes matches the first's
    # height, so both need the lanes they read to have been sampled already.
    order = sorted(range(len(pools)), key=lambda i: "head" in lanes[i])

    events: list[PoseEvent] = []
    for i in order:
        if not pools[i]:
            continue
        # The head follows the hands' schedule when there is one to follow, which
        # also gives it an event per hand move rather than per pair of them, and
        # falls back to a grid of its own when the head is animated alone.
        follow = sorted({e.t_ns for e in events}) if "head" in lanes[i] and events else None
        events.extend(sample_lane(pools[i], list(events), event_count, follow))
    events.sort(key=lambda e: e.t_ns)

    return PoseMotion(
        name=name,
        events=events,
        duration_ns=duration_ns,
        model=model,
        seed=seed,
    )
