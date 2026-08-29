# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import jsonschema
from PIL import Image, ImageDraw

import phanesim.validate as val
from phanesim.posemotion import (
    DEFAULT_DURATION_SECONDS,
    DEFAULT_EVENT_COUNT,
    NS_PER_SECOND,
    PoseAsset,
    sample_pose_motion,
)
from phanesim.skeleton import HAND_CONNECTIONS, LANDMARK_COLORS

# Parent directory of the phanesim package, added to sys.path inside Blender
# so that `import phanesim` works in the headless rendering subprocess.
_PKG_PARENT = str(Path(__file__).parent.parent)


# Short names the --accessories option accepts, kept in step with
# render.ACCESSORIES.  Duplicated here rather than imported because cli.py must
# work outside Blender, where render.py cannot be imported at all.
ACCESSORY_NAMES = ("ring1", "ring2", "watch1", "band1")


def _parse_accessories(value: str) -> str:
    """Turn the --accessories value into the argument to pass into Blender.

    Returns:
        A Python expression building the set of accessories to show.  It is
        always an explicit set, so a render wears exactly what was asked for
        rather than whatever the .blend happened to be saved with.
    """
    picked = {name.strip() for name in value.split(",") if name.strip()}
    if picked == {"all"}:
        picked = set(ACCESSORY_NAMES)
    elif picked == {"none"}:
        picked = set()
    unknown = picked - set(ACCESSORY_NAMES)
    if unknown:
        raise click.BadParameter(
            f"unknown accessory {sorted(unknown)}; choose from {', '.join(ACCESSORY_NAMES)}, all, none"
        )
    # sorted() so the expression is stable between runs.
    return f"set({sorted(picked)!r})"


def _build_lanes(hand: int | None, head: bool) -> tuple[tuple[tuple[str, ...], ...], int, int]:
    """Work out the timelines to sample, and how many events each one gets.

    Two ways of asking exist because they answer different questions.  The
    default is one shared timeline: "give me N poses", each of which happens to
    be a left hand, a right hand or a whole-body pose.  --hand instead gives each
    hand a timeline of its own, so both move on their own schedule and N means N
    poses *each*.

    Returns:
        (lanes, events per lane, total events).
    """
    if hand is not None:
        lanes: tuple[tuple[str, ...], ...] = (("left",), ("right",))
    else:
        # Whole-body poses share the hands' timeline rather than getting their
        # own: each keys both hands, so it is simply another thing that can
        # happen next, not something held alongside them.
        lanes = (("left", "right", "body"),)
    if head:
        lanes = (*lanes, ("head",))
    return lanes, (hand if hand is not None else 0), len(lanes)


SWEEP_DIRECTIONS = ("left", "right", "up", "down")


def _parse_camera(value: str | None) -> str:
    """Turn the --camera value "DIRECTION,DEGREES" into the argument to pass in.

    Returns:
        A Python expression: "None" for a camera that only follows the head, or
        a CameraSweep constructor call.
    """
    if value is None:
        return "None"
    try:
        direction, degrees_s = value.split(",", 1)
        degrees = float(degrees_s)
    except ValueError:
        raise click.BadParameter(f"expected DIRECTION,DEGREES, got {value!r}") from None
    direction = direction.strip().lower()
    if direction not in SWEEP_DIRECTIONS:
        raise click.BadParameter(f"direction must be one of {', '.join(SWEEP_DIRECTIONS)}, got {direction!r}")
    return f"CameraSweep({direction!r}, {degrees!r})"


def _find_blender(blender_bin: str | None) -> str:
    """Resolve the Blender executable path.

    Resolution order:
    1. Explicit --blender argument or BLENDER_BIN environment variable.
    2. Any binary named blender* found on PATH (e.g. blender,blender4,
       blender-4.3).
    3. Any shell alias whose name or target path contains "blender", resolved
       through bash so that aliases defined in ~/.bashrc are visible.
    4. Falls back to "blender" and lets the OS raise a clear error.
    """
    if blender_bin:
        return blender_bin

    # 1. Any blender* binary on PATH.
    for directory in os.environ.get("PATH", "").split(os.pathsep):
        try:
            for name in sorted(Path(directory).iterdir()):
                if name.stem.lower().startswith("blender") and os.access(name, os.X_OK):
                    return str(name)
        except OSError:
            pass

    # 2. Scan all shell aliases for anything whose name or target contains "blender".
    try:
        res = subprocess.run(
            ["bash", "-ic", "alias 2>/dev/null"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in res.stdout.splitlines():
            # Each line: alias blender5='/path/to/blender'
            if "blender" not in line.lower():
                continue
            if "=" not in line:
                continue
            alias_path = line.split("=", 1)[1].strip().strip("'`\"")
            if alias_path and Path(alias_path).is_file() and os.access(alias_path, os.X_OK):
                return alias_path
    except Exception:
        pass

    return "blender"


def _overlay_keypoints(output_path: Path) -> None:
    """Draw the 21-landmark skeleton on every rendered frame found under output_path.

    Reads each cam_*/joints_2d.csv, loads the matching frame_XXXXXX.png files, draws
    per-finger colored dots and skeleton lines, and saves frame_XXXXXX_debug.png.
    """
    _DOT_RADIUS = 3
    _LINE_WIDTH = 1

    for csv_path in sorted(output_path.rglob("joints_2d.csv")):
        cam_dir = csv_path.parent
        with csv_path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue

        # Column pairs: (u_col, v_col) for each of the 21 landmarks per hand.
        headers = list(rows[0].keys())  # timestamp + u/v pairs
        uv_pairs = [(headers[i], headers[i + 1]) for i in range(1, len(headers) - 1, 2)]

        # uv_pairs has 21 pairs per hand; derive hand count from total columns.
        n_hands = max(1, len(uv_pairs) // 21)

        frame_paths = sorted(cam_dir.glob("frame_??????.png"))
        written: list[str] = []
        for frame_path, row in zip(frame_paths, rows, strict=False):
            img = Image.open(frame_path).convert("RGB")
            draw = ImageDraw.Draw(img)

            for hand_idx in range(n_hands):
                offset = hand_idx * 21
                hand_uv: list[tuple[float, float] | None] = []

                for k in range(21):
                    u_col, v_col = uv_pairs[offset + k]
                    try:
                        u, v = float(row[u_col]), float(row[v_col])
                        hand_uv.append(None if (math.isnan(u) or math.isnan(v)) else (u, v))
                    except (ValueError, KeyError):
                        hand_uv.append(None)

                # Draw skeleton lines first (underneath dots).
                for a, b in HAND_CONNECTIONS:
                    pt_a, pt_b = hand_uv[a], hand_uv[b]
                    if pt_a is not None and pt_b is not None:
                        draw.line([pt_a, pt_b], fill=LANDMARK_COLORS[a], width=_LINE_WIDTH)

                # Draw dots on top.
                for k, pt in enumerate(hand_uv):
                    if pt is None:
                        continue
                    u, v = pt
                    r = _DOT_RADIUS
                    draw.ellipse([u - r - 1, v - r - 1, u + r + 1, v + r + 1], fill=(0, 0, 0))
                    draw.ellipse([u - r, v - r, u + r, v + r], fill=LANDMARK_COLORS[k])

            debug_path = frame_path.with_stem(frame_path.stem + "_debug")
            img.save(debug_path)
            written.append(debug_path.name)

        click.echo(f"[phanesim] Debug keypoints written to {cam_dir}:")
        for name in written:
            click.echo(f"  {name}")


def _sys_path_setup() -> str:
    """Python snippet that makes `import phanesim` work inside Blender."""
    return f"import sys; sys.path.insert(0, {_PKG_PARENT!r}); "


def _run_blender(expr: str, blender_bin: str | None) -> int:
    """Run *expr* in headless Blender and return its exit code."""
    blender = _find_blender(blender_bin)
    # LIBGL_ALWAYS_SOFTWARE=1: EEVEE Next requires a display for GPU Vulkan context
    # creation; no display is available in WSL2 headless mode. LLVMpipe (Mesa CPU
    # renderer) provides a valid EGL surfaceless context without a display.
    env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1"}
    result = subprocess.run([blender, "--background", "--factory-startup", "--python-expr", expr], env=env)
    return result.returncode


# Only the two kinds that exist as files of their own.  camera.json and
# body_rig.json are $ref'd from body_sequence.json, so validating a sequence
# already checks the rig and every camera in it.
VALIDATE_KINDS = (
    "body_sequence",
    "pose_motion",
)

_VALIDATE_FNS = {
    "body_sequence": val.validate_body_sequence,
    "pose_motion": val.validate_pose_motion,
}


@click.group()
def cli() -> None:
    """Phanesim command line interface."""


@cli.command()
@click.argument("kind", type=click.Choice(VALIDATE_KINDS, case_sensitive=False))
@click.argument("input_path", type=click.Path(path_type=Path))
def validate(kind: str, input_path: Path) -> None:
    """Validate a config file against its schema."""
    try:
        _VALIDATE_FNS[kind](input_path)
        click.echo(f"OK: {input_path} is a valid {kind}")
    except jsonschema.ValidationError as e:
        click.echo(f"Error: {input_path}: {e.message}", err=True)
        sys.exit(1)
    except (ValueError, OSError) as e:
        click.echo(f"Error: {input_path}: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory the rendered frames and annotation CSVs are written to.",
)
@click.option(
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable. Auto-detected if not set.",
)
@click.option(
    "--frames",
    default=None,
    type=int,
    help="How many frames to render, spread evenly across the whole motion. "
    "2 gives the first and last frame. Overrides the sequence's own 'frames'.",
)
@click.option(
    "--accessories",
    default="none",
    show_default=True,
    metavar="LIST",
    help="Which accessories the model wears: a comma-separated list of "
    "ring1 (right middle finger), ring2 (left ring finger), watch1 (left wrist), "
    "band1 (right wrist). Also accepts 'all'.",
)
@click.option(
    "--rotate",
    type=click.Choice(["0", "90", "180", "270"]),
    default="0",
    show_default=True,
    help="Turn the camera about its own optical axis, in degrees. 90 mounts the "
    "sensor on its side: the file is still 640x480, but the scene inside it is "
    "portrait, as on a headset with a rotated camera. The view itself does not "
    "change, only its orientation in the frame.",
)
@click.option(
    "--camera",
    "camera_sweep",
    default=None,
    metavar="DIRECTION,DEGREES",
    help="Turn the camera steadily as the clip runs, on top of whatever the head "
    "is doing: e.g. --camera right,30 starts at the rest view and ends 30 degrees "
    "to the right. Directions are left, right, up, down. "
    "Omit for a camera that only moves when the head does.",
)
@click.option(
    "--debug_kps",
    is_flag=True,
    default=False,
    help=(
        "After rendering, overlay the projected 21-landmark hand skeleton on each frame "
        "and save frame_XXXXXX_debug.png alongside the rendered images. Also writes "
        "joints_3d.csv with the world-space position and rotation of every joint."
    ),
)
def generate(
    input_path: Path,
    output_path: Path,
    blender_bin: str | None,
    frames: int | None,
    accessories: str,
    rotate: str,
    camera_sweep: str | None,
    debug_kps: bool,
) -> None:
    """Render a body sequence to PNG frames and joint annotations.

    Requires Blender to be installed. Set BLENDER_BIN or pass --blender to
    specify the executable if 'blender' is not on PATH.
    """
    input_abs = str(input_path.resolve())
    output_abs = str(output_path.resolve())

    cls_name, fn_name = "BodySequence", "render_body_sequence"
    extra = f", frames={frames!r}" if frames is not None else ""
    if debug_kps:
        # Ground-truth 3D joint poses are only worth the extra file when the
        # debug pass is asked for; the data itself is already in hand.
        extra += ", write_3d=True"
    extra += f", accessories={_parse_accessories(accessories)}, camera_sweep={_parse_camera(camera_sweep)}"
    extra += f", rotate={float(rotate)!r}"
    expr = (
        _sys_path_setup()
        + "from pathlib import Path; "
        + f"from phanesim.rig import {cls_name}; "
        + "from phanesim.types import CameraSweep; "
        + f"from phanesim.render import {fn_name}; "
        + f"{fn_name}({cls_name}.from_path(Path({input_abs!r})), Path({output_abs!r}){extra})"
    )

    returncode = _run_blender(expr, blender_bin)
    if returncode != 0:
        sys.exit(returncode)
    if debug_kps:
        _overlay_keypoints(output_path)
    sys.exit(0)


@cli.command(name="generate-motion")
@click.option(
    "--model",
    "model_path",
    type=click.Path(path_type=Path, exists=True),
    required=True,
    help="Path to the .blend holding the pose assets (e.g. data/models/model1/model1.blend).",
)
@click.option(
    "--output",
    "output_dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Directory the animation JSON files are written to.",
)
@click.option("--count", default=1, show_default=True, help="Number of animation descriptions to generate.")
@click.option(
    "--duration",
    default=DEFAULT_DURATION_SECONDS,
    show_default=True,
    help="Length of each animation in seconds.",
)
@click.option(
    "--events",
    "event_count",
    default=None,
    type=int,
    help=f"How many poses the clip contains, including the one at t=0. Each is drawn "
    f"from the left-hand, right-hand and whole-body poses together, so one timeline "
    f"covers them all.  [default: {DEFAULT_EVENT_COUNT}]",
)
@click.option(
    "--hand",
    default=None,
    type=int,
    help="Give each hand its own timeline of this many poses, so the two move on "
    "separate schedules and both are always posed. --hand 4 means 4 poses for the "
    "left hand and 4 for the right. Cannot be used together with --events.",
)
@click.option(
    "--head",
    is_flag=True,
    default=False,
    help="Also move the head, on a timeline of its own, so it turns while the hands "
    "are changing pose. The camera is anchored to the head, so this moves the camera "
    "and changes the background too.",
)
@click.option("--rest-asset", default=None, help="Pose to key at t=0. Defaults to a random one.")
@click.option(
    "--seed",
    default=None,
    type=int,
    help="Seed for the first animation; later ones count up from it. Omit it and a "
    "fresh seed is drawn, so running the same command again gives different motion. "
    "Every animation records the seed it was made with, so pass that value back to "
    "reproduce it exactly.",
)
@click.option("--prefix", default="animation", show_default=True, help="Basename of the generated files.")
@click.option(
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable. Auto-detected if not set.",
)
def generate_motion(
    model_path: Path,
    output_dir: Path,
    count: int,
    duration: float,
    event_count: int | None,
    hand: int | None,
    head: bool,
    rest_asset: str | None,
    seed: int | None,
    prefix: str,
    blender_bin: str | None,
) -> None:
    """Generate random pose motion descriptions from a model's pose assets.

    You say how many poses and over how long. Only which poses and when they land
    are random.

    \b
      --events 4 --duration 1   4 poses in 1 second, each of them a left-hand,
                                right-hand or whole-body pose
      --hand 4   --duration 1   4 poses for the left hand and 4 for the right,
                                on separate timelines, in 1 second
      --head                    add head movement on a timeline of its own

    A pose only ever keys its own bones, so poses for different parts of the body
    are held at the same time rather than replacing one another. That is why a
    library of 9 left and 13 right poses covers 117 configurations, not 22.

    Writes animation01.json, animation02.json, ... ; the descriptions hold no bone
    data, so the poses stay in the .blend until `phanesim generate body_sequence`
    runs.

    Blender is launched once to enumerate the pose assets, then all the timelines
    are sampled in-process.
    """
    with tempfile.TemporaryDirectory() as tmp:
        assets_json = Path(tmp) / "pose_assets.json"
        expr = (
            _sys_path_setup()
            + "from pathlib import Path; "
            + "from phanesim.render import enumerate_pose_assets; "
            + f"enumerate_pose_assets(Path({str(model_path.resolve())!r}), Path({str(assets_json)!r}))"
        )
        click.echo(f"[phanesim] Enumerating pose assets in {model_path.name} ...")
        if (code := _run_blender(expr, blender_bin)) != 0:
            sys.exit(code)
        if not assets_json.exists():
            click.echo("Error: Blender did not report any pose assets.", err=True)
            sys.exit(1)
        assets = [PoseAsset.from_dict(a) for a in json.loads(assets_json.read_text())["assets"]]

    if not assets:
        click.echo(f"Error: no asset-marked actions found in {model_path}.", err=True)
        sys.exit(1)

    if event_count is not None and hand is not None:
        raise click.BadParameter("--events and --hand ask for different things; pass one or the other")
    lanes, _, _ = _build_lanes(hand, head)
    per_lane = hand if hand is not None else (event_count if event_count is not None else DEFAULT_EVENT_COUNT)

    counts = {g: sum(1 for a in assets if a.group == g) for g in ("left", "right", "head", "body")}
    # A lane with no assets is skipped by the sampler, so it is not reported as a
    # timeline either -- model2 has no head poses, and --head there is a no-op.
    stocked = [lane for lane in lanes if any(counts[g] for g in lane)]
    for lane in stocked:
        drawn_from = "+".join(g for g in lane if counts[g])
        click.echo(f"    timeline: {drawn_from:<18} {sum(counts[g] for g in lane):>3} pose(s)")
    empty = [lane for lane in lanes if lane not in stocked]
    for lane in empty:
        click.echo(f"    no {'+'.join(lane)} poses in this model, so none are animated")
    used = {g for lane in stocked for g in lane}
    idle = sorted(g for g, n in counts.items() if n and g not in used)
    if idle:
        click.echo(f"    not animated: {', '.join(f'{g} ({counts[g]})' for g in idle)}")

    # Poses only ever key their own bones, so a left and a right pose are held at
    # the same time whichever timeline drew them.  That is what makes the library
    # multiply out instead of merely adding up.
    combinations = 1
    for group in ("left", "right", "head"):
        if group in used and counts[group]:
            combinations *= counts[group]
    click.echo(f"[phanesim] {combinations} distinct pose combinations available.")

    # Drawn once per run rather than per file, so the whole run is reproducible
    # from the single number printed below.
    base_seed = seed if seed is not None else random.randrange(2**31)
    click.echo(f"[phanesim] Seed {base_seed} -- pass --seed {base_seed} to reproduce this run.")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        name = f"{prefix}{i + 1:02d}"
        motion = sample_pose_motion(
            assets,
            duration_ns=int(duration * NS_PER_SECOND),
            name=name,
            event_count=per_lane,
            seed=base_seed + i,
            model=str(model_path),
            rest_asset=rest_asset,
            lanes=lanes,
        )
        out_path = output_dir / f"{name}.json"
        motion.write(out_path)
        click.echo(f"  {out_path}")
        click.echo(f"    {motion.event_count} poses over {duration:g} s  (seed={motion.seed})")
        click.echo(motion.summary())

    click.echo(f"[phanesim] Wrote {count} animation description(s) to {output_dir}")


@cli.command()
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option(
    "--output",
    "output_blend",
    type=click.Path(path_type=Path),
    default="preview.blend",
    show_default=True,
    help="Path for the saved .blend file.",
)
@click.option(
    "--frames",
    default=None,
    type=int,
    help="How many frames to render, spread evenly across the whole motion. "
    "2 gives the first and last frame. Overrides the sequence's own 'frames'.",
)
@click.option(
    "--accessories",
    default="none",
    show_default=True,
    metavar="LIST",
    help="Which accessories the model wears: a comma-separated list of "
    "ring1 (right middle finger), ring2 (left ring finger), watch1 (left wrist), "
    "band1 (right wrist). Also accepts 'all'.",
)
@click.option(
    "--rotate",
    type=click.Choice(["0", "90", "180", "270"]),
    default="0",
    show_default=True,
    help="Turn the camera about its own optical axis, in degrees. 90 mounts the "
    "sensor on its side: the file is still 640x480, but the scene inside it is "
    "portrait, as on a headset with a rotated camera. The view itself does not "
    "change, only its orientation in the frame.",
)
@click.option(
    "--camera",
    "camera_sweep",
    default=None,
    metavar="DIRECTION,DEGREES",
    help="Turn the camera steadily as the clip runs, on top of whatever the head "
    "is doing: e.g. --camera right,30 starts at the rest view and ends 30 degrees "
    "to the right. Directions are left, right, up, down. "
    "Omit for a camera that only moves when the head does.",
)
@click.option(
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable. Auto-detected if not set.",
)
def preview(
    input_path: Path,
    output_blend: Path,
    frames: int | None,
    accessories: str,
    rotate: str,
    camera_sweep: str | None,
    blender_bin: str | None,
) -> None:
    """Bake a sequence as keyframes and save the result as a .blend file.

    Runs Blender headlessly to bake the animation, then prints the path to the
    saved .blend file.  Open it manually in Blender to scrub the timeline and
    inspect the compositor nodes interactively.
    Set BLENDER_BIN or pass --blender to specify the executable.
    """
    input_abs = str(input_path.resolve())
    blend_out = str(Path(output_blend).resolve())

    cls_name, fn_name = "BodySequence", "preview_body_sequence"
    pv_extra = f", frames={frames!r}" if frames is not None else ""
    pv_extra += f", accessories={_parse_accessories(accessories)}, camera_sweep={_parse_camera(camera_sweep)}"
    pv_extra += f", rotate={float(rotate)!r}"
    expr = (
        _sys_path_setup()
        + "from pathlib import Path; "
        + f"from phanesim.rig import {cls_name}; "
        + "from phanesim.types import CameraSweep; "
        + f"from phanesim.render import {fn_name}; "
        + f"{fn_name}({cls_name}.from_path(Path({input_abs!r})), {blend_out!r}{pv_extra})"
    )

    click.echo("Baking keyframes (headless)...")
    if (code := _run_blender(expr, blender_bin)) != 0:
        sys.exit(code)

    click.echo(f"Preview saved: {blend_out}")


def main() -> None:
    cli(prog_name="phanesim")


if __name__ == "__main__":
    main()
