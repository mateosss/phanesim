# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import json
import math
import os
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
    DEFAULT_SEED,
    NS_PER_SECOND,
    PoseAsset,
    sample_pose_motion,
)
from phanesim.skeleton import HAND_CONNECTIONS, LANDMARK_COLORS

# Parent directory of the phanesim package, added to sys.path inside Blender
# so that `import phanesim` works in the headless rendering subprocess.
_PKG_PARENT = str(Path(__file__).parent.parent)


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


VALIDATE_KINDS = (
    "camera",
    "body_rig",
    "body_sequence",
    "pose_motion",
)

GENERATE_KINDS = ("body_sequence",)

PREVIEW_KINDS = ("body_sequence",)

_VALIDATE_FNS = {
    "camera": val.validate_camera,
    "body_rig": val.validate_body_rig,
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
@click.argument("kind", type=click.Choice(GENERATE_KINDS, case_sensitive=False))
@click.argument("input_path", type=click.Path(path_type=Path))
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable. Auto-detected as 'blender5' or 'blender' if not set.",
)
@click.option(
    "--frames",
    default=None,
    type=int,
    help="How many frames to render, spread evenly across the whole motion. "
    "2 gives the first and last frame. Overrides the sequence's own 'frames'.",
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
    kind: str,
    input_path: Path,
    output_path: Path,
    blender_bin: str | None,
    frames: int | None,
    debug_kps: bool,
) -> None:
    """Render a sequence or project by driving Blender headlessly.

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
    expr = (
        _sys_path_setup()
        + "from pathlib import Path; "
        + f"from phanesim.rig import {cls_name}; "
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
    default=DEFAULT_EVENT_COUNT,
    show_default=True,
    help="Exact number of poses per animation, including the one at t=0.",
)
@click.option("--rest-asset", default=None, help="Pose to key at t=0. Defaults to a random one.")
@click.option(
    "--seed",
    default=DEFAULT_SEED,
    show_default=True,
    type=int,
    help="Seed for the first animation; later ones increment from it. "
    "Generation is reproducible by default; pass a different seed for a different timeline.",
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
    event_count: int,
    rest_asset: str | None,
    seed: int,
    prefix: str,
    blender_bin: str | None,
) -> None:
    """Generate random pose motion descriptions from a model's pose assets.

    You say how many poses and over how long — four poses in one second, ten in
    twenty seconds — and only which poses and when they land are random.  Writes
    animation01.json, animation02.json, ... ; the descriptions hold no bone data,
    so the poses stay in the .blend until `phanesim generate body_sequence` runs.

    Blender is launched once to enumerate the pose assets, then all the
    timelines are sampled in-process.
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

    poses = [a for a in assets if not a.is_action]
    actions = [a for a in assets if a.is_action]
    click.echo(f"[phanesim] Found {len(poses)} pose(s) and {len(actions)} animation(s).")

    output_dir.mkdir(parents=True, exist_ok=True)
    for i in range(count):
        name = f"{prefix}{i + 1:02d}"
        motion = sample_pose_motion(
            assets,
            duration_ns=int(duration * NS_PER_SECOND),
            name=name,
            event_count=event_count,
            seed=seed + i,
            model=str(model_path),
            rest_asset=rest_asset,
        )
        out_path = output_dir / f"{name}.json"
        motion.write(out_path)
        click.echo(f"  {out_path}")
        click.echo(f"    {motion.event_count} poses over {duration:g} s  (seed={motion.seed})")
        click.echo(motion.summary())

    click.echo(f"[phanesim] Wrote {count} animation description(s) to {output_dir}")


@cli.command()
@click.argument("kind", type=click.Choice(PREVIEW_KINDS, case_sensitive=False))
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
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable (default: 'blender' on PATH).",
)
def preview(kind: str, input_path: Path, output_blend: Path, frames: int | None, blender_bin: str | None) -> None:
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
    expr = (
        _sys_path_setup()
        + "from pathlib import Path; "
        + f"from phanesim.rig import {cls_name}; "
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
