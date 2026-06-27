# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import csv
import math
import os
import subprocess
import sys
from pathlib import Path

import click
import jsonschema
from PIL import Image, ImageDraw

import phanesim.validate as val
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

        click.echo(f"[phanesim] Debug keypoints: {len(frame_paths)} frame(s) in {cam_dir}")


VALIDATE_KINDS = (
    "camera",
    "camera_motion",
    "hand_motion",
    "hand",
    "camhand_rig",
    "sequence",
    "project",
)

GENERATE_KINDS = (
    "sequence",
    "project",
)

PREVIEW_KINDS = ("sequence",)

_VALIDATE_FNS = {
    "camera": val.validate_camera,
    "camera_motion": val.validate_camera_motion,
    "hand_motion": val.validate_hand_motion,
    "hand": val.validate_hand,
    "camhand_rig": val.validate_camhand_rig,
    "sequence": val.validate_sequence,
    "project": val.validate_project,
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
    "--debug_kps",
    is_flag=True,
    default=False,
    help=(
        "After rendering, overlay the projected 21-landmark hand skeleton on each frame "
        "and save frame_XXXXXX_debug.png alongside the rendered images. "
        "Note: keypoint positions are based on camera intrinsics only and do not account "
        "for the compositor barrel distortion, so the overlay is approximate."
    ),
)
def generate(
    kind: str,
    input_path: Path,
    output_path: Path,
    blender_bin: str | None,
    debug_kps: bool,
) -> None:
    """Render a sequence or project by driving Blender headlessly.

    Requires Blender to be installed. Set BLENDER_BIN or pass --blender to
    specify the executable if 'blender' is not on PATH.
    """
    input_abs = str(input_path.resolve())
    output_abs = str(output_path.resolve())

    sys_path_setup = f"import sys; sys.path.insert(0, {_PKG_PARENT!r}); "
    if kind == "sequence":
        expr = (
            sys_path_setup
            + "from pathlib import Path; "
            + "from phanesim.rig import Sequence; "
            + "from phanesim.render import render_sequence; "
            + f"render_sequence(Sequence.from_path(Path({input_abs!r})), Path({output_abs!r}))"
        )
    else:
        expr = (
            sys_path_setup
            + "from pathlib import Path; "
            + "from phanesim.rig import Project; "
            + "from phanesim.render import render_project; "
            + f"render_project(Project.from_path(Path({input_abs!r})), Path({output_abs!r}))"
        )

    blender = _find_blender(blender_bin)
    # LIBGL_ALWAYS_SOFTWARE=1: EEVEE Next requires a display for GPU Vulkan context
    # creation; no display is available in WSL2 headless mode. LLVMpipe (Mesa CPU
    # renderer) provides a valid EGL surfaceless context without a display.
    env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1"}
    result = subprocess.run([blender, "--background", "--factory-startup", "--python-expr", expr], env=env)
    if result.returncode != 0:
        sys.exit(result.returncode)
    if debug_kps:
        _overlay_keypoints(output_path)
    sys.exit(0)


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
    "--blender",
    "blender_bin",
    default=None,
    envvar="BLENDER_BIN",
    show_envvar=True,
    help="Path to the Blender executable (default: 'blender' on PATH).",
)
def preview(kind: str, input_path: Path, output_blend: Path, blender_bin: str | None) -> None:
    """Bake a sequence as keyframes and save the result as a .blend file.

    Runs Blender headlessly to bake the animation, then prints the path to the
    saved .blend file.  Open it manually in Blender to scrub the timeline and
    inspect the compositor nodes interactively.
    Set BLENDER_BIN or pass --blender to specify the executable.
    """
    input_abs = str(input_path.resolve())
    blend_out = str(Path(output_blend).resolve())

    sys_path_setup = f"import sys; sys.path.insert(0, {_PKG_PARENT!r}); "
    expr = (
        sys_path_setup
        + "from pathlib import Path; "
        + "from phanesim.rig import Sequence; "
        + "from phanesim.render import preview_sequence; "
        + f"preview_sequence(Sequence.from_path(Path({input_abs!r})), {blend_out!r})"
    )

    blender = _find_blender(blender_bin)
    headless_env = {**os.environ, "LIBGL_ALWAYS_SOFTWARE": "1"}

    click.echo("Baking keyframes (headless)...")
    r = subprocess.run(
        [blender, "--background", "--factory-startup", "--python-expr", expr],
        env=headless_env,
    )
    if r.returncode != 0:
        sys.exit(r.returncode)

    click.echo(f"Preview saved: {blend_out}")


def main() -> None:
    cli(prog_name="phanesim")


if __name__ == "__main__":
    main()
