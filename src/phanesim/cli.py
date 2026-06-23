# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import click
import jsonschema

import phanesim.validate as val

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
def generate(kind: str, input_path: Path, output_path: Path, blender_bin: str | None) -> None:
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
    sys.exit(result.returncode)


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
    """Bake a sequence as keyframes and open the result in Blender's GUI.

    Step 1 runs Blender headlessly to bake the animation and save a .blend file.
    Step 2 opens that file in the GUI so you can scrub the timeline interactively.
    Separating the steps avoids WSL2 Wayland/EGL crashes during script execution.
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

    # Step 1: bake keyframes headlessly and save the .blend file.
    click.echo("Step 1/2: baking keyframes (headless)...")
    r1 = subprocess.run(
        [blender, "--background", "--factory-startup", "--python-expr", expr],
        env=headless_env,
    )
    if r1.returncode != 0:
        sys.exit(r1.returncode)

    # Step 2: open the .blend file using the Windows default app via explorer.exe.
    # The Linux Blender GUI crashes in WSL2 (no GPU/EGL), but explorer.exe hands
    # the file to Windows Blender which has full GPU access.
    win_path_result = subprocess.run(["wslpath", "-w", blend_out], capture_output=True, text=True)
    if win_path_result.returncode == 0:
        win_path = win_path_result.stdout.strip()
        click.echo(f"Step 2/2: opening {win_path} in Windows Blender...")
        subprocess.run(["explorer.exe", win_path])
    else:
        click.echo(f"Preview saved to: {blend_out}")
        click.echo("Open this file in Blender to preview the animation.")


def main() -> None:
    cli(prog_name="phanesim")


if __name__ == "__main__":
    main()
