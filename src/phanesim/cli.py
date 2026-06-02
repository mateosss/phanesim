# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from pathlib import Path

import click
import jsonschema

import phanesim.validate as val

# Parent directory of the phanesim package, added to sys.path inside Blender
# so that `import phanesim` works in the headless rendering subprocess.
_PKG_PARENT = str(Path(__file__).parent.parent)

# Site-packages of the active venv, injected into Blender's Python so that
# third-party dependencies (scipy, pandas, numpy) are importable there.
_SITE_PACKAGES = sysconfig.get_paths()["purelib"]


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

    sys_path_setup = f"import sys; sys.path[0:0] = [{_SITE_PACKAGES!r}, {_PKG_PARENT!r}]; "
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
    result = subprocess.run([blender, "--background", "--factory-startup", "--python-expr", expr])
    sys.exit(result.returncode)


def main() -> None:
    cli(prog_name="phanesim")


if __name__ == "__main__":
    main()
