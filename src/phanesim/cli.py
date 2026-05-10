# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import click
import jsonschema

from . import validate as val

# Parent directory of the phanesim package — added to sys.path inside Blender
# so that `import phanesim` works in the headless rendering subprocess.
_PKG_PARENT = str(Path(__file__).parent.parent)

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
    help="Path to the Blender executable (default: 'blender' on PATH).",
)
def generate(kind: str, input_path: Path, output_path: Path, blender_bin: str | None) -> None:
    """Render a sequence or project by driving Blender headlessly.

    Requires Blender to be installed. Set BLENDER_BIN or pass --blender to
    specify the executable if 'blender' is not on PATH.
    """
    input_abs = str(input_path.resolve())
    output_abs = str(output_path.resolve())

    if kind == "sequence":
        expr = (
            f"import sys; sys.path.insert(0, {_PKG_PARENT!r}); "
            "from pathlib import Path; "
            "from phanesim.rig import Sequence; "
            "from phanesim.render import render_sequence; "
            f"render_sequence(Sequence.from_path(Path({input_abs!r})), Path({output_abs!r}))"
        )
    else:
        expr = (
            f"import sys; sys.path.insert(0, {_PKG_PARENT!r}); "
            "from pathlib import Path; "
            "from phanesim.rig import Project; "
            "from phanesim.render import render_project; "
            f"render_project(Project.from_path(Path({input_abs!r})), Path({output_abs!r}))"
        )

    blender = blender_bin or "blender"
    result = subprocess.run([blender, "--background", "--factory-startup", "--python-expr", expr])
    sys.exit(result.returncode)


def main() -> None:
    cli(prog_name="phanesim")


if __name__ == "__main__":
    main()

