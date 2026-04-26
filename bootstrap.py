# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

"""Script used inside blend file"""

import importlib
import sys
from pathlib import Path

import bpy


def main():
    script_dir = Path(bpy.data.filepath).parent
    if script_dir not in sys.path:
        sys.path.append(str(script_dir))

    # scripts = []  # .py scripts in the same directory as this .blend file
    # for file in script_dir.iterdir():
    #     if file.suffix == ".py":
    #         scripts.append(file.stem)
    scripts = ["main", "bootstrap"]  # TODO: only do this for some scripts that are blender-compatible (good imports)

    # Reload all scripts and collect the main module (which registers COLDER operators)
    main_mod = None
    for script in scripts:
        mod = importlib.import_module(script)
        importlib.reload(mod)  # refresh if edited
        if script == "main":
            main_mod = mod

    # Run main module to register COLDER
    assert main_mod is not None, "main.py not found in the same directory as the .blend file"
    main_mod.main()


if __name__ == "__main__":
    main()
