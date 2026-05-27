# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import jsonschema
import pandas as pd
from referencing import Registry, Resource


def _build_registry() -> Registry:
    schemas_dir = importlib.resources.files("phanesim") / "schemas"
    registry: Registry = Registry()
    for entry in schemas_dir.iterdir():  # type: ignore[union-attr]
        if entry.name.endswith(".json"):
            schema = json.loads(entry.read_text())  # type: ignore[arg-type]
            if "$id" in schema:
                registry = registry.with_resource(schema["$id"], Resource.from_contents(schema))
    return registry


def _validate_json(path: Path, schema_name: str) -> None:
    schemas_dir = importlib.resources.files("phanesim") / "schemas"
    schema = json.loads((schemas_dir / schema_name).read_text())  # type: ignore[arg-type]
    data = json.loads(path.read_text())
    jsonschema.Draft202012Validator(schema, registry=_build_registry()).validate(data)


def validate_camera(path: Path) -> None:
    _validate_json(path, "camera.json")


def validate_hand(path: Path) -> None:
    _validate_json(path, "hand.json")


def validate_camhand_rig(path: Path) -> None:
    _validate_json(path, "camhand_rig.json")


def validate_sequence(path: Path) -> None:
    _validate_json(path, "sequence.json")


def validate_project(path: Path) -> None:
    _validate_json(path, "project.json")


_CAMERA_MOTION_COLUMNS = {"timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"}


def validate_camera_motion(path: Path) -> None:
    df = pd.read_csv(path, nrows=0)
    missing = _CAMERA_MOTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"camera_motion CSV missing columns: {sorted(missing)}")


def validate_hand_motion(path: Path) -> None:
    df = pd.read_csv(path, nrows=0)
    if "timestamp" not in df.columns:
        raise ValueError("hand_motion CSV missing 'timestamp' column")

    pose_cols = [c for c in df.columns if c != "timestamp"]
    if len(pose_cols) == 0 or len(pose_cols) % 7 != 0:
        raise ValueError(
            f"hand_motion CSV must have 7 columns per joint after 'timestamp' (got {len(pose_cols)} pose columns)"
        )

    for i in range(0, len(pose_cols), 7):
        group = pose_cols[i : i + 7]
        name = group[0].removesuffix("_x")
        expected = [f"{name}_x", f"{name}_y", f"{name}_z", f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"]
        if group != expected:
            raise ValueError(f"hand_motion CSV: expected columns {expected} at position {i}, got {group}")
