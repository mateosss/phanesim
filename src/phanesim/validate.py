# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import importlib.resources
import json
from pathlib import Path

import jsonschema
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


def validate_body_rig(path: Path) -> None:
    _validate_json(path, "body_rig.json")


def validate_body_sequence(path: Path) -> None:
    _validate_json(path, "body_sequence.json")


def validate_pose_motion(path: Path) -> None:
    """Validate a pose motion description against its schema, then its ordering.

    The schema cannot express that events must be chronological, so that is
    checked here.
    """
    _validate_json(path, "pose_motion.json")

    data = json.loads(path.read_text())
    times = [int(e["t_ns"]) for e in data["events"]]
    if times != sorted(times):
        raise ValueError("pose_motion events must be sorted by t_ns")
    duration = int(data["duration_ns"])
    late = [t for t in times if t > duration]
    if late:
        raise ValueError(f"pose_motion events start after duration_ns={duration}: {late}")
