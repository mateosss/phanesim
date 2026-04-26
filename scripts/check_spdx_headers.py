#!/usr/bin/env python3

# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SIDE_CAR_SUFFIX = ".spdx"
IGNORE_FILES = {"LICENSE"}


def has_spdx_header(file_path: Path) -> bool:
    try:
        head = "\n".join(file_path.read_text(encoding="utf-8").splitlines()[:10])
    except UnicodeDecodeError:
        sidecar_path = file_path.with_name(f"{file_path.name}{SIDE_CAR_SUFFIX}")
        return sidecar_path.is_file() and has_spdx_header(sidecar_path)

    return "SPDX-License-Identifier:" in head


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    tracked_files = subprocess.check_output(
        ["git", "-C", str(repository_root), "ls-files"],
        text=True,
    ).splitlines()
    tracked_files = [f for f in tracked_files if Path(f).name not in IGNORE_FILES]

    missing_headers = []
    for relative_path in tracked_files:
        file_path = repository_root / relative_path
        if not file_path.is_file():
            continue

        if not has_spdx_header(file_path):
            missing_headers.append(relative_path)

    if missing_headers:
        for relative_path in missing_headers:
            print(f"missing SPDX header: {relative_path}")
        return 1

    print("all tracked files contain SPDX headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
