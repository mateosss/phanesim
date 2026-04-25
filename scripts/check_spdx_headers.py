#!/usr/bin/env python3

# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repository_root = Path(__file__).resolve().parents[1]
    tracked_files = subprocess.check_output(
        ["git", "-C", str(repository_root), "ls-files"],
        text=True,
    ).splitlines()

    missing_headers = []
    for relative_path in tracked_files:
        file_path = repository_root / relative_path
        if not file_path.is_file():
            continue

        head = "\n".join(file_path.read_text(encoding="utf-8").splitlines()[:10])
        if "SPDX-License-Identifier:" not in head:
            missing_headers.append(relative_path)

    if missing_headers:
        for relative_path in missing_headers:
            print(f"missing SPDX header: {relative_path}")
        return 1

    print("all tracked files contain SPDX headers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
