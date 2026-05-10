# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Mock bpy and mathutils before any test import so that phanesim.render
can be imported outside of Blender for testing its pure-math functions."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

for _mod in ("bpy", "mathutils"):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
