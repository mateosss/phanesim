# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import pytest

from phanesim.cli import SOFTWARE_GL_VAR, _render_env


class TestRenderEnvironment:
    """Which rasteriser headless Blender is pointed at."""

    def test_software_by_default(self):
        # WSL2 has no display to make a GPU context on, and that is where this
        # runs day to day, so the CPU path has to be what you get for free.
        assert _render_env({})["LIBGL_ALWAYS_SOFTWARE"] == "1"

    def test_turned_off_for_a_machine_with_a_reachable_gpu(self):
        assert "LIBGL_ALWAYS_SOFTWARE" not in _render_env({SOFTWARE_GL_VAR: "0"})

    @pytest.mark.parametrize("value", ["0", "false", "no", "NO", " 0 "])
    def test_the_off_spellings(self, value):
        assert "LIBGL_ALWAYS_SOFTWARE" not in _render_env({SOFTWARE_GL_VAR: value})

    @pytest.mark.parametrize("value", ["1", "true", "yes", "anything"])
    def test_anything_else_keeps_the_software_path(self, value):
        assert _render_env({SOFTWARE_GL_VAR: value})["LIBGL_ALWAYS_SOFTWARE"] == "1"

    def test_an_inherited_setting_is_cleared_when_switched_off(self):
        # Otherwise a stale export in the job script would silently keep the
        # render on the CPU on a node that asked for a GPU.
        env = _render_env({SOFTWARE_GL_VAR: "0", "LIBGL_ALWAYS_SOFTWARE": "1"})
        assert "LIBGL_ALWAYS_SOFTWARE" not in env

    def test_the_rest_of_the_environment_is_carried_through(self):
        assert _render_env({"BLENDER_BIN": "/opt/blender"})["BLENDER_BIN"] == "/opt/blender"
