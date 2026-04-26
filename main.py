# Copyright 2026, Mateo de Mayo.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from pathlib import Path

import bpy  # pyright: ignore[reportMissingImports]

import bootstrap

bl_info = {
    "name": "Phanesim",
    "author": "Mateo de Mayo",
    "version": (0, 1, 0),
    "blender": (5, 1, 1),
    "location": "View3D > Sidebar > Phanesim",
    "description": "Quick rendering controls for Phanesim development",
    "category": "Development",
}


class PHANESIM_OT_render_still(bpy.types.Operator):
    bl_idname = "phanesim.render_still"
    bl_label = "Render"
    bl_description = "Render the current scene in EEVEE to the selected output filename"

    def execute(self, context: bpy.types.Context) -> set[str]:
        scene = context.scene
        output_name = scene.phanesim_output_name.strip() or "output.png"

        if not output_name.lower().endswith(".png"):
            output_name = f"{output_name}.png"

        output_path = output_name if Path(output_name).is_absolute() else bpy.path.abspath(f"//{output_name}")

        previous_engine = scene.render.engine
        previous_path = scene.render.filepath
        previous_format = scene.render.image_settings.file_format

        scene.render.engine = "BLENDER_EEVEE"
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = output_path

        try:
            bpy.ops.render.render(write_still=True)
        finally:
            scene.render.engine = previous_engine
            scene.render.filepath = previous_path
            scene.render.image_settings.file_format = previous_format

        self.report({"INFO"}, f"Rendered to {output_path}")
        return {"FINISHED"}


class PHANESIM_OT_reload_scripts(bpy.types.Operator):
    bl_idname = "phanesim.reload_scripts"
    bl_label = "Reload Scripts"
    bl_description = "Reload all Blender scripts and refresh the current add-on"

    def execute(self, context: bpy.types.Context) -> set[str]:
        bootstrap.main()
        return {"FINISHED"}


class PHANESIM_PT_panel(bpy.types.Panel):
    bl_idname = "PHANESIM_PT_panel"
    bl_label = "Phanesim"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phanesim"

    def draw(self, context: bpy.types.Context) -> None:
        layout = self.layout
        scene = context.scene

        layout.prop(scene, "phanesim_output_name", text="Output image")
        layout.operator(PHANESIM_OT_render_still.bl_idname, text="Render", icon="RENDER_STILL")


class PHANESIM_PT_developer_panel(bpy.types.Panel):
    bl_idname = "PHANESIM_PT_developer_panel"
    bl_label = "Developer"
    bl_space_type = "VIEW_3D"
    bl_region_type = "UI"
    bl_category = "Phanesim"

    def draw(self, context: bpy.types.Context) -> None:
        del context

        layout = self.layout
        layout.operator(PHANESIM_OT_reload_scripts.bl_idname, text="Reload Scripts", icon="FILE_REFRESH")


classes = (
    PHANESIM_OT_render_still,
    PHANESIM_OT_reload_scripts,
    PHANESIM_PT_panel,
    PHANESIM_PT_developer_panel,
)


def register() -> None:
    for cls in classes:
        bpy.utils.register_class(cls)

    bpy.types.Scene.phanesim_output_name = bpy.props.StringProperty(
        name="Output image name",
        default="output.png",
    )


def unregister() -> None:
    del bpy.types.Scene.phanesim_output_name

    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


def main():
    register()


if __name__ == "__main__":
    main()
