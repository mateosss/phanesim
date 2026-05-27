# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Blender EEVEE rendering pipeline for Phanesim sequences.

Coordinate system assumptions
------------------------------
- World frame: right-handed, Z-up (matches Blender's default world).
- Camera motion CSV gives T_world_body: body/head pose in world frame.
- Hand motion CSV gives joint poses in world frame (absolute world positions).
- Camera intrinsics use the OpenCV convention: X right, Y down, Z forward.
- Blender's camera local frame: X right, Y up, Z backward (looks along -Z).
  A 180° rotation around X converts OpenCV→Blender camera orientation.

Output layout per sequence
---------------------------
  <output_path>/
    cam_<name>/
      frame_000000.png
      frame_000001.png
      ...
      joints_2d.csv   -- columns: timestamp, {hand}_{joint}_u, {hand}_{joint}_v, ...

This module must run inside Blender's Python interpreter (provides bpy/mathutils).
Add the phanesim src/ directory to sys.path before importing from an external script.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path

import bpy  # pyright: ignore[reportMissingImports]
import mathutils  # pyright: ignore[reportMissingImports]
import numpy as np

from phanesim.rig import Project, Sequence
from phanesim.types import Camera, CameraModel, Transform, Vector3

# 180° rotation around X: converts OpenCV camera frame to Blender camera frame.
_OPENCV_TO_BLENDER_CAM = mathutils.Matrix.Rotation(math.pi, 4, "X")

# The hand.blend armature has rotation_mode='QUATERNION' with a baked base
# quaternion of 180° around -Z (w=0, z=-1).  Wrist CSV rotations are composed
# with this base so CSV identity preserves the rest-pose appearance.
# NOTE: this constant is specific to hand.blend; a different rig may need a
# different value.  Future work: read it from the blend file at load time.
_ARM_BASE_QUAT = mathutils.Quaternion((0.0, 0.0, 0.0, -1.0))


# ---------------------------------------------------------------------------
# Pure-math projection (no bpy dependency)
# ---------------------------------------------------------------------------


def _project_pinhole(p_cam: Vector3, params: dict[str, float]) -> tuple[float, float]:
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    return params["fx"] * x / z + params["cx"], params["fy"] * y / z + params["cy"]


def _project_kb4(p_cam: Vector3, params: dict[str, float]) -> tuple[float, float]:
    """Kannala-Brandt equidistant projection (kb4)."""
    x, y, z = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(r, z)
    t2 = theta * theta
    d = theta * (1.0 + params["k1"] * t2 + params["k2"] * t2**2 + params["k3"] * t2**3 + params["k4"] * t2**4)
    if r < 1e-9:
        return params["cx"], params["cy"]
    u = params["fx"] * d * x / r + params["cx"]
    v = params["fy"] * d * y / r + params["cy"]
    return u, v


def project_point(p_cam: Vector3, model: CameraModel) -> tuple[float, float]:
    """Project a point in camera frame to image coordinates (u, v) in pixels."""
    if model.name == "pinhole":
        return _project_pinhole(p_cam, model.parameters)
    if model.name == "kb4":
        return _project_kb4(p_cam, model.parameters)
    raise ValueError(f"Unsupported camera model: {model.name!r}. Supported: pinhole, kb4.")


# ---------------------------------------------------------------------------
# Blender scene helpers
# ---------------------------------------------------------------------------


def _set_camera(cam_obj: bpy.types.Object, T_world_cam: Transform) -> None:
    """Position a Blender camera object using a world-frame Transform."""
    cam_obj.matrix_world = mathutils.Matrix(T_world_cam.as_matrix().tolist()) @ _OPENCV_TO_BLENDER_CAM


def _set_hand_bones(arm_obj: bpy.types.Object, joint_names: list[str], joint_poses: list[Transform]) -> None:
    """Position the armature and apply wrist rotation from the CSV.

    The CSV wrist quaternion is relative to the blend file's baked base orientation
    (_ARM_BASE_QUAT).  Total rotation = _ARM_BASE_QUAT @ q_csv, so CSV identity
    preserves the original rest-pose look (back of hand visible from above), while
    a 180° Z CSV rotation flips to show the palm (pronation/supination).

    The armature object location is recalculated each frame so the Wrist bone
    stays exactly at the CSV world position regardless of rotation.
    Non-wrist joints named in the CSV are driven by applying their quaternion
    directly as matrix_basis (bone-local rotation relative to rest pose).
    Finger flexion is rotation around each bone's local X axis.
    """
    # Reset all pose bones to rest pose.
    for pb in arm_obj.pose.bones:
        pb.matrix_basis = mathutils.Matrix.Identity(4)

    wrist_idx = next((i for i, n in enumerate(joint_names) if n.lower() == "wrist"), None)
    if wrist_idx is None:
        return

    world_wrist = mathutils.Vector(joint_poses[wrist_idx].translation.tolist())

    # Compose: blend-file base orientation + CSV relative rotation.
    # CSV identity keeps the original rest-pose look (back of hand up);
    # CSV 180° around Z flips to show the palm.
    # scipy as_quat() → [x, y, z, w]; Blender Quaternion → [w, x, y, z].
    q = joint_poses[wrist_idx].rotation.as_quat()
    q_csv = mathutils.Quaternion((q[3], q[0], q[1], q[2]))
    q_total = _ARM_BASE_QUAT @ q_csv

    arm_obj.rotation_mode = "QUATERNION"
    arm_obj.rotation_quaternion = q_total

    # Keep Wrist bone at CSV world position: location = world_wrist - R @ rest_head.
    rest_head = mathutils.Vector(arm_obj.data.bones["Wrist"].head_local)
    arm_obj.location = world_wrist - q_total.to_matrix() @ rest_head

    # Apply per-finger bone rotations from the CSV.
    # For non-wrist joints the CSV quaternion is interpreted as bone-local rotation
    # (i.e. directly as matrix_basis).  matrix_basis identity = rest pose, so
    # bending is expressed as a rotation relative to rest — no base-quat composition
    # needed here, unlike the whole-armature wrist rotation above.
    # Flexion (curling toward palm) is rotation around each bone's local X axis.
    for i, name in enumerate(joint_names):
        if name.lower() == "wrist":
            continue
        pb = arm_obj.pose.bones.get(name)
        if pb is None:
            # Case-insensitive fallback.
            pb = next((b for b in arm_obj.pose.bones if b.name.lower() == name.lower()), None)
        if pb is None:
            continue
        q = joint_poses[i].rotation.as_quat()  # [x, y, z, w]
        pb.matrix_basis = mathutils.Quaternion((q[3], q[0], q[1], q[2])).to_matrix().to_4x4()


def _load_hand_model(hand_model_path: Path) -> bpy.types.Object:
    """Append all objects from a .blend file and return the first Armature.

    The armature (and its parented mesh) is moved to the world origin so that
    bone matrices in armature-local space equal world-space transforms.  This
    lets the hand-motion CSV supply plain world-space joint positions.
    """
    # Remove Blender factory-startup objects (Cube, Camera, Light) so they
    # don't appear in renders or receive the PBR skin texture.
    for obj in list(bpy.context.scene.objects):
        if obj.name in ("Cube", "Camera", "Light"):
            bpy.data.objects.remove(obj, do_unlink=True)

    with bpy.data.libraries.load(str(hand_model_path), link=False) as (src, dst):
        dst.objects = [name for name in src.objects]  # type: ignore[assignment]
    arm_obj: bpy.types.Object | None = None
    for obj in dst.objects:  # type: ignore[attr-defined]
        bpy.context.collection.objects.link(obj)
        if obj.type == "ARMATURE" and arm_obj is None:
            arm_obj = obj
    if arm_obj is None:
        raise RuntimeError(f"No armature found in hand model: {hand_model_path}")

    # Normalize: move armature to world origin; parented meshes follow automatically.
    arm_obj.location = mathutils.Vector((0.0, 0.0, 0.0))
    arm_obj.rotation_euler = mathutils.Euler((0.0, 0.0, 0.0), "XYZ")
    arm_obj.scale = mathutils.Vector((1.0, 1.0, 1.0))

    # Mute IK constraints so that manual bone posing via matrix_basis is not
    # overridden by the IK solver pulling bones toward its embedded targets.
    for pb in arm_obj.pose.bones:
        for constraint in pb.constraints:
            if constraint.type == "IK":
                constraint.mute = True

    apply_hand_textures(str(hand_model_path))
    return arm_obj


# ---------------------------------------------------------------------------
# The texture application (temporary, will be replaced by a more robust material system in the future)
# ---------------------------------------------------------------------------

# Maps a PBR role to filename keywords that identify it.
_TEXTURE_KEYWORDS: dict[str, list[str]] = {
    "albedo": ["albedo", "diffuse", "color", "basecolor", "base_color"],
    "normal": ["normal", "bump", "nrm"],
    "roughness": ["roughness", "rough"],
    "displacement": ["displacement", "disp", "height"],
    "thickness": ["thickness", "thick", "sss"],
}

_TEXTURE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tga", ".exr"}


def _find_textures_in_dir(directory: str) -> dict[str, str]:
    """Scan *directory* for PBR texture files and return a role→absolute-path map."""
    texture_map: dict[str, str] = {}
    for fname in os.listdir(directory):
        if os.path.splitext(fname)[1].lower() not in _TEXTURE_EXTS:
            continue
        fname_lower = fname.lower()
        for role, keywords in _TEXTURE_KEYWORDS.items():
            if role not in texture_map and any(kw in fname_lower for kw in keywords):
                texture_map[role] = os.path.join(directory, fname)
                break
    return texture_map


def _apply_pbr_material(obj: bpy.types.Object, texture_map: dict[str, str]) -> None:
    """Replace all materials on *obj* with a freshly built Principled BSDF material."""
    mat = bpy.data.materials.new(name=f"_phanesim_pbr_{obj.name}")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.location = (0, 0)
    links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

    # -----------------------------------------------------------------------
    # NEW: Set Matte Skin Default Values
    # These apply if textures are missing or act as a base for blending.
    # -----------------------------------------------------------------------

    # 1. Base skin tone (approximating the image provided)
    bsdf.inputs["Base Color"].default_value = (0.65, 0.40, 0.30, 1.0)

    # 2. High roughness for a matte, non-glossy finish (0.0 is mirror, 1.0 is flat)
    bsdf.inputs["Roughness"].default_value = 0.75

    # 3. Lower specular highlights to prevent the "wet plastic" look
    # (Blender 4.0+ renamed "Specular" to "Specular IOR Level")
    if "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = 0.2
    elif "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = 0.2

    # 4. Optional: Add a tiny bit of Subsurface Scattering for flesh softness
    if "Subsurface Weight" in bsdf.inputs:
        bsdf.inputs["Subsurface Weight"].default_value = 0.05
    elif "Subsurface" in bsdf.inputs:
        bsdf.inputs["Subsurface"].default_value = 0.05

    def _add_tex(role: str, x: float, y: float, color_space: str = "Non-Color") -> bpy.types.Node | None:
        if role not in texture_map:
            return None
        tex = nodes.new("ShaderNodeTexImage")
        tex.location = (x, y)
        img = bpy.data.images.load(texture_map[role], check_existing=True)
        img.colorspace_settings.name = color_space
        tex.image = img
        return tex

    albedo = _add_tex("albedo", -600, 300, "sRGB")
    if albedo:
        links.new(albedo.outputs["Color"], bsdf.inputs["Base Color"])

    rough = _add_tex("roughness", -600, 0)
    if rough:
        # 1. Create an Invert Node
        invert = nodes.new("ShaderNodeInvert")
        invert.location = (-300, 0)  # Place it between the texture and the BSDF

        # 2. Link Texture -> Invert Node -> BSDF Roughness
        links.new(rough.outputs["Color"], invert.inputs["Color"])
        links.new(invert.outputs["Color"], bsdf.inputs["Roughness"])

    bump = _add_tex("normal", -800, -200)
    if bump:
        nmap = nodes.new("ShaderNodeNormalMap")
        nmap.location = (-400, -200)
        links.new(bump.outputs["Color"], nmap.inputs["Color"])
        links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])

    disp_tex = _add_tex("displacement", -600, -450)
    if disp_tex:
        disp = nodes.new("ShaderNodeDisplacement")
        disp.location = (0, -400)
        links.new(disp_tex.outputs["Color"], disp.inputs["Height"])
        links.new(disp.outputs["Displacement"], out.inputs["Displacement"])

    thick = _add_tex("thickness", -600, 150)
    if thick:
        links.new(thick.outputs["Color"], bsdf.inputs["Subsurface Weight"])

    obj.data.materials.clear()
    obj.data.materials.append(mat)


def apply_hand_textures(model_file_path: str) -> None:
    """Build and apply PBR materials to every MESH in the scene.

    Looks for a 'textures' folder next to *model_file_path*, then the folder
    itself.  Safe to call when no textures are present.
    """
    model_dir = os.path.dirname(os.path.abspath(model_file_path))
    search_dirs = [os.path.join(model_dir, "textures"), model_dir]

    texture_map: dict[str, str] = {}
    for d in search_dirs:
        if os.path.isdir(d):
            texture_map = _find_textures_in_dir(d)
            if texture_map:
                print(f"[PBR] Found textures in: {d}")
                break

    if not texture_map:
        print(f"[PBR] No textures found near '{model_file_path}'. Hand will render with default material.")
        return

    print("[PBR] Discovered texture map:")
    for role, path in texture_map.items():
        print(f"        {role:12s} -> {os.path.basename(path)}")

    mesh_count = 0
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _apply_pbr_material(obj, texture_map)
            mesh_count += 1

    print(f"[PBR] Applied textures to {mesh_count} mesh object(s).")


def _add_sun_light() -> tuple[bpy.types.Object, bpy.types.Light]:
    """Add a sun lamp aimed from the +Y side to illuminate the front of the scene."""
    light_data: bpy.types.Light = bpy.data.lights.new(name="_phanesim_sun", type="SUN")
    light_data.energy = 3.0
    light_obj: bpy.types.Object = bpy.data.objects.new("_phanesim_sun", light_data)
    # Rotate 90° around X so the sun shines along +Y (from the camera's side).
    light_obj.rotation_euler = mathutils.Euler((-math.pi / 2, 0.0, 0.0), "XYZ")
    bpy.context.collection.objects.link(light_obj)
    return light_obj, light_data


def _setup_world_light(scene: bpy.types.Scene) -> None:
    """Add a soft white ambient world background so unlit surfaces aren't pure black."""
    world = bpy.data.worlds.new("_phanesim_world")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 0.4
    scene.world = world


def _configure_render(scene: bpy.types.Scene, camera: Camera, cam_obj: bpy.types.Object) -> None:
    """Apply camera properties to the Blender scene render settings."""
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = camera.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.camera = cam_obj

    # Set the Blender camera focal length to match the intrinsic fx parameter.
    # Using a standard 36 mm sensor width: lens_mm = sensor_width * fx / image_width.
    bpy_cam = cam_obj.data
    fx = camera.intrinsics.parameters.get("fx", 500.0)
    sensor_w = 36.0
    bpy_cam.lens = sensor_w * fx / camera.resolution[0]
    bpy_cam.sensor_width = sensor_w
    bpy_cam.sensor_fit = "HORIZONTAL"
    bpy_cam.clip_start = 0.01
    bpy_cam.clip_end = 200.0

    if getattr(camera, "motion_blur", False):
        scene.render.use_motion_blur = True
    if getattr(camera, "shutter", None) == "rolling":
        scene.render.use_motion_blur = True


# ---------------------------------------------------------------------------
# Frame-level rendering
# ---------------------------------------------------------------------------


def _render_frame(scene: bpy.types.Scene, output_file: Path) -> None:
    scene.render.filepath = str(output_file)
    bpy.ops.render.render(write_still=True)


def _joint_columns(seq: Sequence) -> list[str]:
    """Return ordered column names for the joints_2d CSV (excluding timestamp)."""
    cols: list[str] = []
    rig = seq.camhand_rig
    for h, hand in enumerate(rig.hands):
        hand_label = hand.name or f"hand{h}"
        for joint_name in seq.hand_motions[h].joint_names:
            cols.append(f"{hand_label}_{joint_name}_u")
            cols.append(f"{hand_label}_{joint_name}_v")
    return cols


# ---------------------------------------------------------------------------
# Sequence rendering
# ---------------------------------------------------------------------------


def render_sequence(seq: Sequence, output_path: Path) -> None:
    """Render all frames of a Sequence and write images + ground-truth CSVs.

    Args:
        seq:         Loaded Sequence object.
        output_path: Root output directory for this sequence.
    """
    scene = bpy.context.scene
    rig = seq.camhand_rig

    # Determine time range: intersection of all motion trajectory spans.
    t_start = max(
        *(m.ts[0] for m in seq.cam_motions),
        *(m.ts[0] for m in seq.hand_motions),
    )
    t_end = min(
        *(m.ts[-1] for m in seq.cam_motions),
        *(m.ts[-1] for m in seq.hand_motions),
    )

    # Add lighting — factory-startup gives an empty scene with no lights.
    sun_obj, sun_data = _add_sun_light()
    _setup_world_light(scene)

    # Load hand models once; keep reference for bone posing.
    arm_objs: list[bpy.types.Object] = []
    for hand in rig.hands:
        arm_objs.append(_load_hand_model(hand.model))

    joint_cols = _joint_columns(seq)

    for c, (camera, cam_motion) in enumerate(zip(rig.cameras, seq.cam_motions, strict=True)):
        cam_label = camera.name or f"cam{c}"
        cam_dir = output_path / f"cam_{cam_label}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        # Create a Blender camera object for this camera.
        bpy_cam_data: bpy.types.Camera = bpy.data.cameras.new(name=cam_label)
        cam_obj: bpy.types.Object = bpy.data.objects.new(cam_label, bpy_cam_data)
        bpy.context.collection.objects.link(cam_obj)
        _configure_render(scene, camera, cam_obj)

        # Build frame timestamps at this camera's frequency.
        dt_ns = int(1e9 / camera.frequency)
        timestamps = np.arange(t_start, t_end + 1, dt_ns, dtype=np.int64)

        csv_path = cam_dir / "joints_2d.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp"] + joint_cols)

            for frame_idx, ts in enumerate(timestamps):
                # Position the camera.
                T_world_body = cam_motion.get_pose(ts)
                T_b_c = camera.T_b_c
                T_world_cam = T_world_body * T_b_c
                _set_camera(cam_obj, T_world_cam)

                # Position hand bones and collect 2D projections.
                row: list[object] = [int(ts)]
                T_cam_world = T_world_cam.inv()

                for h, (hand_motion, arm_obj) in enumerate(zip(seq.hand_motions, arm_objs, strict=True)):
                    joint_poses = hand_motion.get_joint_poses(ts)

                    # Pose the armature via Python API (no mode switching needed).
                    _set_hand_bones(arm_obj, hand_motion.joint_names, joint_poses)
                    bpy.context.view_layer.update()

                    # T_c_h[c][h]: transform from hand frame to camera frame.
                    # Joint poses from the CSV are in world frame, so we compose:
                    # p_cam = T_cam_world * T_world_hand * p_joint_local
                    # With T_c_h = identity (default) this reduces to T_cam_world.
                    T_c_h = rig.T_c_h[c][h]
                    T_cam_hand = T_cam_world * T_c_h

                    # Project each joint to image coordinates.
                    for joint_pose in joint_poses:
                        p_world = joint_pose.translation
                        p_cam = T_cam_hand.apply(p_world)
                        if p_cam[2] > 0:
                            u, v = project_point(p_cam, camera.intrinsics)
                        else:
                            u, v = float("nan"), float("nan")
                        row.extend([u, v])

                # Render and save frame image.
                frame_file = cam_dir / f"frame_{frame_idx:06d}.png"
                _render_frame(scene, frame_file)
                writer.writerow(row)

        # Clean up camera object after rendering this camera's frames.
        bpy.data.objects.remove(cam_obj)
        bpy.data.cameras.remove(bpy_cam_data)

    # Clean up imported hand models and the sun light.
    for arm_obj in arm_objs:
        bpy.data.objects.remove(arm_obj, do_unlink=True)
    bpy.data.objects.remove(sun_obj, do_unlink=True)
    bpy.data.lights.remove(sun_data)


def render_project(project: Project, output_path: Path) -> None:
    """Render all sequences in a Project."""
    for seq in project.sequences:
        render_sequence(seq, output_path / seq.output_path)
