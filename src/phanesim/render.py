# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Blender EEVEE rendering pipeline for Phanesim sequences.

A BodySequence is rendered by blending named pose assets stored in the model
.blend according to a pose motion description, with the camera derived from the
head bone every frame.

Coordinate system assumptions
------------------------------
- World frame: right-handed, Z-up (matches Blender's default world).
- Body joint positions are read from the posed armature in world frame.
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

A sequence listing several motions renders each as its own take into
<output_path>/<motion name>/ instead.

This module must run inside Blender's Python interpreter (provides bpy/mathutils).
Add the phanesim src/ directory to sys.path before importing from an external script.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import bpy  # pyright: ignore[reportMissingImports]
import mathutils  # pyright: ignore[reportMissingImports]
import numpy as np
from scipy.spatial.transform import Rotation

from phanesim.posemotion import NS_PER_SECOND, PoseAsset, PoseMotion
from phanesim.rig import BodySequence
from phanesim.skeleton import rigify_hand_landmarks
from phanesim.types import Camera, CameraModel, HeadCamera, Shutter, Timestamps, Transform, Vector3

# 180° rotation around X: converts OpenCV camera frame to Blender camera frame.
_OPENCV_TO_BLENDER_CAM = mathutils.Matrix.Rotation(math.pi, 4, "X")


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


def distort_pixel(
    x: float,
    y: float,
    width: int,
    height: int,
    distortion: float,
    scale: float = 1.0,
) -> tuple[float, float]:
    """Map an undistorted pixel to where the compositor puts it.

    Blender's Lens Distortion node is a *gather*: it says which input pixel each
    output pixel reads from.  Ground truth needs the opposite direction — given a
    projected landmark in the undistorted render, where does it appear in the
    written image — so this is the forward map.

    The relation was measured against Blender's CPU compositor rather than
    derived, because the GPU shader in Blender's source implements a different
    parameterisation than the CPU path that headless renders actually use.
    Fitting a grid of known points at several distortion values gives, in
    coordinates normalised about the image centre::

        r_out = r_in * (1 + k) / (1 + k * r_in^2)

    which reproduces Blender's output to ~0.1 px for k up to 0.4, and is exactly
    the identity at k = 0.  Note the normalisation divides x by width/2 and y by
    height/2, and is centred on the image centre — not on the intrinsics
    principal point, which the compositor knows nothing about.

    Args:
        x, y:       Pixel coordinates in the undistorted image.
        width:      Image width in pixels.
        height:     Image height in pixels.
        distortion: The node's Distortion input.
        scale:      Uniform scale applied by the Transform node after distortion,
                    used to crop the borders the distortion opens up.

    Returns:
        The pixel coordinates in the written image.
    """
    half_w = width / 2.0
    half_h = height / 2.0

    # Blender samples at pixel centres, hence the half-pixel offsets.
    xn = (x + 0.5 - half_w) / half_w
    yn = (y + 0.5 - half_h) / half_h

    r2 = xn * xn + yn * yn

    # Beyond r = 1/sqrt(k) the map turns over and starts folding outward points
    # back toward the centre, so a landmark far outside the field of view would
    # be reported inside the image.  Blender draws the same line: its shader
    # writes a transparent pixel once distortion * r^2 exceeds 1.  Outside that
    # radius the point simply has no position in the rendered frame.
    if distortion * r2 > 1.0:
        return float("nan"), float("nan")

    factor = (1.0 + distortion) / (1.0 + distortion * r2) * scale

    return (
        xn * factor * half_w + half_w - 0.5,
        yn * factor * half_h + half_h - 0.5,
    )


# ---------------------------------------------------------------------------
# Blender scene helpers
# ---------------------------------------------------------------------------


def _setup_world_hdri(scene: bpy.types.Scene, hdri_path: Path) -> None:
    """Set the world background to an HDR/EXR environment map.

    The HDRI provides both the visible background and the scene lighting, so
    no additional sun lamp is needed when this is used.
    """
    world = bpy.data.worlds.new("_phanesim_world")
    world.use_nodes = True
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    nodes.clear()

    tex = nodes.new("ShaderNodeTexEnvironment")
    tex.image = bpy.data.images.load(str(hdri_path))
    tex.location = (-300, 0)

    bg = nodes.new("ShaderNodeBackground")
    bg.location = (0, 0)

    out = nodes.new("ShaderNodeOutputWorld")
    out.location = (300, 0)

    links.new(tex.outputs["Color"], bg.inputs["Color"])
    links.new(bg.outputs["Background"], out.inputs["Surface"])
    scene.world = world


def _load_compositor_effects() -> None:
    """Append the Blender built-in compositor effect node groups if not yet loaded.

    Groups: 'Chromatic Aberration', 'Sensor Noise', 'Vignette'.
    They live in compositing_nodes_essentials.blend shipped with every Blender install.
    """
    _EFFECTS = ("Chromatic Aberration", "Sensor Noise", "Vignette")
    missing = [e for e in _EFFECTS if e not in bpy.data.node_groups]
    if not missing:
        return
    major, minor = bpy.app.version[:2]
    asset_blend = (
        Path(bpy.app.binary_path).parent
        / f"{major}.{minor}"
        / "datafiles"
        / "assets"
        / "nodes"
        / "compositing_nodes_essentials.blend"
    )
    if not asset_blend.exists():
        return
    with bpy.data.libraries.load(str(asset_blend), link=False) as (src, dst):
        dst.node_groups = [e for e in missing if e in src.node_groups]


def _setup_compositor(scene: bpy.types.Scene, camera: Camera) -> None:
    """Build the compositor pipeline applied to every rendered frame.

    Uses Blender 5.x built-in effect node groups (Chromatic Aberration, Sensor Noise,
    Vignette) loaded from compositing_nodes_essentials.blend.  These are the same nodes
    visible in the Compositor Add menu → Camera & Lens Effects / Creative.

    Pipeline:
      RenderLayers → ChromaticAberration → SensorNoise → BarrelDistortion
      → Transform(1.2×) → Add(BlackMask, Transform) → [RGBtoBW]
      → Vignette → Viewer + GroupOutput

    noise_std / chroma_noise are read directly from camera and sequence and fed to the
    Sensor Noise node (values in [0-1+] range, no unit conversion).
    Open preview.blend → Compositor editor to adjust parameters interactively.
    """
    # Clean up previously created node group and black mask image.
    if old_ng := bpy.data.node_groups.get("_phanesim_compositor"):
        bpy.data.node_groups.remove(old_ng)
    if old_img := bpy.data.images.get("_phanesim_black"):
        bpy.data.images.remove(old_img)

    _load_compositor_effects()

    scene.render.use_compositing = True
    tree = bpy.data.node_groups.new("_phanesim_compositor", "CompositorNodeTree")
    tree.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    scene.compositing_node_group = tree

    W, H = camera.resolution
    nodes = tree.nodes
    links = tree.links

    def _effect(name: str, loc: tuple) -> bpy.types.Node:
        n = nodes.new("CompositorNodeGroup")
        n.node_tree = bpy.data.node_groups.get(name)
        n.location = loc
        return n

    # ── 1. Render Layers ────────────────────────────────────────────────────
    rl = nodes.new("CompositorNodeRLayers")
    rl.location = (-1400, 0)
    rl.scene = scene
    cur = rl.outputs["Image"]

    # ── 2. Chromatic Aberration (built-in effect group) ──────────────────────
    ca = _effect("Chromatic Aberration", (-1100, 0))
    if ca.node_tree and "Factor" in ca.inputs:
        ca.inputs["Factor"].default_value = camera.ca_factor
    links.new(cur, ca.inputs["Image"])
    cur = ca.outputs["Image"]

    # ── 3. Sensor Noise (built-in effect group) ──────────────────────────────
    # noise_std / chroma_noise map directly to the node's [0-1+] inputs.
    noise = _effect("Sensor Noise", (-800, 0))
    if noise.node_tree:
        noise.inputs["Luminance Noise"].default_value = camera.noise_std
        noise.inputs["Chroma Noise"].default_value = camera.chroma_noise
        noise.inputs["Animated"].default_value = True
    links.new(cur, noise.inputs["Image"])
    cur = noise.outputs["Image"]

    # ── 4. Barrel Distortion (native node, no Fit — Transform handles crop) ──
    ld = nodes.new("CompositorNodeLensdist")
    ld.location = (-500, 0)
    ld.inputs["Distortion"].default_value = camera.distortion
    ld.inputs["Dispersion"].default_value = camera.dispersion
    links.new(cur, ld.inputs["Image"])
    cur = ld.outputs["Image"]

    # ── 5. Transform: zoom in to crop the black borders ──────────────────────
    tf = nodes.new("CompositorNodeTransform")
    tf.location = (-200, 0)
    tf.inputs["X"].default_value = 0.0
    tf.inputs["Y"].default_value = 0.0
    tf.inputs["Angle"].default_value = 0.0
    tf.inputs["Scale"].default_value = camera.lens_scale
    links.new(cur, tf.inputs["Image"])
    tf_out = tf.outputs["Image"]

    # ── 6. Black mask + Add: fill any remaining corner gaps with opaque black ─
    # ShaderNodeMix(RGBA, ADD) replaces the removed CompositorNodeMixRGB.
    # A = black image (fills gaps), B = transformed image; result = A + B.
    black_img = bpy.data.images.new("_phanesim_black", W, H)
    black_img.pixels = [0.0, 0.0, 0.0, 1.0] * (W * H)
    black_node = nodes.new("CompositorNodeImage")
    black_node.image = black_img
    black_node.location = (-200, -300)

    add = nodes.new("ShaderNodeMix")
    add.data_type = "RGBA"
    add.blend_type = "ADD"
    add.clamp_factor = True
    add.clamp_result = False
    add.location = (100, 0)
    add.inputs[0].default_value = 1.0  # Factor
    links.new(black_node.outputs["Image"], add.inputs[6])  # A = black mask
    links.new(tf_out, add.inputs[7])  # B = Transform
    cur = add.outputs[2]  # Color result

    # ── 7. RGB to BW (MONO8 cameras only) ───────────────────────────────────
    if getattr(camera, "pixel_format", "RGB24") == "MONO8":
        rgb2bw = nodes.new("CompositorNodeRGBToBW")
        rgb2bw.location = (400, 0)
        links.new(cur, rgb2bw.inputs["Image"])
        cur = rgb2bw.outputs["Val"]

    # ── 8. Vignette (built-in effect group) ──────────────────────────────────
    vig = _effect("Vignette", (700, 0))
    if vig.node_tree:
        vig.inputs["Factor"].default_value = camera.vignette_factor
        vig.inputs["Feather"].default_value = camera.vignette_feather
    links.new(cur, vig.inputs["Image"])
    cur = vig.outputs["Image"]

    # ── 9. Viewer (interactive preview in Compositor viewport) ────────────────
    viewer = nodes.new("CompositorNodeViewer")
    viewer.location = (1000, -200)
    links.new(cur, viewer.inputs["Image"])

    # ── 10. Group Output → render pipeline receives the composited image ──────
    ngo = nodes.new("NodeGroupOutput")
    ngo.location = (1000, 0)
    links.new(cur, ngo.inputs["Image"])


def _configure_render(scene: bpy.types.Scene, camera: Camera, cam_obj: bpy.types.Object) -> None:
    """Apply camera properties to the Blender scene render settings."""
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x, scene.render.resolution_y = camera.resolution
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_depth = "8"
    if getattr(camera, "pixel_format", "RGB24") == "MONO8":
        scene.render.image_settings.color_mode = "BW"
    else:
        scene.render.image_settings.color_mode = "RGB"
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

    if camera.shutter != Shutter.GLOBAL:
        raise NotImplementedError("Only global shutter is supported in this version.")
    if getattr(camera, "motion_blur", False):
        scene.render.use_motion_blur = True

    scene.eevee.taa_render_samples = 8
    if hasattr(scene.eevee, "use_gtao"):
        scene.eevee.use_gtao = False
    if hasattr(scene.eevee, "use_bloom"):
        scene.eevee.use_bloom = False

    _setup_compositor(scene, camera)


# ---------------------------------------------------------------------------
# Sequence rendering
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Full-body rig: pose assets + head-mounted camera
# ---------------------------------------------------------------------------

# Pose assets are keyed into the baked timeline no finer than this, so a long
# multi-frame action does not explode the keyframe count.
_ACTION_SAMPLE_STEP = 2


def _assign_action(arm_obj: bpy.types.Object, action: bpy.types.Action | None) -> None:
    """Assign *action* to the armature, binding its slot as Blender 4.4+ requires.

    From Blender 4.4 an action carries slots; assigning the action alone leaves
    animation_data unbound and the action evaluates as empty.
    """
    anim = arm_obj.animation_data or arm_obj.animation_data_create()
    anim.action = action
    if action is None:
        return
    slots = getattr(action, "slots", None)
    if slots:
        try:
            anim.action_slot = slots[0]
        except (AttributeError, TypeError) as exc:  # pre-4.4 Blender has no slots
            print(f"[phanesim] slot binding skipped for {action.name!r}: {exc}")


def _custom_props(pose_bone: bpy.types.PoseBone) -> dict[str, object]:
    """Return the bone's custom properties, skipping Blender's UI metadata.

    Rigify stores its IK/FK and stretch switches as custom properties, so they
    have to travel with the pose for a captured pose to look right.
    """
    return {k: pose_bone[k] for k in pose_bone.keys() if k != "_RNA_UI"}


def _capture_pose(arm_obj: bpy.types.Object, action: bpy.types.Action, source_frame: int) -> dict[str, dict]:
    """Evaluate *action* at *source_frame* and snapshot every pose bone.

    Returns a plain-data snapshot so the source action can be unassigned before
    the values are written into the baked timeline.
    """
    _assign_action(arm_obj, action)
    bpy.context.scene.frame_set(int(source_frame))
    bpy.context.view_layer.update()

    return {
        pb.name: {
            "loc": list(pb.location),
            "rot_q": list(pb.rotation_quaternion),
            "rot_e": list(pb.rotation_euler),
            "rot_a": list(pb.rotation_axis_angle),
            "scale": list(pb.scale),
            "props": _custom_props(pb),
        }
        for pb in arm_obj.pose.bones
    }


def _key_pose(arm_obj: bpy.types.Object, snapshot: dict[str, dict], frame: int) -> None:
    """Apply *snapshot* to the armature and keyframe it at *frame*.

    The caller must have assigned the destination action first.
    """
    for pb in arm_obj.pose.bones:
        data = snapshot.get(pb.name)
        if data is None:
            continue

        pb.location = data["loc"]
        if pb.rotation_mode == "QUATERNION":
            pb.rotation_quaternion = data["rot_q"]
            rot_path = "rotation_quaternion"
        elif pb.rotation_mode == "AXIS_ANGLE":
            pb.rotation_axis_angle = data["rot_a"]
            rot_path = "rotation_axis_angle"
        else:
            pb.rotation_euler = data["rot_e"]
            rot_path = "rotation_euler"
        pb.scale = data["scale"]

        pb.keyframe_insert(data_path="location", frame=frame)
        pb.keyframe_insert(data_path=rot_path, frame=frame)
        pb.keyframe_insert(data_path="scale", frame=frame)

        for prop, value in data["props"].items():
            pb[prop] = value
            try:
                pb.keyframe_insert(data_path=f'["{prop}"]', frame=frame)
            except (RuntimeError, TypeError):
                # Non-animatable custom property (e.g. a string); the value is
                # still applied above, it just cannot be keyed.
                pass


def _resolve_pose_assets(motion: PoseMotion) -> dict[str, bpy.types.Action]:
    """Look up every asset the motion references, erroring on the first miss.

    Failing loudly here beats rendering a whole sequence that silently holds one
    pose because an asset name was misspelled.
    """
    resolved: dict[str, bpy.types.Action] = {}
    missing: list[str] = []
    for name in motion.assets():
        action = bpy.data.actions.get(name)
        if action is None:
            missing.append(name)
        else:
            resolved[name] = action
    if missing:
        available = sorted(a.name for a in bpy.data.actions if a.asset_data is not None)
        raise RuntimeError(f"pose assets not found in model: {missing}. Available assets: {available}")
    return resolved


def _bake_pose_motion(
    arm_obj: bpy.types.Object,
    motion: PoseMotion,
    fps: float,
    action_name: str = "_phanesim_poisson",
) -> bpy.types.Action:
    """Bake a pose motion description into a single keyframed action.

    Every event contributes one key at its own ``t_ns``; Blender interpolates
    between consecutive keys, so the armature moves continuously from each pose
    into the next.  Multi-frame assets are additionally sampled across their
    playback span.

    Blender's own F-curve interpolation then supplies every intermediate frame,
    which is why the per-frame render loop only has to call frame_set.

    Args:
        arm_obj:     The armature to bake onto.
        motion:      Timeline to bake.
        fps:         Frames per second the timeline is sampled at.
        action_name: Name for the created action.

    Returns:
        The baked action, already assigned to *arm_obj*.
    """
    assets = _resolve_pose_assets(motion)

    def to_frame(t_ns: int) -> int:
        return int(round(t_ns * fps / NS_PER_SECOND))

    # Plan every key up front, one list per event: (dest_frame, asset, source_frame).
    # A multi-frame asset contributes several, a static pose exactly one.
    event_plans: list[list[tuple[int, str, int]]] = []
    for event in motion.events:
        action = assets[event.asset]
        first = int(event.source_frames[0]) if event.source_frames else int(action.frame_range[0])
        keys = [(to_frame(event.t_ns), event.asset, first)]

        if event.kind == "action" and event.duration_ns > 0 and event.source_frames:
            span = max(1, int(event.source_frames[1]) - first)
            for offset in range(_ACTION_SAMPLE_STEP, span + 1, _ACTION_SAMPLE_STEP):
                t_ns = event.t_ns + int(offset * event.duration_ns / span)
                keys.append((to_frame(t_ns), event.asset, first + offset))
        event_plans.append(keys)

    # Pass 1 — capture each distinct (asset, source frame) once.  Switching
    # actions is the expensive part, so all reads happen before any writes and
    # repeated poses reuse their snapshot.
    snapshots: dict[tuple[str, int], dict[str, dict]] = {}
    for keys in event_plans:
        for _, asset_name, src_frame in keys:
            key = (asset_name, src_frame)
            if key not in snapshots:
                snapshots[key] = _capture_pose(arm_obj, assets[asset_name], src_frame)

    # Pass 2 — write the timeline into a fresh action.
    if old := bpy.data.actions.get(action_name):
        bpy.data.actions.remove(old)
    baked = bpy.data.actions.new(name=action_name)
    _assign_action(arm_obj, baked)

    for keys in event_plans:
        for frame, asset_name, src_frame in keys:
            _key_pose(arm_obj, snapshots[(asset_name, src_frame)], frame)

    # Blender defaults to Bezier with auto-clamped handles, which flattens the
    # curve at every key: motion nearly stops on each pose and accelerates in
    # between, so frames sampled just after a key can be visually identical.
    # Linear keeps velocity constant across each segment, which is what makes
    # every frame differ from the one before it.
    for layer in baked.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fcurve in channelbag.fcurves:
                    for keyframe in fcurve.keyframe_points:
                        keyframe.interpolation = "LINEAR"

    total_keys = sum(len(k) for k in event_plans)
    print(f"[phanesim] Baked {len(motion.events)} pose event(s) into {total_keys} keyframe group(s).")
    return baked


def _load_body_model(model_path: Path, armature_name: str) -> bpy.types.Object:
    """Open the body .blend as the working scene and return its armature.

    The file is opened rather than appended so the model keeps its own
    materials, lights and — crucially — its pose assets, which live as actions
    in that file.  This replaces the current scene, so it must run before any
    render or compositor setup.

    Raises:
        RuntimeError: If the named armature is not present in the file.
    """
    bpy.ops.wm.open_mainfile(filepath=str(model_path))

    arm_obj = bpy.data.objects.get(armature_name)
    if arm_obj is None or arm_obj.type != "ARMATURE":
        available = sorted(o.name for o in bpy.data.objects if o.type == "ARMATURE")
        raise RuntimeError(f"armature {armature_name!r} not found in {model_path}. Available armatures: {available}")

    bpy.context.view_layer.objects.active = arm_obj
    _assign_action(arm_obj, None)
    return arm_obj


def enumerate_pose_assets(model_path: Path, output_json: Path) -> None:
    """Write the pose assets found in *model_path* to *output_json*.

    Runs inside Blender on behalf of the `generate-motion` CLI command, which
    needs the authoritative asset list before it can sample a timeline.  Only
    asset-marked actions are reported, so working actions in the file are not
    mistaken for poses.
    """
    bpy.ops.wm.open_mainfile(filepath=str(model_path))
    fps = bpy.context.scene.render.fps or 24

    assets = [
        PoseAsset(
            name=action.name,
            frame_start=int(action.frame_range[0]),
            frame_end=int(action.frame_range[1]),
            fps=fps,
        )
        for action in bpy.data.actions
        if action.asset_data is not None
    ]
    assets.sort(key=lambda a: a.name)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps({"model": str(model_path), "assets": [a.to_dict() for a in assets]}, indent=2))
    print(f"[phanesim] Found {len(assets)} pose asset(s) in {model_path.name}")


def _mute_scene_lights(scene: bpy.types.Scene) -> int:
    """Hide every light object in the scene from rendering.

    An HDRI is a complete lighting environment on its own.  The body model ships
    with its own key and fill lamps, and leaving them on top of the HDRI stacks
    two full lighting rigs and blows out the skin to flat white.

    Returns:
        The number of lights muted.
    """
    muted = 0
    for obj in scene.objects:
        if obj.type == "LIGHT":
            obj.hide_render = True
            muted += 1
    return muted


def _bone_world(arm_obj: bpy.types.Object, bone_name: str, tail: bool = False) -> mathutils.Vector | None:
    """World-space position of a posed bone's head (or tail), or None if absent."""
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    return arm_obj.matrix_world @ (pb.tail if tail else pb.head)


def _bone_world_pose(
    arm_obj: bpy.types.Object,
    bone_name: str,
    tail: bool = False,
) -> tuple[mathutils.Vector, mathutils.Quaternion] | None:
    """World-space position and orientation of a posed bone, or None if absent.

    Fingertips are the tail of their distal bone and have no orientation of their
    own, so they inherit that bone's rotation — the same convention the 2D
    landmarks use for position.
    """
    pb = arm_obj.pose.bones.get(bone_name)
    if pb is None:
        return None
    matrix = arm_obj.matrix_world @ pb.matrix
    position = arm_obj.matrix_world @ (pb.tail if tail else pb.head)
    return position, matrix.to_quaternion()


def _head_delta(arm_obj: bpy.types.Object, head_cam: HeadCamera) -> mathutils.Matrix:
    """Return the head bone's rest-to-pose delta, in world space.

    Raises:
        RuntimeError: If the anchor bone is missing from the armature.
    """
    bone = arm_obj.data.bones.get(head_cam.anchor_bone)
    pose_bone = arm_obj.pose.bones.get(head_cam.anchor_bone)
    if bone is None or pose_bone is None:
        raise RuntimeError(f"head_camera.anchor_bone {head_cam.anchor_bone!r} not found in armature {arm_obj.name!r}")
    return arm_obj.matrix_world @ pose_bone.matrix @ bone.matrix_local.inverted() @ arm_obj.matrix_world.inverted()


def _head_camera_transform(arm_obj: bpy.types.Object, head_cam: HeadCamera) -> Transform:
    """Compute the head-mounted camera pose for the current pose of the rig.

    Position and direction are both rigidly attached to the head: the configured
    rest-pose values are carried through the head bone's rest-to-pose delta.  The
    camera never turns to follow the hands, so they enter and leave the frame on
    their own, exactly as on a real headset.

    Returns:
        T_world_cam in the OpenCV convention (X right, Y down, Z forward).

    Raises:
        RuntimeError: If the anchor bone is missing or rest_position is unset.
    """
    if head_cam.rest_position is None:
        raise RuntimeError("head_camera.rest_position is required to place a head-mounted camera")

    delta = _head_delta(arm_obj, head_cam)
    position = delta @ mathutils.Vector(tuple(float(c) for c in head_cam.rest_position))

    rest_forward = head_cam.rest_forward if head_cam.rest_forward is not None else (0.0, -1.0, -0.268)
    # to_3x3 so the direction is rotated by the head but not displaced by it.
    rotation_only = delta.to_3x3()
    forward = (rotation_only @ mathutils.Vector(tuple(float(c) for c in rest_forward))).normalized()
    up = (rotation_only @ mathutils.Vector((0.0, 0.0, 1.0))).normalized()

    if abs(forward.dot(up)) > 0.999:
        # Looking straight along the up axis: any other hint gives a stable roll.
        up = mathutils.Vector((0.0, 1.0, 0.0))
    right = forward.cross(up).normalized()
    down = forward.cross(right)

    rotation = np.array(
        [
            [right.x, down.x, forward.x],
            [right.y, down.y, forward.y],
            [right.z, down.z, forward.z],
        ],
        dtype=np.float64,
    )
    return Transform.from_components(
        np.array([position.x, position.y, position.z], dtype=np.float64),
        Rotation.from_matrix(rotation),
    )


def _body_joint_columns(seq: BodySequence) -> tuple[list[str], list[tuple[str, list[tuple[str, str, str]]]]]:
    """Return the joints_2d column names and the per-hand landmark definitions."""
    columns: list[str] = []
    hands: list[tuple[str, list[tuple[str, str, str]]]] = []
    for side in seq.body_rig.body.hands:
        landmarks = rigify_hand_landmarks(side)
        hands.append((side, landmarks))
        for landmark_name, _, _ in landmarks:
            columns.append(f"{side}_{landmark_name}_u")
            columns.append(f"{side}_{landmark_name}_v")
    return columns, hands


def _body_joint_3d_columns(hands: list[tuple[str, list[tuple[str, str, str]]]]) -> list[str]:
    """Return the joints_3d column names: 7 per landmark, xyz then xyzw quaternion.

    The layout matches the hand motion CSVs the earlier pipeline consumed, so the
    two are readable by the same code.
    """
    columns: list[str] = []
    for side, landmarks in hands:
        for landmark_name, _, _ in landmarks:
            stem = f"{side}_{landmark_name}"
            columns += [f"{stem}_x", f"{stem}_y", f"{stem}_z"]
            columns += [f"{stem}_qx", f"{stem}_qy", f"{stem}_qz", f"{stem}_qw"]
    return columns


def _project_landmark(
    point_world: mathutils.Vector,
    T_cam_world: Transform,
    camera: Camera,
) -> tuple[float, float]:
    """Project a world point to distorted pixel coordinates, or NaN if behind the camera.

    The compositor distorts and rescales the image after rendering, so the same
    forward map is applied here — otherwise the CSV would describe an
    undistorted image that was never written to disk.  See distort_pixel.
    """
    p_cam = T_cam_world.apply(np.array([point_world.x, point_world.y, point_world.z], dtype=np.float64))
    if p_cam[2] <= 0:
        return float("nan"), float("nan")

    u, v = project_point(p_cam, camera.intrinsics)
    width, height = camera.resolution
    return distort_pixel(u, v, width, height, camera.distortion, camera.lens_scale)


def render_body_sequence(
    seq: BodySequence,
    output_path: Path,
    frames: int | None = None,
    write_3d: bool = False,
) -> None:
    """Render a BodySequence: pose-asset motion seen by a head-mounted camera.

    Structurally this mirrors render_sequence — bake keyframes and collect
    projections, then render the whole animation in one call — but the motion
    comes from pose assets rather than CSVs and the camera pose is derived from
    the head bone instead of being read from a trajectory.

    When the sequence lists several motions each is rendered as its own take
    into <output_path>/<motion name>/, so a batch of generated animations can be
    rendered in one command.

    Args:
        seq:         Loaded BodySequence.
        output_path: Root output directory for this sequence.
        frames:      Frames to render per motion, overriding seq.frames.
        write_3d:    Also write joints_3d.csv with world-space joint poses.
    """
    for motion in seq.hand_motions:
        take_path = output_path if len(seq.hand_motions) == 1 else output_path / motion.name
        _render_body_take(seq, motion, take_path, frames if frames is not None else seq.frames, write_3d)


def _sample_timestamps(motion: PoseMotion, frames: int | None, frequency: float) -> tuple[Timestamps, float]:
    """Choose the timestamps to render, and the sampling rate they imply.

    A frame count is the direct control: *frames* samples are spread evenly over
    the whole timeline, so 2 gives the first and last frame and any count shows
    the entire motion rather than a truncated opening.  Without one the samples
    fall at the camera's own rate instead.

    Returns:
        The sample timestamps in nanoseconds, and the effective rate in Hz that
        the spacing corresponds to.
    """
    t_end = int(motion.t_end_ns)

    if frames is None:
        dt_ns = int(NS_PER_SECOND / frequency)
        return np.arange(0, t_end + 1, dt_ns, dtype=np.int64), frequency

    count = max(1, int(frames))
    if count == 1 or t_end <= 0:
        return np.zeros(1, dtype=np.int64), 1.0
    timestamps = np.linspace(0, t_end, count).round().astype(np.int64)
    effective_hz = (count - 1) * NS_PER_SECOND / t_end
    return timestamps, effective_hz


def _render_body_take(
    seq: BodySequence,
    motion: PoseMotion,
    output_path: Path,
    frames: int | None = None,
    write_3d: bool = False,
) -> None:
    """Render one pose motion of a BodySequence into *output_path*."""
    rig = seq.body_rig
    head_cam = rig.head_camera

    # Opening the model replaces the scene, so it must happen before any setup.
    arm_obj = _load_body_model(rig.body.model, rig.body.armature)
    scene = bpy.context.scene

    if seq.hdri:
        _setup_world_hdri(scene, seq.hdri)
        muted = _mute_scene_lights(scene)
        print(f"[phanesim] HDRI lighting: muted {muted} light(s) shipped with the model.")

    joint_columns, hands = _body_joint_columns(seq)

    for camera_index, camera in enumerate(rig.cameras):
        cam_label = camera.name or f"cam{camera_index}"
        cam_dir = output_path / f"cam_{cam_label}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        timestamps, effective_hz = _sample_timestamps(motion, frames, camera.frequency)
        # Bake at the sampling rate so keyframe times land exactly on rendered frames.
        _bake_pose_motion(arm_obj, motion, effective_hz)

        cam_data: bpy.types.Camera = bpy.data.cameras.new(name=cam_label)
        cam_obj: bpy.types.Object = bpy.data.objects.new(cam_label, cam_data)
        bpy.context.collection.objects.link(cam_obj)
        _configure_render(scene, camera, cam_obj)
        cam_obj.rotation_mode = "QUATERNION"

        scene.frame_start = 0
        scene.frame_end = len(timestamps) - 1
        scene.render.fps = max(1, int(round(effective_hz)))
        print(
            f"[phanesim] {len(timestamps)} frame(s) over {motion.t_end_ns / NS_PER_SECOND:.2f} s "
            f"= {effective_hz:.3g} Hz effective."
        )

        # Pass 1: step the baked animation, place the camera, collect projections.
        joint_rows: list[list[object]] = []
        joint_3d_rows: list[list[object]] = []
        for frame_idx, ts in enumerate(timestamps):
            scene.frame_set(frame_idx)
            bpy.context.view_layer.update()

            T_world_cam = _head_camera_transform(arm_obj, head_cam)
            T_cam_world = T_world_cam.inv()

            mat = mathutils.Matrix(T_world_cam.as_matrix().tolist()) @ _OPENCV_TO_BLENDER_CAM
            location, rotation, _ = mat.decompose()
            cam_obj.location = location
            cam_obj.rotation_quaternion = rotation
            cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
            cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

            row: list[object] = [int(ts)]
            row_3d: list[object] = [int(ts)]
            nan = float("nan")
            for _side, landmarks in hands:
                for _name, src_type, bone_name in landmarks:
                    pose = _bone_world_pose(arm_obj, bone_name, tail=src_type == "arm_tail")
                    if pose is None:
                        row.extend([nan, nan])
                        row_3d.extend([nan] * 7)
                        continue
                    point, quat = pose
                    row.extend(_project_landmark(point, T_cam_world, camera))
                    # xyz then xyzw: scipy/JSON scalar-last, as everywhere else.
                    row_3d.extend([point.x, point.y, point.z, quat.x, quat.y, quat.z, quat.w])
            joint_rows.append(row)
            joint_3d_rows.append(row_3d)

        csv_path = cam_dir / "joints_2d.csv"
        with csv_path.open("w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["timestamp"] + joint_columns)
            writer.writerows(joint_rows)

        if write_3d:
            csv_3d_path = cam_dir / "joints_3d.csv"
            with csv_3d_path.open("w", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow(["timestamp"] + _body_joint_3d_columns(hands))
                writer.writerows(joint_3d_rows)
            print(f"[phanesim] Wrote {csv_3d_path.name}: world-space joint poses.")

        # Pass 2: render every frame in one call so EEVEE initialises once.
        scene.render.filepath = str(cam_dir / "frame_######")
        bpy.ops.render.render(animation=True)
        print(f"[phanesim] Rendered {len(timestamps)} frame(s) to {cam_dir}")

        bpy.data.objects.remove(cam_obj)
        bpy.data.cameras.remove(cam_data)


def preview_body_sequence(seq: BodySequence, save_path: str | None = None, frames: int | None = None) -> None:
    """Bake a BodySequence as keyframes and optionally save it as a .blend file.

    Only the first camera and first motion are baked; open the result in
    Blender's GUI to scrub the timeline and inspect the compositor.
    """
    rig = seq.body_rig
    head_cam = rig.head_camera
    motion = seq.hand_motions[0]
    camera = rig.cameras[0]

    arm_obj = _load_body_model(rig.body.model, rig.body.armature)
    scene = bpy.context.scene

    if seq.hdri:
        _setup_world_hdri(scene, seq.hdri)
        _mute_scene_lights(scene)

    timestamps, effective_hz = _sample_timestamps(
        motion, frames if frames is not None else seq.frames, camera.frequency
    )
    _bake_pose_motion(arm_obj, motion, effective_hz)

    cam_data: bpy.types.Camera = bpy.data.cameras.new(name="preview_cam")
    cam_obj: bpy.types.Object = bpy.data.objects.new("preview_cam", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    _configure_render(scene, camera, cam_obj)
    cam_obj.rotation_mode = "QUATERNION"

    scene.render.fps = max(1, int(round(effective_hz)))
    scene.frame_start = 0
    scene.frame_end = len(timestamps) - 1
    scene.camera = cam_obj


    for frame_idx in range(len(timestamps)):
        scene.frame_set(frame_idx)
        bpy.context.view_layer.update()
        mat = (
            mathutils.Matrix(_head_camera_transform(arm_obj, head_cam).as_matrix().tolist())
            @ _OPENCV_TO_BLENDER_CAM
        )
        location, rotation, _ = mat.decompose()
        cam_obj.location = location
        cam_obj.rotation_quaternion = rotation
        cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    print(f"[phanesim] Preview ready: {len(timestamps)} frames at {scene.render.fps} fps.")
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"[phanesim] Saved: {save_path}")
