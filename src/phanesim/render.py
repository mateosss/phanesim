# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""Blender EEVEE rendering pipeline for Phanesim sequences.

A BodySequence is rendered by blending named pose assets from the model .blend
according to a pose motion description, with the camera derived from the head
bone every frame.

Conventions: the world frame is right-handed and Z-up, as Blender's default, and
joint positions are read from the posed armature in it.  Camera intrinsics use
OpenCV axes (X right, Y down, Z forward) while Blender's camera looks along -Z,
so a 180° rotation about X converts between them.

Each camera writes <output_path>/cam_<name>/ holding frame_000000.png ... and
joints_2d.csv, whose columns are timestamp then {hand}_{joint}_u/_v per landmark.
A sequence listing several motions renders each as its own take into
<output_path>/<motion name>/ instead.

Must run inside Blender's Python interpreter, with phanesim's src/ on sys.path.
"""

from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path

import bpy  # pyright: ignore[reportMissingImports]
import mathutils  # pyright: ignore[reportMissingImports]
import numpy as np
from scipy.spatial.transform import Rotation

from phanesim.clips import SEQUENCE_FILE, clip_dirs, is_done, mark_done, stray_clip_dirs
from phanesim.posemotion import NS_PER_SECOND, PoseAsset, PoseMotion
from phanesim.rig import BodySequence
from phanesim.skeleton import rigify_hand_landmarks
from phanesim.types import (
    Camera,
    CameraModel,
    CameraSweep,
    HeadCamera,
    Shutter,
    Timestamps,
    Transform,
    Vector3,
)

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

    Blender's Lens Distortion node is a *gather* — it says which input pixel each
    output pixel reads from — but ground truth needs the opposite: where a
    landmark in the undistorted render lands in the written image.

    The relation was measured against Blender's CPU compositor rather than
    derived, because the GPU shader in Blender's source uses a different
    parameterisation from the CPU path headless renders actually take.  In
    coordinates normalised about the image centre::

        r_out = r_in * (1 + k) / (1 + k * r_in^2)

    reproducing Blender to ~0.1 px for k up to 0.4, and exactly the identity at
    k = 0.  The normalisation divides x by width/2 and y by height/2 about the
    image centre, not the intrinsics principal point, which the compositor knows
    nothing about.

    Args:
        x, y:          Pixel coordinates in the undistorted image.
        width, height: Image size in pixels.
        distortion:    The node's Distortion input.
        scale:         Uniform scale the Transform node applies afterwards, to
                       crop the borders the distortion opens up.

    Returns:
        The pixel coordinates in the written image.
    """
    half_w = width / 2.0
    half_h = height / 2.0

    # Blender samples at pixel centres, hence the half-pixel offsets.
    xn = (x + 0.5 - half_w) / half_w
    yn = (y + 0.5 - half_h) / half_h

    r2 = xn * xn + yn * yn

    # Beyond r = 1/sqrt(k) the map folds outward points back toward the centre,
    # so a landmark far outside the field of view would be reported inside the
    # image.  Blender draws the same line: its shader writes a transparent pixel
    # once distortion * r^2 exceeds 1.
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

    The effect node groups are Blender 5.x built-ins loaded from
    compositing_nodes_essentials.blend — the same ones in the Compositor Add menu
    under Camera & Lens Effects / Creative::

      RenderLayers → ChromaticAberration → SensorNoise → BarrelDistortion
      → Transform(1.2×) → Add(BlackMask, Transform) → [RGBtoBW]
      → Vignette → Viewer + GroupOutput

    Open preview.blend in the Compositor editor to adjust parameters by hand.
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
    # ShaderNodeMix(RGBA, ADD) replaces the removed CompositorNodeMixRGB:
    # A = black image, B = transformed image, result = A + B.
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


def _oriented_camera(camera: Camera, rotate: float) -> Camera:
    """Return *camera* as mounted, with the sensor turned by *rotate* degrees.

    A quarter turn stands the sensor on its end, so the file itself becomes
    portrait: 640x480 is written as 480x640, with the scene upright inside it.
    That is what a headset does with a camera mounted sideways — the readout is
    rotated back before it is stored, rather than left lying on its side.

    Swapping the resolution also swaps the intrinsics, so the same rays land on
    the same features and only the frame around them changes.  Because every
    consumer reads the camera's own resolution and intrinsics, returning one
    object here keeps the render, the compositor and the projected ground truth
    in agreement without any of them knowing about rotation.

    Half turns need no swap: the frame keeps its shape and only the roll in
    _head_camera_transform turns the picture.
    """
    if rotate % 180 == 0:
        return camera

    width, height = camera.resolution
    params = dict(camera.intrinsics.parameters)
    for a, b in (("fx", "fy"), ("cx", "cy")):
        if a in params and b in params:
            params[a], params[b] = params[b], params[a]

    turned = copy.copy(camera)
    turned.resolution = (height, width)
    turned.intrinsics = CameraModel(name=camera.intrinsics.name, parameters=params)
    return turned


def _sensor_roll_deg(rotate: float) -> float:
    """Roll about the optical axis left over once the frame shape is accounted for.

    A quarter turn is carried entirely by _oriented_camera swapping the frame, so
    it needs no roll.  The two quarter turns differ by which edge ends up at the
    top, which is a half turn apart, and a half turn cannot be expressed by the
    frame shape at all.
    """
    return 180.0 if rotate % 360 in (180.0, 270.0) else 0.0


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
    """Evaluate *action* at *source_frame* and snapshot the bones it drives.

    Only the action's own bones are captured: snapshotting the whole armature
    would make every pose overwrite the others, so a right-hand pose would reset
    the left hand and the head.  Restricting it to the animated bones is what
    lets poses from different groups layer.  The result is plain data, so the
    source action can be unassigned before it is written into the baked timeline.
    """
    driven = _action_bones(action)
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
        if pb.name in driven
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


def _rest_snapshot(arm_obj: bpy.types.Object, bones: set[str]) -> dict[str, dict]:
    """Snapshot of *bones* at the armature's rest pose.

    Rest is the identity transform in pose space — the T-pose the model was built
    in — so transforms are written directly rather than read off the rig, which
    may be sitting in whatever pose the .blend was last saved with.

    Custom properties are read from the live rig instead: they are switches like
    IK/FK blends, not pose values, so guessing a default would change how the rig
    behaves rather than where it is.  Call before any pose asset is applied,
    while those properties still hold their as-saved values.
    """
    return {
        pb.name: {
            "loc": [0.0, 0.0, 0.0],
            "rot_q": [1.0, 0.0, 0.0, 0.0],
            "rot_e": [0.0, 0.0, 0.0],
            "rot_a": [0.0, 0.0, 1.0, 0.0],
            "scale": [1.0, 1.0, 1.0],
            "props": _custom_props(pb),
        }
        for pb in arm_obj.pose.bones
        if pb.name in bones
    }


def _bake_pose_motion(
    arm_obj: bpy.types.Object,
    motion: PoseMotion,
    fps: float,
    action_name: str = "_phanesim_poisson",
) -> bpy.types.Action:
    """Bake a pose motion description into a single keyframed action.

    Every event contributes one key at its own ``t_ns`` and Blender's F-curve
    interpolation supplies the frames between, which is why the render loop only
    has to call frame_set.

    Every bone the timeline touches is also keyed at rest on frame 0.  An F-curve
    holds its first keyframe's value for every frame *before* it, so a bone whose
    first event comes late would otherwise show that pose from the opening frame
    — a left hand already in a whole-body pose a second before the event that
    causes it.

    Args:
        arm_obj:     The armature to bake onto.
        motion:      Timeline to bake.
        fps:         Frames per second the timeline is sampled at.
        action_name: Name for the created action.

    Returns:
        The baked action, already assigned to *arm_obj*.
    """
    assets = _resolve_pose_assets(motion)

    # Read before pass 1 applies any pose asset, so the custom properties in the
    # snapshot are the rig's own rather than some pose's.
    driven: set[str] = set()
    for name in motion.assets():
        driven |= _action_bones(assets[name])
    rest = _rest_snapshot(arm_obj, driven)

    def to_frame(t_ns: int) -> int:
        return int(round(t_ns * fps / NS_PER_SECOND))

    # Plan every key up front: one per event, since every asset is a single pose.
    # Events landing on the same frame are ordered whole-body first, so a limb
    # pose written at that instant refines the body pose instead of being
    # overwritten by it.  Both hands are pinned to the end of the clip, which is
    # exactly where a whole-body pose can also fall.
    ordered = sorted(motion.events, key=lambda e: (e.t_ns, e.group != "body"))
    plan = [(to_frame(e.t_ns), e.asset, int(assets[e.asset].frame_range[0])) for e in ordered]

    # Pass 1 — capture each distinct (asset, source frame) once.  Switching
    # actions is the expensive part, so all reads happen before any writes and
    # repeated poses reuse their snapshot.
    snapshots: dict[tuple[str, int], dict[str, dict]] = {}
    for _, asset_name, src_frame in plan:
        key = (asset_name, src_frame)
        if key not in snapshots:
            snapshots[key] = _capture_pose(arm_obj, assets[asset_name], src_frame)

    # Pass 2 — write the timeline into a fresh action.
    if old := bpy.data.actions.get(action_name):
        bpy.data.actions.remove(old)
    baked = bpy.data.actions.new(name=action_name)
    _assign_action(arm_obj, baked)

    # Rest first, so that events at frame 0 overwrite it for their own bones and
    # every other bone still has somewhere to start from.
    _key_pose(arm_obj, rest, 0)
    for frame, asset_name, src_frame in plan:
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

    print(f"[phanesim] Baked {len(motion.events)} pose event(s) over a rest pose of {len(rest)} bone(s) at frame 0.")
    return baked


def _load_body_model(model_path: Path, armature_name: str) -> bpy.types.Object:
    """Open the body .blend as the working scene and return its armature.

    Opened rather than appended so the model keeps its own materials, lights and
    — crucially — its pose assets, which live as actions in that file.  This
    replaces the current scene, so it must run before any render setup.

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


def _action_bones(action: bpy.types.Action) -> set[str]:
    """Names of the pose bones an action actually animates."""
    names: set[str] = set()
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fcurve in channelbag.fcurves:
                    if 'pose.bones["' in fcurve.data_path:
                        names.add(fcurve.data_path.split('"')[1])
    return names


def _asset_group(bones: set[str]) -> str:
    """Classify an asset by the side of the body it drives.

    Derived from the animated bones rather than the asset name, so a renamed pose
    keeps working.  Different groups touch disjoint bones and can be applied
    together; a "body" asset touches both sides at once.
    """
    if not bones:
        return "body"
    left = {b for b in bones if b.endswith(".L")}
    right = {b for b in bones if b.endswith(".R")}
    if left and not right:
        return "left"
    if right and not left:
        return "right"
    if not left and not right:
        return "head"
    return "body"


def enumerate_pose_assets(model_path: Path, output_json: Path) -> None:
    """Write the pose assets found in *model_path* to *output_json*.

    Runs inside Blender for `generate-motion`, which needs the asset list before
    it can sample a timeline.  Only single-frame asset-marked actions count:
    working actions are not poses, and multi-frame ones are not either.
    """
    bpy.ops.wm.open_mainfile(filepath=str(model_path))

    assets: list[PoseAsset] = []
    skipped: list[str] = []
    for action in bpy.data.actions:
        if action.asset_data is None:
            continue
        first, last = int(action.frame_range[0]), int(action.frame_range[1])
        if last > first:
            # Only single-frame poses are supported.  Reading frame one of a
            # multi-frame action and calling it a pose would silently produce a
            # timeline nobody authored, so it is reported and left out.
            skipped.append(action.name)
            continue
        assets.append(PoseAsset(name=action.name, frame=first, group=_asset_group(_action_bones(action))))
    assets.sort(key=lambda a: a.name)

    # Which accessories this model actually carries.  model2 has none, so a plan
    # that dressed its clips anyway would write a sequence.json claiming a watch
    # that never appears in the pixels.
    available = sorted(s for s, root in ACCESSORIES.items() if bpy.data.objects.get(root) is not None)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {"model": str(model_path), "assets": [a.to_dict() for a in assets], "accessories": available},
            indent=2,
        )
    )
    print(f"[phanesim] Found {len(assets)} pose asset(s) and {len(available)} accessory(s) in {model_path.name}")
    if skipped:
        print(f"[phanesim] Skipped {len(skipped)} multi-frame action(s), which are not poses: {sorted(skipped)}")


def _mute_scene_lights(scene: bpy.types.Scene) -> int:
    """Hide every light object in the scene from rendering.

    An HDRI is a complete lighting environment on its own, and the body model
    ships with its own key and fill lamps; leaving both on stacks two lighting
    rigs and blows the skin out to flat white.

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
    own, so they inherit that bone's rotation — as the 2D landmarks do for
    position.
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


def _head_camera_transform(
    arm_obj: bpy.types.Object,
    head_cam: HeadCamera,
    sweep: CameraSweep | None = None,
    progress: float = 0.0,
    roll_deg: float = 0.0,
) -> Transform:
    """Compute the head-mounted camera pose for the current pose of the rig.

    Position and direction are both rigidly attached to the head, carried through
    the head bone's rest-to-pose delta.  The camera never turns to follow the
    hands, so they enter and leave the frame on their own, as on a real headset.

    Args:
        arm_obj:  The posed armature the camera is anchored to.
        head_cam: Camera placement.
        sweep:    Optional steady turn away from the head's direction, growing
                  with *progress* so the clip opens on the rest view.
        progress: How far through the clip this frame is, 0.0 to 1.0.
        roll_deg: Constant rotation about the optical axis.  90 puts the sensor
                  on its side: the frame stays 640x480 but the scene inside it
                  is portrait, as on a headset with a rotated camera.

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

    if (sweep is not None and sweep.degrees) or roll_deg:
        # Columns are the camera axes expressed in world space, so a rotation
        # right-multiplied here acts in the camera's own frame: X is its right
        # axis, Y its down axis, Z where it looks.
        basis = mathutils.Matrix(
            ((right.x, down.x, forward.x), (right.y, down.y, forward.y), (right.z, down.z, forward.z))
        )
        if sweep is not None and sweep.degrees:
            basis = basis @ mathutils.Euler(sweep.euler_at(progress), "XYZ").to_matrix()
        if roll_deg:
            # Z is the optical axis, so this spins the sensor without changing
            # where the camera looks.
            basis = basis @ mathutils.Euler((0.0, 0.0, math.radians(roll_deg)), "XYZ").to_matrix()
        right = mathutils.Vector((basis[0][0], basis[1][0], basis[2][0]))
        down = mathutils.Vector((basis[0][1], basis[1][1], basis[2][1]))
        forward = mathutils.Vector((basis[0][2], basis[1][2], basis[2][2]))

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

    The compositor distorts and rescales after rendering, so the same forward map
    is applied here; otherwise the CSV would describe an undistorted image that
    was never written to disk.  See distort_pixel.
    """
    p_cam = T_cam_world.apply(np.array([point_world.x, point_world.y, point_world.z], dtype=np.float64))
    if p_cam[2] <= 0:
        return float("nan"), float("nan")

    u, v = project_point(p_cam, camera.intrinsics)
    width, height = camera.resolution
    return distort_pixel(u, v, width, height, camera.distortion, camera.lens_scale)


# Accessories worn by model1, keyed by the short name the CLI accepts.  Each
# value is the root object; it is parented to a bone, and the mesh it carries
# hangs below it, so showing or hiding one means walking its children too.
ACCESSORIES: dict[str, str] = {
    "ring1": "Ring_R_Root",  # right middle finger
    "ring2": "Ring_Wedding_L",  # left ring finger
    "watch1": "Sketchfab_watch",  # left forearm
    "band1": "Sketchfab_wristband",  # right forearm
}


def _set_accessories(visible: set[str]) -> None:
    """Show only the accessories in *visible*, hiding every other one.

    Always explicit: there is no "leave the file as it was saved" mode, because
    model1.blend is saved with all four showing and inheriting that silently put
    accessories into renders nobody asked for.  An empty set means bare hands.

    An unknown name is an error rather than a silent no-op, so a typo cannot
    quietly render a bare hand.

    Args:
        visible: Short names from ACCESSORIES; empty for none.

    Raises:
        ValueError: If a name is not a known accessory.
    """
    unknown = visible - ACCESSORIES.keys()
    if unknown:
        raise ValueError(f"unknown accessory {sorted(unknown)}; known: {sorted(ACCESSORIES)}")

    shown, missing = [], []
    for short, root_name in ACCESSORIES.items():
        root = bpy.data.objects.get(root_name)
        if root is None:
            # A model that simply does not have this accessory; only worth
            # reporting if it was asked for.
            if short in visible:
                missing.append(short)
            continue
        hide = short not in visible
        for obj in [root, *root.children_recursive]:
            obj.hide_render = hide
            obj.hide_viewport = hide
        if not hide:
            shown.append(short)

    print(f"[phanesim] Accessories: {', '.join(shown) if shown else 'none'}.")
    if missing:
        print(f"[phanesim] Not in this model, skipped: {', '.join(missing)}.")


def render_body_sequence(
    seq: BodySequence,
    output_path: Path,
    frames: int | None = None,
    write_3d: bool = False,
    accessories: set[str] | None = None,
    camera_sweep: CameraSweep | None = None,
    rotate: float = 0.0,
) -> None:
    """Render a BodySequence: pose-asset motion seen by a head-mounted camera.

    Bakes keyframes and collects projections, then renders the whole animation in
    one call so EEVEE initialises once.  Several motions in one sequence each
    become their own take under <output_path>/<motion name>/.

    Args:
        seq:         Loaded BodySequence.
        output_path: Root output directory for this sequence.
        frames:      Frames to render per motion, overriding seq.frames.
        write_3d:    Also write joints_3d.csv with world-space joint poses.
        accessories: Short names to wear, overriding the sequence.  None uses
                     the sequence's own list, which is empty unless it says
                     otherwise.
        camera_sweep: Optional steady turn of the camera across the clip.
        rotate:       Degrees to rotate the sensor about the optical axis; 90
                      makes a portrait view inside the same landscape frame.
    """
    for motion in seq.hand_motions:
        take_path = output_path if len(seq.hand_motions) == 1 else output_path / motion.name
        _render_body_take(
            seq,
            motion,
            take_path,
            frames if frames is not None else seq.frames,
            write_3d,
            accessories,
            camera_sweep,
            rotate,
        )


def _sample_timestamps(motion: PoseMotion, frames: int) -> tuple[Timestamps, float]:
    """Choose the timestamps to render, and the sampling rate they imply.

    The frame count is the only control: samples spread evenly over the whole
    timeline, so 2 gives the first and last frame and any count covers the entire
    motion rather than a truncated opening.  The rate is a consequence.

    Returns:
        The sample timestamps in nanoseconds, and the effective rate in Hz that
        the spacing corresponds to.
    """
    t_end = int(motion.t_end_ns)
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
    frames: int,
    write_3d: bool = False,
    accessories: set[str] | None = None,
    camera_sweep: CameraSweep | None = None,
    rotate: float = 0.0,
) -> None:
    """Render one pose motion of a BodySequence into *output_path*."""
    rig = seq.body_rig
    head_cam = rig.head_camera

    # Opening the model replaces the scene, so it must happen before any setup.
    arm_obj = _load_body_model(rig.body.model, rig.body.armature)
    scene = bpy.context.scene
    _set_accessories(set(seq.accessories) if accessories is None else accessories)

    if seq.hdri:
        _setup_world_hdri(scene, seq.hdri)
        muted = _mute_scene_lights(scene)
        print(f"[phanesim] HDRI lighting: muted {muted} light(s) shipped with the model.")

    joint_columns, hands = _body_joint_columns(seq)

    for camera_index, configured in enumerate(rig.cameras):
        camera = _oriented_camera(configured, rotate)
        cam_label = camera.name or f"cam{camera_index}"
        cam_dir = output_path / f"cam_{cam_label}"
        cam_dir.mkdir(parents=True, exist_ok=True)

        timestamps, effective_hz = _sample_timestamps(motion, frames)
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

            progress = frame_idx / max(1, len(timestamps) - 1)
            T_world_cam = _head_camera_transform(arm_obj, head_cam, camera_sweep, progress, _sensor_roll_deg(rotate))
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


def render_clips(dataset_dir: Path, overwrite: bool = False) -> None:
    """Render every clip under *dataset_dir*, skipping the ones already done.

    One Blender process walks the whole list, so the interpreter and its addons
    start once rather than once per clip.  A clip is marked done only after its
    frames are written, so killing the process mid-run costs at most the clip in
    flight: run the command again and it picks up from there.

    That also covers the process dying on its own — if a long run leaks memory
    and is killed, resuming is the same command.

    Args:
        dataset_dir: Directory holding clip_NNNNN/ subdirectories.
        overwrite:   Re-render clips already marked done.
    """
    clips = clip_dirs(dataset_dir)
    for stray in stray_clip_dirs(dataset_dir):
        print(f"[phanesim] WARNING: {stray.name}/ holds a clip one level too deep and will not be rendered.")
    if not clips:
        raise RuntimeError(f"no clips found in {dataset_dir}; run `phanesim plan-clips` first")

    todo = [d for d in clips if overwrite or not is_done(d)]
    done_already = len(clips) - len(todo)
    print(f"[phanesim] {len(clips)} clip(s), {done_already} already done, {len(todo)} to render.")

    for i, clip in enumerate(todo, 1):
        seq = BodySequence.from_path(clip / SEQUENCE_FILE)
        print(f"[phanesim] ({i}/{len(todo)}) {clip.name}")
        # Accessories come from the clip's own sequence.json.  joints_3d costs
        # nothing extra and is ground truth, so a dataset always gets it; the
        # debug overlays are not written, since keypoints drawn onto the pixels
        # would be learned as features.
        render_body_sequence(seq, clip, write_3d=True)
        mark_done(clip, seq.frames)

    print(f"[phanesim] Done: {len(todo)} clip(s) rendered, {done_already} skipped.")


def preview_body_sequence(
    seq: BodySequence,
    save_path: str | None = None,
    frames: int | None = None,
    accessories: set[str] | None = None,
    camera_sweep: CameraSweep | None = None,
    rotate: float = 0.0,
) -> None:
    """Bake a BodySequence as keyframes and optionally save it as a .blend file.

    Only the first camera and first motion are baked; open the result in
    Blender's GUI to scrub the timeline and inspect the compositor.
    """
    rig = seq.body_rig
    head_cam = rig.head_camera
    motion = seq.hand_motions[0]
    camera = _oriented_camera(rig.cameras[0], rotate)

    arm_obj = _load_body_model(rig.body.model, rig.body.armature)
    scene = bpy.context.scene
    _set_accessories(set(seq.accessories) if accessories is None else accessories)

    if seq.hdri:
        _setup_world_hdri(scene, seq.hdri)
        _mute_scene_lights(scene)

    timestamps, effective_hz = _sample_timestamps(motion, frames if frames is not None else seq.frames)
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
        # progress is passed so the preview sweeps exactly as the render will.
        progress = frame_idx / max(1, len(timestamps) - 1)
        T_world_cam = _head_camera_transform(arm_obj, head_cam, camera_sweep, progress, _sensor_roll_deg(rotate))
        mat = mathutils.Matrix(T_world_cam.as_matrix().tolist()) @ _OPENCV_TO_BLENDER_CAM
        location, rotation, _ = mat.decompose()
        cam_obj.location = location
        cam_obj.rotation_quaternion = rotation
        cam_obj.keyframe_insert(data_path="location", frame=frame_idx)
        cam_obj.keyframe_insert(data_path="rotation_quaternion", frame=frame_idx)

    print(f"[phanesim] Preview ready: {len(timestamps)} frames at {scene.render.fps} fps.")
    if save_path:
        bpy.ops.wm.save_as_mainfile(filepath=save_path)
        print(f"[phanesim] Saved: {save_path}")
