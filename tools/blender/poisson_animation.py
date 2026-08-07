# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

# This script generates a Poisson-distributed sequence of hand poses and animations

import os
import random

import bpy  # type: ignore[import-not-found]  # bpy is Blender's embedded Python module

# ==================== Configuration ====================
events_per_second = 0.4
total_frames = 500
fps = bpy.context.scene.render.fps
rig_name = "rig"

# sample poses:

# Static poses (single frame) — Poisson-sampled transitions between these
pose_assets = [
    "hand_wave",
    "Right_Pat",
    "Right_ThumbUp",
    "Right_fist",
    "Right_ok",
    "Right_one.002",
    "Right_two",
    "Right_three",
    "Right_four",
    "Left_default",
    "Left_grab",
    "Left_pointing",
    "Look_At_Hand",
]
# Dynamic animations (multi-frame) — played back in full
action_assets = ["Wave_Animation"]

# Path to external asset library (This will be deleted, just to keep safe if some assets are missing)
asset_library_path = r"C:\Users\wanyt\Documents\Blender\Assets"
# =======================================================


# ---------- 1. Build an index of asset name -> .blend file ----------
# Assets visible in the asset browser are not necessarily in bpy.data.actions
# of the current file. Assets from external libraries must be appended into
# the current file before they can be accessed.
# (If all poses are already in the current file, this index will be empty — that's expected.)
def build_asset_index(lib_path):
    index = {}
    if not os.path.isdir(lib_path):
        return index
    for root, _dirs, files in os.walk(lib_path):
        for fn in files:
            if fn.lower().endswith(".blend"):
                fp = os.path.join(root, fn)
                try:
                    with bpy.data.libraries.load(fp, assets_only=True) as (src, _dst):
                        for a in src.actions:
                            index.setdefault(a, fp)
                except Exception:
                    pass
    return index


asset_index = build_asset_index(asset_library_path)


# ---------- 2. Ensure the action is in the current file; append from the external library if not ----------
def ensure_action(name):
    act = bpy.data.actions.get(name)
    if act:
        return act
    fp = asset_index.get(name)
    if not fp:
        print(f"WARNING: asset '{name}' not found locally or in asset library — skipping")
        return None
    with bpy.data.libraries.load(fp, link=False, assets_only=True) as (src, dst):
        if name in src.actions:
            dst.actions = [name]
    act = bpy.data.actions.get(name)
    if act:
        print(f"Imported from asset library: '{name}'")
    return act


# ---------- 3. Safely assign action + slot to the armature (Blender 4.4+ slot/layer system) ----------
# In Blender 5.x, assigning action alone is not enough — action_slot must also
# be bound, otherwise the action evaluates as empty.
def assign_action(rig, action):
    ad = rig.animation_data
    ad.action = action
    if action is not None:
        try:
            if len(action.slots) > 0:
                ad.action_slot = action.slots[0]
        except Exception as e:
            print(f"  (slot binding note: {e})")


rig = bpy.data.objects.get(rig_name)

if rig and rig.type == "ARMATURE":
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="POSE")

    if rig.animation_data:
        rig.animation_data.action = None
    else:
        rig.animation_data_create()

    main_action = bpy.data.actions.new(name="Programmatic_Poisson_Animation")
    assign_action(rig, main_action)

    def transfer_pose(source_action_name, source_frame, target_frame):
        source_action = ensure_action(source_action_name)

        # Hard stop: treat a missing asset as a fatal error
        if not source_action:
            print(f"ERROR: transfer failed — '{source_action_name}' not found locally or in asset library")
            return False

        assign_action(rig, source_action)

        # When source_frame is None, read from the asset's actual first keyframe
        if source_frame is None:
            read_frame = int(source_action.frame_range[0])
        else:
            read_frame = int(source_frame)
        bpy.context.scene.frame_set(read_frame)
        bpy.context.view_layer.update()

        bone_data = {}
        for pb in rig.pose.bones:
            bone_data[pb.name] = {
                "loc": list(pb.location),
                "rot_q": list(pb.rotation_quaternion),
                "rot_e": list(pb.rotation_euler),
                "rot_a": list(pb.rotation_axis_angle),
                "scl": list(pb.scale),
                "props": {prop: pb[prop] for prop in pb.keys() if prop not in "_RNA_UI"},
            }

        # Switch back to the main action before inserting keyframes
        assign_action(rig, main_action)
        bpy.context.scene.frame_set(int(target_frame))

        for pb in rig.pose.bones:
            data = bone_data[pb.name]
            pb.location = data["loc"]
            if pb.rotation_mode == "QUATERNION":
                pb.rotation_quaternion = data["rot_q"]
            elif pb.rotation_mode == "AXIS_ANGLE":
                pb.rotation_axis_angle = data["rot_a"]
            else:
                pb.rotation_euler = data["rot_e"]
            pb.scale = data["scl"]

            for prop, val in data["props"].items():
                pb[prop] = val

            pb.keyframe_insert(data_path="location", frame=target_frame)
            if pb.rotation_mode == "QUATERNION":
                pb.keyframe_insert(data_path="rotation_quaternion", frame=target_frame)
            elif pb.rotation_mode == "AXIS_ANGLE":
                pb.keyframe_insert(data_path="rotation_axis_angle", frame=target_frame)
            else:
                pb.keyframe_insert(data_path="rotation_euler", frame=target_frame)
            pb.keyframe_insert(data_path="scale", frame=target_frame)

            for prop in data["props"].keys():
                try:
                    pb.keyframe_insert(data_path=f'["{prop}"]', frame=target_frame)
                except Exception:
                    pass

        return True

    current_frame = 1
    all_choices = pose_assets + action_assets

    # Initialise: key the first pose at frame 1 to prevent a wild interpolation at the start
    if len(pose_assets) > 0:
        transfer_pose(pose_assets[0], None, 1)

    while current_frame < total_frames:
        wait_time_seconds = random.expovariate(events_per_second)
        wait_frames = int(wait_time_seconds * fps)
        if wait_frames < int(fps * 1.5):
            wait_frames = int(fps * 1.5)

        current_frame += wait_frames
        if current_frame > total_frames:
            break

        chosen_asset = random.choice(all_choices)
        blend_frames = int(fps * 0.5)
        target_frame = current_frame + blend_frames
        if target_frame > total_frames:
            target_frame = total_frames

        # Insert a hold keyframe so the previous pose stays stable before transitioning
        bpy.context.scene.frame_set(current_frame)
        for pb in rig.pose.bones:
            pb.keyframe_insert(data_path="location", frame=current_frame)
            if pb.rotation_mode == "QUATERNION":
                pb.keyframe_insert(data_path="rotation_quaternion", frame=current_frame)
            elif pb.rotation_mode == "AXIS_ANGLE":
                pb.keyframe_insert(data_path="rotation_axis_angle", frame=current_frame)
            else:
                pb.keyframe_insert(data_path="rotation_euler", frame=current_frame)
            pb.keyframe_insert(data_path="scale", frame=current_frame)
            for prop in pb.keys():
                if prop not in "_RNA_UI":
                    try:
                        pb.keyframe_insert(data_path=f'["{prop}"]', frame=current_frame)
                    except Exception:
                        pass

        if chosen_asset in pose_assets:
            # Static pose: read from its own first keyframe (source_frame=None -> frame_range[0])
            success = transfer_pose(chosen_asset, None, target_frame)
            if success:
                print(f"Frame {current_frame} -> {target_frame}: transition to pose '{chosen_asset}'")

        elif chosen_asset in action_assets:
            # Dynamic animation: play the full asset frame_range (not a hardcoded frame count)
            src = ensure_action(chosen_asset)
            if src:
                f0 = int(src.frame_range[0])
                f1 = int(src.frame_range[1])
                anim_duration = max(1, f1 - f0)
                success = transfer_pose(chosen_asset, f0, target_frame)
                if success:
                    step = 2  # sample every 2 frames: smooth enough without too many keyframes
                    for offset in range(step, anim_duration + 1, step):
                        apply_frame = target_frame + offset
                        if apply_frame > total_frames:
                            break
                        transfer_pose(chosen_asset, f0 + offset, apply_frame)
                    current_frame = target_frame + anim_duration
                    print(
                        f"Frame {current_frame}: playing animation '{chosen_asset}' (source frames {f0}-{f1}, {anim_duration} frames)"
                    )

    bpy.context.scene.frame_set(1)
    print("Poisson animation generation complete.")
else:
    print("Armature not found — check the rig_name setting")
