# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""MediaPipe-compatible 21-landmark hand skeleton definition.

Kept in a bpy-free module so it can be imported by both render.py (inside
Blender) and cli.py (regular Python environment).
"""

from __future__ import annotations

# Each entry: (output_name, source_type, blender_bone_name)
#   "csv_head"  — joint_pose.translation from hand_motion CSV (bone HEAD in world space)
#   "arm_head"  — bone HEAD read directly from the posed armature (not in CSV)
#   "arm_tail"  — bone TAIL read directly from the posed armature (fingertips)
HAND_LANDMARKS: list[tuple[str, str, str]] = [
    ("Wrist", "arm_head", "Wrist"),
    ("ThumbCMC", "arm_head", "ThumbMetacarpal"),
    ("ThumbMCP", "arm_head", "ThumbProximal"),
    ("ThumbIP", "arm_head", "ThumbDistal"),
    ("ThumbTip", "arm_tail", "ThumbDistal"),
    ("IndexMCP", "arm_head", "IndexProximal"),
    ("IndexPIP", "arm_head", "IndexIntermediate"),
    ("IndexDIP", "arm_head", "IndexDistal"),
    ("IndexTip", "arm_tail", "IndexDistal"),
    ("MiddleMCP", "arm_head", "MiddleProximal"),
    ("MiddlePIP", "arm_head", "MiddleIntermediate"),
    ("MiddleDIP", "arm_head", "MiddleDistal"),
    ("MiddleTip", "arm_tail", "MiddleDistal"),
    ("RingMCP", "arm_head", "RingProximal"),
    ("RingPIP", "arm_head", "RingIntermediate"),
    ("RingDIP", "arm_head", "RingDistal"),
    ("RingTip", "arm_tail", "RingDistal"),
    ("PinkyMCP", "arm_head", "LittleProximal"),
    ("PinkyPIP", "arm_head", "LittleIntermediate"),
    ("PinkyDIP", "arm_head", "LittleDistal"),
    ("PinkyTip", "arm_tail", "LittleDistal"),
]

# Skeleton connectivity: index pairs into HAND_LANDMARKS.
HAND_CONNECTIONS: list[tuple[int, int]] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),  # thumb
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),  # index
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),  # middle
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),  # ring
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),  # pinky
    (5, 9),
    (9, 13),
    (13, 17),  # palm cross-connections
]

# Per-landmark RGB color for debug overlays (finger-grouped).
LANDMARK_COLORS: list[tuple[int, int, int]] = (
    [(255, 255, 255)]  # 0   Wrist — white
    + [(255, 80, 80)] * 4  # 1–4  Thumb — red
    + [(255, 180, 0)] * 4  # 5–8  Index — orange
    + [(80, 220, 80)] * 4  # 9–12 Middle — green
    + [(80, 120, 255)] * 4  # 13–16 Ring — blue
    + [(180, 80, 255)] * 4  # 17–20 Pinky — purple
)
