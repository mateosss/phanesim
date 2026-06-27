# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""MediaPipe-compatible 21-landmark hand skeleton definition.

Kept in a bpy-free module so it can be imported by both render.py (inside
Blender) and cli.py (regular Python environment).
"""

from __future__ import annotations

# Each entry: (output_name, source_type, blender_bone_name)
#   "arm_head"  — proximal end of the bone (root/joint side); used for every joint
#   "arm_tail"  — distal end of the bone (tip side); only used for fingertips,
#                 which have no child bone so their position only exists as the
#                 parent Distal bone's tail
HAND_LANDMARKS: list[tuple[str, str, str]] = [
    ("Wrist", "arm_head", "Wrist"),
    ("ThumbMetacarpal", "arm_head", "ThumbMetacarpal"),
    ("ThumbProximal", "arm_head", "ThumbProximal"),
    ("ThumbDistal", "arm_head", "ThumbDistal"),
    ("ThumbTip", "arm_tail", "ThumbDistal"),
    ("IndexProximal", "arm_head", "IndexProximal"),
    ("IndexIntermediate", "arm_head", "IndexIntermediate"),
    ("IndexDistal", "arm_head", "IndexDistal"),
    ("IndexTip", "arm_tail", "IndexDistal"),
    ("MiddleProximal", "arm_head", "MiddleProximal"),
    ("MiddleIntermediate", "arm_head", "MiddleIntermediate"),
    ("MiddleDistal", "arm_head", "MiddleDistal"),
    ("MiddleTip", "arm_tail", "MiddleDistal"),
    ("RingProximal", "arm_head", "RingProximal"),
    ("RingIntermediate", "arm_head", "RingIntermediate"),
    ("RingDistal", "arm_head", "RingDistal"),
    ("RingTip", "arm_tail", "RingDistal"),
    ("LittleProximal", "arm_head", "LittleProximal"),
    ("LittleIntermediate", "arm_head", "LittleIntermediate"),
    ("LittleDistal", "arm_head", "LittleDistal"),
    ("LittleTip", "arm_tail", "LittleDistal"),
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
    + [(180, 80, 255)] * 4  # 17–20 Little — purple
)
