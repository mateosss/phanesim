# Copyright 2026, Yutong Wan.
# SPDX-License-Identifier: BSD-3-Clause

"""OpenXR-compatible 21-landmark hand skeleton definition.

Landmark names follow the OpenXR hand tracking convention.  Non-thumb
metacarpals are omitted, which leaves exactly 21 joints:

    Wrist
    Thumb:  Metacarpal, Proximal, Distal, Tip
    Index / Middle / Ring / Little:  Proximal, Intermediate, Distal, Tip

Two rigs are supported, each with its own bone naming:
  * the standalone hand rig                    -> HAND_LANDMARKS
  * the Rigify full-body rigs in data/models/  -> rigify_hand_landmarks(side)

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

# Rigify finger-chain names, in OpenXR landmark order.  Rigify numbers each
# finger chain .01/.02/.03 from the knuckle outward, and the thumb chain starts
# one joint earlier at the metacarpal — so thumb.01/02/03 lines up exactly with
# OpenXR Metacarpal/Proximal/Distal.
_RIGIFY_FINGERS: list[tuple[str, str]] = [
    # (OpenXR finger name, Rigify chain name)
    ("Index", "f_index"),
    ("Middle", "f_middle"),
    ("Ring", "f_ring"),
    ("Little", "f_pinky"),
]


def rigify_hand_landmarks(side: str) -> list[tuple[str, str, str]]:
    """Return the 21 landmark definitions for one hand of a Rigify body rig.

    DEF- bones are used because they are the ones that actually deform the mesh.
    The obvious alternative, the ORG- chain, is unreliable once posed: ORG-hand
    carries two Copy Transforms constraints, one from the FK chain and one from
    the IK chain, blended by the arm's IK/FK switch.  When an arm is in IK the
    ORG bone can end up centimetres from the hand it is supposed to represent
    (measured: 6.4 cm on one arm, 0 cm on the other, from the same pose).  DEF
    bones follow the tweak bones and therefore the mesh.

    The two chains are identical in the rest pose and agree on 41 of the 42
    landmarks even when posed, so this only changes the wrist — but it changes
    it from wrong to right.

    Args:
        side: "left"/"L" or "right"/"R" (case-insensitive).

    Returns:
        A list of (openxr_name, source_type, bone_name) triples in the same
        order and format as HAND_LANDMARKS.

    Raises:
        ValueError: If *side* is not recognisable as left or right.
    """
    s = side.strip().lower()
    if s in ("l", "left"):
        suffix = "L"
    elif s in ("r", "right"):
        suffix = "R"
    else:
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")

    landmarks: list[tuple[str, str, str]] = [
        ("Wrist", "arm_head", f"DEF-hand.{suffix}"),
        ("ThumbMetacarpal", "arm_head", f"DEF-thumb.01.{suffix}"),
        ("ThumbProximal", "arm_head", f"DEF-thumb.02.{suffix}"),
        ("ThumbDistal", "arm_head", f"DEF-thumb.03.{suffix}"),
        ("ThumbTip", "arm_tail", f"DEF-thumb.03.{suffix}"),
    ]
    for openxr_name, chain in _RIGIFY_FINGERS:
        landmarks += [
            (f"{openxr_name}Proximal", "arm_head", f"DEF-{chain}.01.{suffix}"),
            (f"{openxr_name}Intermediate", "arm_head", f"DEF-{chain}.02.{suffix}"),
            (f"{openxr_name}Distal", "arm_head", f"DEF-{chain}.03.{suffix}"),
            (f"{openxr_name}Tip", "arm_tail", f"DEF-{chain}.03.{suffix}"),
        ]
    return landmarks


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
