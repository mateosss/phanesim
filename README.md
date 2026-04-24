<!--
Copyright 2026, Mateo de Mayo.
SPDX-License-Identifier: BSD-3-Clause
-->

# Phanesim: Realistic datasets

This project is a synthetic dataset generator for computer vision: aimed at hand
tracking first, and later visual-inertial SLAM, and structure from motion will
be added as well. The initial motivation is to generate synthetic data for 2d
joint detection for hand tracking.

## Overview

1. It uses blender with EEVEE
2. It works completely in headless mode with a CLI interface
3. It also provides a UI in blender for easier project setup and debugging, but the UI is not required to generate datasets.
4. It is easy to parallelize the dataset generation to use in SLURM.
5. It simulates multiple camera artifacts
6. It can simulate camera rigs with multiple cameras. Other sensors may be added in the future like IMU or Lidar.
7. It provides versatile project descriptions through json files describing camera rigs, camera properties, hand properties, camera motion, sensor motion, hand motion, etc

## Technical details

- Uses uv for package management
- Uses ruff for linting
- Uses ruff for formatting
- Uses pyright for type checking
- Uses pytest for testing
- Uses git lfs for large file storage
- Uses SPDX license headers, and has a BSD-3-Clause license
- Uses github actions for CI: ruff, pyright, pylint, pytest, SPDX check
- We use numpy for all the math types, only convert to blender types when needed.

## Other details

- We prefer csv files whenever needed for tabular data. Use pandas with pyarrow to read them.
- We prefer json for configurations and project descriptions
- We prefer int64 timestamps in nanoseconds
- We use the convention T_a_b for the transform from frame b to frame a, i.e. p_a = T_a_b * p_b.
- We can use X_a_b with X = T for 4x4 SE(3), q for quaternion, t for translation, R for rotation matrix, etc. The same convention applies for the subscripts.

## Data specification

This is a prelimianry specifications of the types and data we are working with
in a python-ish syntax. Use this as a guideline but don't assume it's the source
of all truth. Please update it as needed.

```python
# We define some basic types like as numpy arrays like this, we dont have a way to enforce the
# shape of the arrays but we can use type hints to indicate the expected shape.
type Scalar = np.float32
type Timestamp = np.int64 # in nanoseconds
type Duration = np.int64 # in nanoseconds
type Path = pathlib.Path
type Vector3 = npt.NDArray[Scalar] # xyz
type Vector4 = npt.NDArray[Scalar] # xyzw
type Quaternion = Vector4 # xyzw
type Matrix4x4 = npt.NDArray[Scalar] # 4x4
type Color = npt.NDArray[Scalar] # rgb
type Timestamps = npt.NDArray[Timestamp] # Nx1
type Positions = npt.NDArray[Vector3] # Nx3
type Quaternions = npt.NDArray[Quaternion] # Nx4

class Transform:
    pos: Vector3
    quat: Quaternion

    @property def mat() -> Matrix4x4: ... # return the 4x4 matrix representation of the transform
    @property def inv() -> Transform: ... # return the inverse transform
    @property def rotmat() -> Matrix3x3: ... # return the 3x3 rotation matrix of the transform
    def __mul__(self, other: Transform) -> Transform: ... # compose two transforms
    def __mul__(self, other: Vector3) -> Vector3: ... # apply the transform to a point

class Vignette:
    # TODO: Define what vignette is, in basalt it's a spline, see:
    #  https://gitlab.freedesktop.org/mateosss/basalt-headers/-/blob/e6db0fb84c69614bc4923fde6c52154c221c1768/include/basalt/calibration/calibration.hpp#L152-158
    path: Path # otherwise maybe just a path to a png like this: https://cvg.cit.tum.de/_detail/data/datasets/visual-inertial-dataset/vingette_0.png?id=data%3Adatasets%3Avisual-inertial-dataset

class CameraModel:
    name: str # e.g. kb4, rt8, pinhole, etc
    parameters: Dict[str, Scalar] # e.g. for kb4: fx, fy, cx, cy, k1, k2, k3, k4

class Camera:
    name: Optional[str] # simple human readable name
    T_b_c: Transform
    intrinsics: CameraModel
    resolution: Tuple[int, int] # w, h
    frequency: Scalar # fps in Hz
    pixel_format: str # e.g. GRAY8, GRAY12, RGB24, YUYV422, etc
    shutter: GLOBAL | ROLLING # Use python's Enum for this
    vignette: Vignette
    exposure: Duration # 0 means none, -1 means auto, otherwise it's the exposure time in nanoseconds
    gain: int # 0 means none, -1 means auto, otherwise it's the gain value in some unit to be defined
    lense_flare: bool # whether to simulate lens flare or not
    chromatic_aberration: bool # whether to simulate chromatic aberration or not
    motion_blur: bool # whether to simulate motion blur or not


class Hand:
    # TODO: The goal of this class will be to parametrize the hand model
    # something like MANO but much simpler and with direct interpretation.
    # We won't use MANO because of its license. For now we'll limit ourselves
    # to simple scaling and texturing parameters.
    name: Optional[str] # simple human readable name
    model: Path # e.g., path to the file with the mesh and rig
    scale_length: Scalar = 1.0
    scale_breadth: Scalar = 1.0
    scale_thickness: Scalar = 1.0
    texture: Optional[Path] = None # path to a texture file, e.g. png
    color_multiply: Optional[Color] # an optional color to multiply the texture by

class CameraMotion: # Camera trajectory
    source: Path # path to a csv file with columns: timestamp, px, py, pz, qx, qy, qz, qw
    ts: Timestamps
    xyz: Positions
    quats: Quaternions

    def __init__(self, source: Path): ...
    def get_pose(timestamp: Timestamp) -> Transform: ... # get the pose of the camera at a given timestamp, use interpolation if needed


class HandMotion:
    # TODO: We need to see how the hand motion is currently parametrized, globally? wrt wrist bone? relative to parent bone?
    source: Path # path to a csv file with hand joints with columns: timestamp, joint1_x, joint1_y, joint1_z, joint1_qx, joint1_qy, joint1_qz, joint1_qw, ..., jointN_x, jointN_y, jointN_z, jointN_qx, jointN_qy, jointN_qz, jointN_qw
    joint_names: List[str] # joint names as appearing on the csv, length J
    ts: Timestamps # Nx1
    joints_xyz: Positions # NxJx3, J is number of joints
    joints_quats: Quaternions # NxJx4

class CameraHandRig:
    # TODO: This models the relationship between the cameras and the hand.
    # Initially let's just "parent" the hands to the head pose.
    # In the future we could add a still simple but better model:
    # - we split yaw rotations in 8.
    # - head translates -> neck translates with lag -> shoulder translates with lag
    # - head rotates -> neck rotates with lag only when current 8-piece changed -> shoulder rotates with lag.
    # - shoulder is already in hand skeleton so the rest of the bone chain follows
    cameras: List[Camera] # C
    hands: List[Hand] # H
    T_c_h: List[List[Transform]] # CxH, T_c_h[c, h] is hand h pose in camera c frame (converts hand joint positions from hand frame to camera frame)

class Sequence:
    name: str # human readable name
    camhand_rig: CameraHandRig
    cam_motions: List[CameraMotion]
    hand_motions: List[HandMotion]
```
