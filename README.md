<!--
Copyright 2026, Yutong Wan.
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

- Use jsom schemas for validating the project description json files
- Uses uv for package management
- Uses ruff for linting
- Uses ruff for formatting
- Uses pyright for type checking
- Uses pytest for testing
- Uses git lfs for large file storage
- Uses SPDX license headers, and has a BSD-3-Clause license
- Uses github actions for CI: ruff, pyright, pylint, pytest, SPDX check
- We use numpy for all the math types, only convert to blender types when needed.

## Development commands

```bash
uv sync --all-groups # Install with uv (including development tools)
uv run pyright # Run Pyright
uv run ruff check . # Run Ruff lint checks
uv run ruff format --check . # Run Ruff formatter check

```

## Blender workflow

For now we use this through a bootstrap.py script inside the blend file that is
autorun at the beginning and reloads all scripts in the blend file and setups
the Phanesim panel UI in `View3D > Sidebar > Phanesim`. A script reload can be
also triggered by clicking the `Reload Scripts` button, useful for iterating
without closing the blend file.

## Other details

- We prefer csv files whenever needed for tabular data. Use pandas with pyarrow to read them.
- We prefer json for configurations and project descriptions
- We prefer int64 timestamps in nanoseconds
- We use the convention T_a_b for the transform from frame b to frame a, i.e. p_a = T_a_b * p_b.
- We can use X_a_b with X = T for 4x4 SE(3), q for quaternion, t for translation, R for rotation matrix, etc. The same convention applies for the subscripts.
- We use the "Assisted-by: AGENT_NAME:MODEL_VERSION [TOOL1] [TOOL2]" tag in the commit messages to indicate that the commit was assisted by an AI agent, and which tools were used.
- We prefer commits with small diffs and the message should be descriptive, e.g., "Add preliminary camera motion functionality"

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
    output_path: Path # path to the folder where the dataset will be generated, if in a project, relative to the project output path
    camhand_rig: CameraHandRig
    cam_motions: List[CameraMotion]
    hand_motions: List[HandMotion]

class Project:
    name: str # human readable name
    output_path: Path # path to the folder where the dataset will be generated
    sequences: List[Sequence]
```

## CLI

Four commands:

| Command | What it does |
|---|---|
| `generate-motion` | **Step 1.** Invents a random timeline of poses and writes it as `animationNN.json`. |
| `generate` | **Step 2.** Renders that timeline to PNG frames plus `joints_2d.csv`. |
| `preview` | Builds the animation into a `.blend` you can open in Blender. Renders nothing. |
| `validate` | Checks that a JSON file matches its schema. |

The models live in `data/models/model1/` and `data/models/model2/`. Each is a
full body with an armature and a set of **pose assets** — hand poses authored in
Blender and marked as assets.

Making a dataset is two steps: first decide *when* each pose happens, then render
it. The two are separate because a timeline takes a second to make, while
rendering it takes minutes.

### Step 1 — make a motion description

```bash
# 4 random poses in 1 second
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output data/sequences/model1 --events 4 --duration 1

# 10 random poses over 20 seconds, as 5 separate files
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output data/sequences/model1 --events 10 --duration 20 --count 5
```

You choose how many poses and how long. Only *which* poses and *when* they land
are random. This writes `animation01.json`, `animation02.json`, … Each file holds
only the timeline — no bone data, because the poses stay in the `.blend`.

```json
{
  "duration_ns": 1000000000,
  "event_count": 4,
  "events": [
    {"t_ns": 0,          "asset": "Right_three", "group": "right"},
    {"t_ns": 30000000,   "asset": "Left_three", "group": "left"},
    {"t_ns": 120000000,  "asset": "Left_pointing", "group": "left"},
    {"t_ns": 1000000000, "asset": "Look_At_Hand", "group": "body"}
  ]
}
```

Each event says which pose is reached at `t_ns` nanoseconds, and `group` says
which part of the body it moves. `--events 4` means **4 poses in the clip**: each
one is drawn from the left-hand, right-hand and whole-body poses together, so the
draw decides which part of the body moves next.

The body moves continuously from one pose to the next, so no two frames look the
same. Three rules keep it that way: the first pose is at `t=0`, the last is at
`duration_ns`, and the same pose is never used twice in a row.

**Each run gives different motion.** A fresh seed is drawn every time, so running
the same command again builds up a dataset instead of rewriting the same file.
The seed is printed and stored in the JSON:

```
[phanesim] Seed 1727056693 -- pass --seed 1727056693 to reproduce this run.
```

#### Poses combine, they do not replace each other

Every pose asset is sorted into a **group** — `left`, `right`, `head` or `body` —
by the bones it actually keys. A left-hand pose only touches left-hand bones, so
applying it leaves the right hand exactly where it was. That is why the timeline
above still has both hands posed at `0.12s` even though only the left one moved.

It also means a small library goes a long way: `model1`'s 9 left and 13 right
poses cover 9 x 13 = **117** configurations, not 22. The command prints the count
when it runs.

Whole-body poses (`Look_At_Hand`, `Pose_photo`) key both hands and the head at
once, so they are simply another thing the draw can turn up, replacing whatever
was held.

#### One timeline per hand — `--hand`

`--hand N` gives each hand a timeline of its own, so both are always moving and
neither waits for the other:

```bash
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output data/sequences/model1 --hand 4 --duration 1
```

```
8 poses over 1 s  (seed=42)
  0.00s  left  pose  Left_Tel          0.00s  right pose  Right_V
  0.03s  left  pose  Left_grab         0.11s  right pose  Right_stop
  0.80s  left  pose  Left_Two          0.79s  right pose  hand_wave
  1.00s  left  pose  Left_default      1.00s  right pose  Right_rock2
```

`--hand 4` means 4 poses for the left hand **and** 4 for the right — 8 events in
the file, but only 4 for each hand to get through in that second. The two columns
above are the two hands, shown side by side; the command prints one list sorted
by time.

`--events` and `--hand` answer different questions, so pass one or the other, not
both. Whole-body poses do not appear under `--hand`: they key both hands, so
there is no single hand's timeline they belong on.

#### Head movement — `--head`

```bash
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output data/sequences/model1 --hand 4 --duration 1 --head
```

`--head` adds the head on a timeline of its own, so it turns while the hands are
changing pose — the two are independent and can change at the same moment. The
camera is anchored to the head bone, so this moves the camera and changes the
background too. Without the flag the head stays still.

It works with either `--events` or `--hand`, and adds that many head poses on top.

### Step 2 — render

```bash
# bare hands, camera fixed to the head
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder

# add the keypoint overlay to check the ground truth visually
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder --debug_kps

# wear a watch and a ring
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder --accessories watch1,ring1

# pan the camera 30 degrees to the right over the clip
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder --camera right,30

# everything at once: all four accessories, a leftward pan, sensor on its side
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder \
    --accessories all --camera left,25 --rotate 90
```

This writes to `output_folder/cam_<name>/`:

- `frame_000000.png`, `frame_000001.png`, … the rendered images
- `joints_2d.csv` — the 21 hand landmarks per hand, in pixels, for every frame
- `frame_000000_debug.png`, … only with `--debug_kps`: the same images with the
  skeleton drawn on top
- `joints_3d.csv` — only with `--debug_kps`: the same 21 landmarks per hand in
  world space, as position `x, y, z` plus rotation `qx, qy, qz, qw`

`--frames N` renders N frames spread evenly across the whole motion, so
`--frames 2` gives the first and last frame, and any number still covers the
entire animation. Use a small number to check something quickly and a large one
for the real dataset. Rendering takes roughly 5 seconds per frame.

#### Accessories

`--accessories` picks what the model wears. **Nothing is worn by default**, so a
frame shows exactly what you asked for. `model1` has four to choose from:

| Name | What it is | Where |
|---|---|---|
| `ring1` | ring | right middle finger |
| `ring2` | wedding ring | left ring finger |
| `watch1` | wristwatch | left wrist |
| `band1` | braided wristband | right wrist |

```bash
--accessories all              # wear everything
--accessories watch1,ring1     # just these two
--accessories none             # bare hands (the default)
```

Render the same motion twice with different accessories to get two variations of
the same frames. `model2` has none of them, so it prints a note and renders bare.

#### Camera movement

Two things move the camera, and they add up:

- **The head**, if the motion was made with `--head`. This is real head movement,
  so the background changes as a person's would.
- **`--camera DIRECTION,DEGREES`**, a steady turn on top of that. The clip
  starts at the rest view and ends `DEGREES` away from it, so `right,30` pans
  the camera 30 degrees to the right across the frames and the background slides
  left.

```bash
--camera right,30    # pan right, ending 30 degrees off
--camera left,20     # pan left
--camera up,15       # tilt up
--camera down,25     # tilt down
```

Directions are `left`, `right`, `up`, `down`. Leave the option out and the
camera only moves when the head does. It works on `preview` too, so you can
scrub the turn in Blender before rendering anything.

#### Camera rotation — `--rotate`

`--rotate 90` turns the camera about its own optical axis, so the sensor sits on
its side:

```bash
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder --rotate 90
```

The file is still 640x480 and the camera still looks in the same direction — only
the orientation of the scene inside the frame changes, so a 480x640 portrait view
fills a landscape image. This is what a headset with a rotated camera sees, and it
is a cheap way to double a dataset: render the same motion twice, once at `0` and
once at `90`.

`0`, `90`, `180` and `270` are accepted. The keypoints in `joints_2d.csv` rotate
with the image, so the annotations stay correct.

### Preview — look at it in Blender

```bash
uv run phanesim preview data/sequences/model1/sequence.json \
    --frames 15 --output preview.blend
```

**`preview` does not render any images.** It builds the animation and camera into
a `.blend` file and stops. Open that file in Blender to scrub the timeline, check
where the camera is pointing, and adjust the compositor nodes by hand. It takes
seconds instead of minutes, so use it to check a setup before committing to a
full render with `generate`. It accepts `--accessories` and `--camera` as well.

### The sequence file — you edit this one by hand

`sequence.json` is the only file in the workflow that **nothing writes for you**.
`generate-motion` creates `animationNN.json` files but never touches
`sequence.json`. If you want a render to use a different animation, you open the
file and change the name yourself.

```json
{
  "name": "model1",
  "output_path": "model1",
  "body_rig": {
    "cameras":     [ "... resolution, lens, noise, distortion ..." ],
    "body":        { "model": "../../models/model1/model1.blend" },
    "head_camera": { "rest_position": [0.0, -0.21, 1.715] }
  },
  "hand_motions": ["animation01.json"],
  "frames": 21,
  "hdri": "../../hdri/brown_photostudio_02_2k.exr"
}
```

#### Switching to another animation

Say you generated five takes with `--count 5`:

```
data/sequences/model1/
    sequence.json
    animation01.json     <- the one being rendered
    animation02.json
    animation03.json
    animation04.json
    animation05.json
```

`generate` renders whatever `hand_motions` lists. To render `animation03.json`
instead, edit that line:

```json
  "hand_motions": ["animation01.json"],      // before
  "hand_motions": ["animation03.json"],      // after
```

To render several takes in one command, list them all. Each gets its own folder
named after the animation:

```json
  "hand_motions": ["animation01.json", "animation02.json", "animation03.json"]
```

```
output_folder/
    animation01/cam_head0/frame_000000.png ...
    animation02/cam_head0/frame_000000.png ...
    animation03/cam_head0/frame_000000.png ...
```

With a single entry there is no extra folder — the frames go straight into
`output_folder/cam_head0/`.

⚠ The file names are relative to the `sequence.json`, so the animation files must
sit in the same folder.

#### The other fields

- **`body.model`** — which model to use, relative to this file.
- **`head_camera.rest_position`** — where the camera sits on the head. Measured
  per model, and different for each one because the models are different heights.
  `model1` uses `[0.0, -0.21, 1.715]`, `model2` uses `[0.0, -0.19, 1.564]`.
- **`head_camera.rest_forward`** — where it looks. The camera is bolted to the
  head like a real headset and never turns to follow the hands, so they move in
  and out of view on their own. `[0.0, -1.0, -0.268]` points forward and 15
  degrees down, at the space where the hands are.
- **`frames`** — default frame count. `--frames` on the command line wins.
- **`cameras`** — resolution, focal length, and the artifact settings (noise,
  distortion, vignette).

After editing, check the file is still valid:

```bash
uv run phanesim validate body_sequence data/sequences/model1/sequence.json
```

### Validate

```bash
uv run phanesim validate body_sequence data/sequences/model1/sequence.json
uv run phanesim validate pose_motion   data/sequences/model1/animation01.json
```

Prints `OK` or the reason the file is wrong. Useful after editing a
`sequence.json` by hand.

## Third-party assets

These credits must be reproduced wherever the models or rendered datasets are
shared. They are also recorded in `REUSE.toml`.

### Accessory meshes embedded in `data/models/model1/model1.blend`

All three are licensed **CC-BY-4.0** (http://creativecommons.org/licenses/by/4.0/),
which requires the author to be credited. Commercial use is allowed.

> This work is based on "543 - Ring"
> (https://sketchfab.com/3d-models/543-ring-4c55eacf1f264799b5a61126678f8360)
> by Lizardsking (https://sketchfab.com/lizardsking)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

> This work is based on "Seiko Watch"
> (https://sketchfab.com/3d-models/seiko-watch-0796e23ab5c0448c9bdf3fe5c3b3e362)
> by carloshisserich (https://sketchfab.com/carloshisserich)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

> This work is based on "Braided Loop Wristband"
> (https://sketchfab.com/3d-models/braided-loop-wristband-d889ebab38da43fda673eb273945afdc)
> by Mikaeel Irani (https://sketchfab.com/mirani55)
> licensed under CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)

### Body meshes

`model1` and `model2` are derived from MB-Lab base meshes
(Manuel Bastioni, MB-Lab contributors), rigged and posed for this project, and
carry **AGPL-3.0-only**. `model1.blend` is therefore a combined work under
`AGPL-3.0-only AND CC-BY-4.0`.

### Environment map

`data/hdri/brown_photostudio_02_2k.exr` by Sergej Majboroda, **CC0-1.0**
(no attribution required, credited here as a courtesy).
