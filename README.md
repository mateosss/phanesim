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

The commands:

| Command | What it does |
|---|---|
| `generate-motion` | **Step 1.** Invents a random timeline of poses and writes it as `animationNN.json`. |
| `generate` | **Step 2.** Renders that timeline to PNG frames plus `joints_2d.csv`. |
| `preview` | Builds the animation into a `.blend` you can open in Blender. Renders nothing. |
| `plan-clips` | Plans a whole dataset: one directory per clip. Renders nothing. |
| `render-clips` | Renders a planned dataset. Safe to interrupt and re-run. |
| `annotate` | Derives `hand_rect.csv` from `joints_2d.csv` files already on disk. Renders nothing. |
| `visibility` | Reports how many rendered frames actually show a hand. Renders nothing. |
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

`--head` adds head movement, so the head turns while the hands are changing pose.
The camera is anchored to the head bone, so this moves the camera and changes the
background too. Without the flag the head stays still.

It works with either `--events` or `--hand`.

#### The head looks at the hands

The head is not drawn independently of them. The camera is bolted to the head, so
the head *is* where the camera points: a head that looks away from where the arms
went produces a frame with no hand in it, however good the hand pose was. Drawn
independently that happened about half the time.

So the hands are drawn first and the head second, and each head pose is weighed by
how many hands it would leave in frame across the stretch it is held. Two things
follow from that:

- **The head is keyed just after every hand move**, rather than on a schedule of
  its own, so a hand never changes and then sits unseen until the head catches
  up. It still moves at moments the hands do not.
- **Which head pose comes up depends on where the arms are.** Looking ahead or a
  little to one side carries most of the timeline, because that is where a hand
  in front of the body is visible at all. The extremes — `Head_farleft`,
  `Head_leftup` — stay reachable and come up when an arm is somewhere they can
  see, which is what puts a hand at the edge of the frame for a detector to learn.

The mapping from head pose to the arm positions it can see lives in `HEAD_VIEW` in
`posemotion.py`. It is a plain table, tied to the current library's pose names, and
it is the thing to edit if a pose is renamed or the framing changes.

### Step 2 — render

```bash
# bare hands, camera fixed to the head
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder

# write the extra ground truth and draw it, to check it visually
uv run phanesim generate data/sequences/model1/sequence.json \
    --frames 81 --output output_folder --debug_kps

# override what the sequence file says about accessories, just for this run
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
- `hand_rect.csv` — only with `--debug_kps`: per hand, whether it is in the
  picture and the bounding box it occupies. See
  [Hand bounding boxes](#hand-bounding-boxes--hand_rectcsv).
- `joints_3d.csv` — only with `--debug_kps`: the same 21 landmarks per hand in
  world space, as position `x, y, z` plus rotation `qx, qy, qz, qw`
- `frame_000000_debug.png`, … only with `--debug_kps`: the same images with the
  skeleton and the bounding boxes drawn on top

`--frames N` renders N frames spread evenly across the whole motion, so
`--frames 2` gives the first and last frame, and any number still covers the
entire animation. Use a small number to check something quickly and a large one
for the real dataset. Render speed depends heavily on the machine — measure
your own before planning a long run.

#### Accessories

`--accessories` overrides what the model wears. `model1` has four:

| Name | What it is | Where |
|---|---|---|
| `ring1` | ring | right middle finger |
| `ring2` | wedding ring | left ring finger |
| `watch1` | wristwatch | left wrist |
| `band1` | braided wristband | right wrist |

```bash
--accessories all              # wear everything
--accessories watch1,ring1     # just these two
--accessories none             # bare hands
```

The option is for trying something out. Normally you set `accessories` in the
sequence file instead, so the render describes itself:

```json
"accessories": ["watch1", "ring1"]
```

The `.blend`'s saved state is never inherited — `model1.blend` is saved with all
four showing, and inheriting that silently put accessories into renders nobody
asked for. `model2` carries none of the four, so its clips are always bare.

The background works the same way, except it has no command-line option at all:
set `hdri_spin_step_deg` in the sequence file to turn it a little on every frame.

#### Simple Camera movement

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

#### Hand bounding boxes — `hand_rect.csv`

Detection and keypoint estimation are two stages: first find whether there is a
hand and where its box is, then run keypoints on the crop. `hand_rect.csv` is
the ground truth for the first stage — one row per frame:

```
timestamp, left_present, left_x, left_y, left_w, left_h,
           right_present, right_x, right_y, right_w, right_h
```

`x, y` is the top-left corner and `w, h` the size, in pixels. An absent hand is
`present=0` and four `nan`s, so a script that ignores the flag fails loudly
instead of training on a box in the corner.

The box is an upright rectangle even though the lens distortion curves the
hand's real outline, because that is what a detector predicts. A hand counts as
present once 5 of its 21 landmarks are inside the image, so one sliced by the
frame edge is kept and its box clipped rather than dropped. The box is grown
past the landmark hull by 12% of the hull's longer side on all four edges, since
the landmarks are joint centres and the hull runs inside the hand. All three are
`debug.py` constants; the margin is also `phanesim annotate --margin`.

Nothing is re-rendered to produce it — it comes from `joints_2d.csv`, so
`phanesim annotate` adds it to a dataset rendered before it existed:

```bash
uv run phanesim annotate dataset                   # every clip under it
uv run phanesim annotate output_folder --overlay   # and draw it
```

`--debug_kps` draws the boxes alongside the skeleton, cyan for the left hand and
yellow for the right, read back out of `hand_rect.csv` so the picture can only
show what the file says.

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
  "body_rig": {
    "cameras":     [ "... resolution, lens, noise, distortion ..." ],
    "body":        { "model": "../../models/model1/model1.blend" },
    "head_camera": { "rest_position": [0.0, -0.181, 1.723] }
  },
  "hand_motions": ["animation01.json"],
  "frames": 21,
  "hdri": "../../hdri/brown_photostudio_02_2k.exr",
  "accessories": [],
  "hdri_spin_deg": 0.0,
  "hdri_spin_step_deg": 0.0
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

Every field below is set by editing the file. `generate` reads all of them, so a
render needs no options beyond `--output`.

| Field | What it does |
|---|---|
| `hand_motions` | Which animation to render. See above. |
| `frames` | How many frames. `--frames` overrides it. |
| `hdri` | The background and lighting, relative to this file. |
| `accessories` | What the body wears: any of `ring1`, `ring2`, `watch1`, `band1`. Empty is bare hands. `--accessories` overrides it. |
| `hdri_spin_deg` | Angle the background is turned on the first frame. |
| `hdri_spin_step_deg` | Degrees added on each frame after, so every frame gets a different slice of the panorama and a different light direction. `0` holds it still. |
| `body.model` | Which model, relative to this file. |
| `cameras` | Resolution, focal length, and the artifact settings (noise, distortion, vignette). |
| `head_camera.rest_position` | Where the camera sits on the head. Measured per model: `model1` `[0.0, -0.181, 1.723]`, `model2` `[0.0, -0.161, 1.572]`. It has to clear the nose — behind that line the camera renders the inside of the face — but as close to it as possible, because the hands work close to the body. |
| `head_camera.rest_forward` | Where it looks. Bolted to the head like a real headset, never turning to follow the hands. `[0.0, -1.0, -0.601]` is forward and 31 degrees down, which is where the hands are: they work low and close, so a shallower angle leaves them along the bottom edge. `HEAD_VIEW` in `posemotion.py` is calibrated against this angle. |

`plan-clips` writes all of these for you, one clip at a time — see
[Making a dataset](#making-a-dataset--plan-clips-and-render-clips).

After editing, check the file is still valid:

```bash
uv run phanesim validate body_sequence data/sequences/model1/sequence.json
```

### Making a dataset — `plan-clips` and `render-clips`

The two steps above make **one** animation. For a training set you want hundreds,
each with a different body, background and camera. That is what these two do.

A **clip** is one continuous stretch of frames in its own directory, with its own
`sequence.json` and `animation.json`. Everything varies *between* clips —
background, body, camera — and only the motion varies *within* one.

```bash
# 1. Plan. Writes one directory per clip. No rendering, takes seconds.
uv run phanesim plan-clips --template data/sequences/model1/sequence.json \
    --output dataset_test --clips 10 --frames 15 --hand 6

# or on cluster

uv run phanesim plan-clips \
    --template data/sequences/model1/sequence.json \
    --output ~/storage/user/phanesim/dataset \
    --clips 50 \
    --frames 15 \
    --hand 6 \
    --append

# ...then the other body, numbering on from where the first left off.
uv run phanesim plan-clips --template data/sequences/model2/sequence.json \
    --output dataset --clips 10 --append

# 2. Render. Interrupt it whenever; run it again to carry on.
uv run phanesim render-clips dataset

# 3. Check if hands are visible in the dataset
uv run phanesim visibility dataset
```

Planning each body separately is how you control the mix. Passing two
`--template` options to one command works too, but it draws one at random per
clip, so 10 clips can come out 2:8 rather than 5:5.

`plan-clips` refuses to write into a dataset that already has clips. Use
`--append` to add to it, or `--overwrite` to replace it — overwriting also
invalidates any frames already rendered, so they get rendered again.

**One event is a whole limb.** The pose library is split by joint — an asset
moves the upper arm, or the forearm, or one finger — so an event draws one of
each: arm, forearm, wrist, and then the hand, set either by one to five single
fingers or by one whole-hand pose, half the time each. Every pose is blended in
by its own amount. A single joint alone barely changes the picture; the arm
accounts for about 78 px of hand movement against 10 px for all five fingers
together.

The joints are not drawn wholly independently of each other either: folding the
forearm vertical (`Forearm_*_up_*`) is the raise-your-hand pose only from a
lowered upper arm, and on a level or raised one it puts the hand above anything
the camera sees, so that pairing is gated out. The two hands are also drawn to
the same height about half the time, since one head pose can only hold both hands
at once when they are in the same band.

**The head moves by default**, keyed just after each hand move and drawn against
where the arms are — see [The head looks at the hands](#the-head-looks-at-the-hands).
The camera is anchored to the head bone, so this also moves the camera and changes
the background. Pass `--no-head` to hold it still.

**About 1 frame in 25 has no hand in it**, two thirds have both, and the rest
have one — about 1.6 hands per rendered frame, with a hand counted as present when
5 of its 21 landmarks are in frame. Counting only hands no frame edge has cut, it
is 41% both and 17% none. Run `phanesim visibility <dataset>` on a render to
measure it; it reports all three thresholds, and they disagree enough that one
number on its own is misleading.

That is the result of aiming the camera at where the hands actually are rather
than of the pose sampling alone; see
[The head looks at the hands](#the-head-looks-at-the-hands) and
`head_camera.rest_forward`. Two knobs move the balance: the zero entry of
`HEAD_VIEW_WEIGHTS` raises or lowers the share of empty frames, and
`ARM_COUPLE_SHARE` trades two-hand frames against one-hand ones. Both are in
`posemotion.py`.

```
dataset/
  clip_00000/
    sequence.json     # this clip's body, background and camera
    animation.json    # this clip's pose timeline, with its seed
    cam_head0/
      frame_000000.png ... frame_000049.png
      joints_2d.csv
      joints_3d.csv
      hand_rect.csv
    _done.json        # written last: "these frames match these settings"
  clip_00001/
  ...
```

#### Resuming

`render-clips` writes `_done.json` only after a clip's frames are on disk, and
skips clips that already have one. So if the run is killed — by another job on a
shared machine, or by the process dying on its own — **just run the same command
again**. It re-does at most the clip that was in flight.

`_done.json` stores a hash of `sequence.json` and `animation.json`, so editing a
clip's settings marks it un-done and it gets re-rendered. That stops a half-changed
dataset from looking finished. Use `--overwrite` to force everything.

#### What varies between clips

| | How |
|---|---|
| body | picked from the `--template` files |
| background | picked from `data/hdri/` |
| pose timeline | fresh draw, its seed stored in `animation.json` |
| accessories | 50% bare, 30% one, 15% two, 5% three or four |
| background angle | starts anywhere, then turns 25–45° per frame, enough to pass the whole panorama (`--no-spin` holds it) |
| clip length | `--duration` is derived as `frames / 30`, i.e. 30 fps |
| sensor noise | `noise_std` 0.05–0.40 |
| vignette | `vignette_factor` 0.30–0.70 |
| lens distortion | `distortion` 0.25–0.40, with `lens_scale` following it |
| field of view | `fx` = `fy` 210–270, i.e. about 100°–113° |

Three things are deliberately **not** varied:

- **`cx` / `cy`** — Blender renders with the principal point at the image centre
  and no shift is applied, so moving them would put the ground truth in the wrong
  place while the picture stayed the same.
- **`fy` on its own** — only `fx` sets the Blender lens, so `fy` has to match it.
- **`distortion` beyond 0.40** — the forward map in `distort_pixel` was fitted
  and checked against Blender up to about 0.4. Past that the 2D keypoints would
  drift away from the pixels, silently.

`render-clips` always writes `joints_3d.csv` and `hand_rect.csv`, and never
writes the `_debug.png` overlays: keypoints drawn onto the image would be
learned as features. Every CSV is ground truth and belongs in the dataset; the
drawings are only ever a way of looking at it.

#### Choosing `--frames` and `--hand`

They are separate knobs pulling opposite ways. `--frames` decides how densely
the pose path is sampled; `--hand` decides how many configurations are on it.
Raising `--frames` alone samples the *same* path more finely, so frames get more
alike.

What matters is how many frames one change takes — aim for **1 to 2**. The
default is `--frames 10 --hand 10`, one event per frame. `--duration` is derived
from the frame count (30 fps) and does not affect the images.

#### Splitting for training

Split **by clip, never by frame**. A clip's frames share one background, one body
and one set of camera settings, so putting some in train and some in test lets a
network score well on what it has effectively already seen.

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

### Environment maps

The `.exr` files in `data/hdri/` come from
[Poly Haven](https://polyhaven.com/hdris) and
[ambientCG](https://ambientcg.com/list?type=HDRI), and
[Open HDRI](https://openhdri.org/). They are all **CC0-1.0** — public domain,
no attribution required. Each provides both the lighting and the background of
a render.

Each file is still credited to its author in `REUSE.toml` when the source names
one. Open HDRI assets are credited to Grzegorz Wronkowski; ambientCG assets
without a named individual use `NOASSERTION` and retain their source URLs. A
new map needs an entry there; `reuse lint` fails until it has one.
