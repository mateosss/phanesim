<!--
Copyright 2026, Yutong Wan.
SPDX-License-Identifier: BSD-3-Clause
-->

# MB-Lab → Rigify → Pose Library

Pipeline notes for **phanesim**. ⚠ = things that cost time.

**Versions:** MB-Lab works in **Blender 4.0 or lower**. It does not work in 5.x.
Rigify comes with Blender and works in 5.x.
So: make the body in **4.0**, do the rig and poses in **5.x**.

---

## 1. Generate the MB-Lab mesh (Blender 4.0 or lower)

### 1.1 Install the addon
1. Download MB-Lab from GitHub: `github.com/animate1978/MB-Lab` (get the `.zip`).
2. In Blender: `Edit → Preferences → Add-ons → Install…` → pick the zip.
3. Tick the checkbox to enable it.

⚠ Use Blender **4.0 or lower**. In 5.x the addon will not load.

### 1.2 Create the base mesh
1. Press **`N`** in the 3D viewport to open the sidebar. Click the **MB-Lab** tab.
2. Choose a **project** (human, anime, etc.).
3. Choose a **base model** (e.g. Caucasian male, Caucasian female).
4. Set the options: IK or muscles, EEVEE or Cycles, portrait studio lights.
5. Click **Create Character**. The model appears at (0,0,0).

Scale is 1 Blender Unit = 1 meter. Loading the morphs takes a moment.

### 1.3 Customize the body
Use the MB-Lab panels to shape the character:
- **Creation Tools** — body measures, phenotypes, presets, random generator.
- **Skin Editor** — skin, iris, texture maps.

⚠ Only use MB-Lab's own tools here. Do **not** edit the mesh in Edit Mode, add shape keys,
sculpt, or rename the object. Doing so breaks the character.

### 1.4 Finalize (+ save the .json backup)
1. Go to the **Finalize** panel at the bottom.
2. Keep **"Save images and backup character"** ticked (it is on by default).
   This writes a **`.json`** file with all the character parameters, plus the textures.
3. Leave **"Remove modifiers"** unticked unless you want them gone.
4. Press **Finalize**.

⚠ **Finalize cannot be undone.** Always export the `.json` first.
The `.json` lets you restore the character later with the **Import Character** button.
⚠ After finalizing, the character is a normal Blender model. MB-Lab tools no longer apply to it.
⚠ Save the `.blend` as well.

### 1.5 Notes for later
- Different base models (male vs female) have different topology and vertex counts
  (~11.8k vs ~18.2k). You **cannot** reuse a rig or copy weights between them.
- Note the MB-Lab bone names (`upperarm_L`, `lowerarm_L`, `hand_L`, `index01_L`, …).
  They are a useful reference when fitting the metarig.

---

## 2. Rig with Rigify (Blender 5.x)

### 2.0 First: separate the face parts
Separate eyeballs, corneas, teeth, tongue and eyelashes from the body mesh.
Do this **before** automatic weights.

⚠ The body is a closed mesh with **interior geometry** (eyes and teeth sit inside the head).
Automatic weights (bone heat) **cannot solve** on it. It fails and leaves empty `DEF-` groups.
⚠ Separating **destroys the weights**. The parts keep the group *names* but hold no weight data.
They then float outside the head, frozen, while the body moves.
⚠ Eyelashes can break into many loose pieces (here: **136**). Join them back into one object.

### 2.1 Enable Rigify
`Edit → Preferences → Add-ons` → search **Rigify** → tick it. It ships with Blender.

### 2.2 Add and fit the metarig
1. `Add → Armature → Human (Meta-Rig)`.
2. Scale it to the character's height.
3. Press **`Ctrl+A` → Apply Scale**. Rigify needs this.
4. Turn on **In Front** (Object Data Properties) and **X-ray** (`Alt+Z`).
5. Go to **Edit Mode**. Turn on **X-Axis Mirror**, so one side copies to the other.
6. Move the bones onto the joints. Order: **spine → one arm + fingers → one leg**.

⚠ **Fit to the MB-Lab skeleton, not the mesh surface.** That skeleton is already fitted,
fingers included. This saves the most time, especially for hands.
⚠ Bend the elbows slightly **backward** and the knees slightly **forward**.
A straight chain gives broken IK.
⚠ Put the **heel bone** at the back of the heel, flat on the ground.
⚠ The ~94 face bones have no MB-Lab match. Fit them by hand or delete them.

### 2.3 Generate the rig
1. Select the metarig.
2. Go to **Object Data Properties → Rigify**.
3. Press **Generate Rig**. This creates the control rig, named `rig`.
4. Hide or delete the metarig. From now on you pose `rig`.

### 2.4 Automatic weights
1. Select the **body mesh**.
2. **Shift-select** the generated `rig` (rig must be active/last).
3. Press **`Ctrl+P` → Armature Deform → With Automatic Weights**.

⚠ **Check that the weights are not empty.** Generate can look fine while every `DEF-` group
is empty. Empty groups = the mesh does not move at all.
⚠ Turn off **Subsurf / Displace / Corrective Smooth** before this step. Turn them back on after.
⚠ Delete leftover dead Armature modifiers. Re-binding stacks them up
(this file had 7, and 6 pointed at nothing). Keep only the one pointing at `rig`.

### 2.5 Re-attach the separated parts
These parts are rigid. **Bone-parent** them instead of weighting them.

Select the part → shift-select the rig → Pose Mode → pick the bone →
`Ctrl+P` → **Bone**.

| Part | Bone |
|---|---|
| Eyeball_L / Cornea_L | `eye.L` |
| Eyeball_R / Cornea_R | `eye.R` |
| Teeth, Eyelashes | `spine_fk.006` (head) |
| Tongue | `jaw_master` |

⚠ Do this in **Rest Position** (Object Data Properties → Pose Position → Rest).
Otherwise the current pose is baked into the offset and the parts sit wrong.
⚠ Rigify has no `DEF-eye` or `DEF-teeth` bones, so bone-parenting is correct here.
⚠ Teeth as one object will not open with the jaw. Split upper/lower if you need talking.
⚠ Test it: rotate the head and the jaw, and check the parts follow.

Rename the meshes: `Body`, `Eyeball_L`, `Cornea_L`, `Eyeball_R`, `Cornea_R`, `Teeth`,
`Tongue`, `Eyelashes`. Identify them by **material** (`MBlab_cornea`, `MBlab_human_teeth`,
`MBlab_tongue`, `MBlab_eyelash`) and by x-position for left/right. Rename the mesh data too.

---

## 3. Build the pose library

### 3.1 Make one pose
1. Select `rig` → **Pose Mode**.
2. Pose the character.
3. **Select the bones you want the pose to store.**
4. `Pose → Animation → **Create Pose Asset**`.
   (The same button is in the Dope Sheet → **Action Editor** header.)

⚠ Only the **selected** bones are saved into the pose.
⚠ For body poses you want to combine with hand poses, **leave the finger bones out**.
Then a body pose will not overwrite the hand pose layered on top.
⚠ For a self-contained pose, press **`A`** to select everything.

Useful body selection: `hand_ik.L/R`, `upper_arm_ik_target.L/R`, `shoulder.L/R`,
`torso`, `hips`, `chest`, `spine_fk` … `spine_fk.006`.

### 3.2 Name it and file it
1. Open an **Asset Browser**. Set the source to **Current File**.
2. Rename the new pose.
3. Drag it onto a catalog (e.g. `Poses`, or `Poses/Hands` for hand poses).
4. Save the `.blend`.

### 3.3 Use a pose
- **Double-click** a pose in the Asset Browser to apply it.
- If some bones are selected, it only applies to those bones.
- **Apply Pose Flipped** mirrors left to right, so one hand pose covers both hands.

### 3.4 Edit an existing pose (keep the same name)
A pose asset is just an Action with keys at frame 1.
1. Dope Sheet → **Action Editor**.
2. Pick the pose (e.g. `hand_wave`) from the action dropdown.
3. Go to **frame 1**.
4. Fix the bones.
5. Select them and press **`I`** to insert keys, overwriting frame 1.

⚠ Do **not** press Create Pose Asset again. That makes a second asset (`hand_wave.001`).
⚠ The thumbnail does not refresh by itself. Re-render the preview in the Asset Browser sidebar.

### 3.5 Where to keep poses
Optional external library: `Preferences → File Paths → Asset Libraries` → add a folder.

⚠ **Visible in the Asset Browser is not the same as in this file.** The browser shows library
assets without importing them. `bpy.data.actions` only holds local ones. This is why scripts
say *"no action named X"*. A script must append them first.
⚠ **Catalog UUID mismatch**: the same catalog name can have a different UUID in the file and
in the library. Then assets show as **Unassigned**. Use the current file's UUIDs.
⚠ `catalog_simple_name` is read-only. Only `catalog_id` decides placement.

---

## 4. Use the poses in a generation

### 4.0 What the pipeline expects
- The model is `data/models/modelN/modelN.blend`, with the armature named **`rig`**.
- Every pose is an **asset-marked Action** inside that same file.
- The `.blend` is **saved**.

⚠ Only asset-marked actions count. A pose you made but did not mark is invisible to the code,
even though it works fine when you double-click it in Blender.
⚠ Other actions in the file (e.g. `Shader NodetreeAction.*`) are ignored, which is why the
mark matters.

### 4.1 Check the poses are visible to the code
```bash
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output output_folder --events 4 --duration 1
```
It prints how many assets it found. If the number is lower than you expect, the missing poses
are either unmarked or the `.blend` was not saved.

### 4.2 Step 1 — make a motion description
```bash
uv run phanesim generate-motion --model data/models/model1/model1.blend \
    --output data/sequences/model1 --events 8 --duration 4
```
This writes `animation01.json`: a list of *which pose* at *what time*, and nothing else.

```json
{"t_ns": 1040000000, "asset": "Right_one.002", "kind": "pose"}
```

The pose data stays in the `.blend`. The description only stores the **name**, so renaming a
pose in Blender breaks any description that refers to it.

Single-frame poses get `"kind": "pose"`. Multi-frame ones (e.g. `Wave_Animation`) get
`"kind": "action"` and are played back over their own length.

### 4.3 Step 2 — render
```bash
uv run phanesim generate body_sequence data/sequences/model1/sequence.json \
    --frames 21 --output output_folder --debug_kps
```
`sequence.json` names the model and lists the animation files in `hand_motions`.

### 4.4 What happens to a pose during a render
1. The description is read; each pose name is looked up in the model's actions.
2. Each pose is applied once and its bone values are copied out.
3. Those values are keyed onto the armature at the times in the description.
4. Keys are set to **linear**, so the body moves at constant speed between poses.
5. Hand landmarks are read from the `ORG-` bones and written to `joints_2d.csv`.

⚠ If a name in the description is not in the file, the render **stops** with a list of the
names it did find. It does not silently skip.

### 4.5 Things that bite

⚠ **Poses do not transfer between models.** A pose authored on `model1` can be appended into
`model2`, but the bodies have different proportions, so it usually needs re-posing there.
After editing, the two versions are different poses that happen to share a name.

⚠ **Adding a pose only affects the model you saved it in.** There is no shared library.

⚠ **The catalog file must be kept.** `blender_assets.cats.txt` sits next to the `.blend` and
holds the folder definitions. The `.blend` only stores a UUID per asset. Delete the file and
every pose shows as *Unassigned*. Commit it with the model.

⚠ **`.001` suffixes stick.** A pose duplicated in Blender becomes `Right_one.002` and that
name goes into the description and the dataset. Rename it before generating.

⚠ **A pose is never used twice in a row.** The generator skips the pose that is already
showing, because an event that changes nothing freezes every frame until the next one. With
few poses in the file this makes the timeline noticeably repetitive.

⚠ **The opening pose is random** unless you pass `--rest-asset NAME`.
