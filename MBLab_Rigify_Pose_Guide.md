# MB-Lab → Rigify → Pose Library

Pipeline notes for phanesim. ⚠ = things that cost time.

Versions: MB-Lab is Blender 4.0 only. Rigify is native in 5.x.
Generate the body in 4.0, rig and pose in 5.x.

---

## 1. Generate body (4.0)
MB-Lab panel → pick base → set sliders → Finalize (keep MB-Lab skeleton + weights) → save.

⚠ Different bases (male/female) have different topology: we cannot reuse a rig or copy
weights between them. Rig reuse only works for same base and same proportions.

## 2. Separate face parts — BEFORE rigging
Separate eyeballs, corneas, teeth, tongue, eyelashes from the body.

⚠ The body is a closed mesh with interior geometry (eyes/teeth inside the head).
Bone-heat automatic weights cannot solve on it - it fails and leaves empty `DEF-` groups.
⚠ Separating destroys weights: parts keep group *names* but zero weight data → they float
outside the head, frozen at rest.
⚠ Eyelashes can explode into 100+ loose fragments (here: 136). Join them back into one object.


## 3. Fit the metarig (in Blender 5.x)
`Add → Armature → Human (Meta-Rig)` → scale to height → `Ctrl+A` Apply Scale →
X-ray on → Edit Mode → X-Axis Mirror on → snap spine, one arm (+fingers), one leg.

⚠ **Rig the Hand** - rig the hand/finger following the Blender file for finger bones:
Select the mesh, edit, then loop around a finger knuckle, `ALT click`, `shift S` cursor to selected → then back to Object mode, select metarig, edit, select bone，`shift S` selection to cursor.
⚠ **Fit to the MB-Lab skeleton** - it's generally fitted. Biggest time-saver.
⚠ Elbows need a slight backward bend, knees forward, or IK can't resolve. Heel bone at the back of the heel, foot flat.
⚠ The ~94 face bones have no MB-Lab equivalent — fit manually or delete.

## 4. Generate rig + weights
Metarig → Object Data Properties → Rigify → Generate Rig.
Body → shift-select `rig` → `Ctrl+P` → With Automatic Weights.

⚠ Verify weights aren't empty: generate can "succeed" with every `DEF-` group empty
(= no deformation).
⚠ Delete stacked dead Armature modifiers (this file had 7, six pointing at nothing).
⚠ Disable the MBLab automatic generated Subsurf / Displace / Corrective Smooth before auto weights.

## 5. Create pose assets
Pose → select the bones to store → `Pose → Animation → Create Pose Asset` →
Asset Browser (Current File) → rename → drag onto catalog → save.

⚠ Only selected bones are stored. For layerable body poses, exclude fingers so a body pose doesn't overwrite the hand pose on top.

## 6. Asset library / catalogs
Preferences → File Paths → Asset Libraries.

## 7. Scripting poses (5.x)
⚠ Setting `animation_data.action` is not enough — also bind
`animation_data.action_slot = action.slots[0]`, or every pose reads as rest pose.
⚠ `action.fcurves` is gone; use `action.layers[].strips[].channelbag(slot).fcurves`.
⚠ Poses are keyed at `frame_range[0]` (frame 1), **not frame 0**.
⚠ Tell poses from animations by **keyframe count, not name** (`ThumbUp1` was a 446-frame clip).
