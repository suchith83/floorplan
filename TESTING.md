# Checking each component

Every check runs the real pipeline up to that stage, prints numbers, and writes images to
`out/<capture>/check/`. Run them in order: if a stage looks wrong, every later stage will too.

```bash
uv run fp check all <capture_dir>          # every stage below
uv run fp check <stage> <capture_dir>      # one stage
uv run fp run <capture_dir>                # full pipeline -> out/<capture>/plan.svg, plan.json
uv run python eval/arkit_gt.py <capture_dir> out/<capture>   # vs laser GT (ARKitScenes only)
```
`fp check` needs depth + poses: an ARKitScenes raw folder (`lowres_wide/`, `lowres_depth/`,
`lowres_wide_intrinsics/`, `lowres_wide.traj`). Add `--filter` to inspect the frames `fp run` keeps.
For video and photos, `fp run` writes `debug_bev.png` and the report; its depth comes from MapAnything.

---

### 1. `frames`: are the input images usable?
Outputs: `frames_sheet.png` (16 evenly spaced), `frames_worst.png` (4 blurriest + 4 darkest), `frames_quality.csv`.
- **Good:** brightness p50 between 80 and 180; < 10 % blurry; few near-duplicates.
- **Red flags:** dark frames (lens covered, lights off); > 20 % blurry (moving too fast); long runs of duplicates (standing still).
- **Open the images.** If the sheet shows junk (black, floor-only, repeated views), stop here: the data is the problem, not the code.

### 2. `poses`: is the camera track plausible?
Output: `poses_topdown.png` (path from above; green dot = start, red = > 1 m/s).
- **Good:** smooth path that stays inside the room shape; 0 gaps > 0.15 s; rotation p95 < 40°/s.
- **Red flags:** jumps or teleports (tracking lost); rotation > 60°/s (fast panning smears walls, REVIEW #1); "frames with a pose" well below the total.

### 2b. `fusion`: how the point cloud is built, step by step
Outputs: `fusion_1_inputs.png` … `fusion_5_normals.png`, explained in [HOW_IT_WORKS.md](HOW_IT_WORKS.md) §4.
Prints raw → voxel → outlier point counts and the normal classes.
- **Good:**
  - In `fusion_2`, the frame's points sit inside the camera's field-of-view lines.
  - In `fusion_4` (side slice through a wall), the wall is one thin vertical line and the floor is one thin line on the blue z = 0 line.
  - In `fusion_5`, fewer than ~25 % of points are "oblique/noisy".
- **Red flags:** points outside the field of view (wrong pose or K); several parallel streaks in the wall slice (smear, REVIEW #17); the floor sitting above or below the blue line (REVIEW #7).

### 3. `cloud`: does the 3D look like the room?
Output: `cloud.ply`. View it: `uv run fp check view out/<capture>/check/cloud.ply` (mouse rotates, scroll zooms, `n` toggles normals).
- **Good:** walls are thin flat sheets, the floor is one flat surface, and doors and counters are recognisable.
- **Red flags:** doubled or "ghost" walls (pose error); thick fuzzy walls (motion or depth noise); points floating far outside the room (mirrors, windows).

### 4. `align`: are floor, ceiling and axes right?
Prints: yaw, ceiling height, horizontal layers, normal sanity, camera height, Manhattan fit.
- **Good:**
  - Ceiling 2.4–3.2 m for typical homes; your laser ceiling height should match within ~3 cm.
  - ≥ 95 % of floor points face up.
  - Camera height p50 ≈ 1.1–1.5 m. If it's ~0.3 m or ~2.5 m, the wrong floor was picked (REVIEW #7).
- **Red flags:**
  - Large up-facing layers *below* 0. Floor picked too high.
  - Large up-facing layers just above 0. The floor is smeared, or the real floor is higher.
  - Manhattan fit < 50 %. Noisy normals, or a non-rectangular room.

### 5. `walls`: one line per physical wall?
Output: `walls.png` (grey = wall points; green = sharp line; red = smeared > 3 cm). Prints a table: axis, coord, area, spread, length along.
- **Good:** one line per real wall; spread ≤ 2 cm for LiDAR.
- **Red flags:** 2–3 parallel lines ~10 cm apart (smear → duplicates, REVIEW #4); lines in mid-room (furniture/cabinet fronts, REVIEW #5); a real wall with no line (not captured; it'll be dashed in the plan).
- **Compare with tape:** the distance between two opposite wall coords ≈ the room width.

### 6. `footprint`: right room shape?
Output: `footprint.png` (grey = horizontal surfaces from above; blue = wall-line grid; red = fallback lines at the footprint edge, i.e. no wall seen; green = interior cells; black = final polygon).
- **Good:** black polygon hugs the walls; no red lines where a real wall exists.
- **Red flags:** grey area leaking through doorways into another room (REVIEW #6); small steps in the black outline (duplicate wall lines).

### 7. `fp run`: final plan
Outputs: `plan.svg` (open in a browser), `plan.json`, `debug_bev.png`.
- Solid walls were observed; dashed walls were inferred. Numbers in brackets are length confidence (green ≥ 0.7, amber ≥ 0.45, red below).
- **Check:** every wall length against your laser or tape. The error should be smaller where confidence is high. If it isn't, confidence is lying (REVIEW #2).

### 8. `eval/arkit_gt.py`: vs laser ground truth (ARKitScenes only)
Prints a per-wall table: prediction, GT, error in cm, confidence. Caveats: REVIEW #3 (GT search window), and the ceiling can't be checked because the GT depth stops around 1.8 m.

---

## Inputs
| Capture | Status | Run |
|---|---|---|
| Phone **video** (.mp4/.mov) | works; depth and poses from MapAnything on Modal | `fp run video.mp4` |
| Phone **photos**, one folder per room | works; same path as video | `fp run photos/` |
| ARKitScenes-format LiDAR folder | works; frame filter on | `fp run <folder>` |
| iPhone/iPad Pro + 3D Scanner App export | reader not written | — |

For every real capture, also note:
- laser measurements of each wall, the room's length and width, and the ceiling height (3 readings each);
- which phone was used, and roughly how fast you moved.
