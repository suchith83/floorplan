# Code review — CP1 (LiDAR, one room)

Scope: `fp/ingest/arkitscenes.py`, `fp/recon/lidar_fuse.py`, `fp/geometry/{align,plan}.py`,
`fp/report/*`, `fp/cli.py`, `eval/arkit_gt.py`. Evidence comes from `fp check all` on
ARKitScenes `48018559` (galley kitchen + glimpse of living room).

## Verified correct
- **Pose convention** (`.traj` = world→camera, inverted to camera→world). The laser-GT cloud built with the same poses is sharp (walls within one 5 cm bin), so poses and intrinsics are applied correctly.
- **Pose interpolation** (slerp). Poses arrive at 10 Hz and frames at 60 Hz; using the nearest pose put up to 50 ms of motion into each frame.
- **Manhattan yaw** (89.8° here), floor normal orientation (100 % of floor points face up), step-merge index logic, outward label placement in SVG.
- **Length confidence uses the neighbouring walls.** That is the right dependency: a wall's length is the distance between the two planes that bound it.

## Findings (P0 = hurts accuracy now)

| # | Pri | Component | Problem | Evidence | Fix |
|---|---|---|---|---|---|
| 1 | P0 | ingest | **No frame-quality filter.** Blurry frames, fast pans and duplicates all go into the fusion. | 172/824 frames blurry; rotation p95 102°/s, max 147°/s. Keeping rotation < 40°/s plus sharp frames: wall lines 11 → 4, median smear 7.7 → 5.0 cm, width 2.66 → 2.54 m (laser ≈ 2.51 m). | Drop frames with rotation > 40°/s or sharpness < 30 % of median; dedupe by pose change (< 2 cm and < 2°). The same filter applies to the video path. |
| 2 | P0 | confidence | **Confidence ignores smear.** A wall with 11 cm spread still scores 0.87. | `fp check walls`: every line flagged SMEARED while plan confidences are high | Store the per-wall spread in `plan.json` and multiply confidence by `exp(-spread / 3 cm)`. |
| 3 | P0 | eval | **The GT wall is searched only within ±15 cm of *our* wall.** Bigger errors drop out, or match a cabinet front, so the reported error is too optimistic. | 5 of 9 items had no GT in an earlier run | Detect GT planes independently (same histogram method on the GT cloud, outermost strong plane), then match them to ours. |
| 4 | P0 | walls | **One smeared wall becomes 2–3 parallel lines 10 cm apart,** which creates the grid jogs. | x = 3.14 / 3.32 / 3.42; y = 0.66 / 0.76 / 0.85 | Non-maximum suppression with a window tied to the measured spread; keep the outermost inward-facing plane. |
| 5 | P1 | walls | Lines don't store which way they face, so a wall can't be told from a cabinet front, and partition thickness can't be measured. | — | Store the normal sign per line. |
| 6 | P1 | footprint | The footprint spills into rooms seen through doorways (living-room rug area included). | `footprint.png` | Room segmentation (CP4). |
| 7 | **P0** | align | **Floor = lowest layer ≥ 0.3 m², so the detected floor sits ~29 cm below the real one.** Up-facing points spread from −0.2 to +0.5 m, with a tail down to 0.5 m below the floor (likely reflections in glossy tiles). Ceiling height is off by the same amount. | `fusion_4_cleanup.png`: the floor in the wall slice sits ~0.29 m above the blue z = 0 line; the layer table has no sharp floor peak | Use the up-facing layer with the **most** area inside camera height − [0.8, 2.0] m, not the lowest one. |
| 8 | P1 | align | **Crashes** (`min() of empty`) when no floor is seen, instead of telling the user what to recapture. | Reproduced with strict frame filtering | Raise a clear error: "floor not observed — sweep the floor". |
| 9 | P1 | recon | Normals are oriented toward the *nearest* camera, which can sit on the other side of a thin wall or object. | — | Carry each point's viewing direction through voxel averaging. |
| 10 | P1 | recon | ARKit depth-confidence maps aren't used, so flying pixels at depth edges add smear. | — | Download the `confidence` asset and keep only high-confidence depth. |
| 11 | P2 | walls | Only 32–39 % of wall normals are within 5° of an axis (noisy PCA normals), so the axis gate discards points. | `fp check align` | Re-measure after #1/#10; fit plane normals per line. |
| 12 | P2 | grid | `_grid` keeps the *first* of two lines < 10 cm apart, not the stronger one. | — | Keep the one with the larger area. |
| 13 | P2 | polygon | Axis detection uses `abs(dx) < 1e-6` and the polygon isn't validated after `_merge_steps`. | — | Classify by \|dx\| < \|dy\|; check `is_valid`, fall back if not. |
| 14 | P2 | coverage | One stray point marks a whole 5 cm bin as covered. | — | Require ≥ 3 points per bin. |
| 15 | P2 | cli | `SOURCE_PRIOR` values are placeholders. | — | Calibrate them from eval (planned). |
| 16 | P2 | svg | The footer is clipped and labels overlap on short walls. | `plan.svg` | Cosmetic. |
| 17 | P0 | recon | **Every surface carries ~5–8 cm of smear that isn't caused by range.** The floor std is 8.2 / 8.1 / 8.2 cm with max depth 4 / 2.5 / 1.5 m. Smear drops when fast frames are removed, and the laser-GT cloud (same poses) is sharp. | Suspects: a time offset between depth and pose, and ARKit depth quality on glossy tiles. | Experiment: sweep a depth↔pose time offset of ±50 ms and pick the one that minimises wall spread; use the depth-confidence maps (#10). |
| 18 | P1 | recon | 53 % of points have normals that are neither horizontal nor vertical, so both the walls and footprint stages ignore them. | `fusion_5_normals.png` | Follows from #17 (a fuzzy surface gives random PCA normals); re-check after fixing it. |

Platform note: Open3D 0.20's TSDF returns empty clouds on macOS arm64, even for a single frame with an identity pose. We use back-projection plus voxel averaging instead (`fuse_depth`).

**Weakest-result candidate for the brief:** #17 + #1 + #2 (with #7 as a consequence). Fast handheld motion → smeared planes → duplicate walls → wrong dimensions, while confidence stays high. Before/after numbers are in the table above. The fix still needs to be verified on held-out captures.

## Status after the fixes (2026-10-03)
Measured on three ARKitScenes scans of one kitchen; numbers in `eval/results.md`.

| # | Status | What changed |
|---|---|---|
| 1 | **Fixed** | `fp/ingest/quality.py: select_frames` drops frames turning faster than 40°/s, sharpness under 30 % of the median, and frames where the camera moved < 2 cm and < 2°. Used by `fp run` for LiDAR. |
| 2 | **Fixed** | Each wall stores its thickness (`spread_cm`, robust MAD). Confidence × exp(−(thickness − 1 cm) / 3 cm) (`fp/geometry/plan.py: length_confidence`). |
| 3 | **Fixed** | `eval/arkit_gt.py` finds laser walls on the laser cloud alone and compares frame-independent wall-to-wall distances. |
| 4 | Improved | Peaks merge within max(10 cm, 2 × the stronger line's thickness). The filter removes most duplicates (wall lines 11 → 3 on the test scan). |
| 5 | **Fixed** | Each line stores the side it faces (`faces`). The laser eval uses it. |
| 6 | Addressed | The footprint is split into rooms at doorways (`fp/geometry/rooms.py`); a glimpse of another room becomes its own region. |
| 7 | **Fixed** | Floor = biggest up-facing layer 0.8–2.0 m below the median camera, unless a layer 25 cm lower has half its area (`align.pick_floor`). |
| 8 | **Fixed** | `FloorNotFound` with a re-capture instruction. `fp run` then assumes the floor 1.2 m below the camera, says so in `plan.json`, and halves confidence. |
| 9, 10, 11 | Open | Not needed for the current accuracy; listed under "Next" in README. |
| 12, 13, 14, 16 | Open | Minor; no measured effect. |
| 15 | Partly | Priors unchanged (0.9 / 0.65 / 0.5); calibration data now in `eval/results.md`. |
| 17 | Mitigated | The frame filter cuts median wall thickness from 7.4 to 4.0 cm (tuning scan) and 7.0 to 2.0 cm (held out). The depth↔pose time-offset test was not run. |
| 18 | Mitigated | Follows #17. |
