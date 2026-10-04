# Floorplan — phone capture → dimensioned floor plan with damage flags

`fp run <capture>` turns a phone capture into `plan.json`, `plan.svg` and a one-page `report.html`.
The capture can be a handheld **video**, a set of **photos**, or **LiDAR** depth (ARKitScenes format).
All three go through one pipeline and produce the same schema. Only the depth source and the confidence differ.

```
capture ─► CaptureBundle ─► depth + poses ─► point cloud ─► align ─► walls ─► rooms + doors ─► damage ─► report
             (one format)    LiDAR, or MapAnything            (gravity,   (planes)  (split at      (SAM 3, placed
                             on a Modal GPU                    floor, axes)          doorways)      on a surface)
```

## Status at a glance
| Brief | Where | State |
|---|---|---|
| Own pipeline; capture by a protocol a non-engineer can follow | [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md); everything after the raw files is in `fp/` | done |
| Photos, video and LiDAR → the same output, honest confidence | `fp/ingest/`, `fp/recon/`, `fp/geometry/plan.py: length_confidence` | LiDAR validated against a laser scan. Photos and video run end to end but are **not yet validated on a real phone capture** |
| Stitched whole-property plan, rooms placed, connected, dimensioned | one joint reconstruction per walk + `fp/geometry/rooms.py` (rooms split at doorways, doors as connections) | implemented and unit-tested; **not yet run on a real multi-room capture** |
| Surface damage and likely concealed damage, tied to surfaces | `fp/damage/`: SAM 3 masks → 3D → a wall, floor or ceiling; 4 rules labelled "hypothesis" | implemented; recall **not yet measured** on real damage |
| Accuracy and repeatability vs tape/laser; consumer-app benchmark | `eval/run_public.py` (laser), `eval/run_home.py` (tape) | laser: done. Tape on a real home and the Magicplan benchmark: **not done** |
| Weakest result → root cause → measured fix | [below](#weakest-result--root-cause--fix), `fp/ingest/quality.py` | done, verified on held-out scans |
| Runs cold on an unseen space | fixed thresholds, input auto-detected, clear failure messages | checklist in [COLD_RUN.md](COLD_RUN.md) |

## Results
Against a Faro laser scan, on three ARKitScenes scans of one kitchen (visit 483621). The measure is the
room width, wall to wall, scored only when both walls were observed. The fix was tuned on scan 48018559;
48018560 and 48018562 are held out. Every row: [eval/results.md](eval/results.md).

| Input (same three scans) | Widths scored | MAE (cm) | Max (cm) | Repeat SD (cm) | Mean confidence |
|---|---|---|---|---|---|
| LiDAR, original code | 2 of 3 | 10.9 | 16.2 | 8.3 | 0.73 |
| LiDAR, fix without the frame filter | 2 of 3 | 5.1 | 8.0 | 6.9 | 0.10 |
| **LiDAR, shipped** | 2 of 3 | **2.5** | **3.7** | **2.5** | 0.24 |
| Photos, 24 RGB frames | 1 of 3 | 9.4 | 9.4 | — | 0.05 |
| Photos, 12 RGB frames | 1 of 3 | 28.4 | 28.4 | — | 0.06 |
| Video, 120 RGB frames | 0 of 3 | — | — | — | — |

- **LiDAR** meets the 6 cm max-error and 1.5 % size-error targets (worst 3.7 cm, 1.49 %). It misses the 2 cm repeatability target by 0.5 cm.
- **Confidence is honest:** over the 6 scored widths of the current code it ranks the errors correctly (Spearman ρ = −0.75, p = 0.08). Camera-only runs score 0.05–0.06, against 0.12–0.37 for LiDAR.
- **Camera-only** numbers come from ARKitScenes' 256×192 RGB frames, about 1/19 of the pixels of a 1080p phone frame. On these, MapAnything's cloud is too noisy for the width walls to be found in most runs. Treat them as a lower bound; the real test is a phone capture (not run yet).
- One photos-24 run crashed on a degenerate room shape. That bug was fixed after the run.

## Weakest result → root cause → fix
**Weakest result:** the LiDAR room width came out +16.2 cm against the laser (target ≤ 6 cm), with confidence 0.64. It was confidently wrong.

**Root cause, found with the stage checks** (`fp check fusion|walls|align`):
- The point cloud was smeared. A side slice through a wall showed several parallel streaks, and walls were 7.4 cm thick.
- One real wall produced 3 candidate lines, and the plan took the outer one.
- Drivers: fast pans (rotation p95 102°/s) and blurry frames (21 %).
- It wasn't range or pose error: a cloud built from the laser depth with the same poses is sharp, and the smear drops when fast frames are removed.
- The floor detector had also picked a smeared tail 0.29 m below the real floor.

**Fix:**
- `fp/ingest/quality.py: select_frames` drops frames turning faster than 40°/s, frames under 30 % of the median sharpness, and frames where the camera moved less than 2 cm and 2°.
- Wall peaks merge within twice the wall's thickness.
- The floor is the biggest horizontal layer near camera height, not the lowest.
- Confidence falls with wall thickness.

| | Original code | Fix, filter off | **Fix, filter on** |
|---|---|---|---|
| Width MAE (cm) | 10.9 | 5.1 | **2.5** |
| Max error (cm) | 16.2 | 8.0 | **3.7** |
| Held-out scan 48018560 (cm) | +5.6 | +8.0 | **−3.7** |
| Wall lines on the tuning scan (4 real walls) | 11 | 7 | **3** |
| Median wall thickness, tuning scan (cm) | — | 7.4 | **4.0** |
| Mean confidence | 0.73 | 0.10 | 0.24 |

Before/after wall slices: [results/wall_slice_unfiltered.png](results/wall_slice_unfiltered.png) (streaks, 4.4 cm)
and [results/wall_slice_filtered.png](results/wall_slice_filtered.png) (one line, 2.7 cm). Each slice cuts the longest wall its own run found.

**Trade-off:** the filter keeps about 20 % of the frames (163–178). On scan 48018562 one bounding wall is then no longer seen, so its width can't be scored. That scan never shows a floor layer: the plan assumes one, says so, and halves the confidence.

## Quickstart
```bash
uv sync                                         # Python 3.12, uv
uv run python scripts/check_setup.py            # free: Modal login, Hugging Face secret (SETUP.md)

uv run fp run path/to/video.mp4                 # video walkthrough  → out/video/report.html
uv run fp run path/to/photos                    # photos, one folder per room (rooms get the folder names)
uv run fp run data/arkitscenes/48018559         # LiDAR (ARKitScenes layout)
uv run fp run <capture> --wet-rooms R2          # mark a bathroom/kitchen for the concealed-damage rule
uv run fp check all data/arkitscenes/48018559   # every stage, with debug images (TESTING.md)

uv run pytest                                   # unit tests
uv run python eval/run_public.py                # laser-scan evaluation → eval/results.md
uv run python eval/run_home.py                  # a home capture vs tape → eval/results_home.md
```
Capture: [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) (one page, for a non-engineer). Accounts: [SETUP.md](SETUP.md).
ARKitScenes scans go in `data/arkitscenes/<video_id>/` (raw `lowres_wide`, `lowres_depth`, `highres_depth`).

## Known limits
- **Not yet run on a real phone capture.** Photos, video, multi-room and damage are validated only by unit tests and public data.
- **Manhattan rooms only.** Angled walls are snapped to the two main axes.
- **No windows and no room types.** Doors come from narrow necks in the floor area. Rooms are named by photo folder, or R1, R2, …
- **Ceiling height** is reported only when enough ceiling is seen, and it isn't validated: the laser depth stops at about 1.8 m.
- **Confidence** is conservative, and not yet calibrated into ±cm bands.

## Next
- Run the capture guide on a real home; measure against tape; scan the same rooms with Magicplan.
- Calibrate confidence into ±cm bands from more measured rooms.
- LiDAR: test a time offset between depth and pose, and use ARKit's depth-confidence maps (REVIEW #17, #10).
- Damage: measure recall on real defects; tile large images for hairline cracks.

## Repository
| Path | What |
|---|---|
| `fp/ingest/` | readers: ARKitScenes, video, photos; `quality.py` frame filter; `load_capture` detects the input |
| `fp/recon/` | `lidar_fuse.py` depth → point cloud; `mapanything*.py` depth and poses for camera-only input, on Modal |
| `fp/geometry/` | `align.py` gravity, floor, axes; `plan.py` walls, room outlines, confidence; `rooms.py` room split, doors |
| `fp/damage/` | `detect_modal.py` SAM 3 on Modal; `project.py` masks → surfaces; `rules.py` concealed damage |
| `fp/report/` | `svg.py` plan drawing; `html.py` report; `debug.py` top view |
| `eval/` | laser evaluation, home evaluation, results |
| `results/` | sample outputs: plans, reports, before/after images |
| `tests/` | unit tests |
| Docs | [HOW_IT_WORKS.md](HOW_IT_WORKS.md) stage-by-stage walk-through · [REVIEW.md](REVIEW.md) code review and fix status · [TESTING.md](TESTING.md) stage checks · [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md) · [COLD_RUN.md](COLD_RUN.md) · [SETUP.md](SETUP.md) · [PLAN.md](PLAN.md) original plan |

ARKitScenes data and the images derived from it: © Apple, CC BY-NC-SA 4.0. MapAnything weights (`facebook/map-anything-apache`): Apache-2.0. SAM 3 (`facebook/sam3`): Meta's licence, gated.
