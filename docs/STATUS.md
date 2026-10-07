# Status

One section per work order (`prompts/NN-*.md`). Each session updates its own section at
hand-off: done / not done, real numbers, known bugs, and what the next work order needs to know.

## 00 — Repo setup and port of the prototype: **done**
- Sample data moved out of the repo root into `data/stray/<id>` (captures) and `data/zips/`; both
  ignored. Brief at `docs/brief/applied_ai_brief.md`. `planning/` is gitignored (private notes).
- Ported from branch `beta` in three commits (code only, unchanged except tests):
  `pyproject.toml`, `uv.lock`, `fp/bundle.py`, `fp/geometry/*`, `fp/recon/*`, `fp/ingest/*`
  (including `arkitscenes.py`), `fp/damage/*`, `fp/report/*`, `fp/check.py`, `fp/cli.py`, `tests/`.
- Not ported: `eval/` (ARKitScenes ground-truth evaluation), `results/`, `docs/img/`, beta's
  markdown docs, `scripts/collect_results.sh` (copies ARKitScenes outputs; not useful here).
- Tests: 14 pass (`uv run pytest -q`): 13 ported + 1 new Stray-detection guard. Dropped one:
  `test_eval_compares_only_sides_whose_walls_were_both_seen` (imports `eval/arkit_gt.py`, not ported).
- Verifier (fresh clone): `uv sync` + pytest + `fp --help` pass; largest tracked file is `uv.lock`
  (168 KB); no data in main's history.
- Env: Python 3.12.12 via uv (system Python here is 3.14; uv downloads 3.12 itself). See D2.
- `scripts/fetch_data.sh`: a stub that unpacks `data/zips/*.zip` into `data/stray/<id>`; the
  remote source (HF dataset or release) is decided in 08.

**What 01/02 need to know**
- `fp run` on a Stray folder exits with "reader not implemented": `is_stray()` in
  `fp/ingest/__init__.py` runs before the photo check, because `depth/*.png` used to pass as a photo
  folder and sent 3,430 depth maps to MapAnything on Modal. Replace that `SystemExit` with
  `fp/ingest/stray.py` and keep the detection order (test: `test_a_stray_folder_is_not_mistaken_for_a_photo_folder`).
- `rgb.mp4` is nominally 60 fps but really ~46 fps with gaps: use the odometry timestamps.
- `photos.load` writes its JPEG cache to `out/_cache/` relative to the working directory and ignores
  `--out` (known bug, ignored by git; fix in 05).
- Format facts are in `CLAUDE.md`. `arkitscenes.load` passes `up=(0,1,0)`; Stray is also y-up.
  `camera_matrix.csv` differs from per-frame intrinsics (c7d2: fx 1601.1 vs 1581.2): use per-frame.
- `Frame` has no confidence path, IMU or room label yet (GAMEPLAN §3).
- The current `plan.json` is `fp.plan/0.3`, with a 0–1 `confidence` per wall; 01 replaces it with
  `{value, lo, hi}` intervals and a published schema.
- MapAnything and SAM 3 run only on Modal (`fp/recon/mapanything_modal.py`,
  `fp/damage/detect_modal.py`); `modal` is a hard dependency. The local backend is work order 05.
- `fp/check.py` and the CLI help still talk about ARKitScenes; fine to reshape around Stray.

## 01 — Output contract: schema, intervals, renderer: **done**
- **Schema 1.0** (`fp/schema.py`, `schema/plan.schema.json`, `docs/SCHEMA.md`): Pydantic v2, `extra="forbid"`.
  `Measure{value, lo, hi, unit, method, observed}`; value null => observed false. Stable IDs `R1`, `R1.W1`,
  `R1.O1`, `R1.floor`, `R1.ceiling`, `D1`, `C1`, `S1`; `validate()` rejects dangling references, NaN, CW
  polygons, wrong units. Regenerate with `uv run python -m fp.schema` (a test fails if it's stale).
- **Renderer** (`fp/report/svg.py`, `html.py`, `write_outputs`): magicplan-style plan (thick walls, dashed
  inferred walls, door swings, double-line windows, `4.20 m ±0.05` labels, scale bar, +y arrow, legend);
  one-file report.html with tier badge, all tables, evidence crops, warnings, drift, timings. Intervals
  print rounded outward. Fixture render: `out/fixture/plan.svg` (verified visually, no overlaps).
- **CLI**: `fp run <capture> [--tier] [--backend local|modal] [--no-drift-correction] [--no-cache]`,
  `fp render <plan.json> --out DIR`. Always writes plan.json (validated), plan.svg, report.html, debug/.
  Missing stages raise `StageNotBuilt`/`StageError` (fp/contract.py) -> warning + `observed: false`.
- **Provisional intervals** (`fp/contract.py`, D5): plane position error = 1 cm × tier scale (1/3/6) +
  wall-band thickness, ×3 if inferred; length = sum of its two neighbours' errors (+3%/8% for video/photos).
  `intervals.calibrated = false` until 07.
- Tests: 65 pass. Verifier: all acceptance checks pass; its findings (HEIC/zero-frame crash, traceback on
  bad plans, fixture `length.observed` semantics) are fixed and tested.

**What 02 needs to know**
- Replace the `StageNotBuilt` in `load_capture` (fp/ingest/__init__.py) with the Stray reader; `fp run
  data/stray/c00a170fe1` then flows through `_geometry` in fp/cli.py into `contract.fill_from_geometry`.
- Geometry rooms must keep the legacy dict shape that `contract.room_to_schema` reads (`id, name, polygon
  (CCW), walls[{id W<k>, p0, p1, length_m, observed, coverage, spread_cm}], area_m2, ceiling_h_m`), or
  change room_to_schema with it. `extract_rooms`' `source_prior` arg is now unused (passed 1.0).
- Debug images go to `out/<capture>/debug/` (bev.png is written there now).
- Doorway "openings" are still floor-mask necks: width marked `observed: false`, ±10 cm (04 replaces).
- Damage only runs with `--backend modal` (old SAM 3 path, converted by `contract.damage_to_schema`);
  locally it's a warning (06).
- Known gaps: pydantic reports field errors before cross-reference errors (two passes); photos.load still
  ignores HEIC and writes its cache to ./out/_cache (05).
## 02 — LiDAR tier: Stray Scanner ingest and single room end to end: **done** (thickness target missed: 2.2 cm vs 2 cm)
- **Reader** `fp/ingest/stray.py`: odometry.csv poses (OpenCV camera, world y-up, `up=(0,1,0)`), per-frame K,
  timestamps from odometry (not i/60), `depth/` + `confidence/` paths. `rgb.mp4` decoded once into
  quarter-res JPEGs under `out/_cache/stray/<id>/` (4.6 s cold for 1,714 frames; marker keyed on size+mtime);
  `Frame` gained `confidence` and `video_index` (full-res crops for 06). `--tier video` on a Stray folder uses
  its `rgb.mp4`; `--tier photos` raises StageNotBuilt (05). Broken folders give an `ingest` StageError.
- **Fusion** (`fp/recon/lidar_fuse.py`): confidence == 2 (`MIN_CONFIDENCE`), 4 m range, quality filter, 1 cm voxel.
  `free_space_rays` feeds the footprint.
- **Footprint** (`fp/geometry/plan.py`): horizontal surfaces ∪ cells crossed by top-down camera→point rays from
  ≥ 3 frames ∪ a 0.3 m band around the camera path (D9). **Rooms** (`fp/cli.py rank_rooms`): R1 = most camera time ≥ 0.3 m inside; rooms with < 3 s
  are kept and warned "partially observed" (D10). Ceiling not seen → `observed: false` + a `ceiling` warning.
- **c00a170fe1** (`out/c00a170fe1/`): 11 s warm, 16 s cold (verifier). 383/1,714 frames kept
  (276 blurry, 237 fast, 818 still). R1 **19.54 m² [18.85, 20.23]**, 6 walls all observed: 6.05 m and 3.41 m
  outer walls, 4.79 m long side with a 1.26 × 0.87 m notch (kitchen counter); camera 32.1 s inside.
  R2 6.03 m² (seen through the door, 1.1 s inside → partially observed). Footprint history: floor only
  12.7 m² → + seen-through rays 15.1 → + camera path 19.5. Ceiling **not observed** (camera pitch
  −46°…−8°, highest point 1.9 m). Median wall thickness **2.17 cm** (18 walls); slice wall σ 3.0 cm.
  Debug: `debug/fusion_topdown.png`, `debug/wall_slice.png`, `debug/rooms_split.png`, `debug/bev.png`.
- **1a8384c3f6**: 42–55 s, 1,308/5,250 frames, 3 rooms (R1 44 m² is several rooms merged, rough), every ceiling
  `observed: false` (highest point 2.33 m). Median wall thickness 4.18 cm: drift across the flat (03).
- **Experiment D** (`uv run python scripts/exp02_fusion.py data/stray/c00a170fe1`; same 12 wall planes, median cm):

  | variant | median | mean | fuse s |
  |---|---|---|---|
  | conf==2, filter, 2 cm voxel (baseline) | 2.78 | 2.60 | 3.3 |
  | conf>=1, filter | 2.81 | 2.67 | 3.5 |
  | conf==2, no filter (all 1,714 frames) | 2.73 | 2.85 | 13.3 |
  | conf>=1, no filter | 2.93 | 3.00 | 14.0 |
  | depth i + pose i−2 / i−1 / i+1 / i+2 | 4.12 / 3.14 / 3.49 / 4.35 | | |
  | conf==2, filter, **1 cm voxel (chosen)** | **2.17** | 2.52 | 6.8 |

- **Verifier** (fresh agent): pytest, cold run < 3 min, schema-valid, 1a83 ceilings not observed, git clean: pass.
  Its findings, fixed: R1 polygon cut through floor the camera walked on (→ camera-path rule, 15.1 → 19.5 m²);
  truncated odometry passed silently (→ ingest warning "N frames without a pose"); missing `confidence/` had no
  warning (→ added); main room on a too-short capture got a "seen through a doorway" warning (→ R1 exempt);
  `fps_median` 60 was misleading (→ `fps_nominal` 60 + `fps_mean` ≈ 46).

**Known problems / what 03+ need to know**
- Not fixed: with the floor assumed (truncated 50-row capture), a 0.54 m² placeholder room still gets a ±13 %
  interval; only a warning says it's rough. 07 should widen intervals (or drop rooms) when the floor is assumed.
- The bathroom at the bottom of c00a is not split from the living room (doorway neck ratio too wide for
  `NECK_RATIO`); 1a83 merges several rooms into R1. Room splitting needs work on the whole flat in 03/04.
- Floor-area `observed: false` (any inferred wall) has no warning explaining why; the method string is the only hint.
- The longest wall (x = 2.38) shows two stripes ~3 cm apart in `wall_slice.png`: two passes misregistered (drift, 03).
- Whole-flat walls are 4.2 cm thick vs 2.2 cm single room: drift grows with capture length (03's metric).
- `plan["source"]` now carries `room_occupancy_s` and `wall_thickness_cm`; the camera path is in
  `bundle.meta["_trajectory"]` (all frames, before the filter).
- c7d28f72c6 (with ceiling) not run in this work order.
## 03 — Drift accountability and the stitched whole-property plan (LiDAR): **done** (repeatability gate missed: spans ±5.5 cm vs 1 cm)
- **A. Drift** (`fp/recon/drift.py`, on by default, `--no-drift-correction` off): submaps every 3 m / 8 s, point-to-plane
  ICP loop closures (fitness ≥ 0.2, RMSE ≤ 1.5 cm, constrained geometry, plausible size, ≤ 1° tilt), Open3D pose graph with
  line process, yaw-only result (gravity stays ARKit's), corrections **blended in time** between submap centres (D12).
  Plane-anchored fallback exists (`method="plane"`), measured and not used: it never lowers loop residuals.
  `plan.drift.metrics` has off/on thickness, double walls, rooms, footprint, loop + odometry RMSE before/after,
  heading spread (std of per-submap wall angle), max correction. `debug/drift_ablation.png`: off | on, doubles in red.
  Experiment script `scripts/exp03_drift.py` (variants off / posegraph-step / posegraph / plane).
  | capture | thickness cm off→on | double walls off→on | loop RMSE cm | loops accepted | drift s |
  |---|---|---|---|---|---|
  | c00a170fe1 | 2.17 → 1.71 | 4 → 2 | 4.20 → 1.91 | 2/3 | 0.9 |
  | 1a8384c3f6 | 4.18 → 1.94 | 13 → 7 | 3.34 → 1.83 | 8/25 | 5 |
  | c7d28f72c6 | 3.62 → 3.27 | 32 → 15 | 3.22 → 2.12 | 17/138 (5 more pruned by the graph) | 12–15 |
  1a83's heading drift is real: per-submap wall angle +3.0° → −0.8° over 115 s (spread 1.50° → 0.79° corrected). Its
  "max correction 68 cm / 6.9°" is mostly a rigid turn of the whole map toward submap 0, which `align` removes.
  Bug fixed on the way: a "double wall" used to pool every cell on a coordinate across the flat; now it must be one
  continuous side-by-side stretch ≥ 0.5 m (old counts 19/26 off were inflated).
- **B. Stitch** (`fp/geometry/plan.py extract_rooms`, `rooms.py`, `cli.py rank_rooms`, D11), drift on:
  **1a8384c3f6** 9 rooms, 1 connector, 12 connections, ~80–105 s, ceiling not observed (highest point 2.35 m);
  **c7d28f72c6** 9 rooms, 1 connector, 11 connections, ~195–276 s, ceiling 2.44 m; overlaps > 1 % repaired and warned
  (c7d2 R9/R3 10 %). No frame subsampling beyond the quality filter (1,308 / 2,629 frames kept). Known: part of the
  corridor merges into the open kitchen; c7d2 R6 is a region outside a window (warned partially observed).
- **C. Repeatability** (`eval/repeatability_lidar.md`, `scripts/repeatability.py`, `fp/geometry/register2d.py`, D13):
  same flat (yaw 90.4°, 69 % inliers vs 24 % runner-up); 6 rooms matched; wall-to-wall spans **0/16** in the gate,
  median 5.5 cm; drift off: 3/12, 1.9 cm, but 55 % inliers and edges 40 cm vs 12 cm. Correction trades a few cm of
  in-room accuracy for global consistency; segmentation differences dominate room widths/areas.
- Tests: `tests/test_drift.py` (12), `tests/test_stitch.py`. Debug images: `out/<id>/debug/drift_ablation.png`,
  `rooms_split.png`, `bev.png`; experiment images `out/_exp03/<id>/drift_exp03_<variant>.png`.
- **Verifier** (all 6 checks pass): fresh runs 95 s / 174 s, 0.000 % residual overlap, numbers match. Found: a 10° yaw
  injected into half of 1a83 is mostly left uncorrected (plausibility gate rejects the loops; ~2° removed) while loop RMSE
  still looks good -> `fp run` now warns when the heading spread widens by > 0.5° (`HEADING_WORSE_DEG`, fp/cli.py).
  `debug/rooms_split.png` shows the watershed before the room filter (a dropped 1.9 m² strip still coloured, kept R6 grey):
  redraw it from the final rooms in 04. `--max-frames` is a no-op on LiDAR captures (it only caps learned-depth frames).
- **For 04 and later:** openings/connections from 03 are rough (doorway gaps on wall lines, shared walls have
  `opening_id: null`). Ceiling is one flat-wide value (2.44 m on c7d2); per-room ceilings are 04. If 07 shows
  span accuracy matters more than double walls, try a per-room refinement after correction (D13).
  (The old subagent worktree from 03 is removed.)
## 04 — Openings (doors, windows, passages) and per-room ceiling height: **done; ceilings pass, openings gate not met (50 % vs 85 %, by eye)**
- **B. Ceiling** (`fp/geometry/ceiling.py`, D04.5): per room, largest down-facing layer ≥ 1 m² 2.1–3.6 m above the
  room's own floor; value = medians' difference; half = 1 cm + 1.645 √(σc²/patches + σf²/patches) (25 cm patches),
  +3 cm if the room's floor wasn't seen. Other layers -> `ceiling` warnings ("multi-level ceiling").
  **c7d28f72c6:** R1 2.47, R2 2.45, **R3 3.09, R4 3.04**, R5 2.38, R7 2.33 (plan floor, ±4 cm), R8 2.45, R9 2.49 m;
  R6 (balcony, never entered) not observed with a reason; R1 and R8 warn of a 3.05–3.08 m layer (R1 merges the hall
  with the high-ceiling dining area). The verifier found R3/R4 at ~3.05 m plausible (normal high ceiling; hall,
  corridor and bathrooms have dropped soffits). **1a8384c3f6 and c00a170fe1: every room not observed**, as required.
  Repeatability across 1a83/c7d2 can't be scored (1a83 has no ceiling); waits for own repeat captures (07).
- **A. Openings** (`fp/geometry/openings.py`, D04.1–D04.4, D04.6): per wall an elevation grid (2 cm cells) where each
  depth ray votes wall / seen-through / (nothing = unobserved), frames counted not rays; regions classified door
  (floor, 0.6–1.6 m, lintel 1.9–2.4 m, or no lintel seen but seen through ≥ 1.2 m), window (sill 0.3–1.5 m),
  passage (> 1.6 m or no lintel, room behind). Width = jamb planes from reveal points (wall-end percentile
  fallback). Mirror test (reflected points land on the room) and recess test (surface < 40 cm behind); a
  door-shaped recess with a lintel = closed door. One opening seen from both rooms is listed once. 03's floor necks
  remain only between rooms no measured opening links (warned).
  | capture | openings | doors / windows / passages | openings s (+ RGB sheets) | total s |
  |---|---|---|---|---|
  | c7d28f72c6 | 10 | 5 / 1 / 4 | 44.2 (+10.5) | 226 |
  | 1a8384c3f6 | 12 | 8 / 1 / 3 | 22.7 (+6.7) | 102 |
  | c00a170fe1 | 2 | 1 / 0 / 1 | 3.1 (+0.4) | 23 |
  c7d2 doors: R1.O1 0.904 m (lintel 2.08), R1.O2 0.905, R7.O1 0.893, R9.O1 0.713, R2.O2 0.915 (closed). Width
  half-widths 3–12 cm (provisional). **Width accuracy vs ±2 cm cannot be checked: no tape on the sample data.**
- **Verification by eye** (verifier subagent, RGB 5 frames per opening, `debug/openings/walls_R*.jpg` for misses),
  c7d28f72c6. First pass 12 found: 6 correct, 3 phantoms, 3 wrong kind, 4 misses = **38 %**. Fixed (D04.6): closed
  door needs a lintel; an opening must fit its wall; mirror sample fixed; sheets skip occluded boxes. After: 10 found,
  **7 correct, 1 phantom** (vanity mirror, mirror score 0.30 < 0.5: threshold not lowered, real doors score to 0.38),
  **2 wrong kind** (glazed balcony slider R4.O1 and office door R8.O1 called passages), **4 misses** (R5.W2 tall
  narrow window, sill 0.15 m: no rule; R1.W5 and R3.W6 curtained windows; R2.W4 closed door seen only above 1.35 m)
  = **7/14 = 50 %** (in-sample: rules came from this capture). 1a83 spot check (6 of 14, before the fixes): 2 good;
  phantoms there were a standing mirror (R2.O4 window 0.42, still present), a sofa-backed "window" and a vanity
  "closed door" (both now rejected). c00a: R2.O1 door 0.92 correct; its toilet-niche "closed door" is now rejected.
  Adversarial (verifier): zeroed depth -> 0 openings; walls moved/rotated -> no new openings, except a wall moved
  30 cm in gave a "closed door" (fixed by the lintel rule).
- Tests: `tests/test_ceiling.py` (8), `tests/test_openings.py` (7: rendered depth of a two-room scene: door 0.90 m
  within 2 cm, window, closed door vs niche, wardrobe = unobserved, mirror test, schema, dedupe). **121 pass.**
- Debug images (per capture): `debug/openings/<wall_id>.png` (elevation as seen from inside: white unobserved,
  grey wall, blue seen through; boxes red door / blue window / green passage / orange rejected) for every wall with
  an opening or a candidate; `debug/openings/<opening_id>_frames.jpg` (5 RGB frames, opening projected; some have
  only 1–2 frames when few frames see it unoccluded); `debug/openings/walls_<room>.jpg` (2 frames per wall).
- **Known problems / what 05+ need to know**
  - Openings gate (≥ 85 %) not met. Biggest remaining causes: windows behind curtains (no see-through rays); low-sill
    narrow windows (no rule between 0.05 and 0.3 m sill); wide glazed doors and doors with an unseen lintel
    called passages; mirrors at oblique angles (mirror score 0.3–0.5). An RGB check (door leaf/handle/glass) is the
    obvious next step if 09 has time.
  - `plan.svg`: opening labels crowd around small rooms (R7–R9 on c7d2).
  - Camera tiers: `fp run` runs openings on their predicted depth too (no confidence map -> all treated as
    confident); not evaluated. Photos give too few frames to reach MIN_FRAMES = 2 in most cells.
  - `debug/rooms_split.png` still shows the watershed before the room filter (03 note, not redrawn).
  - fp run on c7d2 grew 195 -> 226 s (openings 44 s + sheets 10 s).
## 05 — Video and photo tiers (no depth, no poses): **runs end to end; accuracy weak, photo-tier gates not met** (worktree ../floorplan-cam, branch track/camera, merged 7 Oct)
**Built**
- **Ingest** (`fp/ingest/photos.py`, `video.py`, `media.py`): HEIC (pillow-heif), EXIF rotation, P3 -> sRGB, EXIF 35 mm focal ->
  `_K_prior`; room folders required (loose photo -> StageError); identical doorway photo in two folders -> one frame with
  `_frame_room_sets`; video via bundled ffmpeg (imageio-ffmpeg), HDR tonemap, autorotate, 3 fps, <= 240 keyframes;
  frame cache keyed by content hash under `<out>/../_cache`. Tests: HEIC, 10-bit HLG HEVC with rotation (`tests/test_camera_ingest.py`).
- **Recon** `fp/recon/camera.py`: MapAnything locally (cuda > mps > cpu) or `--backend modal`. **Cache per model pass**
  (key = model + params + frame content ids + K priors; raw outputs only, merge/geometry always rerun). Chunks of
  `CHUNK = 24` views, `OVERLAP = 6`, merged by **median depth ratio** at shared pixels + average pose move; final scale =
  **median of the chunks' metric votes** (D05.2). Warning when a merged chunk still disagrees by > 0.25 m. Photos: one
  joint pass, rooms linked by shared photo / >= 40 inliers, unlinked rooms unplaced + warning. Gravity: **image up**,
  overruled only by 2x support (D05.4). Scale priors widen intervals (D05.5).
- `scripts/make_camera_tiers.py`: derived video gets the **display-rotation tag** an iPhone writes; stills get EXIF
  focal; room photos are **overlapping sweeps** (D05.7, D05.8). `scripts/cache_sync.py` push/pull/list (HF dataset +
  sha256 manifest; not yet tried against a real repo). `eval/cross_tier.py` (+ test).

**Numbers** (M5, 16 GB, MPS; the 03 jobs ran in parallel for part of the session)
- MPS per pass (518x392, bf16): 4 views 12.6 s / 6.6 GB; 16 views 18.7 s / 8.75 GB; **24 views 37.1 s / 9.83 GB**; model
  load 40-50 s. 40 views swaps on 16 GB.
- **~120 keyframes**: c00a video, 109 keyframes, 6 passes: **338 s live, 10.8 s replayed** (peak footprint 14.3 GB).
  The first attempt took 1,538 s while 03's c7d2 job held the memory (swap 15 GB): don't run both together.
- Photos: c00a 7 photos 47 s live; c7d2 17 photos 54 s live (one pass each). **40 photos not timed** (no 40-photo set;
  expect 2 passes, ~2.5 min).
- Cache acceptance: live run vs replay, plan.json **byte-identical outside `timings`** (c00a + c7d2 photos, c00a video).
- vs LiDAR reference (eval/cross_tier.py): c00a video footprint IoU **0.62**, area 30.4 vs 25.6 m², camera 1.25 m above
  floor (LiDAR 1.42-1.48); chunk scales 0.89-1.15 (were 0.78 -> 0.22 before the merge fix). c00a photos IoU 0.21
  (5.4 vs 25.6 m²); c7d2 photos IoU 0.17 (12.6 vs 73.3 m², 2 of 4 rooms, adjacency not recovered), ceiling 2.47 m
  (reference 3.08). No reference area falls inside a camera-tier interval.
- Photo poses vs LiDAR (c7d2): median relative-rotation error 35 deg (farthest-point photos) -> **10 deg** (sweeps);
  positions still unregistered across sweeps/rooms (RMS 3.0 m on a 3.1 m spread).

**Not done / known problems**
- **c7d2 video** (finished just after the merge, pre-merge code): 240 keyframes, 13 passes, **511 s live**, 14.3 GB peak.
  Chunk scales 0.91-1.05 (no drift), metric vote 1.19, camera 1.43 m above floor (LiDAR ~1.49). But **one merged room**:
  82.2 m² vs the reference's 73.3 m² footprint (+12 %), footprint IoU 0.66, adjacency not recovered, ceiling 3.60 m
  (reference 3.08; the scale prior widened intervals 11 %). 11 of 12 chunk merges warn (0.27-1.45 m disagreement):
  pose noise between chunks blurs the doorways, so the watershed split finds no necks.
- `eval/cross_tier_sample.md` not written yet (numbers above come from `eval/cross_tier.py`).
- Acceptance "photos: one stitched plan, correct adjacency, no overlaps": **not met**. No overlaps (0 m²), but rooms are
  undersized and adjacency is wrong. Two causes: (1) the footprint rule needs rays from >= 3 frames (`MIN_RAY_HITS`,
  fp/geometry/plan.py), which 7-17 photos rarely give (c00a: 0.8 m² vs 9.0 m² from >= 1 photo), and room folders are not used
  to split rooms. Fix deferred to after the merge by the user (D05.9): `MIN_RAY_HITS` by tier + per-room footprint from
  each folder's photos. (2) Photo positions are not registered across sweeps (see above).
- c7d2 photos: both plan rooms are named "R2" (`_name_rooms` in fp/cli.py takes the majority folder; no uniqueness).
- Video: the kitchen stretch of c00a (19-37 s) has inconsistent model poses (per-pass RMS vs LiDAR 0.35-0.63 m); a
  warning names it. Camera-tier intervals are far too narrow (07 must calibrate them per tier).
- Verifier subagent not run for 05.

**What the next work orders need to know**
- 08 (protocol): photos must overlap (sweeps, ~half a view between neighbours, a doorway photo in both folders); video
  is the stronger camera tier.
- 07: calibrate camera-tier intervals; `eval/cross_tier.py` gives the per-room errors against the LiDAR reference.
## 06 — Damage regions, concealed-damage flags, scope line items: not started
## 07 — Calibrated intervals and the benchmark harness: not started
## 08 — Capture protocol, device matrix, README, report, compliance matrix: not started
## 09 — The fix loop: not started
## 10 — Cold-run rehearsal: not started
