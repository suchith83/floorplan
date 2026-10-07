# Design decisions

One entry per real decision: context → options → choice → evidence. Newest at the bottom.

## D1. Stray Scanner as the LiDAR capture route
- **Context.** The brief offers two capture routes: ship our own iOS app (TestFlight) or name a
  stock App Store tool plus a one-page protocol. We have no iPhone Pro to build or test an app on.
  The evaluator's LiDAR sample data (three captures) are Stray Scanner exports, and at the walk-in
  they will capture with their own iPhone following our protocol.
- **Options.** (a) Own ARKit/RoomPlan app: most control, but no device to develop on and a
  TestFlight review we can't test. (b) A stock LiDAR logger: Stray Scanner, 3d Scanner App, Polycam
  raw export. (c) RoomPlan-based consumer apps: give a finished model, not raw depth and poses.
- **Choice.** (b) Stray Scanner. It's free, exports raw depth (256×192 mm), per-pixel confidence,
  per-frame camera poses and intrinsics, IMU, and the RGB video, all in plain files.
- **Evidence.** It's the format the evaluator already uses (`data/stray/*`), so our reader is
  exercised on their real data, and the walk-in capture will arrive in the same format. The
  format was checked by back-projecting depth with the logged poses: the OpenCV camera convention
  gives sharp walls (see `CLAUDE.md`).

## D2. Python 3.12 with uv, prototype ported from branch `beta`
- **Context.** The pipeline must go from `git clone` to a running plan in under 15 minutes on a
  clean machine. The round-1 prototype (branch `beta`) already pins Python 3.12 and a `uv.lock`.
- **Options.** (a) uv + lockfile; (b) conda env; (c) plain pip + requirements.txt.
- **Choice.** (a) uv with `requires-python = ">=3.12,<3.13"`. uv downloads 3.12 itself if the
  machine lacks it (this Mac has 3.14 system Python; uv fetched 3.12.12). Open3D wheels lag new
  Python releases, which is why we stay on 3.12 rather than 3.13+.
- **Evidence.** `uv sync` built the env from the lockfile here and `uv run pytest -q` passes (13 ported tests).
  The prototype was ported in three labelled commits rather than one lump, so the history shows
  what was reused; ARKitScenes ground-truth evaluation code (`eval/`) stayed behind, since the
  benchmark is rebuilt around our own captures in work order 07.

## D3. Intervals on every number, not a 0–1 confidence score
- **Context.** The brief asks for "a confidence interval on every measurement", scores calibration
  at every tier, and says confident garbage on thin input caps the total score. The round-1 prototype
  gave each wall a 0–1 confidence (input prior × coverage × smear penalty).
- **Options.** (a) Keep the 0–1 score; (b) a 90% interval `[lo, hi]` in metres on every value;
  (c) a full error distribution per value.
- **Choice.** (b). Every measurement is a `Measure{value, lo, hi, unit, method, observed}`.
  `[lo, hi]` is a stated 90% interval, so about 9 in 10 true values should fall inside it. When nothing
  was observed, `value` is null and `observed` false: never invent a number.
- **Evidence.** A 0–1 score can't be checked against a tape measure; an interval can. Work order 07
  counts how many tape values fall inside their intervals (stated 90% vs observed coverage), which is
  exactly the calibration the brief scores. (c) is more than anyone can check by hand at the defense.

## D4. Stable surface IDs: R1, R1.W1, R1.O1, R1.floor, R1.ceiling
- **Context.** Damage regions, concealed-damage flags and scope line items must be "keyed to surfaces".
- **Options.** (a) Array positions; (b) global counters (W17); (c) room-scoped dotted IDs.
- **Choice.** (c). A wall is `R1.W1`, an opening `R1.O1`, and each room's floor and ceiling are
  `R1.floor` and `R1.ceiling`. Damage (`surface_id`), scope (`surface_id`), openings (`wall_id`) and
  connections (`opening_id`) all refer to them. `fp.schema.validate` rejects any reference to an ID
  that doesn't exist, so a dangling key fails the run instead of shipping.
- **Evidence.** The ID tells a reader which room a surface is in without a lookup, and a quote
  ("repaint R2.W2, 6.0 m²") stays readable.

## D5. Provisional interval model until calibration
- **Context.** Intervals are needed from work order 01 on, but tape ground truth only arrives in 07.
- **Choice.** Every wall plane gets a position error `pos = 1 cm × tier scale + measured wall-band
  thickness`, ×3 if the wall was never seen. A length's half-width is the sum of its two neighbouring
  planes' `pos` (plus 3% / 8% of the length for video / photos, matching the brief's gates, since their
  metric scale is learned). Area half-width = Σ length × pos (+ 2 × scale error × area); perimeter =
  2 Σ pos (+ scale error × perimeter); ceiling = 2 cm × tier scale (+ scale error × height). Measured
  smear is added as-is, not multiplied by the tier scale. Terms add (worst case), so it errs wide.
  Tier scale: LiDAR 1, video 3, photos 6. All constants are in `fp/contract.py`, labelled provisional, and the plan says
  `"calibrated": false`.
- **Evidence.** Each term is one sentence to explain. It widens where the data are thin (inferred
  walls, smeared walls, camera-only tiers), which is the behaviour the brief asks for; 07 scales it.

## D6. Missing stages give warnings and `observed: false`, not crashes
- **Context.** "One command per capture, JSON to the published schema" must hold before every stage exists.
- **Choice.** A stage that isn't built raises `StageNotBuilt`; `fp run` records it as a warning and still
  writes a schema-valid `plan.json`, `plan.svg` and `report.html`. Real errors still raise. The default
  `--backend` is `local`; Modal is opt-in, because the walk-in must run on the evaluator's machine.

## D7. Stray video decoded once into a frame cache
- **Context.** Stray stores RGB as one HEVC `rgb.mp4`; the rest of the pipeline reads frames as image files.
  HEVC seeking is slow, and the frame filter needs every frame's sharpness anyway.
- **Options.** (a) Lazy random-access decode per frame; (b) extract full-res PNGs (1,715 × ~5 MB);
  (c) one front-to-back decode into quarter-resolution JPEGs under `out/_cache/stray/<id>/`.
- **Choice.** (c). Frames carry `video_index`, so later stages (damage crops) can fetch full resolution.
  K is scaled to the cached size; timestamps come from `odometry.csv`, never i/60.
- **Evidence.** Cold decode of c00a170fe1 (1,714 frames) takes 4.6 s, cached 0.1 s; total run 11 s.

## D8. Fusion settings chosen by measured wall thickness
- **Context.** Back-projected LiDAR walls should be thin; thickness (robust sigma of wall points) is a
  direct, ground-truth-free measure of fusion quality.
- **Options.** confidence ≥ 1 vs == 2; quality filter on/off; depth↔pose offset −2..+2 frames; 1 vs 2 cm voxel.
- **Choice.** confidence == 2, filter on, offset 0, 1 cm voxel, 4 m range.
- **Evidence.** `scripts/exp02_fusion.py data/stray/c00a170fe1`, 12 largest wall planes, median thickness:
  baseline (conf 2, filter, 2 cm) 2.78 cm; conf ≥ 1 2.81; no filter 2.73 (mean 2.85 vs 2.60, 4× slower);
  offsets −2/−1/+1/+2: 4.12/3.14/3.49/4.35 (0 is right); 1 cm voxel 2.17 cm (fuse 6.8 s vs 3.3 s).

## D9. Footprint includes air the sensor saw through
- **Context.** On c00a170fe1 the camera faced the fridge, sofa and wardrobe, not the floor, so the
  floor-only footprint lost most of the living area (R1 12.7 m²).
- **Options.** (a) Floor + horizontal surfaces only; (b) add the camera path; (c) add top-down
  camera→point rays (a 2D free-space carve); (d) TSDF free space.
- **Choice.** (c) and (b): a 2 cm cell is inside if rays from ≥ 3 frames crossed it (one mirror reflection
  can't punch through a wall), and a 0.3 m band around the camera path is always inside (the person stood
  there). Both exist for every posed tier, so neither is LiDAR-specific.
- **Evidence.** R1 12.7 → 15.1 (rays) → 19.5 m² (camera path); the verifier found the polygon cutting through
  floor the camera walked on before (b). `debug/fusion_topdown.png`, `debug/rooms_split.png`.

## D10. The main room is the one the camera spent most time in
- **Context.** A single-room capture also sees neighbouring rooms through doorways.
- **Choice.** Rank rooms by camera time at least 0.3 m inside each polygon (R1 = most); rooms with < 3 s are
  kept in the plan but warned as "partially observed". Time, not frame count: the filter drops still frames.
- **Evidence.** c00a170fe1: R1 32.1 s, R2 (glimpsed through the door) 1.1 s → warned. Without the 0.3 m
  inset R2 scored 3.0 s because the camera stood in the doorway. On 1a8384c3f6, 3 of 4 rooms are visited.

## D11. Whole-flat stitch: walls cut out of the footprint, doorways barred, watershed split
- **Context.** On 1a8384c3f6 the 02 pipeline found 3 rooms (R1 = 44 m², several rooms merged): the 18 cm
  footprint closing filled partition walls, and the corridor never got a watershed seed.
- **Options.** (a) Beta's watershed as-is; (b) learned room segmentation; (c) watershed on a footprint with the
  wall evidence put back.
- **Choice.** (c): tall wall evidence (vertical surface in ≥ 2 of 3 height bands) is cut out of the footprint;
  0.5–1.3 m gaps between runs of tall wall on one wall line are barred as doorways (= door connections); seed
  spacing 1.2 → 0.8 m. Overlap > 1 % of the smaller room goes to the room whose own footprint covers more of
  it (warned). Regions never entered with < 20 % floor seen are dropped (air through a window). R1..Rn by area;
  ≤ 1.6 m wide and ≥ 2.5 m long → "connector".
- **Evidence.** 1a8384c3f6: 3 → 9 rooms, 1 connector, 12 connections (drift on). c7d28f72c6: 9 rooms,
  1 connector, 11 connections. Known: part of the corridor merges into the open kitchen (no wall at wall-band
  height); c7d2 R6 is a region outside a window (0 s inside, warned partially observed).

## D12. Drift correction: pose graph over submaps, ICP loop closures, corrections blended in time
- **Context.** ARKit gravity is accurate, but heading and position drift over a multi-minute walk: on 1a83 the
  per-submap wall angle slides +3.0° → −0.8° over 115 s, and walls seen twice land 4–20 cm apart.
- **Options.** (a) poses as-is (fails the work order); (b) plane-anchored yaw: snap each submap's walls to the
  global Manhattan axes; (c) submaps (3 m / 8 s) + point-to-plane ICP loop closures (fitness ≥ 0.2, RMSE ≤ 1.5 cm,
  all directions constrained, plausible size, ≤ 1° tilt) + Open3D pose graph, yaw-only result; (d) (b) after (c).
  For (c), one rigid step per submap or a blend of the two bracketing submaps by time.
- **Choice.** (c) with the time blend (`fp/recon/drift.py`, default; `--no-drift-correction` turns it off).
- **Evidence** (`scripts/exp03_drift.py`; thickness cm / double walls / loop RMSE cm before→after):

  | capture | off | posegraph blended | posegraph step | plane |
  |---|---|---|---|---|
  | c00a170fe1 | 2.17 / 4 / 4.20 | 1.71 / 2 / 1.91 | 1.67 / 1 / 1.91 | 1.77 / 3 / 4.33 |
  | 1a8384c3f6 | 4.18 / 13 / 3.34 | **1.94 / 7 / 1.83** | 1.96 / 8 / 1.83 | 4.10 / 18 / 2.98 |
  | c7d28f72c6 | 3.62 / 32 / 3.22 | **3.27 / 15 / 2.12** | 3.26 / 19 / 2.12 | 3.05 / 36 / 3.48 |

  The plane method never lowers the loop residual and adds doubles on c7d2 (it fixes heading, not position);
  (d) was worse than (c) alone in an earlier run (1a83 thickness 2.34 vs 1.96). The blend beats the step on
  doubles on both flats. On c7d2 the pose graph pruned 5 of 22 ICP-accepted loops as inconsistent.
- **Cost.** Corrections are good to ~1–2 cm, which moves in-room spans by 2–6 cm (D13). Accepted, because the
  alternative (no correction) leaves doubled walls and rooms that don't match across scans.

## D13. Repeatability reported as measured, gate missed
- **Context.** Gate: per wall within max(1 cm, 0.5 %) between the two flat scans.
- **Choice.** 2-D registration (yaw 0/90/180/270 + ICP on wall points), rooms matched by centroid, wall-to-wall
  spans compared (a tape measure's view, independent of where segmentation ends a room). No threshold changed
  after seeing these numbers.
- **Evidence** (`eval/repeatability_lidar.md`). Same flat: 69 % inliers vs 24 % for the runner-up yaw. Spans
  0/16 within the gate, median |Δ| 5.5 cm. Drift off: 3/12 spans, median 1.9 cm, but 55 % inliers, 6/8 rooms
  matched and edges 40 cm vs 12 cm. Same-scan off vs on moves spans 2.3 cm (c7d2) / 5.5 cm (1a83) median.
  Drift correction buys global consistency at a few cm of local accuracy; a better local model (e.g. per-room
  refinement after correction) is future work.

<!-- Work order 05 ran in a parallel worktree (track/camera) while 03 added entries on main; its entries are
numbered D05.x so the merge doesn't clash. -->

## D05.1. MapAnything runs locally by default (MPS/CUDA/CPU); Modal is opt-in
- **Context.** Camera tiers have no depth and no poses, so a learned multi-view model must supply both.
  The brief requires the evaluator's laptop to run cold with no calls to our cloud.
- **Options.** (a) MapAnything Apache weights locally; (b) VGGT (non-commercial weights at the time) or
  DUSt3R/MASt3R + global alignment (pairwise, slower, CC-BY-NC); (c) COLMAP SfM + a monocular depth model
  (no metric scale without a prior, fails on textureless walls); (d) keep Modal as the only backend.
- **Choice.** (a), with `--backend modal` kept as an optional speed-up. Device order cuda > mps > cpu.
- **Evidence.** On this M5 (16 GB, MPS, bf16, 518×392): 4 views 12.6 s / 6.6 GB, 16 views 18.7 s / 8.75 GB,
  24 views 37.1 s / 9.83 GB MPS peak, model load 40–50 s warm. It works on MPS, so no fallback to VGGT was
  needed. Apache-2.0 weights; one model gives metric depth, intrinsics and poses in one shared frame.

## D05.2. 24 views per model pass; chunks merged by depth ratio, scale by median vote
- **Context.** MapAnything attends over all views at once; memory grows ~0.13 GB per view on MPS. Long
  videos run in overlapping chunks that must be put into one frame and one scale.
- **Options.** Chunk size 40 (first guess), 32, 24, 16. Merge: (a) Umeyama Sim(3) on shared frames' 3D
  points (first version); (b) scale = median depth ratio at the same pixels of shared frames, rotation and
  translation = average move between the shared cameras' poses. Final scale: chunk 1's, or the median of
  every chunk's own metric scale.
- **Choice.** `CHUNK = 24`, `OVERLAP = 6`; merge (b); final scale = median of the chunks' votes.
- **Evidence.** 24 views: 37 s and 9.8 GB MPS peak, no swap; 40 views (~13 GB from the slope) swapped.
  With (a) the c00a video chunk scales ran 0.78 → 0.62 → 0.58 → 0.58 → 0.22: a least-squares point fit
  shrinks the scale under noisy pairs and the shrink compounds along the chain (recon 2.5× too small,
  floor lost). With (b): 0.89–1.15, floor found 1.25 m below the camera (LiDAR 1.42–1.48 m). Unit test:
  on clean synthetic data (a) shrinks ~4 % per link, (b) stays within 2 %.

## D05.3. Photos: one joint reconstruction of all rooms; unlinked rooms are not placed
- **Context.** The plan must be one stitched property, not per-room islands. Rooms photographed separately
  share no frame unless the model sees them together.
- **Options.** (a) Reconstruct each room alone, then place rooms by matching doorway photos; (b) one joint
  pass over every room's photos, so the model itself puts all rooms in one frame; (c) place rooms by hand.
- **Choice.** (b), with checks: rooms are linked by a photo saved in both folders (identical bytes) or by
  ≥ 40 SIFT + fundamental-matrix RANSAC inliers between photos of two rooms. A room linked only by matches is
  checked in 3D and re-placed by a Sim(3) on the matches if they disagree by > 0.25 m. A room with no link is
  left out of the plan and named in a warning ("not placed: re-capture with a doorway photo in both folders").
- **Evidence.** A joint pass is how the model was trained (multi-view, one frame); per-room passes would each
  have their own scale too. Never overlapping a room silently is a brief requirement.

## D05.4. Gravity on camera tiers: image up, overruled only by twice the support
- **Context.** Without an IMU the up direction must come from the images.
- **Options.** (a) the cameras' mean "up" only; (b) try the in-image axes, refine each onto the normals of
  horizontal surfaces, keep the most supported (first version, made for sideways Stray frames);
  (c) image up refined the same way, replaced by another axis only with 2× its support.
- **Choice.** (c), `OVERRULE_UP = 2.0`. The sign puts the bigger horizontal layer 0.8–2.0 m from the
  cameras below them (the floor). Checked against an SVD plane fitted to the floor (0.15–1.55° on all runs).
- **Evidence.** Phone images are upright: the camera app orients them by the accelerometer, and the ingest
  applies EXIF and video rotation tags. Under (b) a long wall in the c7d2 photos (support 0.206) beat the
  floor (0.197) and the plan became a side elevation; with (c) the ceiling is found at 2.47–2.78 m. A truly
  sideways input still wins in the unit test (floor + ceiling ≈ 2.4× the wall).

## D05.5. Scale priors widen intervals; they never rescale
- **Context.** Camera-tier metric scale comes only from the model. A wrong scale moves every number.
- **Options.** (a) trust the model; (b) rescale to a prior (camera height, door height, ceiling height);
  (c) keep the model's values and widen intervals by the relative gap to the prior range.
- **Choice.** (c): priors are handheld camera 1.0–1.8 m above the floor and ceiling 2.3–3.2 m. The largest
  relative gap becomes `extra_rel` and contract.fill_from_geometry adds it to every interval's scale term.
- **Evidence.** A prior is a range, not a measurement; rescaling to it would turn a plausible-but-unusual
  room (a 3.5 m ceiling) into a confidently wrong plan. Widening keeps the value honest and says "less sure".

## D05.6. Deterministic recon cache keyed by input bytes; live path always available
- **Context.** A cold laptop run of the camera tiers takes minutes and needs ~5 GB of weights; the second run
  of the same input must give a byte-identical plan.json (except timings).
- **Choice.** One entry per model pass. Key = sha1(model id + model parameters + each frame's content hash +
  K priors); value = an .npz of the model's raw output (depth, conf, mask, K, T_wc, model-input RGB, metric
  scale) under `out/_cache/recon/`. Everything after the model (merge, fusion, geometry) always runs.
  `--no-cache` forces the live run. `scripts/cache_sync.py push|pull` publishes it as a Hugging Face dataset
  with a sha256 manifest checked after download.
- **Evidence.** A cache entry can only replay the exact bytes it was computed from, so it cannot stand in for
  an unseen capture (the walk-in capture always runs live). Live run vs replay: plan.json byte-identical
  outside `timings` (c00a and c7d2 photos; c00a video replayed twice). c00a video: 338 s live, 10.8 s replayed.

## D05.7. Derived camera inputs carry what a real iPhone file carries
- **Context.** The camera tiers are tested on video and stills cut from the Stray LiDAR captures. Stray's
  rgb.mp4 is stored in the sensor's landscape frame with no rotation tag, and stills from it have no EXIF.
- **Options.** (a) feed them as they are; (b) add exactly the metadata a phone writes: the video's display
  rotation (fixed once per video from pose gravity, the most common upright rotation) and each still's
  EXIF `FocalLengthIn35mmFilm` (integer, from Stray's per-frame fx).
- **Choice.** (b), in `scripts/make_camera_tiers.py`. Pixels untouched (ffmpeg `-c copy`).
- **Evidence.** With (a) the model saw portrait frames sideways and its per-chunk depth was 0.6–2.1× the
  LiDAR depth. The rule uses the capture's poses, never the evaluation numbers. The focal prior differs
  from the true fx by 0.7 % (integer rounding), as on a phone.

## D05.8. Photo protocol: overlapping sweeps, not "shoot across from each corner"
- **Context.** The first derived photos were chosen to be as different as possible (farthest-point over
  position and heading). The model registers photos through what they share.
- **Options.** (a) farthest-point; (b) sweeps: from a standing spot, a photo every 25° (upright stills see
  ~48° across, so neighbours share about half) within 1 m of the spot, a second sweep from the farthest
  spot, doorway photos in both rooms' folders; (c) a video instead of photos.
- **Choice.** (b) for the derived photos and for the capture protocol (08); (c) remains the stronger tier.
- **Evidence.** Median error of the relative rotation between photo pairs vs LiDAR, c7d2: (a) 35° → (b) 10°.
  Camera positions are still not registered across sweeps and rooms (RMS 3.0 m on a 3.1 m spread): stills
  from a walking video give sweeps of only 3–5 photos. A real sweep, standing still and turning, should
  give longer chains; this has not been tested on a real phone yet.

## D05.9. The photo-tier footprint fix waits for the 03 merge
- **Context.** The room footprint (fp/geometry/plan.py) = horizontal surfaces ∪ cells crossed by rays from
  ≥ 3 frames (`MIN_RAY_HITS`) ∪ a band around the camera path. With 7–17 photos rays from ≥ 3 frames almost
  never overlap (c00a: 0.8 m² of rays from ≥ 3 photos, 9.0 m² from ≥ 1), there is no camera path, and room
  folders are not used to split rooms. Work order 05 must not edit fp/geometry/* while 03 does.
- **Options.** (a) edit plan.py now (merge conflict with 03); (b) a new photo-room module + a hook in
  cli._geometry; (c) defer, record the fix.
- **Choice.** (c), chosen by the user. Proposed fix after the merge: `MIN_RAY_HITS` by tier (1–2 for photos),
  and a per-room footprint from each folder's photos so the folder labels define the rooms.
- **Evidence.** Photo-tier rooms are undersized: c00a R1 5.4 m² vs 19.5 m² LiDAR reference (eval/cross_tier_sample.md).

## D04.1. Openings: ray casting on wall elevations, not holes in the point cloud
- **Context.** A gap in a wall's points is either an opening or a part of the wall nobody looked at (behind a
  wardrobe, above the camera's view). 03's doorways were floor-footprint necks: a width with ±10 cm and no height.
- **Options.** (a) gaps in the wall's point band; (b) floor necks (03); (c) an RGB door/window detector;
  (d) per wall, an elevation grid (along-wall × height) where every depth ray votes: ended on the plane
  (wall), crossed it and ended > 5 cm behind (seen through), or never reached it (unobserved).
- **Choice.** (d) (`fp/geometry/openings.py`): 2 cm cells, every 4th depth pixel, frames counted not rays,
  a cell is empty when ≥ 60 % of the ≥ 2 frames that reached it saw through. Door = empty region from the floor,
  0.6–1.6 m wide, lintel 1.9–2.4 m; window = sill 0.3–1.5 m; passage = floor gap > 1.6 m or no lintel below the
  ceiling, and another room behind it. 03's necks stay only between rooms no measured opening links (warned).
- **Evidence.** Synthetic scene: a 0.90 m door measured within 2 cm; a wall behind a wardrobe stays unobserved,
  not empty (tests/test_openings.py). c7d28f72c6: 12 openings, e.g. R1.O1 door 0.904 m with lintel 2.08 m;
  44 s for 55 walls on 2,629 frames. 2 cm cells, not the work order's 1 cm: one frame's rays are ~4 cm apart at
  2 m, so 1 cm cells would hold no votes; the width comes from points, not cells.

## D04.2. Width between jamb planes; no RGB refinement
- **Context.** The gate is ±2 cm on width; one LiDAR depth pixel is ~1 cm at 2 m.
- **Options.** (a) the empty region's width (cell-rounded, blurred by ray density); (b) each jamb = median
  along-wall position of reveal points (vertical surfaces inside the wall thickness facing into the opening,
  2–35 cm behind the face, so casings standing proud of the wall are left out); fallback: the 95th
  percentile of the wall-face points next to the gap; (c) (b) plus vertical RGB edges at 1920×1440.
- **Choice.** (b). Interval half = Σ per jamb (1 cm × tier scale + jamb σ, + 2 cm for a wall-end fallback);
  a jamb with neither ("region edge") makes width `observed: false`. (c) not built: the work order allows it
  only "if it measurably helps", and the sample data has no tape widths to show that it does.
- **Evidence.** c7d28f72c6 doors 0.713–0.915 m, reveal σ 1–5 cm per jamb, half-widths 5–12 cm (provisional,
  07 calibrates).

## D04.3. Mirror and recess tests; a door-shaped recess is a closed door
- **Context.** A mirror makes the sensor "see" a room behind the wall; a niche or recessed panel is seen
  > 5 cm behind the plane. Both look like openings to (D04.1). RGB sheets on c7d28f72c6 showed a bathroom mirror
  (R7.W6, "window" 1.64 m) and a closed door at the corridor end seen only above 1.35 m (R2.W4, "window" 0.90 m;
  first read as a recessed panel, corrected by the verifier: handle and casing visible at t = 65838.5).
- **Options.** (a) accept them; (b) RGB classifier; (c) geometry: mirror = reflect the points behind the gap
  back across the plane, and if ≥ 50 % land within 3 cm of the room's own surfaces it is the room again;
  recess = a room-facing surface < 40 cm behind the plane over ≥ 50 % of the gap.
- **Choice.** (c). A recess with a door's shape is kept as a **closed door** (its leaf sits back in the frame):
  R2.W5 on c7d28f72c6 was rejected as a recess until the RGB sheet showed the handle.
- **Evidence.** c7d28f72c6: the mirror scores only 0.48 (real doors 0.09–0.37), so it is caught by the recess
  test (0.92), not the mirror test; c00a170fe1 R3.W4 mirror 0.98; R2.W4 recess 1.00 (a closed door, so a miss). Mirror threshold
  not tuned to these numbers (0.5 = "most of it"). Known limit: a leaf < 5 cm behind the face is "wall" (HIT_TOL).

## D04.4. A door whose top was never seen
- **Context.** On c00a170fe1 the camera pitched −46°…−8°: rays never cross a door plane above ~1.4 m, so no
  lintel is seen and the door rule (head 1.9–2.4 m) found nothing.
- **Choice.** A floor gap 0.6–1.6 m wide with *unobserved* (not wall) cells above is a door if it was seen
  through up to ≥ 1.2 m (above counters and sofa backs); its height is `observed: false`.
- **Evidence.** c00a170fe1: 0 → 2 doors (0.92 m R2–R3).

## D04.5. Ceiling per room: largest down-facing layer 2.1–3.6 m above the room's own floor
- **Context.** One flat-wide ceiling (2.44 m on c7d2) hid rooms at 3.09 m.
- **Options.** lower bound 1.8 m (work order) vs 2.1 m (align.CEILING_RANGE); count points vs patches.
- **Choice.** (`fp/geometry/ceiling.py`) inside the polygon, layers ≥ 1 m² 2.1–3.6 m above the room's floor
  (up-facing points within ±5 cm of z = 0, fallback the plan floor +3 cm); value = medians' difference; half =
  1 cm + 1.645 √(σc²/patches_c + σf²/patches_f), patches = 25 cm squares the layer covers (errors are
  correlated within a patch, so points are not independent samples). Other layers → `ceiling` warnings.
- **Evidence.** 1.8 m admits kitchen cabinet undersides (round 1 picked 1.86 m). c7d28f72c6: 2.33–2.49 m in
  6 rooms, 3.04 / 3.09 m in R3 / R4, R6 not observed; 1a8384c3f6 and c00a170fe1: every room not observed.

## D04.6. Fixes from verification; the openings gate is not met
- **Context.** The verifier checked every c7d28f72c6 opening in 5 RGB frames: 6/12 correct, 3 phantoms (a
  fridge front read as a 2.43 m window on a 1.87 m wall, the vanity mirror, a toilet niche read as a closed
  door), 3 real but wrong kind, 4 misses (proxy correct / (detected + misses) = 38 % vs the 85 % gate).
- **Options.** Lower the mirror threshold to 0.3 (it would catch the vanity mirror at 0.30, but real doors
  score up to 0.38: tuning on the evaluation capture, rejected); add rules that follow from what an opening is.
- **Choice.** (1) A door-shaped recess is a closed door only if a lintel was seen above it (a leaf in a frame
  has wall above; a niche need not). (2) An opening wider than its own wall + 10 cm is not a hole in that
  wall: a window there is rejected; a door-wide gap without a lintel becomes a passage if a room is behind it.
  (3) The mirror test's random sample is fixed per candidate. (4) The RGB sheets skip boxes hidden behind a
  nearer surface in that frame.
- **Evidence.** c7d28f72c6 after: 10 openings, 7 correct, 1 phantom (vanity mirror, 0.30), 2 wrong kind
  (a glazed balcony slider and the office door called passages), 4 misses (a tall narrow window with a
  0.15 m sill that no rule covers, two curtained windows, the corridor-end closed door) = **50 %**, still
  below 85 %. These rules were found on the same capture they are scored on, so 50 % is in-sample (optimistic).
  The adversarial case (a wall line moved 30 cm into c00a's room) no longer produces a closed door.

## D06.1. Local damage detector: Grounding DINO tiny + SAM 2.1 small, not SAM 3
- **Context.** Damage must run on the laptop (no calls to our cloud). Beta used SAM 3 on Modal.
- **Options.** (a) SAM 3 locally (text -> masks, one model); (b) Grounding DINO (text -> boxes) + SAM 2.1
  (box -> mask); (c) OWLv2 + SAM 2.
- **Choice.** (b), `fp/damage/detect_local.py`: one text pass for all classes ("crack. water stain. mold. peeling
  paint. hole in wall."), the published default thresholds (box 0.35, text 0.25, not tuned), a matched phrase
  must name exactly one class, masks over 25 % of the image are the wall itself and dropped. Same per-image
  deterministic cache as 05. SAM 3 stays as `--backend modal`.
- **Evidence.** SAM 3's weights are gated: 401 on this Mac, and an evaluator's cold run would need HF approval
  first. (b) is ungated Apache-2.0. MPS: 0.65 s detection + 0.3 s masks per full-res frame, ~30 s model load;
  c00a damage stage 16 s warm, c7d2 43 s.

## D06.2. Metric extent = mask coverage x pixel footprint on the surface plane
- **Context.** Beta's extent was a bounding square, an upper bound marked not observed.
- **Choice.** The mask is area-resized onto the depth image (fractional coverage, so a 2 px crack counts in a
  7.5 px LiDAR depth pixel); each covered pixel with depth adds coverage × z²/(fx·fy·|cos a|), a = angle between
  its ray and the surface normal (capped at 5×). Only points within 8 cm of the surface plane count (mask spill
  onto furniture doesn't), and ≥ 50 % of the mask must lie on it, else the detection is on furniture and dropped.
  Interval: the wider of 20 % (+ 2 × the tier's scale term: area goes with scale²) and half the views' spread.
- **Evidence.** Synthetic rendered frame: 0.20 × 0.20 m stain = 0.040 m² measured within 15 %, face-on and
  34° off; a mask on a cupboard front 0.6 m from the wall is dropped (tests/test_damage.py).

## D06.3. A detection must be confirmed from a second viewpoint
- **Context.** On c00a170fe1, 8 detections gave 4 items; by eye all 4 were false: shadows, a reflection, tile
  edges. Raising thresholds would be tuning on the sample data.
- **Choice.** Look again: run the detector on up to 2 other frames that see the same spot unoccluded, from a
  camera ≥ 25 cm away; keep the detection only if the same class lands on the same surface within 30 cm.
  Unconfirmed detections are not damage; they are listed in a warning with their crop (damage/unconfirmed_k.jpg).
- **Evidence.** A stain stays put when you move; a reflection moves and a lighting artefact disappears.
  c00a 4 -> 1 item, c7d2 5 -> 1, 1a83 6 -> 3. Static shadows survive it (they also stay put): the remaining
  items on the sample data are all false (listed in STATUS 06).

## D06.4. Concealed damage: transparent rules, labelled hypotheses
- **Context.** No dataset of hidden damage behind visible symptoms exists to learn from.
- **Choice.** Five rules (`fp/damage/rules.py`), each flag carrying its rule id, rule text and evidence, label
  "hypothesis": C1 wet damage on a ceiling; C2 wet damage / peeling paint below 0.5 m on a wall; C3 wet damage on
  a wall shared with a wet room; C4 crack within 0.6 m of a door; C5 (new) wet damage on a ceiling or the top
  0.3 m of a wall in or next to a wet room (pipes in the ceiling void). Wet rooms come from `--wet-rooms`.
- **Evidence.** Each rule is one sentence a surveyor would say; tests check each fires and none fire without
  their trigger.

## D06.5. Scope by interval arithmetic
- **Choice.** `fp/scope.py`: repaint wall = length × ceiling height − openings (assumed 2.4 m [2.1, 3.2] when the
  ceiling wasn't observed, then not observed); patch and fill = count of cracks/holes; mould treatment /
  stain-block = sum of extents; repaint ceiling and replace floor finish = floor area; skirting = perimeter −
  door and passage widths. Each [lo, hi] is pushed through the formula worst-case (lo uses length lo × height lo
  − openings hi).
- **Evidence.** Test: wall 3.0 × 2.5 − 0.9 × 2.05 door = 5.655 m² with lo < value < hi; every surface_id validates.
