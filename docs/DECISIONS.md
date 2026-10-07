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
