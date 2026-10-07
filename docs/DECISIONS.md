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
