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
