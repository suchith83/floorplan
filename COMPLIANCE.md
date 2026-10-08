# Compliance matrix

Every requirement in [the brief](docs/brief/applied_ai_brief.md) (Parts 1–5, the gates, Deliverables 1–8, the walk-in
test, Constraints) → where it lives → what the artifact is → status. **Done** = built and evidenced. **Partial** = built
but the gate or the evidence falls short (the reason is given). **Not done** = missing (the reason is given).

`req #` numbers the brief's requirements one atomic clause at a time (157 in all). A subagent that had read only the
brief produced that list independently, and `scripts/check_compliance.py` confirms every number appears below and every
path exists. Benchmark numbers are from the fix-after run (`make benchmark`, [eval/BENCHMARK.md](eval/BENCHMARK.md)).

**Three facts behind most Partial / Not done rows.**
(1) The author has no iPhone. The recruiter said to run the LiDAR tier on her sample data (`data/stray/`, 3 Stray
Scanner captures) and capture photo and video on the author's own phone.
(2) Those own captures (`data/own/`: photos, video, tape, app export, staged damage) **were not taken** (no time before the
deadline). Every row that needs them is Not done or Partial. The harness is ready for them: `make benchmark` picks up `data/own/`.
(3) There is no tape on the sample data. LiDAR is scored by repeatability, camera tiers against the LiDAR plan of the same
capture (a reference, not truth), and openings by eye.

## Part 1: capture route and input tiers

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 1, 2, 5 | Own the problem from the sensors; choose a route; name the tool | `CAPTURE_PROTOCOL.md`, `docs/DECISIONS.md` (D1) | Route 2: Stray Scanner (LiDAR), the native Camera app (video, photos) | Done |
| 3, 4 | Route 1 (own iOS app, TestFlight, < 10 min install) | `docs/DECISIONS.md` (D1) | Not taken; Route 2 chosen | Done (n/a: Route 2) |
| 6, 7, 8, 9, 10, 11, 12 | One-page protocol for a non-engineer: install, walk, length, avoid, hand-off | `CAPTURE_PROTOCOL.md`, `CAPTURE_PROTOCOL.pdf`, `docs/img/walk_path.svg` | Prints on one A4 page; imperatives with numbers; walk-path figure | Done |
| 13 | Protocol unambiguous when followed literally | `CAPTURE_PROTOCOL.md`, `docs/STATUS.md` (08), `docs/DECISIONS.md` (D08.2) | Two rounds by a literal-reader subagent (3 role-plays: LiDAR, video, photos in a flat with open plan); its 7 + 2 blocking points were fixed, one by a code change (EXIF photo identity) | Done (not yet tried by a non-engineer in person) |
| 14, 15 | Three tiers, same output contract | `fp/ingest/__init__.py`, `fp/schema.py`, `schema/plan.schema.json` | `load_capture` auto-detects Stray folder / video / photo folders; one schema for every tier | Done |
| 16, 22 | Intervals widen as data thins | `fp/contract.py`, `fp/calibration.json`, `eval/CALIBRATION.md` | Tier scale 1/3/6 + fitted per-tier constants (video wall b = 4.09, photos 3.12, LiDAR k = 1.42) | Done |
| 17, 19, 20 | Photos: 2–8 stills per room, no depth or poses, one folder per room | `fp/ingest/photos.py`, `fp/recon/camera.py`, `CAPTURE_PROTOCOL.md` | Room folders required; MapAnything predicts depth and poses; protocol: 6 per room + doorway photos (a many-door hallway can pass 8; the code has no cap) | Done |
| 18, 25 | Photo and video tiers on any iPhone 15+ | `docs/DEVICE_MATRIX.md`, `tests/test_camera_ingest.py` | HEIC, HEVC, 10-bit HDR, rotation handled (tested on synthetic files) | Partial: no real iPhone file tested (inputs derived from the sample captures) |
| 21, 51 | Photo folders give the same stitched whole-property plan | `fp/cli.py`, `eval/BENCHMARK.md` | Runs end to end and produces a multi-room plan | Partial: c7d2 photos 3 rooms vs 9, footprint −79 % |
| 23 | Any picture in, results out | `fp/contract.py` (`StageNotBuilt`/`StageError`), `tests/test_cli.py` | A failed stage gives a warning and `observed: false`; a valid plan is always written | Done |
| 24 | Video: handheld walkthrough clip | `fp/ingest/video.py` | 3 fps keyframes, ≤ 240, HDR tonemap, autorotate | Done |
| 26 | LiDAR: depth, poses, intrinsics on Pro devices | `fp/ingest/stray.py`, `fp/recon/lidar_fuse.py` | Stray Scanner reader; confidence-2 depth fusion | Done (sample data only) |
| 27, 28, 29 | Device matrix: tier × hardware × honest accuracy | `docs/DEVICE_MATRIX.md` | Each row says what was actually tested and links its accuracy to `eval/` | Done (LiDAR validated only on the sample data, as stated) |

## Part 2: output contract

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 30, 31, 33, 50 | Dimensioned per-room plan: walls, floor area | `fp/geometry/plan.py`, `fp/contract.py`, `fixloop/after/c7d28f72c6/plan.json` | Walls with length intervals, room area intervals | Done |
| 32 | Ceiling height per room | `fp/geometry/ceiling.py`, `tests/test_ceiling.py` | c7d2: 8 of 9 rooms 2.33–3.09 m; scans with no ceiling in view say "not observed" | Done (not tape-checked) |
| 34 | Openings per room | `fp/geometry/openings.py`, `tests/test_openings.py` | Doors, windows, passages with widths | Partial: detection 50 % by eye (gate row 65) |
| 35, 47, 48, 49 | Stitched multi-room plan, every room placed, connected | `fp/geometry/rooms.py`, `fixloop/after/c7d28f72c6/plan.svg` | LiDAR c7d2: 9 rooms, 11 connections, overlaps repaired | Done at LiDAR; Partial at camera tiers (rooms merge) |
| 36 | Correct adjacency | `fixloop/after/c7d28f72c6/plan.json` (`connections`) | Connections from openings and doorway necks | Partial: correct by eye on LiDAR; wrong on c7d2 photos |
| 37, 38, 39 | Damage regions per surface, with class and metric extent | `fp/damage/detect_local.py`, `fp/damage/project.py`, `tests/test_damage.py` | Grounding DINO + SAM 2.1, mask projected on the surface plane; painted stain +11 % extent | Partial: no real damage on the sample data; a drawn crack is missed |
| 40, 41 | Concealed-damage flags with the rule that fired | `fp/damage/rules.py` | Rules C1–C5; each flag carries rule_id, rule_text, evidence | Done (rules tested; no real damage to fire them yet) |
| 42 | Scope line items keyed to surfaces | `fp/scope.py` | Repaint, patch, mould treatment, floor, skirting; interval arithmetic | Done |
| 43 | Interval on every measurement | `fp/schema.py` (`Measure`), `docs/SCHEMA.md` | `{value, lo, hi, unit, method, observed}`; schema rejects a bare number | Done |
| 44 | One command per capture | `fp/cli.py`, `README.md` | `uv run fp run <capture>` | Done |
| 45 | JSON to the published schema | `schema/plan.schema.json`, `docs/SCHEMA.md` | JSON Schema generated from `fp/schema.py`; a test fails if it's stale | Done |
| 46 | Rendered plan | `fp/report/svg.py`, `fp/report/html.py` | `plan.svg` + one-file `report.html` | Done |

## Part 2: benchmark set

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 52 | Build the benchmark set yourself | `eval/run_benchmark.py`, `eval/ground_truth/TEMPLATE.yaml` | Harness ready; own set not taken | Partial: sample data only (fact 2) |
| 53, 54 | Multi-room capture, 3+ rooms + connector | `eval/BENCHMARK.md` | c7d28f72c6 / 1a8384c3f6: whole flat, corridor + rooms (evaluator's data, not ours) | Partial: own multi-room capture not made |
| 55, 56 | Furnished room, staged damage, 2 classes | `scripts/exp06_damage.py`, `docs/STATUS.md` (06) | Painted test on real frames (stain found, crack missed) | Not done: own staged capture not taken |
| 57, 58, 59 | Same rooms at all three tiers, multi-room included, photos as room folders | `scripts/make_camera_tiers.py`, `eval/BENCHMARK.md` | Video and photo-folder inputs cut from the same Stray captures | Partial: derived from the sample data, not captured on a phone |
| 60 | One room captured twice, same tier | `eval/repeatability_lidar.md` | LiDAR: the two whole-flat scans | Partial: camera-tier repeats not done (own captures not taken) |
| 61, 63 | Laser or tape ground truth on everything, submitted | `eval/ground_truth/TEMPLATE.yaml`, `eval/match_gt.py` | Format, matcher and scorer ready | Not done: no tape on the sample data; own tape not taken |
| 62 | Raw sensor data submitted | `scripts/fetch_data.sh` | Unpacks the evaluator's sample zips from `data/zips/`; tries a Hugging Face dataset first and falls back to the local zips | Not done: dataset not published, time. The evaluator's own sample zips are the raw data (README: put them in `data/zips/`) |

## Part 2: gates

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 64 | Round 1 gates apply | `eval/BENCHMARK.md` | Wall, ceiling, opening, repeatability, calibration rows per tier | Partial: see rows below |
| 65, 66, 67 | Openings ≤ 2 cm on ≥ 85 %, misses and phantoms count | `eval/BENCHMARK.md`, `eval/ground_truth/c7d28f72c6.openings_by_eye.yaml` | 7/14 = 50 % by eye (1 phantom, 4 missed, 2 wrong kind) | Partial: FAIL; widths not taped |
| 68 | Ceiling ≤ 1.5 cm per room | `eval/BENCHMARK.md` | 8/9 rooms observed on c7d2 | Not done: NOT SCORED, no tape |
| 69, 70 | Ceiling spread ≤ 1 cm across captures; say biased vs unrepeatable | `eval/BENCHMARK.md` | 1a83 never saw the ceiling, so no pair exists | Not done: NOT SCORED; needs own repeat |
| 71, 72 | Repeatability within 1 cm or 0.5 % per wall | `eval/repeatability_lidar.md`, `scripts/repeatability.py` | 0/16 spans within; median 5.5 cm; diagnosed as unrepeatable, not biased | Partial: FAIL, measured and explained |
| 73, 74, 75, 76 | Drift: state the method, ablation on and off, not poses-as-is | `fp/recon/drift.py`, `docs/REPORT.md`, `docs/img/drift_ablation_c7d2.png` | Submap pose graph + ICP loop closures; `--no-drift-correction` ablation | Done (PASS on 1a83 and c7d2) |
| 77, 79, 82 | Photo folders → one stitched plan, no overlaps, multi-room | `fp/cli.py`, `eval/BENCHMARK.md` | One plan from 3 room folders; overlaps 0 m² | Partial: one plan with no overlaps, but rooms undersized |
| 78, 80, 81 | Photo stitch: correct adjacency, footprint ±8 %, calibrated | `eval/BENCHMARK.md` | c7d2 photos −79 %, adjacency wrong | Partial: FAIL |
| 83, 84 | Photo walls ±8 %, calibrated | `eval/BENCHMARK.md`, `eval/CALIBRATION.md` | p90 24 % / 62 %; held-out coverage 67 % | Partial: FAIL |
| 85 | Video walls ±3 % | `eval/BENCHMARK.md`, `fixloop/RESULT.md` | p90 71 % / 76 % (the fix-loop gate, was 105 %) | Partial: FAIL |
| 86, 87 | Calibration scored at every tier; no confident garbage | `eval/CALIBRATION.md`, `fp/calibration.py` | Split conformal per tier; held-out 81 % LiDAR / 92 % video / 67 % photos; camera intervals ±300–400 % ("we don't know") | Partial: only video passes, and only by widening |

## Part 3: head-to-head

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 88, 89, 90, 94, 95 | LiDAR tier vs a consumer app on 2 rooms; one table; beat or tie ≥ 70 % | `eval/HEAD_TO_HEAD.md` | Protocol and empty table; substitute (photo/video vs magicplan) ready | Not done: no LiDAR iPhone (recruiter-approved); substitute not run (own captures not taken) |
| 91, 92, 93 | Name the app and version, submit its export | `eval/HEAD_TO_HEAD.md` | magicplan free tier; export goes to `data/own/app_export/` | Not done: own captures not taken |
| 96 | Cost is no excuse | `eval/HEAD_TO_HEAD.md` | Free tier chosen | Done |

## Part 4: fix loop

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 97–103 | One-page declaration: worst gate + number, root cause + evidence, fix, predicted number | `fixloop/FIX_DECLARATION.md` | Committed before the fix (tag `fix-before`): video walls p90 105 %, predicted 76 % | Done |
| 104, 113 | Ship the fix | `fp/geometry/plan.py` (`seen_through_walls`), `tests/test_stitch.py` | Seen-through wall copies ignored on camera tiers | Done |
| 105, 106, 107, 108, 114 | Before and after runs, regenerable | `Makefile` (`fix-before`, `fix-after`), `scripts/fixloop.sh`, `fixloop/before/BENCHMARK.md`, `fixloop/after/BENCHMARK.md` | Each side reruns the whole benchmark at its git tag in a temporary worktree | Done |
| 109 | Readable diff | `fixloop/RESULT.md` | `git diff fix-before fix-after` (5 files) and the key hunks explained | Done |
| 110, 111, 112 | Gate passes, or the report says why not; honest post-mortem | `fixloop/RESULT.md`, `docs/REPORT.md` | 105 → 76 % exactly as predicted; still FAIL; camera-tier poses metres off (Sim(3) scale 0.48) | Partial: meaningful movement, gate not passed |

## Part 5: process evidence

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 115, 116, 117 | Commit as you work; history belongs to the builder | `docs/STATUS.md`, `docs/DECISIONS.md` | ~75 small commits over 6–8 Oct, one per working piece, with key numbers in the messages | Done |
| 118 | Every decision defensible live | `docs/DECISIONS.md`, `docs/defense/00-setup.md`, `docs/defense/09-fix-loop.md` | One entry per decision (context → options → choice → evidence); Q&A per work order | Done |

## Deliverables

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 119 | Compliance matrix | `COMPLIANCE.md`, `scripts/check_compliance.py` | This file; the script checks coverage and paths | Done |
| 120, 121 | Capture route + device matrix | `CAPTURE_PROTOCOL.md`, `docs/DEVICE_MATRIX.md` | Route 2 protocol, device matrix | Done |
| 122, 125 | Repo, one command per capture | `README.md`, `fp/cli.py` | `uv run fp run <capture>` | Done |
| 123, 124 | README to running in < 15 min on a clean machine | `README.md`, `scripts/fetch_data.sh`, `scripts/fetch_weights.sh` | `uv sync` → zips into `data/zips/` → fetch → `fp run` | Done: fresh clone to the first plan in 2 min 10 s, warm uv/HF caches ([docs/COLD_RUN.md](docs/COLD_RUN.md)); a truly cold machine adds the uv and weight downloads |
| 126, 127, 128 | Reproduction bundle: every number from raw inputs; cache deterministic; live path runs | `Makefile` (`benchmark`), `eval/run_benchmark.py`, `fp/recon/camera.py`, `scripts/cache_sync.py` | `make benchmark` regenerates every table; MapAnything cache keyed by input bytes; live replay byte-identical outside timings | Partial: raw data and cache not published (not done: dataset not published, time); without the cache every camera-tier run is live |
| 129 | Benchmark: gates at all three tiers | `eval/BENCHMARK.md` | 27 gate rows, PASS 3 / FAIL 13 / NOT SCORED 10 / NOT DONE 1 | Done (as measured) |
| 130 | Repeatability table | `eval/BENCHMARK.md`, `eval/repeatability_lidar.md` | LiDAR pair table; own repeats not taken | Partial |
| 131 | Head-to-head table | `eval/HEAD_TO_HEAD.md` | Empty table and protocol | Not done (see Part 3) |
| 132 | Timing | `eval/BENCHMARK.md` | Per-stage timings per capture | Done |
| 133 | Fix loop bundle | `fixloop/FIX_DECLARATION.md`, `fixloop/RESULT.md`, `fixloop/before/BENCHMARK.md`, `fixloop/after/BENCHMARK.md` | Declaration, result, both runs, diff | Done |
| 134–141 | Technical report ≤ 6 pages: architecture, tiers + device matrix, drift, error budget, calibration, fix loop, failure modes | `docs/REPORT.md`, `docs/REPORT.pdf`, `docs/img/architecture.svg` | Every section, PDF export ≤ 6 pages (`scripts/export_report.py`) | Done |
| 142 | Raw data: sensor logs | `scripts/fetch_data.sh` | Stray Scanner sample captures (evaluator's zips); derived camera inputs rebuilt by `scripts/make_camera_tiers.py` | Not done: dataset not published, time; own captures not taken |
| 143 | Raw data: ground truth | `eval/ground_truth/c7d28f72c6.openings_by_eye.yaml` | By-eye opening labels | Not done: no tape exists yet |
| 144 | Raw data: app exports | `eval/HEAD_TO_HEAD.md` | Slot: `data/own/app_export/` | Not done: own captures not taken |

## Walk-in test

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 145, 147 | Unseen space, their iPhone, cold run in front of them | `README.md`, `fp/cli.py`, `docs/WALKIN.md`, `docs/COLD_RUN.md` | Live path on a laptop (MPS/CPU); no cloud calls; weights pre-fetched; walk-in script; unseen-input matrix (truncated scan, 2 photos/room, junk files, path with spaces: 4/4 give a plan after one crash fix) | Partial: rehearsed on sample-derived inputs only; video with fast pan / dark / mirror not run (time) |
| 146 | Route followed exactly | `CAPTURE_PROTOCOL.md` | Literal-reader checked | Done |
| 148 | Accurate against their laser | `docs/DEVICE_MATRIX.md`, `eval/BENCHMARK.md` | Honest accuracy per tier | Partial: camera tiers far off (median wall error 37–39 %) |
| 149 | All three tiers ready on the day | `fp/ingest/__init__.py` | All three run end to end; c7d2 video ~8.5 min live | Done (runs); accuracy as above |

## Constraints

| req # | requirement | path | artifact | status |
|---|---|---|---|---|
| 150 | Handheld consumer capture only | `CAPTURE_PROTOCOL.md` | Phone apps only | Done |
| 151 | Pretrained models disclosed | `docs/REPORT.md`, `scripts/fetch_weights.sh`, `docs/DECISIONS.md` (D05.1, D06.1) | MapAnything (Apache-2.0), Grounding DINO tiny, SAM 2.1 small; SAM 3 optional on Modal | Done |
| 152 | Runs without calling our infrastructure | `fp/recon/camera.py`, `fp/damage/detect_local.py` | Local by default; Modal only with `--backend modal` | Done |
| 153 | Weights and large binaries fetched by script | `scripts/fetch_weights.sh`, `scripts/fetch_data.sh` | Nothing large in git (largest tracked file < 5 MB) | Done (the data script falls back to local `data/zips/`) |
| 154 | Mirrors | `fp/geometry/openings.py` (mirror test), `docs/REPORT.md` | Reflected points landing on the room are rejected; one vanity mirror still a phantom | Partial |
| 155 | Glass | `docs/REPORT.md`, `fixloop/RESULT.md` | Documented: glazed slider called a passage; glass merges rooms on camera tiers | Partial: documented, not handled |
| 156 | Wet-look surfaces | `docs/REPORT.md`, `fp/recon/lidar_fuse.py` | Confidence-2 filter drops unreliable depth; failure mode documented | Partial: not tested on a wet-look floor |
| 157 | Low light | `docs/REPORT.md`, `fp/ingest/quality.py`, `CAPTURE_PROTOCOL.md` | Blur filter drops dark, smeared frames; the protocol says turn on all lights | Partial: not tested in low light |
