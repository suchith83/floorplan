# Status

One section per work order (`planning/prompts/NN-*.md`). Each session updates its own section at
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
## 02 — LiDAR tier: Stray Scanner ingest and single room end to end: not started
## 03 — Drift accountability and the stitched whole-property plan (LiDAR): not started
## 04 — Openings (doors, windows, passages) and per-room ceiling height: not started
## 05 — Video and photo tiers (no depth, no poses): not started
## 06 — Damage regions, concealed-damage flags, scope line items: not started
## 07 — Calibrated intervals and the benchmark harness: not started
## 08 — Capture protocol, device matrix, README, report, compliance matrix: not started
## 09 — The fix loop: not started
## 10 — Cold-run rehearsal: not started
