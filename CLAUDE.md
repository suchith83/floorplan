# Floorplan: phone capture → dimensioned, stitched floor plan

## Goal
Turn a handheld iPhone capture (LiDAR via Stray Scanner, a video walkthrough, or 2–8 photos per room)
into one whole-property plan: per-room walls, ceiling height, floor area, openings, adjacency,
damage regions and scope line items, with an interval on every number. One command per capture,
one JSON schema for every tier, and it runs cold on an evaluator's laptop (no calls to our cloud).
Brief: [docs/brief/applied_ai_brief.md](docs/brief/applied_ai_brief.md). Strategy and work orders:
`planning/GAMEPLAN.md` and `prompts/` (work orders, `CHECKLIST.md`; local only, not committed).

## Where things stand
Read `docs/STATUS.md` (one section per work order 00–10) and `docs/DECISIONS.md` before starting.
Per-work-order defense notes live in `docs/defense/NN-<topic>.md`.

## Sample data (LiDAR tier; evaluator-provided, not in git)
Under `data/stray/<id>/` (zips in `data/zips/`); `scripts/fetch_data.sh` will fetch them.

| id | content |
|---|---|
| `c00a170fe1` | single room, ~4.8 × 6 m, 37 s, 1,715 frames, glimpses of neighbouring rooms |
| `1a8384c3f6` | whole flat ~10 × 10 m, corridor + ~6 rooms, 115 s, 5,251 frames. **No ceiling captured.** |
| `c7d28f72c6` | very likely the same flat with ceiling, 215 s, 9,745 frames, loops/revisits (drift work) |

Stray Scanner format (checked):
- `rgb.mp4`: 1920×1440 HEVC, nominally 60 fps but really ~46 fps with gaps: use `odometry.csv`
  timestamps, never i/60. Video frame *i* = row *i* of `odometry.csv` = `depth/{i:06d}.png`
  = `confidence/{i:06d}.png`.
- `depth/*.png`: 256×192 uint16, **millimetres**. `confidence/*.png`: values 0/1/2; keep 2 (maybe 1).
- `odometry.csv`: `timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, …` per frame.
  Pose = camera→world. **Camera axes are the OpenCV convention (x right, y down, z forward)**; the
  world is y-up (gravity-aligned). Verified: the ARKit convention (y up, −z forward) gives a smeared
  cloud, OpenCV gives sharp walls. Intrinsics are per frame at 1920×1440: scale by 256/1920 for depth.
  `camera_matrix.csv` holds one 3×3 K that differs from the per-frame values (c00a fx 1599.7 vs
  1597.9; c7d2 1601.1 vs 1581.2), so use the per-frame values.
- `imu.csv`: ~100 Hz `timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z`.
- Floor is ~1.45–1.5 m below the starting camera. In `c7d28f72c6` ceiling layers sit ~2.3–2.5 m
  above the floor, with another layer near 3.1 m (unverified).
- The floor-only scan must report its ceiling as **not observed** (or a very wide interval), never
  a confident guess. The two whole-flat scans are a repeatability pair. `rgb.mp4` doubles as a
  video-tier input and its stills as photo-tier input of the same rooms. LiDAR output is a
  *reference*, not ground truth.

## Repo layout
```
fp/                 the pipeline package (ported from round-1 prototype, branch `beta`)
  bundle.py         CaptureBundle / Frame / Recon: every input becomes a bundle
  cli.py            `fp run`, `fp check`
  check.py          per-stage inspection images + numbers
  ingest/           load_capture() auto-detect; photos, video, quality filter, ARKitScenes reader
  recon/            lidar_fuse (depth back-projection), mapanything (learned depth, Modal for now)
  geometry/         align (gravity, floor, Manhattan yaw, ceiling), plan (walls, room polygon),
                    rooms (watershed split at doorways, connections)
  damage/           SAM 3 detection, projection to surfaces, concealed-damage rules
  report/           svg plan, html report, debug BEV image
tests/              unit tests on synthetic inputs (no data needed)
scripts/            fetch_data.sh (stub)
docs/               brief/, STATUS.md, DECISIONS.md, defense/
data/  out/         ignored: raw captures, pipeline outputs
```
Code comments like `REVIEW #n`, `TESTING.md`, `HOW_IT_WORKS.md`, `SETUP.md` refer to round-1 docs
that stay on branch `beta` (`git show beta:REVIEW.md`).

## Commands
```
uv sync                                  # Python 3.12 env from uv.lock
uv run pytest -q                         # unit tests
uv run fp --help
uv run fp run data/stray/c00a170fe1      # → out/c00a170fe1/{plan.json,plan.svg,report.html}
uv run fp run <capture> --tier video --backend modal   # tier override; Modal is opt-in (paid)
uv run fp render <plan.json> --out DIR   # re-render plan.svg + report.html from a plan
uv run python -m fp.schema               # regenerate schema/plan.schema.json after editing fp/schema.py
uv run fp check <stage> <capture>        # frames|poses|fusion|cloud|align|walls|footprint|all|view
```
Until a stage exists it raises `StageNotBuilt` (fp/contract.py) and `fp run` still writes a valid plan
with a warning: the Stray reader arrives in 02, drift in 03, local MapAnything in 05, local damage in 06. The benchmark command arrives in work order 07.
Add dependencies with `uv add`, never pip.

## Conventions
- **Measurements** are `Measure{value, lo, hi, unit, method, observed}` (fp/schema.py, docs/SCHEMA.md);
  `[lo, hi]` is a stated 90% interval. Value null => observed false; "not observed" beats a guess.
  Build them with `fp.contract.measure()`; provisional interval constants live in fp/contract.py.
- **Surface IDs** are stable: `R1`, `R1.W1`, `R1.O1`, `R1.floor`, `R1.ceiling`, `D1`, `C1`, `S1`.
  Damage, scope and openings key off them; `fp.schema.validate` rejects dangling references.
- **Plan frame**: z up, floor at z = 0, metres, x/y aligned to the dominant (Manhattan) wall axes.
- **Thresholds** are module-level constants with a comment explaining the value. **No per-capture
  tuning**, and never tune on evaluation data to make a gate pass.
- Same `plan.json` schema for every tier; one CLI: `uv run fp run <capture>`.
- Debug images go to `out/<capture>/debug/`; name every image you mention in STATUS/defense notes.
- The live path must run on a laptop (CPU/MPS); GPU/Modal may only be an optional `--backend`.
  Today MapAnything and SAM 3 run only on Modal (no `--backend` flag yet; work order 05).
- Prefer numpy / scipy / open3d / opencv / shapely.

## Commits
- Small commits, as you go, each saying what changed and why (plus the key number if any).
- Stage files by name; check `git status` first. Never commit `data/`, `out/`, zips, weights or caches.
  No tracked file over 5 MB.

## How to make decisions
The user defends every decision live, **without tools**. Prefer simple, explainable methods over
clever ones, and log each real decision in `docs/DECISIONS.md` (context → options → choice →
evidence). Each work order ends with `docs/defense/NN-*.md`: likely evaluator questions with short
answers the user can say from memory.
