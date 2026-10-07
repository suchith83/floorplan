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
- **Evidence.** `uv sync` built the env from the lockfile here and `uv run pytest -q` passes (13 tests).
  The prototype was ported in three labelled commits rather than one lump, so the history shows
  what was reused; ARKitScenes ground-truth evaluation code (`eval/`) stayed behind, since the
  benchmark is rebuilt around our own captures in work order 07.
