# floorplan

Turns a phone capture of a home (a Stray Scanner LiDAR scan, a video walkthrough, or 2 to 8 photos per room) into
one stitched floor plan: walls, floor area, ceiling height, doors and windows, how rooms connect, damage and repair
scope. Every number comes with a 90 % interval. One command per capture; everything runs on your laptop.

- How to capture: [CAPTURE_PROTOCOL.md](CAPTURE_PROTOCOL.md) (one page)
- Which phone, which tier, how accurate: [docs/DEVICE_MATRIX.md](docs/DEVICE_MATRIX.md)
- Technical report: [docs/REPORT.md](docs/REPORT.md). Requirement by requirement: [COMPLIANCE.md](COMPLIANCE.md)
- Measured accuracy: [eval/BENCHMARK.md](eval/BENCHMARK.md)

## Prerequisites

- macOS or Linux, 16 GB RAM (the video tier peaked at 14.3 GB), about 15 GB free disk
  (Python environment 2 GB on a Mac, more on Linux x86 where PyTorch pulls its CUDA wheels; model weights 5.8 GB;
  sample data and cache 2.2 GB; plus the uv download cache).
- `git`, and `unzip` (preinstalled on macOS; `sudo apt install unzip` on minimal Linux).
- `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh` (then open a new terminal). uv downloads Python 3.12
  itself. ffmpeg comes bundled (the `imageio-ffmpeg` package); you don't install it.

Tested on an Apple M5 MacBook (16 GB, MPS). Linux has not been tested. It should run on CPU; the LiDAR tier does not
need the big model, but the video and photo tiers will be much slower on CPU (not timed): use an NVIDIA GPU there, or
`--backend modal` (paid, opt-in). See [docs/DEVICE_MATRIX.md](docs/DEVICE_MATRIX.md#laptop).

## Quick start

```sh
git clone https://github.com/suchith83/floorplan.git
cd floorplan
uv sync                          # Python 3.12 + dependencies from uv.lock
mkdir -p data/zips               # copy the three sample zips here (see "Sample data" below)
scripts/fetch_data.sh            # unpacks data/zips/ into data/stray/<id>/
scripts/fetch_weights.sh         # model weights into the Hugging Face cache (optional: they also download on first use)
uv run fp run data/stray/c00a170fe1
```

Open `out/c00a170fe1/report.html` in a browser. The output folder holds:

```
out/c00a170fe1/plan.json     the plan (schema: schema/plan.schema.json, explained in docs/SCHEMA.md)
out/c00a170fe1/plan.svg      the drawing
out/c00a170fe1/report.html   plan, numbers with intervals, warnings, damage, scope
out/c00a170fe1/debug/        one image per pipeline stage
```

**Sample data.** The raw captures are not in git and the Hugging Face dataset is not published. Get the three
sample zips from the link Bhavana sent with the brief (`single_room.zip`, `single_scan_floor_only.zip`,
`single_scan_with_ceiling.zip`, 0.8 GB together), put them in `data/zips/`, then run `scripts/fetch_data.sh`. It tries
the dataset `FP_DATA_REPO` (default `suchith83/floorplan-data`) first; when that fails it prints a warning and unpacks
the local zips into `data/stray/<id>/`. Without the dataset there is no model-output cache, so camera-tier runs are
live. The video and photo inputs cut from the sample captures are rebuilt with
`uv run python scripts/make_camera_tiers.py data/stray/c00a170fe1 data/stray/c7d28f72c6` (it first makes a LiDAR
reference plan of each capture, a few minutes).

`fetch_weights.sh` pre-downloads `facebook/map-anything-apache` (depth and poses for video and photos, 4.9 GB),
`IDEA-Research/grounding-dino-tiny` and `facebook/sam2.1-hiera-small` (damage detection, 0.9 GB together). The
LiDAR tier needs only the two damage models, so the first LiDAR plan does not wait for the 4.9 GB download.

## One command per capture

The tier is detected from what you pass in:

```sh
uv run fp run path/to/stray_folder        # LiDAR: a Stray Scanner folder (odometry.csv + depth/)
uv run fp run path/to/IMG_1234.MOV        # Video: one .mov / .mp4 file
uv run fp run path/to/my_home             # Photos: a folder of room folders (my_home/kitchen/*.HEIC, ...)
```

Output goes to `out/<name>/` (`--out DIR` to change it). On a Stray folder, `--tier video` or `--tier photos` runs
the same capture as video or photos (its RGB only), to compare tiers on the same rooms. `uv run fp run --help` lists
the other flags (`--no-damage`, `--wet-rooms R2,R3`, `--no-drift-correction`, ...). A photo folder with photos
directly in it (not in room folders) is refused with a message saying how to fix it.

**How long** (M5 MacBook, 16 GB; [eval/BENCHMARK.md](eval/BENCHMARK.md#timing-per-stage-s), [docs/STATUS.md](docs/STATUS.md) section 05):

| capture | time |
|---|---|
| LiDAR, 1 room (`c00a170fe1`) | 33 s (75 s the first time: the video is decoded once and cached) |
| LiDAR, whole flat (`c7d28f72c6`) | about 4 min |
| Video, live model | 338 s (1 room) to 511 s (whole flat) |
| Video, replayed from the cache | 14 s to 72 s |
| Photos, live model | about 50 s for 7 to 17 photos (one model pass); a 5-room home (~40 photos) needs 2 passes, about 2.5 min (estimated, not timed) |

Model outputs are cached under `out/_cache/`, keyed by the input bytes. A rerun on the same input replays them;
`--no-cache` forces the live model. A new capture always runs live.

## Reproduce every number

```sh
uv run pytest -q          # 152 unit tests on synthetic inputs, no data needed
make benchmark            # every sample capture, every tier; refits the intervals; writes eval/BENCHMARK.md (~16 min)
make benchmark-tables     # rebuild the tables from the plans already in out/ (seconds)
make fix-before           # fix loop: rerun the whole benchmark at the tag before the fix (~16 min) -> fixloop/before/
make fix-after            # same at the tag after the fix -> fixloop/after/
```

`make benchmark` replays the learned models from `out/_cache` (fetched by `fetch_data.sh`); any single run can be
redone live with `uv run fp run <capture> --no-cache`. The fix-loop story is in [fixloop/RESULT.md](fixloop/RESULT.md).

Inspect one stage: `uv run fp check <stage> <capture>`, stage one of
`frames poses fusion cloud align walls footprint all view`. Re-render a plan: `uv run fp render out/<name>/plan.json`.

## No cloud

Nothing calls our infrastructure. Model weights come from Hugging Face once, then run locally (CUDA, else Apple
MPS, else CPU). `--backend modal` runs the learned models on Modal GPUs instead; it is paid, opt-in, and never
needed.

## Repo layout

```
fp/          the pipeline: ingest/ (Stray, video, photos), recon/, geometry/, damage/, report/, cli.py
eval/        benchmark harness and results (BENCHMARK.md, CALIBRATION.md, repeatability_lidar.md)
fixloop/     fix-loop declaration, before/after results
scripts/     fetch_data.sh, fetch_weights.sh, make_camera_tiers.py, fixloop.sh
schema/      plan.schema.json (generated from fp/schema.py)
tests/       unit tests
docs/        STATUS.md, DECISIONS.md, SCHEMA.md, DEVICE_MATRIX.md, defense/, brief/
data/ out/   not in git: inputs (fetched) and outputs
```
