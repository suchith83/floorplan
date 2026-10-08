# Walk-in script

What to type, what to say while it runs, where to look, and what to do when it fails. Times are on the M5 MacBook
(16 GB, MPS) from [README.md](../README.md#one-command-per-capture) and [COLD_RUN.md](COLD_RUN.md).

## Before you start (once, ~2 min if the zips are at hand)

```sh
cd floorplan
uv sync
ls data/zips/            # single_room.zip single_scan_floor_only.zip single_scan_with_ceiling.zip (from Bhavana's link)
scripts/fetch_data.sh    # dataset download fails (not published) -> warning, then unpacks data/zips/ into data/stray/
uv run pytest -q         # 152 tests, no data needed: shows the environment works
```

Open each `out/<name>/report.html` in a browser as soon as the run prints `wrote`.

## 1. LiDAR tier (start here: fast and the most accurate)

```sh
uv run fp run data/stray/c00a170fe1            # 1 room, ~35 s (~75-110 s the first time: video decode is cached after)
uv run fp run data/stray/1a8384c3f6            # whole flat, no ceiling captured, ~3-4 min
```

**Say while it runs:** "Depth maps are back-projected with the phone's own poses (OpenCV camera axes, odometry
timestamps, confidence-2 pixels only), the cloud is levelled by gravity and the floor, rotated to the dominant wall
axes, then walls come from a horizontal slice at wall-band height and rooms from a watershed split at doorway necks.
Every number is a `Measure` with a 90 % interval fitted on the benchmark."

**Where to look in the report:**
- *Plan*: the drawing; dashed edges are inferred, solid are observed.
- *Rooms*: area and ceiling with `[lo, hi]`. On `1a8384c3f6` every ceiling says **not observed**: that is on purpose,
  the scan has no ceiling, and "not observed" beats a guess.
- *Warnings*: partial rooms (camera < 3 s inside), mirror-rejected openings (`debug/openings/*.png`), unconfirmed damage.
- *Drift correction* (whole flats): before/after loop closure, `debug/drift_ablation.png`.
- `debug/`: `fusion_topdown.png`, `wall_slice.png`, `rooms_split.png`, one image per stage.

## 2. Video tier

```sh
uv run fp run data/stray/c00a170fe1 --tier video     # the same capture's rgb.mp4 alone, no depth or poses
uv run fp run path/to/IMG_1234.MOV                     # a phone video
```

Replayed from `out/_cache` (only on the author's laptop: the cache is not published) 14-72 s; **live 338 s (1 room)
to 511 s (whole flat)**. Start it, then talk.

**Say:** "Keyframes are picked by sharpness and parallax, MapAnything predicts metric depth and poses locally on MPS,
then the same geometry code as LiDAR runs. Camera tiers have no drift correction, so the intervals are wider; the
fix loop (fixloop/RESULT.md) moved wall error p90 from 105 % to 76 %, still a FAIL against ±3 %."

## 3. Photo tier

```sh
uv run fp run data/derived/c7d28f72c6/photos          # 3 room folders, 19 photos, ~1 min live
uv run fp run path/to/my_home                          # my_home/<room>/*.HEIC|*.jpg
```

**Say:** "One folder per room, 2 to 8 photos each; a doorway photo saved in both rooms' folders links them (the loader
dedupes it by bytes or by EXIF capture time). Rooms with no link are placed apart and warned about."

**Look at:** *Connections* (which rooms the doorway photos linked) and *Warnings* (unlinked or unplaced rooms).

## If it fails

| symptom | do this |
|---|---|
| `fetch_data.sh` prints `missing data/stray/<id>` | the zip is not in `data/zips/` or has another name; copy it in and rerun |
| `data/derived/` missing (no video/photo inputs for the samples) | `uv run python scripts/make_camera_tiers.py data/stray/c00a170fe1 data/stray/c7d28f72c6` (a few min), or use `--tier video` / `--tier photos` on the Stray folder instead |
| model weights downloading for minutes | `scripts/fetch_weights.sh` beforehand; LiDAR runs need only the 0.9 GB damage models; `--no-damage` needs none |
| video run too slow for the room | `--max-frames 40` (fewer keyframes; accuracy cost not measured), or show the LiDAR tier and the committed `fixloop/after/` plans |
| damage step slow or crashes | `--no-damage` |
| out of memory on a 16 GB machine | close the browser, `--max-frames 40`, or `--backend modal` (paid, needs a Modal login) |
| a photo folder with photos directly in it | refused with a message: put the photos in one subfolder per room |
| any stack trace | rerun with `--no-damage`; if it persists, show `out/<name>/` from the last good run and say which stage failed |

## Questions to expect

- *Why does the floor-only scan say "not observed" for the ceiling?* No down-facing layer of at least 1 m² 2.1–3.6 m
  above the floor; a confident guess would be worse than no number.
- *Is the LiDAR plan ground truth?* No, a reference. No tape exists for the sample data; tape gates are NOT SCORED.
- *What happens on an incomplete scan?* See [COLD_RUN.md](COLD_RUN.md): a scan cut to 40 % still gives a plan, with
  partial-room warnings.
