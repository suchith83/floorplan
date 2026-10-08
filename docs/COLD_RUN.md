# Cold run (work order 10), 8 Oct 2026

M5 MacBook, 16 GB, MPS. Scratch copies only; nothing here is in git except this file.

## Fresh clone to the first plan

`git clone` of the repo into an empty folder, then the README's quick start literally, with the three sample zips copied
into `data/zips/` (the Hugging Face dataset is not published, so the download is expected to fail).

| step | time | note |
|---|---|---|
| `git clone` | 1 s | local clone of the committed tree |
| `uv sync` | 2 s | **warm** uv cache on this Mac; a cold machine downloads ~2 GB of wheels first (not timed) |
| copy the zips into `data/zips/` | 1 s | 0.8 GB |
| `scripts/fetch_data.sh` | 14 s | download fails with *Repository Not Found* → warning, then the three zips unpacked into `data/stray/` |
| `uv run fp run data/stray/c00a170fe1` | 112 s | first run decodes the video; damage weights were already in the HF cache |
| **total** | **2 min 10 s** | plan identical to the main checkout's (rooms 9.07 / 8.80 / 6.61 m²) |

What a stranger would need that the README did not say before this work order: where the sample zips come from
(now: "from Bhavana's link, into `data/zips/`"), and that `fetch_data.sh` must not stop when the dataset is missing
(fixed: it warns and continues). Not tested: a machine with an empty uv and Hugging Face cache (adds the wheel download
and 0.9 GB of damage weights; 5.8 GB with the camera-tier model).

## Unseen-input matrix

All live (`--no-cache`), inputs the code was never tuned on, built from the sample data.

| input | result | time | first message |
|---|---|---|---|
| Stray `c00a170fe1` truncated to its first 40 % (686 of 1,715 poses, depth, confidence; video cut at the last pose) | **crash, fixed** → PASS: plan, 2 rooms (8.33 [7.01–9.65] m², R2 partial) | 57 s → 51 s | before: `plan.json does not match schema 1.0: drift.metrics.loop_rmse_before_cm: Input should be a finite number`. After: `rgb.mp4 has 692 frames but odometry.csv only 686 poses: 6 frames without a pose were ignored (truncated capture?)` |
| Photos, only 2 per room (first two stills of each room of `c7d28f72c6`; no doorway photo survives) | PASS: plan with R1 only (6.99 [0.00–29.63] m²) | 51 s | `Room 'R2' shares no photo and too few feature matches (< 40) with the other rooms ... NOT placed in the plan (not guessed, not overlapped). Re-capture with a doorway photo saved in ...` |
| Photo folder with stray files: `notes.txt`, `.DS_Store`, an empty `empty.heic`, a text file renamed `IMG_9999.JPG` | PASS: plan, 3 rooms linked | 78 s | `Skipped 2 unreadable photo(s): R2/empty.heic (UnidentifiedImageError), R3/IMG_9999.JPG (UnidentifiedImageError)`; non-image extensions ignored silently |
| Stray capture at a path with spaces (`my scans/flat one/c00a copy`) | PASS: same plan as the original | 80 s | none beyond the usual warnings |

**The crash.** Drift correction reports the loop-closure residual as NaN when the scan has at least two submaps but
never revisits a place (no loop candidates). NaN is not valid JSON, so schema validation refused the plan. The full
sample scans all revisit, so this never showed; any short scan that walks one way would have hit it. Fix:
`fp/cli.py` writes non-finite drift metrics as `null` ("not measured"); the plan's drift section now reads
`0 ICP loop closures accepted of 0 candidates (camera never revisited a place: poses kept)`.

## Not run (time)

Mixed HEIC/JPEG with portrait and landscape; one room with no doorway photo among linked rooms; video with a fast pan, a
dark section and a mirror (needs an own capture, and a live video run is > 5 min); the runtime-budget step (times are in
the README from work order 05/07).
