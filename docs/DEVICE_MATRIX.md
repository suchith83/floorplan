# Device matrix

Which tier runs on which phone, what we actually tested, and the accuracy we measured. Status as of 8 October 2026.

**Read this first.** We have no iPhone. Every number below comes from the evaluator's three Stray Scanner sample
captures (`data/stray/`), or from video and photos cut out of those same captures (`data/derived/`, made by
[`scripts/make_camera_tiers.py`](../scripts/make_camera_tiers.py)). No capture from any phone of ours has been run
yet (`data/own/` does not exist). None of the numbers is against a tape: there is no tape on the sample data, so the
camera tiers are scored against the LiDAR plan of the same capture, and LiDAR against itself (two scans of the same flat).
How to capture: [CAPTURE_PROTOCOL.md](../CAPTURE_PROTOCOL.md).

## Tier × phone

| phone | LiDAR tier (Stray Scanner) | Video tier | Photo tier |
|---|---|---|---|
| iPhone 15, 15 Plus | not available: no LiDAR sensor | runs; not tested on this phone | runs; not tested on this phone |
| iPhone 15 Pro / Pro Max | runs; not tested on this phone | runs; not tested on this phone | runs; not tested on this phone |
| iPhone 16, 16 Plus, 16e | not available: no LiDAR sensor | runs; not tested on this phone | runs; not tested on this phone |
| iPhone 16 Pro / Pro Max | runs; not tested on this phone | runs; not tested on this phone | runs; not tested on this phone |
| iPhone 17, iPhone Air | not available: no LiDAR sensor | runs; not tested on this phone | runs; not tested on this phone |
| iPhone 17 Pro / Pro Max | runs; not tested on this phone | runs; not tested on this phone | runs; not tested on this phone |
| iPad Pro with LiDAR | expected to run (same Stray Scanner format); not tested | expected to run; not tested | expected to run; not tested |
| Android phone | not available: Stray Scanner is iOS only | expected to run; not tested | expected to run; not tested |
| **What was actually tested** | 3 Stray Scanner exports from the evaluator, from one Pro-class iPhone of unknown model (the files carry no device name: only intrinsics, fx ≈ 1,600 px at 1920×1440) | the `rgb.mp4` of 2 of those exports, with the rotation tag an iPhone writes added (D05.7) | 7 stills (1 room) and 17 stills (3 rooms) cut from 2 of those exports, in room folders, with EXIF focal length added (D05.7, D05.8) |

"Runs" means the input format is one our reader handles (Stray Scanner folder; `.mov`/`.mp4` HEVC or H.264, HDR or
SDR, any rotation; HEIC/JPEG/PNG). It does not mean we have run a file from that phone.

**Formats.** iPhone video is HEVC, often 10-bit HDR (HLG or Dolby Vision); we decode it with the bundled ffmpeg,
tone-map it to SDR and apply the rotation tag. This is covered by a unit test on a synthetic 10-bit HLG HEVC clip with
rotation ([`tests/test_camera_ingest.py`](../tests/test_camera_ingest.py)), not by a real iPhone clip. Photos: HEIC via
pillow-heif, EXIF rotation, Display P3 converted to sRGB, focal length from EXIF. Android HDR video (PQ / HDR10) goes
through the same tone-map path; untested on a real file. Photos without EXIF focal length still run, with the focal
length estimated and a warning.

**Own capture.** The plan was to capture the photo and video tiers on my own phone (not an iPhone); those captures were not taken before the deadline.

## Accuracy we measured

All from `make benchmark` ([`eval/BENCHMARK.md`](../eval/BENCHMARK.md)).

| tier | measure | result | against what | source |
|---|---|---|---|---|
| LiDAR | repeatability, wall-to-wall spans (same flat, two scans) | median difference **5.5 cm**, p90 11.8 cm; 0 of 16 spans within 1 cm | the other scan | [BENCHMARK, Repeatability](../eval/BENCHMARK.md#repeatability); [repeatability_lidar.md §3](../eval/repeatability_lidar.md) |
| LiDAR | wall thickness in the cloud (a drift measure) | single room 2.2 cm; whole flat 4.2 → 1.9 cm (1a83) and 3.6 → 3.3 cm (c7d2) with drift correction | itself | [BENCHMARK, Gates](../eval/BENCHMARK.md#gates) (drift rows); [STATUS 02](STATUS.md) |
| LiDAR | ceiling height per room | c7d2: 8 of 9 rooms observed, 2.33 to 3.09 m; 1a83 (ceiling never filmed): every room "not observed" | nothing (not taped) | [STATUS 04](STATUS.md); [BENCHMARK, Gates](../eval/BENCHMARK.md#gates) |
| LiDAR | openings found (c7d2) | 7 of 14 right (50 %): 4 missed, 1 phantom, 2 wrong kind | by eye in the video | [BENCHMARK, Openings](../eval/BENCHMARK.md#openings-c7d28f72c6-detection-by-eye-widths-not-taped) |
| Video | wall length error | median **39 %** / **37 %**, p90 71 % / 76 % (c00a / c7d2); 0 walls within 3 % | LiDAR plan of the same capture | [BENCHMARK, Gates](../eval/BENCHMARK.md#gates) |
| Video | footprint area | **+19 %** / **+22 %** (c00a / c7d2) | LiDAR plan | [BENCHMARK, Camera tiers](../eval/BENCHMARK.md#camera-tiers-vs-the-lidar-reference-same-capture) |
| Photos | wall length error | median **15 %** / **22 %**, p90 24 % / 62 % (c00a / c7d2) | LiDAR plan | [BENCHMARK, Gates](../eval/BENCHMARK.md#gates) |
| Photos | footprint area of the stitched plan | **−70 %** / **−79 %** (c00a / c7d2); room adjacency wrong on c7d2 | LiDAR rooms the photos were taken in | [BENCHMARK, Gates](../eval/BENCHMARK.md#gates) (photo stitch rows) |

In plain words: the LiDAR tier gets walls to within a few centimetres of itself but misses the 1 cm repeatability
gate. The video tier gets the rough shape and area (within about a quarter) but single walls are off by a third or
more. The photo tier recovers only a fraction of the floor. The intervals in `plan.json` are fitted to these errors,
so camera-tier wall intervals are very wide (about ±310 % of the length for photos, ±410 % for video, floored at
0 m); the honest reading is "we don't know this wall". See [`eval/CALIBRATION.md`](../eval/CALIBRATION.md).

## Laptop

| machine | tested? | LiDAR tier | video tier | photo tier |
|---|---|---|---|---|
| MacBook, Apple M5, 16 GB, MPS | yes | c00a (1 room): 33 s; c7d2 (whole flat): 233 s | 338 s (c00a, 109 keyframes) and 511 s (c7d2, 240 keyframes) with the model running; 14 s and 72 s replayed from the cache | 47 s (7 photos) and 54 s (17 photos) live; 2 s and 5 s replayed |
| Linux, CPU only | no | expected to run (no learned depth; damage models on CPU); not timed | expected to run, much slower (MapAnything on CPU); not timed | expected to run; not timed |
| Linux with CUDA GPU | no | not timed | not timed | not timed |
| less than 16 GB RAM | no | not timed | not timed; the video run peaked at 14.3 GB on 16 GB | not timed |

Sources: [BENCHMARK, Timing](../eval/BENCHMARK.md#timing-per-stage-s) (replayed runs, LiDAR) and
[STATUS 05](STATUS.md) (live camera-tier runs). The first LiDAR run of a capture adds the one-off video decode
(c00a: 75 s instead of 33 s).
