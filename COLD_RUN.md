# Cold run on an unseen space — checklist

The system runs with fixed thresholds and no per-scene tuning. On the day, change no code.

## Before you go (5 min)
1. `uv run python scripts/check_setup.py` → all ✓ (Modal login, Hugging Face secret).
2. `uv run pytest` → all tests pass.
3. Phone: 5 GB free, battery above 60 %, camera at 1080p 30 fps, stabilisation off (CAPTURE_GUIDE.md).

## On site (about 20 min)
1. Follow CAPTURE_GUIDE.md §1: one slow video walkthrough of every room. Turn slowly.
2. If there is time: §2 photos per room, and §4 two tape measurements of one room.
3. Note any visible damage before you look at any result.

## Run (about 5 min, needs the internet for Modal)
```bash
uv run fp run path/to/video.mp4                 # plan.json, plan.svg, report.html in out/video/
uv run fp run path/to/video.mp4 --wet-rooms R2  # after a first look: mark the bathroom/kitchen rooms
open out/video/report.html
```

## Read the result honestly
- Low confidence (red badge) or a dashed wall means "measure this by hand". Say so; do not defend the number.
- `floor: assumed` in the report header means the floor was not seen. Re-film with the camera tilted down.
- If MapAnything or SAM 3 fails (network, GPU queue), the floor plan still comes out; damage shows the error.
- Compare one or two room sizes with the tape measure on the spot, and write down the error.
