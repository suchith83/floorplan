"""Video file -> keyframes. Samples about 3 frames a second, drops blurry frames and frames where the
camera did not move, then keeps the sharpest frame in each of up to MAX_KEYFRAMES equal time slots.
MapAnything sees all keyframes in one joint pass, so the cap sets the GPU memory."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from fp.bundle import CaptureBundle, Frame
from fp.ingest.quality import MIN_SHARP_FRAC, image_stats

MAX_KEYFRAMES = 150
SAMPLE_FPS = 3.0
SAVE_WIDTH = 1280
DUPLICATE_DIFF = 1.5  # mean grey difference to the previous kept frame


def load(path: Path, max_frames: int = MAX_KEYFRAMES) -> CaptureBundle:
    path = Path(path)
    work = Path("out/_cache") / path.stem / "frames"
    work.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(path))  # OpenCV applies the phone's rotation flag
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps / SAMPLE_FPS))
    saved, times, i = [], [], 0
    while True:
        ok, im = cap.read()
        if not ok:
            break
        if i % step == 0:
            if im.shape[1] > SAVE_WIDTH:
                im = cv2.resize(im, None, fx=SAVE_WIDTH / im.shape[1], fy=SAVE_WIDTH / im.shape[1], interpolation=cv2.INTER_AREA)
            p = work / f"{i:06d}.jpg"
            cv2.imwrite(str(p), im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved.append(p)
            times.append(i / fps)
        i += 1
    cap.release()
    if not saved:
        raise SystemExit(f"Could not read any frame from {path}")
    st = image_stats(saved)
    good = (st["sharpness"] >= MIN_SHARP_FRAC * np.median(st["sharpness"])) & (st["diff_prev"] >= DUPLICATE_DIFF)
    idx = np.flatnonzero(good)
    if len(idx) > max_frames:  # sharpest frame in each equal time slot
        idx = np.array([s[np.argmax(st["sharpness"][s])] for s in np.array_split(idx, max_frames)])
    frames = [Frame(rgb=saved[k], timestamp=times[k]) for k in idx]
    st_src = path.stat()
    source_ids = [f"{path.resolve()}:{st_src.st_size}:{st_src.st_mtime_ns}:{saved[k].stem}" for k in idx]
    return CaptureBundle(source="video", frames=frames, up=None,
                         meta={"file": path.name, "duration_s": round(i / fps, 1), "frames_sampled": len(saved),
                               "dropped_blurry_or_still": int((~good).sum()), "keyframes": len(frames),
                               "_source_ids": source_ids})
