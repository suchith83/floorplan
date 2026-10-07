"""Folder of photos -> frames. One sub-folder per room (photos/kitchen/*.jpg) is optional; its name is
kept so rooms in the plan can be labelled. EXIF rotation is applied; near-duplicates are dropped."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from fp.bundle import CaptureBundle, Frame
from fp.ingest.quality import image_stats

MAX_PHOTOS = 150
SAVE_SIDE = 1600
DUPLICATE_DIFF = 1.5


def load(path: Path, max_frames: int = MAX_PHOTOS) -> CaptureBundle:
    path = Path(path)
    files = sorted(p for p in path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    work = Path("out/_cache") / path.name / "photos"
    work.mkdir(parents=True, exist_ok=True)
    saved, rooms = [], []
    for k, p in enumerate(files):
        im = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        im.thumbnail((SAVE_SIDE, SAVE_SIDE))
        out = work / f"{k:04d}.jpg"
        im.save(out, quality=95)
        saved.append(out)
        rooms.append(p.parent.name if p.parent != path else "")
    st = image_stats(saved)
    keep = np.flatnonzero(st["diff_prev"] >= DUPLICATE_DIFF)
    if len(keep) > max_frames:
        keep = keep[np.linspace(0, len(keep) - 1, max_frames).astype(int)]
    frames = [Frame(rgb=saved[k], timestamp=float(i)) for i, k in enumerate(keep)]
    return CaptureBundle(source="photos", frames=frames, up=None,
                         meta={"folder": path.name, "photos": len(files), "used": len(frames),
                               "_frame_rooms": [rooms[k] for k in keep],
                               "_source_ids": [f"{files[k].resolve()}:{files[k].stat().st_mtime_ns}" for k in keep]})
