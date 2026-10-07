"""Stray Scanner export -> CaptureBundle (LiDAR tier).

Layout (checked on the evaluator's three captures, see CLAUDE.md):
  rgb.mp4            1920x1440 HEVC; nominally 60 fps, really ~46 fps with gaps -> use odometry timestamps
  odometry.csv       timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, ...  (one row per video frame)
                     pose = camera->world, OpenCV camera axes (x right, y down, z forward), world y-up
  depth/{i:06d}.png  256x192 uint16 millimetres
  confidence/{i:06d}.png  uint8 0/1/2 (ARKit low/medium/high)
Video frame i = odometry row i = depth/{i:06d}.png = confidence/{i:06d}.png.

The video is decoded once, front to back (HEVC seeks are slow; a full pass runs at ~340 fps), and every
frame is cached as a quarter-resolution JPEG under <cache>/stray/<id>/. Downstream code reads frames
as images like every other tier; `Frame.video_index` keeps the link to the full-resolution video."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from fp.bundle import CaptureBundle, Frame
from fp.contract import StageError

RGB_SCALE = 0.25          # cached frames are 480x360: enough for sharpness, colour and the debug views
JPEG_QUALITY = 90
DEFAULT_CACHE = Path("out/_cache")


def read_odometry(d: Path) -> dict[str, np.ndarray]:
    """odometry.csv -> {"t": (n,), "index": (n,), "T_wc": (n,4,4), "K": (n,3,3) at 1920x1440}."""
    with open(Path(d) / "odometry.csv", newline="") as fh:
        rows = list(csv.reader(fh))
    head = [h.strip() for h in rows[0]]
    col = {h: k for k, h in enumerate(head)}
    data = [r for r in rows[1:] if len(r) >= 13 and r[0].strip()]
    num = lambda name: np.array([float(r[col[name]]) for r in data])
    q = np.stack([num("qx"), num("qy"), num("qz"), num("qw")], 1)   # scipy order: x, y, z, w
    T = np.tile(np.eye(4), (len(data), 1, 1))
    T[:, :3, :3] = Rotation.from_quat(q).as_matrix()
    T[:, :3, 3] = np.stack([num("x"), num("y"), num("z")], 1)
    K = np.tile(np.eye(3), (len(data), 1, 1))
    K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = num("fx"), num("fy"), num("cx"), num("cy")
    return {"t": num("timestamp"), "index": num("frame").astype(int), "T_wc": T, "K": K}


def _cache_frames(video: Path, cache: Path, n_expected: int) -> int:
    """Decode the whole video once into cache/{i:06d}.jpg. Returns the number of frames decoded.
    A marker file records the video's size and mtime, so an edited capture is re-decoded."""
    st = video.stat()
    key = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "scale": RGB_SCALE}
    marker = cache / "frames.json"
    if marker.is_file():
        done = json.loads(marker.read_text())
        if done.get("key") == key:
            return int(done["n"])
    cache.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    n = 0
    while n < n_expected:
        ok, img = cap.read()
        if not ok:
            break
        small = cv2.resize(img, None, fx=RGB_SCALE, fy=RGB_SCALE, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(cache / f"{n:06d}.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        n += 1
    cap.release()
    marker.write_text(json.dumps({"key": key, "n": n}))
    return n


def load(d: Path, cache_dir: Path | None = None) -> CaptureBundle:
    """Every odometry row that has a decoded video frame and a depth map becomes a Frame
    (the frame filter in fp/ingest/quality.py thins them later). K is scaled to the cached RGB size."""
    d = Path(d)
    od = read_odometry(d)
    if not len(od["t"]):
        raise StageError("ingest", f"Stray Scanner capture {d} has no poses in odometry.csv.")
    if not (d / "rgb.mp4").is_file():
        raise StageError("ingest", f"Stray Scanner capture {d} has no rgb.mp4.")
    cache = Path(cache_dir or DEFAULT_CACHE) / "stray" / d.name
    n_video = _cache_frames(d / "rgb.mp4", cache, len(od["t"]))
    S = np.diag([RGB_SCALE, RGB_SCALE, 1.0])
    frames, missing = [], 0
    for k, i in enumerate(od["index"]):
        depth, conf = d / "depth" / f"{i:06d}.png", d / "confidence" / f"{i:06d}.png"
        if i >= n_video or not depth.is_file():
            missing += 1
            continue
        frames.append(Frame(rgb=cache / f"{i:06d}.jpg", K=S @ od["K"][k], T_wc=od["T_wc"][k], depth=depth,
                            timestamp=float(od["t"][k]), confidence=conf if conf.is_file() else None,
                            video_index=int(i)))
    dt = np.diff(od["t"])
    meta = {"dataset": "Stray Scanner", "capture": d.name, "odometry_rows": len(od["t"]),
            "video_frames": n_video, "frames_missing": missing,
            "duration_s": round(float(od["t"][-1] - od["t"][0]), 1) if len(dt) else 0.0,
            "fps_median": round(float(1 / np.median(dt)), 1) if len(dt) else None,
            "has_confidence": all(f.confidence is not None for f in frames) and bool(frames)}
    # Stray (ARKit) world frame is gravity-aligned with +Y up.
    return CaptureBundle(source="lidar", frames=frames, up=np.array([0.0, 1.0, 0.0]), meta=meta)
