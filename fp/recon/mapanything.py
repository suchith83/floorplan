"""Camera-only input (photos, video) -> frames with metric depth, K and pose, predicted by MapAnything.

Depth maps are written as uint16 mm PNGs, like LiDAR depth, so `fuse_depth` and every later stage run
unchanged. Low-confidence pixels (the bottom 10 %) and pixels the model masks out are set to 0 (= no depth)."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import cv2
import numpy as np

from fp.bundle import CaptureBundle, Frame
from fp.recon.mapanything_modal import MODEL_ID

CONF_PERCENTILE = 10
UPLOAD_MAX_SIDE = 1024  # MapAnything works at about 518 px; no need to upload full-resolution photos


def _jpeg(path: Path) -> bytes:
    im = cv2.imread(str(path))
    s = UPLOAD_MAX_SIDE / max(im.shape[:2])
    if s < 1:
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()


def _cache_key(bundle: CaptureBundle) -> str:
    """Identifies the input by its SOURCE files (video/photo frames are re-extracted on every run)."""
    h = hashlib.sha1(MODEL_ID.encode())
    ids = bundle.meta.get("_source_ids") or [f"{Path(f.rgb).resolve()}:{Path(f.rgb).stat().st_mtime_ns}"
                                             for f in bundle.frames]
    for i in ids:
        h.update(i.encode())
    return h.hexdigest()


def predict_depth(bundle: CaptureBundle, work: Path) -> CaptureBundle:
    """Cached per input: a second run on the same frames (a geometry change, a flaky network) skips Modal."""
    work.mkdir(parents=True, exist_ok=True)
    cache, key = work / "mapanything.npz", _cache_key(bundle)
    if cache.exists() and str(np.load(cache)["key"]) == key:
        print(f"MapAnything: reusing {cache}")
        z = np.load(cache)
    else:
        import modal

        from fp.recon.mapanything_modal import app, infer
        images = [_jpeg(f.rgb) for f in bundle.frames]
        print(f"MapAnything on Modal: {len(images)} views ...")
        with modal.enable_output(), app.run():
            raw = infer.remote(images)
        z = dict(np.load(io.BytesIO(raw)))
        np.savez_compressed(cache, key=key, **z)
    conf_floor = np.percentile(z["conf"][z["mask"]].astype(np.float32), CONF_PERCENTILE)
    frames = []
    for i, f in enumerate(bundle.frames):
        ok = z["mask"][i] & (z["conf"][i] >= conf_floor)
        depth_mm = np.where(ok, np.round(z["depth"][i].astype(np.float32) * 1000), 0).clip(0, 65535).astype(np.uint16)
        rgb_p, depth_p = work / f"{i:04d}.jpg", work / f"{i:04d}_depth.png"
        cv2.imwrite(str(rgb_p), z["rgb"][i][..., ::-1])
        cv2.imwrite(str(depth_p), depth_mm)
        frames.append(Frame(rgb=rgb_p, K=z["K"][i].astype(float), T_wc=z["T_wc"][i].astype(float),
                            depth=depth_p, timestamp=f.timestamp))
    bundle.meta["_original_rgb"] = [str(f.rgb) for f in bundle.frames]
    bundle.meta["mapanything_scale"] = round(float(np.median(z["scale"])), 3)
    bundle.frames = frames
    return bundle
