"""Damage masks (SAM 3) -> 3D points -> one surface of the plan (a wall, a floor or a ceiling),
merged across the frames that saw the same defect. A detection that lands on no surface (furniture,
clutter, a view out of a window) is dropped and counted, because the brief asks for damage tied to
surfaces."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import Point, Polygon

from fp.bundle import CaptureBundle
from fp.geometry.align import to_plan
from fp.ingest.quality import image_stats

N_IMAGES = 30            # sharpest frame in each of 30 equal slots
WALL_DIST = 0.15         # m: a defect this close to a wall plane is on that wall
FLOOR_CEIL = 0.15        # m from the floor (z = 0) or the ceiling
MERGE_DIST = 0.30        # m: the same defect seen from several frames
DETECT_MAX_SIDE = 1280   # px sent to SAM 3


def pick_frames(paths: list[Path], n: int = N_IMAGES) -> list[int]:
    sharp = image_stats(paths)["sharpness"]
    return [int(s[np.argmax(sharp[s])]) for s in np.array_split(np.arange(len(paths)), min(n, len(paths)))]


def to_depth_image(mask: np.ndarray, proc_hw: tuple[int, int]) -> np.ndarray:
    """A mask on the original photo -> the image MapAnything predicted depth for. MapAnything picks the
    target size closest to the photo's aspect ratio, so a plain resize maps it (checked on scan 48018559:
    mean grey difference 1.8-3.7, against 2.4-5.1 for resize-and-centre-crop)."""
    return cv2.resize(mask, (proc_hw[1], proc_hw[0]), interpolation=cv2.INTER_NEAREST)


def mask_points(frame, mask: np.ndarray, T_plan: np.ndarray) -> np.ndarray:
    """Plan-frame 3D points of the masked pixels that have depth."""
    D = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED) / 1000.0
    rgb_w = cv2.imread(str(frame.rgb)).shape[1]
    m = to_depth_image(mask, D.shape) if mask.shape != D.shape else mask
    v, u = np.nonzero((m > 0) & (D > 0.1) & (D < 5.0))
    s = D.shape[1] / rgb_w
    fx, fy, cx, cy = frame.K[0, 0] * s, frame.K[1, 1] * s, frame.K[0, 2] * s, frame.K[1, 2] * s
    z = D[v, u]
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    return to_plan(T_plan, cam @ frame.T_wc[:3, :3].T + frame.T_wc[:3, 3])


def snap(c: np.ndarray, plan: dict) -> dict | None:
    """Which surface a defect centred at c (plan metres) sits on."""
    rooms = plan["rooms"]
    H = rooms[0].get("ceiling_h_m")
    best = None
    for r in rooms:
        for w in r["walls"]:
            ax = 0 if w["axis"] == "x" else 1
            lo, hi = sorted([w["p0"][1 - ax], w["p1"][1 - ax]])
            d = abs(c[ax] - w["coord"])
            if d < WALL_DIST and lo - 0.2 < c[1 - ax] < hi + 0.2 and (best is None or d < best[0]):
                best = (d, r, w, c[1 - ax] - min(w["p0"][1 - ax], w["p1"][1 - ax]))
    if best and c[2] >= 0.1:
        _, r, w, along = best
        return {"kind": "wall", "room": r["id"], "wall": w["id"], "along_wall_m": round(float(along), 2)}
    inside = [r for r in rooms if Polygon(r["polygon"]).buffer(0.3).contains(Point(c[0], c[1]))]
    if not inside:
        return None
    if abs(c[2]) < FLOOR_CEIL:
        return {"kind": "floor", "room": inside[0]["id"]}
    if H and abs(c[2] - H) < FLOOR_CEIL:
        return {"kind": "ceiling", "room": inside[0]["id"]}
    return None


def assess(bundle: CaptureBundle, plan: dict, out: Path) -> tuple[list[dict], dict]:
    """Run SAM 3 on the sharpest frames, place each detection on a surface, merge repeats.
    Returns (damage items for plan.json, counts)."""
    import modal

    from fp.damage.detect_modal import app, detect
    originals = [Path(p) for p in bundle.meta.get("_original_rgb", [f.rgb for f in bundle.frames])]
    idx = pick_frames(originals)
    images = []
    for i in idx:
        im = cv2.imread(str(originals[i]))
        s = min(1.0, DETECT_MAX_SIDE / max(im.shape[:2]))
        images.append(cv2.imencode(".jpg", cv2.resize(im, None, fx=s, fy=s) if s < 1 else im)[1].tobytes())
    print(f"SAM 3 on Modal: {len(images)} frames ...")
    with modal.enable_output(), app.run():
        found = detect.remote(images)
    T = np.array(plan["T_plan_world"])
    (out / "damage").mkdir(parents=True, exist_ok=True)
    raw, counts = [], {"detections": len(found), "no_depth": 0, "no_surface": 0}
    for d in found:
        frame = bundle.frames[idx[d["image"]]]
        mask = cv2.imdecode(np.frombuffer(d["mask_png"], np.uint8), cv2.IMREAD_GRAYSCALE)
        pts = mask_points(frame, mask, T)
        if len(pts) < 10:
            counts["no_depth"] += 1
            continue
        c = np.median(pts, 0)
        surf = snap(c, plan)
        if surf is None:
            counts["no_surface"] += 1
            continue
        lo, hi = np.percentile(pts, [5, 95], axis=0)
        raw.append({"type": d["prompt"], "score": d["score"], "center": c, "size_m": round(float(np.max(hi - lo)), 2),
                    "surface": surf, "image": d["image"], "box": d["box"], "mask": mask})
    items = []
    for d in sorted(raw, key=lambda d: -d["score"]):  # merge: same type, same surface, close together
        same = next((it for it in items if it["type"] == d["type"] and it["_key"] == _key(d["surface"])
                     and np.linalg.norm(it["_c"] - d["center"]) < MERGE_DIST), None)
        if same:
            same["views"] += 1
            continue
        k = len(items) + 1
        crop = _evidence(originals[idx[d["image"]]], d["mask"], d["box"])
        cv2.imwrite(str(out / "damage" / f"D{k}.jpg"), crop)
        items.append({"id": f"D{k}", "type": d["type"], "score": d["score"], "views": 1,
                      "room": d["surface"]["room"], "surface": _key(d["surface"]), **d["surface"],
                      "position_m": [round(float(v), 2) for v in d["center"]], "height_m": round(float(d["center"][2]), 2),
                      "size_m": d["size_m"], "evidence": f"damage/D{k}.jpg", "_key": _key(d["surface"]), "_c": d["center"]})
    for it in items:
        it.pop("_key"); it.pop("_c")
    counts["items"] = len(items)
    return items, counts


def _key(surf: dict) -> str:
    return f"{surf['room']} {surf['wall']}" if surf["kind"] == "wall" else f"{surf['room']} {surf['kind']}"


def _evidence(path: Path, mask: np.ndarray, box) -> np.ndarray:
    """The detection's area of the original photo, the mask outlined in red."""
    im = cv2.imread(str(path))
    s = min(1.0, DETECT_MAX_SIDE / max(im.shape[:2]))
    im = cv2.resize(im, None, fx=s, fy=s) if s < 1 else im
    m = cv2.resize(mask, (im.shape[1], im.shape[0]), interpolation=cv2.INTER_NEAREST)
    cs, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
    x0, y0, x1, y1 = box
    pad = 0.4 * max(x1 - x0, y1 - y0, 60)
    return im[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)]
