"""Damage masks -> one surface of the plan (a wall, a floor or a ceiling) with a metric extent, merged across
the frames that saw the same defect. Works on the schema plan (surface ids R1.W3, R1.floor, R1.ceiling).

Per detection:
 1. Frame choice: the sharpest frame in each of N_IMAGES equal slots of the capture, at full resolution
    (LiDAR: decoded from rgb.mp4 by video index; camera tiers: the original photo or video frame), turned
    upright using the camera pose so the detector sees the room the right way up.
 2. Mask -> depth image: the mask's coverage of each depth pixel (area resize, so a 2 px crack still counts
    in a 7.5 px LiDAR depth pixel). Each covered pixel -> a 3-D point (sensor depth for LiDAR, model depth for
    the camera tiers).
 3. Surface: the median point is snapped to the nearest wall plane (within WALL_DIST, inside the wall's
    extent), else the floor or the ceiling of the room it is in. At least ON_SURFACE_FRAC of the mask's points
    must lie within ON_SURFACE_DIST of that plane, or the detection is on furniture and is dropped.
 4. Extent (m^2) = sum over covered depth pixels of coverage x the pixel's footprint on the surface plane,
    z^2 / (fx fy |cos a|), a = angle between the pixel's ray and the surface normal.
 5. Merge: the same class on the same surface within MERGE_DIST is one defect; its extent is the median of
    the views, its interval the wider of a relative term (mask edges, tier scale) and the views' spread."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from fp import contract

N_IMAGES = 30            # sharpest frame in each of 30 equal slots (work order 06: "sharpest N frames spread over")
DETECT_MAX_SIDE = 1280   # px: Grounding DINO resizes to ~800-1333 anyway
WALL_DIST = 0.15         # m: a defect centre this close to a wall plane is on that wall
FLOOR_CEIL = 0.15        # m from the floor (z = 0) or the room's ceiling
ON_SURFACE_DIST = 0.08   # m: a mask point this close to the surface plane lies on it (LiDAR noise + wall smear)
ON_SURFACE_FRAC = 0.5    # ...and at least half of them must: otherwise it is a picture, a towel, furniture
MIN_POINTS = 10          # depth pixels with coverage, else no depth there (sky through a window, too far)
MIN_VALID = 0.5          # fraction of the mask's coverage that must have depth, else the extent is a guess
MERGE_DIST = 0.30        # m: the same defect seen from several frames
CONFIRM_FRAMES = 2       # a detection is looked at again in this many other frames...
CONFIRM_MOVE = 0.25      # m: ...taken from a camera at least this far away (a new viewpoint, not the same image)
CONFIRM_DIST = (0.5, 4.0)  # m from the camera to the defect in a confirming frame
EXTENT_REL = 0.20        # relative half-width of one extent: mask edges +-1-2 px on a 10-20 px wide defect
DEFAULT_WALL_TOP = 3.6   # m: walls are searched up to this height when the room's ceiling was not observed


def pick_frames(paths: list[Path], n: int = N_IMAGES) -> list[int]:
    from fp.ingest.quality import image_stats
    if not paths:
        return []
    sharp = image_stats(paths)["sharpness"]
    return [int(s[np.argmax(sharp[s])]) for s in np.array_split(np.arange(len(paths)), min(n, len(paths)))]


def upright_k(T_pc: np.ndarray) -> int:
    """Number of 90 deg counter-clockwise turns (np.rot90 k) that put plan-down at the bottom of the image.
    T_pc: camera -> plan rotation (3x3); plan-down in camera coords decides it (OpenCV camera, y down)."""
    d = T_pc[:3, :3].T @ np.array([0.0, 0.0, -1.0])
    dx, dy = d[0], d[1]
    if abs(dy) >= abs(dx):
        return 0 if dy > 0 else 2
    return 1 if dx < 0 else 3     # rot90 k=1 turns the image counter-clockwise: its left side goes to the bottom


def detection_image(bundle, i: int, T: np.ndarray, video: Path | None):
    """Full-resolution RGB of frame i, upright. Returns (rgb, k) with k the np.rot90 turns applied."""
    f = bundle.frames[i]
    img = None
    if video is not None and f.video_index is not None:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f.video_index))
        ok, img = cap.read()
        cap.release()
        img = img if ok else None
    if img is None:
        orig = bundle.meta.get("_original_rgb")
        img = cv2.imread(str(orig[i] if orig else f.rgb))
    s = min(1.0, DETECT_MAX_SIDE / max(img.shape[:2]))
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    # camera-tier originals are already upright (ingest applies EXIF / rotation tags); LiDAR video is not
    k = upright_k((T @ f.T_wc)[:3, :3]) if video is not None else 0
    return np.ascontiguousarray(np.rot90(img[:, :, ::-1], k)), k


def _surfaces(plan: dict) -> list[dict]:
    out = []
    for r in plan["rooms"]:
        H = r["ceiling_height"]["value"]
        for w in r["walls"]:
            p0, p1 = np.array(w["p0"], float), np.array(w["p1"], float)
            L = float(np.linalg.norm(p1 - p0))
            if L < 1e-6:
                continue
            u = (p1 - p0) / L
            out.append({"id": w["id"], "room": r["id"], "kind": "wall", "p0": p0, "u": u,
                        "n": np.array([u[1], -u[0], 0.0]), "L": L, "top": H or DEFAULT_WALL_TOP})
    return out


def snap(c: np.ndarray, plan: dict, walls: list[dict] | None = None) -> dict | None:
    """Surface a defect centred at c (plan metres) sits on: {id, room, kind, normal, offset} or None."""
    walls = walls if walls is not None else _surfaces(plan)
    best = None
    for w in walls:
        rel = c[:2] - w["p0"]
        d, a = abs(float(rel @ w["n"][:2])), float(rel @ w["u"])
        if d < WALL_DIST and -0.2 < a < w["L"] + 0.2 and 0.05 < c[2] < w["top"] + 0.1 and (best is None or d < best[0]):
            best = (d, w)
    if best:
        w = best[1]
        return {"id": w["id"], "room": w["room"], "kind": "wall", "normal": w["n"],
                "offset": float(np.r_[w["p0"], 0.0] @ w["n"]), "wall": w}
    for r in plan["rooms"]:
        if not contains_xy(Polygon(r["polygon"]).buffer(0.3), c[0], c[1]):
            continue
        if abs(c[2]) < FLOOR_CEIL:
            return {"id": f"{r['id']}.floor", "room": r["id"], "kind": "floor", "normal": np.array([0, 0, 1.0]),
                    "offset": 0.0}
        H = r["ceiling_height"]["value"]
        if H and abs(c[2] - H) < FLOOR_CEIL:
            return {"id": f"{r['id']}.ceiling", "room": r["id"], "kind": "ceiling",
                    "normal": np.array([0, 0, -1.0]), "offset": -H}
    return None


def mask_geometry(frame, mask: np.ndarray, T: np.ndarray):
    """Mask (bool, at the frame's own image orientation, any resolution) -> (points (n,3) plan, coverage (n,),
    ray directions (n,3) plan, per-pixel area factor z^2/(fx fy) (n,), valid fraction)."""
    D = cv2.imread(str(frame.depth), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    h, w = D.shape
    cov = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
    rgb_w = cv2.imread(str(frame.rgb)).shape[1]
    s = w / rgb_w
    fx, fy, cx, cy = frame.K[0, 0] * s, frame.K[1, 1] * s, frame.K[0, 2] * s, frame.K[1, 2] * s
    total = float(cov.sum())
    ok = (cov > 0.05) & (D > 0.1) & (D < 6.0)
    v, u = np.nonzero(ok)
    z = D[v, u]
    cam = np.stack([(u - cx) / fx * z, (v - cy) / fy * z, z], 1)
    Tp = T @ frame.T_wc
    pts = cam @ Tp[:3, :3].T + Tp[:3, 3]
    rays = cam @ Tp[:3, :3].T
    rays /= np.linalg.norm(rays, axis=1, keepdims=True) + 1e-12
    valid = float(cov[v, u].sum()) / total if total > 0 else 0.0
    return pts, cov[v, u], rays, z * z / (fx * fy), valid


def place(frame, mask: np.ndarray, T: np.ndarray, plan: dict, walls=None) -> dict | None:
    """One mask -> {surface, center, extent_m2, uv points on the surface} or a dict with 'drop' = reason."""
    pts, cov, rays, area_k, valid = mask_geometry(frame, mask, T)
    if len(pts) < MIN_POINTS or valid < MIN_VALID:
        return {"drop": "no_depth"}
    core = cov >= 0.25
    c = np.median(pts[core] if core.sum() >= MIN_POINTS else pts, 0)
    surf = snap(c, plan, walls)
    if surf is None:
        return {"drop": "no_surface"}
    dist = np.abs(pts @ surf["normal"] - surf["offset"])
    on = dist < ON_SURFACE_DIST
    if (cov[on].sum() / cov.sum()) < ON_SURFACE_FRAC:
        return {"drop": "off_surface"}
    cosa = np.clip(np.abs(rays @ surf["normal"]), 0.2, 1.0)        # grazing views capped at 5x
    # pixels with no depth are assumed to be like the rest (valid >= MIN_VALID); mask spill off the plane is not counted
    extent = float((cov * area_k / cosa)[on].sum() / max(valid, 1e-9))
    P = pts[on]
    if surf["kind"] == "wall":
        w = surf["wall"]
        uv = np.stack([(P[:, :2] - w["p0"]) @ w["u"], P[:, 2]], 1)
    else:
        uv = P[:, :2]
    return {"surface": surf, "center": c, "extent_m2": extent, "uv": uv}


def assess(bundle, plan: dict, out: Path, cache_dir: Path | None = None, backend: str = "local",
           use_cache: bool = True) -> tuple[list[dict], dict]:
    """Detect damage on the sharpest frames, place each detection on a surface, merge repeats.
    Returns (schema Damage dicts with internal keys prefixed '_', counts)."""
    T = np.array(plan["frame"]["T_plan_world"], float)
    frames = bundle.frames
    if not frames or any(f.depth is None or f.T_wc is None for f in frames):
        return [], {"frames": 0}
    video = Path(plan["source"]["path"]) / "rgb.mp4" if plan["tier"] == "lidar" else None
    video = video if video is not None and video.is_file() else None
    idx = pick_frames([f.rgb for f in frames])
    imgs, turns = zip(*[detection_image(bundle, i, T, video) for i in idx]) if idx else ((), ())
    if backend == "modal":
        dets = _detect_modal(list(imgs))
    else:
        from fp.damage.detect_local import detect
        dets = detect(list(imgs), cache_dir, use_cache)
    walls = _surfaces(plan)
    counts = {"frames": len(idx), "detections": 0, "no_depth": 0, "no_surface": 0, "off_surface": 0}
    raw = []
    for fi, (i, img, k, d) in enumerate(zip(idx, imgs, turns, dets)):
        for b, s, cls, m in zip(d["boxes"], d["scores"], d["classes"], d["masks"]):
            counts["detections"] += 1
            p = place(frames[i], np.rot90(m, -k), T, plan, walls)     # mask back to the frame's orientation
            if "drop" in p:
                counts[p["drop"]] += 1
                continue
            raw.append({**p, "class": str(cls), "score": float(s), "img": fi, "box": b, "mask": m})
    raw, unconfirmed = _confirm(raw, bundle, idx, T, plan, walls, video, cache_dir, use_cache, backend, counts)
    items = _merge(raw, plan["tier"])
    (out / "damage").mkdir(parents=True, exist_ok=True)
    for old in (out / "damage").glob("D*.jpg"):
        old.unlink()
    for it in items:
        r = it["_best"]
        cv2.imwrite(str(out / "damage" / f"{it['id']}.jpg"), _evidence(imgs[r["img"]], r["mask"], r["box"]))
        it["evidence_image"] = f"damage/{it['id']}.jpg"
    counts["items"] = len(items)
    counts["unconfirmed"] = []
    for k, r in enumerate(unconfirmed):
        name = f"damage/unconfirmed_{k + 1}.jpg"
        cv2.imwrite(str(out / name), _evidence(imgs[r["img"]], r["mask"], r["box"]))
        counts["unconfirmed"].append({"class": r["class"], "surface_id": r["surface"]["id"],
                                      "score": round(r["score"], 3), "evidence_image": name})
    return items, counts


def _sees(f, T, X, img_wh):
    """Does frame f see plan point X in its image, unoccluded, at a usable distance?"""
    Tcp = np.linalg.inv(T @ f.T_wc)
    C = Tcp[:3, :3] @ X + Tcp[:3, 3]
    if not CONFIRM_DIST[0] < C[2] < CONFIRM_DIST[1]:
        return False
    u, v = (f.K @ C)[:2] / C[2]
    if not (0.1 * img_wh[0] < u < 0.9 * img_wh[0] and 0.1 * img_wh[1] < v < 0.9 * img_wh[1]):
        return False
    D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
    d = D[int(v * D.shape[0] / img_wh[1]), int(u * D.shape[1] / img_wh[0])] / 1000.0
    return not (0 < d < C[2] - 0.15)


def _confirm(raw, bundle, idx, T, plan, walls, video, cache_dir, use_cache, backend, counts):
    """Keep a detection only if the detector finds the same class on the same surface (within MERGE_DIST)
    in another frame from a different viewpoint: a stain or crack stays put when you look again, a reflection,
    a shadow edge or a lighting artefact moves or disappears. Detections already seen in two sampled frames
    count as confirmed. Returns (confirmed, unconfirmed)."""
    from fp.damage.detect_local import detect
    frames = bundle.frames
    img_wh = cv2.imread(str(frames[0].rgb)).shape[1::-1]
    cams = np.array([(T @ f.T_wc)[:3, 3] for f in frames])
    same = lambda a, b: (a["class"] == b["class"] and a["surface"]["id"] == b["surface"]["id"]
                         and np.linalg.norm(a["center"] - b["center"]) < MERGE_DIST)
    confirmed, unconfirmed, extra = [], [], []
    for d in raw:
        if any(same(d, e) and e["img"] != d["img"] for e in raw):
            confirmed.append(d)
            continue
        src = cams[idx[d["img"]]]
        cand = [j for j in np.argsort(np.abs(np.arange(len(frames)) - idx[d["img"]]))
                if np.linalg.norm(cams[j] - src) >= CONFIRM_MOVE and _sees(frames[j], T, d["center"], img_wh)]
        picked = []
        for j in cand:              # nearest in time first, each one a further CONFIRM_MOVE from the others
            if all(np.linalg.norm(cams[j] - cams[k]) >= CONFIRM_MOVE for k in picked):
                picked.append(int(j))
            if len(picked) == CONFIRM_FRAMES:
                break
        ok = False
        for j in picked:
            img, k = detection_image(bundle, j, T, video)
            det = (_detect_modal([img]) if backend == "modal" else detect([img], cache_dir, use_cache))[0]
            counts["confirm_frames"] = counts.get("confirm_frames", 0) + 1
            for b, s, cls, m in zip(det["boxes"], det["scores"], det["classes"], det["masks"]):
                if str(cls) != d["class"]:
                    continue
                p = place(frames[j], np.rot90(m, -k), T, plan, walls)
                if "drop" not in p and same(d, {**p, "class": str(cls)}):
                    ok = True
                    p.update({"class": str(cls), "score": float(s), "img": d["img"], "box": d["box"], "mask": d["mask"]})
                    extra.append(p)              # the second view joins the merge (its extent counts)
                    break
            if ok:
                break
        (confirmed if ok else unconfirmed).append(d)
    return confirmed + extra, unconfirmed


def _merge(raw: list[dict], tier: str) -> list[dict]:
    groups = []
    for d in sorted(raw, key=lambda d: -d["score"]):
        g = next((g for g in groups if g[0]["class"] == d["class"] and g[0]["surface"]["id"] == d["surface"]["id"]
                  and np.linalg.norm(g[0]["center"] - d["center"]) < MERGE_DIST), None)
        (g.append(d) if g else groups.append([d]))
    rel = EXTENT_REL + 2 * contract.SCALE_REL[tier]          # area scales with the square of a scale error
    items = []
    for k, g in enumerate(groups):
        ex = np.array([d["extent_m2"] for d in g])
        v = float(np.median(ex))
        half = max(rel * v, 0.5 * float(ex.max() - ex.min()))
        uv = np.concatenate([d["uv"] for d in g])
        u0, v0 = np.percentile(uv, 5, axis=0)
        u1, v1 = np.percentile(uv, 95, axis=0)
        best = g[0]
        items.append({
            "id": f"D{k + 1}", "class": best["class"], "surface_id": best["surface"]["id"],
            "position": [round(float(x), 3) for x in np.median([d["center"] for d in g], 0)],
            "extent_m2": contract.measure(v, half, "m2", f"mask area projected on the surface plane, median of "
                                                         f"{len(g)} view(s)", observed=True),
            "bbox_on_surface": {"u0": round(float(u0), 3), "v0": round(float(v0), 3), "u1": round(float(u1), 3),
                                "v1": round(float(v1), 3)},
            "evidence_image": None, "score": round(float(best["score"]), 3),
            "_views": len(g), "_room": best["surface"]["room"], "_kind": best["surface"]["kind"], "_best": best})
    return items


def to_schema(items: list[dict]) -> list[dict]:
    return [{k: v for k, v in it.items() if not k.startswith("_")} for it in items]


def _detect_modal(images: list[np.ndarray]) -> list[dict]:
    """SAM 3 on Modal (optional, paid): same output shape as detect_local.detect."""
    import modal

    from fp.damage.detect_local import MAX_MASK_FRAC, label_class
    from fp.damage.detect_modal import app, detect
    jpgs = [cv2.imencode(".jpg", im[:, :, ::-1])[1].tobytes() for im in images]
    with modal.enable_output(), app.run():
        found = detect.remote(jpgs)
    out = [{"boxes": [], "scores": [], "classes": [], "masks": []} for _ in images]
    for d in found:
        m = cv2.imdecode(np.frombuffer(d["mask_png"], np.uint8), cv2.IMREAD_GRAYSCALE) > 0
        cls = label_class(d["prompt"])
        if cls and m.mean() <= MAX_MASK_FRAC:
            o = out[d["image"]]
            o["boxes"].append(d["box"]); o["scores"].append(d["score"]); o["classes"].append(cls); o["masks"].append(m)
    return out


def _evidence(rgb: np.ndarray, mask: np.ndarray, box) -> np.ndarray:
    """The detection's area of the (upright, full-resolution) frame, the mask outlined in red. BGR."""
    im = np.ascontiguousarray(rgb[:, :, ::-1])
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(im, cs, -1, (0, 0, 255), 2)
    x0, y0, x1, y1 = box
    pad = 0.4 * max(x1 - x0, y1 - y0, 60)
    return im[max(0, int(y0 - pad)):int(y1 + pad), max(0, int(x0 - pad)):int(x1 + pad)]
