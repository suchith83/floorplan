"""Camera-only input (photos, video) -> frames with metric depth, intrinsics and camera poses.

The model is MapAnything (Apache-2.0 weights): images in, per-view metric depth, intrinsics and poses
out, all views in ONE shared world frame. It runs on this machine by default (CUDA, else Apple MPS, else
CPU); `--backend modal` sends the same images to a Modal GPU instead (optional, never needed).

Three jobs on top of the model:
 1. Deterministic cache, one entry per model pass. Key = sha1(model id + parameters + the content id of
    every frame in the pass, which is a hash of the source bytes, fp/ingest, + intrinsics priors). Value =
    one .npz of the model's raw output: depth, conf, mask, K, T_wc, the processed RGB and the metric scale.
    A cache hit replays exactly what the live pass produced; everything after the model (chunk merge,
    fusion, geometry) always runs. `--no-cache` forces live passes. The cache lives in
    <out>/../_cache/recon/ (scripts/cache_sync.py publishes/fetches it).
 2. Chunks. The model attends over all views at once, so memory grows with the view count. Long videos
    run in overlapping chunks of <= CHUNK views. Chunk k is put into chunk k-1's frame using the frames
    they share: scale = median ratio of their depths at the same pixel (same image, two predictions);
    rotation and translation = the average over shared frames of the move between their two camera poses.
    Each chunk also predicted its own metric scale, so the final scale is the median of every chunk's
    vote, not chunk 1's alone.
 3. Photos: ALL rooms' photos go into one joint pass, so every room lands in one frame and the rooms
    stitch. Rooms are linked by a photo saved in both rooms' folders (identical bytes) or by enough
    feature matches between their photos. A room linked only by matches is checked: if its matched
    points disagree in 3D, it is re-placed by a Sim(3) on those matches. A room with no link at all
    is not placed (its frames are dropped from the plan) and named in a warning, never overlapped.

Gravity: the cameras' mean "up" is only a guess (a phone may be held sideways or tilted), so each of the four in-image directions is tried and refined onto the horizontal surfaces
(floor, ceiling, table tops); the winner has the floor below the cameras. The result is checked against a
plane fitted to the floor.
Scale: metric scale comes from the model alone (it learned the size of things: doors, tiles, furniture).
It is sanity-checked against priors (handheld camera 1.0-1.8 m above the floor, ceiling 2.3-3.2 m); a
disagreement widens every interval by the relative gap (contract.fill_from_geometry), never rescales."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np

from fp.bundle import CaptureBundle, Frame, Recon
from fp.contract import StageError

MODEL_ID = "facebook/map-anything-apache"
# Everything that changes the model's output is part of the cache key.
MODEL_PARAMS = {"resolution_set": 518, "amp": "bf16", "apply_mask": True, "mask_edges": True,
                "intrinsics_prior": True, "version": 1}
CHUNK = 24            # views per model pass: 24 views = 37 s, 9.8 GB MPS peak on a 16 GB M5; 40 views swaps (STATUS 05)
OVERLAP = 6           # frames shared by consecutive video chunks; the Sim(3) between chunks is fitted on them
CONF_PERCENTILE = 10  # drop the least confident 10 % of pixels (sky-like voids, reflections, far clutter)
SIM3_SAMPLES = 4000   # 3D point pairs per Sim(3) fit / residual check
SIM3_TRIM = 0.5       # refit on the best half of the pairs: robust to bad depth on one side
MERGE_WARN_M = 0.25   # m, median 3D gap between two chunks' predictions of their shared frames after the merge
                      # above which the run warns: larger than a wall band, so walls from the two chunks double
# Photo rooms
MIN_MATCHES = 40      # RANSAC inliers between two photos of different rooms that count as "same view"
MATCH_SIDE = 800      # px, longest side for feature matching
MATCH_AGREE_M = 0.25  # m, median 3D disagreement of matched points above which a room is re-placed
# Gravity and scale priors
CAM_H = (1.0, 1.8)    # m: a handheld phone, standing adult
CEILING_H = (2.3, 3.2)  # m: residential ceilings (low 2.3 m modern flats to 3.2 m older buildings)


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------- model passes

_MODEL = None


def _model(dev: str):
    global _MODEL
    if _MODEL is None:
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # an op MPS lacks runs on CPU instead of failing
        from mapanything.models import MapAnything
        _MODEL = MapAnything.from_pretrained(MODEL_ID).to(dev).eval()
    return _MODEL


def _infer_local(paths: list[Path], K_prior: list) -> dict:
    """One joint model pass on this machine. K_prior: per image a 3x3 (at the image's resolution) or None."""
    import torch
    from mapanything.utils.image import preprocess_inputs
    from PIL import Image
    dev = device()
    model = _model(dev)
    views = []
    for p, K in zip(paths, K_prior):
        v = {"img": np.asarray(Image.open(p).convert("RGB"))}
        if K is not None and MODEL_PARAMS["intrinsics_prior"]:
            v["intrinsics"] = torch.tensor(np.asarray(K, np.float32))
        views.append(v)
    views = preprocess_inputs(views, resolution_set=MODEL_PARAMS["resolution_set"])
    with torch.no_grad():
        preds = model.infer(views, memory_efficient_inference=True, use_amp=True, amp_dtype=MODEL_PARAMS["amp"],
                            apply_mask=MODEL_PARAMS["apply_mask"], mask_edges=MODEL_PARAMS["mask_edges"])
    out = _stack(preds)
    if dev == "mps":
        torch.mps.empty_cache()
    return out


def _stack(preds) -> dict:
    s = lambda k: np.stack([p[k][0].float().cpu().numpy() for p in preds])
    rgb = s("img_no_norm")
    rgb = rgb * 255 if rgb.max() <= 1.0 else rgb
    return {"depth": s("depth_z")[..., 0].astype(np.float16), "conf": s("conf").astype(np.float16),
            "mask": s("mask")[..., 0] > 0.5, "K": s("intrinsics").astype(np.float32),
            "T_wc": s("camera_poses").astype(np.float64), "rgb": rgb.clip(0, 255).astype(np.uint8),
            "scale": np.array([float(p["metric_scaling_factor"].float().reshape(-1)[0]) for p in preds])}


def _infer_modal(paths: list[Path], K_prior: list) -> dict:
    """The same pass on a Modal GPU (opt-in). The Modal function takes images only, so no intrinsics prior."""
    import modal

    from fp.recon.mapanything_modal import app, infer
    images = []
    for p in paths:
        im = cv2.imread(str(p))
        images.append(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes())
    with modal.enable_output(), app.run():
        raw = infer.remote(images)
    z = dict(np.load(io.BytesIO(raw)))
    return {k: z[k] for k in ("depth", "conf", "mask", "K", "T_wc", "rgb", "scale")}


def _infer(paths, K_prior, backend: str) -> dict:
    try:
        return _infer_modal(paths, K_prior) if backend == "modal" else _infer_local(paths, K_prior)
    except Exception as e:   # no weights offline, out of memory, Modal down: a warning, not a lost plan
        raise StageError("recon", f"MapAnything failed on {len(paths)} views ({backend}): {type(e).__name__}: {e}") from e


# ----------------------------------------------------------------------------- Sim(3)

def umeyama(A: np.ndarray, B: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares s, R, t with B ~ s R A + t (Umeyama 1991). A, B: (n,3)."""
    ma, mb = A.mean(0), B.mean(0)
    a, b = A - ma, B - mb
    U, S, Vt = np.linalg.svd(b.T @ a / len(A))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / a.var(0).sum())
    return s, R, mb - s * R @ ma


def robust_sim3(A, B, trim: float = SIM3_TRIM, iters: int = 3):
    """Umeyama, then refit on the `trim` fraction of pairs that agree best (twice). Returns s, R, t and the
    median residual (m) on the kept pairs."""
    keep = np.arange(len(A))
    for _ in range(iters):
        s, R, t = umeyama(A[keep], B[keep])
        r = np.linalg.norm(A @ R.T * s + t - B, axis=1)
        keep = np.argsort(r)[: max(10, int(trim * len(A)))]
    return s, R, t, float(np.median(r[keep]))


def _sim3_matrix(s, R, t) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = s * R, t
    return T


def _apply_sim3(z: dict, idx, s, R, t) -> None:
    """Move views `idx` of a pass by x -> s R x + t: poses rotate and translate, depth scales by s."""
    for i in idx:
        T = z["T_wc"][i]
        Tn = np.eye(4)
        Tn[:3, :3] = R @ T[:3, :3]
        Tn[:3, 3] = s * R @ T[:3, 3] + t
        z["T_wc"][i] = Tn
        z["depth"][i] = (z["depth"][i].astype(np.float32) * s).astype(np.float16)


def view_points(z: dict, i: int, uv: np.ndarray | None = None):
    """World points of view i at pixels uv (n,2 int, processed-image coords), or all valid pixels.
    Returns (points (n,3), valid (n,))."""
    D = z["depth"][i].astype(np.float32)
    ok = z["mask"][i] & (D > 0)
    if uv is None:
        v, u = np.nonzero(ok)
        valid = np.ones(len(u), bool)
    else:
        u, v = uv[:, 0], uv[:, 1]
        valid = ok[v, u]
    K, T = z["K"][i], z["T_wc"][i]
    d = D[v, u]
    cam = np.stack([(u - K[0, 2]) / K[0, 0] * d, (v - K[1, 2]) / K[1, 1] * d, d], 1)
    return cam @ T[:3, :3].T + T[:3, 3], valid


def _shared_view_pairs(za, ia, zb, ib, rng) -> tuple[np.ndarray, np.ndarray]:
    """The same image in two passes: the same pixel gives a 3D point in each pass's world."""
    ok = za["mask"][ia] & zb["mask"][ib] & (za["depth"][ia] > 0) & (zb["depth"][ib] > 0)
    v, u = np.nonzero(ok)
    sel = rng.choice(len(u), min(len(u), SIM3_SAMPLES // 2), replace=False) if len(u) else []
    uv = np.stack([u[sel], v[sel]], 1)
    return view_points(za, ia, uv)[0], view_points(zb, ib, uv)[0]


# ----------------------------------------------------------------------------- chunked passes

def chunks(n: int, size: int = CHUNK, overlap: int = OVERLAP) -> list[list[int]]:
    """[0..size), [size-overlap..2size-overlap), ...; the last chunk is pulled back to be full."""
    if n <= size:
        return [list(range(n))]
    step = size - overlap
    starts = list(range(0, n - size, step)) + [n - size]
    return [list(range(s, s + size)) for s in starts]


def _rotation_mean(Rs: list[np.ndarray]) -> np.ndarray:
    """Chordal mean of rotations: the rotation nearest (Frobenius) to their sum."""
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    return U @ D @ Vt


def _relative_sim3(z: dict, zi: list[int], prev: dict, pi: list[int]) -> tuple[float, np.ndarray, np.ndarray]:
    """Sim(3) taking pass `z` into pass `prev`'s frame from frames both passes saw (z view zi[k] = prev view pi[k]).
    Scale: median over shared pixels of depth_prev / depth_z (the same pixel of the same image, predicted
    twice). Unlike a least-squares point fit, a ratio median is not biased towards 0 by noisy pairs, so the
    error doesn't compound along a chain of chunks. Rotation/translation: average of the per-frame moves
    R_prev R_z^T and c_prev - s R c_z between the two camera poses of each shared frame."""
    ratios, Rs = [], []
    for a, b in zip(zi, pi):
        ok = z["mask"][a] & prev["mask"][b] & (z["depth"][a] > 0) & (prev["depth"][b] > 0)
        ratios.append(prev["depth"][b][ok].astype(np.float32) / z["depth"][a][ok].astype(np.float32))
        Rs.append(prev["T_wc"][b][:3, :3] @ z["T_wc"][a][:3, :3].T)
    r = np.concatenate(ratios)
    if len(r) < 100:
        raise StageError("recon", "Two video chunks share too little valid depth to be merged.")
    s = float(np.median(r))
    R = _rotation_mean(Rs)
    t = np.median([prev["T_wc"][b][:3, 3] - s * R @ z["T_wc"][a][:3, 3] for a, b in zip(zi, pi)], axis=0)
    return s, R, t


def _merge_chunks(parts: list[tuple[list[int], dict]], n: int) -> tuple[dict, list[dict], float]:
    """Chain chunk k onto chunk k-1 (already in chunk 0's frame) by the Sim(3) from their shared frames,
    then rescale everything by the median of the chunks' metric-scale votes. A frame in two chunks keeps
    the earlier chunk's prediction. Returns (arrays, per-merge stats, the final metric correction)."""
    rng = np.random.default_rng(0)
    first_idx, z0 = parts[0]
    out = {k: [None] * n for k in z0}
    stats = []
    votes = [1.0]   # chunk k (scaled by s_k into chunk 0's units) says metric = chunk-0 units / s_k
    prev_idx, prev = first_idx, z0
    for k, (idx, z) in enumerate(parts):
        if k > 0:
            shared = sorted(set(idx) & set(prev_idx))
            s, R, t = _relative_sim3(z, [idx.index(g) for g in shared], prev, [prev_idx.index(g) for g in shared])
            _apply_sim3(z, range(len(idx)), s, R, t)
            res = [np.median(np.linalg.norm(a - b, axis=1)) for a, b in
                   (_shared_view_pairs(z, idx.index(g), prev, prev_idx.index(g), rng) for g in shared)]
            votes.append(votes[-1] / s)
            stats.append({"chunk": k + 1, "views": [idx[0], idx[-1]], "shared_frames": len(shared), "scale": round(s, 4),
                          "median_residual_m": round(float(np.median(res)), 4)})
        for j, g in enumerate(idx):
            if out["depth"][g] is None:
                for key in out:
                    out[key][g] = z[key][j]
        prev_idx, prev = idx, z
    zz = {k: np.stack(v) for k, v in out.items()}
    corr = float(np.median(votes))
    _apply_sim3(zz, range(n), corr, np.eye(3), np.zeros(3))
    return zz, stats, corr


# ----------------------------------------------------------------------------- cache

def pass_key(ids: list[str], K_prior: list) -> str:
    """Cache key of one model pass: model id + parameters + each frame's content id + intrinsics prior."""
    h = hashlib.sha1(json.dumps({"model": MODEL_ID, **MODEL_PARAMS}, sort_keys=True).encode())
    for i, K in zip(ids, K_prior):
        h.update(i.encode())
        h.update(b"none" if K is None else np.asarray(K, np.float64).round(3).tobytes())
    return h.hexdigest()


def _pass(paths, ids, Kp, cache_dir: Path, backend: str, use_cache: bool) -> tuple[dict, bool]:
    """One model pass, replayed from the cache when present. Returns (raw outputs, from_cache)."""
    path = Path(cache_dir) / "recon" / f"{pass_key(ids, Kp)}.npz"
    if use_cache and path.is_file():
        return dict(np.load(path, allow_pickle=False)), True
    print(f"recon: MapAnything ({backend}, {device() if backend == 'local' else 'gpu'}) on {len(paths)} views ...")
    z = _infer(paths, Kp, backend)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **z)
    tmp.replace(path)
    return z, False


def _predict(bundle: CaptureBundle, cache_dir: Path, backend: str, use_cache: bool) -> tuple[dict, dict]:
    """All passes for this bundle (cached or live), chunks merged. Returns (per-frame arrays, info)."""
    ids = bundle.meta.get("_source_ids")
    if not ids or len(ids) != len(bundle.frames):
        raise StageError("recon", "Frames carry no content ids; cannot build a deterministic cache key.")
    paths = [f.rgb for f in bundle.frames]
    Kp = bundle.meta.get("_K_prior") or [None] * len(paths)
    groups = bundle.meta.get("_recon_groups") or [list(range(len(paths)))]
    n = len(paths)
    merged = {}
    chunk_stats, corrections, keys, hits = [], [], [], 0
    for g in groups:   # photos: one group per set of linked rooms; video: one group (chunked)
        parts = []
        for c in chunks(len(g)):
            idx = [g[j] for j in c]
            z, hit = _pass([paths[i] for i in idx], [ids[i] for i in idx], [Kp[i] for i in idx], cache_dir, backend, use_cache)
            hits += hit
            keys.append(pass_key([ids[i] for i in idx], [Kp[i] for i in idx]))
            parts.append((idx, z))
        if len(parts) > 1:
            z, st, corr = _merge_chunks([([g.index(i) for i in idx], zz) for idx, zz in parts], len(g))
            chunk_stats += st
            corrections.append(round(corr, 4))
        else:
            z = parts[0][1]
        for key_ in z:
            merged.setdefault(key_, [None] * n)
            for j, i in enumerate(g):
                merged[key_][i] = z[key_][j]
    if hits:
        print(f"recon: {hits}/{len(keys)} model passes replayed from {Path(cache_dir) / 'recon'}")
    # frames outside every group (should not happen) get empty predictions
    shapes = {k: next(v for v in merged[k] if v is not None) for k in merged}
    z = {k: np.stack([v if v is not None else np.zeros_like(shapes[k]) for v in merged[k]]) for k in merged}
    info = {"backend": backend, "model": MODEL_ID, "views": n, "passes": len(keys), "chunk_merges": chunk_stats,
            "metric_vote_correction": corrections, "cache_keys": keys}
    return z, info


# ----------------------------------------------------------------------------- photo rooms

def _sift(img):
    """SIFT keypoints (pixel coords of `img`) and descriptors. img: a path or an RGB array."""
    g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE) if isinstance(img, (str, Path)) else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    s = MATCH_SIDE / max(g.shape)
    g = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    kp, des = cv2.SIFT_create(2000).detectAndCompute(g, None)
    return np.array([k.pt for k in kp], np.float32) / min(s, 1.0), des


def match_pair(fa, fb) -> tuple[np.ndarray, np.ndarray]:
    """SIFT + ratio test + fundamental-matrix RANSAC -> inlier pixel pairs (in the cached image coords)."""
    (pa, da), (pb, db) = fa, fb
    if da is None or db is None or len(da) < 8 or len(db) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    m = cv2.BFMatcher().knnMatch(da, db, k=2)
    good = [a for a, b in (x for x in m if len(x) == 2) if a.distance < 0.75 * b.distance]
    if len(good) < 8:
        return np.zeros((0, 2)), np.zeros((0, 2))
    A, B = pa[[g.queryIdx for g in good]], pb[[g.trainIdx for g in good]]
    _, inl = cv2.findFundamentalMat(A, B, cv2.FM_RANSAC, 2.0, 0.999)
    if inl is None:
        return np.zeros((0, 2)), np.zeros((0, 2))
    inl = inl.ravel().astype(bool)
    return A[inl], B[inl]


def room_links(bundle: CaptureBundle) -> tuple[dict, list[dict]]:
    """Which rooms are tied together, and how. Shared photo (identical bytes in both folders) = a link;
    otherwise the best feature-matched photo pair between two rooms with >= MIN_MATCHES inliers = a link.
    Returns ({room: component id}, [link dicts])."""
    rooms_of = bundle.meta.get("_frame_room_sets") or [[r] if r else [] for r in bundle.meta.get("_frame_rooms", [])]
    rooms = sorted({r for rs in rooms_of for r in rs})
    links = []
    for i, rs in enumerate(rooms_of):
        for a in rs:
            for b in rs:
                if a < b:
                    links.append({"rooms": [a, b], "how": "shared photo", "frames": [i, i]})
    linked = {tuple(l["rooms"]) for l in links}
    feats = None
    for ai, a in enumerate(rooms):
        for b in rooms[ai + 1:]:
            if (a, b) in linked:
                continue
            feats = feats or {}
            fa = [i for i, rs in enumerate(rooms_of) if a in rs]
            fb = [i for i, rs in enumerate(rooms_of) if b in rs]
            best = (0, None)
            for i in fa:
                for j in fb:
                    for k in (i, j):
                        if k not in feats:
                            feats[k] = _sift(bundle.frames[k].rgb)
                    n = len(match_pair(feats[i], feats[j])[0])
                    if n > best[0]:
                        best = (n, (i, j))
            if best[0] >= MIN_MATCHES:
                links.append({"rooms": [a, b], "how": "feature matches", "frames": list(best[1]), "inliers": best[0]})
    comp = {r: r for r in rooms}
    find = lambda r: r if comp[r] == r else find(comp[r])
    for l in links:
        comp[find(l["rooms"][0])] = find(l["rooms"][1])
    return {r: find(r) for r in rooms}, links


def _check_matched_rooms(z: dict, bundle: CaptureBundle, links: list[dict], warnings: list[str]) -> list[dict]:
    """A room tied to the others only by feature matches: its photos went through the same joint pass, so
    the matched pixels should land on the same 3D points. If they disagree by more than MATCH_AGREE_M,
    move that room's views by the Sim(3) that brings the matches together."""
    rooms_of = bundle.meta["_frame_room_sets"]
    fixes = []
    for l in links:
        if l["how"] != "feature matches":
            continue
        i, j = l["frames"]
        # match again on the images the model saw (cropped + resized), so pixels index its depth maps directly
        A_px, B_px = match_pair(_sift(z["rgb"][i]), _sift(z["rgb"][j]))
        ua, ub = np.round(A_px).astype(int), np.round(B_px).astype(int)
        H, W = z["depth"].shape[1:]
        inb = (ua[:, 0] < W) & (ua[:, 1] < H) & (ub[:, 0] < W) & (ub[:, 1] < H)
        ua, ub = ua[inb], ub[inb]
        Pa, va = view_points(z, i, ua)
        Pb, vb = view_points(z, j, ub)
        ok = va & vb
        if ok.sum() < 10:
            continue
        d = float(np.median(np.linalg.norm(Pa[ok] - Pb[ok], axis=1)))
        l["median_disagreement_m"] = round(d, 3)
        if d <= MATCH_AGREE_M:
            continue
        room = l["rooms"][1]   # move the second room onto the first
        s, R, t, res = robust_sim3(Pb[ok], Pa[ok])
        idx = [k for k, rs in enumerate(rooms_of) if room in rs and l["rooms"][0] not in rs]
        _apply_sim3(z, idx, s, R, t)
        fixes.append({"room": room, "onto": l["rooms"][0], "before_m": round(d, 3), "after_m": round(res, 3)})
        warnings.append(f"Room '{room}' was mis-placed by the joint reconstruction ({d:.2f} m between matched "
                        f"points with '{l['rooms'][0]}'); it was re-placed by a Sim(3) on {int(ok.sum())} feature "
                        f"matches (residual {res:.2f} m).")
    return fixes


def _photo_groups(bundle: CaptureBundle, warnings: list[str]) -> dict:
    """Decide which photos go into the joint pass: the largest set of linked rooms. Rooms with no link to it
    are left out (unplaced) and named in a warning."""
    rooms_of = bundle.meta.get("_frame_room_sets")
    if not rooms_of or not any(rooms_of):
        return {"groups": [list(range(len(bundle.frames)))], "links": [], "unplaced": []}
    comp, links = room_links(bundle)
    sizes = {}
    for i, rs in enumerate(rooms_of):
        for r in rs:
            sizes[comp[r]] = sizes.get(comp[r], 0) + 1
    main = max(sorted(sizes), key=lambda c: sizes[c])
    unplaced = sorted(r for r in comp if comp[r] != main)
    keep = [i for i, rs in enumerate(rooms_of) if any(comp[r] == main for r in rs)]
    for r in unplaced:
        warnings.append(f"Room '{r}' shares no photo and too few feature matches (< {MIN_MATCHES}) with the other "
                        f"rooms, so its position is unknown: it is NOT placed in the plan (not guessed, not "
                        f"overlapped). Re-capture with a doorway photo saved in both rooms' folders.")
    return {"groups": [keep], "links": links, "unplaced": unplaced}


# ----------------------------------------------------------------------------- gravity and scale

def _refine_up(N: np.ndarray, up: np.ndarray) -> tuple[np.ndarray, float]:
    """Pull `up` onto the normals of horizontal surfaces near it (same cones as fp.geometry.align.estimate_up).
    Returns (up, fraction of normals within 18 deg of it)."""
    for cone in (0.7, 0.85, 0.95):
        d = N @ up
        sel = np.abs(d) > cone
        if sel.sum() < 100:
            break
        v = (N[sel] * np.sign(d[sel])[:, None]).sum(0)
        up = v / np.linalg.norm(v)
    return up, float(np.mean(np.abs(N @ up) > 0.95))


def estimate_gravity(rec: Recon) -> tuple[np.ndarray, dict]:
    """Try each in-image direction (camera -y, +y, -x, +x averaged over all views) as the start, refine onto
    horizontal surfaces, and keep the one with most horizontal support. A sideways start can't tell up from
    down, so the sign is chosen so that the larger horizontal layer 0.8-2.0 m from the cameras is below them
    (the floor: handheld cameras look level or down, so they see more floor than ceiling)."""
    R = np.array([f.T_wc[:3, :3] for f in rec.frames])
    starts = {"-y": -R[:, :, 1].mean(0), "+y": R[:, :, 1].mean(0), "-x": -R[:, :, 0].mean(0), "+x": R[:, :, 0].mean(0)}
    best = None
    for name, u in starts.items():
        if np.linalg.norm(u) < 1e-6:
            continue
        up, support = _refine_up(rec.normals, u / np.linalg.norm(u))
        if best is None or support > best[2] + 1e-9:
            best = (name, up, support)
    name, up, support = best
    cams = np.array([f.T_wc[:3, 3] for f in rec.frames])
    h = rec.points @ up - np.median(cams @ up)
    horiz = np.abs(rec.normals @ up) > 0.95
    # the floor is the big horizontal layer 0.8-2.0 m from the camera; it must be below (h < 0)
    below = np.sum(horiz & (h < -0.8) & (h > -2.0))
    above = np.sum(horiz & (h > 0.8) & (h < 2.0))
    if above > below:
        up = -up
    return up, {"start": name, "horizontal_support": round(support, 3)}


def floor_check(rec: Recon, up: np.ndarray) -> float | None:
    """Angle (deg) between `up` and a plane fitted (SVD) to the floor layer's points. None if no floor layer."""
    cams = np.array([f.T_wc[:3, 3] for f in rec.frames])
    h = rec.points @ up - np.median(cams @ up)
    horiz = (rec.normals @ up > 0.9) & (h < -0.8) & (h > -2.0)
    if horiz.sum() < 200:
        return None
    hist, e = np.histogram(h[horiz], np.arange(-2.0, -0.8, 0.04))
    z0 = 0.5 * (e[np.argmax(hist)] + e[np.argmax(hist) + 1])
    P = rec.points[horiz & (np.abs(h - z0) < 0.04)]
    if len(P) < 100:
        return None
    n = np.linalg.svd(P - P.mean(0), full_matrices=False)[2][2]
    return float(np.degrees(np.arccos(min(1.0, abs(n @ up)))))


def scale_check(rec: Recon) -> dict:
    """Compare the model's metric scale with priors: camera height above the floor (handheld) and ceiling
    height. `extra_rel` = the largest relative gap to a prior range; contract adds it to every interval's
    scale term. The values themselves are never rescaled (a prior is weaker evidence than the model)."""
    from fp.geometry.align import FloorNotFound, align
    try:
        al = align(rec)
    except FloorNotFound:
        return {"camera_h_m": None, "ceiling_h_m": None, "extra_rel": 0.0,
                "note": "no floor found, so the scale could not be checked against priors"}
    gaps, notes = [], []
    for name, v, (lo, hi) in (("camera height", al["camera_h"], CAM_H), ("ceiling height", al["ceiling_h"], CEILING_H)):
        if v is None:
            continue
        gap = (lo - v) / v if v < lo else (v - hi) / v if v > hi else 0.0
        gaps.append(gap)
        if gap > 0:
            notes.append(f"{name} {v:.2f} m is outside the prior {lo}-{hi} m ({100 * gap:.0f} %)")
    return {"camera_h_m": round(al["camera_h"], 3),
            "ceiling_h_m": None if al["ceiling_h"] is None else round(al["ceiling_h"], 3),
            "priors": {"camera_h_m": list(CAM_H), "ceiling_h_m": list(CEILING_H)},
            "extra_rel": round(max(gaps, default=0.0), 4), "note": "; ".join(notes) or "within priors"}


# ----------------------------------------------------------------------------- entry point

def reconstruct(bundle: CaptureBundle, work: Path, cache_dir: Path, backend: str = "local",
                use_cache: bool = True, voxel: float = 0.02, max_depth: float = 5.0) -> tuple[CaptureBundle, Recon]:
    """Bundle without depth or poses -> bundle with predicted depth/K/T_wc + fused, gravity-aligned cloud."""
    from fp.recon.lidar_fuse import fuse_depth
    warnings: list[str] = []
    info = {}
    if bundle.source == "photos":
        g = _photo_groups(bundle, warnings)
        bundle.meta["_recon_groups"] = g["groups"]
        info["room_links"] = [{k: v for k, v in l.items() if k != "frames"} for l in g["links"]]
        info["unplaced_rooms"] = g["unplaced"]
    z, pinfo = _predict(bundle, cache_dir, backend, use_cache)
    info.update(pinfo)
    for st in info["chunk_merges"]:
        if st["median_residual_m"] > MERGE_WARN_M:
            t0, t1 = (bundle.frames[i].timestamp for i in st["views"])
            span = f" ({t0:.0f}-{t1:.0f} s)" if t0 is not None and t1 is not None else ""
            warnings.append(f"Chunk {st['chunk']}{span} disagrees with the previous chunk by "
                            f"{st['median_residual_m']:.2f} m on the frames they share: the model's camera poses are "
                            f"inconsistent there (fast motion or close-ups), so walls seen in that stretch may be doubled "
                            f"or misplaced.")
    if bundle.source == "photos" and info.get("room_links"):
        info["replaced_rooms"] = _check_matched_rooms(z, bundle, g["links"], warnings)
    keep = sorted({i for grp in bundle.meta.get("_recon_groups") or [range(len(bundle.frames))] for i in grp})
    if not keep:
        raise StageError("recon", "No frames left to reconstruct.")

    # depth maps as uint16 mm PNGs (like LiDAR), low-confidence pixels zeroed, so fuse_depth runs unchanged
    work.mkdir(parents=True, exist_ok=True)
    conf_ok = z["mask"][keep]
    floor_c = np.percentile(z["conf"][keep][conf_ok].astype(np.float32), CONF_PERCENTILE) if conf_ok.any() else 0
    frames = []
    for i in keep:
        f = bundle.frames[i]
        ok = z["mask"][i] & (z["conf"][i].astype(np.float32) >= floor_c)
        depth_mm = np.where(ok, np.round(z["depth"][i].astype(np.float32) * 1000), 0).clip(0, 65535).astype(np.uint16)
        rgb_p, depth_p = work / f"{i:04d}.jpg", work / f"{i:04d}_depth.png"
        cv2.imwrite(str(rgb_p), z["rgb"][i][..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(depth_p), depth_mm)
        frames.append(Frame(rgb=rgb_p, K=z["K"][i].astype(float), T_wc=z["T_wc"][i].astype(float), depth=depth_p,
                            timestamp=f.timestamp, video_index=f.video_index))
    for k in ("_frame_rooms", "_frame_room_sets", "_source_ids", "_K_prior"):
        if bundle.meta.get(k) is not None:
            bundle.meta[k] = [bundle.meta[k][i] for i in keep]
    bundle.meta["_original_rgb"] = [str(bundle.frames[i].rgb) for i in keep]
    bundle.frames = frames
    bundle.meta.pop("_recon_groups", None)

    rec = fuse_depth(bundle, voxel=voxel, max_depth=max_depth)
    up, ginfo = estimate_gravity(rec)
    rec.up = up
    ginfo["floor_plane_angle_deg"] = None if (a := floor_check(rec, up)) is None else round(a, 2)
    rec.meta["up_source"] = "estimated (horizontal surfaces, 4 starts; checked against the floor plane)"
    sc = scale_check(rec)
    if sc["extra_rel"] > 0:
        warnings.append(f"Metric scale disagrees with priors ({sc['note']}); every interval is widened by "
                        f"{100 * sc['extra_rel']:.0f} % of the value (values are not rescaled).")
    if bundle.source == "video" and len(frames) > 1:
        # keyframes are sparse (~1 per second): interpolate the path at 10 Hz so room occupancy counts time
        t = np.array([f.timestamp for f in frames])
        c = np.array([f.T_wc[:3, 3] for f in frames])
        tt = np.arange(t[0], t[-1], 0.1)
        bundle.meta["_trajectory"] = (tt, np.stack([np.interp(tt, t, c[:, k]) for k in range(3)], 1))
    info.update(gravity=ginfo, scale_check=sc, metric_scale_model=round(float(np.median(z["scale"][keep])), 3))
    bundle.meta["recon"] = info
    rec.meta["warnings"] = warnings
    return bundle, rec
