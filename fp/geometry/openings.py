"""Doors, windows and passages from wall elevations: what the depth sensor saw *through* each wall plane.

Method (one wall at a time, plan frame: z up, floor at z = 0):
 1. Elevation grid on the wall plane: along-wall u (from p0, a margin past each end) x height z, CELL cells.
 2. Ray casting: every depth pixel is a ray from its camera to the measured point. Per cell, count the frames
    whose rays *ended* on the plane (within HIT_TOL: wall there) and the frames whose rays *crossed* the plane
    and ended more than HIT_TOL behind it (air there: the sensor saw through). A cell no ray reached is
    *unobserved* (furniture in front, or never looked at), which is not the same as empty: that is the point
    of casting rays instead of looking for holes in the point cloud.
 3. Openings = connected regions of empty cells. Below the region: occupied wall up to the sill (none for a
    door or passage). Above it: occupied wall from the head (lintel) up (none for a passage).
    door: no sill, 0.6-1.6 m wide, head 1.9-2.4 m.   window: sill 0.3-1.5 m, head below the ceiling.
    passage: no sill, wider than 1.6 m or no lintel below the ceiling, and another room behind it.
 4. Width = distance between the two jamb planes. Each jamb is the median along-wall position of the reveal
    points (vertical surfaces facing into the opening, inside the wall's thickness). With no reveal points
    (a passage at a room-split line, a jamb in shadow), the 95th percentile of the wall-end points (on the
    plane, next to the gap) stands in. The empty region's own width is never used: its edges are
    rounded to cells and blurred by ray density.
 5. One physical opening is seen from both rooms (both faces of a partition): keep one, linked to both rooms.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy import ndimage
from shapely import contains_xy
from shapely.geometry import Polygon

CELL = 0.02          # m: elevation cell. Rays are 4 depth pixels apart (~4 cm at 2 m) in one frame, so 1 cm cells
                     # would be mostly empty of votes; jambs are measured from points, not from cells
RAY_STRIDE = 4       # every 4th depth pixel in x and y (64 x 48 rays per frame)
HIT_TOL = 0.05       # m: a ray ending within 5 cm of the plane hit the wall (LiDAR noise ~1 cm, smear 2-4 cm)
U_MARGIN = 0.30      # m past each wall end: a jamb can sit at the room corner
Z_MAX = 3.6          # m: top of the elevation (highest ceiling the pipeline accepts, align.CEILING_RANGE)
MAX_RAY_DEPTH = 8.0  # m: a ray only has to get past the plane, so far (noisier) returns still count as see-through
MIN_FRAMES = 2       # frames that must agree before a cell counts as observed (one frame can be a glitch)
EMPTY_FRAC = 0.6     # a cell is empty if >= 60 % of the frames that reached it saw through it
MIN_OPEN_AREA = 0.15  # m^2 of empty cells: smaller blobs are noise or gaps between objects
DOOR_W = (0.6, 1.6)  # m: door leaves are 0.6-1.0 m; double doors up to 1.6 m (work order 04)
DOOR_HEAD = (1.9, 2.4)  # m: door heights (work order 04)
DOOR_MIN_SEEN_TOP = 1.2  # m: with no lintel seen (a camera held low and pointing down never sees a door's top), a
                         # door-wide gap from the floor must be seen through up to at least here: above counters
                         # (0.9 m) and sofa backs, so a gap between furniture does not pass; its height is unobserved
SILL_FLOOR = 0.05    # m: a "sill" no higher than this is the floor (door, passage)
WINDOW_SILL = (0.3, 1.5)  # m (work order 04)
WINDOW_MIN = 0.3     # m: narrower or shorter "windows" are vents or gaps
PASSAGE_MIN_W = 1.6  # m: a floor gap wider than any door is a passage (work order 04)
HEAD_AT_CEILING = 0.15  # m: a lintel within this of the ceiling is the ceiling itself: no lintel
DEFAULT_CEILING = 2.6   # m: elevation top / passage test when the ceiling was not observed
JAMB_SEARCH = 0.12   # m: look for a jamb this far either side of the empty region's edge (6 cells)
REVEAL_DEPTH = 0.35  # m behind the wall face: reveals are inside the wall thickness (partitions 8-30 cm)
REVEAL_NORMAL = 0.8  # |normal . along-wall| for a reveal (faces into the opening)
MIN_JAMB_AREA = 0.02  # m^2 of reveal surface for a jamb plane (a 1 m tall x 2 cm strip); points x voxel^2 = area
REVEAL_SETBACK = 0.02  # m behind the wall face: door casings stand 1-2 cm proud of the face; their side edges also
                       # face into the opening but sit 5-8 cm outside the clear width, so they are left out
JAMB_POS = 0.01      # m: one jamb plane's position error from LiDAR (256 px depth at 2 m ~ 1 cm per pixel)
WALL_END_POS = 0.02  # m: extra error when a jamb comes from wall-end points, not a reveal plane
MIRROR_TOL = 0.03    # m: a reflected point this close to a surface of the room counts as "the room again"
MIRROR_BEHIND = (0.15, 3.0)  # m behind the plane: where a mirror's virtual room appears
MIRROR_FRAC = 0.5    # a mirror if >= half the points behind it, reflected back, land on the room's own surfaces
                     # (measured on c7d28f72c6: see docs/DECISIONS.md D04.3)
RECESS_DEPTH = 0.40  # m: a surface facing the room this close behind the plane is the back of a niche
RECESS_FRAC = 0.5    # ...covering >= half the opening's cells: a recess, not an opening -- unless it has a door's
                     # shape: then it is a closed door, its leaf set back 3-5 cm in the frame (seen on c7d28f72c6)
PARTNER_DIST = 0.40  # m: an opening on a parallel wall this close is the same opening seen from the other room
WALL_FIT_TOL = 0.10  # m: an opening wider than its own wall + 10 cm is not in that wall (verifier, c7d28f72c6 R3.W1)
SAME_PAIR_DIST = 1.0  # m: two openings between the same two rooms closer than this are one (a doorway at a corner
                      # is cut by both rooms' wall lines; real double doors between two rooms are rarer than that)
PROBE = (0.05, 0.7)  # m: walk this far out from an opening to find the room behind it


def _walls(rooms):
    """Flatten room walls: p0, unit u (along), outward normal n (right of a CCW edge), length."""
    out = []
    for r in rooms:
        for w in r["walls"]:
            p0, p1 = np.array(w["p0"], float), np.array(w["p1"], float)
            L = float(np.linalg.norm(p1 - p0))
            if L < 0.3:
                continue
            u = (p1 - p0) / L
            out.append({"room": r["id"], "wall": w, "id": f"{r['id']}.{w['id']}", "p0": p0, "u": u,
                        "n": np.array([u[1], -u[0]]), "L": L, "observed": bool(w.get("observed"))})
    return out


def cast_votes(frames, T, walls, z_max=Z_MAX):
    """Per wall: (through, hit) frame counts per elevation cell (u index, z index). frames need depth,
    confidence (optional), K at RGB resolution, T_wc (world); T maps world -> plan."""
    nz = int(np.ceil(z_max / CELL))
    nu = [int(np.ceil((w["L"] + 2 * U_MARGIN) / CELL)) for w in walls]
    off = np.r_[0, np.cumsum([n * nz for n in nu])]
    thr, hit = np.zeros(off[-1], np.int32), np.zeros(off[-1], np.int32)
    if not walls or not frames:
        return [(thr[a:b].reshape(n, nz), hit[a:b].reshape(n, nz)) for a, b, n in zip(off[:-1], off[1:], nu)]
    P0 = np.array([w["p0"] for w in walls]); U = np.array([w["u"] for w in walls])
    Nn = np.array([w["n"] for w in walls]); L = np.array([w["L"] for w in walls])
    NU = np.array(nu); OFF = off[:-1]
    rgb_w = None
    for f in frames:
        D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
        if D is None:
            continue
        if rgb_w is None:
            rgb_w = cv2.imread(str(f.rgb)).shape[1]
        s = D.shape[1] / rgb_w
        fx, fy, cx, cy = f.K[0, 0] * s, f.K[1, 1] * s, f.K[0, 2] * s, f.K[1, 2] * s
        d = D[::RAY_STRIDE, ::RAY_STRIDE] / 1000.0
        v, uu = np.mgrid[0:d.shape[0], 0:d.shape[1]] * RAY_STRIDE
        conf = (cv2.imread(str(f.confidence), cv2.IMREAD_UNCHANGED)[::RAY_STRIDE, ::RAY_STRIDE]
                if f.confidence is not None else np.full(d.shape, 2, np.uint8))
        m = (d > 0.1) & (d < MAX_RAY_DEPTH) & (conf >= 1)
        z = d[m]
        cam = np.stack([(uu[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], 1)
        Twp = T @ f.T_wc
        E = cam @ Twp[:3, :3].T + Twp[:3, 3]           # ray end points, plan frame
        c = Twp[:3, 3]
        strong = conf[m] >= 2                           # a hit must be a confident return (as in the cloud)
        sc = np.einsum("wk,wk->w", c[None, :2] - P0, Nn)   # camera side of each plane (W,)
        uc = np.einsum("wk,wk->w", c[None, :2] - P0, U)
        rel = E[:, None, :2] - P0[None]                 # (n, W, 2)
        se = np.einsum("nwk,wk->nw", rel, Nn)
        ue = np.einsum("nwk,wk->nw", rel, U)
        # hits: end point on the plane
        hm = (np.abs(se) <= HIT_TOL) & strong[:, None]
        r_, w_ = np.nonzero(hm)
        keys_h = _cell_keys(ue[r_, w_], E[r_, 2], w_, NU, OFF, nz)
        # through: camera and end point on opposite sides, end point clearly behind the plane
        tm = (sc[None] * se < 0) & (np.abs(se) > HIT_TOL) & (np.abs(sc)[None] > HIT_TOL)
        r_, w_ = np.nonzero(tm)
        t = sc[w_] / (sc[w_] - se[r_, w_])
        ux = uc[w_] + t * (ue[r_, w_] - uc[w_])
        zx = c[2] + t * (E[r_, 2] - c[2])
        keys_t = _cell_keys(ux, zx, w_, NU, OFF, nz)
        hit[np.unique(keys_h)] += 1                     # count frames, not rays
        thr[np.unique(keys_t)] += 1
    return [(thr[a:b].reshape(n, nz), hit[a:b].reshape(n, nz)) for a, b, n in zip(off[:-1], off[1:], nu)]


def _cell_keys(u, z, w, NU, OFF, nz):
    iu = np.floor((u + U_MARGIN) / CELL).astype(np.int64)
    iz = np.floor(z / CELL).astype(np.int64)
    ok = (iu >= 0) & (iu < NU[w]) & (iz >= 0) & (iz < nz)
    return OFF[w[ok]] + iu[ok] * nz + iz[ok]


def cell_states(thr, hit):
    """0 unobserved, 1 occupied (wall), 2 empty (seen through)."""
    obs = (thr + hit) >= MIN_FRAMES
    empty = obs & (thr >= EMPTY_FRAC * (thr + hit))
    st = np.zeros(thr.shape, np.uint8)
    st[obs] = 1
    st[empty] = 2
    return st


def _runs(col: np.ndarray, value: int):
    """Contiguous runs [a, b) of `value` in a 1-D array."""
    m = np.r_[False, col == value, False]
    d = np.diff(m.astype(np.int8))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))


def _regions(st, ceiling):
    """Candidate openings in one wall's state image: connected empty regions, cleaned up, with per-column
    sill (top of the occupied wall below) and head (bottom of the occupied wall above)."""
    empty = st == 2
    empty = ndimage.binary_closing(empty, np.ones((3, 3)))   # ray-density pinholes
    empty = ndimage.binary_opening(empty, np.ones((3, 3)))   # single-cell specks
    lab, n = ndimage.label(empty)
    out = []
    for k in range(1, n + 1):
        m = lab == k
        if m.sum() * CELL * CELL < MIN_OPEN_AREA:
            continue
        cols = np.nonzero(m.any(1))[0]
        sills, heads, tops, bots = [], [], [], []
        for i in cols:
            zz = np.nonzero(m[i])[0]
            lo, hi = zz.min(), zz.max()
            below = np.nonzero(st[i, :lo] == 1)[0]
            above = np.nonzero(st[i, hi + 1:] == 1)[0]
            sills.append((below.max() + 1) * CELL if len(below) else 0.0)
            heads.append((hi + 1 + above.min()) * CELL if len(above) else np.nan)
            bots.append(lo * CELL)
            tops.append((hi + 1) * CELL)
        heads = np.array(heads)
        core = slice(len(cols) // 4, max(len(cols) // 4 + 1, 3 * len(cols) // 4))   # middle half: away from jambs
        head = float(np.nanmedian(heads[core])) if np.isfinite(heads[core]).any() else None
        if head is not None and head >= ceiling - HEAD_AT_CEILING:
            head = None
        out.append({"mask": m, "i0": int(cols.min()), "i1": int(cols.max()) + 1,
                    "sill": float(np.median(np.array(sills)[core])), "head": head,
                    "bottom": float(np.median(np.array(bots)[core])), "top": float(np.median(np.array(tops)[core])),
                    "area": float(m.sum() * CELL * CELL)})
    return out


def _classify(r, width, ceiling):
    """door / window / passage, or None (not an opening)."""
    s, h = r["sill"], r["head"]
    if s <= SILL_FLOOR:
        if width > PASSAGE_MIN_W or (h is None and width >= DOOR_W[0] and r["top"] >= ceiling - HEAD_AT_CEILING):
            return "passage"
        if DOOR_W[0] <= width <= DOOR_W[1] and ((h is not None and DOOR_HEAD[0] <= h <= DOOR_HEAD[1])
                                                 or (h is None and r["top"] >= DOOR_MIN_SEEN_TOP)):
            return "door"
        return None
    if WINDOW_SILL[0] <= s <= WINDOW_SILL[1] and width >= WINDOW_MIN and (r["top"] - s) >= WINDOW_MIN:
        return "window"
    return None


def _jamb(P, N, w, u_edge, side, zlo, zhi, voxel):
    """Along-wall position of one jamb. side -1 = left jamb (opening to its right, +u), +1 = right jamb.
    Returns (u, sigma, n, method)."""
    rel = P[:, :2] - w["p0"]
    u, s = rel @ w["u"], rel @ w["n"]
    band = (P[:, 2] > zlo) & (P[:, 2] < zhi) & (np.abs(u - u_edge) < JAMB_SEARCH)
    nu = N[:, :2] @ w["u"]
    # reveal: inside the wall's thickness, facing into the opening (left jamb faces +u)
    rv = band & (s > REVEAL_SETBACK) & (s < REVEAL_DEPTH) & (-side * nu > REVEAL_NORMAL)
    if rv.sum() * voxel ** 2 >= MIN_JAMB_AREA:
        x = u[rv]
        return float(np.median(x)), float(1.4826 * np.median(np.abs(x - np.median(x)))), int(rv.sum()), "reveal"
    # fallback: the end of the wall face next to the gap (points on the plane, outside the empty region)
    on = band & (np.abs(s) < HIT_TOL) & ((u - u_edge) * side > -CELL)
    if on.sum() * voxel ** 2 >= MIN_JAMB_AREA / 2:
        x = u[on]
        q = np.percentile(x, 5 if side > 0 else 95)
        return float(q), float(1.4826 * np.median(np.abs(x - np.median(x)))) / 2, int(on.sum()), "wall end"
    return float(u_edge), CELL * 2, 0, "region edge"


def behind_tests(P, N, tree, w, u0, u1, z0, z1):
    """(mirror fraction, recess fraction) for a candidate opening on wall w spanning [u0, u1] x [z0, z1].
    Mirror: a mirror makes the depth sensor 'see' a copy of the room behind the wall. Reflect the points
    behind the opening back across the wall plane; for a mirror most land on surfaces of the room itself.
    Recess: a wall surface facing into the room just behind the plane across the opening (a niche, a
    recessed panel, a closed door set deep in its frame) means nothing was seen through it."""
    rel = P[:, :2] - w["p0"]
    u, s = rel @ w["u"], rel @ w["n"]
    near = (u > u0 - 0.5) & (u < u1 + 0.5)
    beh = near & (s > MIRROR_BEHIND[0]) & (s < MIRROR_BEHIND[1])
    idx = np.nonzero(beh)[0]
    rng = np.random.default_rng(0)                   # same sample for the same candidate, whatever ran before
    mirror = 0.0
    if len(idx) >= 50:
        idx = rng.choice(idx, min(len(idx), 4000), replace=False)
        Q = P[idx].copy()
        Q[:, :2] -= 2 * s[idx, None] * w["n"][None]          # reflect across the plane
        dist, _ = tree.query(Q, distance_upper_bound=MIRROR_TOL)
        mirror = float(np.mean(np.isfinite(dist)))
    inn = (u > u0) & (u < u1) & (P[:, 2] > z0) & (P[:, 2] < z1)
    back = inn & (s > HIT_TOL) & (s < RECESS_DEPTH) & ((N[:, :2] @ w["n"]) < -0.8)
    nu, nz = max(1, int((u1 - u0) / CELL)), max(1, int((z1 - z0) / CELL))
    cells = np.zeros((nu, nz), bool)
    iu = np.clip(((u[back] - u0) / CELL).astype(int), 0, nu - 1)
    iz = np.clip(((P[back, 2] - z0) / CELL).astype(int), 0, nz - 1)
    cells[iu, iz] = True
    return mirror, float(cells.mean())


def _room_behind(rooms_poly, owner, c, n):
    for d in np.arange(*PROBE, 0.05):
        p = c + n * d
        for rid, poly in rooms_poly:
            if rid != owner and contains_xy(poly, p[0], p[1]):
                return rid
    return None


def detect(rooms, frames, T, P, N, voxel: float, ceilings: dict | None = None, debug_dir: Path | None = None,
           tier_scale: float = 1.0):
    """rooms: geometry rooms (id, polygon, walls with p0/p1/observed). frames: fused frames with depth.
    ceilings: room id -> ceiling height (None = not observed). Returns (openings, info): openings are dicts
    {room, wall_id, kind, rooms, width(value, half, method, observed), height, sill, center, ...};
    info has per-wall vote images and rejected candidates for the debug images."""
    ceilings = ceilings or {}
    walls = _walls(rooms)
    votes = cast_votes(frames, T, walls)
    polys = [(r["id"], Polygon(r["polygon"])) for r in rooms]
    from scipy.spatial import cKDTree
    tree = cKDTree(P)
    found, rejected, states = [], [], {}
    for w, (thr, hit) in zip(walls, votes):
        st = cell_states(thr, hit)
        states[w["id"]] = st
        H = ceilings.get(w["room"]) or DEFAULT_CEILING
        for r in _regions(st, H):
            ue0, ue1 = r["i0"] * CELL - U_MARGIN, r["i1"] * CELL - U_MARGIN
            if ue1 < 0.05 or ue0 > w["L"] - 0.05:                     # wholly past the wall's ends
                continue
            zhi = (r["head"] if r["head"] is not None else r["top"]) - 0.05
            zlo = max(r["sill"], 0.0) + 0.05
            zlo, zhi = (zlo, zhi) if zhi - zlo > 0.3 else (r["bottom"], r["top"])
            jl = _jamb(P, N, w, ue0, -1, zlo, zhi, voxel)
            jr = _jamb(P, N, w, ue1, +1, zlo, zhi, voxel)
            width = jr[0] - jl[0]
            kind = _classify(r, width, H)
            if kind == "door" and r["head"] is None and width > w["L"] + WALL_FIT_TOL:
                kind = "passage"            # wider than its wall and no lintel: an open-plan gap, not a door
            cxy = w["p0"] + w["u"] * 0.5 * (jl[0] + jr[0])
            top = r["head"] if r["head"] is not None else r["top"]
            mirror, recess = behind_tests(P, N, tree, w, jl[0], jr[0], r["sill"], top) if kind else (0.0, 0.0)
            behind = _room_behind(polys, w["room"], cxy, w["n"])
            cand = {"room": w["room"], "wall_id": w["id"], "kind": kind, "u0": jl[0], "u1": jr[0],
                    "width_m": round(width, 3), "jambs": [jl, jr], "region": r, "center": cxy.round(3).tolist(),
                    "behind": behind, "mirror": round(mirror, 3), "recess": round(recess, 3), "wall_observed": w["observed"], "n": w["n"], "u": w["u"]}
            why = None
            if kind is None:
                why = "shape is not a door, window or passage"
            elif jl[3] == "region edge" and jr[3] == "region edge":
                why = "no jamb on either side: air beside a wall, not a hole in it"
            elif mirror >= MIRROR_FRAC:
                why = f"mirror: {100 * mirror:.0f} % of what was seen behind it is the room itself, reflected"
            elif width > w["L"] + WALL_FIT_TOL and (kind == "window" or behind is None):
                why = (f"{width:.2f} m wide on a {w['L']:.2f} m wall: the wall line is wrong here (seen through "
                       "past its ends), not a hole in it")
            elif recess >= RECESS_FRAC and kind == "door" and r["head"] is None:
                why = "door-shaped recess with no lintel seen: a niche, not a closed door in its frame"
            elif recess >= RECESS_FRAC and kind == "door":
                cand["closed"] = True                     # a door leaf set back in its frame: a closed door
            elif recess >= RECESS_FRAC:
                why = f"recess: a surface {100 * recess:.0f} % covers it less than {RECESS_DEPTH:.1f} m behind the wall"
            elif kind == "passage" and behind is None:
                why = "floor-to-ceiling gap with no room behind it: the wall itself was not where the plan put it"
            elif not w["observed"] and kind != "passage":
                why = "on an inferred wall"
            if why:
                cand["why"] = why
                rejected.append(cand)
            else:
                found.append(cand)
    found = _dedupe(found)
    out = [_to_opening(c, ceilings, tier_scale) for c in found]
    if debug_dir is not None:
        _debug_images(Path(debug_dir), walls, states, found, rejected)
    return out, {"states": states, "rejected": rejected, "walls": walls}


def _score(c):
    """Prefer the copy whose jambs are reveal planes, then the one with more jamb points."""
    return (sum(j[3] == "reveal" for j in c["jambs"]), sum(j[2] for j in c["jambs"]))


def _dedupe(cands):
    """One opening seen from both rooms (both faces of a partition) -> keep the better measured copy."""
    keep = []
    for c in sorted(cands, key=_score, reverse=True):
        dup = False
        for k in keep:
            pair = c["behind"] is not None and {c["room"], c["behind"]} == {k["room"], k["behind"]}
            if pair and np.hypot(*np.subtract(c["center"], k["center"])) < SAME_PAIR_DIST:
                dup = True                      # same two rooms, same place: one opening on two walls' lines
                break
            if abs(float(np.dot(c["u"], k["u"]))) < 0.99:
                continue
            d = np.subtract(c["center"], k["center"])
            if abs(float(np.dot(d, k["n"]))) < PARTNER_DIST and abs(float(np.dot(d, k["u"]))) < 0.5 * max(
                    c["width_m"], k["width_m"]):
                dup = True
                if k["behind"] is None and c["room"] != k["room"]:
                    k["behind"] = c["room"]
                break
        if not dup:
            keep.append(c)
    return keep


def _to_opening(c, ceilings, tier_scale):
    """Candidate -> internal opening dict (contract.py turns it into the schema Opening)."""
    jl, jr = c["jambs"]
    side_half = lambda j: (JAMB_POS * tier_scale + (WALL_END_POS if j[3] != "reveal" else 0.0)
                           + (CELL if j[3] == "region edge" else 0.0) + j[1])
    methods = sorted({j[3] for j in c["jambs"]})
    observed = all(j[3] != "region edge" for j in c["jambs"])
    r = c["region"]
    width = {"value": c["width_m"], "half": side_half(jl) + side_half(jr), "observed": observed,
             "method": ("closed door (leaf set back in its frame): " if c.get("closed") else "")
                       + "distance between jamb planes (" + " + ".join(methods) + ")"}
    height = None
    if c["kind"] in ("door", "window"):
        if r["head"] is not None:
            height = {"value": r["head"] - (r["sill"] if c["kind"] == "window" else 0.0),
                      "half": 2 * CELL * tier_scale, "observed": True, "method": "lintel (wall above the gap) "
                      + ("minus sill" if c["kind"] == "window" else "above the floor")}
        else:
            height = {"value": None, "half": None, "observed": False, "method": "lintel not observed"}
    sill = None
    if c["kind"] == "window":
        sill = {"value": r["sill"], "half": 2 * CELL * tier_scale, "observed": True,
                "method": "top of the wall below the gap"}
    rooms = None
    if c["kind"] != "window" and c["behind"] is not None:
        rooms = [c["room"], c["behind"]]
    top = r["head"] if r["head"] is not None else r["top"]
    return {"room": c["room"], "wall_id": c["wall_id"], "kind": c["kind"], "rooms": rooms,
            "mirror": c["mirror"], "recess": c["recess"], "closed": bool(c.get("closed")), "width": width,
            "height": height, "sill": sill, "center": c["center"], "u": [float(v) for v in c["u"]],
            "z": [round(float(r["sill"]), 3), round(float(top), 3)],
            "jambs": [{"u": round(j[0], 3), "sigma_m": round(j[1], 4), "n": j[2], "method": j[3]} for j in c["jambs"]]}


# ----------------------------------------------------------------------------- debug images
PX = 3  # screen pixels per cell
COL = {0: (255, 255, 255), 1: (90, 90, 90), 2: (240, 200, 150)}   # BGR: unobserved white, wall grey, empty blue
KCOL = {"door": (0, 0, 220), "window": (220, 90, 0), "passage": (0, 150, 0), None: (0, 170, 255)}


def _debug_images(d: Path, walls, states, found, rejected):
    """debug/openings/<wall_id>.png for every wall with an opening or a rejected candidate: the wall seen from
    inside the room (u to the right, z up). White = unobserved, grey = wall (rays ended on the plane), light
    blue = seen through; boxes: red door, blue window, green passage, orange rejected (dashed); thin black
    lines = the wall's own ends.
    Drawn as seen from inside the room, so p0 is on the right."""
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("*.png"):
        old.unlink()
    by_wall = {}
    for c in found:
        by_wall.setdefault(c["wall_id"], []).append((c, True))
    for c in rejected:
        by_wall.setdefault(c["wall_id"], []).append((c, False))
    for w in walls:
        if w["id"] not in by_wall:
            continue
        st = states[w["id"]]
        img = np.array([COL[k] for k in range(3)], np.uint8)[st.T[::-1]]
        img = np.ascontiguousarray(cv2.resize(img, None, fx=PX, fy=PX, interpolation=cv2.INTER_NEAREST)[:, ::-1])
        H, Wd = img.shape[:2]
        X = lambda u: int(Wd - (u + U_MARGIN) / CELL * PX)   # mirrored: as seen from inside the room
        Y = lambda z: int(H - z / CELL * PX)
        for u in (0.0, w["L"]):
            cv2.line(img, (X(u), 0), (X(u), H), (0, 0, 0), 1)
        for z in np.arange(0.5, Z_MAX, 0.5):
            cv2.line(img, (0, Y(z)), (6, Y(z)), (0, 0, 0), 1)
            cv2.putText(img, f"{z:.1f}", (8, Y(z) + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)
        for c, ok in by_wall[w["id"]]:
            r = c["region"]
            top = r["head"] if r["head"] is not None else r["top"]
            col = KCOL[c["kind"]] if ok else KCOL[None]
            p0, p1 = (X(c["u1"]), Y(top)), (X(c["u0"]), Y(r["sill"]))
            cv2.rectangle(img, p0, p1, col, 2 if ok else 1)
            txt = (f"{c['kind']} {c['width_m']:.2f} m" if ok else f"rejected: {c.get('why', '')[:40]}")
            cv2.putText(img, txt, (p0[0] + 3, max(12, p0[1] - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
            for j in c["jambs"]:
                if ok:
                    cv2.line(img, (X(j[0]), Y(r["sill"])), (X(j[0]), Y(top)), col, 1)
        band = np.full((22, img.shape[1], 3), 255, np.uint8)
        cv2.putText(band, f"{w['id']}  L={w['L']:.2f} m  ({'observed' if w['observed'] else 'inferred'} wall), "
                          "seen from inside", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        cv2.imwrite(str(d / f"{w['id']}.png"), np.vstack([band, img]))


# ----------------------------------------------------------------------------- RGB evidence sheets
SHEET_FRAMES = 5      # frames per opening (work order 04: look at 5 frames per opening)
SHEET_MIN_GAP_S = 2.0  # s between chosen frames, so they show different viewpoints
VIEW_DIST = (0.8, 5.0)  # m from the camera to the opening centre


def _project(f, T, X):
    """Plan points (n,3) -> pixel coords in the frame's RGB image and camera depth."""
    Tcp = np.linalg.inv(T @ f.T_wc)
    C = X @ Tcp[:3, :3].T + Tcp[:3, 3]
    z = C[:, 2]
    uv = (C @ f.K.T)[:, :2] / np.maximum(z, 1e-6)[:, None]
    return uv, z


def _views(frames, T, X, n_wall, k=SHEET_FRAMES):
    """Frames that see plan point X (centre) unoccluded and roughly face-on, spread in time."""
    img_wh = None
    cand = []
    for i, f in enumerate(frames):
        if img_wh is None:
            h, w = cv2.imread(str(f.rgb)).shape[:2]
            img_wh = (w, h)
        uv, z = _project(f, T, X[None])
        if not (VIEW_DIST[0] < z[0] < VIEW_DIST[1]):
            continue
        u, v = uv[0]
        if not (0.1 * img_wh[0] < u < 0.9 * img_wh[0] and 0.1 * img_wh[1] < v < 0.9 * img_wh[1]):
            continue
        if f.depth is not None:                         # occluded: something nearer than the opening
            D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
            if D is not None:
                sx, sy = D.shape[1] / img_wh[0], D.shape[0] / img_wh[1]
                dd = D[min(int(v * sy), D.shape[0] - 1), min(int(u * sx), D.shape[1] - 1)] / 1000.0
                if 0 < dd < z[0] - 0.3:
                    continue
        cam = (T @ f.T_wc)[:3, 3]
        d = X - cam
        face = abs(float(np.dot(d[:2] / max(np.linalg.norm(d[:2]), 1e-6), n_wall)))
        cand.append((face - 0.1 * abs(z[0] - 2.0), i, f.timestamp or i))
    cand.sort(reverse=True)
    out = []
    for _, i, t in cand:
        if all(abs(t - t2) >= SHEET_MIN_GAP_S for _, t2 in out):
            out.append((i, t))
        if len(out) == k:
            break
    return [frames[i] for i, _ in sorted(out, key=lambda x: x[1])]


def _visible(f, T, X, tol=0.3):
    """Is plan point X in front of the camera and not hidden behind a nearer surface in this frame's depth?"""
    uv, z = _project(f, T, X[None])
    if z[0] <= 0.1 or f.depth is None:
        return z[0] > 0.1
    D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
    h, w = cv2.imread(str(f.rgb)).shape[:2]
    i, j = int(uv[0, 1] * D.shape[0] / h), int(uv[0, 0] * D.shape[1] / w)
    if not (0 <= i < D.shape[0] and 0 <= j < D.shape[1]):
        return True                                  # outside the image: the box is clipped anyway
    d = D[i, j] / 1000.0
    return not (0 < d < z[0] - tol)


def _upright(img, f, T, X):
    """Rotate a frame by a multiple of 90 deg so that plan down (-z) at X points down in the image
    (Stray stores frames in the sensor's landscape orientation)."""
    uv, _ = _project(f, T, np.array([X, X - [0, 0, 0.3]]))
    dx, dy = uv[1] - uv[0]
    if abs(dy) >= abs(dx):
        return img if dy > 0 else cv2.rotate(img, cv2.ROTATE_180)
    return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE if dx < 0 else cv2.ROTATE_90_CLOCKWISE)


def _draw_box(img, f, T, p0, u, a, b, z0, z1, col, label=None):
    X = np.array([[*(p0 + u * a), z0], [*(p0 + u * b), z0], [*(p0 + u * b), z1], [*(p0 + u * a), z1]])
    # densify edges so lines bend correctly near the image border; drop points behind the camera
    E = np.concatenate([np.linspace(X[k], X[(k + 1) % 4], 20) for k in range(4)])
    uv, z = _project(f, T, E)
    ok = z > 0.1
    pts = uv[ok].astype(np.int32)
    if len(pts) > 1:
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], True, col, 2)
        if label:
            cv2.putText(img, label, tuple(int(v) for v in pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)


def _same_height(tiles, h=360):
    return [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]


def evidence_sheets(d: Path, openings: list[dict], walls_by_room: dict, frames, T):
    """debug/openings/<opening id>_frames.jpg: SHEET_FRAMES RGB frames that see each opening, the opening drawn
    (red door, blue window, green passage); debug/openings/walls_<room>.jpg: per wall of each room, 2 frames
    looking at its middle with the wall outlined in yellow and the detected openings drawn, for spotting
    openings that were missed. Needs ids on the openings (`id`)."""
    d.mkdir(parents=True, exist_ok=True)
    for old in list(d.glob("*_frames.jpg")) + list(d.glob("walls_*.jpg")):
        old.unlink()
    col = {k: v for k, v in KCOL.items() if k}
    for o in openings:
        u, c = np.array(o["u"]), np.array(o["center"])
        w = o["width"]["value"]
        n = np.array([u[1], -u[0]])
        X = np.array([*c, 0.5 * (o["z"][0] + o["z"][1])])
        tiles = []
        for f in _views(frames, T, X, n):
            img = cv2.imread(str(f.rgb))
            _draw_box(img, f, T, c, u, -w / 2, w / 2, o["z"][0], o["z"][1], col[o["kind"]])
            img = np.ascontiguousarray(_upright(img, f, T, X))
            cv2.putText(img, f"t={f.timestamp:.1f}s", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            tiles.append(img)
        if tiles:
            tiles = _same_height(tiles)
            band = np.full((24, sum(t.shape[1] for t in tiles), 3), 255, np.uint8)
            cv2.putText(band, f"{o['id']} {o['kind']} {w:.2f} m on {o['wall_id']} rooms={o['rooms']}", (4, 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.imwrite(str(d / f"{o['id']}_frames.jpg"), np.vstack([band, np.hstack(tiles)]))
    for rid, walls in walls_by_room.items():
        rows = []
        for w in walls:
            X = np.array([*(w["p0"] + w["u"] * w["L"] / 2), 1.2])
            tiles = []
            for f in _views(frames, T, X, w["n"], k=2):
                img = cv2.imread(str(f.rgb))
                _draw_box(img, f, T, w["p0"], w["u"], 0, w["L"], 0.0, 2.3, (0, 220, 255), w["id"])
                for o in openings:
                    if not _visible(f, T, np.array([*o["center"], 0.5 * (o["z"][0] + o["z"][1])])):
                        continue                    # hidden behind a wall in this frame: don't draw it
                    if o["wall_id"] == w["id"] or (o["rooms"] and w["room"] in o["rooms"]):
                        uo, co = np.array(o["u"]), np.array(o["center"])
                        ww = o["width"]["value"]
                        _draw_box(img, f, T, co, uo, -ww / 2, ww / 2, o["z"][0], o["z"][1], col[o["kind"]],
                                  o["id"])
                tiles.append(np.ascontiguousarray(_upright(img, f, T, X)))
            if not tiles:
                continue
            tiles = _same_height(tiles)
            while len(tiles) < 2:
                tiles.append(np.zeros_like(tiles[0]))
            row = np.hstack(tiles)
            band = np.full((20, row.shape[1], 3), 255, np.uint8)
            cv2.putText(band, f"{w['id']} L={w['L']:.2f} m {'observed' if w['observed'] else 'inferred'}", (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            rows.append(np.vstack([band, row]))
        if rows:
            W = max(r.shape[1] for r in rows)
            rows = [np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 255, np.uint8)]) for r in rows]
            cv2.imwrite(str(d / f"walls_{rid}.jpg"), np.vstack(rows))
