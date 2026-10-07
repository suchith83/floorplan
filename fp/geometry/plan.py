"""Aligned point cloud -> room polygon(s) with wall-to-wall dimensions.

Method (explainable, Manhattan):
 1. Wall candidates = peaks in 1-D histograms of vertical-surface points along x and y.
 2. Room footprint = all horizontal surfaces (floor, counter/bed tops, ceiling) rasterised
    top-down: furniture hides the floor but its top surface is still inside the room.
 3. Grid of candidate wall lines -> cells; a cell is interior if the footprint covers it.
    Union of interior cells = rectilinear polygon whose edges sit exactly on wall planes.
 4. An edge with no wall evidence is kept but marked inferred (dashed, low confidence)."""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

RES = 0.02          # BEV raster (m)
MIN_WALL_AREA = 0.25  # m^2 of vertical surface to count as a wall candidate
MERGE = 0.10        # merge wall lines closer than this (m)
MIN_JOG = 0.30      # polygon features narrower than this are noise (m)


def spread(v, c):
    """Robust wall thickness (m): 1.4826 x median absolute deviation of the points within +-15 cm of the
    plane. A sharp LiDAR wall is about 1 cm; smear from fast pans gives 4-8 cm (REVIEW #17)."""
    d = v[np.abs(v - c) < 0.15] - c
    return float(1.4826 * np.median(np.abs(d - np.median(d)))) if len(d) > 20 else 0.0


WALL_BAND = (0.3, 2.1)  # m above the floor: above skirting and low furniture, below the ceiling


def _wall_lines(P, N, H, voxel):
    # A fixed band, not one tied to the ceiling height: a wrong ceiling used to change which walls were found.
    zlo, zhi = WALL_BAND
    band = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
    lines = []
    for ax in (0, 1):
        m = band & (np.abs(N[:, ax]) > 0.9)
        v = P[m, ax]
        if len(v) == 0:
            continue
        bins = np.arange(v.min() - 0.05, v.max() + 0.05, 0.01)
        h, e = np.histogram(v, bins)
        h = np.convolve(h, np.ones(3), "same")
        for i in np.argsort(-h):
            if h[i] * voxel ** 2 < MIN_WALL_AREA:
                break
            c0 = 0.5 * (e[i] + e[i + 1])
            # one smeared wall shows as 2-3 peaks: merge within twice the stronger line's thickness (REVIEW #4)
            if any(l["axis"] == ax and abs(l["coord"] - c0) < max(MERGE, 2 * l["spread_m"]) for l in lines):
                continue
            near = m.copy(); near[m] = np.abs(v - c0) < 0.03
            c = float(np.median(P[near, ax]))
            lines.append({"axis": ax, "coord": c, "area_m2": float(near.sum() * voxel ** 2),
                          "spread_m": spread(v, c),
                          "faces": int(np.sign(N[near, ax].sum())),  # +1 = surface faces +axis (REVIEW #5)
                          "other": P[near, 1 - ax]})  # extent along the wall
    return lines


def _footprint(P, N, H):
    # Every horizontal surface between floor and ceiling is inside the room: floor, counter
    # tops, bed tops, cabinet undersides, ceiling. Floor alone under-estimates kitchens.
    top = (H + 0.1) if H else 2.6
    hor = (np.abs(N[:, 2]) > 0.9) & (P[:, 2] > -0.08) & (P[:, 2] < top)
    Q = P[hor, :2]
    o = Q.min(0) - 0.2
    ij = ((Q - o) / RES).astype(int)
    img = np.zeros(ij.max(0) + 1 + int(0.2 / RES), np.uint8)
    img[ij[:, 0], ij[:, 1]] = 1
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    img = ndimage.binary_fill_holes(img)
    lab, n = ndimage.label(img)
    if n > 1:
        img = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return img.astype(bool), o


def _grid(coords, lo, hi):
    vals = sorted([(c, True) for c in coords] + [(lo, False), (hi, False)])
    out = []
    for c, is_wall in vals:
        if out and c - out[-1][0] < MERGE:
            if is_wall and not out[-1][1]:
                out[-1] = (c, True)
            continue
        out.append((c, is_wall))
    return [c for c, _ in out]


def _coverage(line, a, b):
    """Fraction of segment [a,b] along the wall that has wall points (5 cm bins)."""
    if line is None or b - a < 1e-3:
        return 0.0
    o = line["other"]
    o = o[(o >= a) & (o <= b)]
    nb = max(1, int((b - a) / 0.05))
    hit = np.unique(np.clip(((o - a) / (b - a) * nb).astype(int), 0, nb - 1))
    return len(hit) / nb


def _support(lines, ax, c):
    l = min((l for l in lines if l["axis"] == ax), key=lambda l: abs(l["coord"] - c), default=None)
    return l["area_m2"] if l is not None and abs(l["coord"] - c) < 0.03 else 0.0


def _merge_steps(pts, lines):
    """Remove short connector edges (< MIN_JOG) between two parallel edges by snapping both
    onto the line with more wall evidence. Cabinet fronts / skirting create these steps."""
    pts = [np.array(p, float) for p in pts]
    changed = True
    while changed and len(pts) > 4:
        changed = False
        n = len(pts)
        for k in range(n):
            a, b = pts[k], pts[(k + 1) % n]
            if np.linalg.norm(b - a) >= MIN_JOG:
                continue
            # connector k is along axis 'ax_c'; its neighbours lie on lines of the other axis
            ax = 1 if abs(a[0] - b[0]) < 1e-6 else 0      # neighbours are x=c (0) or y=c (1)
            ca, cb = a[ax], b[ax]
            c = ca if _support(lines, ax, ca) >= _support(lines, ax, cb) else cb
            pts[(k - 1) % n][ax] = c
            pts[(k + 2) % n][ax] = c
            for i in sorted({k, (k + 1) % n}, reverse=True):
                pts.pop(i)
            changed = True
            break
        # drop collinear vertices
        pts = [p for i, p in enumerate(pts) if not _collinear(pts[i - 1], p, pts[(i + 1) % len(pts)])]
    return [tuple(p) for p in pts]


def _collinear(a, b, c):
    return abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])) < 1e-9


SHARP_WALL = 0.01   # m: thickness of a well-captured LiDAR wall
SMEAR_SCALE = 0.03  # m: each 3 cm of extra thickness divides confidence by e


def length_confidence(source_prior: float, nb_a: dict, nb_b: dict) -> float:
    """A wall's length is the distance between its two neighbours' planes, so its confidence comes from
    them: how much of each was seen (coverage) and how thick it is (smear, REVIEW #2)."""
    cov = min(nb_a["coverage"], nb_b["coverage"])
    smear = max(nb_a.get("spread_cm") or 0, nb_b.get("spread_cm") or 0) / 100
    return round(source_prior * (0.3 + 0.7 * cov) * float(np.exp(-max(0.0, smear - SHARP_WALL) / SMEAR_SCALE)), 2)


def extract_room(P, N, ceiling_h, voxel, source_prior, debug: dict | None = None):
    """The whole footprint as ONE room (single-room scans, the laser eval, fp check).
    debug: if a dict is passed, intermediate results are stored in it."""
    lines = _wall_lines(P, N, ceiling_h, voxel)
    fp, o = _footprint(P, N, ceiling_h)
    if debug is not None:
        debug.update(lines=lines, footprint=fp, origin=o)
    return _room(fp, o, lines, ceiling_h, source_prior, debug)


def extract_rooms(P, N, ceiling_h, voxel, source_prior):
    """The whole floor: footprint split into rooms at doorways (fp/geometry/rooms.py), one polygon per
    room on the shared wall lines, and the doors between rooms. Returns (rooms, connections, debug)."""
    from fp.geometry.rooms import doors, split_rooms
    lines = _wall_lines(P, N, ceiling_h, voxel)
    fp, o = _footprint(P, N, ceiling_h)
    lab = split_rooms(fp, RES)
    rooms = []
    for k in range(1, lab.max() + 1):
        try:
            r = _room(lab == k, o, lines, ceiling_h, source_prior)
        except (ValueError, IndexError):  # region too thin for a polygon: leave it out
            continue
        if len(r["polygon"]) < 3 or r["area_m2"] <= 0:
            continue
        r["id"] = r["name"] = f"R{k}"
        rooms.append(r)
    kept = {r["id"] for r in rooms}
    conns = [c for c in doors(lab, o, RES) if set(c["rooms"]) <= kept]
    return rooms, conns, {"lines": lines, "footprint": fp, "labels": lab, "origin": o}


def _crosses(line, x0, x1, y0, y1, margin=0.3):
    """Does a wall line lie within (or just outside) this room's box and span part of it?"""
    lo, hi, alo, ahi = (x0, x1, y0, y1) if line["axis"] == 0 else (y0, y1, x0, x1)
    o = line["other"]
    return lo - margin < line["coord"] < hi + margin and o.max() > alo and o.min() < ahi


def _room(fp, o, all_lines, H, source_prior, debug: dict | None = None):
    """One room's footprint mask -> rectilinear polygon whose edges sit on wall planes, and its walls."""
    ii, jj = np.nonzero(fp)
    x0, x1, y0, y1 = o[0] + ii.min() * RES, o[0] + ii.max() * RES, o[1] + jj.min() * RES, o[1] + jj.max() * RES
    lines = [l for l in all_lines if _crosses(l, x0, x1, y0, y1)]
    xs = _grid([l["coord"] for l in lines if l["axis"] == 0], x0, x1)
    ys = _grid([l["coord"] for l in lines if l["axis"] == 1], y0, y1)

    cells = []
    for cx0, cx1 in zip(xs[:-1], xs[1:]):
        for cy0, cy1 in zip(ys[:-1], ys[1:]):
            i0, i1 = int((cx0 - o[0]) / RES), int((cx1 - o[0]) / RES)
            j0, j1 = int((cy0 - o[1]) / RES), int((cy1 - o[1]) / RES)
            sub = fp[max(i0, 0):max(i1, i0 + 1), max(j0, 0):max(j1, j0 + 1)]
            if sub.size and sub.mean() > 0.5:
                cells.append(box(cx0, cy0, cx1, cy1))
    if debug is not None:
        debug.update(xs=xs, ys=ys, cells=cells)
    poly = unary_union(cells)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    # Morphological open/close with mitre joins removes jogs < 30 cm but keeps right angles.
    poly = poly.buffer(-MIN_JOG / 2, join_style=2).buffer(MIN_JOG, join_style=2).buffer(-MIN_JOG / 2, join_style=2)
    if poly.geom_type == "MultiPolygon":
        poly = max(poly.geoms, key=lambda g: g.area)
    poly = orient(Polygon(poly.exterior).simplify(0.005), 1.0)

    pts = _merge_steps(list(poly.exterior.coords)[:-1], lines)
    poly = orient(Polygon(pts), 1.0)
    pts = list(poly.exterior.coords)[:-1]
    walls = []
    for k in range(len(pts)):
        p0, p1 = np.array(pts[k]), np.array(pts[(k + 1) % len(pts)])
        ax = 0 if abs(p0[0] - p1[0]) < 1e-6 else 1          # edge lies on x=c (ax 0) or y=c
        c = p0[ax]
        line = min((l for l in lines if l["axis"] == ax), key=lambda l: abs(l["coord"] - c), default=None)
        if line is not None and abs(line["coord"] - c) > 0.03:
            line = None
        a, b = sorted([p0[1 - ax], p1[1 - ax]])
        cov = _coverage(line, a, b)
        walls.append({"id": f"W{k + 1}", "p0": p0.round(4).tolist(), "p1": p1.round(4).tolist(),
                      "length_m": round(float(b - a), 3), "axis": "x" if ax == 0 else "y",
                      "coord": round(float(c), 4), "observed": line is not None,
                      "coverage": round(cov, 2),
                      "spread_cm": None if line is None else round(100 * line["spread_m"], 1)})
    for k, w in enumerate(walls):
        w["confidence"] = length_confidence(source_prior, walls[k - 1], walls[(k + 1) % len(walls)])

    minx, miny, maxx, maxy = poly.bounds
    return {
        "polygon": [list(map(lambda v: round(v, 4), p)) for p in pts],
        "walls": walls,
        "area_m2": round(poly.area, 2),
        "extent_x_m": round(maxx - minx, 3),
        "extent_y_m": round(maxy - miny, 3),
        "ceiling_h_m": None if H is None else round(float(H), 3),
        "n_wall_candidates": len(lines),
    }
