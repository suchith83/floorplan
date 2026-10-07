"""Aligned point cloud -> room polygon(s) with wall-to-wall dimensions.

Method (explainable, Manhattan):
 1. Wall candidates = peaks in 1-D histograms of vertical-surface points along x and y.
 2. Room footprint = all horizontal surfaces (floor, counter/bed tops, ceiling) rasterised
    top-down: furniture hides the floor but its top surface is still inside the room. Plus, when the
    depth rays are given, every cell that several camera->point rays crossed: the sensor saw through
    that air, so it is inside the room even where the camera never looked down at the floor. And the
    camera path itself (CAMERA_RADIUS around it): the person stood on floor inside the room.
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


MIN_RAY_HITS = 3   # a cell is seen-through free space if rays from >= 3 frames crossed it (one stray ray,
                   # e.g. a mirror reflection, must not open a hole through a wall)


def _seen_through(rays, o, shape):
    """rays: list of (camera xyz, (n,3) measured points) per frame, plan coords. Returns, per RES cell, the
    number of frames whose top-down rays crossed it."""
    cnt = np.zeros(shape, np.uint16)
    frame = np.zeros(shape[::-1], np.uint8)          # cv2 draws in (row=j, col=i) order
    for cam, ends in rays:
        if not len(ends):
            continue
        frame[:] = 0
        c = tuple(((cam[:2] - o) / RES).astype(int))
        for e in ((ends[:, :2] - o) / RES).astype(int):
            cv2.line(frame, c, (int(e[0]), int(e[1])), 1, 1)
        cnt += frame.T
    return cnt


CAMERA_RADIUS = 0.3  # m: the phone is held ~0.3 m in front of the body; that floor is inside the room


def _footprint(P, N, H, rays=None, cams=None):
    # Every horizontal surface between floor and ceiling is inside the room: floor, counter
    # tops, bed tops, cabinet undersides, ceiling. Floor alone under-estimates kitchens.
    top = (H + 0.1) if H else 2.6
    hor = (np.abs(N[:, 2]) > 0.9) & (P[:, 2] > -0.08) & (P[:, 2] < top)
    Q = P[hor, :2]
    o = P[:, :2].min(0) - 0.2                        # grid over the whole cloud: rays end on walls, not floor
    img = np.zeros(((P[:, :2].max(0) - o) / RES).astype(int) + 1 + int(0.2 / RES), np.uint8)
    ij = ((Q - o) / RES).astype(int)
    img[ij[:, 0], ij[:, 1]] = 1
    if rays is not None and len(rays[0]):
        img |= (_seen_through(rays, o, img.shape) >= MIN_RAY_HITS).astype(np.uint8)
    if cams is not None and len(cams):
        path = np.zeros(img.shape[::-1], np.uint8)
        c = ((cams[:, :2] - o) / RES).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(path, [c], False, 1, 2 * int(CAMERA_RADIUS / RES))
        img |= path.T
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


def extract_room(P, N, ceiling_h, voxel, source_prior, debug: dict | None = None, rays=None, cams=None):
    """The whole footprint as ONE room (single-room scans, the laser eval, fp check).
    debug: if a dict is passed, intermediate results are stored in it."""
    lines = _wall_lines(P, N, ceiling_h, voxel)
    fp, o = _footprint(P, N, ceiling_h, rays, cams)
    if debug is not None:
        debug.update(lines=lines, footprint=fp, origin=o)
    return _room(fp, o, lines, ceiling_h, source_prior, debug)


TALL_BANDS = ((0.3, 0.9), (0.9, 1.5), (1.5, 2.1))  # m above the floor; furniture rarely reaches the top band
TALL_MIN_BANDS = 2    # a wall (or a full-height wardrobe) has vertical surface in at least 2 of the 3 bands
TALL_MIN_PTS = 3      # points of one band in a 2 cm cell; fewer is a stray return
DOOR_GAP = (0.5, 1.3)  # m: a gap this wide between two runs of tall wall on one wall line is a doorway
                       # (door leaves are 0.6-1.0 m; narrower gaps are pipes/columns, wider are open-plan)
MIN_RUN = 0.25        # m of tall evidence along a wall line on each side of a doorway (a jamb or more)
EXTERIOR_FLOOR_FRAC = 0.2  # a region the camera never entered with less than 20 % seen floor is outside air
                           # seen through a window or doorway (rays carve it, nobody stood there)
ADJ_GAP = 0.35        # m: two rooms whose edges are this close along a wall share that wall (partition <= 30 cm)
ADJ_MIN_LEN = 0.5     # m of shared wall to call two rooms adjacent
OVERLAP_TOL = 0.01    # polygon overlap above 1 % of the smaller room is repaired


def tall_wall_mask(P, N, o, shape):
    """Cells (RES grid at origin o) with vertical surface in >= TALL_MIN_BANDS height bands: walls and tall
    cupboards. Partition walls are 8-15 cm thick, so the footprint's 18 cm closing fills them; cutting this
    mask back out restores the walls between rooms."""
    vert = np.abs(N[:, 2]) < 0.3
    nb = np.zeros(shape, np.uint8)
    for lo, hi in TALL_BANDS:
        m = vert & (P[:, 2] > lo) & (P[:, 2] < hi)
        ij = ((P[m, :2] - o) / RES).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
        h = np.zeros(shape, np.int32)
        np.add.at(h, (ij[ok, 0], ij[ok, 1]), 1)
        nb += h >= TALL_MIN_PTS
    return nb >= TALL_MIN_BANDS


def close_doorways(tall, lines, o):
    """Along every wall line, a 0.5-1.3 m gap between two runs of tall wall is a doorway: bar it (3 cells
    thick) so the rooms on either side become separate regions, and remember it as a door between them.
    Returns (bar mask, [{axis, coord, a, b}] in plan metres)."""
    bar, gaps = np.zeros(tall.shape, bool), []
    for l in lines:
        ax = l["axis"]
        k = int(round((l["coord"] - o[ax]) / RES))
        if not 1 <= k < tall.shape[ax] - 1:
            continue
        along = tall.take(range(k - 1, k + 2), axis=ax).any(axis=ax)   # tall evidence within 2 cm of the line
        idx = np.nonzero(along)[0]
        if len(idx) < 2:
            continue
        br = np.nonzero(np.diff(idx) > 1)[0]
        runs = [(s, e) for s, e in zip(np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]])
                if (e - s + 1) * RES >= MIN_RUN]
        for (_, e0), (s1, _) in zip(runs, runs[1:]):
            if DOOR_GAP[0] <= (s1 - e0 - 1) * RES <= DOOR_GAP[1]:
                if ax == 0:
                    bar[k - 1:k + 2, e0 + 1:s1] = True
                else:
                    bar[e0 + 1:s1, k - 1:k + 2] = True
                gaps.append({"axis": ax, "coord": l["coord"], "a": o[1 - ax] + (e0 + 1) * RES,
                             "b": o[1 - ax] + s1 * RES})
    return bar, gaps


def _gap_doors(gaps, lab, o):
    """Each closed doorway -> a connection between the regions found just either side of it."""
    out = []
    for g in gaps:
        ax, m = g["axis"], 0.5 * (g["a"] + g["b"])
        side = []
        for sgn in (-1, 1):
            for d in np.arange(0.06, 0.5, 0.02):           # walk away from the bar until a room is hit
                p = np.array([g["coord"] + sgn * d, m] if ax == 0 else [m, g["coord"] + sgn * d])
                i, j = ((p - o) / RES).astype(int)
                if 0 <= i < lab.shape[0] and 0 <= j < lab.shape[1] and lab[i, j]:
                    side.append(int(lab[i, j]))
                    break
        if len(side) == 2 and side[0] != side[1]:
            c = [g["coord"], m] if ax == 0 else [m, g["coord"]]
            out.append({"rooms": [f"R{min(side)}", f"R{max(side)}"], "kind": "door",
                        "width_m": round(g["b"] - g["a"], 2), "center": [round(float(v), 3) for v in c]})
    return out


def extract_rooms(P, N, ceiling_h, voxel, source_prior, rays=None, cams=None):
    """The whole floor: footprint minus tall walls, doorways closed along wall lines, split into rooms
    (fp/geometry/rooms.py watershed for the open necks that remain), one rectilinear polygon per room on the
    wall lines, overlaps repaired, and the connections between rooms (doors and shared walls).
    Returns (rooms, connections, debug)."""
    from fp.geometry.rooms import doors, split_rooms
    lines = _wall_lines(P, N, ceiling_h, voxel)
    fp, o = _footprint(P, N, ceiling_h, rays, cams)
    tall = tall_wall_mask(P, N, o, fp.shape)
    bar, gaps = close_doorways(tall, lines, o)
    free = fp & ~ndimage.binary_dilation(tall) & ~bar
    free = ndimage.binary_opening(free, iterations=2)       # drop 1-2 cell slivers left along the walls
    lab = split_rooms(free, RES)
    floor = _surface_cells(P, N, ceiling_h, o, fp.shape)
    inside_cam = np.zeros(fp.shape, bool)
    if cams is not None and len(cams):
        ij = ((cams[:, :2] - o) / RES).astype(int)
        ok = (ij >= 0).all(1) & (ij[:, 0] < fp.shape[0]) & (ij[:, 1] < fp.shape[1])
        inside_cam[ij[ok, 0], ij[ok, 1]] = True
    rooms, masks, dropped = [], {}, []
    for k in range(1, lab.max() + 1):
        m = lab == k
        if not inside_cam[m].any() and floor[m].mean() < EXTERIOR_FLOOR_FRAC:
            dropped.append({"label": k, "area_m2": round(float(m.sum() * RES * RES), 2),
                            "floor_frac": round(float(floor[m].mean()), 2)})
            continue
        try:
            poly, near = _mask_polygon(m, o, lines)
        except (ValueError, IndexError):  # region too thin for a polygon: leave it out
            continue
        if poly.is_empty or poly.area <= 0 or len(poly.exterior.coords) < 4:
            continue
        rooms.append({"id": f"R{k}", "poly": poly, "lines": near})
        masks[f"R{k}"] = m
    overlaps = _repair_overlaps(rooms, masks, o)
    out = []
    for r in rooms:
        if r["poly"].is_empty or r["poly"].area <= 0:
            continue
        d = _room_from_polygon(r["poly"], r["lines"], ceiling_h, source_prior)
        d.update(id=r["id"], name=r["id"], width_m=polygon_width(r["poly"]))
        out.append(d)
    kept = {r["id"] for r in out}
    conns = [c for c in doors(lab, o, RES) + _gap_doors(gaps, lab, o) if set(c["rooms"]) <= kept]
    conns = _dedupe(conns) + _wall_adjacency(out, conns)
    return out, conns, {"lines": lines, "footprint": fp, "labels": lab, "origin": o, "tall": tall, "bar": bar,
                        "gaps": gaps, "dropped": dropped, "overlaps": overlaps}


def polygon_width(poly, tol=0.01) -> float:
    """Diameter of the widest circle that fits in the polygon (bisection on an inward offset). A corridor's is
    its width; the room polygon ignores furniture, which the free-space mask does not."""
    lo, hi = 0.0, 0.5 * min(poly.bounds[2] - poly.bounds[0], poly.bounds[3] - poly.bounds[1]) + tol
    while hi - lo > tol / 2:
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if not poly.buffer(-mid).is_empty else (lo, mid)
    return round(2 * lo, 2)


def _surface_cells(P, N, ceiling_h, o, shape):
    """Cells with an up-facing surface between the floor and the ceiling (floor, beds, counters)."""
    top = (ceiling_h + 0.1) if ceiling_h else 2.6
    m = (N[:, 2] > 0.9) & (P[:, 2] > -0.08) & (P[:, 2] < top)
    ij = ((P[m, :2] - o) / RES).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    out = np.zeros(shape, bool)
    out[ij[ok, 0], ij[ok, 1]] = True
    return ndimage.binary_closing(out, iterations=2)


def _repair_overlaps(rooms, masks, o):
    """Rooms are drawn on a shared wall-line grid, so two polygons can claim the same cell. An overlap above
    OVERLAP_TOL of the smaller room goes to the room whose own footprint (bounded by its walls) covers more of
    it; the other polygon loses it. Returns what was repaired, for the warnings."""
    from shapely import contains_xy
    out = []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            A, B = rooms[a], rooms[b]
            inter = A["poly"].intersection(B["poly"])
            small = min(A["poly"].area, B["poly"].area)
            if inter.area <= OVERLAP_TOL * small:
                continue
            x0, y0, x1, y1 = inter.bounds
            gx, gy = np.meshgrid(np.arange(x0, x1, RES) + RES / 2, np.arange(y0, y1, RES) + RES / 2, indexing="ij")
            ins = contains_xy(inter, gx, gy)
            i, j = ((gx[ins] - o[0]) / RES).astype(int), ((gy[ins] - o[1]) / RES).astype(int)
            score = lambda r: int(masks[r["id"]][np.clip(i, 0, masks[r["id"]].shape[0] - 1),
                                                np.clip(j, 0, masks[r["id"]].shape[1] - 1)].sum())
            win, lose = (A, B) if score(A) >= score(B) else (B, A)
            rest = lose["poly"].difference(win["poly"])
            if rest.geom_type == "MultiPolygon":
                rest = max(rest.geoms, key=lambda g: g.area)
            rest = orient(Polygon(rest.exterior).simplify(0.005), 1.0) if not rest.is_empty else rest
            out.append({"rooms": [win["id"], lose["id"]], "overlap_m2": round(inter.area, 2),
                        "frac_of_smaller": round(inter.area / small, 3)})
            lose["poly"] = rest
    return out


def _dedupe(conns, tol=0.5):
    """One connection per opening: the watershed border and a closed doorway can find the same door."""
    out = []
    for c in conns:
        if not any(set(c["rooms"]) == set(d["rooms"]) and np.hypot(*np.subtract(c["center"], d["center"])) < tol
                   for d in out):
            out.append(c)
    return out


def _wall_adjacency(rooms, conns):
    """Rooms that share a wall (edges within ADJ_GAP for >= ADJ_MIN_LEN) but have no door between them:
    adjacency without an opening."""
    have = {tuple(sorted(c["rooms"])) for c in conns}
    out = []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            A, B = (Polygon(rooms[k]["polygon"]) for k in (a, b))
            key = tuple(sorted((rooms[a]["id"], rooms[b]["id"])))
            if key in have:
                continue
            inter = A.buffer(ADJ_GAP / 2, join_style=2).intersection(B.buffer(ADJ_GAP / 2, join_style=2))
            if inter.is_empty:
                continue
            x0, y0, x1, y1 = inter.bounds
            if max(x1 - x0, y1 - y0) >= ADJ_MIN_LEN:
                c = inter.centroid
                out.append({"rooms": list(key), "kind": "wall", "width_m": None,
                            "center": [round(c.x, 3), round(c.y, 3)]})
    return out


def _crosses(line, x0, x1, y0, y1, margin=0.3):
    """Does a wall line lie within (or just outside) this room's box and span part of it?"""
    lo, hi, alo, ahi = (x0, x1, y0, y1) if line["axis"] == 0 else (y0, y1, x0, x1)
    o = line["other"]
    return lo - margin < line["coord"] < hi + margin and o.max() > alo and o.min() < ahi


def _room(fp, o, all_lines, H, source_prior, debug: dict | None = None):
    """One room's footprint mask -> rectilinear polygon whose edges sit on wall planes, and its walls."""
    poly, lines = _mask_polygon(fp, o, all_lines, debug)
    return _room_from_polygon(poly, lines, H, source_prior)


def _mask_polygon(fp, o, all_lines, debug: dict | None = None):
    """Footprint mask -> (rectilinear shapely polygon on the wall-line grid, the wall lines near it)."""
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
    return orient(Polygon(pts), 1.0), lines


def _room_from_polygon(poly, lines, H, source_prior):
    """Rectilinear polygon -> room dict: one wall per edge, matched to the wall line it sits on (observed) or not."""
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
