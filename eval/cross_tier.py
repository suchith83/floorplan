"""Compare a camera-tier plan (video or photos) with the LiDAR-tier plan of the same capture, room by room.

The LiDAR plan is a REFERENCE, not ground truth: it has its own errors (drift, merged rooms, no ceiling on
some scans). This tells us how far the camera tiers are from it, not how far either is from the truth.

How plans are put on top of each other (they come out in different frames):
 1. Both plans are already floor at z = 0 and x/y on their dominant wall axes, so only a yaw that is a
    multiple of 90 deg and a 2D translation are unknown. No scale is fitted: scale error is what we measure.
 2. For each of the 4 yaws, a translation grid search (coarse then fine) maximises the IoU of the two
    footprints (union of room polygons). The best yaw + translation wins.
 3. Rooms are matched one-to-one by polygon IoU (Hungarian); a pair with IoU < MATCH_IOU is unmatched.
 4. Walls of matched rooms are matched by direction (parallel) and nearest midpoint (< WALL_MATCH_M).

    uv run python eval/cross_tier.py out/c00a170fe1-lidar-ref/plan.json out/c00a170fe1-video/plan.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely import affinity
from shapely.geometry import Polygon
from shapely.ops import unary_union

MATCH_IOU = 0.2      # rooms overlapping less than this after alignment are not the same room
WALL_MATCH_M = 0.5   # m, a test wall's midpoint must be this close to the reference wall's line to be its match
COARSE = (1.5, 0.1)  # m, translation search half-width and step around the centroid offset
FINE = (0.12, 0.01)


def _polys(plan):
    return {r["id"]: Polygon(r["polygon"]) for r in plan["rooms"]}


def _iou(a, b):
    u = a.union(b).area
    return a.intersection(b).area / u if u > 0 else 0.0


def align(ref, test):
    """Best (yaw_deg, dx, dy, iou) moving `test` onto `ref`."""
    A = unary_union(list(_polys(ref).values())).buffer(0)
    B0 = unary_union(list(_polys(test).values())).buffer(0)
    best = (0, 0.0, 0.0, -1.0)
    for yaw in (0, 90, 180, 270):
        B = affinity.rotate(B0, yaw, origin=(0, 0))
        c = np.array(A.centroid.coords[0]) - np.array(B.centroid.coords[0])
        for half, step in (COARSE, FINE):
            grid = np.arange(-half, half + 1e-9, step)
            scores = [(_iou(A, affinity.translate(B, c[0] + x, c[1] + y)), c[0] + x, c[1] + y) for x in grid for y in grid]
            s, dx, dy = max(scores)
            c = np.array([dx, dy])
        if s > best[3]:
            best = (yaw, float(c[0]), float(c[1]), float(s))
    return best


def _move(geom_xy, yaw, dx, dy):
    t = np.radians(yaw)
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return np.asarray(geom_xy, float) @ R.T + [dx, dy]


def compare(ref, test) -> dict:
    yaw, dx, dy, fiou = align(ref, test)
    P = _polys(ref)
    Q = {k: Polygon(_move(v.exterior.coords, yaw, dx, dy)) for k, v in _polys(test).items()}
    rk, tk = list(P), list(Q)
    M = np.array([[_iou(P[a], Q[b]) for b in tk] for a in rk]) if rk and tk else np.zeros((len(rk), len(tk)))
    pairs = []
    if M.size:
        ri, ti = linear_sum_assignment(-M)
        pairs = [(rk[i], tk[j], float(M[i, j])) for i, j in zip(ri, ti) if M[i, j] >= MATCH_IOU]
    R = {r["id"]: r for r in ref["rooms"]}
    T = {r["id"]: r for r in test["rooms"]}
    rows = []
    for a, b, iou in pairs:
        ra, tb = R[a], T[b]
        row = {"ref": a, "test": b, "test_name": tb.get("name"), "iou": round(iou, 3)}
        for key in ("floor_area", "perimeter", "ceiling_height"):
            va, vb = ra[key]["value"], tb[key]["value"]
            row[key] = {"ref": va, "test": vb, "test_lo": tb[key]["lo"], "test_hi": tb[key]["hi"],
                        "rel_err": None if va is None or vb is None else round((vb - va) / va, 4),
                        "ref_in_test_interval": None if va is None or vb is None else bool(tb[key]["lo"] <= va <= tb[key]["hi"])}
        row["walls"] = _walls(ra, tb, yaw, dx, dy)
        rows.append(row)
    room_map = {b: a for a, b, _ in pairs}
    adj = lambda plan, f=lambda x: x: {tuple(sorted(f(r) for r in c["rooms"])) for c in plan["connections"]}
    ref_adj = adj(ref)
    test_adj = {tuple(sorted(room_map.get(r, f"?{r}") for r in c)) for c in adj(test)}
    fa = lambda p: p["footprint_area"]["value"]
    return {"align": {"yaw_deg": yaw, "dx": round(dx, 3), "dy": round(dy, 3), "footprint_iou": round(fiou, 3)},
            "rooms_ref": len(P), "rooms_test": len(Q), "unmatched_ref": sorted(set(rk) - {a for a, _, _ in pairs}),
            "unmatched_test": sorted(set(tk) - {b for _, b, _ in pairs}), "rooms": rows,
            "footprint_area": {"ref": fa(ref), "test": fa(test),
                               "rel_err": None if not fa(ref) or fa(test) is None else round((fa(test) - fa(ref)) / fa(ref), 4)},
            "adjacency": {"ref": sorted(ref_adj), "test": sorted(test_adj), "common": sorted(ref_adj & test_adj)},
            "overlap_test_m2": round(_overlap(list(Q.values())), 3)}


def _overlap(polys):
    """Total pairwise overlap area between the test plan's rooms (should be ~0: rooms must not overlap)."""
    return sum(polys[i].intersection(polys[j]).area for i in range(len(polys)) for j in range(i + 1, len(polys)))


def _walls(ra, tb, yaw, dx, dy):
    """Pair each reference wall with the parallel test wall whose midpoint is nearest its line."""
    out = []
    tw = []
    for w in tb["walls"]:
        p0, p1 = _move([w["p0"], w["p1"]], yaw, dx, dy)
        tw.append((w, p0, p1))
    for w in ra["walls"]:
        a0, a1 = np.array(w["p0"]), np.array(w["p1"])
        d = (a1 - a0) / (np.linalg.norm(a1 - a0) + 1e-9)
        n = np.array([-d[1], d[0]])
        best = None
        for v, b0, b1 in tw:
            e = (b1 - b0) / (np.linalg.norm(b1 - b0) + 1e-9)
            if abs(e @ d) < 0.9:
                continue
            m = 0.5 * (b0 + b1)
            off = abs((m - a0) @ n)
            along = (m - a0) @ d
            if off < WALL_MATCH_M and -0.5 < along < np.linalg.norm(a1 - a0) + 0.5 and (best is None or off < best[1]):
                best = (v, off)
        if best:
            v = best[0]
            la, lb = w["length"]["value"], v["length"]["value"]
            out.append({"ref": w["id"], "test": v["id"], "ref_len": la, "test_len": lb, "err_m": round(lb - la, 3),
                        "offset_m": round(best[1], 3), "ref_in_test_interval": bool(v["length"]["lo"] <= la <= v["length"]["hi"])})
    return out


def summary(c) -> dict:
    errs = [abs(w["err_m"]) for r in c["rooms"] for w in r["walls"]]
    cover = [w["ref_in_test_interval"] for r in c["rooms"] for w in r["walls"]]
    return {"walls_matched": len(errs), "wall_len_median_abs_err_m": round(float(np.median(errs)), 3) if errs else None,
            "wall_len_p90_abs_err_m": round(float(np.percentile(errs, 90)), 3) if errs else None,
            "ref_in_interval_frac": round(float(np.mean(cover)), 2) if cover else None}


if __name__ == "__main__":
    ref, test = (json.loads(Path(p).read_text()) for p in sys.argv[1:3])
    c = compare(ref, test)
    c["summary"] = summary(c)
    print(json.dumps(c, indent=1))
