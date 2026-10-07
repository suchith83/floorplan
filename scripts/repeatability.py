"""Repeatability of the LiDAR tier: two scans of the same flat -> same rooms, same dimensions?

    uv run python scripts/repeatability.py out/1a8384c3f6 out/c7d28f72c6 [--md eval/repeatability_lidar.md]

1. Register the scans' wall points top-down (fp/geometry/register2d.py: yaw from 0/90/180/270 + 2-D ICP).
   If they don't register as the same property, say so and stop.
2. Move scan B's rooms into A's frame; match rooms one-to-one by centroid (Hungarian, <= MATCH_M apart).
3. Per matched room: width (x extent), length (y extent), area in both scans and the difference.
4. Wall-to-wall spans (what a tape measures): in each matched room, wall planes found in both scans are
   paired (same axis, <= WALL_MATCH_M apart); every two opposite matched planes give a span in A and in B.
5. Per wall: matched walls (same axis in A's frame, planes <= WALL_MATCH_M apart, overlapping >= 50 %)
   compared by length. Gate (work order 03): |difference| <= max(1 cm, 0.5 % of the length).
Numbers are the pipeline's own outputs; nothing here is ground truth."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon

from fp.geometry.register2d import register

MATCH_M = 1.0        # m between matched room centroids
WALL_MATCH_M = 0.10  # m between matched wall planes
GATE_ABS, GATE_REL = 0.01, 0.005


def _load(d: Path):
    plan = json.loads((d / "plan.json").read_text())
    return plan, np.load(d / "debug" / "wall_points_2d.npy").astype(float)


def _walls(room, R, t):
    out = []
    for w in room["walls"]:
        p0, p1 = np.array(w["p0"]) @ R.T + t, np.array(w["p1"]) @ R.T + t
        ax = 0 if abs(p0[0] - p1[0]) < abs(p0[1] - p1[1]) else 1      # 0: wall runs along y (x = c)
        out.append({"id": w["id"], "axis": ax, "c": 0.5 * (p0[ax] + p1[ax]),
                    "a": min(p0[1 - ax], p1[1 - ax]), "b": max(p0[1 - ax], p1[1 - ax]),
                    "len": w["length"]["value"], "observed": w["observed"]})
    return out


MIN_SPAN = 0.5  # m: shorter plane pairs are jambs and cupboard faces, not wall-to-wall distances


def _spans(Wa, Wb, ida, idb):
    """Distances between opposite wall planes found in both scans of one room (observed walls only)."""
    out, seen = [], set()
    for ax in (0, 1):
        pa = sorted({round(w["c"], 3) for w in Wa if w["axis"] == ax and w["observed"]})
        pb = sorted({round(w["c"], 3) for w in Wb if w["axis"] == ax and w["observed"]})
        pairs = [(c, min(pb, key=lambda d: abs(d - c))) for c in pa if pb and min(abs(d - c) for d in pb) <= WALL_MATCH_M]
        for k in range(len(pairs)):
            for m in range(k + 1, len(pairs)):
                (a1, b1), (a2, b2) = pairs[k], pairs[m]
                sa, sb = abs(a2 - a1), abs(b2 - b1)
                if sa < MIN_SPAN or b1 == b2 or (ax, a1, a2) in seen:
                    continue
                seen.add((ax, a1, a2))
                out.append({"a": ida, "b": idb, "axis": "x" if ax == 0 else "y", "span_a": round(sa, 3),
                            "span_b": round(sb, 3), "diff_cm": round(100 * (sb - sa), 1),
                            "pass": abs(sb - sa) <= max(GATE_ABS, GATE_REL * sa)})
    return out


def compare(dir_a: Path, dir_b: Path) -> dict:
    A, wa = _load(dir_a)
    B, wb = _load(dir_b)
    reg = register(wa, wb)
    res = {"a": A["capture_id"], "b": B["capture_id"], "registration": {k: v for k, v in reg.items() if k not in ("R", "t")}}
    res["registration"]["t"] = [round(float(v), 3) for v in reg["t"]]
    if not reg["same_property"]:
        return res
    R, t = reg["R"], reg["t"]
    pa = [Polygon(r["polygon"]) for r in A["rooms"]]
    pb = [Polygon(np.array(r["polygon"]) @ R.T + t) for r in B["rooms"]]
    D = np.array([[p.centroid.distance(q.centroid) for q in pb] for p in pa])
    ia, ib = linear_sum_assignment(D)
    rooms, walls, spans = [], [], []
    for i, j in zip(ia, ib):
        if D[i, j] > MATCH_M:
            continue
        ra, rb = A["rooms"][i], B["rooms"][j]
        ext = lambda p: (p.bounds[2] - p.bounds[0], p.bounds[3] - p.bounds[1])
        (xa, ya), (xb, yb) = ext(pa[i]), ext(pb[j])
        rooms.append({"a": ra["id"], "b": rb["id"], "name": ra["name"], "centroid_gap_m": round(D[i, j], 2),
                      "x_a": xa, "x_b": xb, "y_a": ya, "y_b": yb,
                      "area_a": ra["floor_area"]["value"], "area_b": rb["floor_area"]["value"],
                      "iou": round(pa[i].intersection(pb[j]).area / pa[i].union(pb[j]).area, 2)})
        Wa, Wb = _walls(ra, np.eye(2), np.zeros(2)), _walls(rb, R, t)
        for w in Wa:
            cand = [v for v in Wb if v["axis"] == w["axis"] and abs(v["c"] - w["c"]) <= WALL_MATCH_M
                    and min(w["b"], v["b"]) - max(w["a"], v["a"]) >= 0.5 * min(w["b"] - w["a"], v["b"] - v["a"])]
            if not cand:
                continue
            v = min(cand, key=lambda v: abs(v["c"] - w["c"]))
            d = v["len"] - w["len"]
            walls.append({"a": f"{ra['id']}.{w['id'].split('.')[-1]}", "b": f"{rb['id']}.{v['id'].split('.')[-1]}",
                          "len_a": w["len"], "len_b": v["len"], "diff_cm": round(100 * d, 1),
                          "plane_gap_cm": round(100 * abs(v["c"] - w["c"]), 1),
                          "both_observed": w["observed"] and v["observed"],
                          "pass": abs(d) <= max(GATE_ABS, GATE_REL * w["len"])})
        spans += _spans(Wa, Wb, ra["id"], rb["id"])
    res.update(rooms=rooms, walls=walls, spans=spans, n_rooms=(len(A["rooms"]), len(B["rooms"])),
               drift=(A["drift"], B["drift"]))
    return res


def markdown(r: dict) -> str:
    g = r["registration"]
    L = [f"# Repeatability, LiDAR tier: `{r['a']}` vs `{r['b']}`", "",
         "Generated by `uv run python scripts/repeatability.py out/" + r["a"] + " out/" + r["b"] + "`. Both are the "
         "pipeline's own outputs on the evaluator's sample captures; neither is ground truth.", "",
         "## 1. Same flat?", "",
         f"- 2-D registration of wall points (yaw from 0/90/180/270, then 2-D ICP): yaw **{g['yaw_deg']:.2f}°**, "
         f"shift ({g['t'][0]:.2f}, {g['t'][1]:.2f}) m.",
         f"- **{100 * g['inliers']:.0f} %** of scan B's wall points land within 5 cm of scan A's walls "
         f"(median {g['median_inlier_cm']:.1f} cm among them); the next-best start yaw ({g['runner_up_yaw0']}°) "
         f"gets {100 * g['runner_up_inliers']:.0f} %.",
         f"- Verdict: **{'same flat' if g['same_property'] else 'NOT shown to be the same flat; stopping here'}**.", ""]
    if not g["same_property"]:
        return "\n".join(L)
    rooms, walls = r["rooms"], r["walls"]
    L += ["## 2. Rooms matched by centroid", "",
          f"{r['n_rooms'][0]} rooms in A, {r['n_rooms'][1]} in B; {len(rooms)} matched (centroids ≤ {MATCH_M} m apart "
          "after registration). Width = x extent, length = y extent in A's frame.", "",
          "| A | B | name (A) | width A | width B | Δ | length A | length B | Δ | area A | area B | Δ | IoU |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in rooms:
        L.append(f"| {m['a']} | {m['b']} | {m['name']} | {m['x_a']:.2f} | {m['x_b']:.2f} | {100 * (m['x_b'] - m['x_a']):+.0f} cm "
                 f"| {m['y_a']:.2f} | {m['y_b']:.2f} | {100 * (m['y_b'] - m['y_a']):+.0f} cm "
                 f"| {m['area_a']:.2f} | {m['area_b']:.2f} | {m['area_b'] - m['area_a']:+.2f} m² | {m['iou']:.2f} |")
    sp = r["spans"]
    ds = np.abs([x["diff_cm"] for x in sp]) if sp else np.array([np.nan])
    L += ["", "## 3. Wall-to-wall spans (gate: |Δ| ≤ max(1 cm, 0.5 %))", "",
          "What a tape measures: the distance between two opposite wall planes that **both** scans found in the "
          "same room (observed walls only, planes matched within "
          f"{100 * WALL_MATCH_M:.0f} cm after registration, spans ≥ {MIN_SPAN} m). This isolates measurement "
          "repeatability from segmentation (where each scan ends a room).", "",
          f"- Spans compared: **{len(sp)}**; within the gate: **{sum(x['pass'] for x in sp)} / {len(sp)}** "
          f"({100 * sum(x['pass'] for x in sp) / max(len(sp), 1):.0f} %).",
          f"- |Δ span|: median **{np.median(ds):.1f} cm**, 90th percentile {np.percentile(ds, 90):.1f} cm, max {np.max(ds):.1f} cm.", "",
          "| room A | room B | axis | span A | span B | Δ cm | gate |", "|---|---|---|---|---|---|---|"]
    for x in sorted(sp, key=lambda x: (x["a"], x["axis"], x["span_a"])):
        L.append(f"| {x['a']} | {x['b']} | {x['axis']} | {x['span_a']:.3f} | {x['span_b']:.3f} | {x['diff_cm']:+.1f} | "
                 f"{'pass' if x['pass'] else 'fail'} |")
    n_pass = sum(w["pass"] for w in walls)
    obs = [w for w in walls if w["both_observed"]]
    d = np.abs([w["diff_cm"] for w in walls]) if walls else np.array([np.nan])
    do = np.abs([w["diff_cm"] for w in obs]) if obs else np.array([np.nan])
    L += ["", "## 4. Per-wall polygon edge lengths (gate: |Δ| ≤ max(1 cm, 0.5 %))", "",
          "An edge's length runs between its two neighbouring walls, so it changes whenever the two scans split "
          "rooms differently (a doorway assigned to the corridor in one scan and to the room in the other), even "
          "when every wall plane agrees. Reported as-is.", "",
          f"- Walls matched (same axis, planes ≤ {100 * WALL_MATCH_M:.0f} cm apart, ≥ 50 % overlap): **{len(walls)}**.",
          f"- Within the gate: **{n_pass} / {len(walls)}** ({100 * n_pass / max(len(walls), 1):.0f} %).",
          f"- |Δ length|: median **{np.median(d):.1f} cm**, 90th percentile {np.percentile(d, 90):.1f} cm; "
          f"walls observed in both scans only ({len(obs)}): median {np.median(do):.1f} cm.",
          f"- Wall planes, matched: median gap {np.median([w['plane_gap_cm'] for w in walls]) if walls else float('nan'):.1f} cm after registration.", "",
          "| wall A | wall B | length A | length B | Δ cm | plane gap cm | both seen | gate |", "|---|---|---|---|---|---|---|---|"]
    for w in sorted(walls, key=lambda w: w["a"]):
        L.append(f"| {w['a']} | {w['b']} | {w['len_a']:.3f} | {w['len_b']:.3f} | {w['diff_cm']:+.1f} | {w['plane_gap_cm']:.1f} "
                 f"| {'yes' if w['both_observed'] else 'no'} | {'pass' if w['pass'] else 'fail'} |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", type=Path)
    ap.add_argument("b", type=Path)
    ap.add_argument("--md", type=Path)
    ap.add_argument("--json", type=Path)
    a = ap.parse_args()
    r = compare(a.a, a.b)
    md = markdown(r)
    print(md)
    if a.md:
        a.md.write_text(md)
    if a.json:
        a.json.write_text(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
