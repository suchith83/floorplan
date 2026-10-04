"""Room size against ARKitScenes laser ground truth, without sharing a coordinate frame.

Laser side of a room = the distance between the strongest wall plane facing +axis and the strongest
facing -axis, found by our wall-line detector run on the laser cloud alone (highres_depth, rendered
from a Faro scan). Ours = the plan's reported short / long side, wall to wall, used only when both
bounding walls were observed. Each laser measure is matched to our side of nearest length (and only
within 25 %), so camera-only runs, whose frame is MapAnything's, are compared the same way.
This replaces the +-15 cm search around our own walls, which hid big errors (REVIEW #3).

Usage: uv run python eval/arkit_gt.py <scene_dir> <out_dir with plan.json>"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from fp.geometry.align import FloorNotFound, align, to_plan
from fp.geometry.plan import _wall_lines
from fp.ingest import arkitscenes
from fp.recon.lidar_fuse import backproject

CACHE = Path("out/_gt")
MATCH_TOL = 0.25


def gt_sides(scene: Path) -> dict:
    """Laser wall-to-wall distances per axis, cached. A laser scan covers part of a room only, so an
    axis without walls facing both ways has no measure."""
    scene = Path(scene)
    cache = CACHE / f"{scene.name}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    frames = arkitscenes.load_gt_frames(scene, stride=2)  # every 2nd laser frame
    rec = backproject(frames, np.array([0, 1.0, 0]), pix_stride=4, voxel=0.01)
    try:
        al, floor = align(rec), "detected"
    except FloorNotFound:  # only the wall height band needs the floor: assume a 1.0 m handheld camera
        al, floor = align(rec, assume_floor_below_camera=1.0), "assumed 1.0 m below the camera"
    T = al["T_plan_world"]
    lines = _wall_lines(to_plan(T, rec.points), rec.normals @ T[:3, :3].T, al["ceiling_h"], 0.01)
    measures = []
    for ax in (0, 1):
        pos = [l for l in lines if l["axis"] == ax and l["faces"] > 0]
        neg = [l for l in lines if l["axis"] == ax and l["faces"] < 0]
        if pos and neg:
            a, b = max(pos, key=lambda l: l["area_m2"]), max(neg, key=lambda l: l["area_m2"])
            if b["coord"] > a["coord"]:
                measures.append({"axis": "xy"[ax], "laser_m": round(b["coord"] - a["coord"], 3),
                                 "walls": [round(a["coord"], 3), round(b["coord"], 3)]})
    out = {"scene": scene.name, "floor": floor, "laser_frames": len(frames), "measures": measures,
           "lines": [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in l.items() if k != "other"} for l in lines]}
    CACHE.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out, indent=1))
    return out


def sides(room: dict) -> list[dict]:
    """The plan's two sides: wall-to-wall extent along x and y, whether both bounding walls were
    observed, and the confidence of the walls whose length it is."""
    out = []
    for name in ("x", "y"):
        bound = [w for w in room["walls"] if w["axis"] == name]   # walls lying on x=c (or y=c) planes
        lo, hi = min(bound, key=lambda w: w["coord"]), max(bound, key=lambda w: w["coord"])
        along = [w for w in room["walls"] if w["axis"] != name]   # walls whose length spans this side
        out.append({"extent_m": round(hi["coord"] - lo["coord"], 3), "observed": lo["observed"] and hi["observed"],
                    "confidence": max((w["confidence"] for w in along), default=None),
                    "spread_cm": max((w.get("spread_cm") or 0) for w in (lo, hi))})
    return out


def compare(room: dict, gt: dict) -> list[dict]:
    rows = []
    ours = [s for s in sides(room) if s["observed"]]
    for m in gt["measures"]:
        best = min(ours, key=lambda s: abs(s["extent_m"] - m["laser_m"]), default=None)
        if best is None or abs(best["extent_m"] - m["laser_m"]) > MATCH_TOL * m["laser_m"]:
            rows.append({"laser_m": m["laser_m"], "pred_m": None, "err_cm": None, "confidence": None, "spread_cm": None})
            continue
        rows.append({"laser_m": m["laser_m"], "pred_m": best["extent_m"],
                     "err_cm": round(100 * (best["extent_m"] - m["laser_m"]), 1),
                     "confidence": best["confidence"], "spread_cm": best["spread_cm"]})
    return rows


def main(scene, out):
    plan = json.loads((Path(out) / "plan.json").read_text())
    gt = gt_sides(Path(scene))
    rows = [r | {"room": rm["name"]} for rm in plan["rooms"] for r in compare(rm, gt)]
    (Path(out) / "gt_compare.json").write_text(json.dumps(rows, indent=1))
    print(f"laser floor: {gt['floor']}")
    print(f"{'room':<8}{'laser m':>9}{'ours m':>9}{'err cm':>8}{'conf':>6}")
    for r in rows:
        print(f"{r['room']:<8}{r['laser_m']:>9}{str(r['pred_m']):>9}{str(r['err_cm']):>8}{str(r['confidence']):>6}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
