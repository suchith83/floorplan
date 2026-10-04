"""fp run <capture> [--out DIR] [--no-filter] [--as video|photos] [--max-frames N]

Auto-detects the input (ARKitScenes LiDAR folder, video file, photo folder) and writes plan.json, plan.svg
and report.html.
Every input goes through the same CaptureBundle -> point cloud -> geometry path; only the depth source differs."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from fp.bundle import CaptureBundle
from fp.geometry.align import FloorNotFound, align, to_plan
from fp.geometry.plan import extract_room, extract_rooms
from fp.ingest import load_capture
from fp.ingest.quality import select_frames
from fp.recon.lidar_fuse import fuse_depth
from fp.report.debug import bev_png
from fp.report.html import write_report
from fp.report.svg import render

SOURCE_PRIOR = {"lidar": 0.9, "video": 0.65, "photos": 0.5}
SCALE_METHOD = {"lidar": "LiDAR depth (metric)", "video": "MapAnything metric depth (learned)",
                "photos": "MapAnything metric depth (learned)"}
PREDICTED_MAX_DEPTH = 5.0  # m; predicted depth is less reliable far away
ASSUMED_CAMERA_H = 1.2    # m above the floor, used only when no floor is seen
FLOOR_ASSUMED_PENALTY = 0.5


def reconstruct(bundle: CaptureBundle, work: Path, filter_frames: bool = True):
    """Bundle -> (bundle with depth, point cloud, frame-filter counts)."""
    stats = {}
    if bundle.has_depth:
        if filter_frames:
            bundle.frames, stats = select_frames(bundle.frames)
        return bundle, fuse_depth(bundle), stats
    from fp.recon.mapanything import predict_depth
    bundle = predict_depth(bundle, work / "depth")
    return bundle, fuse_depth(bundle, max_depth=PREDICTED_MAX_DEPTH), stats


def run(capture: Path, out: Path, *, filter_frames: bool = True, as_source: str | None = None,
        max_frames: int | None = None, damage: bool = True, wet_rooms: set[str] = frozenset()) -> dict:
    t0 = time.time()
    bundle = load_capture(capture, as_source=as_source, max_frames=max_frames)
    bundle, rec, fstats = reconstruct(bundle, out, filter_frames)
    prior = SOURCE_PRIOR[bundle.source]
    try:
        al, floor = align(rec), "detected"
    except FloorNotFound as e:  # still draw the plan, but say so and halve the confidence (REVIEW #8)
        print(f"WARNING: {e}\n         Assuming the floor {ASSUMED_CAMERA_H} m below the camera.")
        al, floor = align(rec, assume_floor_below_camera=ASSUMED_CAMERA_H), "assumed (not observed)"
        prior *= FLOOR_ASSUMED_PENALTY
    T = al["T_plan_world"]
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    voxel = rec.meta.get("voxel", 0.02)
    rooms, conns, _ = extract_rooms(P, N, al["ceiling_h"], voxel, prior)
    layout = f"{len(rooms)} room(s) split at doorways"
    if not rooms:  # the split failed: fall back to the whole footprint as one room
        rooms, conns, layout = [extract_room(P, N, al["ceiling_h"], voxel, prior)], [], "single region (split failed)"
        rooms[0]["id"] = rooms[0]["name"] = "R1"
    _name_rooms(rooms, bundle, T)
    plan = {
        "schema": "fp.plan/0.3",
        "source": {"input": bundle.source, "scale": SCALE_METHOD[bundle.source], "n_frames": len(bundle.frames),
                   "gravity": al["up_source"], "floor": floor, "frame_filter": fstats,
                   **{k: v for k, v in bundle.meta.items() if not k.startswith("_")}},
        "layout": layout,
        "rooms": rooms,
        "connections": conns,
        "damage": [],
        "damage_meta": {"skipped": True},
        "debug": {"yaw_deg": al["yaw_deg"], "camera_h_m": round(al["camera_h"], 2), "n_points": len(P),
                  "runtime_s": round(time.time() - t0, 1)},
        "T_plan_world": T.round(6).tolist(),
    }
    out.mkdir(parents=True, exist_ok=True)
    if damage:
        _assess_damage(bundle, plan, out, wet_rooms)
    plan["debug"]["runtime_s"] = round(time.time() - t0, 1)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    (out / "plan.svg").write_text(render(plan))
    write_report(plan, out)
    bev_png(out / "debug_bev.png", P, N, plan, al["ceiling_h"])
    np.save(out / "points_plan.npy", np.c_[P, N].astype(np.float32))
    return plan


def _name_rooms(rooms, bundle, T):
    """Photo input in per-room folders (photos/kitchen/...): name each plan room after the folder most of
    the cameras inside it came from. Other inputs keep R1, R2, ..."""
    from shapely.geometry import Point, Polygon
    names = bundle.meta.get("_frame_rooms")
    if not names or not any(names):
        return
    cams = to_plan(T, np.array([f.T_wc[:3, 3] for f in bundle.frames]))
    for r in rooms:
        poly = Polygon(r["polygon"]).buffer(0.2)
        votes = [n for n, c in zip(names, cams) if n and poly.contains(Point(c[0], c[1]))]
        if votes:
            r["name"] = max(set(votes), key=votes.count)


def _assess_damage(bundle, plan, out, wet_rooms):
    """SAM 3 damage on surfaces + concealed-damage rules. A failure here never loses the floor plan."""
    from fp.damage.detect_modal import MODEL_ID, PROMPTS
    from fp.damage.project import assess
    from fp.damage.rules import concealed
    try:
        items, counts = assess(bundle, plan, out)
        concealed(items, plan, wet_rooms)
        plan["damage"] = items
        plan["damage_meta"] = {"model": f"{MODEL_ID}, zero-shot text prompts", "prompts": PROMPTS, **counts}
    except Exception as e:
        print(f"WARNING: damage step failed: {type(e).__name__}: {e}")
        plan["damage_meta"] = {"error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser(prog="fp")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="capture -> plan.json + plan.svg")
    r.add_argument("capture", type=Path)
    r.add_argument("--out", type=Path)
    r.add_argument("--no-filter", action="store_true", help="keep blurry and fast-turning LiDAR frames (the 'before' run)")
    r.add_argument("--as", dest="as_source", choices=["video", "photos"],
                   help="ARKitScenes only: use the RGB frames alone, as a phone video or photo set would")
    r.add_argument("--max-frames", type=int, help="cap on frames sent to MapAnything")
    r.add_argument("--no-damage", action="store_true", help="skip SAM 3 damage detection")
    r.add_argument("--wet-rooms", default="", help="comma-separated room ids that are bathrooms/kitchens, e.g. R2,R3")
    c = sub.add_parser("check", help="inspect one pipeline stage (see TESTING.md)")
    c.add_argument("stage", choices=["frames", "poses", "fusion", "cloud", "align", "walls", "footprint", "all", "view"])
    c.add_argument("capture", type=Path); c.add_argument("--out", type=Path)
    c.add_argument("--filter", action="store_true", help="apply the frame filter first, as `fp run` does")
    a = ap.parse_args()
    if a.cmd == "check":
        from fp import check
        return check.run(a.stage, a.capture, a.out, a.filter)
    name = a.capture.stem if a.capture.is_file() else a.capture.name
    out = a.out or Path("out") / (name + (f"-{a.as_source}" if a.as_source else "") + ("-nofilter" if a.no_filter else ""))
    plan = run(a.capture, out, filter_frames=not a.no_filter, as_source=a.as_source, max_frames=a.max_frames,
               damage=not a.no_damage, wet_rooms={r.strip() for r in a.wet_rooms.split(",") if r.strip()})
    for rm in plan["rooms"]:
        print(f"{out}/plan.svg  {rm['name']}: area={rm['area_m2']} m²  extents={rm['extent_x_m']}x{rm['extent_y_m']} m  "
              f"ceiling={rm['ceiling_h_m']}  walls={len(rm['walls'])}  t={plan['debug']['runtime_s']}s")
        for w in rm["walls"]:
            print(f"  {w['id']}: {w['length_m']:.3f} m  observed={w['observed']} cov={w['coverage']} "
                  f"spread={w['spread_cm']} cm  conf={w['confidence']}")
    for c in plan["connections"]:
        print(f"  {c['kind']} {c['rooms'][0]}-{c['rooms'][1]}: {c['width_m']} m")
    for d in plan["damage"]:
        print(f"  {d['id']} {d['type']} on {d['surface']} at {d['height_m']} m  views={d['views']} score={d['score']}"
              + "".join(f"\n     hypothesis {c['rule']}: {c['hypothesis']}" for c in d["concealed"]))
    if plan["source"]["frame_filter"]:
        print(f"frame filter: {plan['source']['frame_filter']}")


if __name__ == "__main__":
    main()
