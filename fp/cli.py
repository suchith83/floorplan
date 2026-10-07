"""fp run <capture> [--out DIR] [--tier auto|lidar|video|photos] [--backend local|modal] [--no-drift-correction] [--no-cache]
fp render <plan.json> [--out DIR]
fp check <stage> <capture>

`fp run` detects the tier (Stray Scanner folder -> lidar, video file -> video, folder of room folders of photos ->
photos) and always writes plan.json (schema 1.0, validated), plan.svg, report.html and debug images under
out/<capture>/. A stage that isn't built yet leaves its fields `observed: false` with a warning, so the output
contract holds from day one. Every tier goes through the same CaptureBundle -> point cloud -> geometry path."""
from __future__ import annotations

import argparse
import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from fp import contract
from fp.bundle import CaptureBundle
from fp.contract import StageNotBuilt

PREDICTED_MAX_DEPTH = 5.0  # m; predicted depth is less reliable far away
ASSUMED_CAMERA_H = 1.2    # m above the floor, used only when no floor is seen


@contextmanager
def _timed(timings: dict, stage: str):
    t = time.perf_counter()
    try:
        yield
    finally:
        timings[stage] = round(time.perf_counter() - t, 2)


def reconstruct(bundle: CaptureBundle, work: Path, filter_frames: bool = True, backend: str = "local"):
    """Bundle -> (bundle with depth, point cloud, frame-filter counts)."""
    from fp.ingest.quality import select_frames
    from fp.recon.lidar_fuse import fuse_depth
    stats = {}
    if bundle.has_depth:
        if filter_frames:
            bundle.frames, stats = select_frames(bundle.frames)
        return bundle, fuse_depth(bundle), stats
    if backend == "local":
        raise StageNotBuilt("recon", "Local MapAnything backend not built yet (work order 05); "
                                     "camera-only tiers need --backend modal for now.")
    from fp.recon.mapanything import predict_depth
    bundle = predict_depth(bundle, work / "depth")
    return bundle, fuse_depth(bundle, max_depth=PREDICTED_MAX_DEPTH), stats


def run(capture: Path, out: Path, *, tier: str = "auto", backend: str = "local", drift_correction: bool = True,
        cache: bool = True, filter_frames: bool = True, max_frames: int | None = None, damage: bool = True,
        wet_rooms: set[str] = frozenset()) -> dict:
    from fp.ingest import detect_tier
    from fp.schema import validate
    t0 = time.perf_counter()
    capture = Path(capture)
    tier = detect_tier(capture) if tier == "auto" else tier
    plan = contract.empty_plan(tier, contract.capture_id(capture), source={
        "path": str(capture), "scale": contract.TIER_NAME[tier], "backend": backend, "cache": cache})
    warn = lambda stage, msg: plan["warnings"].append({"stage": stage, "message": msg})
    timings = plan["timings"]
    debug = out / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    plan["drift"]["method"] = ("disabled (--no-drift-correction): poses used as-is" if not drift_correction
                               else "not built yet (work order 03): poses used as-is")
    if drift_correction:
        warn("drift", "Drift correction not built yet (work order 03); poses used as-is.")
    try:
        legacy = _geometry(capture, out, plan, timings, tier, backend, filter_frames, max_frames, warn)
        if damage and backend == "modal":
            with _timed(timings, "damage"):
                _assess_damage(legacy, plan, out, wet_rooms, warn)
        elif damage:
            warn("damage", "Local damage backend not built yet (work order 06); damage not assessed.")
    except StageNotBuilt as e:
        warn(e.stage, str(e))
    if not plan["rooms"]:
        warn("rooms", "No rooms reconstructed.")
    warn("intervals", "Intervals are provisional (fp/contract.py), not yet calibrated on ground truth (work order 07).")
    timings["total"] = round(time.perf_counter() - t0, 2)
    plan = validate(plan).model_dump(mode="json", by_alias=True)
    _write(plan, out)
    return plan


def _geometry(capture, out, plan, timings, tier, backend, filter_frames, max_frames, warn) -> dict:
    """Ingest -> recon -> align -> rooms, filling `plan` in place. Returns the internal (pre-schema) result
    that the damage step still works on."""
    from fp.geometry.align import FloorNotFound, align, to_plan
    from fp.geometry.plan import extract_room, extract_rooms
    from fp.ingest import load_capture
    from fp.report.debug import bev_png
    with _timed(timings, "ingest"):
        bundle = load_capture(capture, tier=tier, max_frames=max_frames)
    with _timed(timings, "recon"):
        bundle, rec, fstats = reconstruct(bundle, out, filter_frames, backend)
    plan["source"].update(n_frames=len(bundle.frames), frame_filter=fstats,
                          **{k: v for k, v in bundle.meta.items() if not k.startswith("_")})
    with _timed(timings, "align"):
        try:
            al = align(rec)
        except FloorNotFound as e:
            warn("align", f"{e} Assuming the floor {ASSUMED_CAMERA_H} m below the camera.")
            al = align(rec, assume_floor_below_camera=ASSUMED_CAMERA_H)
    T = al["T_plan_world"]
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    with _timed(timings, "rooms"):
        rooms, conns, _ = extract_rooms(P, N, al["ceiling_h"], rec.meta.get("voxel", 0.02), 1.0)
        if not rooms:
            warn("rooms", "Room split failed; the whole footprint is one room.")
            rooms, conns = [extract_room(P, N, al["ceiling_h"], rec.meta.get("voxel", 0.02), 1.0)], []
            rooms[0]["id"] = rooms[0]["name"] = "R1"
        _name_rooms(rooms, bundle, T)
    contract.fill_from_geometry(plan, rooms, conns)
    plan["frame"]["T_plan_world"] = T.round(6).tolist()
    plan["source"].update(gravity=al["up_source"], yaw_deg=al["yaw_deg"], camera_h_m=round(al["camera_h"], 2))
    bev_png(out / "debug" / "bev.png", P, N, plan, al["ceiling_h"])
    return {"bundle": bundle, "rooms": rooms, "connections": conns, "T_plan_world": T.round(6).tolist()}


def _name_rooms(rooms, bundle, T):
    """Photo input in per-room folders (photos/kitchen/...): name each plan room after the folder most of
    the cameras inside it came from. Other inputs keep R1, R2, ..."""
    from shapely.geometry import Point, Polygon

    from fp.geometry.align import to_plan
    names = bundle.meta.get("_frame_rooms")
    if not names or not any(names):
        return
    cams = to_plan(T, np.array([f.T_wc[:3, 3] for f in bundle.frames]))
    for r in rooms:
        poly = Polygon(r["polygon"]).buffer(0.2)
        votes = [n for n, c in zip(names, cams) if n and poly.contains(Point(c[0], c[1]))]
        if votes:
            r["name"] = max(set(votes), key=votes.count)


def _assess_damage(legacy, plan, out, wet_rooms, warn):
    """SAM 3 damage on surfaces + concealed-damage rules (Modal). A failure here never loses the floor plan."""
    from fp.damage.project import assess
    from fp.damage.rules import concealed
    try:
        items, _ = assess(legacy["bundle"], legacy, out)
        concealed(items, legacy, wet_rooms)
        contract.damage_to_schema(plan, items)
    except Exception as e:
        warn("damage", f"Damage step failed: {type(e).__name__}: {e}")


def _write(plan: dict, out: Path) -> None:
    from fp.report import write_outputs
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    write_outputs(plan, out)


def render(plan_path: Path, out: Path | None) -> Path:
    """plan.json -> plan.svg + report.html (validated first). Default output: next to the plan."""
    from fp.report import write_outputs
    from fp.schema import validate
    plan = validate(plan_path).model_dump(mode="json", by_alias=True)
    out = out or plan_path.parent
    out.mkdir(parents=True, exist_ok=True)
    write_outputs(plan, out)
    return out


def _print_summary(plan: dict, out: Path) -> None:
    fmt = lambda m: "not observed" if m["value"] is None else f"{m['value']:.2f} [{m['lo']:.2f}-{m['hi']:.2f}] {m['unit']}"
    print(f"{out}/plan.json  tier={plan['tier']}  rooms={len(plan['rooms'])}  t={plan['timings'].get('total')}s")
    for r in plan["rooms"]:
        print(f"  {r['id']} {r['name']}: area {fmt(r['floor_area'])}  ceiling {fmt(r['ceiling_height'])}")
        for w in r["walls"]:
            print(f"    {w['id']}: {fmt(w['length'])}  {'observed' if w['observed'] else 'inferred'}")
    for c in plan["connections"]:
        print(f"  {c['rooms'][0]}-{c['rooms'][1]} via {c['opening_id']}")
    for w in plan["warnings"]:
        print(f"  WARNING [{w['stage']}] {w['message']}")


def main():
    ap = argparse.ArgumentParser(prog="fp", description="Phone capture -> dimensioned, stitched floor plan.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="capture -> plan.json + plan.svg + report.html")
    r.add_argument("capture", type=Path, help="Stray Scanner folder, video file, or folder of room photo folders")
    r.add_argument("--out", type=Path, help="output folder (default out/<capture>)")
    r.add_argument("--tier", default="auto", choices=["auto", "lidar", "video", "photos"],
                   help="auto-detect, or run a LiDAR capture as video/photos (its RGB alone)")
    r.add_argument("--backend", default="local", choices=["local", "modal"],
                   help="where learned models run: this machine (default) or Modal GPUs (optional)")
    r.add_argument("--no-drift-correction", action="store_true", help="use the capture's poses as-is (ablation)")
    r.add_argument("--no-cache", action="store_true", help="recompute model outputs instead of replaying the cache")
    r.add_argument("--no-filter", action="store_true", help="keep blurry and fast-turning LiDAR frames")
    r.add_argument("--max-frames", type=int, help="cap on frames sent to the depth model")
    r.add_argument("--no-damage", action="store_true", help="skip damage detection")
    r.add_argument("--wet-rooms", default="", help="comma-separated room ids that are bathrooms/kitchens, e.g. R2,R3")
    g = sub.add_parser("render", help="plan.json -> plan.svg + report.html")
    g.add_argument("plan", type=Path)
    g.add_argument("--out", type=Path, help="output folder (default: next to the plan)")
    c = sub.add_parser("check", help="inspect one pipeline stage (see fp/check.py)")
    c.add_argument("stage", choices=["frames", "poses", "fusion", "cloud", "align", "walls", "footprint", "all", "view"])
    c.add_argument("capture", type=Path)
    c.add_argument("--out", type=Path)
    c.add_argument("--filter", action="store_true", help="apply the frame filter first, as `fp run` does")
    a = ap.parse_args()
    if a.cmd == "check":
        from fp import check
        return check.run(a.stage, a.capture, a.out, a.filter)
    if a.cmd == "render":
        out = render(a.plan, a.out)
        print(f"{out}/plan.svg\n{out}/report.html")
        return
    name = contract.capture_id(a.capture)
    out = a.out or Path("out") / (name + ("" if a.tier == "auto" else f"-{a.tier}") + ("-nofilter" if a.no_filter else "")
                                  + ("-nodrift" if a.no_drift_correction else ""))
    plan = run(a.capture, out, tier=a.tier, backend=a.backend, drift_correction=not a.no_drift_correction,
               cache=not a.no_cache, filter_frames=not a.no_filter, max_frames=a.max_frames, damage=not a.no_damage,
               wet_rooms={r.strip() for r in a.wet_rooms.split(",") if r.strip()})
    _print_summary(plan, out)


if __name__ == "__main__":
    main()
