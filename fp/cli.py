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
from fp.contract import StageError, StageNotBuilt

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
    if bundle.frames and all(f.T_wc is not None and f.timestamp is not None for f in bundle.frames):
        # the whole camera path, before the filter thins it: room occupancy needs time, not kept frames
        bundle.meta["_trajectory"] = (np.array([f.timestamp for f in bundle.frames]),
                                      np.array([f.T_wc[:3, 3] for f in bundle.frames]))
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
    t0 = time.perf_counter()
    capture = Path(capture)
    tier = detect_tier(capture) if tier == "auto" else tier
    plan = contract.empty_plan(tier, contract.capture_id(capture), source={
        "path": str(capture), "scale": contract.TIER_NAME[tier], "backend": backend, "cache": cache})
    warn = lambda stage, msg: plan["warnings"].append({"stage": stage, "message": msg})
    timings = plan["timings"]
    debug = out / "debug"
    debug.mkdir(parents=True, exist_ok=True)
    try:
        legacy = _geometry(capture, out, plan, timings, tier, backend, filter_frames, max_frames, warn)
        if damage and backend == "modal":
            with _timed(timings, "damage"):
                _assess_damage(legacy, plan, out, wet_rooms, warn)
        elif damage:
            warn("damage", "Local damage backend not built yet (work order 06); damage not assessed.")
    except StageError as e:
        warn(e.stage, str(e))
    if not plan["rooms"]:
        plan["drift"]["method"] = "not run: no poses reached the geometry stage"
    elif not drift_correction:
        plan["drift"]["method"] = "disabled (--no-drift-correction): poses used as-is"
    else:
        plan["drift"]["method"] = "not built yet (work order 03): poses used as-is"
        warn("drift", "Drift correction not built yet (work order 03); poses used as-is.")
    if not plan["rooms"]:
        warn("rooms", "No rooms reconstructed.")
    warn("intervals", "Intervals are provisional (fp/contract.py), not yet calibrated on ground truth (work order 07).")
    timings["total"] = round(time.perf_counter() - t0, 2)
    try:
        plan = _validated(plan)
    except SystemExit:  # a bug in our own output: keep it for debugging, then stop
        (out / "plan.invalid.json").write_text(json.dumps(plan, indent=1))
        raise
    _write(plan, out)
    return plan


def _validated(plan: dict | Path) -> dict:
    """Schema-validate a plan (dict or JSON path); on failure exit with every error on its own line, not a
    traceback. Pydantic checks fields before the cross-reference pass, so ID errors show once fields pass."""
    from pydantic import ValidationError

    from fp.schema import validate
    if isinstance(plan, Path) and not plan.is_file():
        raise SystemExit(f"No such plan file: {plan}")
    try:
        return validate(plan).model_dump(mode="json", by_alias=True)
    except ValidationError as e:
        lines = [f"  {'.'.join(map(str, err['loc'])) or '<plan>'}: {err['msg']}" for err in e.errors()]
        raise SystemExit("plan.json does not match schema 1.0:\n" + "\n".join(lines))


def _geometry(capture, out, plan, timings, tier, backend, filter_frames, max_frames, warn) -> dict:
    """Ingest -> recon -> align -> rooms, filling `plan` in place. Returns the internal (pre-schema) result
    that the damage step still works on."""
    from fp.geometry.align import CEILING_MIN_AREA, FloorNotFound, align, to_plan
    from fp.geometry.plan import extract_room, extract_rooms
    from fp.ingest import load_capture
    from fp.report.debug import bev_png
    with _timed(timings, "ingest"):
        bundle = load_capture(capture, tier=tier, max_frames=max_frames)
    for msg in bundle.meta.get("_warnings", []):
        warn("ingest", msg)
    with _timed(timings, "recon"):
        bundle, rec, fstats = reconstruct(bundle, out, filter_frames, backend)
    plan["source"].update(n_frames=len(bundle.frames), frame_filter=fstats,
                          **{k: v for k, v in bundle.meta.items() if not k.startswith("_")})
    with _timed(timings, "align"):
        try:
            al = align(rec)
        except FloorNotFound as e:
            warn("align", f"{e} Assuming the floor {ASSUMED_CAMERA_H} m below the camera; every room below rests "
                          "on that assumption, so treat its walls and area as rough, whatever their intervals say.")
            al = align(rec, assume_floor_below_camera=ASSUMED_CAMERA_H)
    T = al["T_plan_world"]
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    if al["ceiling_h"] is None:
        warn("ceiling", f"No ceiling layer of at least {CEILING_MIN_AREA:g} m2 was seen (highest point "
                        f"{P[:, 2].max():.2f} m above the floor), so ceiling height is not observed.")
    with _timed(timings, "rooms"):
        rays = _plan_rays(rec, T)
        traj = bundle.meta.get("_trajectory")
        cams = to_plan(T, traj[1]) if traj is not None else None
        rooms, conns, gdbg = extract_rooms(P, N, al["ceiling_h"], rec.meta.get("voxel", 0.02), 1.0, rays, cams)
        wall_lines = gdbg["lines"]
        _labels_png(out / "debug" / "rooms_split.png", gdbg)
        if not rooms:
            warn("rooms", "Room split failed; the whole footprint is one room.")
            rooms, conns = [extract_room(P, N, al["ceiling_h"], rec.meta.get("voxel", 0.02), 1.0, rays=rays,
                                         cams=cams)], []
            rooms[0]["id"] = rooms[0]["name"] = "R1"
        for d in gdbg.get("dropped", []):
            warn("rooms", f"Left out a {d['area_m2']:.1f} m2 region the camera never entered, with only "
                          f"{100 * d['floor_frac']:.0f} % of its floor seen: air seen through a window or doorway.")
        rooms, conns = rank_rooms(rooms, conns, bundle, T, plan, warn)
        final = {r.get("label", r["id"]): r["id"] for r in rooms}
        for o in gdbg.get("overlaps", []):
            a, b = (final.get(k, k) for k in o["rooms"])
            warn("rooms", f"Rooms {a} and {b} overlapped by {o['overlap_m2']:.2f} m2 "
                          f"({100 * o['frac_of_smaller']:.0f} % of the smaller); repaired: the overlap went to "
                          f"{a}, whose own footprint covers more of it.")
        for a, b, f in check_overlaps(rooms):
            warn("rooms", f"{a} and {b} still overlap by {100 * f:.1f} % of the smaller room.")
        _name_rooms(rooms, bundle, T)
    contract.fill_from_geometry(plan, rooms, conns)
    plan["frame"]["T_plan_world"] = T.round(6).tolist()
    plan["source"].update(gravity=al["up_source"], yaw_deg=al["yaw_deg"], camera_h_m=round(al["camera_h"], 2))
    bev_png(out / "debug" / "bev.png", P, N, plan, al["ceiling_h"])
    _fusion_debug(out / "debug", P, N, rec, bundle, T, plan, al["ceiling_h"], wall_lines)
    return {"bundle": bundle, "rooms": rooms, "connections": conns, "T_plan_world": T.round(6).tolist()}


def _plan_rays(rec, T):
    """Camera->point rays of the fused frames in plan coords (None for inputs without depth frames)."""
    from fp.geometry.align import to_plan
    from fp.recon.lidar_fuse import free_space_rays
    if not rec.frames or any(f.depth is None or f.T_wc is None for f in rec.frames):
        return None
    return [(to_plan(T, c[None])[0], to_plan(T, p)) for c, p in free_space_rays(rec.frames)]


MIN_ROOM_VISIT_S = 3.0  # s: a room the camera spent less time in was only seen from outside (through a door)
MAX_DT_S = 0.2          # s: a longer gap between poses is a tracking dropout, not time spent there
INSIDE_M = 0.3          # m: count only time spent this far inside a room; standing in its doorway doesn't count


def room_occupancy(rooms, t: np.ndarray, cams: np.ndarray) -> dict[str, float]:
    """Seconds the camera spent at least INSIDE_M inside each room polygon (plan x/y). Each pose counts for the
    time until the next one."""
    from shapely import contains_xy
    from shapely.geometry import Polygon
    dt = np.clip(np.diff(t, append=t[-1]), 0, MAX_DT_S)
    inner = {r["id"]: Polygon(r["polygon"]).buffer(-INSIDE_M, join_style=2) for r in rooms}
    return {k: float(dt[contains_xy(p, cams[:, 0], cams[:, 1])].sum()) if not p.is_empty else 0.0
            for k, p in inner.items()}


CONNECTOR_MAX_W = 1.6   # m: a region whose widest inscribed circle is narrower than this is a corridor-width space
CONNECTOR_MIN_LEN = 2.5  # m: ...and its mean length (area / width) at least this: a hall runs past two doors
                         # (2 x ~1.2 m); shorter narrow spaces are lobbies or closets


def rank_rooms(rooms, conns, bundle, T, plan, warn):
    """Number rooms by floor area (R1 = largest), name long narrow ones "connector", and warn about rooms the
    camera never really entered: they were seen through a doorway, so their walls and area are partial.
    Those rooms stay in the plan (we don't hide what was seen), flagged in the warnings."""
    from fp.geometry.align import to_plan
    traj = bundle.meta.get("_trajectory")
    if not rooms:
        return rooms, conns
    occ = room_occupancy(rooms, traj[0], to_plan(T, traj[1])) if traj is not None else {}
    order = sorted(rooms, key=lambda r: -r["area_m2"])
    new = {r["id"]: f"R{k + 1}" for k, r in enumerate(order)}
    for r in order:
        r["occupancy_s"] = round(occ.get(r["id"], 0.0), 1)
        r["label"], r["id"] = r["id"], new[r["id"]]
        w = r.get("width_m")
        if len(order) > 1 and w and w <= CONNECTOR_MAX_W and r["area_m2"] / w >= CONNECTOR_MIN_LEN:
            r["name"] = "connector"
        elif r["name"] in new:
            r["name"] = r["id"]
    for c in conns:
        c["rooms"] = sorted((new[c["rooms"][0]], new[c["rooms"][1]]), key=lambda i: int(i[1:]))
    if traj is None:
        return order, conns
    plan["source"]["room_occupancy_s"] = {r["id"]: r["occupancy_s"] for r in order}
    main = max(order, key=lambda r: r["occupancy_s"])
    for r in order:   # the room the camera spent most time in is the main room, even on a very short capture
        if r is not main and r["occupancy_s"] < MIN_ROOM_VISIT_S:
            warn("rooms", f"{r['id']} is partially observed: the camera spent {r['occupancy_s']:.1f} s inside it "
                          f"(< {MIN_ROOM_VISIT_S:.0f} s), so it was seen mostly through a doorway; its walls and "
                          f"area are partial.")
    return order, conns


def check_overlaps(rooms) -> list[tuple[str, str, float]]:
    """Pairs of room polygons overlapping by more than 1 % of the smaller one (should be none after repair)."""
    from shapely.geometry import Polygon
    from fp.geometry.plan import OVERLAP_TOL
    out = []
    for a in range(len(rooms)):
        for b in range(a + 1, len(rooms)):
            A, B = Polygon(rooms[a]["polygon"]), Polygon(rooms[b]["polygon"])
            f = A.intersection(B).area / max(min(A.area, B.area), 1e-9)
            if f > OVERLAP_TOL:
                out.append((rooms[a]["id"], rooms[b]["id"], round(f, 3)))
    return out


def _labels_png(path, gdbg):
    """debug/rooms_split.png: the footprint (grey) split into rooms (one colour each), tall walls black, closed
    doorways red; +x right, +y up."""
    import cv2
    lab = gdbg["labels"]
    pal = np.array([[255, 255, 255]] + [[int(v) for v in np.random.default_rng(k).integers(60, 230, 3)]
                                         for k in range(1, lab.max() + 1)], np.uint8)
    img = pal[lab]
    img[(lab == 0) & gdbg["footprint"]] = 160
    if "tall" in gdbg:
        img[gdbg["tall"]] = 0                 # tall wall evidence (black)
        img[gdbg["bar"]] = (0, 0, 255)        # doorways closed along wall lines (red)
    cv2.imwrite(str(path), np.ascontiguousarray(img.transpose(1, 0, 2)[::-1]))


def _fusion_debug(debug, P, N, rec, bundle, T, plan, ceiling_h, lines):
    """debug/fusion_topdown.png (cloud, camera path, rooms) and debug/wall_slice.png (cut through the longest wall)."""
    from fp.geometry.align import to_plan
    from fp.report.debug import fusion_topdown_png, wall_slice_png
    traj = bundle.meta.get("_trajectory")
    cams = to_plan(T, traj[1] if traj is not None else np.array([f.T_wc[:3, 3] for f in rec.frames]))
    fusion_topdown_png(debug / "fusion_topdown.png", P, rec.colors, cams, plan["rooms"])
    # wall points (plan x/y, wall band, 2 cm thinned) for registering two scans of one property (scripts/repeatability.py)
    wb = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > 0.3) & (P[:, 2] < 2.1)
    _, keep = np.unique(np.floor(P[wb, :2] / 0.02).astype(np.int64), axis=0, return_index=True)
    np.save(debug / "wall_points_2d.npy", P[wb, :2][keep].astype(np.float32))
    if not lines:
        return
    # the main room's longest seen wall, matched back to its detected plane
    seen = [w for w in (plan["rooms"][0]["walls"] if plan["rooms"] else []) if w["observed"]]
    longest = max(lines, key=lambda l: l["area_m2"])
    if seen:
        w = max(seen, key=lambda w: w["length"]["value"])
        ax = 0 if abs(w["p0"][0] - w["p1"][0]) < 1e-6 else 1
        cand = [l for l in lines if l["axis"] == ax and abs(l["coord"] - w["p0"][ax]) < 0.03]
        longest = max(cand, key=lambda l: l["area_m2"]) if cand else longest
    wall_slice_png(debug / "wall_slice.png", P, N, longest, ceiling_h)
    sp = [l["spread_m"] for l in lines]
    plan["source"]["wall_thickness_cm"] = {
        "median": round(100 * float(np.median(sp)), 2),
        "longest_wall": round(100 * longest["spread_m"], 2),
        "n_walls": len(sp),
        "method": "1.4826 x MAD of wall points within 15 cm of each detected wall plane (fp.geometry.plan.spread)"}


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
    plan = _validated(Path(plan_path))
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
