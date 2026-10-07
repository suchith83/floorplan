"""Derive the camera-only tier inputs (video, per-room photos) from a Stray Scanner LiDAR capture.

Usage:
    uv run python scripts/make_camera_tiers.py data/stray/c00a170fe1 data/stray/c7d28f72c6 [--out data/derived]

Why: the brief compares tiers on the same rooms. The sample captures are LiDAR-only, so the video and
photo tiers are cut from the same `rgb.mp4`; the LiDAR plan is the *reference* that tells us which room
each frame was taken in. For every capture <id> this writes (and overwrites) <out>/<id>/:

    video/<id>.mp4                the capture's rgb.mp4 alone (byte copy; no depth, no poses)
    photos/<room id>/fNNNNNN.jpg  2-8 full-resolution upright stills per room (JPEG q95), NNNNNN =
                                  video frame index = odometry.csv row
    selection.json                every still (frame, timestamp, rooms, why, sharpness, camera pose in
                                  the plan), the LiDAR room polygons used, a per-room and per-connection summary

How stills are chosen (same constants for every capture, no per-capture tuning):
 1. Reference plan: out/<id>-lidar-ref/plan.json (made with `fp run --no-damage` if missing).
    Camera position in the plan = T_plan_world @ T_wc[:3,3]; viewing direction = T_plan_world rotation
    @ T_wc[:3,2] (OpenCV camera, +z forward). Pitch = elevation of that direction, yaw = its heading.
 2. Room membership: the LiDAR room polygon that contains the camera position (frames outside every
    room are ignored).
 3. Candidates: every CANDIDATE_STEP-th frame that is inside a room, roughly level (|pitch| <=
    MAX_PITCH_DEG, as a person following the protocol would hold the phone), and sharp (variance of the
    Laplacian of a 480x360 grey image >= SHARPNESS_FRACTION x the median over the level in-room
    sampled frames of that capture: a relative rule, so it adapts to scene texture without tuning).
 4. Doorways first: for each LiDAR connection A-B with centre c (the opening centre, or the middle of
    the shared boundary when no opening was detected) pick the candidate in A or B whose camera is
    DOOR_MIN_DIST..DOOR_RADIUS from c, whose view points at c within DOOR_MAX_ANGLE_DEG and whose view
    ray (DOOR_RAY_LEN, horizontal) enters the other room's polygon; among those, the one looking most
    squarely at c. It is written once and copied byte-for-byte into both rooms' folders, as in the
    capture protocol ("in every doorway take 1 photo looking into the next room and save it in both
    rooms' folders"); the photo loader dedupes identical photos and uses them to link rooms.
 5. Room photos: fill each room up to PER_ROOM_MAX stills (at least PER_ROOM_MIN) by greedy
    farthest-point selection over (camera x, y, yaw), seeded by that room's doorway stills (or, if none,
    by its sharpest candidate), so the shots come from different spots and look in different
    directions ("stand in each corner and shoot across the room"). Rooms with fewer than PER_ROOM_MIN
    candidates (seen only through a door) get no photos and are listed as "not photographed";
    connections touching them get no doorway still.

Upright stills: each still is rotated by k*90 deg so that world up points closest to image up (what a
phone camera app does with its accelerometer); a portrait-held frame becomes 1440x1920. The rotation is
recorded as `rotated_deg` (clockwise) in selection.json. video/<id>.mp4 is left untouched.

Decoding: rgb.mp4 is read front to back twice (HEVC seeks are slow): once to score sharpness of the
sampled frames, once to write the chosen stills. The output is deterministic (no wall-clock data).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from shapely.geometry import LineString, Point, Polygon

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fp.ingest.stray import read_odometry  # noqa: E402

CANDIDATE_STEP = 10          # every 10th frame: ~4-5 candidates per second at the real ~46 fps, plenty
SHARP_SIZE = (480, 360)      # sharpness is scored on a quarter-resolution grey image (fast, less noise)
SHARPNESS_FRACTION = 0.5     # keep frames at least half as sharp as the capture's median: drops motion blur
MAX_PITCH_DEG = 35.0         # roughly level photos; steeper ones show mostly floor or ceiling
PER_ROOM_MIN = 2             # the photo protocol asks for 2-8 photos per room
PER_ROOM_MAX = 8
YAW_WEIGHT_M_PER_RAD = 1.0   # diversity metric: 1 rad (57 deg) of heading counts like 1 m of position
DOOR_RADIUS = 1.0            # a doorway photo is taken within 1 m of the opening centre ...
DOOR_MIN_DIST = 0.3          # ... but not inside the opening itself (heading to the centre is unstable)
DOOR_MAX_ANGLE_DEG = 45.0    # the opening centre must be in front of the camera
DOOR_RAY_LEN = 3.0           # and the view ray must reach into the other room within 3 m
SHARED_EDGE_BUFFER = 0.3     # connection without an opening: centre of the boundary within 0.3 m of both rooms
JPEG_QUALITY = 95


def _angdiff(a, b):
    return np.abs((np.asarray(a) - b + np.pi) % (2 * np.pi) - np.pi)


def reference_plan(capture: Path) -> dict:
    ref = Path("out") / f"{capture.name}-lidar-ref"
    if not (ref / "plan.json").is_file():
        print(f"  no LiDAR reference yet; running fp run -> {ref}")
        subprocess.run(["uv", "run", "fp", "run", str(capture), "--no-damage", "--out", str(ref)], check=True)
    return json.loads((ref / "plan.json").read_text())


def sharpness_pass(video: Path, wanted: set[int]) -> dict[int, float]:
    """Variance of the Laplacian for the wanted frame indices, in one sequential decode."""
    cap = cv2.VideoCapture(str(video))
    out, i, last = {}, 0, max(wanted)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            break
        if i in wanted:
            grey = cv2.cvtColor(cv2.resize(img, SHARP_SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            out[i] = float(cv2.Laplacian(grey, cv2.CV_64F).var())
        i += 1
    cap.release()
    return out


# clockwise image rotation (deg) -> cv2 rotate code; 0 = keep the sensor's landscape frame
_ROTATE = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def upright_rotation(R_wc: np.ndarray) -> int:
    """Clockwise k*90 deg rotation that makes world up point closest to image up, as a phone camera app
    does with its accelerometer. World up (+y, Stray/ARKit world) in camera axes is R_wc^T @ (0,1,0);
    its image-plane part (ux, uy) (OpenCV: x right, y down) maps to, after rotating the image clockwise by
    0 / 90 / 180 / 270 deg: (ux, uy) / (-uy, ux) / (-ux, -uy) / (uy, -ux). Image up is -y, so the score
    of each option is minus its new y component; ties go to the smaller rotation."""
    ux, uy = R_wc[1, 0], R_wc[1, 1]            # row 1 of R_wc = R_wc^T @ (0, 1, 0)
    score = {0: -uy, 90: -ux, 180: uy, 270: ux}
    return max(score, key=lambda k: (round(float(score[k]), 9), -k))


def write_stills(video: Path, targets: dict[int, list[Path]], rot: dict[int, int]) -> None:
    """Decode front to back once; rotate each wanted frame upright (rot[i] deg clockwise), write it to
    its first path and copy that file byte-for-byte to the others."""
    cap = cv2.VideoCapture(str(video))
    i, last = 0, max(targets)
    while i <= last:
        ok, img = cap.read()
        if not ok:
            raise RuntimeError(f"{video}: video ended at frame {i}, before frame {last}")
        if i in targets:
            first, *rest = targets[i]
            first.parent.mkdir(parents=True, exist_ok=True)
            if _ROTATE[rot[i]] is not None:
                img = cv2.rotate(img, _ROTATE[rot[i]])
            cv2.imwrite(str(first), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            for p in rest:
                p.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(first, p)
        i += 1
    cap.release()


def farthest_points(feat: np.ndarray, seeds: list[int], k: int) -> list[int]:
    """Greedy farthest-point selection on rows of feat = (x, y, yaw); seeds are already chosen rows."""
    chosen = list(seeds)
    if not len(feat):
        return chosen

    def dist(j):
        dp = np.hypot(feat[:, 0] - feat[j, 0], feat[:, 1] - feat[j, 1])
        return np.hypot(dp, YAW_WEIGHT_M_PER_RAD * _angdiff(feat[:, 2], feat[j, 2]))

    dmin = np.full(len(feat), np.inf)
    for j in chosen:
        dmin = np.minimum(dmin, dist(j))
    while len(chosen) < min(k, len(feat)):
        dmin[chosen] = -1.0
        j = int(np.argmax(dmin))                 # argmax takes the first (lowest index) on ties
        chosen.append(j)
        dmin = np.minimum(dmin, dist(j))
    return chosen


def connection_centre(plan: dict, conn: dict, polys: dict[str, Polygon]) -> tuple[list[float], str]:
    openings = {o["id"]: o for r in plan["rooms"] for o in r["openings"]}
    if conn.get("opening_id") in openings:
        return list(openings[conn["opening_id"]]["center"]), f"opening {conn['opening_id']}"
    a, b = (polys[r] for r in conn["rooms"])
    shared = a.buffer(SHARED_EDGE_BUFFER).intersection(b.buffer(SHARED_EDGE_BUFFER))
    c = shared.centroid if not shared.is_empty else LineString([a.centroid, b.centroid]).centroid
    return [c.x, c.y], "shared boundary (no opening detected)"


def process(capture: Path, out_root: Path) -> dict:
    t0 = time.time()
    cid = capture.name
    plan = reference_plan(capture)
    P = np.asarray(plan["frame"]["T_plan_world"], float)
    rooms = plan["rooms"]
    polys = {r["id"]: Polygon(r["polygon"]) for r in rooms}

    od = read_odometry(capture)
    n = len(od["t"])
    if not np.array_equal(od["index"], np.arange(n)):
        raise RuntimeError(f"{capture}: odometry frame column is not 0..n-1; video frame i must equal row i")
    pos = (P[:3, :3] @ od["T_wc"][:, :3, 3].T).T + P[:3, 3]
    fwd = (P[:3, :3] @ od["T_wc"][:, :3, 2].T).T
    pitch = np.degrees(np.arcsin(np.clip(fwd[:, 2], -1, 1)))
    yaw = np.arctan2(fwd[:, 1], fwd[:, 0])

    room_of = {}
    for i in range(0, n, CANDIDATE_STEP):
        pt = Point(pos[i, 0], pos[i, 1])
        for r in rooms:                          # plan order; rooms do not overlap
            if polys[r["id"]].contains(pt):
                room_of[i] = r["id"]
                break
    level = sorted(i for i in room_of if abs(pitch[i]) <= MAX_PITCH_DEG)
    sharp = sharpness_pass(capture / "rgb.mp4", set(level)) if level else {}
    med = float(np.median([sharp[i] for i in level if i in sharp])) if sharp else 0.0
    cands = [i for i in level if sharp.get(i, 0.0) >= SHARPNESS_FRACTION * med]
    by_room = {r["id"]: [i for i in cands if room_of[i] == r["id"]] for r in rooms}
    photographed = {rid for rid, c in by_room.items() if len(c) >= PER_ROOM_MIN}

    reasons: dict[int, dict] = {}                # frame -> {"rooms": [...], "why": [...]}

    def add(i, rid, why):
        e = reasons.setdefault(i, {"rooms": [], "why": []})
        if rid not in e["rooms"]:
            e["rooms"].append(rid)
        if why not in e["why"]:
            e["why"].append(why)

    conn_out = []
    for conn in plan.get("connections", []):
        A, B = conn["rooms"]
        c, src = connection_centre(plan, conn, polys)
        rec = {"rooms": [A, B], "opening_id": conn.get("opening_id"), "center": [round(v, 3) for v in c],
               "center_from": src, "frame": None}
        missing = [r for r in (A, B) if r not in photographed]
        if missing:
            rec["status"] = f"no doorway frame: {', '.join(missing)} not photographed"
            conn_out.append(rec)
            continue
        best = None
        for i in by_room[A] + by_room[B]:
            other = B if room_of[i] == A else A
            v = np.array(c) - pos[i, :2]
            d = float(np.hypot(*v))
            if not DOOR_MIN_DIST <= d <= DOOR_RADIUS:
                continue
            ang = float(np.degrees(_angdiff(np.arctan2(v[1], v[0]), yaw[i])))
            if ang >= DOOR_MAX_ANGLE_DEG:
                continue
            ray = LineString([pos[i, :2], pos[i, :2] + DOOR_RAY_LEN * np.array([np.cos(yaw[i]), np.sin(yaw[i])])])
            if not ray.intersects(polys[other]):
                continue
            key = (ang, -sharp[i], i)
            if best is None or key < best[0]:
                best = (key, i, d)
        if best is None:
            rec["status"] = "no doorway frame"
        else:
            (ang, _, i), d = best[0], best[2]
            rec.update(frame=i, status="ok", camera_room=room_of[i], distance_m=round(d, 3),
                       angle_to_center_deg=round(ang, 1))
            add(i, A, "doorway"), add(i, B, "doorway")
        conn_out.append(rec)

    for rid in sorted(photographed, key=lambda r: [x["id"] for x in rooms].index(r)):
        cl = by_room[rid]
        feat = np.stack([pos[cl, 0], pos[cl, 1], yaw[cl]], 1)
        door_here = [k for k, i in enumerate(cl) if i in reasons and rid in reasons[i]["rooms"]]
        # doorway stills taken from the other room still count as already-chosen views of this room
        n_door_total = sum(1 for e in reasons.values() if rid in e["rooms"])
        seeds = door_here or [int(np.argmax([sharp[i] for i in cl]))]
        budget = max(PER_ROOM_MIN, PER_ROOM_MAX - (n_door_total - len(door_here)))
        picked = farthest_points(feat, seeds, budget)
        if not door_here:
            add(cl[seeds[0]], rid, "room")
        for k in picked[len(seeds):]:
            add(cl[k], rid, "room")

    out = out_root / cid
    if out.exists():
        shutil.rmtree(out)
    (out / "video").mkdir(parents=True)
    shutil.copyfile(capture / "rgb.mp4", out / "video" / f"{cid}.mp4")
    rot = {i: upright_rotation(od["T_wc"][i, :3, :3]) for i in reasons}
    targets = {i: [out / "photos" / rid / f"f{i:06d}.jpg" for rid in e["rooms"]] for i, e in sorted(reasons.items())}
    if targets:
        write_stills(capture / "rgb.mp4", targets, rot)

    stills = [{"frame": i, "timestamp": round(float(od["t"][i]), 6),
               "files": [str(p.relative_to(out)) for p in targets[i]], "rooms": e["rooms"], "why": e["why"],
               "sharpness": round(sharp[i], 1), "sharpness_rel_median": round(sharp[i] / med, 2),
               "camera_room": room_of[i], "camera_xy": [round(float(v), 3) for v in pos[i, :2]],
               "camera_height_m": round(float(pos[i, 2]), 3), "yaw_deg": round(float(np.degrees(yaw[i])), 1),
               "pitch_deg": round(float(pitch[i]), 1), "rotated_deg": rot[i],
               "size_wh": [1440, 1920] if rot[i] in (90, 270) else [1920, 1440]} for i, e in sorted(reasons.items())]
    room_out = []
    for r in rooms:
        rid = r["id"]
        files = sorted(i for i, e in reasons.items() if rid in e["rooms"])
        room_out.append({"id": rid, "floor_area_m2": r["floor_area"]["value"],
                         "polygon": [[round(x, 3), round(y, 3)] for x, y in r["polygon"]],
                         "sampled_frames_inside": sum(1 for i in room_of.values() if i == rid),
                         "candidates": len(by_room[rid]),
                         "status": "photographed" if rid in photographed else "not photographed",
                         "photos": len(files), "photo_frames": files,
                         "doorway_photos": sum(1 for i in files if "doorway" in reasons[i]["why"])})
    sel = {"capture": cid, "source": str(capture), "reference_plan": f"out/{cid}-lidar-ref/plan.json",
           "T_plan_world": P.round(6).tolist(),
           "constants": {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))},
           "odometry_rows": n, "sampled_frames": len(range(0, n, CANDIDATE_STEP)),
           "sampled_in_rooms": len(room_of), "level_in_rooms": len(level), "candidates": len(cands),
           "median_sharpness": round(med, 1), "rooms": room_out, "connections": conn_out,
           "stills": stills, "total_stills": len(stills),
           "total_files": sum(len(t) for t in targets.values())}
    (out / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    sel["_runtime_s"] = round(time.time() - t0, 1)
    return sel


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("captures", nargs="+", type=Path, help="Stray Scanner capture folders")
    ap.add_argument("--out", type=Path, default=Path("data/derived"))
    args = ap.parse_args()
    for cap in args.captures:
        print(f"{cap.name}:")
        s = process(cap, args.out)
        for r in s["rooms"]:
            print(f"  {r['id']}: {r['floor_area_m2']} m2, {r['candidates']} candidates, {r['photos']} photos "
                  f"({r['doorway_photos']} doorway) [{r['status']}]")
        for c in s["connections"]:
            print(f"  {c['rooms'][0]}-{c['rooms'][1]}: {c['status']}" + (f" (frame {c['frame']})" if c["frame"] is not None else ""))
        print(f"  {s['total_stills']} stills, {s['total_files']} files, {s['_runtime_s']} s")


if __name__ == "__main__":
    main()
