"""Per-component checks. Each stage runs the real pipeline code up to that point and writes
an image + prints numbers so a human can judge it. See TESTING.md for what "good" looks like.

  fp check frames|poses|fusion|cloud|align|walls|footprint|all <capture_dir> [--out DIR]
  fp check view <file.ply>"""
from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from fp.geometry.align import align, to_plan
from fp.geometry.plan import RES, extract_room
from fp.ingest import arkitscenes, load_capture
from fp.ingest.quality import image_stats, rotation_speed, select_frames
from fp.recon.lidar_fuse import fuse_depth

STAGES = ["frames", "poses", "fusion", "cloud", "align", "walls", "footprint"]


FILTER = False  # set by `fp check ... --filter`: inspect the frames `fp run` actually keeps


def _load(capture: Path):
    b = load_capture(capture)
    if not b.has_depth:
        raise SystemExit("fp check needs depth + poses (an ARKitScenes folder). For video or photos, "
                         "run `fp run` first; its debug images are in the output folder.")
    if FILTER:
        b.frames, st = select_frames(b.frames)
        print(f"frame filter: {st}")
    return b


def _thumb(path, text, w=256):
    im = cv2.imread(str(path))
    im = cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1])))
    cv2.putText(im, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    return im


def _sheet(ims, cols=4):
    while len(ims) % cols:
        ims.append(np.zeros_like(ims[0]))
    return np.vstack([np.hstack(ims[i:i + cols]) for i in range(0, len(ims), cols)])


# ---------------------------------------------------------------- frames
def check_frames(capture, out):
    b = _load(capture)
    st = image_stats([f.rgb for f in b.frames])
    br, sh, df = st["brightness"], st["sharpness"], st["diff_prev"]
    rows = [{"frame": f.rgb.name, "brightness": round(float(br[i]), 1), "sharpness": round(float(sh[i]), 1),
             "diff_prev": None if np.isinf(df[i]) else round(float(df[i]), 2)} for i, f in enumerate(b.frames)]
    with open(out / "frames_quality.csv", "w", newline="") as fh:
        wr = csv.DictWriter(fh, rows[0].keys()); wr.writeheader(); wr.writerows(rows)
    dark, blurry, dup = br < 25, sh < 0.3 * np.median(sh), df < 1.5
    print(f"frames used: {len(rows)}  dark(<25): {dark.sum()}  blurry(<30% median sharpness): {blurry.sum()}  "
          f"near-duplicate: {dup.sum()}")
    print(f"brightness p5/p50/p95: {np.percentile(br, [5, 50, 95]).round(1)}  sharpness p5/p50/p95: {np.percentile(sh, [5, 50, 95]).round(1)}")
    even = np.linspace(0, len(rows) - 1, 16).astype(int)
    cv2.imwrite(str(out / "frames_sheet.png"), _sheet([_thumb(b.frames[i].rgb, f"#{i} b{br[i]:.0f} s{sh[i]:.0f}") for i in even]))
    worst = np.argsort(sh)[:4].tolist() + np.argsort(br)[:4].tolist()
    cv2.imwrite(str(out / "frames_worst.png"), _sheet([_thumb(b.frames[i].rgb, f"#{i} b{br[i]:.0f} s{sh[i]:.0f}") for i in worst]))
    print(f"-> {out}/frames_sheet.png (16 evenly spaced), frames_worst.png (4 blurriest + 4 darkest), frames_quality.csv")


# ---------------------------------------------------------------- poses
def check_poses(capture, out):
    ts, poses = arkitscenes.load_traj(capture)
    b = _load(capture)
    n_rgb = len(sorted((capture / "lowres_wide").glob("*.png"))[::3])
    C = poses[:, :3, 3]
    dt = np.diff(ts)
    speed = np.linalg.norm(np.diff(C, axis=0), axis=1) / dt
    ang = rotation_speed(ts, poses)
    print(f"pose samples: {len(ts)} over {ts[-1] - ts[0]:.1f}s ({len(ts) / (ts[-1] - ts[0]):.1f} Hz)  "
          f"gaps>0.15s: {(dt > 0.15).sum()}  path length: {np.linalg.norm(np.diff(C, axis=0), axis=1).sum():.1f} m")
    print(f"speed p95/max: {np.percentile(speed, 95):.2f}/{speed.max():.2f} m/s   "
          f"rotation p95/max: {np.percentile(ang, 95):.0f}/{ang.max():.0f} deg/s   "
          f"frames with a pose: {len(b.frames)}/{n_rgb}")
    xy = C[:, [0, 2]]  # ARKit: y is up, so top view = x,z
    lo, hi = xy.min(0) - 0.3, xy.max(0) + 0.3
    s = 600 / (hi - lo).max()
    img = np.full((int((hi - lo)[1] * s) + 1, int((hi - lo)[0] * s) + 1, 3), 255, np.uint8)
    p = ((xy - lo) * s).astype(int)
    for i in range(len(p) - 1):
        col = (0, 0, 255) if speed[i] > 1.0 else (200, 120, 0)
        cv2.line(img, tuple(p[i]), tuple(p[i + 1]), col, 2)
    cv2.circle(img, tuple(p[0]), 6, (0, 160, 0), -1)
    cv2.imwrite(str(out / "poses_topdown.png"), img)
    print(f"-> {out}/poses_topdown.png (green dot = start, red = moving > 1 m/s)")


# ---------------------------------------------------------------- fusion: how the 3D cloud is built
def _topdown(P, C, lo, hi, res, img=None):
    """Paint plan-frame points seen from above (highest point wins). C: (N,3) BGR uint8."""
    w, h = ((hi - lo) / res).astype(int) + 1
    if img is None:
        img = np.full((h, w, 3), 255, np.uint8)
    o = np.argsort(P[:, 2])
    x = ((P[o, 0] - lo[0]) / res).astype(int)
    y = (h - 1 - (P[o, 1] - lo[1]) / res).astype(int)
    ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
    img[y[ok], x[ok]] = C[o][ok]
    return img


def _label(img, text, org=(8, 22), scale=0.55):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(img, (org[0] - 4, org[1] - th - 6), (org[0] + tw + 4, org[1] + 6), (255, 255, 255), -1)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def _grey(n, v=225):
    return np.full((n, 3), v, np.uint8)


def check_fusion(capture, out):
    """Replays the point-cloud construction: one frame's inputs -> that frame in 3D ->
    accumulation over frames -> voxel/outlier clean-up -> normals."""
    b = _load(capture)
    dbg = {}
    rec = fuse_depth(b, debug=dbg)
    al = align(rec)
    T, H, m = al["T_plan_world"], al["ceiling_h"] or 2.4, rec.meta
    counts = np.array(dbg["frame_counts"]); ends = np.cumsum(counts)
    raw = to_plan(T, dbg["raw"])
    rawc = (dbg["raw_colors"][:, ::-1] * 255).astype(np.uint8)
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    cams = to_plan(T, np.array([f.T_wc[:3, 3] for f in b.frames]))
    lo, hi = P[:, :2].min(0) - 0.3, P[:, :2].max(0) + 0.3
    res = max((hi - lo).max() / 520, 0.005)
    below = P[:, 2] < H - 0.15            # hide the ceiling so we can see inside
    print(f"frames: {len(b.frames)}   points per frame: median {np.median(counts):.0f}, max {counts.max()}")
    print(f"raw points {m['n_raw']:,} -> 2 cm voxel average {m['n_voxel']:,} -> "
          f"outliers removed {m['n_final']:,}   normals flipped toward camera: {100 * m['n_flipped'] / m['n_final']:.0f}%")

    # 1. inputs of one frame
    k = len(b.frames) // 2
    f = b.frames[k]
    rgb = cv2.imread(str(f.rgb))
    D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
    W = 384
    sz = (W, int(W * rgb.shape[0] / rgb.shape[1]))
    dcol = cv2.applyColorMap(np.clip(D / 4000 * 255, 0, 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    dcol[D == 0] = 0
    fig = np.hstack([cv2.resize(rgb, sz), cv2.resize(dcol, sz, interpolation=cv2.INTER_NEAREST)])
    _label(fig, f"frame #{k} RGB {rgb.shape[1]}x{rgb.shape[0]}", (8, 20), 0.45)
    _label(fig, f"depth {D.shape[1]}x{D.shape[0]} (blue 0 m .. red 4 m)", (W + 8, 20), 0.45)
    cv2.imwrite(str(out / "fusion_1_inputs.png"), fig)
    print(f"frame #{k}: fx={f.K[0, 0]:.1f} fy={f.K[1, 1]:.1f} cx={f.K[0, 2]:.1f} cy={f.K[1, 2]:.1f}  "
          f"depth range {D[D > 0].min() / 1000:.2f}-{D.max() / 1000:.2f} m  camera at {cams[k].round(2)} m (plan frame)")

    # 2. that frame back-projected into 3D, seen from above, with its camera and field of view
    img = _topdown(P[below], _grey(below.sum()), lo, hi, res)
    s0 = ends[k] - counts[k]
    fr = raw[s0:ends[k]]
    img = _topdown(fr, rawc[s0:ends[k]], lo, hi, res, img)
    h = img.shape[0]
    px = lambda p: (int((p[0] - lo[0]) / res), int(h - 1 - (p[1] - lo[1]) / res))
    R = T[:3, :3] @ f.T_wc[:3, :3]
    for u in (0, rgb.shape[1]):           # left / right edge of the image -> field-of-view rays
        ray = R @ np.array([(u - f.K[0, 2]) / f.K[0, 0], 0, 1.0])
        cv2.line(img, px(cams[k]), px(cams[k] + 2.0 * ray / np.linalg.norm(ray[:2])), (200, 120, 0), 1, cv2.LINE_AA)
    cv2.circle(img, px(cams[k]), 6, (200, 120, 0), -1)
    _label(img, f"frame #{k}: {len(fr):,} points (colour) on the final cloud (grey)")
    cv2.imwrite(str(out / "fusion_2_one_frame.png"), img)

    # 3. accumulation over frames
    panels = []
    for n in [1, max(2, len(b.frames) // 10), len(b.frames) // 2, len(b.frames)]:
        sub = slice(0, ends[n - 1], max(1, ends[n - 1] // 1_500_000))
        Q, C = raw[sub], rawc[sub]
        keep = Q[:, 2] < H - 0.15
        pimg = _topdown(Q[keep], C[keep], lo, hi, res)
        cv2.polylines(pimg, [np.array([px(c) for c in cams[:n]], np.int32)], False, (200, 120, 0), 2)
        _label(pimg, f"after {n} frames: {ends[n - 1]:,} raw points")
        panels.append(pimg)
    cv2.imwrite(str(out / "fusion_3_growth.png"), np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))

    # 4. clean-up, as a side-view slice through the longest observed wall
    room = extract_room(P, N, al["ceiling_h"], m["voxel"], 0.9)
    wall = max((w for w in room["walls"] if w["observed"]), key=lambda w: w["length_m"], default=room["walls"][0])
    ax = 0 if wall["axis"] == "x" else 1                  # wall is the plane P[:, ax] = coord
    mid = (wall["p0"][1 - ax] + wall["p1"][1 - ax]) / 2   # middle of the wall, measured along it
    inside = np.sign(np.mean(np.array(room["polygon"])[:, ax]) - wall["coord"]) or 1.0
    cres, half = 0.005, 0.75
    wpx, hpx = int(2 * half / cres), int((H + 0.2) / cres)
    panels = []
    for name, Q in [("raw", raw), ("2 cm voxel average", to_plan(T, dbg["voxel"])), ("outliers removed", P)]:
        sel = np.abs(Q[:, 1 - ax] - mid) < 0.15                  # 30 cm thick slice across the wall
        off, z = (Q[sel, ax] - wall["coord"]) * inside, Q[sel, 2]  # off > 0 = into the room
        ok = (np.abs(off) < half) & (z > -0.1) & (z < H + 0.1)
        off, z = off[ok], z[ok]
        cnt = np.zeros((hpx, wpx))
        np.add.at(cnt, (np.clip(((H + 0.1 - z) / cres).astype(int), 0, hpx - 1),
                        np.clip(((off + half) / cres).astype(int), 0, wpx - 1)), 1)
        pimg = cv2.cvtColor((255 - 255 * np.log1p(cnt) / np.log1p(cnt.max())).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.line(pimg, (wpx // 2, 0), (wpx // 2, hpx), (0, 0, 220), 1)                       # detected wall
        cv2.line(pimg, (0, int((H + 0.1) / cres)), (wpx, int((H + 0.1) / cres)), (220, 120, 0), 1)  # floor
        near = (np.abs(off) < 0.15) & (z > 0.3) & (z < H - 0.2)
        _label(pimg, f"{name}: {ok.sum():,} pts", (8, 20), 0.45)
        _label(pimg, f"+-15 cm of wall: std {100 * np.std(off[near]):.1f} cm", (8, 44), 0.45)
        panels.append(pimg)
    _label(panels[0], "behind wall | room", (wpx // 2 - 75, hpx - 12), 0.45)
    cv2.imwrite(str(out / "fusion_4_cleanup.png"), np.hstack(panels))
    print(f"slice: 30 cm thick, across wall {wall['id']} ({wall['axis']}={wall['coord']:.2f}); "
          f"red = detected wall, blue = floor, darker = more points")

    # 5. normals -> which later stage uses each point
    Pb, Nb = P[below], N[below]
    vert = np.abs(Nb[:, 2]) < 0.2
    wx, wy = vert & (np.abs(Nb[:, 0]) > 0.9), vert & (np.abs(Nb[:, 1]) > 0.9)
    classes = {"up (floor, tops)": (Nb[:, 2] > 0.9, (150, 150, 150)),
               "down (undersides)": (Nb[:, 2] < -0.9, (220, 140, 60)),
               "wall facing x": (wx, (40, 40, 220)),
               "wall facing y": (wy, (40, 170, 40)),
               "wall off-axis": (vert & ~wx & ~wy, (0, 150, 255)),
               "oblique / noisy": (~vert & (np.abs(Nb[:, 2]) <= 0.9), (225, 225, 225))}
    sides = []
    for title, names in [("horizontal -> footprint stage", ["up (floor, tops)", "down (undersides)"]),
                         ("vertical -> walls stage", ["wall off-axis", "wall facing x", "wall facing y"])]:
        msk = classes["oblique / noisy"][0]
        img = _topdown(Pb[msk], _grey(msk.sum(), 225), lo, hi, res)
        for i, n in enumerate(names):
            msk, col = classes[n]
            img = _topdown(Pb[msk], np.full((msk.sum(), 3), col, np.uint8), lo, hi, res, img)
            cv2.rectangle(img, (8, 34 + 20 * i), (22, 48 + 20 * i), col, -1)
            _label(img, f"{n}: {100 * msk.mean():.0f}%", (28, 47 + 20 * i), 0.45)
        _label(img, title, (8, 20), 0.5)
        _label(img, f"light grey = oblique/noisy normals ({100 * classes['oblique / noisy'][0].mean():.0f}%), "
                    f"used by neither", (8, img.shape[0] - 10), 0.4)
        sides.append(img)
    cv2.imwrite(str(out / "fusion_5_normals.png"), np.hstack(sides))
    print("normal classes (below ceiling): " + ", ".join(f"{n} {100 * k.mean():.0f}%" for n, (k, _) in classes.items()))
    print(f"-> {out}/fusion_1_inputs.png .. fusion_5_normals.png")


# ---------------------------------------------------------------- cloud / align
def _cloud(capture):
    b = _load(capture)
    return b, fuse_depth(b)


def check_cloud(capture, out):
    import open3d as o3d
    b, rec = _cloud(capture)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(rec.points))
    pcd.normals = o3d.utility.Vector3dVector(rec.normals)
    if rec.colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(rec.colors)
    o3d.io.write_point_cloud(str(out / "cloud.ply"), pcd)
    ext = rec.points.max(0) - rec.points.min(0)
    print(f"points: {len(rec.points)} from {len(b.frames)} frames  bbox: {ext.round(2)} m")
    print(f"-> {out}/cloud.ply   view: uv run fp check view {out}/cloud.ply")


def check_align(capture, out):
    b, rec = _cloud(capture)
    al = align(rec)
    T = al["T_plan_world"]
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    vox = rec.meta["voxel"]
    print(f"yaw {al['yaw_deg']:.1f} deg  ceiling height: {al['ceiling_h']}")
    print("horizontal layers (z above detected floor, area m^2):")
    for name, m in [("up-facing  ", N[:, 2] > 0.9), ("down-facing", N[:, 2] < -0.9)]:
        h, e = np.histogram(P[m, 2], np.arange(-0.5, 3.2, 0.05))
        top = [(round(e[i], 2), round(h[i] * vox ** 2, 2)) for i in np.argsort(-h)[:6] if h[i]]
        print(f"  {name} {sorted(top)}")
    near_floor = np.abs(P[:, 2]) < 0.03
    flat = near_floor & (np.abs(N[:, 2]) > 0.9)
    print(f"normal sanity: {100 * (N[flat, 2] > 0).mean():.0f}% of floor points face up (want > 95%)")
    cams = np.array([to_plan(T, f.T_wc[None, :3, 3])[0] for f in b.frames])
    print(f"camera height above floor p5/p50/p95: {np.percentile(cams[:, 2], [5, 50, 95]).round(2)} m (handheld: ~1.0-1.7)")
    vert = np.abs(N[:, 2]) < 0.2
    ang = np.degrees(np.mod(np.arctan2(N[vert, 1], N[vert, 0]), np.pi / 2))
    on_axis = np.minimum(ang, 90 - ang) < 5
    print(f"Manhattan fit: {100 * on_axis.mean():.0f}% of wall normals within 5 deg of x/y axes")


def _prep_room(capture):
    b, rec = _cloud(capture)
    al = align(rec)
    T = al["T_plan_world"]
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    dbg = {}
    room = extract_room(P, N, al["ceiling_h"], rec.meta["voxel"], 0.9, debug=dbg)
    return P, N, al, room, dbg


def _bev_canvas(P, res=0.01):
    lo = P[:, :2].min(0) - 0.3
    hi = P[:, :2].max(0) + 0.3
    w, h = ((hi - lo) / res).astype(int)
    px = lambda x, y: (int((x - lo[0]) / res), int(h - 1 - (y - lo[1]) / res))
    return np.full((h, w, 3), 255, np.uint8), px, lo, h


def check_walls(capture, out):
    P, N, al, room, dbg = _prep_room(capture)
    img, px, lo, h = _bev_canvas(P)
    H = al["ceiling_h"] or 2.4
    band = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > 0.3) & (P[:, 2] < H - 0.2)
    for x, y in P[band, :2]:
        img[px(x, y)[1], px(x, y)[0]] = (150, 150, 150)
    print(f"{'axis':<5}{'coord m':>9}{'area m2':>9}{'spread cm':>10}{'along m':>9}  note")
    for l in sorted(dbg["lines"], key=lambda l: (l["axis"], l["coord"])):
        ax, c = l["axis"], l["coord"]
        spread = 100 * l["spread_m"]
        a, bb = np.percentile(l["other"], [2, 98])
        note = "SMEARED (> 3 cm)" if spread > 3 else ""
        print(f"{'x' if ax == 0 else 'y':<5}{c:>9.3f}{l['area_m2']:>9.2f}{spread:>10.1f}{bb - a:>9.2f}  {note}")
        p0 = px(c, a) if ax == 0 else px(a, c)
        p1 = px(c, bb) if ax == 0 else px(bb, c)
        cv2.line(img, p0, p1, (0, 0, 220) if spread > 3 else (0, 150, 0), 2)
    cv2.imwrite(str(out / "walls.png"), img)
    print(f"-> {out}/walls.png (grey = wall points, green = sharp wall line, red = smeared)")


def check_footprint(capture, out):
    P, N, al, room, dbg = _prep_room(capture)
    fp, o = dbg["footprint"], dbg["origin"]
    s = int(RES / 0.01)
    img = np.repeat(np.repeat(np.where(fp.T[::-1, :, None], 200, 255).astype(np.uint8), s, 0), s, 1)
    img = np.ascontiguousarray(np.repeat(img, 3, 2))
    h = img.shape[0]
    px = lambda x, y: (int((x - o[0]) / 0.01), int(h - 1 - (y - o[1]) / 0.01))
    shade = img.copy()
    for cell in dbg["cells"]:
        x0, y0, x1, y1 = cell.bounds
        cv2.rectangle(shade, px(x0, y1), px(x1, y0), (120, 220, 120), -1)
    img = cv2.addWeighted(shade, 0.35, img, 0.65, 0)
    wall_x = {round(l["coord"], 3) for l in dbg["lines"] if l["axis"] == 0}
    wall_y = {round(l["coord"], 3) for l in dbg["lines"] if l["axis"] == 1}
    for x in dbg["xs"]:
        col = (200, 0, 0) if round(x, 3) in wall_x else (0, 0, 220)
        cv2.line(img, px(x, dbg["ys"][0]), px(x, dbg["ys"][-1]), col, 1)
    for y in dbg["ys"]:
        col = (200, 0, 0) if round(y, 3) in wall_y else (0, 0, 220)
        cv2.line(img, px(dbg["xs"][0], y), px(dbg["xs"][-1], y), col, 1)
    poly = np.array([px(*p) for p in room["polygon"]], np.int32)
    cv2.polylines(img, [poly], True, (0, 0, 0), 3)
    cv2.imwrite(str(out / "footprint.png"), img)
    print(f"grid: {len(dbg['xs'])} x-lines, {len(dbg['ys'])} y-lines, {len(dbg['cells'])} interior cells -> "
          f"{len(room['walls'])} walls, area {room['area_m2']} m^2")
    print(f"-> {out}/footprint.png (grey = footprint, blue = wall line, red = footprint-edge fallback, "
          f"green = interior cell, black = final polygon)")


def view(ply: Path):
    import open3d as o3d
    o3d.visualization.draw_geometries([o3d.io.read_point_cloud(str(ply))], point_show_normal=False)


def run(stage: str, capture: Path, out: Path | None, filter_frames: bool = False):
    global FILTER
    FILTER = filter_frames
    if stage == "view":
        return view(capture)
    out = out or Path("out") / capture.name / ("check-filtered" if filter_frames else "check")
    out.mkdir(parents=True, exist_ok=True)
    for s in STAGES if stage == "all" else [stage]:
        print(f"\n=== {s}")
        globals()[f"check_{s}"](capture, out)
