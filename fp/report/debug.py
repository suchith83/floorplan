"""Top-down debug image: points by surface type + room polygon. Used to diagnose failures."""
from __future__ import annotations

import cv2
import numpy as np

RES = 0.01


def bev_png(path, P, N, plan, ceiling_h=None):
    lo = P[:, :2].min(0) - 0.3
    hi = P[:, :2].max(0) + 0.3
    w, h = ((hi - lo) / RES).astype(int)
    img = np.full((h, w, 3), 255, np.uint8)
    px = lambda q: (((q[..., 0] - lo[0]) / RES).astype(int), (h - 1 - (q[..., 1] - lo[1]) / RES).astype(int))
    wall = np.abs(N[:, 2]) < 0.2
    floor = (N[:, 2] > 0.9) & (np.abs(P[:, 2]) < 0.08)
    ceil = (N[:, 2] < -0.9) & (P[:, 2] > (ceiling_h or 9) - 0.1)
    for m, col in [(floor, (225, 225, 225)), (ceil, (240, 215, 190)), (wall, (40, 40, 40))]:
        x, y = px(P[m])
        img[np.clip(y, 0, h - 1), np.clip(x, 0, w - 1)] = col
    for r in plan["rooms"]:
        for wl in r["walls"]:
            a, b = px(np.array(wl["p0"])), px(np.array(wl["p1"]))
            col = (60, 160, 40) if wl["observed"] else (40, 40, 220)
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), col, 2)
    cv2.imwrite(str(path), img)


def _canvas(P, res, pad=0.3):
    lo = P[:, :2].min(0) - pad
    w, h = ((P[:, :2].max(0) + pad - lo) / res).astype(int) + 1
    px = lambda q: np.stack([((q[..., 0] - lo[0]) / res), (h - 1 - (q[..., 1] - lo[1]) / res)], -1).astype(int)
    return np.full((h, w, 3), 255, np.uint8), px


def fusion_topdown_png(path, P, colors, cams, rooms=(), res=0.01):
    """Top-down view of the fused cloud in plan coordinates: RGB of the highest point in each 1 cm cell,
    the camera path in red (start = green dot), room polygons in blue with their ids, and a 1 m scale bar."""
    img, px = _canvas(np.vstack([P[:, :2], cams[:, :2]]), res)
    o = np.argsort(P[:, 2])                      # draw low first, so the top surface wins
    q = px(P[o])
    img[q[:, 1], q[:, 0]] = (colors[o, ::-1] * 255).astype(np.uint8) if colors is not None else 90
    c = px(cams[:, :2])
    cv2.polylines(img, [c.reshape(-1, 1, 2)], False, (0, 0, 230), 2)
    cv2.circle(img, tuple(map(int, c[0])), 7, (0, 170, 0), -1)
    for r in rooms:
        poly = px(np.array(r["polygon"]))
        cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, (220, 120, 0), 2)
        cx, cy = poly.mean(0).astype(int)
        cv2.putText(img, r["id"], (int(cx) - 12, int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 120, 0), 2)
    h = img.shape[0]
    cv2.line(img, (20, h - 20), (20 + int(1 / res), h - 20), (0, 0, 0), 3)
    cv2.putText(img, "1 m", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    cv2.imwrite(str(path), img)


def wall_slice_png(path, P, N, line, ceiling_h=None, res=0.005, half_depth=0.15, band=0.3):
    """A thin cut through one wall: points within +-band/2 m (along the wall) of the 10 cm stretch with the most
    wall points, drawn as (offset from the plane, height). A sharp wall is a thin vertical stripe; smear shows
    as width. The red lines mark the plane and +-1 robust sigma (the `spread` used for thickness)."""
    ax, c = line["axis"], line["coord"]
    o = line["other"]
    h, e = np.histogram(o, np.arange(o.min(), o.max() + 0.1, 0.1))
    mid = float(e[np.argmax(h)] + 0.05)
    sel = (np.abs(P[:, ax] - c) < half_depth) & (np.abs(P[:, 1 - ax] - mid) < band / 2)
    d, z = P[sel, ax] - c, P[sel, 2]
    zmax = max(2.0, float(z.max()) if len(z) else 2.0, (ceiling_h or 0) + 0.1)
    scale = 6                                    # draw the depth axis 6x wider than the height axis
    W, H = int(2 * half_depth / res * scale), int((zmax + 0.2) / res)
    img = np.full((H, W, 3), 255, np.uint8)
    u = ((d + half_depth) / res * scale).astype(int).clip(0, W - 1)
    v = (H - 1 - (z + 0.1) / res).astype(int).clip(0, H - 1)
    img[v, u] = (40, 40, 40)
    s = line["spread_m"]
    for off, col in ((0.0, (0, 0, 220)), (-s, (120, 120, 255)), (s, (120, 120, 255))):
        x = int((off + half_depth) / res * scale)
        cv2.line(img, (x, 0), (x, H - 1), col, 1)
    for zz in np.arange(0, zmax + 0.01, 0.5):     # height ticks every 50 cm
        y = int(H - 1 - (zz + 0.1) / res)
        cv2.line(img, (0, y), (12, y), (0, 0, 0), 1)
        cv2.putText(img, f"{zz:.1f}m", (14, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    for cm in (-10, -5, 5, 10):                   # depth ticks every 5 cm
        x = int((cm / 100 + half_depth) / res * scale)
        cv2.line(img, (x, H - 1), (x, H - 10), (0, 0, 0), 1)
    label = f"{'x' if ax == 0 else 'y'}={c:.2f} m at {mid:.1f}  sigma {100 * s:.1f} cm (whole wall)"
    cv2.putText(img, label, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    cv2.putText(img, "x6 horizontal; ticks 5 cm", (5, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 200), 1)
    cv2.imwrite(str(path), img)


def drift_ablation_png(path, panels, res=0.01):
    """debug/drift_ablation.png: side-by-side top-down views of the wall band (0.3-2.1 m, vertical surfaces),
    one panel per run (drift correction off | on), with that run's stitched room polygons in blue. A sharp wall is a thin dark line; a wall seen twice by
    drifted poses shows as two lines. Double walls found by fp.recon.drift.wall_quality are drawn in red.
    panels: [{"title": str, "lines": [str], "P": (n,3), "N": (n,3), "double_walls": [{axis, coord_a, coord_b, lo, hi}]}]"""
    imgs = []
    allP = np.vstack([p["P"][:, :2] for p in panels])
    for p in panels:
        img, px = _canvas(allP, res)
        P, N = p["P"], p["N"]
        m = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > 0.3) & (P[:, 2] < 2.1)
        q = px(P[m])
        h, w = img.shape[:2]
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        cnt = np.zeros((h, w), np.int32)
        np.add.at(cnt, (q[ok, 1], q[ok, 0]), 1)
        shade = np.clip(255 - 25 * cnt, 40, 255).astype(np.uint8)   # more wall points per cell -> darker
        img[:] = shade[..., None]
        for d in p.get("double_walls", []):
            for c in (d["coord_a"], d["coord_b"]):
                a = np.array([c, d["lo"]] if d["axis"] in (0, "x") else [d["lo"], c])
                b = np.array([c, d["hi"]] if d["axis"] in (0, "x") else [d["hi"], c])
                cv2.line(img, tuple(map(int, px(a))), tuple(map(int, px(b))), (0, 0, 230), 3)
        for r in p.get("rooms", []):          # the stitched footprint of this run
            poly = px(np.array(r["polygon"]))
            cv2.polylines(img, [poly.reshape(-1, 1, 2)], True, (220, 120, 0), 2)
        for k, t in enumerate([p["title"]] + list(p.get("lines", []))):
            cv2.putText(img, t, (20, 40 + 32 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.9 if k == 0 else 0.7, (0, 0, 0), 2)
        cv2.line(img, (20, h - 20), (20 + int(1 / res), h - 20), (0, 0, 0), 3)
        cv2.putText(img, "1 m", (20, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        imgs.append(img)
    sep = np.full((imgs[0].shape[0], 12, 3), 255, np.uint8)
    out = imgs[0]
    for im in imgs[1:]:
        out = np.hstack([out, sep, im])
    cv2.imwrite(str(path), out)
