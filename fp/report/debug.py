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
