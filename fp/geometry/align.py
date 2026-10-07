"""World -> plan frame: z up, floor at z=0, dominant walls along x/y (Manhattan)."""
from __future__ import annotations

import numpy as np

from fp.bundle import Recon

MIN_LAYER_AREA = 0.3  # m^2
CEILING_MIN_AREA = 1.0  # m^2
CEILING_RANGE = (2.1, 3.6)  # m above the floor


class FloorNotFound(RuntimeError):
    pass


def _basis_from_up(up):
    z = up / np.linalg.norm(up)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 0, 1.0])
    x = np.cross(a, z); x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])  # rows: new axes


def _refine(zs, z0):
    return float(np.median(zs[np.abs(zs - z0) < 0.04]))


def estimate_up(rec: Recon) -> np.ndarray:
    """Gravity for inputs without an IMU (photos, video). Start from the cameras' mean 'up' (a phone
    is held roughly level; OpenCV camera y points down), then pull it onto the normals of the
    horizontal surfaces near it (floor, ceiling, table tops), narrowing the cone each pass."""
    up = -np.mean([f.T_wc[:3, 1] for f in rec.frames], axis=0)
    up /= np.linalg.norm(up)
    for cone in (0.7, 0.85, 0.95):  # cos 45, 32, 18 degrees
        d = rec.normals @ up
        sel = np.abs(d) > cone
        if sel.sum() < 100:
            break
        v = (rec.normals[sel] * np.sign(d[sel])[:, None]).sum(0)
        up = v / np.linalg.norm(v)
    return up


def _layers(zs, lo, hi, min_pts):
    """Horizontal layers: 2 cm bins summed over a 4 cm window -> [(centre z, count)] with count >= min_pts."""
    h, e = np.histogram(zs, np.arange(lo, hi, 0.02))
    h = h + np.r_[h[1:], 0]
    return [(0.5 * (e[i] + e[i + 1]), int(h[i])) for i in range(len(h)) if h[i] >= min_pts]


def pick_floor(up_layers, cam_z: float) -> float:
    """Floor = the biggest up-facing layer 0.8-2.0 m below the median camera height (handheld), unless a
    layer more than 25 cm lower has at least half its area (a floor under a bed or table top).
    The old rule, 'lowest layer with 0.3 m^2', picked a smeared tail 0.29 m below the floor (REVIEW #7)."""
    cand = [(z, c) for z, c in up_layers if cam_z - 2.0 <= z <= cam_z - 0.8]
    if not cand:
        raise FloorNotFound("Floor not observed: no large horizontal surface 0.8-2.0 m below the camera. "
                            "Re-capture and point the camera at the floor in every room.")
    zb, cb = max(cand, key=lambda zc: zc[1])
    lower = [(z, c) for z, c in cand if z < zb - 0.25 and c >= 0.5 * cb]
    return max(lower, key=lambda zc: zc[1])[0] if lower else zb


def align(rec: Recon, assume_floor_below_camera: float | None = None) -> dict:
    """assume_floor_below_camera: for laser ground truth only. If no floor is seen, put it this far
    below the median camera (it then only sets the height band used to find walls)."""
    up = rec.up if rec.up is not None else estimate_up(rec)
    R0 = _basis_from_up(up)
    P, N = rec.points @ R0.T, rec.normals @ R0.T

    # Manhattan yaw from horizontal normals of vertical surfaces.
    vert = np.abs(N[:, 2]) < 0.2
    ang = np.mod(np.arctan2(N[vert, 1], N[vert, 0]), np.pi / 2)
    h, e = np.histogram(ang, bins=180, range=(0, np.pi / 2))
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(5) / 5, "same")[3:-3]  # circular smooth
    yaw = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    c, s = np.cos(-yaw), np.sin(-yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    R = Rz @ R0
    P, N = rec.points @ R.T, rec.normals @ R.T
    cam_z = float(np.median([f.T_wc[:3, 3] @ R[2] for f in rec.frames]))

    # Layers need real area (absolute, not "fraction of the biggest layer").
    vox = rec.meta.get("voxel", 0.02)
    min_pts = MIN_LAYER_AREA / vox ** 2
    lo, hi = P[:, 2].min() - 0.05, P[:, 2].max() + 0.05
    up_z, dn_z = P[N[:, 2] > 0.9, 2], P[N[:, 2] < -0.9, 2]
    try:
        floor_z = _refine(up_z, pick_floor(_layers(up_z, lo, hi, min_pts), cam_z))
    except FloorNotFound:
        if assume_floor_below_camera is None:
            raise
        floor_z = cam_z - assume_floor_below_camera
    # ceiling = the biggest down-facing layer of at least 1 m^2 at a plausible height (2.1-3.6 m).
    # 'Highest layer' picked stray points (3.98 m); a looser range picked cabinet undersides (1.86 m).
    # None is an honest answer: a handheld camera often does not see enough of the ceiling.
    dn_l = [(z, c) for z, c in _layers(dn_z, lo, hi, CEILING_MIN_AREA / vox ** 2)
            if floor_z + CEILING_RANGE[0] < z < floor_z + CEILING_RANGE[1]]
    ceil_z = _refine(dn_z, max(dn_l, key=lambda zc: zc[1])[0]) if dn_l else None

    T = np.eye(4)
    T[:3, :3] = R
    T[2, 3] = -floor_z
    return {"T_plan_world": T, "ceiling_h": None if ceil_z is None else ceil_z - floor_z,
            "yaw_deg": float(np.degrees(yaw)), "camera_h": cam_z - floor_z,
            "up_source": "sensor" if rec.up is not None else "estimated"}


def to_plan(T, pts):
    return pts @ T[:3, :3].T + T[:3, 3]
