"""Register two floor plans of the same property (repeatability): a top-down 2-D rigid transform.

Both plans are already Manhattan-aligned (fp/geometry/align.py), so the rotation between them is one of
0/90/180/270 degrees plus a small residual. For each of the four yaws: rasterise both sets of wall points
(5 cm cells), find the best translation by FFT cross-correlation, then refine rotation + translation with
2-D point-to-point ICP. The yaw with the most wall points landing within INLIER_M of the other scan's walls
wins. The two scans are called the same property only if that inlier fraction is high and clearly beats
the runner-up yaw (a symmetric box would score the same at 0 and 180)."""
from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

CELL = 0.05        # m: raster for the coarse translation search
INLIER_M = 0.05    # m: a wall point within 5 cm of the other scan's walls agrees
ICP_ITERS = 30
SAME_MIN_INLIERS = 0.5   # at least half the wall points agree...
SAME_MIN_MARGIN = 1.3    # ...and 1.3x more than with the next-best yaw


def _rot(deg: float) -> np.ndarray:
    a = np.radians(deg)
    return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def _raster(P, o, shape):
    img = np.zeros(shape, np.float32)
    ij = ((P - o) / CELL).astype(int)
    ok = (ij >= 0).all(1) & (ij[:, 0] < shape[0]) & (ij[:, 1] < shape[1])
    np.add.at(img, (ij[ok, 0], ij[ok, 1]), 1)
    return np.minimum(img, 1)


def _coarse_shift(A, B):
    """Translation t maximising the overlap of raster(A) and raster(B + t)."""
    o = np.minimum(A.min(0), B.min(0)) - 1
    shape = tuple(((np.maximum(A.max(0), B.max(0)) - o + 1) / CELL).astype(int) + 1)
    ra, rb = _raster(A, o, shape), _raster(B, o, shape)
    cc = fftconvolve(ra, rb[::-1, ::-1], mode="full")
    i, j = np.unravel_index(np.argmax(cc), cc.shape)
    return np.array([i - (shape[0] - 1), j - (shape[1] - 1)]) * CELL


def icp2d(A, B, R, t, iters=ICP_ITERS, max_d=0.3):
    """Refine x -> R x + t taking B onto A (point-to-point, Kabsch per iteration)."""
    tree = cKDTree(A)
    for _ in range(iters):
        Bt = B @ R.T + t
        d, k = tree.query(Bt, distance_upper_bound=max_d)
        m = np.isfinite(d)
        if m.sum() < 10:
            break
        p, q = Bt[m], A[k[m]]
        pc, qc = p.mean(0), q.mean(0)
        U, _, Vt = np.linalg.svd((p - pc).T @ (q - qc))
        dR = (U @ Vt).T
        if np.linalg.det(dR) < 0:
            Vt[-1] *= -1
            dR = (U @ Vt).T
        R, t = dR @ R, dR @ (t - pc) + qc
        max_d = max(INLIER_M * 2, max_d * 0.8)
    return R, t


def register(A: np.ndarray, B: np.ndarray, max_points: int = 40000, seed: int = 0) -> dict:
    """A, B: (n,2) wall points of two plans. Returns the transform taking B into A's frame and the evidence."""
    rng = np.random.default_rng(seed)
    sub = lambda X: X[rng.choice(len(X), min(len(X), max_points), replace=False)]
    A, B = sub(A), sub(B)
    tree = cKDTree(A)
    tries = []
    for yaw in (0, 90, 180, 270):
        R0 = _rot(yaw)
        Br = B @ R0.T
        t0 = _coarse_shift(A, Br)
        R, t = icp2d(A, B, R0, t0)
        d, _ = tree.query(B @ R.T + t)
        tries.append({"yaw0": yaw, "R": R, "t": t, "inliers": float((d < INLIER_M).mean()),
                      "median_cm": float(100 * np.median(d[d < INLIER_M])) if (d < INLIER_M).any() else None})
    tries.sort(key=lambda x: -x["inliers"])
    best, second = tries[0], tries[1]
    yaw = float(np.degrees(np.arctan2(best["R"][1, 0], best["R"][0, 0])))
    same = best["inliers"] >= SAME_MIN_INLIERS and best["inliers"] >= SAME_MIN_MARGIN * max(second["inliers"], 1e-9)
    return {"R": best["R"], "t": best["t"], "yaw_deg": yaw, "inliers": best["inliers"],
            "median_inlier_cm": best["median_cm"], "runner_up_inliers": second["inliers"],
            "runner_up_yaw0": second["yaw0"], "same_property": bool(same)}
