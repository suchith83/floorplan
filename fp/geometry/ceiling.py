"""Per-room ceiling height: the room's own ceiling layer minus the room's own floor layer.

One sentence: inside the room polygon, the ceiling is the largest down-facing horizontal layer of at
least 1 m^2 at 2.1-3.6 m above the room's floor, the floor is the up-facing layer near z = 0, the height
is the difference of their medians, and the 90 % interval is a 1 cm systematic term plus 1.645 standard
errors, where each layer's standard error is its robust spread over the square root of the number of
25 cm patches it covers (neighbouring points are not independent, patches roughly are).

A room with no qualifying down-facing layer is reported as not observed with a reason; never a guess.
"""
from __future__ import annotations

import numpy as np
from shapely import contains_xy
from shapely.geometry import Polygon

from fp.geometry.align import CEILING_MIN_AREA, CEILING_RANGE

# Ceiling search band above the room's floor: reuse align's (2.1, 3.6) m. The work order's "> 1.8 m"
# would admit kitchen cabinet undersides (round 1 picked one at 1.86 m); 2.1 m is the lowest legal
# habitable ceiling we expect in the brief's housing, and above 3.6 m we only saw stray points (3.98 m).
CEILING_BAND = CEILING_RANGE
# A ceiling layer must cover at least 1 m^2 (align's CEILING_MIN_AREA): smaller down-facing patches are
# shelf undersides, light fittings or door heads, not a ceiling.
LAYER_MIN_AREA = CEILING_MIN_AREA
BIN = 0.02            # m: histogram bin; LiDAR layers are ~1-2 cm thick, so a 2 cm bin resolves them
LAYER_HALF = 0.04     # m: points within +-4 cm of a layer belong to it (2 bins each side; matches align._refine)
LAYER_SEP = 0.10      # m: peaks closer than 10 cm are one smeared layer (drift smear is a few cm), not two
FLOOR_HALF = 0.05     # m: the room's floor = up-facing points within +-5 cm of the plan floor (drift tilt is a few cm)
FLOOR_MIN_AREA = 0.5  # m^2: less floor than a small rug and we do not trust a local floor; use the plan floor
PATCH = 0.25          # m: independence scale; drift and smear are correlated over tens of cm, so we count
                      # 25 cm patches, not points, as independent samples of a layer's height
PATCH_MIN_FILL = 0.10 # a patch counts if the layer covers at least 10 % of it (drops lone stray points)
SYS_HALF = 0.01       # m: LiDAR range bias between two surfaces at different distances (~1 cm), not averaged away
FLOOR_FALLBACK_HALF = 0.03  # m: extra when the room's floor was not seen: the global floor can be a few cm
                            # off in one room of a whole-flat scan (drift tilt)
Z90 = 1.645           # two-sided 90 % normal quantile
MAD_TO_SIGMA = 1.4826 # MAD -> standard deviation for a normal distribution


def _cells(xy: np.ndarray, cell: float) -> np.ndarray:
    """One int64 key per point for its cell on a cell-sized xy grid (a fast 1-D unique)."""
    ij = np.floor(xy / cell).astype(np.int64)
    return ij[:, 0] * (1 << 32) + ij[:, 1]


def _xy_area(xy: np.ndarray, cell: float) -> float:
    """Area covered on a cell-sized xy grid (a thick or doubly scanned layer does not count twice)."""
    return len(np.unique(_cells(xy, cell))) * cell ** 2 if len(xy) else 0.0


def _patches(xy: np.ndarray, voxel: float) -> int:
    """Number of distinct PATCH x PATCH cells holding at least PATCH_MIN_FILL of their area in points."""
    if len(xy) == 0:
        return 0
    _, cnt = np.unique(_cells(xy, PATCH), return_counts=True)
    return int((cnt * voxel ** 2 >= PATCH_MIN_FILL * PATCH ** 2).sum())


def _layer_stats(z: np.ndarray, xy: np.ndarray, z0: float, voxel: float) -> dict:
    """Median height, robust spread, patch count and area of the points within LAYER_HALF of z0."""
    m = np.abs(z - z0) < LAYER_HALF
    zm = float(np.median(z[m]))
    m = np.abs(z - zm) < LAYER_HALF  # re-centre on the median once
    zs = z[m]
    zm = float(np.median(zs))
    sigma = MAD_TO_SIGMA * float(np.median(np.abs(zs - zm)))
    return {"z": zm, "sigma": sigma, "n": int(m.sum()), "area_m2": _xy_area(xy[m], voxel),
            "patches": _patches(xy[m], voxel)}


def _peaks(z: np.ndarray, lo: float, hi: float) -> list[float]:
    """Histogram peaks (2 cm bins summed over a 4 cm window), strongest first, non-max suppressed at LAYER_SEP."""
    edges = np.arange(lo, hi + BIN, BIN)
    if len(z) == 0 or len(edges) < 3:
        return []
    h, _ = np.histogram(z, edges)
    h = h + np.r_[h[1:], 0]                       # window [e_i, e_i+2), centred on e_i+1
    centres = edges[1:len(h) + 1]
    left, right = np.r_[-1, h[:-1]], np.r_[h[1:], -1]
    idx = [i for i in np.argsort(-h, kind="stable") if h[i] > 0 and h[i] >= left[i] and h[i] >= right[i]]
    kept: list[float] = []
    for i in idx:
        if all(abs(centres[i] - k) >= LAYER_SEP for k in kept):
            kept.append(float(centres[i]))
    return kept


def _floor(P: np.ndarray, N: np.ndarray, voxel: float) -> dict:
    up = N[:, 2] > 0.9
    sel = up & (np.abs(P[:, 2]) < FLOOR_HALF)
    if sel.sum() * voxel ** 2 >= FLOOR_MIN_AREA:  # cheap upper bound before the real area below
        st = _layer_stats(P[sel, 2], P[sel, :2], float(np.median(P[sel, 2])), voxel)
        if st["area_m2"] >= FLOOR_MIN_AREA and st["patches"] > 0:
            return {**st, "source": "room floor layer"}
    return {"z": 0.0, "sigma": None, "n": int(sel.sum()), "area_m2": _xy_area(P[sel, :2], voxel),
            "patches": 0, "source": "plan floor (room floor not seen)"}


def room_ceiling(P: np.ndarray, N: np.ndarray, polygon: list[list[float]], voxel: float,
                 room_id: str = "") -> dict:
    """Ceiling height of one room. P, N: (n,3) plan-frame points/normals (z up, floor at 0, metres).
    Returns {h, half, observed, method, reason, ceiling, floor, other_layers}; `half` is the LiDAR-scale
    90 % half-width (the caller multiplies by the tier scale)."""
    where = f" inside {room_id}" if room_id else " inside the room"
    poly = Polygon(polygon)
    x0, y0, x1, y1 = poly.bounds
    # only horizontal points matter; test the cheap conditions before the polygon
    idx = np.flatnonzero((P[:, 0] >= x0) & (P[:, 0] <= x1) & (P[:, 1] >= y0) & (P[:, 1] <= y1))
    idx = idx[np.abs(N[idx, 2]) > 0.9]
    idx = idx[contains_xy(poly, P[idx, 0], P[idx, 1])]
    Pr, Nr = np.asarray(P[idx], float), np.asarray(N[idx], float)

    floor = _floor(Pr, Nr, voxel)
    fz = floor["z"]
    lo, hi = fz + CEILING_BAND[0], fz + CEILING_BAND[1]
    dn = Nr[:, 2] < -0.9
    zd, xyd = Pr[dn, 2], Pr[dn, :2]
    band = (zd > lo - LAYER_HALF) & (zd < hi + LAYER_HALF)

    layers = []
    for z0 in _peaks(zd[band], lo - LAYER_HALF, hi + LAYER_HALF):
        st = _layer_stats(zd[band], xyd[band], z0, voxel)
        if lo <= st["z"] <= hi and st["area_m2"] >= LAYER_MIN_AREA:
            layers.append(st)
    # two peaks can re-centre onto the same layer: keep the bigger one
    layers.sort(key=lambda s: -s["area_m2"])
    uniq: list[dict] = []
    for s in layers:
        if all(abs(s["z"] - u["z"]) >= LAYER_SEP for u in uniq):
            uniq.append(s)

    out = {"h": None, "half": None, "observed": False, "method": "", "reason": None,
           "ceiling": None, "floor": floor, "other_layers": []}
    if not uniq:
        top = f"highest down-facing point {zd.max() - fz:.2f} m" if len(zd) else "no down-facing points"
        best = max((_xy_area(xyd[np.abs(zd - z0) < LAYER_HALF], voxel)
                    for z0 in _peaks(zd[band], lo - LAYER_HALF, hi + LAYER_HALF)), default=0.0)
        out["method"] = "not observed"
        out["reason"] = (f"no down-facing layer of at least {LAYER_MIN_AREA:g} m² "
                         f"{CEILING_BAND[0]:g}–{CEILING_BAND[1]:g} m above the floor{where}"
                         + (f" (largest {best:.2f} m²)" if best > 0 else "") + f"; {top}")
        return out

    ceil, others = uniq[0], uniq[1:]
    h = ceil["z"] - fz
    var = ceil["sigma"] ** 2 / max(ceil["patches"], 1)
    if floor["sigma"] is not None:
        var += floor["sigma"] ** 2 / floor["patches"]
        extra, fl = 0.0, "room floor"
    else:
        extra, fl = FLOOR_FALLBACK_HALF, "plan floor (room floor not seen)"
    half = SYS_HALF + Z90 * float(np.sqrt(var)) + extra
    out.update(h=float(h), half=float(half), observed=True, ceiling=ceil,
               method=f"largest down-facing layer in the room minus {fl}",
               other_layers=[{"h": float(s["z"] - fz), "area_m2": float(s["area_m2"])} for s in others])
    return out
