"""Whole-floor footprint -> rooms and the doors between them.

A doorway is a narrow neck in the floor area. The distance transform of the footprint peaks in the
middle of each room and dips at necks, so a watershed from its peaks splits the floor at doorways.
Watershed over-splits long rooms (a galley kitchen has several peaks), so neighbours whose shared
border is nearly as wide as the WIDER of the two are merged back: that border is not a door. (Against
the narrower one, a corridor always matched its own opening and merged into the room it leads from.)"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed

MIN_ROOM_AREA = 2.0       # m^2; smaller regions join their neighbour (closets, alcoves)
MIN_CENTRE_GAP = 0.8      # m between room centres. Was 1.2: a 1.2 m corridor's centre ridge (0.6 m) lies within
                          # 1.2 m of the higher ridge of the room it opens into, so no corridor ever got a seed
                          # and corridors merged into rooms. Extra seeds in big rooms are merged back below.
MIN_HALF_WIDTH = 0.4      # m: a room centre is this far from the nearest edge (corridors from 0.8 m wide)
NECK_RATIO = 0.7          # border >= 70 % of the wider region's width -> same room
DOOR_MAX = 1.4            # m; a wider neck is an open-plan opening, not a door


def _borders(lab: np.ndarray) -> dict[tuple[int, int], np.ndarray]:
    """Pixels on the border between each pair of touching regions (4-neighbourhood)."""
    out: dict[tuple[int, int], list] = {}
    for di, dj in ((1, 0), (0, 1)):
        a, b = lab[: lab.shape[0] - di, : lab.shape[1] - dj], lab[di:, dj:]
        m = (a != b) & (a > 0) & (b > 0)
        for i, j in zip(*np.nonzero(m)):
            key = (min(a[i, j], b[i, j]), max(a[i, j], b[i, j]))
            out.setdefault(key, []).append((i, j))
    return {k: np.array(v) for k, v in out.items()}


def _border_width(px: np.ndarray, res: float) -> float:
    """Length of a border: the longer side of its pixels' bounding box."""
    return float((px.max(0) - px.min(0) + 1).max() * res)


def split_rooms(fp: np.ndarray, res: float) -> np.ndarray:
    """Footprint mask -> label image (0 outside, 1..n rooms)."""
    dist = ndimage.distance_transform_edt(fp) * res
    peaks = peak_local_max(dist, min_distance=max(1, int(MIN_CENTRE_GAP / res)), threshold_abs=MIN_HALF_WIDTH,
                           labels=fp.astype(int), exclude_border=False)
    if len(peaks) < 2:
        return fp.astype(int)
    markers = np.zeros(fp.shape, int)
    markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1)
    lab = watershed(-dist, markers, mask=fp)
    changed = True
    while changed:  # merge a pair at a time, widest relative border first
        changed = False
        width = {k: 2 * dist[lab == k].max() for k in np.unique(lab) if k}
        area = {k: (lab == k).sum() * res * res for k in width}
        best, best_score = None, 0.0
        for (a, b), px in _borders(lab).items():
            score = _border_width(px, res) / max(width[a], width[b])
            small = min(area[a], area[b]) < MIN_ROOM_AREA
            if (score >= NECK_RATIO or small) and (score + 10 * small) > best_score:
                best, best_score = (a, b), score + 10 * small
        if best:
            lab[lab == best[1]] = best[0]
            changed = True
    _, lab = np.unique(lab, return_inverse=True)  # relabel 0..n, keeping 0 = outside
    return lab.reshape(fp.shape)


def doors(lab: np.ndarray, origin: np.ndarray, res: float) -> list[dict]:
    """One connection per pair of touching rooms: its width and centre (plan metres)."""
    out = []
    for (a, b), px in sorted(_borders(lab).items()):
        w = _border_width(px, res)
        c = origin + px.mean(0) * res
        out.append({"rooms": [f"R{a}", f"R{b}"], "kind": "door" if w <= DOOR_MAX else "opening",
                    "width_m": round(w, 2), "center": [round(float(c[0]), 3), round(float(c[1]), 3)]})
    return out
