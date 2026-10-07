"""Work order 02, experiment D: which fusion settings give the thinnest walls?

    uv run python scripts/exp02_fusion.py data/stray/c00a170fe1

Every variant is measured on the SAME wall planes (found once in the baseline cloud, in the same plan
frame), so only the cloud changes. Thickness = robust sigma (1.4826 x MAD) of wall-band points within
15 cm of the plane (fp.geometry.plan.spread). Lower is sharper. Variants:
  confidence >= 1 vs == 2; quality filter on vs off; depth i paired with pose i+k for k in -2..2
  (a depth<->pose time offset would smear walls when the camera turns); 1 cm vs 2 cm voxels."""
from __future__ import annotations

import sys
import time
from dataclasses import replace

import numpy as np

from fp.geometry.align import align, to_plan
from fp.geometry.plan import WALL_BAND, _wall_lines, spread
from fp.ingest import stray
from fp.ingest.quality import select_frames
from fp.recon.lidar_fuse import fuse_depth

TOP_WALLS = 12  # the 12 largest wall planes: small ones are furniture sides


def thickness(rec, T, lines):
    P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
    band = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > WALL_BAND[0]) & (P[:, 2] < WALL_BAND[1])
    out = []
    for l in lines:
        ax = l["axis"]
        m = band & (np.abs(N[:, ax]) > 0.9)
        out.append(spread(P[m, ax], l["coord"]))
    return 100 * np.array(out)


def shifted(frames_all, kept, k):
    """Kept frames with each depth map paired to the pose (and K) of frame i+k."""
    by_idx = {f.video_index: f for f in frames_all}
    out = []
    for f in kept:
        g = by_idx.get(f.video_index + k)
        if g is not None:
            out.append(replace(f, T_wc=g.T_wc, K=g.K))
    return out


def main(path):
    b = stray.load(path)
    frames_all = list(b.frames)
    kept, st = select_frames(frames_all)
    print(f"frames: {len(frames_all)} all, {len(kept)} after the quality filter {st}")

    def run(frames, **kw):
        t = time.perf_counter()
        rec = fuse_depth(replace(b, frames=frames), **kw)
        return rec, time.perf_counter() - t

    base, tb = run(kept)
    al = align(base)
    T = al["T_plan_world"]
    P, N = to_plan(T, base.points), base.normals @ T[:3, :3].T
    lines = sorted(_wall_lines(P, N, al["ceiling_h"], base.meta["voxel"]), key=lambda l: -l["area_m2"])[:TOP_WALLS]
    rows = [("conf==2, filter, 2 cm (baseline)", base, tb)]
    rows.append(("conf>=1, filter, 2 cm", *run(kept, min_conf=1)))
    rows.append(("conf==2, no filter (all frames)", *run(frames_all)))
    rows.append(("conf>=1, no filter (all frames)", *run(frames_all, min_conf=1)))
    for k in (-2, -1, 1, 2):
        rows.append((f"conf==2, filter, depth i + pose i{k:+d}", *run(shifted(frames_all, kept, k))))
    rows.append(("conf==2, filter, 1 cm voxel", *run(kept, voxel=0.01)))
    print(f"{len(lines)} wall planes, areas {[round(l['area_m2'], 1) for l in lines]} m2")
    print(f"{'variant':42s} {'median cm':>9s} {'mean cm':>8s} {'points':>9s} {'fuse s':>7s}")
    for name, rec, t in rows:
        th = thickness(rec, T, lines)
        print(f"{name:42s} {np.median(th):9.2f} {th.mean():8.2f} {len(rec.points):9d} {t:7.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/stray/c00a170fe1")
