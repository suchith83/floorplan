"""Experiment 03: drift correction ablation (off / posegraph / plane) on one Stray Scanner capture.

    uv run python scripts/exp03_drift.py data/stray/c7d28f72c6 [--cache out/_cache] [--out out/<id>/debug]

For each variant: drift step on the quality-filtered frames (as `fp run` does), full 1 cm fusion
(fp.recon.lidar_fuse.fuse_depth), alignment to the plan frame (fp.geometry.align), then
fp.recon.drift.wall_quality. Prints one table and writes a top-down image per variant
(`drift_exp03_<variant>.png`, vertical surfaces in the wall band, double walls in red)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

from fp.bundle import CaptureBundle
from fp.geometry.align import align, to_plan
from fp.ingest.quality import select_frames
from fp.ingest.stray import load
from fp.recon.drift import correct_drift, wall_quality
from fp.recon.lidar_fuse import fuse_depth

PX = 0.01  # m per pixel in the debug image


def bev(P, N, q, path):
    band = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > 0.3) & (P[:, 2] < 2.1)
    Q = P[band, :2]
    lo = Q.min(0) - 0.2
    sz = np.ceil((Q.max(0) + 0.2 - lo) / PX).astype(int)
    img = np.zeros((sz[1], sz[0]), np.float32)
    ij = ((Q - lo) / PX).astype(int)
    np.add.at(img, (sz[1] - 1 - ij[:, 1], ij[:, 0]), 1)
    img = (255 * np.clip(img / 4, 0, 1)).astype(np.uint8)
    out = cv2.cvtColor(255 - img, cv2.COLOR_GRAY2BGR)
    for d in q["double_walls"]:
        ax = d["axis"]
        for c in (d["coord_a"], d["coord_b"]):
            near = band & (np.abs(P[:, ax] - c) < 0.02) & (np.sign(N[:, ax]) == d["faces"])
            ij = ((P[near, :2] - lo) / PX).astype(int)
            out[sz[1] - 1 - ij[:, 1], ij[:, 0]] = (0, 0, 255)
    cv2.imwrite(str(path), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--cache", type=Path, default=Path("out/_cache"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--variants", default="off,posegraph-step,posegraph,plane")
    a = ap.parse_args()
    out = a.out or Path("out") / a.capture.name / "debug"
    out.mkdir(parents=True, exist_ok=True)
    b = load(a.capture, cache_dir=a.cache)
    frames, _ = select_frames(b.frames)
    rows = []
    for v in a.variants.split(","):
        m, t0 = {}, time.perf_counter()
        if v == "off":
            fr = frames
        else:
            # "<method>-step": one rigid correction per submap instead of the default blend in time
            r = correct_drift(frames, b.up, method=v.removesuffix("-step"), smooth=not v.endswith("-step"))
            fr, m = r.frames, r.metrics
            print(f"[{v}] {r.method}")
            for l in r.loops:
                if l["accepted"] or "pruned" in (l["reason"] or ""):
                    print(f"   loop {l['i']:2d}-{l['j']:2d} fit {l['fitness']:.2f} rmse {l['rmse_cm']:.2f} cm "
                          f"move {l['move_cm']:.1f} cm rot {l['rot_deg']:.2f} deg  residual "
                          f"{l.get('residual_before_cm', '')} -> {l.get('residual_after_cm', '')} {l['reason'] or ''}")
        t_drift = time.perf_counter() - t0
        t0 = time.perf_counter()
        rec = fuse_depth(CaptureBundle("lidar", fr, up=b.up))
        t_fuse = time.perf_counter() - t0
        T = align(rec)["T_plan_world"]
        P, N = to_plan(T, rec.points), rec.normals @ T[:3, :3].T
        q = wall_quality(P, N, rec.meta["voxel"])
        bev(P, N, q, out / f"drift_exp03_{v}.png")
        rows.append((v, q["thickness_median_cm"], q["n_walls"], len(q["double_walls"]),
                     m.get("n_submaps", "-"), f"{m.get('n_loop_accepted', '-')}/{m.get('n_loop_candidates', '-')}",
                     m.get("loop_rmse_before_cm", float("nan")), m.get("loop_rmse_after_cm", float("nan")),
                     m.get("heading_spread_before_deg", float("nan")), m.get("heading_spread_after_deg", float("nan")),
                     m.get("max_correction_cm", 0.0), m.get("max_correction_deg", 0.0), t_drift, t_fuse))
        for d in q["double_walls"]:
            print(f"   [{v}] double wall axis {'xy'[d['axis']]} {d['coord_a']:.2f}/{d['coord_b']:.2f} "
                  f"gap {d['gap_cm']} cm overlap {d['overlap_m']} m faces {d['faces']:+d}")
    print(f"\n{a.capture.name}: {len(frames)} frames kept of {len(b.frames)}")
    hdr = ("variant", "thick_cm", "walls", "doubles", "submaps", "loops", "loop_b_cm", "loop_a_cm", "head_b_deg", "head_a_deg",
           "max_corr_cm", "max_corr_deg", "drift_s", "fuse_s")
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for r in rows:
        print("| " + " | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r) + " |")


if __name__ == "__main__":
    main()
