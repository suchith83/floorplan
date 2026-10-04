"""Evaluation on public data: three ARKitScenes scans of one kitchen (visit 483621), each with a Faro
laser scan. Every input type runs on the same scans, so their errors are directly comparable:

  original        the code before this round of fixes (plans kept in eval/baseline/)
  lidar-nofilter  current code, frame filter off: isolates what the filter does
  lidar           LiDAR with the frame filter (the shipped fix)
  video         RGB frames only (120), depth and poses predicted by MapAnything on Modal
  photos-24/12  RGB only, 24 or 12 frames: fewer views, as a photo set

The fix was tuned on 48018559 only; 48018560 and 48018562 are held out.
Usage: uv run python eval/run_public.py [--configs lidar-before,lidar] [--scenes 48018559,...]
Writes eval/results.json and eval/results.md (the tables summarised in README)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from arkit_gt import compare, gt_sides  # noqa: E402

from fp.cli import run  # noqa: E402

SCENES = ["48018559", "48018560", "48018562"]
TUNED_ON = "48018559"
CONFIGS = {
    "original": None,
    "lidar-nofilter": dict(filter_frames=False, damage=False),
    "lidar": dict(filter_frames=True, damage=False),
    "video": dict(as_source="video", max_frames=120, damage=False),
    "photos-24": dict(as_source="photos", max_frames=24, damage=False),
    "photos-12": dict(as_source="photos", max_frames=12, damage=False),
}
RESULTS = Path(__file__).parent / "results.json"


def evaluate(configs, scenes):
    results = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    for cfg in configs:
        for s in scenes:
            out = Path("out/eval") / s / cfg
            print(f"=== {cfg} {s}", flush=True)
            try:
                if CONFIGS[cfg] is None:  # the original code's plan, run before the fixes
                    plan = json.loads((Path(__file__).parent / "baseline" / f"{s}.json").read_text())
                    plan["source"].setdefault("frame_filter", None)
                else:
                    plan = run(Path("data/arkitscenes") / s, out, **CONFIGS[cfg])
            except Exception as e:  # a failed run is a result too
                results[f"{cfg}|{s}"] = {"config": cfg, "scene": s, "error": f"{type(e).__name__}: {e}"}
                print("   failed:", e)
                RESULTS.write_text(json.dumps(results, indent=1))
                continue
            room = max(plan["rooms"], key=lambda r: r["area_m2"])  # the kitchen; other regions are glimpses
            rows = compare(room, gt_sides(Path("data/arkitscenes") / s))
            results[f"{cfg}|{s}"] = {
                "config": cfg, "scene": s, "held_out": s != TUNED_ON, "rows": rows,
                "frames": plan["source"]["n_frames"], "frame_filter": plan["source"].get("frame_filter"),
                "wall_lines": room["n_wall_candidates"], "walls": len(room["walls"]), "floor": plan["source"].get("floor", "detected"),
                "median_spread_cm": float(np.median([w["spread_cm"] for w in room["walls"] if w.get("spread_cm") is not None] or [np.nan])),
                "ceiling_m": room["ceiling_h_m"], "area_m2": room["area_m2"], "runtime_s": plan["debug"]["runtime_s"]}
            print("  ", rows)
            RESULTS.write_text(json.dumps(results, indent=1))  # after every run: a stalled GPU loses nothing
    return results


def table(results) -> str:
    lines = ["| Input | Scan | Frames | Wall lines | Median wall thickness (cm) | Laser width (m) | Ours (m) | Error (cm) | Confidence |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    errs, summary = {}, []
    for key in sorted(results, key=lambda k: (list(CONFIGS).index(results[k]["config"]), results[k]["scene"])):
        r = results[key]
        tag = f"{r['scene']}{'' if r.get('held_out', True) else ' (tuning)'}{' floor assumed' if 'assumed' in r.get('floor', '') else ''}"
        if "error" in r:
            lines.append(f"| {r['config']} | {tag} | | | | | | failed: {r['error'][:60]} | |")
            continue
        for row in r["rows"]:
            lines.append(f"| {r['config']} | {tag} | {r['frames']} | {r['wall_lines']} | {'—' if np.isnan(r['median_spread_cm']) else f"{r['median_spread_cm']:.1f}"} | "
                         f"{row['laser_m']:.3f} | {row['pred_m'] if row['pred_m'] is not None else 'not measured'} | "
                         f"{row['err_cm'] if row['err_cm'] is not None else '—'} | {row['confidence'] if row['confidence'] is not None else '—'} |")
            if row["err_cm"] is not None:
                errs.setdefault(r["config"], []).append((row["err_cm"], row["confidence"], r["scene"], row["pred_m"]))
    lines += ["", "| Input | Measured | MAE (cm) | Max (cm) | Repeat SD of width (cm) | Mean confidence |", "| --- | --- | --- | --- | --- | --- |"]
    all_pairs = []
    for cfg in CONFIGS:
        e = errs.get(cfg, [])
        if not e:
            continue
        abs_e = [abs(x[0]) for x in e]
        sd = np.std([x[3] for x in e], ddof=1) * 100 if len(e) > 1 else float("nan")
        lines.append(f"| {cfg} | {len(e)} of {len(SCENES)} | {np.mean(abs_e):.1f} | {max(abs_e):.1f} | "
                     f"{sd:.1f} | {np.mean([x[1] for x in e]):.2f} |")
        if cfg != "original":  # rho judges the current confidence model, not the old code's
            all_pairs += [(x[1], abs(x[0])) for x in e]
    if len(all_pairs) > 3:
        rho, p = spearmanr([c for c, _ in all_pairs], [a for _, a in all_pairs])
        lines += ["", f"Confidence vs |error| over the {len(all_pairs)} measured widths of the current code: Spearman ρ = {rho:.2f} "
                      f"(p = {p:.2f}); negative means higher confidence goes with smaller error."]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--scenes", default=",".join(SCENES))
    a = ap.parse_args()
    res = evaluate(a.configs.split(","), a.scenes.split(","))
    md = table(res)
    (Path(__file__).parent / "results.md").write_text("# Results on public data (ARKitScenes visit 483621)\n\n"
                                                      "Generated by `eval/run_public.py`.\n\n" + md + "\n")
    print(md)
