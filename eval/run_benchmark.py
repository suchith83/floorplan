"""One command for every reported number: gates per tier, repeatability, cross-tier, timing, interval calibration.

    uv run python eval/run_benchmark.py --all      # (= make benchmark) run every capture, fit, rerun, write tables
    uv run python eval/run_benchmark.py            # reuse plans already in out/, refit, rewrite the tables

Writes eval/BENCHMARK.md, eval/benchmark.json, eval/CALIBRATION.md and fp/calibration.json.

What counts as truth (said in every table, never blurred):
 - TAPE: eval/ground_truth/<capture>.yaml (the user's own captures, data/own/). None exists yet; every row that
   needs it is NOT SCORED, with the slot named.
 - LiDAR REFERENCE: for the sample data the LiDAR-tier plan stands in for truth when scoring the camera tiers
   (derived from the same captures, data/derived/). It has its own errors (drift, room splits), so a camera-tier
   error here is "distance from the LiDAR plan", not from the truth.
 - REPEAT: the two whole-flat sample scans (1a8384c3f6, c7d28f72c6) are the same flat; their disagreement is
   precision, never bias.
 - BY EYE: openings in c7d28f72c6 were judged in RGB frames (eval/ground_truth/c7d28f72c6.openings_by_eye.yaml).

Steps with --all: run fp on every capture (model outputs replay from out/_cache) -> fit the interval constants
on the evidence -> if they changed, run fp again so every plan carries the fitted intervals -> score.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "eval"), str(ROOT / "scripts")]

import cross_tier  # noqa: E402
import gt as gtlib  # noqa: E402
import repeatability  # noqa: E402

from fp import calibration as cal  # noqa: E402

OUT = ROOT / "out"
GT_DIR = ROOT / "eval" / "ground_truth"
OWN = ROOT / "data" / "own"

# --- the brief's gates (docs/brief/applied_ai_brief.md, Part 2) ---
OPENING_TOL_M, OPENING_FRAC = 0.02, 0.85   # width within 2 cm on >= 85 % of openings (misses + phantoms count)
CEILING_TOL_M, CEILING_SPREAD_M = 0.015, 0.01
REPEAT_ABS_M, REPEAT_REL = 0.01, 0.005     # two captures agree within 1 cm or 0.5 % per wall
WALL_REL = {"video": 0.03, "photos": 0.08}  # wall lengths within +-3 % / +-8 %
FOOTPRINT_REL = 0.08                        # photo stitch: footprint within +-8 %
# A "within X" gate on many walls passes when >= 90 % of them are within X: the same 9-in-10 our intervals state.
WALL_FRAC = 0.9
OPENING_MATCH_M = 0.6                      # a plan opening matches a by-eye label within 0.6 m of its centre
MIN_FIT_N = 3                              # fewer evidence rows than this: the constant stays provisional

SAMPLE = [  # name, tier, input, extra args. LiDAR first: the camera tiers are scored against it.
    ("c00a170fe1", "lidar", "data/stray/c00a170fe1", []),
    ("1a8384c3f6", "lidar", "data/stray/1a8384c3f6", []),
    ("c7d28f72c6", "lidar", "data/stray/c7d28f72c6", []),
    ("c00a170fe1-video", "video", "data/derived/c00a170fe1/video/c00a170fe1.mp4", []),
    ("c00a170fe1-photos", "photos", "data/derived/c00a170fe1/photos", []),
    ("c7d28f72c6-video", "video", "data/derived/c7d28f72c6/video/c7d28f72c6.mp4", []),
    ("c7d28f72c6-photos", "photos", "data/derived/c7d28f72c6/photos", []),
]
REPEAT_PAIR = ("1a8384c3f6", "c7d28f72c6")


def own_captures() -> list[tuple]:
    """The user's own captures (CHECKLIST Phase 2), when present: photos_run1/2, video_run1, video_room_repeat."""
    out = []
    for name, tier in (("photos_run1", "photos"), ("photos_run2", "photos")):
        if (OWN / name).is_dir():
            out.append((f"own-{name}", tier, str((OWN / name).relative_to(ROOT)), []))
    for stem in ("video_run1", "video_room_repeat"):
        for ext in (".mp4", ".mov", ".MOV", ".MP4"):
            if (OWN / (stem + ext)).is_file():
                out.append((f"own-{stem}", "video", str((OWN / (stem + ext)).relative_to(ROOT)), []))
                break
    return out


# ---------------------------------------------------------------- running

def run_fp(name, tier, inp, args, log) -> dict:
    out = OUT / name
    cmd = [sys.executable, "-c", "from fp.cli import main; main()", "run", inp, "--out", str(out), "--tier", tier, *args]
    t0 = time.perf_counter()
    with open(OUT / "logs" / f"bench_{name}.log", "w") as fh:
        rc = subprocess.run(cmd, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT).returncode
    s = time.perf_counter() - t0
    log(f"  {name}: exit {rc}, {s:.0f} s")
    return {"exit": rc, "wall_s": round(s, 1)}


def load_plan(name) -> dict | None:
    p = OUT / name / "plan.json"
    return json.loads(p.read_text()) if p.is_file() else None


# ---------------------------------------------------------------- evidence for calibration

def half(m) -> float:
    return m["hi"] - m["value"]          # lo is clipped at 0, so the upper side is the true half-width


def evidence_a(plan, typ, m) -> float:
    """Recover a Measure's evidence term a from its interval and the k, b the plan was made with."""
    used = plan["source"].get("interval_model", {}).get(typ) or cal.provisional()[plan["tier"]][typ]
    extra = ((plan["source"].get("recon") or {}).get("scale_check") or {}).get("extra_rel", 0.0)
    mult = 2 if typ in cal.AREA_TYPES else 1
    return max(half(m) - (used["b"] + mult * extra) * abs(m["value"]), 0.0) / used["k"]


def _extra(plan):
    return ((plan["source"].get("recon") or {}).get("scale_check") or {}).get("extra_rel", 0.0)


def lidar_repeat_rows(A, B, rep) -> tuple[list, list]:
    """Calibration rows from the repeat pair. Fit set: wall-to-wall spans (what a tape measures: the distance
    between two planes both scans found) and matched room areas. Stress set (coverage only, not fitted): the
    per-wall polygon edges, which also change when the scans split rooms at different doorways."""
    posA, posB = A["source"].get("wall_plane_pos", {}), B["source"].get("wall_plane_pos", {})
    R, t = rep["_R"], rep["_t"]
    fit, stress = [], []
    rooms = sorted(rep["rooms"], key=lambda m: m["a"])
    ra = {r["id"]: r for r in A["rooms"]}
    rb = {r["id"]: r for r in B["rooms"]}
    for m in rooms:
        Wa = repeatability._walls(ra[m["a"]], np.eye(2), np.zeros(2))
        Wb = repeatability._walls(rb[m["b"]], R, t)
        pairs = []
        for w in Wa:
            if not w["observed"]:
                continue
            cand = [v for v in Wb if v["axis"] == w["axis"] and v["observed"] and abs(v["c"] - w["c"]) <= repeatability.WALL_MATCH_M]
            if cand:
                pairs.append((w, min(cand, key=lambda v: abs(v["c"] - w["c"]))))
        seen = set()
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                (wa1, wb1), (wa2, wb2) = pairs[i], pairs[j]
                if wa1["axis"] != wa2["axis"] or wb1["id"] == wb2["id"]:
                    continue
                sa, sb = abs(wa2["c"] - wa1["c"]), abs(wb2["c"] - wb1["c"])
                key = (wa1["axis"], round(min(wa1["c"], wa2["c"]), 3), round(max(wa1["c"], wa2["c"]), 3))
                if sa < repeatability.MIN_SPAN or key in seen:
                    continue
                seen.add(key)
                aA = posA.get(wa1["id"], 0) + posA.get(wa2["id"], 0)
                aB = posB.get(wb1["id"], 0) + posB.get(wb2["id"], 0)
                if not (aA and aB):
                    continue
                # the whole disagreement is charged to each scan (conservative: no /sqrt(2))
                fit.append({"tier": "lidar", "type": "wall_length", "value": sa, "err": sb - sa, "a": 0.5 * (aA + aB),
                            "fold": None, "room": m["a"], "what": f"span {m['a']}/{m['b']} {wa1['id']}-{wa2['id']}"})
        fa, fb = ra[m["a"]]["floor_area"], rb[m["b"]]["floor_area"]
        fit.append({"tier": "lidar", "type": "floor_area", "value": fa["value"], "err": fb["value"] - fa["value"],
                    "a": 0.5 * (evidence_a(A, "floor_area", fa) + evidence_a(B, "floor_area", fb)), "fold": None, "room": m["a"],
                    "what": f"area {m['a']}/{m['b']}"})
    # two folds by room (a wall plane must never be in both folds, or held-out spans share errors with fitted ones),
    # balanced greedily: the room with most span rows first, each into the fold that has fewer so far
    per_room = {}
    for r in fit:
        per_room.setdefault(r["room"], []).append(r)
    sizes = {"rooms A": 0, "rooms B": 0}
    for room, rs in sorted(per_room.items(), key=lambda kv: -sum(x["type"] == "wall_length" for x in kv[1])):
        f = min(sizes, key=sizes.get)
        sizes[f] += sum(x["type"] == "wall_length" for x in rs) or 1
        for x in rs:
            x["fold"] = f
    WA = {w["id"]: w for r in A["rooms"] for w in r["walls"]}
    WB = {w["id"]: w for r in B["rooms"] for w in r["walls"]}
    for w in rep["walls"]:
        la, lb = WA[w["a"]]["length"], WB[w["b"]]["length"]
        stress.append({"tier": "lidar", "type": "wall_length", "value": la["value"], "err": lb["value"] - la["value"],
                       "a": 0.5 * (evidence_a(A, "wall_length", la) + evidence_a(B, "wall_length", lb)),
                       "fold": "edges", "what": f"edge {w['a']}/{w['b']}"})
    return fit, stress


def camera_rows(ref, test, cmp, fold) -> list:
    """Calibration rows: a camera-tier plan against the LiDAR reference of the same capture."""
    rows, tier, ex = [], test["tier"], _extra(test)
    T = {r["id"]: r for r in test["rooms"]}
    TW = {w["id"]: w for r in test["rooms"] for w in r["walls"]}
    for r in cmp["rooms"]:
        tr = T[r["test"]]
        for typ, key in (("floor_area", "floor_area"), ("perimeter", "perimeter"), ("ceiling", "ceiling_height")):
            d, m = r[key], tr[key]
            if d["ref"] is None or m["value"] is None:
                continue
            rows.append({"tier": tier, "type": typ, "value": m["value"], "err": m["value"] - d["ref"],
                         "a": evidence_a(test, typ, m), "extra_rel": ex, "fold": fold, "what": f"{r['ref']}/{r['test']} {key}"})
        for w in r["walls"]:
            m = TW[w["test"]]["length"]
            rows.append({"tier": tier, "type": "wall_length", "value": m["value"], "err": m["value"] - w["ref_len"],
                         "a": evidence_a(test, "wall_length", m), "extra_rel": ex, "fold": fold,
                         "what": f"{w['ref']}/{w['test']}"})
    return rows


def gt_rows(plan, scored, fold) -> list:
    """Calibration rows from tape ground truth (own captures), when eval/gt.py scored them."""
    rows, tier, ex = [], plan["tier"], _extra(plan)
    meas = {w["id"]: w["length"] for r in plan["rooms"] for w in r["walls"]}
    for r in plan["rooms"]:
        meas[f"{r['id']}.ceiling"] = r["ceiling_height"]
    for row in scored.get("walls", []):
        m = meas.get(row.get("plan"))
        if m and m["value"] is not None and row.get("gt") is not None:
            rows.append({"tier": tier, "type": "wall_length", "value": m["value"], "err": m["value"] - row["gt"],
                         "a": evidence_a(plan, "wall_length", m), "extra_rel": ex, "fold": fold, "what": row.get("plan")})
    return rows


def gt_file(plan) -> Path | None:
    """Tape GT for an own capture: eval/ground_truth/<capture_id>.yaml, else the shared eval/ground_truth/own.yaml
    (all own captures are of the same rooms)."""
    for p in (GT_DIR / f"{plan['capture_id']}.yaml", GT_DIR / "own.yaml"):
        if p.is_file():
            return p
    return None


def score_own(plans, names) -> tuple[list, list, dict]:
    """Gates + calibration rows for the user's own captures against tape."""
    G, rows, scored = [], [], {}
    for n in names:
        plan, p = plans[n], gt_file(plans[n])
        tier = plan["tier"]
        if p is None:
            G.append(gate("Wall lengths vs tape", tier, n, "—", "tier gate", "NOT SCORED", "tape",
                          f"no eval/ground_truth/{plan['capture_id']}.yaml or own.yaml"))
            continue
        g = gtlib.load_gt(p)
        s = gtlib.score(plan, g, gtlib.match(plan, g, gtlib.load_overrides(GT_DIR / f"{g['capture']}.match.yaml")))
        scored[n] = s
        sm = s["summary"]
        w = sm["walls"]
        if w["n"]:
            rel = WALL_REL.get(tier)
            frac = w["frac_within_3pct"] if tier == "video" else w["frac_within_8pct"] if tier == "photos" else None
            if rel:
                G.append(gate(f"Wall lengths within ±{100 * rel:.0f} % vs tape", tier, n,
                              f"{100 * frac:.0f} % of {w['n']} within; p90 |err| {w['p90_abs_err_m']} m", f"≥ {100 * WALL_FRAC:.0f} %",
                              "PASS" if frac >= WALL_FRAC else "FAIL", "tape", f"interval coverage {w['coverage']}",
                              ratio=WALL_FRAC / max(frac, 1e-3)))
        c = sm["ceilings"]
        if c["n"]:
            G.append(gate("Ceiling ≤ 1.5 cm per room vs tape", tier, n, f"max |err| {c['max_abs_err_m']} m over {c['n']} rooms",
                          "≤ 1.5 cm", "PASS" if c["pass"] else "FAIL", "tape", "",
                          ratio=(c["max_abs_err_m"] or 9) / CEILING_TOL_M))
        o = sm["openings"]
        if o["n_gt"]:
            G.append(gate("Openings: width ≤ 2 cm on ≥ 85 % vs tape", tier, n,
                          f"{o['n_width_ok']} ok of {o['n_gt']} real + {o['n_phantom']} phantom ({100 * o['frac_ok']:.0f} %)",
                          "≥ 85 %", "PASS" if o["pass"] else "FAIL", "tape", f"{o['n_missed']} missed",
                          ratio=OPENING_FRAC / max(o["frac_ok"], 1e-3)))
        rows += gt_rows(plan, s, fold=n)
    return G, rows, scored


# ---------------------------------------------------------------- gates

def gate(name, tier, capture, number, threshold, status, truth, note="", ratio=None) -> dict:
    """ratio: how far from the threshold, in threshold units (> 1 = fails by that factor); ranks the worst gate."""
    return {"gate": name, "tier": tier, "capture": capture, "number": number, "threshold": threshold,
            "status": status, "truth": truth, "note": note, "ratio": None if ratio is None else round(float(ratio), 2)}


def openings_by_eye(plan, labels) -> dict:
    found = [o for r in plan["rooms"] for o in r["openings"]]
    used, rows = set(), []
    for lab in labels["openings"]:
        c = np.array(lab["center"], float)
        best = min(((float(np.linalg.norm(np.array(o["center"][:2]) - c)), o) for o in found if o["id"] not in used),
                   default=(None, None), key=lambda x: x[0])
        if best[1] is not None and best[0] <= OPENING_MATCH_M:
            used.add(best[1]["id"])
            ok = best[1]["kind"] == lab["kind"]
            rows.append({"label": lab["name"], "kind": lab["kind"], "plan": best[1]["id"], "plan_kind": best[1]["kind"],
                         "verdict": "correct" if ok else "wrong kind",
                         "width": best[1]["width"]["value"] if best[1]["width"] else None})
        else:
            rows.append({"label": lab["name"], "kind": lab["kind"], "plan": None, "verdict": "missed"})
    phantoms = [{"plan": o["id"], "plan_kind": o["kind"], "verdict": "phantom"} for o in found if o["id"] not in used]
    n_ok = sum(r["verdict"] == "correct" for r in rows)
    denom = len(rows) + len(phantoms)
    return {"rows": rows, "phantoms": phantoms, "n_correct": n_ok, "n_real": len(rows), "n_phantom": len(phantoms),
            "frac": n_ok / denom if denom else None}


def _union_area(polys):
    from shapely.ops import unary_union
    return unary_union(polys).area if polys else 0.0


def _to_plan(poly_xy, T_from, T_to):
    """2-D floor polygon in one plan frame -> another, through world coordinates (both 4x4 plan<-world)."""
    P = np.c_[np.asarray(poly_xy, float), np.zeros(len(poly_xy)), np.ones(len(poly_xy))]
    return (np.asarray(T_to) @ np.linalg.inv(np.asarray(T_from)) @ P.T).T[:, :2]


def photo_stitch(test, selection, lidar) -> dict:
    """Photo tier vs the rooms its folders were taken in (data/derived/<id>/selection.json). Those polygons come from an
    older LiDAR plan, so they are moved into the CURRENT LiDAR plan's frame and clipped to its footprint: the reference
    area is the current LiDAR floor inside the photographed rooms."""
    from shapely.ops import unary_union
    folders = sorted({p.name for p in (ROOT / "data" / "derived" / selection["capture"] / "photos").iterdir() if p.is_dir()})
    ref_rooms = {r["id"]: r for r in selection["rooms"] if r["id"] in folders}
    U = unary_union([Polygon(r["polygon"]).buffer(0) for r in lidar["rooms"]])
    moved = [Polygon(_to_plan(r["polygon"], selection["T_plan_world"], lidar["frame"]["T_plan_world"])).buffer(0)
             for r in ref_rooms.values()]
    ref_area = unary_union(moved).intersection(U).area
    ref_adj = {tuple(sorted(c["rooms"])) for c in selection["connections"] if all(x in ref_rooms for x in c["rooms"])}
    name = {r["id"]: r["name"] for r in test["rooms"]}
    test_adj = {tuple(sorted((name[a], name[b]))) for a, b in (c["rooms"] for c in test["connections"]) if name[a] != name[b]}
    names = [r["name"] for r in test["rooms"]]
    polys = [Polygon(r["polygon"]).buffer(0) for r in test["rooms"]]
    overlap = sum(polys[i].intersection(polys[j]).area for i in range(len(polys)) for j in range(i + 1, len(polys)))
    fp = test["footprint_area"]
    return {"folders": folders, "ref_area": round(ref_area, 2), "test_area": fp["value"], "test_lo": fp["lo"], "test_hi": fp["hi"],
            "rel_err": None if fp["value"] is None else round((fp["value"] - ref_area) / ref_area, 4),
            "ref_in_interval": None if fp["value"] is None else bool(fp["lo"] <= ref_area <= fp["hi"]),
            "ref_adjacency": sorted(ref_adj), "test_adjacency": sorted(test_adj),
            "rooms_named": names, "duplicate_names": len(set(names)) != len(names),
            "missing_folders": sorted(set(folders) - set(names)),
            "adjacency_correct": test_adj == ref_adj and len(set(names)) == len(names) and set(folders) <= set(names),
            "overlap_m2": round(overlap, 3)}


def score(plans, cmps, rep, stress_cov, cal_info) -> list:
    G = []
    # -- LiDAR tier
    for name in ("c00a170fe1", "1a8384c3f6", "c7d28f72c6"):
        G.append(gate("Wall length vs tape", "lidar", name, "—", "brief: reference tier", "NOT SCORED",
                      "tape", "no tape on the evaluator's sample data; slot: eval/ground_truth/<capture>.yaml"))
    lab = yaml.safe_load((GT_DIR / "c7d28f72c6.openings_by_eye.yaml").read_text())
    ob = openings_by_eye(plans["c7d28f72c6"], lab)
    G.append(gate("Openings: width ≤ 2 cm on ≥ 85 % (misses, phantoms count)", "lidar", "c7d28f72c6",
                  f"{ob['n_correct']}/{ob['n_real'] + ob['n_phantom']} = {100 * ob['frac']:.0f} % detected right (upper bound: widths not taped)",
                  "≥ 85 %", "PASS" if ob["frac"] >= OPENING_FRAC else "FAIL", "by eye",
                  f"{ob['n_real']} real, {ob['n_phantom']} phantom, "
                  f"{sum(r['verdict'] == 'missed' for r in ob['rows'])} missed, "
                  f"{sum(r['verdict'] == 'wrong kind' for r in ob['rows'])} wrong kind; in-sample (04's rules were written on it)",
                  ratio=OPENING_FRAC / max(ob["frac"], 1e-3)))
    for name in ("c00a170fe1", "1a8384c3f6", "c7d28f72c6"):
        p = plans[name]
        seen = [r for r in p["rooms"] if r["ceiling_height"]["value"] is not None]
        G.append(gate("Ceiling ≤ 1.5 cm per room vs tape", "lidar", name,
                      f"{len(seen)}/{len(p['rooms'])} rooms observed" + (" (none: correctly reported not observed)" if not seen else ""),
                      "≤ 1.5 cm", "NOT SCORED", "tape", "no tape on the sample data"))
    G.append(gate("Ceiling spread across captures ≤ 1 cm", "lidar", "1a8384c3f6 + c7d28f72c6", "—", "≤ 1 cm", "NOT SCORED",
                  "repeat", "1a8384c3f6 has no ceiling in view, so no room has two ceiling measurements; needs own repeat"))
    sp = rep["spans"]
    n_ok = sum(s["pass"] for s in sp)
    med = float(np.median(np.abs([s["diff_cm"] for s in sp]))) if sp else None
    G.append(gate("Repeatability: wall-to-wall spans within 1 cm or 0.5 %", "lidar", " vs ".join(REPEAT_PAIR),
                  f"{n_ok}/{len(sp)} within; median |Δ| {med:.1f} cm", "all spans", "PASS" if sp and n_ok == len(sp) else "FAIL",
                  "repeat", "unrepeatable (no tape, so bias can't be seen); per-wall edges: "
                  f"{sum(w['pass'] for w in rep['walls'])}/{len(rep['walls'])} within",
                  ratio=med / 1.0 if med is not None else None))
    for name in ("1a8384c3f6", "c7d28f72c6"):
        d = plans[name]["drift"]
        m = d.get("metrics", {})
        ok = d.get("enabled") and "footprint_m2_off" in m and (OUT / name / "debug" / "drift_ablation.png").is_file()
        G.append(gate("Drift accountability: ablation on/off present", "lidar", name,
                      f"footprint {m.get('footprint_m2_off')} → {m.get('footprint_m2_on')} m², walls {m.get('thickness_median_cm_off', 0):.1f} → "
                      f"{m.get('thickness_median_cm_on', 0):.1f} cm thick" if ok else "missing",
                      "ablation shown", "PASS" if ok else "FAIL", "—", f"debug/drift_ablation.png; {d.get('method', '')[:80]}"))
    # -- camera tiers vs the LiDAR reference
    for name, c in cmps.items():
        tier = plans[name]["tier"]
        errs = [abs(w["err_m"]) / w["ref_len"] for r in c["rooms"] for w in r["walls"] if w["ref_len"] > 0]
        within = float(np.mean([e <= WALL_REL[tier] for e in errs])) if errs else 0.0
        p90 = float(np.percentile(errs, 90)) if errs else None
        G.append(gate(f"Wall lengths within ±{100 * WALL_REL[tier]:.0f} %", tier, name,
                      f"{100 * within:.0f} % of {len(errs)} matched walls within; median |err| "
                      f"{100 * float(np.median(errs)) if errs else float('nan'):.0f} %, p90 {100 * (p90 or 0):.0f} %",
                      f"≥ {100 * WALL_FRAC:.0f} % of walls (p90 ≤ {100 * WALL_REL[tier]:.0f} %)",
                      "PASS" if errs and within >= WALL_FRAC else "FAIL", "LiDAR reference",
                      f"footprint IoU {c['align']['footprint_iou']:.2f}; rooms {c['rooms_test']} vs {c['rooms_ref']}",
                      ratio=(p90 / WALL_REL[tier]) if p90 is not None else 99))
        fa = c["footprint_area"]
        if tier == "video":
            G.append(gate("Footprint area within ±3 % (video gate applied to area)", tier, name,
                          f"{fa['test']} vs {fa['ref']} m² ({100 * fa['rel_err']:+.0f} %)", "±3 %",
                          "PASS" if abs(fa["rel_err"]) <= 0.03 else "FAIL", "LiDAR reference", "",
                          ratio=abs(fa["rel_err"]) / 0.03))
    for name in ("c00a170fe1-photos", "c7d28f72c6-photos"):
        sel = json.loads((ROOT / "data" / "derived" / name.split("-")[0] / "selection.json").read_text())
        st = photo_stitch(plans[name], sel, plans[name.split("-")[0]])
        multi = len(st["folders"]) > 1
        okfp = st["rel_err"] is not None and abs(st["rel_err"]) <= FOOTPRINT_REL and st["ref_in_interval"]
        status = "PASS" if (okfp and st["overlap_m2"] < 0.01 and (st["adjacency_correct"] or not multi)) else "FAIL"
        G.append(gate("Photo stitch: adjacency, no overlaps, footprint ±8 % with calibrated interval", "photos", name,
                      f"footprint {st['test_area']} vs {st['ref_area']} m² ({100 * st['rel_err']:+.0f} %), interval "
                      f"[{st['test_lo']:.1f}, {st['test_hi']:.1f}] {'covers' if st['ref_in_interval'] else 'misses'} it; overlaps "
                      f"{st['overlap_m2']} m²; adjacency {'correct' if st['adjacency_correct'] else 'wrong'}"
                      + ("" if multi else " (one folder: adjacency n/a)"),
                      "±8 %, 0 m², correct", status, "LiDAR reference",
                      f"folders {st['folders']}, plan rooms named {st['rooms_named']}"
                      + (f"; missing {st['missing_folders']}" if st["missing_folders"] else "")
                      + ("; duplicate room names" if st["duplicate_names"] else ""),
                      ratio=abs(st["rel_err"]) / FOOTPRINT_REL if st["rel_err"] is not None else 99))
    for tier in ("video", "photos"):
        G.append(gate("Repeatability (same room twice, same tier)", tier, "own capture", "—", "1 cm or 0.5 %", "NOT SCORED",
                      "repeat", f"pending own capture: data/own/{'video_room_repeat.mp4' if tier == 'video' else 'photos_run2/'}"))
    # -- calibration: held-out coverage of the stated 90 %
    for tier in ("lidar", "video", "photos"):
        held = [r for r in cal_info["two_fold"] if r["tier"] == tier and r["type"] == "wall_length" and r["coverage"] is not None]
        if held:
            n = sum(r["n_test"] for r in held)
            covw = sum(r["coverage"] * r["n_test"] for r in held) / n
            G.append(gate("Calibration: held-out coverage of stated 90 % (wall lengths)", tier,
                          "2-fold: " + ", ".join(sorted({r["test_on"] for r in held})), f"{100 * covw:.0f} % of {n}",
                          "≥ 90 %", "PASS" if covw >= 0.9 else "FAIL",
                          "repeat" if tier == "lidar" else "LiDAR reference",
                          f"n is small (one miss moves it {100 / n:.0f} points); held-out = fit on the other fold", ratio=0.9 / max(covw, 1e-3)))
    G.append(gate("Calibration: LiDAR per-wall edges between the repeat scans (stress set, not fitted)", "lidar",
                  " vs ".join(REPEAT_PAIR), f"{100 * stress_cov['frac']:.0f} % of {stress_cov['n']}", "≥ 90 %",
                  "PASS" if stress_cov["frac"] >= 0.9 else "FAIL", "repeat",
                  "edges also move when the scans split rooms at different doorways; intervals model plane error, not room splits",
                  ratio=0.9 / max(stress_cov["frac"], 1e-3)))
    G.append(gate("Head-to-head vs consumer app (LiDAR tier)", "lidar", "—", "—", "beat or tie ≥ 70 %", "NOT DONE", "tape",
                  "no iPhone Pro (recruiter-approved); substitute protocol in eval/HEAD_TO_HEAD.md, pending own capture"))
    G.append(gate("Staged damage (2 classes) found on the right surface", "all", "own capture", "—", "both found", "NOT SCORED",
                  "staged", "pending own capture; painted test on c00a: stain found, drawn crack missed (STATUS 06)"))
    return G


# ---------------------------------------------------------------- markdown

def _md_table(rows, cols):
    L = ["| " + " | ".join(h for h, _ in cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        L.append("| " + " | ".join(str(f(r)) for _, f in cols) + " |")
    return L


def bench_md(B) -> str:
    G = B["gates"]
    worst = B["worst_gate"]
    L = ["# Benchmark", "",
         f"Generated by `make benchmark` (`uv run python eval/run_benchmark.py --all`) on {B['generated']}; "
         f"wall clock **{B['wall_clock_s']:.0f} s**{' (runs reused)' if not B['ran'] else ''}. Every number below comes from that command.", "",
         "**What is truth here.** There is no tape on the evaluator's sample data, and the user's own captures "
         "(`data/own/`) don't exist yet. So: camera tiers are scored against the **LiDAR plan of the same capture** "
         "(a reference, not truth), LiDAR is scored by **repeatability** across the two scans of the same flat, and "
         "openings by **eye** in RGB frames. Rows that need tape say NOT SCORED and name the slot.", "",
         f"**Worst gate:** {worst['gate']} ({worst['tier']}, `{worst['capture']}`): {worst['number']} vs {worst['threshold']}, "
         f"{worst['ratio']}× the threshold." if worst else "", "",
         "## Gates", ""]
    L += _md_table(G, [("gate", lambda g: g["gate"]), ("tier", lambda g: g["tier"]), ("capture", lambda g: f"`{g['capture']}`"),
                       ("number", lambda g: g["number"]), ("threshold", lambda g: g["threshold"]),
                       ("result", lambda g: f"**{g['status']}**"), ("vs", lambda g: g["truth"]), ("note", lambda g: g["note"])])
    L += ["", f"Status counts: " + ", ".join(f"{k} {v}" for k, v in B["status_counts"].items()), "",
          "Ranking (failing gates by distance to threshold, × threshold):", ""]
    for g in B["failing_ranked"]:
        L.append(f"- {g['ratio']}× — {g['gate']} ({g['tier']}, `{g['capture']}`): {g['number']}")
    # cross tier
    L += ["", "## Camera tiers vs the LiDAR reference (same capture)", "",
          "Plans aligned by footprint IoU over yaw 0/90/180/270 + translation (no scale fitted: scale error is what we measure); "
          "rooms matched one-to-one by IoU ≥ 0.2; walls by direction and midpoint (`eval/cross_tier.py`). "
          "The camera-tier inputs are derived from the same Stray captures (`data/derived/`, `scripts/make_camera_tiers.py`).", ""]
    rows = []
    for name, c in B["cross_tier"].items():
        s = c["summary"]
        rows.append({"name": name, "iou": c["align"]["footprint_iou"], "rooms": f"{c['rooms_test']} vs {c['rooms_ref']}",
                     "fp": f"{c['footprint_area']['test']} vs {c['footprint_area']['ref']} ({100 * (c['footprint_area']['rel_err'] or 0):+.0f} %)",
                     "walls": s["walls_matched"], "med": s["wall_len_median_abs_err_m"], "p90": s["wall_len_p90_abs_err_m"],
                     "cov": s["ref_in_interval_frac"]})
    L += _md_table(rows, [("plan", lambda r: f"`{r['name']}`"), ("footprint IoU", lambda r: r["iou"]), ("rooms (test vs ref)", lambda r: r["rooms"]),
                          ("footprint m² (test vs ref)", lambda r: r["fp"]), ("walls matched", lambda r: r["walls"]),
                          ("median |err| m", lambda r: r["med"]), ("p90 |err| m", lambda r: r["p90"]),
                          ("ref inside interval", lambda r: r["cov"])])
    # repeatability
    rp = B["repeatability"]
    L += ["", "## Repeatability", "",
          f"LiDAR, `{rp['a']}` vs `{rp['b']}` (same flat: registration yaw {rp['registration']['yaw_deg']:.1f}°, "
          f"{100 * rp['registration']['inliers']:.0f} % wall inliers). Full tables: `eval/repeatability_lidar.md`.", "",
          "| measure | n | within gate | median |Δ| | p90 |Δ| | which failure |", "|---|---|---|---|---|---|"]
    for k, lab in (("spans", "wall-to-wall spans"), ("walls", "per-wall edges")):
        d = np.abs([x["diff_cm"] for x in rp[k]])
        L.append(f"| {lab} | {len(d)} | {sum(x['pass'] for x in rp[k])} | {np.median(d):.1f} cm | {np.percentile(d, 90):.1f} cm | "
                 "unrepeatable (spread, not a constant offset: Δ has both signs" + f", mean {np.mean([x['diff_cm'] for x in rp[k]]):+.1f} cm) |")
    L += ["| ceiling | 0 | — | — | — | not scorable: 1a8384c3f6 never saw the ceiling |",
          "| own repeat, video / photos | — | — | — | — | pending own capture (`data/own/video_room_repeat.mp4`, `photos_run2/`) |", "",
          "Bias (a repeatable offset from the truth) can only be seen against tape; the sample data has none."]
    # openings
    ob = B["openings_by_eye"]
    L += ["", "## Openings, c7d28f72c6 (detection by eye; widths not taped)", ""]
    L += _md_table(ob["rows"] + ob["phantoms"], [("real opening (by eye)", lambda r: r.get("label", "—")), ("kind", lambda r: r.get("kind", "—")),
                                                 ("plan", lambda r: r.get("plan") or "—"), ("plan kind", lambda r: r.get("plan_kind", "—")),
                                                 ("verdict", lambda r: r["verdict"]), ("width m", lambda r: r.get("width") or "—")])
    # timing
    L += ["", "## Timing per stage (s)", "",
          "From each plan's `timings` on this run (MapAnything outputs replayed from `out/_cache/recon`; a cold camera-tier "
          "run adds the model passes: c00a video 338 s, c7d2 video 511 s live, STATUS 05). `wall` = the whole `fp run` process."]
    stages = sorted({k for t in B["timings"].values() for k in t["stages"]} - {"total"})
    L += [""] + _md_table([{"name": k, **v} for k, v in B["timings"].items()],
                          [("capture", lambda r: f"`{r['name']}`")] + [(s, (lambda s: lambda r: r["stages"].get(s, ""))(s)) for s in stages]
                          + [("total", lambda r: r["stages"].get("total", "")), ("wall", lambda r: r.get("wall_s") or "—")])
    L += ["", "## Own captures", "", "None yet (`data/own/` missing). When they exist, `make benchmark` picks up "
          "`photos_run1/`, `photos_run2/`, `video_run1.mp4`, `video_room_repeat.mp4`, scores them against "
          "`eval/ground_truth/<capture>.yaml` (format: `eval/ground_truth/TEMPLATE.yaml`; matching: `eval/match_gt.py`)."
          if not B["own"] else "Own captures: " + ", ".join(B["own"])]
    return "\n".join(L) + "\n"


def cal_md(C) -> str:
    consts = C["constants"]
    L = ["# Interval calibration", "",
         f"Generated by `make benchmark` on {C['generated']}. Constants: `fp/calibration.json` (read by `fp run`). Model and fit: `fp/calibration.py`.", "",
         "## What a 90 % interval means here", "",
         "Every number `value [lo, hi]` claims: across many measurements like this one, at least 9 in 10 true values fall inside. "
         "One model for every Measure: `half = k·a + b·value`. `a` is the measurement's own evidence (plane fit + measured wall-band "
         "smear, ×3 for a wall never seen; jamb reveals for openings; ceiling patch spread), `b·value` is the relative (scale) term. "
         "k and b are fitted per tier and measurement type.", "",
         "## Evidence used (no tape yet: n is small and the truth is a stand-in)", "",
         "| tier | type | evidence | n | folds |", "|---|---|---|---|---|"]
    for (t, ty), d in sorted(C["evidence_summary"].items(), key=lambda x: x[0]):
        L.append(f"| {t} | {ty} | {d['source']} | {d['n']} | {', '.join(d['folds'])} |")
    L += ["", "- **LiDAR**: the two scans of the same flat. A wall length is the distance between two planes, so the fit uses "
          "wall-to-wall spans found in both scans, and the **whole** disagreement is charged to each scan (no ÷√2: conservative). "
          "This measures precision only; a bias shared by both scans is invisible without tape.",
          "- **Camera tiers**: the video/photo plans against the LiDAR plan of the same capture. The reference's own error is "
          "charged to the camera tier (conservative).",
          f"- **Rules**: split conformal (the ⌈(n+1)·0.9⌉-th smallest needed factor); n < 9 → the largest (flagged); "
          f"n < {MIN_FIT_N} → stays provisional; a fitted constant is never below its provisional value (no tape, so no narrowing). "
          "LiDAR fits k (metric scale, b = 0); camera tiers fit b with k = 1.",
          "- **Known optimism**: LiDAR spans only include wall planes the two scans matched within 10 cm, so a larger "
          "disagreement never enters the fit; camera-tier constants are fitted on the same two captures the gates score, "
          "so any in-sample coverage (e.g. BENCHMARK's 'ref inside interval') is 100 % by construction: read the held-out columns.",
          "- **Incoherence from small n**: video floor-area b (from 3 rooms) is below video wall b (from 9 walls), "
          "although an area error should be at least the length error; refit with tape.", "",
          "## Fitted constants", "", "| tier | type | k | b | n | how |", "|---|---|---|---|---|---|"]
    for t in ("lidar", "video", "photos"):
        for ty in cal.TYPES:
            c = consts[t][ty]
            how = ("provisional (no evidence)" if not c.get("fitted") else
                   f"from wall_length" if c.get("from") else
                   ("largest of n < 9" if c.get("small_n") else "conformal rank") + (f"; raw {c['raw']} floored to provisional" if c.get("floored") else ""))
            L.append(f"| {t} | {ty} | {c['k']:.3g} | {c['b']:.3g} | {c.get('n', 0)} | {how} |")
    L += ["", "## Coverage: stated 90 % vs observed", "",
          "Held-out: fit on one fold, count how often the other fold's reference value falls inside the interval.", "",
          "| tier | type | fit on | test on | n fit | n test | constant | coverage (fitted) | coverage (provisional) |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in C["two_fold"]:
        if r["coverage"] is None:
            L.append(f"| {r['tier']} | {r['type']} | {r['fit_on']} | — | {r['n_fit']} | 0 | — | — (one fold) | — |")
        else:
            L.append(f"| {r['tier']} | {r['type']} | {r['fit_on']} | {r['test_on']} | {r['n_fit']} | {r['n_test']} | "
                     f"{r['constant']:.3g} | {100 * r['coverage']:.0f} % | {100 * r['coverage_provisional']:.0f} % |")
    L += ["", "In-sample (all evidence, final constants) and the stress set:", "", "| set | n | covered | coverage |", "|---|---|---|---|"]
    for k, v in C["in_sample"].items():
        L.append(f"| {k} | {v['n']} | {v['covered']} | {100 * v['frac']:.0f} % |" if v["n"] else f"| {k} | 0 | — | — |")
    L += ["", "## What this does and doesn't show", "",
          "- Camera-tier intervals become very wide (b is a fraction of the value, not a few %). That is the honest result: on "
          "these derived inputs the camera tiers are far from the LiDAR plan, and a narrow interval there would be confident garbage.",
          "- LiDAR intervals cover plane-to-plane disagreement between two scans. They do **not** cover a room boundary placed at a "
          "different doorway (the per-wall stress set): that is a segmentation failure, reported in BENCHMARK.md, not noise.",
          "- Not calibrated (no evidence): LiDAR ceilings (only one scan sees a ceiling), openings at every tier (no tape on widths), "
          "damage extents (own model, D06.2). Their intervals stay provisional and the plan's `intervals.method` says so.",
          "- With tape (own captures) the same code adds `eval/ground_truth/*.yaml` rows and refits; tape then replaces the stand-ins.", ""]
    return "\n".join(L)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--all", action="store_true", help="run fp on every capture (replaying model caches), fit, rerun, score")
    ap.add_argument("--no-fit", action="store_true", help="keep fp/calibration.json as it is")
    a = ap.parse_args()
    t0 = time.perf_counter()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    log = lambda s: print(s, flush=True)
    caps = SAMPLE + own_captures()
    runs = {}
    if a.all:
        log("run 1: every capture")
        runs = {n: run_fp(n, t, i, x, log) for n, t, i, x in caps}
    plans = {n: load_plan(n) for n, *_ in caps}
    missing = [n for n, p in plans.items() if p is None]
    if missing:
        sys.exit(f"no plan.json for {missing}: run with --all")

    def evidence():
        A, Bp = plans[REPEAT_PAIR[0]], plans[REPEAT_PAIR[1]]
        rep = repeatability.compare(OUT / REPEAT_PAIR[0], OUT / REPEAT_PAIR[1])
        from fp.geometry.register2d import register
        reg = register(np.load(OUT / REPEAT_PAIR[0] / "debug" / "wall_points_2d.npy").astype(float),
                       np.load(OUT / REPEAT_PAIR[1] / "debug" / "wall_points_2d.npy").astype(float))
        rep["_R"], rep["_t"] = reg["R"], reg["t"]
        fit_rows, stress = lidar_repeat_rows(A, Bp, rep)
        cmps = {}
        for n, t, *_ in SAMPLE:
            if t != "lidar":
                ref = plans[n.split("-")[0]]
                cmps[n] = cross_tier.compare(ref, plans[n])
                cmps[n]["summary"] = cross_tier.summary(cmps[n])
                fit_rows += camera_rows(ref, plans[n], cmps[n], fold=n.split("-")[0])
        own_g, own_rows, own_scored = score_own(plans, [n for n, *_ in own_captures()])
        return rep, cmps, fit_rows + own_rows, stress, own_g, own_scored

    rep, cmps, rows, stress, own_g, own_scored = evidence()
    consts = cal.fit([r for r in rows if _enough(rows, r)])
    if not a.no_fit:
        old = cal.load(cal.PATH)["constants"] if cal.PATH.is_file() else None
        cal.PATH.write_text(json.dumps({"source": "sample-data stand-ins (no tape): LiDAR repeat pair "
                                        f"{' vs '.join(REPEAT_PAIR)}; camera tiers vs the LiDAR plan of the same capture "
                                        "(eval/CALIBRATION.md)", "generated": dt.date.today().isoformat(),
                                        "constants": consts}, indent=1) + "\n")
        cal.reset_cache()
        changed = old is None or _differs(old, consts)
        if a.all and changed:
            log("constants changed: run 2, so every plan carries the fitted intervals")
            runs = {n: run_fp(n, t, i, x, log) for n, t, i, x in caps}
            plans.update({n: load_plan(n) for n, *_ in caps})
            rep, cmps, rows, stress, own_g, own_scored = evidence()
    final = cal.load(cal.PATH)["constants"] if cal.PATH.is_file() else cal.provisional()
    two = cal.two_fold([r for r in rows if _enough(rows, r)])
    stress_cov = cal.coverage(stress, final)
    by_type = {}
    for r in rows:
        by_type.setdefault((r["tier"], r["type"]), []).append(r)
    cal_info = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "constants": final, "two_fold": two,
                "evidence_summary": {k: {"n": len(v), "folds": sorted({r["fold"] for r in v}),
                                         "source": "repeat pair" if k[0] == "lidar" else "vs LiDAR reference"} for k, v in by_type.items()},
                "in_sample": {f"{t} {ty}": cal.coverage(v, final) for (t, ty), v in sorted(by_type.items())}
                | {"lidar per-wall edges (stress, not fitted)": stress_cov},
                "rows": rows, "stress_rows": stress}
    G = score(plans, cmps, rep, stress_cov, cal_info) + own_g
    failing = sorted([g for g in G if g["status"] == "FAIL" and g["ratio"] is not None], key=lambda g: -g["ratio"])
    counts = {s: sum(g["status"] == s for g in G) for s in ("PASS", "FAIL", "NOT SCORED", "NOT DONE")}
    lab = yaml.safe_load((GT_DIR / "c7d28f72c6.openings_by_eye.yaml").read_text())
    B = {"generated": dt.datetime.now().isoformat(timespec="minutes"), "ran": a.all,
         "wall_clock_s": round(time.perf_counter() - t0, 1), "gates": G, "worst_gate": failing[0] if failing else None,
         "failing_ranked": failing, "status_counts": counts,
         "cross_tier": {k: {kk: vv for kk, vv in v.items()} for k, v in cmps.items()},
         "repeatability": {k: v for k, v in rep.items() if not k.startswith("_")},
         "openings_by_eye": openings_by_eye(plans["c7d28f72c6"], lab),
         "timings": {n: {"stages": (plans[n] or {}).get("timings", {}), "wall_s": runs.get(n, {}).get("wall_s")} for n, *_ in caps},
         "own": [n for n, *_ in own_captures()], "own_scored": own_scored,
         "calibration": {**{k: v for k, v in cal_info.items() if k not in ("rows", "stress_rows", "evidence_summary")},
                         "evidence_summary": {f"{t}/{ty}": v for (t, ty), v in cal_info["evidence_summary"].items()}}}
    (ROOT / "eval" / "BENCHMARK.md").write_text(bench_md(B))
    (ROOT / "eval" / "CALIBRATION.md").write_text(cal_md(cal_info))
    (ROOT / "eval" / "benchmark.json").write_text(json.dumps({**B, "calibration_rows": rows, "stress_rows": stress},
                                                             indent=1, default=_json) + "\n")
    log(f"wrote eval/BENCHMARK.md, eval/CALIBRATION.md, eval/benchmark.json in {B['wall_clock_s']:.0f} s; "
        f"worst gate: {B['worst_gate']['gate'] if B['worst_gate'] else None}")


def _enough(rows, r) -> bool:
    return sum(x["tier"] == r["tier"] and x["type"] == r["type"] for x in rows) >= MIN_FIT_N


def _differs(old, new) -> bool:
    return any(abs(old[t][ty]["k"] - new[t][ty]["k"]) > 1e-6 or abs(old[t][ty]["b"] - new[t][ty]["b"]) > 1e-6
               for t in new for ty in new[t])


def _json(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
