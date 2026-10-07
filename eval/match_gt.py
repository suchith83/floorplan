"""Show how a ground-truth file matches a plan.json, and optionally write the proposal as an overrides file.

    uv run python eval/match_gt.py out/<capture>/plan.json eval/ground_truth/<capture>.yaml [--write-overrides]

Prints every GT room / wall / opening with the plan ID it was matched to and `how`
(override | name | dir | auto-length | wall+width | unmatched). Lines marked CHECK are guesses from length
alone: confirm them on plan.svg. With --write-overrides the proposal is written to
eval/ground_truth/<capture>.match.yaml (entries already there are kept), so you only edit the wrong lines.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gt import GTError, load_gt, load_overrides, match, outward_dir  # noqa: E402

GT_DIR = Path(__file__).resolve().parent / "ground_truth"


def _f(v, nd=3):
    return "-" if v is None else f"{v:.{nd}f}"


def report(plan: dict, gt: dict, m: dict) -> list[str]:
    rooms = {r["id"]: r for r in plan["rooms"]}
    walls = {w["id"]: (r, w) for r in plan["rooms"] for w in r["walls"]}
    ops = {o["id"]: o for r in plan["rooms"] for o in r["openings"]}
    L = [f"GT {gt['capture']} ({gt['tool']})  vs  plan {plan.get('capture_id')} [{plan.get('tier')}]", ""]
    L.append(f"{'ROOM':28s} {'plan':6s} how")
    for g in gt["rooms"]:
        x = m["rooms"][g["name"]]
        L.append(f"{g['name']:28s} {x['plan'] or '-':6s} {x['how']}" + (f"   <- {x['note']}" if x.get("note") else ""))
    L += ["", f"{'WALL (gt room/wall)':34s} {'gt m':>7s} {'dir':>3s}  {'plan':8s} {'plan m':>7s} {'faces':>5s}  how"]
    for g in gt["rooms"]:
        for w in g["walls"]:
            key = f"{g['name']}/{w['name']}"
            x = m["walls"][key]
            pv, face = None, None
            if x["plan"]:
                r, pw = walls[x["plan"]]
                pv, face = pw["length"]["value"], outward_dir(r, pw)
            flag = "   CHECK (length-only guess)" if x["how"] == "auto-length" else ""
            note = f"   <- {x['note']}" if x.get("note") else ""
            L.append(f"{key:34s} {_f(w['length']['value']):>7s} {w['dir'] or '-':>3s}  {x['plan'] or '-':8s} "
                     f"{_f(pv):>7s} {face or '-':>5s}  {x['how']}{flag}{note}")
    if any(g["openings"] for g in gt["rooms"]):
        L += ["", f"{'OPENING (gt room/opening)':34s} {'kind':7s} {'gt w':>6s}  {'plan':8s} {'plan w':>6s}  how"]
        for g in gt["rooms"]:
            for o in g["openings"]:
                key = f"{g['name']}/{o['name']}"
                x = m["openings"][key]
                pw = ops[x["plan"]]["width"]["value"] if x["plan"] else None
                note = f"   <- {x['note']}" if x.get("note") else ""
                L.append(f"{key:34s} {o['kind']:7s} {_f(o['width']['value']):>6s}  {x['plan'] or '-':8s} {_f(pw):>6s}  {x['how']}{note}")
    if gt["cross_lines"]:
        L += ["", f"{'CROSS LINE':34s} {'gt m':>7s}  plan walls"]
        for c in gt["cross_lines"]:
            ids = [m["walls"][f"{c[e]['room']}/{c[e]['wall']}"]["plan"] or "-" for e in ("from", "to")]
            L.append(f"{c['name']:34s} {_f(c['length']['value']):>7s}  {ids[0]} -> {ids[1]}")
    if any(x["plan"] is None for x in m["rooms"].values()):
        L += ["", "Plan rooms (to pick from; walls as id:faces:length):"]
        for r in rooms.values():
            ws = " ".join(f"{w['id'].split('.')[1]}:{outward_dir(r, w) or '?'}:{_f(w['length']['value'], 2)}" for w in r["walls"])
            L.append(f"  {r['id']:4s} {r['name']:12s} {_f(r['floor_area']['value'], 1):>5s} m2  {ws}")
    n_check = sum(x["how"] == "auto-length" for x in m["walls"].values())
    n_un = sum(x["plan"] is None and x["how"] != "override" for t in m.values() for x in t.values())
    L += ["", f"{n_check} wall(s) to CHECK, {n_un} unmatched item(s)."]
    L += [f"warning: {w}" for w in gt["warnings"]]
    return L


def _q(s: str) -> str:
    return json.dumps(s)  # a JSON string is a valid YAML scalar: safe for names with spaces or ':'


def write_overrides(path: Path, existing: dict, m: dict) -> None:
    """Write existing entries (kept as they are) plus every new proposal; unmatched ones as comments."""
    lines = ["# Matches between GT names and plan IDs. Edit the wrong lines; null = the plan has no such item.",
             "# Lines that are comments were not matched: fill in an ID and uncomment them.", ""]
    for sec in ("rooms", "walls", "openings"):
        lines.append(f"{sec}:")
        for k, v in existing[sec].items():
            lines.append(f"  {_q(k)}: {v if v is not None else 'null'}   # yours")
        for k, x in m[sec].items():
            if k in existing[sec]:
                continue
            if x["plan"] is None:
                lines.append(f"  # {_q(k)}: ?   # unmatched: {x.get('note', '')}")
            else:
                lines.append(f"  {_q(k)}: {x['plan']}   # {x['how']}" + ("  CHECK" if x["how"] == "auto-length" else ""))
        lines.append("")
    path.write_text("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan")
    ap.add_argument("gt")
    ap.add_argument("--overrides", help="overrides file (default eval/ground_truth/<capture>.match.yaml)")
    ap.add_argument("--write-overrides", action="store_true")
    a = ap.parse_args(argv)
    try:
        gt = load_gt(a.gt)
        opath = Path(a.overrides) if a.overrides else GT_DIR / f"{gt['capture']}.match.yaml"
        ov = load_overrides(opath)
        plan = json.loads(Path(a.plan).read_text())
        m = match(plan, gt, ov)
    except GTError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print("\n".join(report(plan, gt, m)))
    if a.write_overrides:
        write_overrides(opath, ov, m)
        print(f"wrote {opath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
