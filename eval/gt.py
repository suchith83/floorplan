"""Tape/laser ground truth (GT): load it, match it to a plan.json, score the plan against it.

GT is hand-written YAML (see eval/ground_truth/TEMPLATE.yaml). Walls and openings are named the way a
person names them ("kitchen / window wall"), so they must be matched to plan IDs ("R2.W3"):

  rooms     plan_room in the GT  >  overrides file  >  plan room with the same name (case-insensitive)
  walls     plan_wall in the GT  >  overrides file  >  Hungarian on |length difference| among the room's
            walls, restricted to walls facing `dir` when the GT gives it ("dir"); without `dir` the
            match is a guess from length alone ("auto-length": confirm it by eye on plan.svg)
  openings  overrides file  >  Hungarian on |width difference| among plan openings on the matched wall
            of the same kind

The overrides file (eval/ground_truth/<capture>.match.yaml) is where the user fixes wrong guesses:
    rooms:    {kitchen: R2}
    walls:    {kitchen/window wall: R2.W3}
    openings: {kitchen/back door: R2.O1, kitchen/hatch: null}     # null = the plan has no such opening

Every scored row carries `in_interval` (GT value inside the plan's [lo, hi]) so 90% coverage can be checked.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment
from shapely.geometry import LineString, Point, Polygon

# Gates from docs/brief/applied_ai_brief.md (no per-capture tuning).
CEILING_GATE_M = 0.015      # m, ceiling height error per room
OPENING_GATE_M = 0.02       # m, opening width error
OPENING_PASS_FRAC = 0.85    # fraction of openings (GT + phantoms) that must be within OPENING_GATE_M
WALL_PCT_LOOSE = 0.08       # photo-tier wall gate (and photo footprint gate)
WALL_PCT_TIGHT = 0.03       # video-tier wall gate

# Matching / sanity constants.
AXIS_COS = 0.9              # a wall faces +x if its outward normal has x-component >= this (about 25 deg)
NORMAL_STEP_M = 0.05        # m, step off a wall's midpoint to test which side is the room interior
OPENING_NEAR_WALL_M = 0.35  # m, an opening owned by the neighbour room counts as on our wall if its centre
                            # is this close to it (a wall is ~0.1-0.3 m thick)
PARALLEL_COS = 0.98         # cross-line walls must be parallel to within ~11 deg
MAX_SPREAD_M = 0.05         # m, readings of one dimension spreading more than this are probably a typo
BIG = 1e6                   # cost of a forbidden pair in a Hungarian assignment

DIRS = {"+x": (1.0, 0.0), "-x": (-1.0, 0.0), "+y": (0.0, 1.0), "-y": (0.0, -1.0)}
KINDS = ("door", "window", "passage")


class GTError(ValueError):
    """A ground-truth or overrides file is malformed; the message names the bad field."""


# ---------------------------------------------------------------- loading

def _readings(v, where: str) -> dict:
    """1-3 readings (or one number) -> {value: mean, spread: max-min, readings}."""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        v = [v]
    if not isinstance(v, list) or not 1 <= len(v) <= 3:
        raise GTError(f"{where}: expected 1-3 readings in metres like [2.415, 2.417, 2.416], got {v!r}")
    for r in v:
        if isinstance(r, bool) or not isinstance(r, (int, float)) or not 0 < r < 100:
            raise GTError(f"{where}: reading {r!r} is not a length in metres (0-100)")
    spread = max(v) - min(v)
    return {"value": float(np.mean(v)), "spread": float(spread), "readings": [float(r) for r in v],
            "warn": f"{where}: readings spread {spread * 100:.1f} cm (typo?)" if spread > MAX_SPREAD_M else None}


def _req(d: dict, key: str, where: str):
    if not isinstance(d, dict):
        raise GTError(f"{where}: expected a mapping (key: value lines), got {d!r}")
    if d.get(key) in (None, ""):
        raise GTError(f"{where}: missing required field '{key}'")
    return d[key]


def _list(d: dict, key: str, where: str) -> list:
    v = d.get(key) or []
    if not isinstance(v, list):
        raise GTError(f"{where}.{key}: expected a list (lines starting with '-'), got {type(v).__name__}")
    return v


def _name(d: dict, where: str) -> str:
    n = str(_req(d, "name", where)).strip()
    if "/" in n:
        raise GTError(f"{where}.name: {n!r} must not contain '/' (it separates room/wall in the overrides file)")
    return n


def load_gt(path) -> dict:
    """Read and validate a GT YAML file; every length becomes {value, spread, readings}."""
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text())
    except yaml.YAMLError as e:
        raise GTError(f"{path}: not valid YAML: {e}") from e
    if not isinstance(raw, dict):
        raise GTError(f"{path}: top level must be a mapping with capture/tool/rooms")
    gt = {"capture": str(_req(raw, "capture", str(path))), "tool": str(raw.get("tool") or "unknown"),
          "rooms": [], "cross_lines": [], "warnings": []}
    names = set()
    for i, r in enumerate(_list(raw, "rooms", "")):
        rn = _name(r, f"rooms[{i}]")
        w = f"rooms[{rn}]"
        if rn.lower() in names:
            raise GTError(f"{w}: room name used twice")
        names.add(rn.lower())
        room = {"name": rn, "plan_room": r.get("plan_room"),
                "ceiling": _readings(r["ceiling"], f"{w}.ceiling") if r.get("ceiling") is not None else None,
                "area_m2": None, "walls": [], "openings": []}
        if r.get("area_m2") is not None:
            a = r["area_m2"]
            if isinstance(a, bool) or not isinstance(a, (int, float)) or a <= 0:
                raise GTError(f"{w}.area_m2: expected a positive number of m2, got {a!r}")
            room["area_m2"] = float(a)
        wnames = set()
        for j, wl in enumerate(_list(r, "walls", w)):
            wn = _name(wl, f"{w}.walls[{j}]")
            ww = f"{w}.walls[{wn}]"
            if wn in wnames:
                raise GTError(f"{ww}: wall name used twice in this room")
            wnames.add(wn)
            d = wl.get("dir")
            if d is not None and d not in DIRS:
                raise GTError(f"{ww}.dir: must be one of {sorted(DIRS)} (or left out), got {d!r}")
            room["walls"].append({"name": wn, "length": _readings(_req(wl, "length", ww), f"{ww}.length"),
                                  "dir": d, "plan_wall": wl.get("plan_wall")})
        onames = set()
        for j, o in enumerate(_list(r, "openings", w)):
            on = _name(o, f"{w}.openings[{j}]")
            ow = f"{w}.openings[{on}]"
            if on in onames:
                raise GTError(f"{ow}: opening name used twice in this room")
            onames.add(on)
            kind = _req(o, "kind", ow)
            if kind not in KINDS:
                raise GTError(f"{ow}.kind: must be one of {KINDS}, got {kind!r}")
            wall = str(_req(o, "wall", ow))
            if wall not in wnames:
                raise GTError(f"{ow}.wall: {wall!r} is not a wall of {rn} (walls: {sorted(wnames)})")
            room["openings"].append({
                "name": on, "kind": kind, "wall": wall,
                "width": _readings(_req(o, "width", ow), f"{ow}.width"),
                "height": _readings(o["height"], f"{ow}.height") if o.get("height") is not None else None,
                "sill": _readings(o["sill"], f"{ow}.sill") if o.get("sill") is not None else None})
        gt["rooms"].append(room)
    walls_of = {r["name"]: {w["name"] for w in r["walls"]} for r in gt["rooms"]}
    for i, c in enumerate(_list(raw, "cross_lines", "")):
        cn = str(_req(c, "name", f"cross_lines[{i}]"))
        cw = f"cross_lines[{cn}]"
        ends = {}
        for end in ("from", "to"):
            e = _req(c, end, cw)
            rn, wn = str(_req(e, "room", f"{cw}.{end}")), str(_req(e, "wall", f"{cw}.{end}"))
            if rn not in walls_of:
                raise GTError(f"{cw}.{end}.room: {rn!r} is not a GT room")
            if wn not in walls_of[rn]:
                raise GTError(f"{cw}.{end}.wall: {wn!r} is not a wall of {rn} (walls: {sorted(walls_of[rn])})")
            ends[end] = {"room": rn, "wall": wn}
        gt["cross_lines"].append({"name": cn, **ends, "length": _readings(_req(c, "length", cw), f"{cw}.length")})
    gt["warnings"] = [x["warn"] for x in _iter_readings(gt) if x.get("warn")]
    return gt


def _iter_readings(gt):
    for r in gt["rooms"]:
        if r["ceiling"]:
            yield r["ceiling"]
        for w in r["walls"]:
            yield w["length"]
        for o in r["openings"]:
            yield from (o[k] for k in ("width", "height", "sill") if o[k])
    for c in gt["cross_lines"]:
        yield c["length"]


def load_overrides(path) -> dict:
    """Read <capture>.match.yaml -> {rooms: {}, walls: {}, openings: {}}; empty when the file is absent."""
    path = Path(path)
    out = {"rooms": {}, "walls": {}, "openings": {}}
    if not path.exists():
        return out
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise GTError(f"{path}: top level must be a mapping with rooms/walls/openings")
    for key in out:
        v = raw.get(key) or {}
        if not isinstance(v, dict):
            raise GTError(f"{path}: '{key}' must be a mapping like  gt_name: plan_id")
        out[key] = {str(k): (None if x is None else str(x)) for k, x in v.items()}
    return out


# ---------------------------------------------------------------- plan geometry

def outward_dir(room: dict, wall: dict) -> str | None:
    """Axis direction ('+x', ...) the wall faces away from its room's interior; None if not axis-aligned."""
    p0, p1 = np.asarray(wall["p0"], float), np.asarray(wall["p1"], float)
    d = p1 - p0
    if np.linalg.norm(d) < 1e-9:
        return None
    n = np.array([d[1], -d[0]]) / np.linalg.norm(d)  # right-hand side = outward for a CCW polygon
    mid = (p0 + p1) / 2
    if Polygon(room["polygon"]).contains(Point(*(mid + NORMAL_STEP_M * n))):
        n = -n                                        # wall stored clockwise: flip
    for name, axis in DIRS.items():
        if float(n @ axis) >= AXIS_COS:
            return name
    return None


def _wall_index(plan: dict) -> dict:
    return {w["id"]: (r, w) for r in plan["rooms"] for w in r["walls"]}


def _len(w):
    return w["length"]["value"]


# ---------------------------------------------------------------- matching

def match(plan: dict, gt: dict, overrides: dict | None = None) -> dict:
    """Propose plan IDs for every GT room, wall and opening. Each entry: {plan, how, note}."""
    ov = overrides or {"rooms": {}, "walls": {}, "openings": {}}
    rooms_by_id = {r["id"]: r for r in plan["rooms"]}
    walls = _wall_index(plan)
    openings = {o["id"]: o for r in plan["rooms"] for o in r["openings"]}

    def check(pid, table, what):
        if pid is not None and pid not in table:
            raise GTError(f"{what}: plan has no {pid!r} (known: {', '.join(sorted(table)) or 'none'})")
        return pid

    out = {"rooms": {}, "walls": {}, "openings": {}}
    by_name = {}
    for r in plan["rooms"]:
        by_name.setdefault(str(r.get("name", "")).lower(), r["id"])
    for g in gt["rooms"]:
        n = g["name"]
        if g["plan_room"]:
            m = {"plan": check(g["plan_room"], rooms_by_id, f"GT rooms[{n}].plan_room"), "how": "override"}
        elif n in ov["rooms"]:
            m = {"plan": check(ov["rooms"][n], rooms_by_id, f"overrides rooms[{n}]"), "how": "override"}
        elif n.lower() in by_name:
            m = {"plan": by_name[n.lower()], "how": "name"}
        else:
            m = {"plan": None, "how": "unmatched",
                 "note": f"no plan room named {n!r}: add  rooms: {{{n}: R?}}  to the overrides file"}
        out["rooms"][n] = m

    # Walls, pass 1: overrides (GT plan_wall, then the overrides file), so pass 2 never reuses those walls.
    todo_by_room = {}
    for g in gt["rooms"]:
        room = rooms_by_id.get(out["rooms"][g["name"]]["plan"])
        todo_by_room[g["name"]] = (room, [])
        for w in g["walls"]:
            key = f"{g['name']}/{w['name']}"
            if w["plan_wall"]:
                out["walls"][key] = {"plan": check(w["plan_wall"], walls, f"GT {key}.plan_wall"), "how": "override"}
            elif key in ov["walls"]:
                out["walls"][key] = {"plan": check(ov["walls"][key], walls, f"overrides walls[{key}]"), "how": "override"}
            elif room is None:
                out["walls"][key] = {"plan": None, "how": "unmatched", "note": "room not matched"}
            else:
                todo_by_room[g["name"]][1].append((key, w))
    # Pass 2: per room, Hungarian on |length difference|, only among walls facing `dir` when it is given.
    for room, todo in todo_by_room.values():
        if room is None or not todo:
            continue
        taken = {x["plan"] for x in out["walls"].values()} - {None}
        free = [w for w in room["walls"] if w["id"] not in taken and _len(w) is not None]
        faces = {w["id"]: outward_dir(room, w) for w in free}
        cost = np.full((len(todo), max(len(free), 1)), BIG)
        for i, (_, gw) in enumerate(todo):
            for j, pw in enumerate(free):
                if gw["dir"] is None or faces[pw["id"]] == gw["dir"]:
                    cost[i, j] = abs(_len(pw) - gw["length"]["value"])
        ri, ci = linear_sum_assignment(cost) if free else ([], [])
        got = {i: j for i, j in zip(ri, ci) if cost[i, j] < BIG}
        for i, (key, gw) in enumerate(todo):
            if i in got:
                out["walls"][key] = {"plan": free[got[i]]["id"], "how": "dir" if gw["dir"] else "auto-length"}
            else:
                why = f"no free plan wall faces {gw['dir']}" if gw["dir"] else "no free plan wall left"
                out["walls"][key] = {"plan": None, "how": "unmatched", "note": why}

    # Openings: overrides first, then Hungarian on width among same-kind openings on the matched wall.
    todo = []
    for g in gt["rooms"]:
        for o in g["openings"]:
            key = f"{g['name']}/{o['name']}"
            if key in ov["openings"]:
                pid = check(ov["openings"][key], openings, f"overrides openings[{key}]")
                out["openings"][key] = {"plan": pid, "how": "override"}
            else:
                todo.append((key, o, out["walls"][f"{g['name']}/{o['wall']}"]["plan"]))
    taken = {m["plan"] for m in out["openings"].values()} - {None}
    free = [o for o in openings.values() if o["id"] not in taken]
    cost = np.full((len(todo), max(len(free), 1)), BIG)
    for i, (_, go, pw) in enumerate(todo):
        for j, po in enumerate(free):
            if pw is not None and po["kind"] == go["kind"] and _on_wall(po, pw, walls):
                cost[i, j] = abs(po["width"]["value"] - go["width"]["value"])
    ri, ci = linear_sum_assignment(cost) if free and todo else ([], [])
    got = {i: j for i, j in zip(ri, ci) if cost[i, j] < BIG}
    for i, (key, go, pw) in enumerate(todo):
        if i in got:
            out["openings"][key] = {"plan": free[got[i]]["id"], "how": "wall+width"}
            continue
        if pw is None:
            note = "its wall is not matched"
        else:
            other = [po["id"] + " (" + po["kind"] + ")" for po in free if _on_wall(po, pw, walls)]
            note = f"no {go['kind']} on {pw}" + (f"; plan has {', '.join(other)} there (kind differs)" if other else "")
        out["openings"][key] = {"plan": None, "how": "unmatched", "note": note}
    return out


def _on_wall(o: dict, wall_id: str, walls: dict) -> bool:
    """Is plan opening `o` on plan wall `wall_id`? Its own wall, or (owned by the neighbour room) close to it."""
    if o["wall_id"] == wall_id:
        return True
    room, w = walls[wall_id]
    if not o.get("rooms") or room["id"] not in o["rooms"]:
        return False
    return LineString([w["p0"], w["p1"]]).distance(Point(o["center"])) <= OPENING_NEAR_WALL_M


# ---------------------------------------------------------------- scoring

def _row(gt_val: float, m: dict | None) -> dict:
    """Compare a GT value with a plan Measure: plan value, interval, abs and % error, in_interval."""
    if m is None or m.get("value") is None:
        return {"gt": round(gt_val, 4), "plan": None, "lo": None, "hi": None,
                "err_m": None, "err_pct": None, "in_interval": False}
    v = m["value"]
    return {"gt": round(gt_val, 4), "plan": v, "lo": m["lo"], "hi": m["hi"], "err_m": round(v - gt_val, 4),
            "err_pct": round(100 * (v - gt_val) / gt_val, 2), "in_interval": bool(m["lo"] <= gt_val <= m["hi"])}


def _cross_distance(wa: dict, wb: dict) -> float:
    """Distance between the planes of two parallel axis-aligned walls (|difference of their constant coord|)."""
    da = np.subtract(wa["p1"], wa["p0"]) / np.linalg.norm(np.subtract(wa["p1"], wa["p0"]))
    db = np.subtract(wb["p1"], wb["p0"]) / np.linalg.norm(np.subtract(wb["p1"], wb["p0"]))
    if abs(float(da @ db)) < PARALLEL_COS:
        raise GTError(f"{wa['id']} and {wb['id']} are not parallel: a cross line needs two parallel walls")
    n = np.array([-da[1], da[0]])  # wall normal; works for any orientation, = constant coord when axis-aligned
    ca = n @ (np.add(wa["p0"], wa["p1"]) / 2)
    cb = n @ (np.add(wb["p0"], wb["p1"]) / 2)
    return float(abs(ca - cb))


def score(plan: dict, gt: dict, matches: dict) -> dict:
    """Score rows for walls, ceilings, openings, room areas, footprint, cross lines, plus a summary."""
    rooms_by_id = {r["id"]: r for r in plan["rooms"]}
    walls = _wall_index(plan)
    openings = {o["id"]: o for r in plan["rooms"] for o in r["openings"]}
    res = {"capture": gt["capture"], "plan_capture": plan.get("capture_id"), "tier": plan.get("tier"),
           "walls": [], "ceilings": [], "openings": [], "phantoms": [], "areas": [], "footprint": None,
           "cross_lines": [], "unmatched": [], "warnings": list(gt.get("warnings", []))}

    for g in gt["rooms"]:
        rm = matches["rooms"][g["name"]]
        room = rooms_by_id.get(rm["plan"])
        if room is None:
            res["unmatched"].append(f"room {g['name']}")
        for w in g["walls"]:
            key = f"{g['name']}/{w['name']}"
            m = matches["walls"].get(key, {"plan": None, "how": "unmatched"})
            pw = walls[m["plan"]][1] if m["plan"] else None
            if pw is None:
                res["unmatched"].append(f"wall {key}")
            res["walls"].append({"gt_room": g["name"], "gt_wall": w["name"], "plan_id": m["plan"], "how": m["how"],
                                 "gt_spread": w["length"]["spread"], **_row(w["length"]["value"], pw and pw["length"])})
        if g["ceiling"] is not None:
            row = _row(g["ceiling"]["value"], room and room["ceiling_height"])
            row["pass"] = row["err_m"] is not None and abs(row["err_m"]) <= CEILING_GATE_M
            res["ceilings"].append({"gt_room": g["name"], "plan_id": rm["plan"], **row})
        if g["area_m2"] is not None:
            res["areas"].append({"gt_room": g["name"], "plan_id": rm["plan"],
                                 **_row(g["area_m2"], room and room["floor_area"])})
        for o in g["openings"]:
            key = f"{g['name']}/{o['name']}"
            m = matches["openings"].get(key, {"plan": None, "how": "unmatched"})
            po = openings.get(m["plan"]) if m["plan"] else None
            row = {"gt_room": g["name"], "gt_opening": o["name"], "kind": o["kind"], "plan_id": m["plan"],
                   "how": m["how"], "detected": po is not None, **_row(o["width"]["value"], po and po["width"])}
            row["width_ok"] = row["err_m"] is not None and abs(row["err_m"]) <= OPENING_GATE_M
            for k in ("height", "sill"):
                if o[k] is not None:
                    row[k] = _row(o[k]["value"], po and po.get(k))
            res["openings"].append(row)

    # Phantoms: plan openings touching a matched room that no GT opening claimed.
    matched_rooms = {m["plan"] for m in matches["rooms"].values()} - {None}
    claimed = {m["plan"] for m in matches["openings"].values()} - {None}
    for oid, o in openings.items():
        touches = walls[o["wall_id"]][0]["id"] in matched_rooms or bool(set(o.get("rooms") or []) & matched_rooms)
        if touches and oid not in claimed:
            res["phantoms"].append({"plan_id": oid, "kind": o["kind"], "wall_id": o["wall_id"],
                                    "width": o["width"]["value"]})

    if gt["rooms"] and all(g["area_m2"] is not None for g in gt["rooms"]):
        res["footprint"] = _row(sum(g["area_m2"] for g in gt["rooms"]), plan.get("footprint_area"))

    for c in gt["cross_lines"]:
        ids = [matches["walls"].get(f"{c[e]['room']}/{c[e]['wall']}", {}).get("plan") for e in ("from", "to")]
        row = {"name": c["name"], "plan_walls": ids, "gt": round(c["length"]["value"], 4), "plan": None,
               "err_m": None, "err_pct": None, "in_interval": False}
        if None in ids:
            row["note"] = "an end wall is not matched"
        else:
            d = _cross_distance(walls[ids[0]][1], walls[ids[1]][1])
            row.update(plan=round(d, 4), err_m=round(d - row["gt"], 4), err_pct=round(100 * (d - row["gt"]) / row["gt"], 2),
                       in_interval=None)  # plan.json has no interval for a derived distance
        res["cross_lines"].append(row)

    res["summary"] = _summary(res)
    return res


def _summary(res: dict) -> dict:
    def q(a, p):
        return round(float(np.percentile(a, p)), 4) if len(a) else None

    w = [r for r in res["walls"] if r["err_m"] is not None]
    werr = np.abs([r["err_m"] for r in w])
    wpct = np.abs([r["err_pct"] for r in w]) / 100
    c = res["ceilings"]
    cerr = [abs(r["err_m"]) for r in c if r["err_m"] is not None]
    o = res["openings"]
    n_ok = sum(r["width_ok"] for r in o)
    n_ph = len(res["phantoms"])
    denom = len(o) + n_ph
    x = [abs(r["err_m"]) for r in res["cross_lines"] if r["err_m"] is not None]
    return {
        "walls": {"n": len(w), "n_unmatched": len(res["walls"]) - len(w),
                  "median_abs_err_m": q(werr, 50), "p90_abs_err_m": q(werr, 90),
                  "frac_within_8pct": round(float(np.mean(wpct <= WALL_PCT_LOOSE)), 3) if w else None,
                  "frac_within_3pct": round(float(np.mean(wpct <= WALL_PCT_TIGHT)), 3) if w else None,
                  "coverage": round(float(np.mean([r["in_interval"] for r in w])), 3) if w else None},
        "ceilings": {"n": len(c), "n_not_observed": len(c) - len(cerr),
                     "max_abs_err_m": round(max(cerr), 4) if cerr else None,
                     "coverage": round(float(np.mean([r["in_interval"] for r in c if r["err_m"] is not None])), 3) if cerr else None,
                     "pass": all(r["pass"] for r in c) if c else None},
        "openings": {"n_gt": len(o), "n_detected": sum(r["detected"] for r in o),
                     "n_missed": sum(not r["detected"] for r in o), "n_phantom": n_ph, "n_width_ok": n_ok,
                     "frac_ok": round(n_ok / denom, 3) if denom else None,
                     "pass": (n_ok / denom >= OPENING_PASS_FRAC) if denom else None},
        "cross_lines": {"n": len(x), "median_abs_err_m": q(x, 50)},
    }
