"""Scope line items: deterministic repair quantities from the plan's surfaces and damage.

Every item is keyed to a surface id (R1.W3, R1.floor, R1.ceiling) and its quantity carries an interval
propagated from the geometry by interval arithmetic: each input's [lo, hi] is pushed through the formula
taking the worst case (e.g. wall area lo = length lo x height lo - openings hi). Simple, conservative, and
explainable line by line.

Rules (one item per surface and action):
  any damage on a wall          -> "repaint wall"           m2  = wall length x ceiling height - openings on it
  crack or hole on a surface    -> "patch and fill"         count of cracks + holes
  mould on a surface            -> "mould treatment"        m2  = sum of the mould extents
  water stain on a surface      -> "stain-block primer"     m2  = sum of the stain extents
  damage on a ceiling           -> "repaint ceiling"        m2  = room floor area
  water stain / mould on floor  -> "replace floor finish"   m2  = room floor area
                                   "replace skirting"       m   = perimeter - widths of doors and passages"""
from __future__ import annotations

ASSUMED_H = (2.4, 2.1, 3.2)   # m (value, lo, hi): wall height when the room's ceiling was not observed
DOOR_H = (2.05, 1.9, 2.4)     # m: door height when its lintel was not observed (work order 04 door range)
WET = ("water_stain", "mold")


def _iv(m: dict | None, default=None):
    """Measure -> (value, lo, hi); a missing value falls back to `default`."""
    if m is None or m.get("value") is None:
        return default
    return m["value"], m["lo"], m["hi"]


def _q(v, lo, hi, unit, method, observed=True) -> dict:
    lo, hi = max(0.0, min(lo, v)), max(hi, v)
    return {"value": round(v, 4), "lo": round(lo, 4), "hi": round(hi, 4), "unit": unit, "method": method,
            "observed": observed}


def wall_area(plan: dict, room: dict, wall: dict) -> dict:
    """Wall area net of the openings in it (m2), with an interval."""
    L = _iv(wall["length"])
    H = _iv(room["ceiling_height"], ASSUMED_H)
    observed = room["ceiling_height"]["value"] is not None and wall["length"]["observed"]
    v, lo, hi = L[0] * H[0], L[1] * H[1], L[2] * H[2]
    for o in (o for r in plan["rooms"] for o in r["openings"] if o["wall_id"] == wall["id"]):
        w = _iv(o["width"])
        h = _iv(o["height"], H if o["kind"] == "passage" else DOOR_H if o["kind"] == "door" else (1.2, 0.6, 1.8))
        if o["kind"] == "window" and o["height"] is None:
            observed = False
        v, lo, hi = v - w[0] * h[0], lo - w[2] * h[2], hi - w[1] * h[1]
    method = ("wall length x ceiling height - openings" if room["ceiling_height"]["value"] is not None
              else f"wall length x assumed height {ASSUMED_H[0]} m [{ASSUMED_H[1]}, {ASSUMED_H[2]}] (ceiling not "
                   "observed) - openings")
    return _q(v, lo, hi, "m2", method, observed)


def skirting(plan: dict, room: dict) -> dict:
    P = _iv(room["perimeter"])
    v, lo, hi = P
    for o in (o for r in plan["rooms"] for o in r["openings"]
              if o["kind"] in ("door", "passage") and (o["rooms"] and room["id"] in o["rooms"]
                                                        or o["wall_id"].startswith(room["id"] + "."))):
        w = _iv(o["width"])
        v, lo, hi = v - w[0], lo - w[2], hi - w[1]
    return _q(v, lo, hi, "m", "room perimeter - door and passage widths", room["perimeter"]["observed"])


def build(plan: dict) -> list[dict]:
    """Scope items (schema ScopeItem dicts, ids S1..) for the plan's damage."""
    rooms = {r["id"]: r for r in plan["rooms"]}
    walls = {w["id"]: (r, w) for r in plan["rooms"] for w in r["walls"]}
    by_surface: dict[str, list[dict]] = {}
    for d in plan["damage"]:
        by_surface.setdefault(d["surface_id"], []).append(d)
    items = []

    def add(sid, action, q):
        items.append({"id": f"S{len(items) + 1}", "surface_id": sid, "action": action, "quantity": q,
                      "unit": q["unit"]})

    for sid in sorted(by_surface, key=lambda s: (s.split(".")[0], s)):
        ds = by_surface[sid]
        classes = [d["class"] for d in ds]
        room = rooms[sid.split(".")[0]]
        if sid in walls:
            add(sid, "repaint wall (prepare, 2 coats)", wall_area(plan, *walls[sid]))
        n = sum(c in ("crack", "hole") for c in classes)
        if n:
            add(sid, "patch and fill cracks / holes", _q(n, n, n, "count", "count of crack and hole items"))
        for cls, action in (("mold", "mould treatment"), ("water_stain", "stain-block primer")):
            ex = [d["extent_m2"] for d in ds if d["class"] == cls]
            if ex:
                add(sid, action, _q(sum(e["value"] for e in ex), sum(e["lo"] for e in ex), sum(e["hi"] for e in ex),
                                    "m2", f"sum of {cls} extents on the surface"))
        if sid.endswith(".ceiling"):
            a = _iv(room["floor_area"])
            add(sid, "repaint ceiling", _q(*a, "m2", "room floor area", room["floor_area"]["observed"]))
        if sid.endswith(".floor") and any(c in WET for c in classes):
            a = _iv(room["floor_area"])
            add(sid, "replace floor finish", _q(*a, "m2", "room floor area", room["floor_area"]["observed"]))
            add(sid, "replace skirting", skirting(plan, room))
    return items
