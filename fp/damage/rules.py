"""Likely concealed damage: transparent rules over a visible defect and where it sits in the plan.

No dataset supports learning concealed damage, so these are hypotheses, labelled as such, each carrying the
rule that fired (id + text) and its evidence. Rules read schema damage items (fp.damage.project) and the plan.

  C1  water stain or mould on a ceiling        -> leak from above (roof, pipes, a bathroom upstairs)
  C2  water stain / mould / peeling paint on a wall below 0.5 m -> rising damp or a leaking pipe in the wall
  C3  water stain or mould on a wall shared with a wet room      -> plumbing leak from that room
  C4  crack within 0.6 m of a door                               -> structural movement around the frame
  C5  water stain or mould on a ceiling or the top 0.3 m of a wall, in or next to a wet room (sharing a wall)
                                                                 -> leak from wet-room pipes in the ceiling void
      (a wall stain in a room whose ceiling was not observed can't be called "high", so C5 can't fire there)"""
from __future__ import annotations

import numpy as np

WET = ("water_stain", "mold")
WET_OR_PAINT = WET + ("peeling_paint",)
LOW_WALL = 0.5          # m: rising damp rarely reaches above ~0.5-1 m; a stain lower than this is suspicious
DOOR_NEAR = 0.6         # m from a door's centre line: frame corners crack under movement
CEILING_BAND = 0.3      # m below the ceiling: a wall stain this high came from above, not from the floor
PARTITION = 0.35        # m: a parallel wall of another room within this is the other face of the same wall

RULES = {
    "C1": "Possible leak from above: roof, pipes or a bathroom on the floor above.",
    "C2": "Possible rising damp or a leaking pipe inside the wall.",
    "C3": "Possible plumbing leak from the wet room behind this wall.",
    "C4": "Possible structural movement around the door frame.",
    "C5": "Possible leak from wet-room pipes running in the ceiling void.",
}


def _walls(plan):
    return {w["id"]: (r, w) for r in plan["rooms"] for w in r["walls"]}


def shared_wall_rooms(plan: dict, wall_id: str) -> list[str]:
    """Rooms on the other side of a wall: a parallel wall of another room within one partition thickness."""
    r, w = _walls(plan)[wall_id]
    p0, p1 = np.array(w["p0"]), np.array(w["p1"])
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    n = np.array([u[1], -u[0]])
    out = []
    for r2 in plan["rooms"]:
        if r2["id"] == r["id"]:
            continue
        for v in r2["walls"]:
            q0, q1 = np.array(v["p0"]), np.array(v["p1"])
            if abs(np.dot((q1 - q0) / max(np.linalg.norm(q1 - q0), 1e-9), u)) < 0.99:
                continue
            if abs(np.dot(q0 - p0, n)) > PARTITION:
                continue
            a, b = sorted([np.dot(q0 - p0, u), np.dot(q1 - p0, u)])
            if min(L, b) - max(0.0, a) > 0.3:
                out.append(r2["id"])
                break
    return out


def _room(plan, rid):
    return next(r for r in plan["rooms"] if r["id"] == rid)


def _doors(plan):
    return [o for r in plan["rooms"] for o in r["openings"] if o["kind"] == "door"]


def concealed(items: list[dict], plan: dict, wet_rooms: set[str] = frozenset()) -> list[dict]:
    """Schema ConcealedFlag dicts for the damage items (ids C1.. in order)."""
    walls = _walls(plan)
    ceil = {r["id"]: r["ceiling_height"]["value"] for r in plan["rooms"]}
    flags = []
    for d in items:
        cls, sid, (x, y, z) = d["class"], d["surface_id"], d["position"]
        kind = "ceiling" if sid.endswith(".ceiling") else "floor" if sid.endswith(".floor") else "wall"
        room = sid.split(".")[0]
        hits = []
        if kind == "ceiling" and cls in WET:
            hits.append(("C1", f"{cls} on the ceiling of {room}"))
        if kind == "wall" and cls in WET_OR_PAINT and z < LOW_WALL:
            hits.append(("C2", f"{cls} {z:.2f} m above the floor on {sid} (rule: below {LOW_WALL} m)"))
        behind = shared_wall_rooms(plan, sid) if kind == "wall" else []
        if kind == "wall" and cls in WET:
            wet_behind = [r for r in behind if r in wet_rooms]
            if wet_behind:
                hits.append(("C3", f"{cls} on {sid}, a wall shared with {', '.join(wet_behind)} (marked wet)"))
        if cls == "crack":
            for o in _doors(plan):
                if np.hypot(o["center"][0] - x, o["center"][1] - y) < DOOR_NEAR + o["width"]["value"] / 2:
                    hits.append(("C4", f"crack within {DOOR_NEAR} m of door {o['id']}"))
                    break
        H = ceil.get(room)
        high = kind == "ceiling" or (kind == "wall" and H is not None and z > H - CEILING_BAND)
        if cls in WET and high:
            # a ceiling's neighbours: every room sharing one of its walls (wet-room pipes run over the partition)
            near = behind if kind == "wall" else sorted({x for w in _room(plan, room)["walls"]
                                                         for x in shared_wall_rooms(plan, w["id"])})
            wet_near = [r for r in [room] + near if r in wet_rooms]
            if wet_near:
                hits.append(("C5", f"{cls} high on {sid}, in or next to wet room(s) {', '.join(wet_near)}"))
        for rule, ev in hits:
            flags.append({"id": f"C{len(flags) + 1}", "damage_id": d["id"], "rule_id": rule, "rule_text": RULES[rule],
                          "evidence": ev, "label": "hypothesis"})
    return flags
