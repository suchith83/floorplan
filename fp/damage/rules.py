"""Likely concealed damage: transparent rules over a visible defect and where it sits in the plan.

No dataset supports learning concealed damage, so these are hypotheses, labelled as such, each
showing the rule that fired and its evidence."""
from __future__ import annotations

import numpy as np

WET = ("water stain", "mold")
WET_OR_PAINT = WET + ("peeling paint",)


def _shared_wall_rooms(plan: dict, room_id: str, wall_id: str, thickness: float = 0.35) -> list[str]:
    """Rooms on the other side of a wall: a parallel wall of another room within one wall thickness."""
    room = next(r for r in plan["rooms"] if r["id"] == room_id)
    w = next(x for x in room["walls"] if x["id"] == wall_id)
    ax = 0 if w["axis"] == "x" else 1
    lo, hi = sorted([w["p0"][1 - ax], w["p1"][1 - ax]])
    out = []
    for r in plan["rooms"]:
        if r["id"] == room_id:
            continue
        for v in r["walls"]:
            vlo, vhi = sorted([v["p0"][1 - ax], v["p1"][1 - ax]])
            if v["axis"] == w["axis"] and abs(v["coord"] - w["coord"]) < thickness and min(hi, vhi) - max(lo, vlo) > 0.3:
                out.append(r["id"])
                break
    return out


def concealed(items: list[dict], plan: dict, wet_rooms: set[str] = frozenset()) -> None:
    """Adds `concealed` hypotheses to each damage item, in place."""
    for d in items:
        hyp = []
        if d["kind"] == "ceiling" and d["type"] in WET:
            hyp.append(("C1", "Possible leak from above: roof, pipes or a bathroom on the floor above.",
                        f"{d['type']} on the ceiling"))
        if d["kind"] == "wall" and d["type"] in WET_OR_PAINT and d["height_m"] < 0.5:
            hyp.append(("C2", "Possible rising damp or a leaking pipe inside the wall.",
                        f"{d['type']} {d['height_m']:.2f} m above the floor (rule: below 0.5 m)"))
        if d["kind"] == "wall" and d["type"] in WET:
            behind = [r for r in _shared_wall_rooms(plan, d["room"], d["wall"]) if r in wet_rooms]
            if behind:
                hyp.append(("C3", "Possible plumbing leak from the wet room behind this wall.",
                            f"{d['type']} on a wall shared with {', '.join(behind)}, marked as a wet room"))
        if d["type"] == "crack":
            near = [c for c in plan.get("connections", [])
                    if np.hypot(c["center"][0] - d["position_m"][0], c["center"][1] - d["position_m"][1]) < 0.6]
            if near:
                hyp.append(("C4", "Possible structural movement around the door frame.",
                            f"crack within 0.6 m of the door between {' and '.join(near[0]['rooms'])}"))
        d["concealed"] = [{"rule": r, "hypothesis": h, "evidence": e, "label": "hypothesis"} for r, h, e in hyp]
