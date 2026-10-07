"""Pipeline results -> plan.json (schema 1.0, see fp/schema.py and docs/SCHEMA.md).

Two jobs:
 1. `empty_plan` gives a valid plan before any stage has run, so a stage that isn't built yet leaves
    its fields `observed: false` with a warning instead of breaking the contract.
 2. `fill_from_geometry` converts what fp/geometry produces (walls, room polygons, doorway necks)
    into Measures with PROVISIONAL 90% intervals. Work order 07 replaces the constants below with
    values calibrated on tape ground truth.

Interval model (simple on purpose, every term explainable):
 - Each wall plane has a position uncertainty  pos = POS_BASE x TIER_SCALE + SMEAR_K x spread,
   where spread is the measured thickness of the wall's point band (smear from drift or fast pans).
   A wall that was never seen (inferred from the floor edge) gets  pos x INFERRED_FACTOR.
 - A wall's length is the distance between its two neighbouring planes, so its half-width is the
   sum of their two `pos` values, plus SCALE_REL x length for tiers whose metric scale is learned.
 - Area: moving wall i outward by d adds length_i x d, so half-width = sum(length_i x pos_i),
   plus 2 x SCALE_REL x area (a scale error s changes every length by s, so the area by about 2s).
 - Perimeter: moving a wall by d lengthens its two neighbours by d each, so half-width = 2 x sum(pos_i),
   plus SCALE_REL x perimeter.
 - Ceiling: CEILING_HALF x TIER_SCALE + SCALE_REL x height.
 - Smear is measured in the cloud itself, so it is added as-is, not multiplied by the tier scale.
 - Each half-width adds its terms (worst case), so the intervals are conservative until 07 calibrates them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SCHEMA_VERSION = "1.0"
LEVEL = 0.9  # the stated coverage of every [lo, hi]

# --- provisional interval constants (calibrated in work order 07) ---
TIER_SCALE = {"lidar": 1.0, "video": 3.0, "photos": 6.0}  # interval multiplier vs LiDAR (report badge "x N")
POS_BASE = 0.01        # m: plane-fit error of a sharp LiDAR wall (a clean wall band is about 1 cm thick)
SMEAR_K = 1.0          # m per m of wall-band thickness: a 4 cm smear means the plane could be 4 cm off
INFERRED_FACTOR = 3.0  # a wall that was not seen sits on the floor's edge: 3x less certain
# Learned metric depth (MapAnything) has a global scale error; the brief's own gates are 3% (video)
# and 8% (photos), so the provisional relative term matches them.
SCALE_REL = {"lidar": 0.0, "video": 0.03, "photos": 0.08}
CEILING_HALF = 0.02    # m: LiDAR ceiling-minus-floor layer distance, before the tier scale
NECK_HALF = 0.10       # m: a doorway taken as the narrowest neck of the floor mask, not jamb planes

TIER_NAME = {"lidar": "LiDAR depth (metric)", "video": "MapAnything metric depth (learned)",
             "photos": "MapAnything metric depth (learned)"}


class StageError(Exception):
    """A stage could not produce its output from this capture (e.g. no usable frames). The CLI records it
    as a warning and still writes a valid plan, with the missing parts `observed: false`."""

    def __init__(self, stage: str, message: str):
        super().__init__(message)
        self.stage = stage


class StageNotBuilt(StageError):
    """Raised by a pipeline stage that doesn't exist yet."""


def measure(value, half, unit: str, method: str, observed: bool = True) -> dict:
    """A Measure dict with a symmetric interval value +- half (half=None -> no interval)."""
    if value is None:
        return {"value": None, "lo": None, "hi": None, "unit": unit, "method": method, "observed": False}
    value, half = float(value), float(half)
    return {"value": round(value, 4), "lo": round(value - half, 4), "hi": round(value + half, 4),
            "unit": unit, "method": method, "observed": bool(observed)}


def not_observed(unit: str, method: str) -> dict:
    return measure(None, None, unit, method, observed=False)


def empty_plan(tier: str, capture_id: str, source: dict, device: str | None = None) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "tier": tier,
        "capture_id": capture_id,
        "device": device,
        "frame": {"up": "z", "floor_z": 0.0, "units": "m", "T_plan_world": None},
        "intervals": {"level": LEVEL, "scale": TIER_SCALE[tier], "calibrated": False,
                      "method": "provisional: plane position error x tier scale (fp/contract.py); "
                                "calibrated in work order 07"},
        "source": source,
        "rooms": [],
        "connections": [],
        "footprint_area": not_observed("m2", "no rooms reconstructed"),
        "drift": {"method": "none", "enabled": False, "metrics": {}},
        "damage": [],
        "concealed": [],
        "scope": [],
        "timings": {},
        "warnings": [],
    }


def _pos_half(wall: dict, tier: str) -> float:
    pos = POS_BASE * TIER_SCALE[tier] + SMEAR_K * (wall.get("spread_cm") or 0.0) / 100
    return pos * (1.0 if wall["observed"] else INFERRED_FACTOR)


def room_to_schema(room: dict, tier: str, extra_rel: float = 0.0) -> dict:
    """One room from fp.geometry.plan._room (+ id/name) -> schema Room. Openings are added later.
    extra_rel: added to the tier's relative scale term when the learned scale disagrees with priors."""
    rid, walls = room["id"], room["walls"]
    rel = SCALE_REL[tier] + extra_rel
    pos = [_pos_half(w, tier) for w in walls]
    out_walls = []
    for k, w in enumerate(walls):
        prev, nxt = walls[k - 1], walls[(k + 1) % len(walls)]
        half = pos[k - 1] + pos[(k + 1) % len(walls)] + rel * w["length_m"]
        both_seen = prev["observed"] and nxt["observed"]
        method = "distance between neighbouring wall planes" + ("" if both_seen else " (a neighbour was inferred)")
        out_walls.append({"id": f"{rid}.{w['id']}", "p0": w["p0"], "p1": w["p1"],
                          "length": measure(w["length_m"], half, "m", method, observed=both_seen),
                          "observed": bool(w["observed"]), "coverage": float(w["coverage"])})
    lengths = np.array([w["length_m"] for w in walls])
    all_seen = all(w["observed"] for w in walls)
    area_half = float(np.sum(lengths * pos)) + 2 * rel * room["area_m2"]
    perim_half = 2 * float(np.sum(pos)) + rel * float(lengths.sum())
    H = room.get("ceiling_h_m")
    ceiling = (measure(H, CEILING_HALF * TIER_SCALE[tier] + rel * H, "m",
                       "ceiling layer minus floor layer (whole capture)")
               if H is not None else not_observed("m", "no ceiling layer found"))
    return {
        "id": rid, "name": room.get("name") or rid,
        "polygon": [[float(x), float(y)] for x, y in room["polygon"]],
        "floor_area": measure(room["area_m2"], area_half, "m2", "polygon area on wall planes", observed=all_seen),
        "ceiling_height": ceiling,
        "perimeter": measure(float(lengths.sum()), perim_half, "m", "sum of wall lengths", observed=all_seen),
        "walls": out_walls,
        "openings": [],
    }


def _nearest_wall(room: dict, c) -> dict:
    def dist(w):
        a, b, p = np.array(w["p0"]), np.array(w["p1"]), np.array(c, float)
        t = np.clip(np.dot(p - a, b - a) / max(np.dot(b - a, b - a), 1e-12), 0, 1)
        return float(np.linalg.norm(a + t * (b - a) - p))
    return min(room["walls"], key=dist)


def fill_from_geometry(plan: dict, rooms: list[dict], connections: list[dict]) -> None:
    """Rooms and doorway necks from fp.geometry -> plan rooms, openings, connections, footprint."""
    tier = plan["tier"]
    # camera tiers: a learned scale that disagrees with priors widens intervals (fp/recon/camera.scale_check)
    extra = ((plan["source"].get("recon") or {}).get("scale_check") or {}).get("extra_rel", 0.0)
    plan["rooms"] = [room_to_schema(r, tier, extra) for r in rooms]
    by_id = {r["id"]: r for r in plan["rooms"]}
    for c in connections:
        a, b = c["rooms"]
        if a not in by_id or b not in by_id or not by_id[a]["walls"]:
            continue
        room = by_id[a]
        oid = f"{a}.O{len(room['openings']) + 1}"
        room["openings"].append({
            "id": oid, "kind": "passage" if c.get("kind") == "opening" else "door",
            "wall_id": _nearest_wall(room, c["center"])["id"], "rooms": [a, b],
            "width": measure(c["width_m"], NECK_HALF * TIER_SCALE[tier], "m",
                             "narrowest neck of the floor footprint between the rooms (not jamb planes)",
                             observed=False),
            "height": None, "sill": None, "center": [float(v) for v in c["center"]]})
        plan["connections"].append({"rooms": [a, b], "opening_id": oid})
    if plan["rooms"]:
        areas = [r["floor_area"] for r in plan["rooms"]]
        v = sum(m["value"] for m in areas)
        half = sum(m["hi"] - m["value"] for m in areas)
        plan["footprint_area"] = measure(v, half, "m2", "sum of room floor areas",
                                         observed=all(m["observed"] for m in areas))


def damage_to_schema(plan: dict, items: list[dict]) -> None:
    """Damage items from fp.damage.project.assess (+ concealed rules) -> plan damage / concealed.
    The detection's largest 5-95% spread s bounds its area by s^2, so extent is stated as [0, s^2] with
    s^2/2 as the value, and marked not observed (work order 06 measures the mask on the surface)."""
    for d in items:
        surface = f"{d['room']}.{d['wall']}" if d["kind"] == "wall" else f"{d['room']}.{d['kind']}"
        s, (x, y, z) = d["size_m"], d["position_m"]
        u, v = (d.get("along_wall_m", 0.0), z) if d["kind"] == "wall" else (x, y)
        plan["damage"].append({
            "id": d["id"], "class": d["type"].replace(" ", "_"), "surface_id": surface,
            "position": [float(x), float(y), float(z)],
            "extent_m2": measure(s * s / 2, s * s / 2, "m2",
                                 "bounding square of the detection's 3D spread (upper bound)", observed=False),
            "bbox_on_surface": _bbox(u, v, s),
            "evidence_image": d.get("evidence"), "score": float(d["score"])})
        for c in d.get("concealed", []):
            plan["concealed"].append({
                "id": f"C{len(plan['concealed']) + 1}", "damage_id": d["id"], "rule_id": c["rule"],
                "rule_text": c["hypothesis"], "evidence": c["evidence"], "label": "hypothesis"})


def _bbox(u: float, v: float, s: float) -> dict:
    """An s x s box centred on (u, v), clipped to the surface's positive quadrant (along-wall >= 0, above floor)."""
    u0, v0 = max(0.0, u - s / 2), max(0.0, v - s / 2)
    return {"u0": u0, "v0": v0, "u1": max(u0, u + s / 2), "v1": max(v0, v + s / 2)}


def capture_id(path: Path) -> str:
    path = Path(path)
    return path.stem if path.is_file() else path.name
