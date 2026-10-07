"""eval/cross_tier.py: a plan rotated by 90 deg and shifted is put back on itself, rooms and walls match."""
import importlib.util
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("cross_tier", Path(__file__).parents[1] / "eval" / "cross_tier.py")
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)


def _m(v):
    return {"value": v, "lo": v * 0.95, "hi": v * 1.05, "unit": "m", "method": "t", "observed": True}


def _room(rid, poly):
    poly = np.asarray(poly, float)
    walls = [{"id": f"{rid}.W{k + 1}", "p0": poly[k].tolist(), "p1": poly[(k + 1) % len(poly)].tolist(),
              "length": _m(float(np.linalg.norm(poly[(k + 1) % len(poly)] - poly[k])))} for k in range(len(poly))]
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return {"id": rid, "name": rid, "polygon": poly.tolist(), "walls": walls, "floor_area": _m(area),
            "perimeter": _m(sum(w["length"]["value"] for w in walls)),
            "ceiling_height": {"value": None, "lo": None, "hi": None, "unit": "m", "method": "t", "observed": False}}


def _plan(rooms):
    return {"rooms": rooms, "connections": [{"rooms": ["A", "B"]}],
            "footprint_area": _m(sum(r["floor_area"]["value"] for r in rooms))}


def test_rotated_shifted_plan_aligns_back():
    a = [[0, 0], [4, 0], [4, 3], [0, 3]]
    b = [[4, 0], [6, 0], [6, 1.5], [4, 1.5]]   # L-shaped footprint: a symmetric one has two equally good yaws
    ref = _plan([_room("A", a), _room("B", b)])
    rot = lambda P: (np.asarray(P, float) @ np.array([[0, -1], [1, 0]]).T + [10.3, -2.7]).tolist()   # +90 deg, shifted
    test = _plan([_room("B", rot(b)), _room("A", rot(a))])
    c = ct.compare(ref, test)
    assert c["align"]["footprint_iou"] > 0.98
    assert {(r["ref"], r["test"]) for r in c["rooms"]} == {("A", "A"), ("B", "B")}
    assert c["adjacency"]["common"] == [("A", "B")]
    s = ct.summary(c)
    assert s["walls_matched"] == 8 and s["wall_len_median_abs_err_m"] < 1e-6
    assert c["overlap_test_m2"] < 1e-9
