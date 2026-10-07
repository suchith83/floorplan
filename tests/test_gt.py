"""eval/gt.py: GT loading, matching (dir / override / auto-length), and scoring on a synthetic two-room plan."""
import importlib.util
import sys
from pathlib import Path

import pytest

_EVAL = Path(__file__).parents[1] / "eval"
sys.path.insert(0, str(_EVAL))
spec = importlib.util.spec_from_file_location("gt", _EVAL / "gt.py")
gt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gt)


def _m(v, rel=0.02):
    return {"value": v, "lo": v * (1 - rel), "hi": v * (1 + rel), "unit": "m", "method": "t", "observed": True}


def _room(rid, name, poly, lengths, ceiling, openings=()):
    walls = [{"id": f"{rid}.W{k + 1}", "p0": poly[k], "p1": poly[(k + 1) % len(poly)], "length": _m(lengths[k]),
              "observed": True, "coverage": 1.0} for k in range(len(poly))]
    return {"id": rid, "name": name, "polygon": poly, "walls": walls, "openings": list(openings),
            "floor_area": _m(1.0), "perimeter": _m(sum(lengths)), "ceiling_height": _m(ceiling, 0.005)}


def _op(oid, kind, wall, width, center, rooms=None):
    return {"id": oid, "kind": kind, "wall_id": wall, "rooms": rooms, "width": _m(width),
            "height": None, "sill": None, "center": center}


def _plan():
    # R1 kitchen 4 x 3 at the origin, R2 (unnamed) 2 x 3 to its east; CCW polygons.
    # Walls: W1 bottom (-y), W2 right (+x), W3 top (+y), W4 left (-x).
    r1 = _room("R1", "Kitchen", [[0, 0], [4, 0], [4, 3], [0, 3]], [4.02, 3.01, 3.99, 2.98], 2.41,
               [_op("R1.O1", "door", "R1.W2", 0.80, [4, 1.5], ["R1", "R2"]),
                _op("R1.O2", "window", "R1.W3", 1.20, [2, 3])])
    r2 = _room("R2", "R2", [[4, 0], [6, 0], [6, 3], [4, 3]], [2.0, 3.0, 2.0, 3.0], 2.45,
               [_op("R2.O1", "window", "R2.W2", 1.00, [6, 1.5])])
    return {"capture_id": "synthetic", "tier": "lidar", "rooms": [r1, r2], "footprint_area": _m(18.0)}


GT_YAML = """
capture: synthetic
tool: test tape
rooms:
  - name: kitchen
    ceiling: [2.40, 2.41, 2.40]
    walls:
      - {name: south, length: [4.00, 4.01], dir: -y}
      - {name: north, length: 4.00, dir: +y}
      - {name: east, length: [3.00]}
      - {name: west, length: [3.00, 3.00, 3.00], dir: -x}
    openings:
      - {name: door, kind: door, wall: east, width: [0.81, 0.81]}
      - {name: window, kind: window, wall: north, width: 1.19}
      - {name: hatch, kind: window, wall: south, width: 0.5}
  - name: bedroom
    plan_room: R2
    ceiling: [2.50]
    walls:
      - {name: east, length: 3.0, dir: +x}
      - {name: far, length: 2.0, plan_wall: R2.W3}
cross_lines:
  - name: kitchen west to bedroom east
    from: {room: kitchen, wall: west}
    to: {room: bedroom, wall: east}
    length: [5.98, 5.99]
"""


@pytest.fixture
def gtfile(tmp_path):
    p = tmp_path / "synthetic.yaml"
    p.write_text(GT_YAML)
    return p


def test_load_normalises_readings(gtfile):
    g = gt.load_gt(gtfile)
    south = g["rooms"][0]["walls"][0]["length"]
    assert south["value"] == pytest.approx(4.005) and south["spread"] == pytest.approx(0.01)
    assert g["rooms"][0]["walls"][1]["length"]["readings"] == [4.0]


def test_outward_dir_both_windings():
    r = _plan()["rooms"][0]
    assert [gt.outward_dir(r, w) for w in r["walls"]] == ["-y", "+x", "+y", "-x"]
    w = dict(r["walls"][0], p0=[4, 0], p1=[0, 0])  # same wall stored the other way round
    assert gt.outward_dir(r, w) == "-y"


def test_match_dir_name_override_autolength(gtfile):
    m = gt.match(_plan(), gt.load_gt(gtfile))
    assert m["rooms"]["kitchen"] == {"plan": "R1", "how": "name"}
    assert m["rooms"]["bedroom"]["how"] == "override" and m["rooms"]["bedroom"]["plan"] == "R2"
    w = m["walls"]
    assert (w["kitchen/south"]["plan"], w["kitchen/south"]["how"]) == ("R1.W1", "dir")
    assert (w["kitchen/west"]["plan"], w["kitchen/west"]["how"]) == ("R1.W4", "dir")
    assert (w["kitchen/east"]["plan"], w["kitchen/east"]["how"]) == ("R1.W2", "auto-length")
    assert (w["bedroom/far"]["plan"], w["bedroom/far"]["how"]) == ("R2.W3", "override")
    assert w["bedroom/east"]["plan"] == "R2.W2"


def test_override_precedence(gtfile, tmp_path):
    o = tmp_path / "synthetic.match.yaml"
    o.write_text('rooms: {bedroom: R1}\nwalls: {"kitchen/south": R1.W3}\nopenings: {"kitchen/door": null}\n')
    ov = gt.load_overrides(o)
    m = gt.match(_plan(), gt.load_gt(gtfile), ov)
    assert m["rooms"]["bedroom"]["plan"] == "R2"            # plan_room in the GT beats the overrides file
    assert m["walls"]["kitchen/south"] == {"plan": "R1.W3", "how": "override"}  # file beats dir
    assert m["walls"]["kitchen/north"]["plan"] is None      # its only +y wall was taken by the override
    assert m["openings"]["kitchen/door"] == {"plan": None, "how": "override"}
    assert gt.load_overrides(tmp_path / "absent.yaml") == {"rooms": {}, "walls": {}, "openings": {}}


def test_unknown_override_id_is_an_error(gtfile, tmp_path):
    o = tmp_path / "x.match.yaml"
    o.write_text("rooms: {kitchen: R9}\n")
    with pytest.raises(gt.GTError, match="R9"):
        gt.match(_plan(), gt.load_gt(gtfile), gt.load_overrides(o))


def test_score_rows_and_summary(gtfile):
    plan, g = _plan(), gt.load_gt(gtfile)
    s = gt.score(plan, g, gt.match(plan, g))
    walls = {(r["gt_room"], r["gt_wall"]): r for r in s["walls"]}
    south = walls[("kitchen", "south")]
    assert south["err_m"] == pytest.approx(0.015) and south["in_interval"]
    # openings: door + window detected, hatch missed, R2.O1 is a phantom
    sm = s["summary"]["openings"]
    assert (sm["n_gt"], sm["n_detected"], sm["n_missed"], sm["n_phantom"]) == (3, 2, 1, 1)
    assert sm["n_width_ok"] == 2 and sm["frac_ok"] == pytest.approx(0.5) and sm["pass"] is False
    assert [p["plan_id"] for p in s["phantoms"]] == ["R2.O1"]
    # ceilings: kitchen 2.41 vs 2.4033 passes (<= 1.5 cm), bedroom 2.45 vs 2.50 fails
    c = {r["gt_room"]: r for r in s["ceilings"]}
    assert c["kitchen"]["pass"] and not c["bedroom"]["pass"]
    assert s["summary"]["ceilings"]["pass"] is False
    assert s["summary"]["ceilings"]["max_abs_err_m"] == pytest.approx(0.05)
    # cross line: x = 0 to x = 6
    x = s["cross_lines"][0]
    assert x["plan"] == pytest.approx(6.0) and x["err_m"] == pytest.approx(0.015)
    sw = s["summary"]["walls"]
    assert sw["n"] == 6 and sw["coverage"] == 1.0 and sw["frac_within_3pct"] == 1.0
    assert s["footprint"] is None                           # no area_m2 in this GT


def test_opening_kind_mismatch_is_a_miss(gtfile, tmp_path):
    p = tmp_path / "k.yaml"
    p.write_text(GT_YAML.replace("{name: door, kind: door", "{name: door, kind: passage"))
    plan, g = _plan(), gt.load_gt(p)
    m = gt.match(plan, g)
    assert m["openings"]["kitchen/door"]["plan"] is None and "kind differs" in m["openings"]["kitchen/door"]["note"]
    s = gt.score(plan, g, m)
    assert s["summary"]["openings"]["n_phantom"] == 2       # the unclaimed plan door is now a phantom too


def test_not_parallel_cross_line_is_an_error(gtfile, tmp_path):
    p = tmp_path / "np.yaml"
    p.write_text(GT_YAML.replace("to: {room: bedroom, wall: east}", "to: {room: bedroom, wall: far}"))
    plan, g = _plan(), gt.load_gt(p)
    with pytest.raises(gt.GTError, match="not parallel"):
        gt.score(plan, g, gt.match(plan, g))


def test_unmatched_room_is_reported(gtfile, tmp_path):
    p = tmp_path / "u.yaml"
    p.write_text(GT_YAML.replace("name: kitchen\n", "name: lounge\n").replace("room: kitchen", "room: lounge"))
    plan, g = _plan(), gt.load_gt(p)
    m = gt.match(plan, g)
    assert m["rooms"]["lounge"]["plan"] is None and m["walls"]["lounge/south"]["how"] == "unmatched"
    s = gt.score(plan, g, m)
    assert "room lounge" in s["unmatched"] and s["cross_lines"][0]["plan"] is None


def test_malformed_gt_names_the_field(tmp_path):
    cases = [
        (GT_YAML.replace("{name: north, length: 4.00, dir: +y}", "{name: north, dir: +y}"), "walls[north]: missing required field 'length'"),
        (GT_YAML.replace("dir: +y}", "dir: up}"), "walls[north].dir"),
        (GT_YAML.replace("length: 4.00, dir", "length: [4.0, 4.1, 4.0, 4.0], dir"), "1-3 readings"),
        (GT_YAML.replace("wall: north, width", "wall: nroth, width"), "'nroth' is not a wall of kitchen"),
        (GT_YAML.replace("kind: door,", "kind: doro,"), "openings[door].kind"),
        ("capture: x\nrooms: [\n", "not valid YAML"),
    ]
    for i, (text, msg) in enumerate(cases):
        p = tmp_path / f"bad{i}.yaml"
        p.write_text(text)
        with pytest.raises(gt.GTError) as e:
            gt.load_gt(p)
        assert msg in str(e.value), (msg, str(e.value))


def test_template_loads():
    g = gt.load_gt(_EVAL / "ground_truth" / "TEMPLATE.yaml")
    assert g["rooms"] and g["cross_lines"] and not g["warnings"]
