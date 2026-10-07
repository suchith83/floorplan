"""plan.json contract (fp/schema.py): the fixture validates, the exported schema is current,
and the rules reject what they should."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from fp.schema import (
    SCHEMA_PATH,
    Damage,
    Measure,
    Property,
    json_schema,
    measure,
    not_observed,
    validate,
)

FIXTURE = Path(__file__).parent / "fixtures" / "plan_two_rooms.json"


@pytest.fixture
def plan() -> dict:
    return json.loads(FIXTURE.read_text())


def test_fixture_validates_from_dict_str_and_path(plan):
    for src in (plan, FIXTURE.read_text(), FIXTURE, str(FIXTURE)):
        p = validate(src)
        assert isinstance(p, Property)
        assert [r.id for r in p.rooms] == ["R1", "R2"]


def test_fixture_round_trips(plan):
    p = validate(plan)
    out = p.model_dump(mode="json", by_alias=True, exclude_none=False)
    assert out == plan
    assert out["damage"][0]["class"] == "water_stain"
    assert "class_" not in out["damage"][0]
    assert p.dump() == plan
    assert validate(out).dump() == plan


def test_surface_ids(plan):
    ids = validate(plan).surface_ids()
    assert {"R1.W1", "R1.floor", "R1.ceiling", "R2.W6", "R2.floor", "R2.ceiling"} <= set(ids)
    assert len(ids) == 4 + 2 + 6 + 2


def test_schema_file_is_current():
    assert SCHEMA_PATH.is_file(), "run: uv run python -m fp.schema"
    assert json.loads(SCHEMA_PATH.read_text()) == json_schema(), "stale: run uv run python -m fp.schema"


def test_fixture_against_exported_json_schema(plan):
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.validate(plan, schema)
    bad = copy.deepcopy(plan)
    bad["rooms"][0]["walls"][0]["surprise"] = 1
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(bad, schema)


def test_empty_plan_is_valid(plan):
    empty = copy.deepcopy(plan)
    empty.update(
        rooms=[], connections=[], damage=[], concealed=[], scope=[],
        footprint_area=not_observed("m2", "rooms not built yet").model_dump(),
        warnings=[{"stage": "rooms", "message": "room stage not built yet"}],
    )
    p = validate(empty)
    assert p.rooms == [] and p.footprint_area.value is None and not p.footprint_area.observed
    assert p.surface_ids() == []


def test_measure_constructors_and_rules():
    m = measure(2.48, 2.46, 2.50, "m", "plane distance")
    assert m.observed and m.value == 2.48
    assert measure(2.5, 2.2, 2.8, "m", "assumed", observed=False).observed is False
    assert not_observed("m", "ceiling not captured").value is None
    with pytest.raises(ValidationError, match="lo <= value <= hi"):
        Measure(value=2.0, lo=2.1, hi=2.5, unit="m", method="x", observed=True)
    with pytest.raises(ValidationError, match="never invent"):
        Measure(value=None, lo=None, hi=None, unit="m", method="x", observed=True)
    with pytest.raises(ValidationError, match="lo and hi are required"):
        Measure(value=2.0, lo=None, hi=None, unit="m", method="x", observed=False)
    with pytest.raises(ValidationError):
        Measure(value=2.0, lo=1.9, hi=2.1, unit="ft", method="x", observed=True)
    with pytest.raises(ValidationError):
        Measure(value=float("nan"), lo=1.9, hi=2.1, unit="m", method="x", observed=True)


def test_damage_class_alias():
    kw = dict(id="D1", surface_id="R1.floor", position=None, evidence_image=None, score=0.5,
              extent_m2=measure(0.1, 0.05, 0.15, "m2", "mask"),
              bbox_on_surface={"u0": 0, "v0": 0, "u1": 1, "v1": 1})
    d = Damage(class_="crack", **kw)
    assert d.model_dump(mode="json")["class"] == "crack"
    assert Damage.model_validate({"class": "crack", **kw}).class_ == "crack"


# (case id, JSON path in the fixture, new value or a callable that edits the node, expected error)
REJECTIONS = [
    ("lo_above_value", ("rooms", 0, "floor_area", "lo"), 16.0, "lo <= value <= hi"),
    ("null_but_observed", ("rooms", 1, "ceiling_height", "observed"), True, "never invent"),
    ("unknown_damage_surface", ("damage", 0, "surface_id"), "R2.W9", "unknown surface_id R2.W9"),
    ("opening_on_other_rooms_wall", ("rooms", 0, "openings", 0, "wall_id"), "R2.W1",
     "wall_id R2.W1 is not a wall of R1"),
    ("duplicate_wall_id", ("rooms", 0, "walls", 3, "id"), "R1.W3", "duplicate wall id R1.W3"),
    ("extra_field", ("rooms", 0, "walls", 0, "surprise"), 1, "Extra inputs are not permitted"),
    ("extra_top_level_field", ("surprise",), 1, "Extra inputs are not permitted"),
    ("wall_under_wrong_room", ("rooms", 1, "walls", 0, "id"), "R1.W9",
     "wall R1.W9 is listed under room R2"),
    ("unknown_scope_surface", ("scope", 1, "surface_id"), "R3.floor", "unknown surface_id R3.floor"),
    ("scope_unit_mismatch", ("scope", 1, "unit"), "m", "!= quantity.unit"),
    ("connection_unknown_opening", ("connections", 0, "opening_id"), "R1.O9", "unknown opening R1.O9"),
    ("connection_unknown_room", ("connections", 0, "rooms"), ["R1", "R7"], "unknown room R7"),
    ("concealed_unknown_damage", ("concealed", 0, "damage_id"), "D9", "unknown damage_id D9"),
    ("opening_unknown_room", ("rooms", 0, "openings", 0, "rooms"), ["R1", "R5"], "unknown room R5"),
    ("duplicate_room_id", ("rooms", 1, "id"), "R1", "duplicate room id R1"),
    ("clockwise_polygon", ("rooms", 0, "polygon"), lambda poly: poly[::-1], "counter-clockwise"),
    ("closed_polygon", ("rooms", 0, "polygon"), lambda poly: poly + [poly[0]], "must not be closed"),
    ("wrong_unit_for_area", ("rooms", 0, "floor_area", "unit"), "m", "unit must be 'm2'"),
    ("bad_schema_version", ("schema_version",), "0.9", "Input should be '1.0'"),
    ("bad_point_arity", ("rooms", 0, "walls", 0, "p0"), [0.0, 0.0, 0.0], "at most 2 items"),
]


@pytest.mark.parametrize("path,new,match", [r[1:] for r in REJECTIONS], ids=[r[0] for r in REJECTIONS])
def test_rejections(plan, path, new, match):
    node = plan
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = new(node[path[-1]]) if callable(new) else new
    with pytest.raises(ValidationError, match=match):
        validate(plan)


def test_validate_rejects_non_json_string():
    with pytest.raises(ValueError):
        validate("no/such/plan.json")
