"""Renderer tests on the hand-written schema 1.0 fixture (no geometry code, no data needed)."""
from __future__ import annotations

import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from fp.report import write_outputs
from fp.report.html import render_report
from fp.report.svg import fmt_len, render

FIXTURE = Path(__file__).parent / "fixtures" / "plan_two_rooms.json"
NS = "{http://www.w3.org/2000/svg}"


@pytest.fixture
def plan() -> dict:
    return json.loads(FIXTURE.read_text())


def _parse(svg: str):
    root = ET.fromstring(svg)
    texts = ["".join(t.itertext()) for t in root.iter(f"{NS}text")]
    return root, texts


def _empty(plan: dict) -> dict:
    p = copy.deepcopy(plan)
    for k in ("rooms", "connections", "damage", "concealed", "scope"):
        p[k] = []
    p["footprint_area"] = {"value": None, "lo": None, "hi": None, "unit": "m2",
                           "method": "no rooms", "observed": False}
    return p


def test_fmt_len():
    assert fmt_len({"value": 3.42, "lo": 3.40, "hi": 3.45, "unit": "m"}) == "3.42 m ±0.03"
    assert fmt_len({"value": None, "lo": None, "hi": None, "unit": "m"}) == "? m"
    assert fmt_len(None) == "? m"


def test_svg_fixture_content(plan):
    root, texts = _parse(render(plan))
    joined = "\n".join(texts)
    for r in plan["rooms"]:
        assert r["name"] in texts
    assert "15.12 m² ±0.27" in texts and "10.01 m² ±0.21" in texts
    assert "h 2.48 m ±0.02" in texts and "h not observed" in texts
    assert "4.20 m ±0.05" in texts  # R1.W1: bounded by the inferred R1.W4, so wider
    assert "3.60 m ±0.02" in texts  # R1.W4 unseen, but its length is between two seen planes
    assert "door 0.85 m ±0.02" in texts and "window 1.20 m ±0.03" in texts
    assert "1 m" in texts and "+y (plan)" in joined
    assert "D1" in texts  # damage pin

    arcs = [e for e in root.iter(f"{NS}path") if e.get("class") == "door-arc"]
    assert len(arcs) == 1 and " A " in arcs[0].get("d")
    windows = [e for e in root.iter(f"{NS}line") if e.get("class") == "window"]
    assert len(windows) == 2  # thin double line across the wall thickness
    dashed = [e for e in root.iter() if e.get("data-wall") == "R1.W4" and e.get("stroke-dasharray")]
    assert dashed, "inferred wall R1.W4 must be drawn dashed"
    solid = [e for e in root.iter() if e.get("data-wall") == "R1.W1"]
    assert solid and not any(e.get("stroke-dasharray") for e in solid)


def test_svg_wall_labels_one_per_wall(plan):
    """Shared wall R1.W2 / R2.W6: one label each, in different places (no stacking)."""
    root, _ = _parse(render(plan))
    labels = [(e.get("x"), e.get("y")) for e in root.iter(f"{NS}text") if e.get("class") == "wall-label"]
    n_walls = sum(len(r["walls"]) for r in plan["rooms"])
    assert len(labels) == n_walls
    assert len(set(labels)) == n_walls


def test_svg_scale_bar_is_true_to_scale(plan):
    """The 1 m segment of the scale bar is as long as 1 m of plan: compare with the 4.2 m room."""
    root, _ = _parse(render(plan))
    seg = next(e for e in root.iter(f"{NS}rect") if e.get("height") == "6")
    floor = next(e for e in root.iter(f"{NS}polygon") if e.get("data-room") == "R1")
    xs = [float(p.split(",")[0]) for p in floor.get("points").split()]
    assert float(seg.get("width")) == pytest.approx((max(xs) - min(xs)) / 4.2, abs=0.2)


def test_svg_deterministic(plan):
    assert render(plan) == render(copy.deepcopy(plan))


def test_svg_empty_plan(plan):
    root, texts = _parse(render(_empty(plan)))
    assert "No rooms reconstructed" in texts
    assert "1 m" in texts


def test_svg_tolerates_nulls_and_bad_refs(plan):
    p = copy.deepcopy(plan)
    r1 = p["rooms"][0]
    r1["walls"][0]["length"] = {"value": None, "lo": None, "hi": None, "unit": "m", "method": "x", "observed": False}
    r1["openings"].append({"id": "R1.O9", "kind": "door", "wall_id": "R7.W1", "rooms": None, "center": [1, 1],
                           "width": {"value": 0.8, "lo": 0.7, "hi": 0.9, "unit": "m", "method": "x", "observed": True},
                           "height": None, "sill": None})
    r1["openings"][1]["width"] = {"value": None, "lo": None, "hi": None, "unit": "m", "method": "x", "observed": False}
    p["damage"].append({**p["damage"][0], "id": "D2", "position": None, "evidence_image": None, "score": None})
    p["rooms"].append({"id": "R3", "name": "Cupboard", "polygon": [], "walls": [], "openings": [],
                       "floor_area": None, "ceiling_height": None, "perimeter": None})
    root, texts = _parse(render(p))
    assert "? m" in texts
    assert "window ? m" in texts
    render_report(p, Path("/nonexistent"))  # must not raise either


def test_write_outputs(plan, tmp_path):
    paths = write_outputs(plan, tmp_path / "o")
    assert set(paths) == {"plan.svg", "report.html"}
    assert all(p.exists() for p in paths.values())
    ET.fromstring(paths["plan.svg"].read_text())
    html = paths["report.html"].read_text()
    # R2's ceiling is not observed
    r2_row = next(row for row in html.split("<tr>") if "Bedroom" in row and "m²" in row)
    assert "not observed" in r2_row
    assert "Hypothesis (rule C2)" in html
    assert "evidence image missing" in html
    assert '<span class="badge tier">LiDAR</span>' in html
    assert "90% intervals" in html and "provisional" in html
    assert "inferred / assumed" in html  # S1 quantity and R1.W1/W3 lengths are not observed
    assert "<svg" in html and "http://" not in html.replace('xmlns="http://www.w3.org/2000/svg"', "")
    assert render_report(plan, tmp_path / "o") == html  # deterministic


def test_report_embeds_evidence(plan, tmp_path):
    from PIL import Image

    (tmp_path / "damage").mkdir()
    Image.new("RGB", (8, 8), (200, 50, 50)).save(tmp_path / "damage" / "D1.jpg")
    html = render_report(plan, tmp_path)
    assert "data:image/jpeg;base64," in html
    assert "evidence image missing" not in html


def test_report_empty_plan(plan, tmp_path):
    html = write_outputs(_empty(plan), tmp_path)["report.html"].read_text()
    assert "No rooms reconstructed" in html and "No damage reported." in html


def test_intervals_are_printed_rounded_outward():
    from fp.report.html import measure
    from fp.report.svg import fmt_len
    m = {"value": 2.48, "lo": 2.465, "hi": 2.495, "unit": "m", "method": "x", "observed": True}
    assert "[2.46–2.50]" in measure(m)                       # never narrower than [2.465, 2.495]
    assert fmt_len({**m, "lo": 2.4649, "hi": 2.48}) == "2.48 m ±0.02"
    assert fmt_len({**m, "lo": 2.4549, "hi": 2.48}) == "2.48 m ±0.03"   # 2.51 cm rounds up, not down
