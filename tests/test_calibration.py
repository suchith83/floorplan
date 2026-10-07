"""Interval calibration (fp/calibration.py): one model, half = k*a + b*value, constants fitted by split conformal."""
import json

import pytest

from fp import calibration as cal
from fp import contract


@pytest.fixture(autouse=True)
def _fresh_cache():
    cal.reset_cache()
    yield
    cal.reset_cache()


def _rows(tier, typ, errs, a=0.02, value=3.0, fold="x"):
    return [{"tier": tier, "type": typ, "err": e, "a": a, "value": value, "fold": fold} for e in errs]


def test_conformal_rank_needs_nine_points_for_ninety_percent():
    assert cal.conformal_rank(8) is None and cal.conformal_rank(9) == 9 and cal.conformal_rank(19) == 18
    assert cal.quantile([1, 2, 3]) == (3, True)
    assert cal.quantile(list(range(1, 20))) == (18, False)


def test_lidar_fits_k_and_covers_its_own_evidence():
    errs = [0.01 * i for i in range(1, 21)]          # 1..20 cm, evidence a = 2 cm -> needed k = 0.5..10
    f = cal.fit_one(_rows("lidar", "wall_length", errs), "lidar", "wall_length")
    assert f["k"] == pytest.approx(9.5) and f["b"] == 0.0 and not f["small_n"] and f["n"] == 20
    consts = cal.provisional()
    consts["lidar"]["wall_length"] = f
    assert cal.coverage(_rows("lidar", "wall_length", errs), consts)["frac"] >= 0.9


def test_a_fit_never_narrows_below_provisional():
    f = cal.fit_one(_rows("lidar", "wall_length", [0.001] * 10), "lidar", "wall_length")
    assert f["k"] == 1.0 and f["floored"] and f["raw"] < 1.0
    g = cal.fit_one(_rows("photos", "wall_length", [0.0] * 10), "photos", "wall_length")
    assert g["b"] == cal.SCALE_REL["photos"]


def test_camera_tiers_fit_the_relative_term():
    errs = [0.3 * (i + 1) / 10 for i in range(10)]   # up to 30 % of a 3 m wall
    f = cal.fit_one(_rows("video", "wall_length", errs, a=0.0), "video", "wall_length")
    assert f["k"] == 1.0 and f["b"] == pytest.approx(0.1, abs=1e-3)   # rank ceil(11 x 0.9) = 10: 0.3 m / 3 m
    assert f["small_n"] is False and f["n"] == 10


def test_area_b_includes_the_factor_two():
    # a 10 % scale error changes a 10 m2 area by 2 m2: b for areas is per unit area, so 0.2
    rows = _rows("video", "floor_area", [2.0] * 9, a=0.0, value=10.0)
    assert cal.fit_one(rows, "video", "floor_area")["b"] == pytest.approx(0.2)


def test_two_fold_reports_held_out_coverage():
    rows = _rows("photos", "wall_length", [0.1] * 9, fold="c1") + _rows("photos", "wall_length", [0.5] * 9, fold="c2")
    out = {r["test_on"]: r for r in cal.two_fold(rows)}
    assert out["c1"]["coverage"] == 1.0          # fit on the bad capture covers the good one
    assert out["c2"]["coverage"] == 0.0          # fit on the good capture misses the bad one: reported, not hidden


def test_fitted_file_drives_the_contract(tmp_path, monkeypatch):
    consts = cal.provisional()
    consts["lidar"]["wall_length"].update(k=3.0, fitted=True, n=12)
    p = tmp_path / "calibration.json"
    p.write_text(json.dumps({"constants": consts, "source": "test"}))
    monkeypatch.setattr(cal, "PATH", p)
    cal.reset_cache()
    room = {"id": "R1", "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]], "area_m2": 12.0,
            "walls": [{"id": f"W{i}", "p0": [0, 0], "p1": [1, 0], "length_m": L, "observed": True, "coverage": 1.0,
                       "spread_cm": 1.0} for i, L in enumerate([4, 3, 4, 3], 1)]}
    r = contract.room_to_schema(room, "lidar")
    pos = contract.POS_BASE + 0.01
    assert r["walls"][0]["length"]["hi"] - 4 == pytest.approx(3 * 2 * pos)
    plan = contract.empty_plan("lidar", "x", source={})
    assert plan["intervals"]["calibrated"] and "wall_length" in plan["intervals"]["method"]
    assert plan["source"]["interval_model"]["wall_length"]["k"] == 3.0
    assert not contract.empty_plan("video", "x", source={})["intervals"]["calibrated"]


def test_lo_is_clipped_at_zero():
    m = contract.measure(0.5, 2.0, "m", "x")
    assert m["lo"] == 0.0 and m["hi"] == 2.5


def test_opening_evidence_is_scaled_for_the_tier_once():
    plan = contract.empty_plan("video", "x", source={})
    plan["rooms"] = [{"id": "R1", "openings": []}]
    w = {"value": 0.9, "half": 0.06, "observed": True, "method": "jambs"}   # already x tier scale (openings.py)
    contract.add_openings(plan, [{"room": "R1", "rooms": None, "kind": "door", "wall_id": "R1.W1", "width": w,
                                  "height": None, "sill": None, "center": [0, 0]}])
    width = plan["rooms"][0]["openings"][0]["width"]
    assert width["hi"] - 0.9 == pytest.approx(0.06 + 0.03 * 0.9)
