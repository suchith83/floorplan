"""The output contract from the CLI side: provisional intervals and a valid plan even when stages are missing."""
import json

import cv2
import numpy as np
import pytest

from fp import contract
from fp.cli import run


def _rect_room(spread_cm=1.0, w4_seen=True):
    """A 4 x 3 m room as fp.geometry.plan._room returns it."""
    pts = [[0, 0], [4, 0], [4, 3], [0, 3]]
    walls = [{"id": f"W{k + 1}", "p0": pts[k], "p1": pts[(k + 1) % 4], "length_m": [4, 3, 4, 3][k],
              "observed": k != 3 or w4_seen, "coverage": 0.8, "spread_cm": spread_cm} for k in range(4)]
    return {"id": "R1", "name": "R1", "polygon": pts, "walls": walls, "area_m2": 12.0, "ceiling_h_m": 2.5}


def test_provisional_intervals_follow_the_plane_position_model():
    r = contract.room_to_schema(_rect_room(), "lidar")
    pos = contract.POS_BASE + 0.01                     # 1 cm base + 1 cm wall-band thickness
    assert r["walls"][0]["length"]["hi"] - 4 == pytest.approx(2 * pos)   # two neighbouring planes
    assert r["floor_area"]["hi"] - 12 == pytest.approx(14 * pos)         # sum of length x pos
    assert r["perimeter"]["hi"] - 14 == pytest.approx(8 * pos)           # 2 x sum of pos
    assert r["ceiling_height"]["value"] == 2.5 and r["walls"][0]["id"] == "R1.W1"


def test_an_inferred_wall_widens_its_neighbours_and_marks_them_not_observed():
    r = contract.room_to_schema(_rect_room(w4_seen=False), "lidar")
    w1, w2 = r["walls"][0], r["walls"][1]               # W1 touches the inferred W4, W2 doesn't
    assert not w1["length"]["observed"] and w2["length"]["observed"]
    assert w1["length"]["hi"] - w1["length"]["value"] > w2["length"]["hi"] - w2["length"]["value"]
    assert not r["floor_area"]["observed"]


def test_camera_tiers_get_wider_intervals_than_lidar():
    half = lambda t: (lambda m: m["hi"] - m["value"])(contract.room_to_schema(_rect_room(), t)["walls"][0]["length"])
    assert half("lidar") < half("video") < half("photos")
    assert half("video") >= 0.03 * 4                    # at least the learned-scale term


def test_a_missing_ceiling_is_not_observed_not_invented():
    room = _rect_room()
    room["ceiling_h_m"] = None
    m = contract.room_to_schema(room, "lidar")["ceiling_height"]
    assert m["value"] is None and m["observed"] is False


def _fake_stray(d):
    (d / "depth").mkdir(parents=True)
    cv2.imwrite(str(d / "depth" / "000000.png"), np.zeros((192, 256), np.uint16))
    (d / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
    return d


def test_run_writes_a_valid_plan_even_when_the_capture_is_unreadable(tmp_path):
    from fp.schema import validate
    out = tmp_path / "out"
    plan = run(_fake_stray(tmp_path / "cap"), out)
    assert plan["tier"] == "lidar" and plan["rooms"] == []
    assert {w["stage"] for w in plan["warnings"]} >= {"ingest", "rooms"}
    assert plan["drift"]["method"].startswith("not run")     # no poses were used, so none "as-is"
    assert plan["footprint_area"]["observed"] is False and plan["footprint_area"]["value"] is None
    for f in ("plan.json", "plan.svg", "report.html"):
        assert (out / f).stat().st_size > 0
    validate(json.loads((out / "plan.json").read_text()))
    assert "total" in plan["timings"]


def test_a_video_whose_model_fails_still_gives_a_valid_plan(tmp_path, monkeypatch):
    """The depth model can fail (no weights offline, out of memory): the plan is still written, with a warning."""
    from fp.recon import camera

    def boom(*a, **k):
        raise RuntimeError("MPS backend out of memory")
    monkeypatch.setattr(camera, "_infer_local", boom)
    monkeypatch.chdir(tmp_path)
    vid = tmp_path / "walk.mp4"
    vw = cv2.VideoWriter(str(vid), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(30):
        vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    vw.release()
    plan = run(vid, tmp_path / "o", damage=False)
    assert plan["tier"] == "video" and plan["intervals"]["scale"] == contract.TIER_SCALE["video"]
    assert any(w["stage"] == "recon" for w in plan["warnings"])


def test_a_heic_only_folder_still_gives_a_valid_plan(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photos" / "kitchen").mkdir(parents=True)
    (tmp_path / "photos" / "kitchen" / "IMG_0001.HEIC").write_bytes(b"not really heic")
    plan = run(tmp_path / "photos", tmp_path / "o")
    assert plan["tier"] == "photos" and plan["rooms"] == []
    assert any(w["stage"] == "ingest" and "HEIC" in w["message"] for w in plan["warnings"])
    assert (tmp_path / "o" / "report.html").exists()


def test_an_empty_bundle_has_no_depth():
    from fp.bundle import CaptureBundle
    assert not CaptureBundle(source="photos", frames=[]).has_depth


def test_render_lists_every_field_error_instead_of_a_traceback(tmp_path):
    from pathlib import Path

    from fp.cli import render
    plan = json.loads(Path("tests/fixtures/plan_two_rooms.json").read_text())
    plan["rooms"][0]["walls"][0]["length"]["lo"] = 4.5           # lo above the value
    plan["rooms"][1]["ceiling_height"]["observed"] = True        # null value claimed as observed
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(plan))
    with pytest.raises(SystemExit) as e:
        render(bad, tmp_path / "o")
    msg = str(e.value)
    assert "does not match schema" in msg and "rooms.0.walls.0.length" in msg and "rooms.1.ceiling_height" in msg
    with pytest.raises(SystemExit, match="No such plan file"):
        render(tmp_path / "missing.json", None)


def test_the_report_badge_shows_the_interval_scale_plainly(tmp_path):
    from fp.report import write_outputs
    plan = json.loads(open("tests/fixtures/plan_two_rooms.json").read())
    write_outputs(plan, tmp_path)
    assert "intervals ×1<" in (tmp_path / "report.html").read_text()
