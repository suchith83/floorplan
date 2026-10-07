"""Damage placement, metric extent, concealed rules and scope on synthetic inputs (no model, no data):
masks are drawn on rendered depth frames of a known scene, so surface and extent have exact answers."""
import cv2
import numpy as np
import pytest

from fp import contract
from fp.bundle import Frame
from fp.damage import detect_local, project
from fp.damage.rules import concealed
from fp.schema import validate
from fp.scope import build
from test_openings import F, H, W, _look, _rect, _render


def _plan(ceiling=2.5):
    """R1 x 0-4, y 0-3 and R2 x 4.1-6 (wet), a 0.9 m door in the shared wall at y 1.0-1.9."""
    def room(rid, x0, x1):
        pts = [[x0, 0.0], [x1, 0.0], [x1, 3.0], [x0, 3.0]]
        walls = [{"id": f"W{k + 1}", "p0": pts[k], "p1": pts[(k + 1) % 4], "observed": True,
                  "length_m": float(np.hypot(*np.subtract(pts[(k + 1) % 4], pts[k]))), "coverage": 1.0,
                  "spread_cm": 1.0} for k in range(4)]
        return {"id": rid, "name": rid, "polygon": pts, "walls": walls, "area_m2": (x1 - x0) * 3.0}
    plan = contract.empty_plan("lidar", "t", source={"path": "nowhere"})
    door = {"room": "R1", "wall_id": "R1.W2", "kind": "door", "rooms": ["R1", "R2"],
            "width": {"value": 0.9, "half": 0.02, "observed": True, "method": "jambs"},
            "height": {"value": 2.05, "half": 0.04, "observed": True, "method": "lintel"}, "sill": None,
            "center": [4.0, 1.45]}
    ceil = {r: {"h": ceiling, "half": 0.012, "method": "layer", "reason": None} for r in ("R1", "R2")}
    contract.fill_from_geometry(plan, [room("R1", 0.0, 4.0), room("R2", 4.1, 6.0)], [], [door], ceil)
    plan["frame"]["T_plan_world"] = np.eye(4).tolist()
    plan["timings"]["total"] = 0.0
    return plan


SCENE = [_rect(1, 0.0, +1, (0, 4), (0, 2.5)), _rect(2, 0.0, +1, (0, 4), (0, 3)), _rect(2, 2.5, -1, (0, 4), (0, 3)),
         _rect(0, 0.0, +1, (0, 3), (0, 2.5)), _rect(0, 4.0, -1, (0, 3), (0, 2.5)), _rect(1, 3.0, -1, (0, 4), (0, 2.5))]


def _frame(tmp_path, c, yaw, pitch, scene=SCENE, name="f"):
    R = _look(yaw, pitch)
    D, _, _ = _render(scene, np.array(c, float), R)
    rgb, dp = tmp_path / f"{name}.jpg", tmp_path / f"{name}.png"
    cv2.imwrite(str(rgb), np.zeros((H, W, 3), np.uint8))
    cv2.imwrite(str(dp), D)
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c
    return Frame(rgb=rgb, K=np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]]), T_wc=T, depth=dp, timestamp=0.0)


def _mask_of(frame, corners, scale=4):
    """Mask (at scale x the image size, like a full-resolution frame) of a plan-frame polygon."""
    Tcp = np.linalg.inv(frame.T_wc)
    C = np.asarray(corners, float) @ Tcp[:3, :3].T + Tcp[:3, 3]
    uv = (C @ frame.K.T)[:, :2] / C[:, 2:3] * scale
    m = np.zeros((H * scale, W * scale), np.uint8)
    cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 1)
    return m.astype(bool)


STAIN = [[1.9, 0.0, 1.1], [2.1, 0.0, 1.1], [2.1, 0.0, 1.3], [1.9, 0.0, 1.3]]   # 0.20 x 0.20 m on R1.W1 (y = 0)


@pytest.mark.parametrize("yaw", [-np.pi / 2, -np.pi / 2 + 0.6])      # face-on, and 34 deg off
def test_a_stain_lands_on_its_wall_with_its_true_area(tmp_path, yaw):
    plan = _plan()
    f = _frame(tmp_path, (2.0 - 1.2 * np.sin(yaw + np.pi / 2), 1.6, 1.3), yaw, -0.1)
    p = project.place(f, _mask_of(f, STAIN), np.eye(4), plan)
    assert p["surface"]["id"] == "R1.W1"
    assert abs(p["extent_m2"] - 0.04) < 0.006                 # 15 %: mask edges at depth resolution
    u0, v0 = np.percentile(p["uv"], 5, axis=0)
    assert abs(u0 - 1.9) < 0.04 and abs(v0 - 1.1) < 0.04       # along the wall from p0, height above the floor


def test_a_mask_on_furniture_in_front_of_the_wall_is_dropped(tmp_path):
    box = SCENE + [_rect(1, 0.6, +1, (1.5, 2.5), (0.0, 1.6))]  # a cupboard front 0.6 m from the wall
    f = _frame(tmp_path, (2.0, 2.0, 1.3), -np.pi / 2, -0.1, box)
    front = [[1.9, 0.6, 1.1], [2.1, 0.6, 1.1], [2.1, 0.6, 1.3], [1.9, 0.6, 1.3]]
    assert project.place(f, _mask_of(f, front), np.eye(4), _plan())["drop"] == "no_surface"


def test_a_ceiling_mask_snaps_to_the_rooms_ceiling(tmp_path):
    f = _frame(tmp_path, (2.0, 1.5, 1.3), -np.pi / 2, 1.2)
    patch = [[1.8, 0.8, 2.5], [2.2, 0.8, 2.5], [2.2, 1.2, 2.5], [1.8, 1.2, 2.5]]
    p = project.place(f, _mask_of(f, patch), np.eye(4), _plan())
    assert p["surface"]["id"] == "R1.ceiling" and abs(p["extent_m2"] - 0.16) < 0.025


def test_upright_turns_put_the_floor_at_the_bottom():
    R = _look(0.0, 0.0)                                        # level camera: already upright
    assert project.upright_k(R) == 0
    roll = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])      # camera rolled 90 deg about its optical axis
    k = project.upright_k(R @ roll)
    d = (R @ roll).T @ [0, 0, -1.0]                            # plan-down in camera coords
    img = np.zeros((4, 6)); img[int(2 + np.sign(d[1])), int(3 + 2 * np.sign(d[0]))] = 1   # a dot at "down"
    r = np.rot90(img, k)
    assert np.argwhere(r)[0][0] > r.shape[0] / 2               # the dot is in the bottom half after turning


def test_labels_map_to_one_class_or_none():
    assert detect_local.label_class("water stain") == "water_stain"
    assert detect_local.label_class("hole in wall") == "hole"
    assert detect_local.label_class("mold stain") == "mold"
    assert detect_local.label_class("wall") is None and detect_local.label_class("stain hole") is None


def test_detections_replay_from_the_cache(tmp_path, monkeypatch):
    calls = []
    fake = {"boxes": np.array([[1, 2, 3, 4]], np.float32), "scores": np.array([0.5], np.float32),
            "classes": np.array(["crack"]), "masks": np.eye(8, dtype=bool)[None]}
    monkeypatch.setattr(detect_local, "_infer", lambda rgb: calls.append(1) or fake)
    img = np.full((8, 8, 3), 7, np.uint8)
    a = detect_local.detect([img], tmp_path)[0]
    b = detect_local.detect([img], tmp_path)[0]
    assert len(calls) == 1 and (a["masks"] == b["masks"]).all() and list(b["classes"]) == ["crack"]


def _damage(plan, items):
    plan["damage"] = [{"id": f"D{k + 1}", "class": c, "surface_id": s, "position": list(p),
                       "extent_m2": contract.measure(e, 0.2 * e, "m2", "mask area", True),
                       "bbox_on_surface": {"u0": 0, "v0": 0, "u1": 0.1, "v1": 0.1}, "evidence_image": None,
                       "score": 0.5} for k, (c, s, p, e) in enumerate(items)]
    return plan


def test_rules_fire_with_their_ids_and_are_hypotheses():
    plan = _damage(_plan(), [("water_stain", "R1.W1", (2.0, 0.0, 0.3), 0.05),       # low wall stain: C2
                             ("crack", "R1.W2", (4.0, 1.9, 1.9), 0.01),             # by the door: C4
                             ("mold", "R1.ceiling", (1.0, 1.0, 2.5), 0.2),          # ceiling: C1
                             ("water_stain", "R1.W2", (4.0, 0.5, 2.35), 0.03)])     # high, wall shared with R2
    flags = concealed(plan["damage"], plan, wet_rooms={"R2"})
    got = {(f["damage_id"], f["rule_id"]) for f in flags}
    assert got == {("D1", "C2"), ("D2", "C4"), ("D3", "C1"), ("D3", "C5"), ("D4", "C3"), ("D4", "C5")}
    assert all(f["label"] == "hypothesis" and f["rule_text"] for f in flags)
    assert not {f["rule_id"] for f in concealed(plan["damage"], plan)} & {"C3", "C5"}   # no wet room known


def test_scope_items_reference_real_surfaces_and_carry_intervals():
    plan = _damage(_plan(), [("crack", "R1.W2", (4.0, 1.9, 1.9), 0.01), ("hole", "R1.W2", (4.0, 2.5, 1.0), 0.01),
                             ("mold", "R1.ceiling", (1.0, 1.0, 2.5), 0.2),
                             ("water_stain", "R2.floor", (5.0, 1.0, 0.0), 0.3)])
    plan["concealed"] = concealed(plan["damage"], plan)
    plan["scope"] = build(plan)
    validate(plan)                                              # every surface_id exists, units match
    by = {(s["surface_id"], s["action"]): s["quantity"] for s in plan["scope"]}
    wall = by[("R1.W2", "repaint wall (prepare, 2 coats)")]
    assert abs(wall["value"] - (3.0 * 2.5 - 0.9 * 2.05)) < 1e-6   # length x height - the door
    assert wall["lo"] < wall["value"] < wall["hi"]
    assert by[("R1.W2", "patch and fill cracks / holes")]["value"] == 2
    assert by[("R1.ceiling", "repaint ceiling")]["value"] == 12.0
    sk = by[("R2.floor", "replace skirting")]
    assert abs(sk["value"] - (2 * (1.9 + 3.0) - 0.9)) < 1e-6      # perimeter - the door into R1


def test_a_door_is_taken_off_both_faces_of_its_partition_and_an_assumed_height_is_not_observed():
    plan = _damage(_plan(), [("crack", "R2.W4", (4.1, 1.0, 1.0), 0.01), ("crack", "R1.W2", (4.0, 2.5, 1.0), 0.01)])
    by = {s["surface_id"]: s["quantity"] for s in build(plan) if s["action"].startswith("repaint wall")}
    assert abs(by["R2.W4"]["value"] - (3.0 * 2.5 - 0.9 * 2.05)) < 1e-6       # R1's door is in R2's face too
    assert by["R2.W4"]["value"] == by["R1.W2"]["value"]
    plan["rooms"][0]["openings"][0]["height"] = contract.not_observed("m", "lintel not seen")
    by = {s["surface_id"]: s["quantity"] for s in build(plan) if s["action"].startswith("repaint wall")}
    assert by["R1.W2"]["observed"] is False                                   # door height assumed, not measured


def test_c5_fires_on_a_ceiling_stain_next_to_a_wet_room():
    plan = _damage(_plan(), [("water_stain", "R1.ceiling", (3.5, 1.5, 2.5), 0.05)])
    rules = {f["rule_id"] for f in concealed(plan["damage"], plan, wet_rooms={"R2"})}
    assert rules == {"C1", "C5"}
