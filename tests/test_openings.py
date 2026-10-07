"""Openings from wall elevations on synthetic scenes: depth maps rendered by a tiny ray caster (axis-aligned
rectangles), so ray casting, jamb planes, classification and the mirror/recess tests run end to end."""
import cv2
import numpy as np
from scipy.spatial import cKDTree

from fp import contract
from fp.bundle import Frame
from fp.geometry import openings as O

W, H, F = 256, 192, 200.0          # depth image and focal length (px): ~65 deg across, like the iPhone LiDAR


def _rect(axis, coord, normal, r1, r2):
    """A rectangle on plane x|y|z = coord; r1, r2 = ranges on the other two axes in order; normal = +-1."""
    return (axis, coord, normal, r1, r2)


def _scene(hole_y=(1.0, 1.9), hole_z=(0.0, 2.05), recess=None):
    """Room R1 x 0-3, room R2 x 3.1-6, y 0-3, ceiling 2.5; a 10 cm partition (faces x = 3.0 and 3.1) with a hole.
    recess: depth (m) of a panel closing the hole behind the R1 face (a closed door / niche)."""
    (y0, y1), (z0, z1) = hole_y, hole_z
    S = [_rect(2, 0.0, +1, (0, 6), (0, 3)), _rect(2, 2.5, -1, (0, 6), (0, 3)),
         _rect(1, 0.0, +1, (0, 6), (0, 2.5)), _rect(1, 3.0, -1, (0, 6), (0, 2.5)),
         _rect(0, 0.0, +1, (0, 3), (0, 2.5)), _rect(0, 6.0, -1, (0, 3), (0, 2.5))]
    for x, nrm in ((3.0, -1), (3.1, +1)):        # partition faces around the hole
        S += [_rect(0, x, nrm, (0, y0), (0, 2.5)), _rect(0, x, nrm, (y1, 3), (0, 2.5)),
              _rect(0, x, nrm, (y0, y1), (z1, 2.5)), _rect(0, x, nrm, (y0, y1), (0, z0))]
    S += [_rect(1, y0, +1, (3.0, 3.1), (z0, z1)), _rect(1, y1, -1, (3.0, 3.1), (z0, z1)),   # reveals (jambs)
          _rect(2, z1, -1, (3.0, 3.1), (y0, y1))]                                          # head (soffit)
    if z0 > 0:
        S.append(_rect(2, z0, +1, (3.0, 3.1), (y0, y1)))                                     # window sill
    if recess is not None:
        S.append(_rect(0, 3.0 + recess, -1, (y0, y1), (z0, z1)))
    return S


def _render(S, c, R):
    """Depth (mm, uint16) and world points/normals for a camera at c with rotation R (camera -> world, OpenCV)."""
    v, u = np.mgrid[0:H, 0:W]
    d = np.stack([(u - W / 2) / F, (v - H / 2) / F, np.ones_like(u, float)], -1).reshape(-1, 3) @ R.T
    best = np.full(len(d), np.inf)
    nrm = np.zeros((len(d), 3))
    for ax, coord, n, r1, r2 in S:
        o1, o2 = [k for k in range(3) if k != ax]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (coord - c[ax]) / d[:, ax]
            p1, p2 = c[o1] + t * d[:, o1], c[o2] + t * d[:, o2]
        ok = (t > 1e-6) & (t < best) & (p1 >= r1[0]) & (p1 <= r1[1]) & (p2 >= r2[0]) & (p2 <= r2[1])
        best[ok] = t[ok]
        nrm[ok] = 0
        nrm[ok, ax] = n
    z = np.where(np.isfinite(best), best, 0.0)        # camera-frame depth = t, since d has unit z in camera frame
    pts = c + best[:, None] * d
    keep = np.isfinite(best)
    return (z.reshape(H, W) * 1000).astype(np.uint16), pts[keep], nrm[keep]


def _look(yaw, pitch):
    """Camera -> world rotation looking along +x rotated by yaw (about z) and pitch (up positive)."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    r = np.cross(f, [0, 0, 1.0]); r /= np.linalg.norm(r)
    return np.stack([r, np.cross(f, r), f], 1)       # columns: camera x (right), y (down), z (forward)


def _capture(tmp_path, S, cams):
    rgb = tmp_path / "rgb.jpg"
    cv2.imwrite(str(rgb), np.zeros((H, W, 3), np.uint8))
    K = np.array([[F, 0, W / 2], [0, F, H / 2], [0, 0, 1]])
    frames, P, N = [], [], []
    for k, (c, yaw, pitch) in enumerate(cams):
        R = _look(yaw, pitch)
        D, p, n = _render(S, np.array(c, float), R)
        dp = tmp_path / f"{k:06d}.png"
        cv2.imwrite(str(dp), D)
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = c
        frames.append(Frame(rgb=rgb, K=K, T_wc=T, depth=dp, timestamp=float(k)))
        P.append(p[::3]); N.append(n[::3])
    return frames, np.concatenate(P), np.concatenate(N)


def _rooms():
    def room(rid, x0, x1):
        pts = [[x0, 0.0], [x1, 0.0], [x1, 3.0], [x0, 3.0]]
        walls = [{"id": f"W{k + 1}", "p0": pts[k], "p1": pts[(k + 1) % 4], "observed": True} for k in range(4)]
        return {"id": rid, "polygon": pts, "walls": walls}
    return [room("R1", 0.0, 3.0), room("R2", 3.1, 6.0)]


CAMS = [((x, y, 1.4), yaw, pitch) for x in (0.8, 1.4) for y in (1.0, 1.5, 2.0)
        for yaw in (-0.3, 0.0, 0.3) for pitch in (-0.35, 0.0, 0.3)]


def test_a_door_is_found_once_measured_between_its_jambs_and_links_both_rooms(tmp_path):
    frames, P, N = _capture(tmp_path, _scene(), CAMS)
    ops, info = O.detect(_rooms(), frames, np.eye(4), P, N, 0.01, {"R1": 2.5, "R2": 2.5}, tmp_path / "dbg")
    doors = [o for o in ops if o["kind"] == "door"]
    assert len(doors) == 1, [(o["wall_id"], o["kind"], o["width"]["value"]) for o in ops]
    d = doors[0]
    assert abs(d["width"]["value"] - 0.90) < 0.02          # jamb planes, not the cell-rounded gap
    assert d["width"]["observed"] and "reveal" in d["width"]["method"]
    assert abs(d["height"]["value"] - 2.05) < 0.05
    assert sorted(d["rooms"]) == ["R1", "R2"]
    assert (tmp_path / "dbg" / f"{d['wall_id']}.png").exists()


def test_a_hole_with_a_sill_is_a_window_with_no_rooms(tmp_path):
    frames, P, N = _capture(tmp_path, _scene(hole_y=(1.0, 2.2), hole_z=(0.9, 2.0)), CAMS)
    ops, _ = O.detect(_rooms(), frames, np.eye(4), P, N, 0.01, {"R1": 2.5, "R2": 2.5})
    win = [o for o in ops if o["kind"] == "window"]
    assert len(win) == 1 and win[0]["rooms"] is None
    assert abs(win[0]["width"]["value"] - 1.20) < 0.03
    assert abs(win[0]["sill"]["value"] - 0.9) < 0.05


def test_a_door_leaf_set_back_in_its_frame_is_a_closed_door_and_a_niche_is_nothing(tmp_path):
    frames, P, N = _capture(tmp_path, _scene(recess=0.08), CAMS)   # leaf 8 cm back (5 cm = HIT_TOL is "wall")
    ops, _ = O.detect(_rooms(), frames, np.eye(4), P, N, 0.01, {"R1": 2.5, "R2": 2.5})
    assert [o["kind"] for o in ops] == ["door"] and "closed door" in ops[0]["width"]["method"]
    frames, P, N = _capture(tmp_path, _scene(hole_y=(1.0, 2.2), hole_z=(0.9, 2.0), recess=0.08), CAMS)
    ops, info = O.detect(_rooms(), frames, np.eye(4), P, N, 0.01, {"R1": 2.5, "R2": 2.5})
    assert ops == [] and any(c.get("why", "").startswith("recess") for c in info["rejected"])


def test_a_wall_hidden_behind_furniture_is_unobserved_not_an_opening(tmp_path):
    S = _scene(hole_y=(1.0, 1.0001), hole_z=(0.0, 0.0001))          # no hole
    S.append(_rect(0, 2.5, -1, (0.8, 2.0), (0, 2.2)))                  # a wardrobe front 0.5 m from the wall
    S += [_rect(1, 0.8, -1, (2.5, 3.0), (0, 2.2)), _rect(1, 2.0, +1, (2.5, 3.0), (0, 2.2))]
    frames, P, N = _capture(tmp_path, S, [c for c in CAMS if c[0][0] == 0.8])
    ops, info = O.detect(_rooms(), frames, np.eye(4), P, N, 0.01, {"R1": 2.5, "R2": 2.5})
    assert ops == []
    st = info["states"]["R1.W2"]                       # x = 3 wall of R1, u = y
    iu = int((1.4 + O.U_MARGIN) / O.CELL)
    assert (st[iu, int(1.0 / O.CELL)] == 0)            # behind the wardrobe: not seen, and not called empty


def test_mirror_test_sees_the_room_reflected_behind_the_wall():
    w = {"p0": np.array([3.0, 0.0]), "u": np.array([0.0, 1.0]), "n": np.array([1.0, 0.0])}
    rng = np.random.default_rng(1)
    room = np.column_stack([rng.uniform(0, 3, 6000), rng.uniform(0, 3, 6000), rng.uniform(0, 2.5, 6000)])
    virtual = room[room[:, 0] > 1.5].copy()
    virtual[:, 0] = 6.0 - virtual[:, 0]                 # the room again, mirrored across x = 3
    other = np.column_stack([rng.uniform(3.2, 6, 3000), rng.uniform(0, 3, 3000), rng.uniform(0, 2.5, 3000)])
    for behind, expect_mirror in ((virtual, True), (other, False)):
        P = np.vstack([room, behind])
        N = np.zeros_like(P)
        m, _ = O.behind_tests(P, N, cKDTree(P), w, 1.0, 2.0, 0.9, 2.0)
        assert (m >= O.MIRROR_FRAC) == expect_mirror, m


def test_openings_become_schema_openings_and_replace_the_neck_connection():
    plan = contract.empty_plan("lidar", "t", source={})
    rooms = [{"id": r["id"], "name": r["id"], "polygon": r["polygon"], "area_m2": 9.0,
              "walls": [dict(w, length_m=3.0, coverage=1.0, spread_cm=1.0) for w in r["walls"]]} for r in _rooms()]
    op = {"room": "R1", "wall_id": "R1.W2", "kind": "door", "rooms": ["R1", "R2"],
          "width": {"value": 0.9, "half": 0.02, "observed": True, "method": "jambs"},
          "height": {"value": 2.05, "half": 0.04, "observed": True, "method": "lintel"}, "sill": None,
          "center": [3.0, 1.45]}
    neck = {"rooms": ["R1", "R2"], "kind": "door", "width_m": 0.8, "center": [3.05, 1.4]}
    ceil = {"R1": {"h": 2.5, "half": 0.012, "method": "layer", "reason": None},
            "R2": {"h": None, "half": None, "method": "layer", "reason": "no layer"}}
    contract.fill_from_geometry(plan, rooms, [neck], [op], ceil)
    plan["timings"]["total"] = 0.0
    from fp.schema import validate
    validate(plan)
    assert plan["connections"] == [{"rooms": ["R1", "R2"], "opening_id": "R1.O1"}]
    assert plan["rooms"][0]["openings"][0]["width"]["value"] == 0.9
    assert plan["rooms"][0]["ceiling_height"]["value"] == 2.5
    assert plan["rooms"][1]["ceiling_height"]["observed"] is False
