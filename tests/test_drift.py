"""Drift correction on synthetic submaps (no capture data): loop recovery, false-loop rejection,
the pose graph, edge cases, and the double-wall metric."""
from pathlib import Path

import numpy as np

from fp.bundle import Frame
from fp.recon import drift
from fp.recon.drift import Submap

UP = np.array([0, 1.0, 0])   # world is y-up, as ARKit / Stray Scanner


def _plane(origin, u, v, nu, nv, normal, step=0.03):
    a, b = np.meshgrid(np.arange(0, nu, step), np.arange(0, nv, step))
    P = origin + a.reshape(-1, 1) * u + b.reshape(-1, 1) * v
    return P, np.repeat(np.asarray(normal, float)[None], len(P), 0)


def _room(w=4.0, d=5.0, h=2.5, x0=0.0, z0=0.0, furniture=True):
    """A y-up box room with inward normals (floor, 4 walls), plus a cabinet so yaw is well pinned."""
    X, Y, Z = np.eye(3)
    o = np.array([x0, 0, z0])
    parts = [_plane(o, X, Z, w, d, Y),                          # floor
             _plane(o, X, Y, w, h, Z), _plane(o + d * Z, X, Y, w, h, -Z),
             _plane(o, Z, Y, d, h, X), _plane(o + w * X, Z, Y, d, h, -X)]
    if furniture:
        c = o + np.array([0.5, 0, 0.0])
        parts += [_plane(c + 0.6 * Z, X, Y, 1.2, 0.9, Z), _plane(c + 1.2 * X, Z, Y, 0.6, 0.9, X),
                  _plane(c + 0.9 * Y, X, Z, 1.2, 0.6, Y)]
    return np.concatenate([p for p, _ in parts]), np.concatenate([n for _, n in parts])


def _T(yaw_deg, t):
    T = np.eye(4)
    T[:3, :3] = drift._rot_about(UP, np.radians(yaw_deg))
    T[:3, 3] = t
    return T


def _sub(P, N, cams, path_m, T=np.eye(4), t=(0.0, 1.0)):
    return Submap(P @ T[:3, :3].T + T[:3, 3], N @ T[:3, :3].T, cams @ T[:3, :3].T + T[:3, 3], t, path_m)


def test_loop_recovers_injected_yaw_and_offset():
    P, N = _room()
    cams = np.array([[2.0, 1.4, 2.5], [2.2, 1.4, 2.0]])
    inj = _T(1.0, [0.06, 0.0, -0.04])          # the revisit landed 1 deg and 7 cm off
    a, b = _sub(P, N, cams, 0.0), _sub(P, N, cams, 10.0, inj)
    r = drift.register_loop(b, a, path_between=10.0, turn_between=360.0)
    assert r["accepted"], r["reason"]
    fix = r["T"] @ inj                          # should undo the injection
    assert np.allclose(fix[:3, :3], np.eye(3), atol=2e-3)
    assert np.linalg.norm(fix[:3, 3]) < 0.01


def test_degenerate_corridor_loop_is_rejected():
    """Two parallel walls and a floor: ICP can slide along the corridor, so the loop must not be trusted."""
    X, Y, Z = np.eye(3)
    parts = [_plane(np.zeros(3), X, Z, 1.2, 6.0, Y), _plane(np.zeros(3), Z, Y, 6.0, 2.5, X),
             _plane(1.2 * X, Z, Y, 6.0, 2.5, -X)]
    P, N = np.concatenate([p for p, _ in parts]), np.concatenate([n for _, n in parts])
    cams = np.array([[0.6, 1.4, 3.0]])
    a, b = _sub(P, N, cams, 0.0), _sub(P, N, cams, 10.0, _T(0.0, [0, 0, 0.08]))
    r = drift.register_loop(b, a, path_between=10.0)
    assert not r["accepted"] and "degenerate" in r["reason"]


def test_implausible_correction_is_rejected():
    """Same room, but 'drifted' 25 cm after only 2 m of walking: more than ARKit can drift, so a wrong match."""
    P, N = _room()
    cams = np.array([[2.0, 1.4, 2.5]])
    a, b = _sub(P, N, cams, 0.0), _sub(P, N, cams, 2.0, _T(0.0, [0.12, 0, 0.0]))
    r = drift.register_loop(b, a, path_between=2.0)
    assert not r["accepted"] and "plausible" in r["reason"]


def test_candidates_need_overlap_and_revisit():
    P, N = _room()
    cams = np.array([[2.0, 1.4, 2.5]])
    far = _room(x0=20.0)
    subs = [_sub(P, N, cams, 0), _sub(P, N, cams + [1, 0, 0], 3), _sub(P, N, cams, 6),       # 0-2: revisit
            _sub(*far, cams + [20, 0, 0], 9)]                                                # far away
    assert drift.loop_candidates(subs) == [(0, 2)]


def test_posegraph_closes_the_loop():
    """Submap 2 revisits submap 0's room with an injected error; submap 1 (in between) sees another room,
    so only weak odometry ties them and the loop dominates."""
    P, N = _room()
    P1, N1 = _room(x0=6.0)
    cams = np.array([[2.0, 1.4, 2.5]])
    inj = _T(1.0, [0.06, 0.0, -0.04])
    subs = [_sub(P, N, cams, 0.0), _sub(P1, N1, np.array([[3.0, 1.4, 2.5]]), 5.0), _sub(P, N, cams, 10.0, inj)]
    subs[2].turn_deg = 360.0
    C, loops = drift.posegraph_corrections(subs, UP)
    assert [(l["i"], l["j"], l["accepted"]) for l in loops] == [(0, 2, True)]
    assert np.allclose(C[0], np.eye(4), atol=1e-6)                 # reference node
    fix = C[2] @ inj
    assert np.allclose(fix[:3, :3], np.eye(3), atol=3e-3)
    assert np.linalg.norm(fix[:3, :3] @ cams[0] + fix[:3, 3] - cams[0]) < 0.02
    for k in range(3):                                               # gravity untouched
        assert np.allclose(C[k][:3, :3] @ UP, UP, atol=1e-9)
    before, after = drift.loop_residuals(subs, loops, C)
    assert after < 0.5 * before


def test_plane_method_snaps_yaw():
    P, N = _room()
    cams = np.array([[2.0, 1.4, 2.5]])
    subs = [_sub(P, N, cams, 0.0), _sub(P, N, cams, 3.0), _sub(P, N, cams, 6.0, _T(-2.0, [0, 0, 0]))]
    C, yaws, _ = drift.plane_corrections(subs, UP)
    assert abs(np.degrees(yaws[2]) - 2.0) < 0.3 and abs(yaws[0]) < 1e-3
    assert np.allclose(C[2][:3, :3] @ UP, UP)


def test_heading_spread_drops_when_drift_is_undone_and_ignores_a_global_turn():
    P, N = _room()
    cams = np.array([[2.0, 1.4, 2.5]])
    yaws = [0.0, 1.0, 2.0, 3.0]                       # heading drifting 1 deg per submap
    subs = [_sub(P, N, cams, 3.0 * k, _T(a, [0, 0, 0])) for k, a in enumerate(yaws)]
    undo = np.stack([_T(-a + 5.0, [0, 0, 0]) for a in yaws])   # undo the drift, then turn everything 5 deg
    before, after = drift.heading_spread(subs, undo, UP)
    assert abs(before - np.std(yaws)) < 0.2 and after < 0.1


def test_yaw_only_keeps_gravity_and_pivot():
    T = np.eye(4)
    T[:3, :3] = drift._rot_about([0.1, 1, 0.05], np.radians(2.0))   # a correction with a little tilt
    T[:3, 3] = [0.1, 0.02, -0.05]
    c = np.array([3.0, 1.2, 4.0])
    Y = drift._yaw_only(T, UP, c)
    assert np.allclose(Y[:3, :3] @ UP, UP)
    assert np.allclose(Y[:3, :3] @ c + Y[:3, 3], T[:3, :3] @ c + T[:3, 3])


def test_split_submaps_by_travel_and_time():
    t = np.arange(0, 20, 0.5)
    pos = np.zeros((len(t), 3))
    pos[:, 0] = np.minimum(t, 10) * 0.5          # walks 5 m in 10 s, then stands still for 10 s
    idx = drift.split_submaps(t, pos)
    assert idx[0] == 0 and np.all(np.diff(idx) >= 0)
    assert idx.max() >= 2                         # cut by travel (3 m) and then by time (8 s)


def test_one_submap_and_correct_poses(tmp_path: Path):
    frames = [Frame(rgb=tmp_path / "x.jpg", T_wc=np.eye(4), timestamp=0.1 * i) for i in range(10)]
    r = drift.correct_drift(frames, UP)
    assert r.metrics["n_submaps"] == 1 and "no correction" in r.method
    assert np.allclose(r.corrections, np.eye(4)) and r.frames[0] is not frames[0]
    # correct_poses picks the submap containing each time, or the nearest one
    r.submap_t = np.array([[0.0, 1.0], [2.0, 3.0]])
    r.corrections = np.stack([np.eye(4), _T(0, [1, 0, 0])])
    t, I = np.array([0.5, 1.2, 1.9, 5.0]), np.repeat(np.eye(4)[None], 4, 0)
    r.smooth = False
    assert list(drift.correct_poses(r, t, I)[:, 0, 3]) == [0, 0, 1, 1]
    # smooth (default): linear between the submap centres 0.5 s and 2.5 s, held beyond them
    r.smooth = True
    assert np.allclose(drift.correct_poses(r, t, I)[:, 0, 3], [0, 0.35, 0.7, 1])
    # a yaw correction blends too: halfway between 0 and 2 deg is 1 deg, and the camera lands halfway
    r.corrections = np.stack([np.eye(4), _T(2.0, [0, 0, 0])])
    P = np.repeat(np.eye(4)[None], 1, 0)
    P[0, :3, 3] = [3.0, 1.4, 0.0]
    out = drift.correct_poses(r, np.array([1.5]), P)[0]
    assert abs(np.degrees(drift._yaw_of(out[:3, :3], UP)) - 1.0) < 1e-6
    assert np.allclose(out[:3, 3], 0.5 * (P[0, :3, 3] + _T(2.0, [0, 0, 0])[:3, :3] @ P[0, :3, 3]))


def _face(x, sign, y0=0.0, length=3.0, voxel=0.01):
    y, z = np.meshgrid(np.arange(y0, y0 + length, voxel), np.arange(0.3, 2.1, voxel))
    P = np.stack([np.full(y.size, x), y.ravel(), z.ravel()], 1)
    N = np.tile([sign, 0.0, 0.0], (len(P), 1))
    return P, N


def test_wall_quality_counts_same_facing_double_but_not_partition():
    # one wall seen twice by two drifted passes, 6 cm apart, both facing +x
    P1, N1 = _face(0.0, +1)
    P2, N2 = _face(0.06, +1)
    q = drift.wall_quality(np.r_[P1, P2], np.r_[N1, N2], 0.01)
    assert len(q["double_walls"]) == 1
    d = q["double_walls"][0]
    assert d["axis"] == 0 and d["faces"] == 1 and abs(d["gap_cm"] - 6) <= 1 and d["overlap_m"] >= 2.5
    assert d["lo"] <= 0.05 and d["hi"] >= 2.95
    # a 10 cm partition: faces point opposite ways -> not a double wall
    P3, N3 = _face(0.0, -1)
    P4, N4 = _face(0.10, +1)
    q = drift.wall_quality(np.r_[P3, P4], np.r_[N3, N4], 0.01)
    assert q["double_walls"] == []
    # same facing but side by side along the wall (no overlap) -> not a double wall
    P5, N5 = _face(0.06, +1, y0=4.0)
    q = drift.wall_quality(np.r_[P1, P5], np.r_[N1, N5], 0.01)
    assert q["double_walls"] == []


def test_double_wall_needs_one_continuous_side_by_side_stretch():
    # two separate walls on the line x = 0, each shadowed 6 cm off by a 0.3 m sliver: 0.6 m shared in total,
    # but never 0.5 m side by side in one place -> not a double wall
    A1, M1 = _face(0.0, +1, y0=0.0, length=3.0)
    A2, M2 = _face(0.0, +1, y0=6.0, length=3.0)
    B1, K1 = _face(0.06, +1, y0=2.7, length=0.3)
    B2, K2 = _face(0.06, +1, y0=6.0, length=0.3)
    q = drift.wall_quality(np.r_[A1, A2, B1, B2], np.r_[M1, M2, K1, K2], 0.01)
    assert q["double_walls"] == []
