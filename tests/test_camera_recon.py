"""fp/recon/camera.py on synthetic inputs: Sim(3) fits, chunk merging, gravity from a sideways phone,
cache keys. No model and no data needed."""
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from fp.bundle import CaptureBundle, Frame, Recon
from fp.recon import camera


def _sim3(seed=0):
    rng = np.random.default_rng(seed)
    return 1.3, Rotation.from_rotvec(rng.normal(size=3)).as_matrix(), rng.normal(size=3)


def test_umeyama_recovers_scale_rotation_translation():
    A = np.random.default_rng(1).normal(size=(500, 3))
    s, R, t = _sim3()
    s2, R2, t2 = camera.umeyama(A, A @ R.T * s + t)
    assert abs(s2 - s) < 1e-9 and np.allclose(R2, R) and np.allclose(t2, t)


def test_robust_sim3_ignores_a_third_of_outliers():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(600, 3))
    s, R, t = _sim3(3)
    B = A @ R.T * s + t
    B[:200] += rng.normal(scale=2.0, size=(200, 3))
    s2, R2, t2, res = camera.robust_sim3(A, B)
    assert abs(s2 - s) < 1e-3 and np.allclose(R2, R, atol=1e-3) and res < 1e-6


def test_chunks_overlap_and_cover_everything():
    cs = camera.chunks(100, size=40, overlap=6)
    assert all(len(c) == 40 for c in cs)
    assert sorted(set(i for c in cs for i in c)) == list(range(100))
    for a, b in zip(cs, cs[1:]):
        assert len(set(a) & set(b)) >= 6
    assert camera.chunks(10, size=40) == [list(range(10))]


def _fake_pass(T_world: np.ndarray, H=24, W=32, depth=2.0) -> dict:
    """A model pass over views with given cam->world poses, all seeing a fronto-parallel plane at `depth`."""
    n = len(T_world)
    K = np.array([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]], np.float32)
    rng = np.random.default_rng(0)
    return {"depth": (depth + 0.3 * rng.random((n, H, W))).astype(np.float16),
            "conf": np.ones((n, H, W), np.float16), "mask": np.ones((n, H, W), bool),
            "K": np.repeat(K[None], n, 0), "T_wc": T_world.copy(), "rgb": np.zeros((n, H, W, 3), np.uint8),
            "scale": np.ones(n)}


def test_merge_chunks_undoes_a_per_chunk_similarity():
    n = 10
    T = np.repeat(np.eye(4)[None], n, 0)
    T[:, 0, 3] = np.linspace(0, 2, n)        # camera walks along x
    T[:, :3, :3] = Rotation.from_euler("y", np.linspace(0, 40, n)[:, None], degrees=True).as_matrix()
    full = _fake_pass(T)
    a_idx, b_idx = list(range(0, 6)), list(range(3, 10))   # 3 shared frames
    za = {k: v[a_idx].copy() for k, v in full.items()}
    zb = {k: v[b_idx].copy() for k, v in full.items()}
    s, R, t = _sim3(4)                         # chunk b came out in its own frame and scale
    zb["T_wc"] = np.array([np.r_[np.c_[R @ M[:3, :3], s * R @ M[:3, 3] + t], [[0, 0, 0, 1]]] for M in zb["T_wc"]])
    zb["depth"] = (zb["depth"].astype(np.float32) * s).astype(np.float16)
    z, stats, corr = camera._merge_chunks([(a_idx, za), (b_idx, zb)], n)
    assert stats[0]["shared_frames"] == 3
    assert abs(stats[0]["scale"] - 1 / s) < 0.01
    # chunk b's own metric claim was off by s, chunk a's was right: the median of the two votes is between
    assert abs(corr - (1 + s) / 2) < 0.01
    assert np.allclose(z["T_wc"][:, :3, 3], corr * T[:, :3, 3], atol=0.02)


def test_chained_merge_scale_does_not_drift_with_noisy_depth():
    """Five chunks, every pass predicting the same views with independent 15 % depth noise: the depth-ratio
    median keeps each link's scale at 1 (a least-squares point fit shrinks it, and the shrink compounds)."""
    n, size, ov = 30, 10, 4
    T = np.repeat(np.eye(4)[None], n, 0)
    T[:, 0, 3] = np.linspace(0, 6, n)
    full = _fake_pass(T)
    parts = []
    for k, c in enumerate(camera.chunks(n, size, ov)):
        z = {key: v[c].copy() for key, v in full.items()}
        noise = np.random.default_rng(k).normal(1, 0.15, z["depth"].shape)
        z["depth"] = (z["depth"].astype(np.float32) * noise).astype(np.float16)
        parts.append((c, z))
    z, stats, corr = camera._merge_chunks(parts, n)
    assert all(abs(st["scale"] - 1) < 0.02 for st in stats), stats
    assert abs(corr - 1) < 0.03


def _room_cloud(up: np.ndarray, cams_world: np.ndarray):
    """Floor (big, 1.5 m below the cameras), smaller ceiling 1.1 m above, 4 walls; returned in a frame where
    'up' is the given vector."""
    rng = np.random.default_rng(5)
    floor = np.c_[rng.uniform(-2, 2, (4000, 2)), np.full(4000, -1.5)]
    ceil = np.c_[rng.uniform(-1, 1, (800, 2)), np.full(800, 1.1)]
    wall = np.c_[np.full(2000, 2.0), rng.uniform(-2, 2, 2000), rng.uniform(-1.5, 1.1, 2000)]
    P = np.r_[floor, ceil, wall]
    N = np.r_[np.tile([0, 0, 1.0], (4000, 1)), np.tile([0, 0, -1.0], (800, 1)), np.tile([-1.0, 0, 0], (2000, 1))]
    # rotate the z-up room so that z maps onto `up`
    R = Rotation.align_vectors([up], [[0, 0, 1.0]])[0].as_matrix()
    return P @ R.T, N @ R.T, R


def test_gravity_from_a_sideways_phone():
    """Portrait phone, landscape-stored frames with no rotation tag: the image x axis points to gravity and
    the cameras' mean -y is horizontal. Floor + ceiling outweigh the wall by > 2x, so the x axis overrules."""
    up = np.array([0.0, 0.0, 1.0])
    P, N, R = _room_cloud(up, None)
    frames = []
    for yaw in np.linspace(0, 300, 6):
        fwd = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), -0.3])
        fwd /= np.linalg.norm(fwd)
        x = -up                                     # image x points down: phone rotated 90 degrees
        y = np.cross(fwd, x)
        T = np.eye(4)
        T[:3, :3] = np.c_[x, y, fwd]
        frames.append(Frame(rgb=Path("x.jpg"), T_wc=T))
    rec = Recon(points=P, normals=N, up=None, frames=frames)
    est, info = camera.estimate_gravity(rec)
    assert est @ up > 0.99, (est, info)
    assert info["start"] == "-x"
    assert camera.floor_check(rec, est) < 1.0


def test_pass_key_depends_on_content_priors_and_params_only(monkeypatch):
    k = camera.pass_key(["a", "b", "c"], [None] * 3)                     # no paths in the key at all
    assert k == camera.pass_key(["a", "b", "c"], [None] * 3)
    assert k != camera.pass_key(["a", "B", "c"], [None] * 3)              # bytes do
    assert k != camera.pass_key(["a", "b", "c"], [None, np.eye(3), None])  # intrinsics priors do
    monkeypatch.setitem(camera.MODEL_PARAMS, "resolution_set", 1)
    assert k != camera.pass_key(["a", "b", "c"], [None] * 3)              # parameters do


def test_photo_rooms_without_any_link_are_left_unplaced(tmp_path):
    """Two rooms tied by a shared doorway photo are placed together; a third room with nothing in common
    (blank images: no features to match) is reported unplaced, never merged blindly."""
    import cv2
    paths = []
    for i in range(5):
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), np.full((60, 80, 3), 40 * i, np.uint8))
        paths.append(p)
    sets = [["hall"], ["hall", "kitchen"], ["kitchen"], ["attic"], ["attic"]]
    b = CaptureBundle("photos", [Frame(rgb=p) for p in paths],
                      meta={"_frame_room_sets": sets, "_frame_rooms": [s[0] for s in sets]})
    warnings = []
    g = camera._photo_groups(b, warnings)
    assert g["groups"] == [[0, 1, 2]] and g["unplaced"] == ["attic"]
    assert any("attic" in w and "NOT placed" in w for w in warnings)


def test_gravity_keeps_image_up_when_a_big_wall_nearly_ties():
    """Upright photos in a flat with a long wall: the wall's normals almost tie with the floor's. Image up
    must win (only a 2x larger support may overrule it)."""
    up = np.array([0.0, 0.0, 1.0])
    rng = np.random.default_rng(1)
    floor = np.c_[rng.uniform(-3, 3, (3000, 2)), np.full(3000, -1.5)]
    wall = np.c_[np.full(3200, 2.0), rng.uniform(-3, 3, 3200), rng.uniform(-1.5, 1.1, 3200)]
    P = np.r_[floor, wall]
    N = np.r_[np.tile([0, 0, 1.0], (3000, 1)), np.tile([-1.0, 0, 0], (3200, 1))]
    frames = []
    for yaw in np.linspace(-40, 40, 6):            # all looking roughly at the wall, phone upright
        fwd = np.array([np.cos(np.radians(yaw)), np.sin(np.radians(yaw)), -0.3])
        fwd /= np.linalg.norm(fwd)
        y = -up + (fwd @ up) * fwd                   # image down = gravity projected into the image
        y /= np.linalg.norm(y)
        T = np.eye(4)
        T[:3, :3] = np.c_[np.cross(y, fwd), y, fwd]
        frames.append(Frame(rgb=Path("x.jpg"), T_wc=T))
    est, info = camera.estimate_gravity(Recon(points=P, normals=N, up=None, frames=frames))
    assert info["start"] == "-y" and est @ up > 0.99, info
