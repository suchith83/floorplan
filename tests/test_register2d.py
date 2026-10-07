import numpy as np

from fp.geometry.register2d import register


def _flat():
    """Wall points of an asymmetric L-shaped flat with an inner partition."""
    segs = [((0, 0), (8, 0)), ((8, 0), (8, 4)), ((8, 4), (3, 4)), ((3, 4), (3, 9)), ((3, 9), (0, 9)), ((0, 9), (0, 0)),
            ((0, 5), (3, 5)), ((5, 0), (5, 4))]
    pts = []
    for (x0, y0), (x1, y1) in segs:
        n = int(np.hypot(x1 - x0, y1 - y0) / 0.01)
        s = np.linspace(0, 1, n)[:, None]
        pts.append(np.array([x0, y0]) + s * np.array([x1 - x0, y1 - y0]))
    return np.concatenate(pts)


def test_register_recovers_a_quarter_turn_and_offset():
    A = _flat()
    a = np.radians(90.6)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    B = (A - [1.3, -2.1]) @ R            # B = R^T (A - t): A seen from another start pose
    B = B + np.random.default_rng(1).normal(0, 0.005, B.shape)
    r = register(A, B)
    assert r["same_property"]
    assert abs(r["yaw_deg"] - 90.6) < 0.3
    assert np.abs(B @ r["R"].T + r["t"] - A).max() < 0.05


def test_register_says_no_for_a_different_layout():
    A = _flat()
    rng = np.random.default_rng(2)
    B = rng.uniform(0, 9, (5000, 2))     # no walls in common
    assert not register(A, B)["same_property"]
