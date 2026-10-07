"""Per-room ceiling height on synthetic plan-frame clouds (fp/geometry/ceiling.py)."""
import numpy as np
import pytest

from fp.geometry.ceiling import room_ceiling

VOX = 0.02
ROOM = [[0, 0], [4, 0], [4, 3], [0, 3]]


def _sheet(x0, x1, y0, y1, z, nz, noise=0.005, seed=0):
    """A horizontal sheet on a VOX grid, normal (0,0,nz), with Gaussian height noise."""
    g = np.random.default_rng(seed)
    xs, ys = np.meshgrid(np.arange(x0, x1, VOX) + VOX / 2, np.arange(y0, y1, VOX) + VOX / 2)
    P = np.c_[xs.ravel(), ys.ravel(), z + g.normal(0, noise, xs.size)]
    N = np.tile([0.0, 0.0, nz], (len(P), 1))
    return P, N


def _cloud(*sheets):
    return np.vstack([s[0] for s in sheets]), np.vstack([s[1] for s in sheets])


def test_flat_room_ceiling():
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.0, 1), _sheet(0, 4, 0, 3, 2.50, -1, seed=1))
    r = room_ceiling(P, N, ROOM, VOX)
    assert r["observed"] and r["h"] == pytest.approx(2.50, abs=0.01)
    assert 0.01 <= r["half"] < 0.02
    assert r["ceiling"]["patches"] == 16 * 12 and r["floor"]["source"] == "room floor layer"
    assert r["other_layers"] == []


def test_no_ceiling_is_not_observed():
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.0, 1))
    r = room_ceiling(P, N, ROOM, VOX, room_id="R3")
    assert not r["observed"] and r["h"] is None and r["half"] is None
    assert "R3" in r["reason"] and "no down-facing points" in r["reason"]


def test_two_layers_dominant_is_largest():
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.0, 1),
                  _sheet(0, 2, 0, 1.5, 2.40, -1, seed=1),      # 3 m^2
                  _sheet(2, 3, 0, 1.5, 3.10, -1, seed=2))      # 1.5 m^2
    r = room_ceiling(P, N, ROOM, VOX)
    assert r["h"] == pytest.approx(2.40, abs=0.01)
    assert len(r["other_layers"]) == 1
    assert r["other_layers"][0]["h"] == pytest.approx(3.10, abs=0.01)
    assert r["other_layers"][0]["area_m2"] == pytest.approx(1.5, rel=0.05)


def test_cabinet_undersides_ignored():
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.0, 1), _sheet(0, 4, 0, 0.6, 1.90, -1, seed=1))  # 2.4 m^2 at 1.9 m
    r = room_ceiling(P, N, ROOM, VOX)
    assert not r["observed"] and "1.9" in r["reason"]
    P2, N2 = _cloud((P, N), _sheet(0, 4, 0.6, 3, 2.45, -1, seed=2))
    r = room_ceiling(P2, N2, ROOM, VOX)
    assert r["h"] == pytest.approx(2.45, abs=0.01) and r["other_layers"] == []


def test_ceiling_outside_polygon_ignored():
    # next room (x 4-8) has a 2.6 m ceiling; this room's own ceiling was not captured
    P, N = _cloud(_sheet(0, 8, 0, 3, 0.0, 1), _sheet(4, 8, 0, 3, 2.60, -1, seed=1))
    assert not room_ceiling(P, N, ROOM, VOX)["observed"]
    assert room_ceiling(P, N, [[4, 0], [8, 0], [8, 3], [4, 3]], VOX)["h"] == pytest.approx(2.60, abs=0.01)


def test_tilted_local_floor_is_subtracted():
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.03, 1), _sheet(0, 4, 0, 3, 2.53, -1, seed=1))
    r = room_ceiling(P, N, ROOM, VOX)
    assert r["floor"]["z"] == pytest.approx(0.03, abs=0.003)
    assert r["h"] == pytest.approx(2.50, abs=0.01)


def test_unseen_floor_falls_back_with_wider_interval():
    P, N = _cloud(_sheet(0, 4, 0, 3, 2.50, -1, seed=1))
    r = room_ceiling(P, N, ROOM, VOX)
    assert r["floor"]["source"].startswith("plan floor") and r["h"] == pytest.approx(2.50, abs=0.01)
    assert r["half"] > 0.04


def test_smeared_layer_counts_once():
    # one layer smeared over 6 cm (two passes 6 cm apart) must not appear as a second layer
    P, N = _cloud(_sheet(0, 4, 0, 3, 0.0, 1), _sheet(0, 4, 0, 1.5, 2.44, -1, seed=1),
                  _sheet(0, 4, 1.5, 3, 2.50, -1, seed=2))
    r = room_ceiling(P, N, ROOM, VOX)
    assert r["observed"] and r["other_layers"] == []
