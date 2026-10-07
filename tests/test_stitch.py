"""Whole-floor stitching on synthetic clouds: partition walls cut back out of the footprint, doorways closed
along wall lines, overlap repair, connector naming."""
import numpy as np
from shapely.geometry import Polygon

from fp.geometry.plan import extract_rooms


def _wall(axis, coord, a, b, face, step=0.01):
    """Points on a vertical wall plane x=coord (axis 0) or y=coord (axis 1), from a to b along it, 0-2.4 m high;
    normals face +face along the axis."""
    s, z = np.meshgrid(np.arange(a, b, step), np.arange(0.05, 2.4, step), indexing="ij")
    s, z = s.ravel(), z.ravel()
    c = np.full_like(s, coord)
    P = np.stack([c, s, z], 1) if axis == 0 else np.stack([s, c, z], 1)
    N = np.zeros_like(P)
    N[:, axis] = face
    return P, N


def _floor(x0, x1, y0, y1, step=0.02):
    x, y = np.meshgrid(np.arange(x0, x1, step), np.arange(y0, y1, step), indexing="ij")
    P = np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], 1)
    return P, np.tile([0.0, 0, 1], (len(P), 1))


def _two_rooms(door=(1.0, 1.9)):
    """Room A x 0-3, room B x 3.1-6, both y 0-3, a 10 cm partition (faces x=3.0 and 3.1) with a doorway."""
    parts = [_floor(0, 3, 0, 3), _floor(3.1, 6, 0, 3),
             _wall(1, 0.0, 0, 6.1, +1), _wall(1, 3.0, 0, 6.1, -1), _wall(0, 0.0, 0, 3, +1), _wall(0, 6.1, 0, 3, -1)]
    for lo, hi in ((0, door[0]), (door[1], 3.0)):
        parts += [_wall(0, 3.0, lo, hi, -1), _wall(0, 3.1, lo, hi, +1)]
    P = np.concatenate([p for p, _ in parts])
    N = np.concatenate([n for _, n in parts])
    cams = np.array([[x, 1.5, 1.4] for x in np.linspace(0.8, 5.2, 60)])
    return P, N, cams


def test_a_partition_wall_with_a_door_gives_two_rooms_and_one_door():
    P, N, cams = _two_rooms()
    rooms, conns, dbg = extract_rooms(P, N, None, 0.01, 1.0, None, cams)
    assert len(rooms) == 2
    areas = sorted(r["area_m2"] for r in rooms)
    assert all(abs(a - 9.0) < 0.6 for a in areas)          # each ~3.0 x 2.9-3.0 m
    doors = [c for c in conns if c["kind"] == "door"]
    assert len(doors) == 1 and abs(doors[0]["width_m"] - 0.9) < 0.1
    assert abs(doors[0]["center"][1] - 1.45) < 0.15
    A, B = (Polygon(r["polygon"]) for r in rooms)
    assert A.intersection(B).area < 0.01 * min(A.area, B.area)


def test_a_wall_without_a_door_gives_adjacency_without_an_opening():
    P, N, cams = _two_rooms(door=(1.0, 1.0001))             # no gap: a solid partition
    cams = np.concatenate([cams[:20], cams[-20:]])          # the camera stays clear of the wall
    rooms, conns, _ = extract_rooms(P, N, None, 0.01, 1.0, None, cams)
    assert len(rooms) == 2
    assert [c["kind"] for c in conns] == ["wall"]


def test_rank_rooms_names_a_long_narrow_room_connector():
    from fp.bundle import CaptureBundle
    from fp.cli import rank_rooms
    rooms = [{"id": "R1", "name": "R1", "area_m2": 12.0, "width_m": 3.0, "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]]},
             {"id": "R2", "name": "R2", "area_m2": 5.0, "width_m": 1.1, "polygon": [[4, 0], [9, 0], [9, 1], [4, 1]]},
             {"id": "R3", "name": "R3", "area_m2": 1.5, "width_m": 1.0, "polygon": [[0, 3], [1, 3], [1, 4.5], [0, 4.5]]}]
    plan = {"source": {}, "warnings": []}
    out, _ = rank_rooms(rooms, [], CaptureBundle("lidar", []), np.eye(4), plan, lambda *a: None)
    assert [(r["id"], r["name"]) for r in out] == [("R1", "R1"), ("R2", "connector"), ("R3", "R3")]
