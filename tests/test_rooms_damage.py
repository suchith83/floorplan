"""Room split, damage placement and concealed-damage rules on small synthetic inputs."""
import numpy as np
import pytest

from fp.damage.project import snap
from fp.damage.rules import concealed
from fp.geometry.rooms import doors, split_rooms

RES = 0.02


def test_split_rooms_cuts_at_a_doorway():
    fp = np.zeros((300, 160), bool)        # x by y pixels, 2 cm each
    fp[10:140, 10:150] = True              # room A, 2.6 x 2.8 m
    fp[160:290, 10:150] = True             # room B
    fp[140:160, 60:100] = True             # doorway 0.8 m wide
    lab = split_rooms(fp, RES)
    assert lab.max() == 2
    d = doors(lab, np.zeros(2), RES)
    assert len(d) == 1 and d[0]["kind"] == "door"
    assert d[0]["width_m"] == pytest.approx(0.8, abs=0.06)


def test_split_rooms_keeps_a_long_galley_as_one_room():
    fp = np.zeros((300, 80), bool)
    fp[10:290, 10:70] = True               # 5.6 x 1.2 m, like the test kitchen
    assert split_rooms(fp, RES).max() == 1


ROOM = {"id": "R1", "ceiling_h_m": 2.5, "polygon": [[0, 0], [4, 0], [4, 3], [0, 3]],
        "walls": [{"id": "W1", "axis": "y", "coord": 0.0, "p0": [0, 0], "p1": [4, 0]},
                  {"id": "W2", "axis": "x", "coord": 4.0, "p0": [4, 0], "p1": [4, 3]},
                  {"id": "W3", "axis": "y", "coord": 3.0, "p0": [4, 3], "p1": [0, 3]},
                  {"id": "W4", "axis": "x", "coord": 0.0, "p0": [0, 3], "p1": [0, 0]}]}


def test_snap_places_a_defect_on_a_wall_the_floor_or_the_ceiling():
    plan = {"rooms": [ROOM]}
    on_wall = snap(np.array([2.0, 0.05, 1.2]), plan)
    assert (on_wall["kind"], on_wall["wall"], on_wall["along_wall_m"]) == ("wall", "W1", 2.0)
    assert snap(np.array([2.0, 1.5, 0.02]), plan)["kind"] == "floor"
    assert snap(np.array([2.0, 1.5, 2.48]), plan)["kind"] == "ceiling"
    assert snap(np.array([2.0, 1.5, 1.0]), plan) is None      # mid-air: furniture, not a surface


def test_rules_flag_a_low_wall_stain_and_a_crack_by_a_door():
    plan = {"rooms": [ROOM], "connections": [{"rooms": ["R1", "R2"], "center": [4.0, 1.5]}]}
    items = [{"type": "water stain", "kind": "wall", "room": "R1", "wall": "W1", "height_m": 0.3, "position_m": [2, 0, 0.3]},
             {"type": "crack", "kind": "wall", "room": "R1", "wall": "W2", "height_m": 1.9, "position_m": [4.0, 1.8, 1.9]},
             {"type": "mold", "kind": "ceiling", "room": "R1", "height_m": 2.5, "position_m": [1, 1, 2.5]}]
    concealed(items, plan)
    assert [[c["rule"] for c in d["concealed"]] for d in items] == [["C2"], ["C4"], ["C1"]]
    assert all(c["label"] == "hypothesis" for d in items for c in d["concealed"])


def test_split_rooms_keeps_a_corridor_apart_from_the_room_it_leads_from():
    fp = np.zeros((360, 160), bool)
    fp[10:150, 10:150] = True              # room, 2.8 x 2.8 m
    fp[150:350, 55:105] = True             # corridor 1.0 m wide, 4 m long, leaving the room
    assert split_rooms(fp, RES).max() == 2
