"""Room split on small synthetic inputs (damage tests: tests/test_damage.py)."""
import numpy as np
import pytest

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


def test_split_rooms_keeps_a_corridor_apart_from_the_room_it_leads_from():
    fp = np.zeros((360, 160), bool)
    fp[10:150, 10:150] = True              # room, 2.8 x 2.8 m
    fp[150:350, 55:105] = True             # corridor 1.0 m wide, 4 m long, leaving the room
    assert split_rooms(fp, RES).max() == 2
