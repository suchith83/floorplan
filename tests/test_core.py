"""Unit tests for the pure pieces of the pipeline. Run: uv run pytest"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from fp.bundle import Frame
from fp.geometry.align import FloorNotFound, pick_floor
from fp.geometry.plan import length_confidence, spread
from fp.ingest.arkitscenes import _interp_pose
from fp.ingest.quality import rotation_angle_deg, select_frames
from fp.recon.lidar_fuse import frame_points

sys.path.insert(0, str(Path(__file__).parents[1] / "eval"))
from arkit_gt import compare  # noqa: E402


def test_backprojection_worked_example(tmp_path):
    # HOW_IT_WORKS.md §4.3: pixel (200, 96) at 2 m with scene 48018559's K -> x = 0.675 m
    rgb, dep = tmp_path / "rgb.png", tmp_path / "depth.png"
    cv2.imwrite(str(rgb), np.zeros((192, 256, 3), np.uint8))
    cv2.imwrite(str(dep), np.full((192, 256), 2000, np.uint16))
    K = np.array([[212.9, 0, 128.1], [0, 212.9, 96.1], [0, 0, 1]])
    pts, _ = frame_points(Frame(rgb=rgb, K=K, T_wc=np.eye(4), depth=dep), pix_stride=1, max_depth=4.0)
    assert pts.reshape(192, 256, 3)[96, 200] == pytest.approx([0.6754, -0.0009, 2.0], abs=1e-3)


def test_pose_interpolation_blends_position_and_rotation():
    p1 = np.eye(4)
    p1[:3, :3] = cv2.Rodrigues(np.array([0, 0, np.pi / 2]))[0]
    p1[:3, 3] = [1, 0, 0]
    T = _interp_pose(np.array([10.0, 10.1]), np.array([np.eye(4), p1]), 10.03)
    assert T[:3, 3] == pytest.approx([0.3, 0, 0])
    assert rotation_angle_deg(np.eye(3), T[:3, :3]) == pytest.approx(27.0)


def test_select_frames_drops_blurry_fast_and_still(tmp_path):
    sharp = (np.indices((96, 128)).sum(0) % 2 * 255).astype(np.uint8)  # checkerboard
    yaw_deg = [0, 1, 2, 3, 4, 5, 10, 15, 16, 16]   # frames 5-7 turn at 60-100 deg/s
    x = [0.05 * i for i in range(9)] + [0.40]      # frame 9 = frame 8: the camera stood still
    frames = []
    for i in range(10):
        p = tmp_path / f"{i}.png"
        cv2.imwrite(str(p), cv2.GaussianBlur(sharp, (15, 15), 5) if i == 3 else sharp)
        T = np.eye(4)
        T[:3, :3] = cv2.Rodrigues(np.array([0, 0, np.radians(yaw_deg[i])]))[0]
        T[0, 3] = x[i]
        frames.append(Frame(rgb=p, T_wc=T, timestamp=0.05 * i))
    kept, st = select_frames(frames)
    assert [frames.index(f) for f in kept] == [0, 1, 2, 4, 8]
    assert (st["dropped_blurry"], st["dropped_fast"], st["dropped_duplicate"]) == (1, 3, 1)


def test_pick_floor_takes_the_biggest_layer_not_the_smeared_tail():
    layers = [(-0.20, 800), (-0.10, 1200), (0.0, 3000), (0.05, 2500), (0.9, 1500)]  # (z, points)
    assert pick_floor(layers, cam_z=1.0) == 0.0


def test_pick_floor_finds_the_floor_under_a_big_bed():
    assert pick_floor([(0.0, 1500), (0.5, 2500)], cam_z=1.4) == 0.0


def test_pick_floor_raises_when_no_floor_is_seen():
    with pytest.raises(FloorNotFound):
        pick_floor([(1.2, 3000)], cam_z=1.4)


def test_confidence_keeps_the_old_formula_for_sharp_walls_and_falls_with_smear():
    a, b = {"coverage": 0.66, "spread_cm": None}, {"coverage": 0.59, "spread_cm": 1.0}
    assert length_confidence(0.9, a, b) == 0.64            # 0.9 x (0.3 + 0.7 x 0.59)
    smeared = {"coverage": 0.59, "spread_cm": 7.0}          # 6 cm above a sharp wall -> divided by e^2
    assert length_confidence(0.9, a, smeared) == pytest.approx(0.6417 * np.exp(-2), abs=0.01)


def test_spread_mostly_ignores_a_cabinet_front_near_the_wall():
    rng = np.random.default_rng(0)
    pts = np.r_[rng.normal(0, 0.02, 5000), rng.normal(0.12, 0.005, 800)]  # 2 cm wall + a cabinet 12 cm out
    assert spread(pts, 0.0) < 0.026                          # robust: close to the wall's own 2 cm
    assert np.std(pts[np.abs(pts) < 0.15]) > 0.04            # a plain std would double it


def test_eval_compares_only_sides_whose_walls_were_both_seen():
    room = {"polygon": [[0, 0], [5, 0], [5, 2.5], [0, 2.5]], "walls": [
        {"axis": "x", "coord": 0.0, "observed": False, "confidence": 0.3, "spread_cm": None},
        {"axis": "y", "coord": 0.0, "observed": True, "confidence": 0.5, "spread_cm": 2.0},
        {"axis": "x", "coord": 5.0, "observed": True, "confidence": 0.3, "spread_cm": 2.0},
        {"axis": "y", "coord": 2.5, "observed": True, "confidence": 0.5, "spread_cm": 2.0}]}
    rows = compare(room, {"measures": [{"laser_m": 2.49}, {"laser_m": 5.1}]})
    assert rows[0]["err_cm"] == 1.0          # 2.50 m vs laser 2.49 m
    assert rows[1]["pred_m"] is None         # the 5 m side has an inferred wall: not scored
