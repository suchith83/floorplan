"""Stray Scanner reader, confidence filter, room occupancy and seen-through footprint, on synthetic inputs."""
import cv2
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from fp.geometry.plan import MIN_RAY_HITS, _footprint
from fp.ingest import detect_tier, load_capture, stray
from fp.recon.lidar_fuse import frame_points

W, H = 64, 48            # synthetic video size (the real one is 1920x1440)
DW, DH = 32, 24          # synthetic depth size (the real one is 256x192)


def _stray(d, n=5, t0=100.0, rotvec=(0, 0, 0), skip_depth=()):
    """A tiny Stray folder: n frames, irregular timestamps, one rotation for every pose."""
    (d / "depth").mkdir(parents=True)
    (d / "confidence").mkdir()
    vw = cv2.VideoWriter(str(d / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 60, (W, H))
    q = Rotation.from_rotvec(rotvec).as_quat()  # x, y, z, w
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"]
    for i in range(n):
        vw.write(np.full((H, W, 3), 40 * i, np.uint8))
        t = t0 + 0.0217 * i + (0.05 if i >= 3 else 0)   # ~46 fps with a gap, like the real captures
        rows.append(f"{t}, {i:06d}, {0.1 * i}, 1.5, 0.0, {q[0]}, {q[1]}, {q[2]}, {q[3]}, 50.0, 50.0, 32.0, 24.0, , ")
        if i not in skip_depth:
            cv2.imwrite(str(d / "depth" / f"{i:06d}.png"), np.full((DH, DW), 2000, np.uint16))
            conf = np.full((DH, DW), 2, np.uint8)
            conf[:, : DW // 2] = 1                       # left half medium confidence
            cv2.imwrite(str(d / "confidence" / f"{i:06d}.png"), conf)
    vw.release()
    (d / "odometry.csv").write_text("\n".join(rows) + "\n")
    return d


def test_reader_uses_odometry_timestamps_poses_and_per_frame_intrinsics(tmp_path):
    d = _stray(tmp_path / "cap")
    assert detect_tier(d) == "lidar"
    b = stray.load(d, cache_dir=tmp_path / "cache")
    assert b.source == "lidar" and len(b.frames) == 5
    assert b.up == pytest.approx([0, 1, 0])
    f = b.frames[3]
    assert f.timestamp == pytest.approx(100.0 + 3 * 0.0217 + 0.05)      # not 3/60
    assert f.T_wc[:3, 3] == pytest.approx([0.3, 1.5, 0.0])
    assert f.K[0, 0] == pytest.approx(50.0 * stray.RGB_SCALE)            # K follows the cached RGB size
    assert cv2.imread(str(f.rgb)).shape[:2] == (round(H * stray.RGB_SCALE), round(W * stray.RGB_SCALE))
    assert f.confidence.name == "000003.png" and f.video_index == 3
    assert b.meta["has_confidence"] and b.meta["video_frames"] == 5


def test_reader_skips_rows_without_depth_and_reuses_the_frame_cache(tmp_path):
    d = _stray(tmp_path / "cap", skip_depth=(2,))
    b = stray.load(d, cache_dir=tmp_path / "cache")
    assert [f.video_index for f in b.frames] == [0, 1, 3, 4] and b.meta["frames_missing"] == 1
    jpg = b.frames[0].rgb
    mtime = jpg.stat().st_mtime_ns
    stray.load(d, cache_dir=tmp_path / "cache")
    assert jpg.stat().st_mtime_ns == mtime                              # decoded once


def test_opencv_camera_convention_and_confidence_filter(tmp_path):
    # camera turned 90 deg about world +y (up): OpenCV forward (+z) then points along world +x
    d = _stray(tmp_path / "cap", n=1, rotvec=(0, np.pi / 2, 0))
    f = stray.load(d, cache_dir=tmp_path / "cache").frames[0]
    p2, _ = frame_points(f, pix_stride=1, max_depth=4.0, min_conf=2)
    p1, _ = frame_points(f, pix_stride=1, max_depth=4.0, min_conf=1)
    assert len(p1) == DW * DH and len(p2) == DW * DH // 2              # level-1 half dropped
    centre = p1[np.argmin(np.linalg.norm(p1 - p1.mean(0), axis=1))]
    assert centre == pytest.approx([2.0, 1.5, 0.0], abs=0.1)            # 2 m in front, along +x


def test_load_capture_routes_stray_to_the_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(stray, "DEFAULT_CACHE", tmp_path / "cache")
    b = load_capture(_stray(tmp_path / "cap"))
    assert b.has_depth and len(b.frames) == 5


def test_room_occupancy_counts_time_inside_not_in_the_doorway():
    from fp.cli import room_occupancy
    rooms = [{"id": "A", "polygon": [[0, 0], [4, 0], [4, 4], [0, 4]]},
             {"id": "B", "polygon": [[4, 0], [8, 0], [8, 4], [4, 4]]}]
    t = np.arange(0, 10, 0.1)
    xy = np.where(t[:, None] < 8, [[2.0, 2.0]], [[4.1, 2.0]])            # 8 s in A, 2 s just past B's door
    cams = np.c_[xy, np.full(len(t), 1.4)]
    occ = room_occupancy(rooms, t, cams)
    assert occ["A"] == pytest.approx(8.0, abs=0.15) and occ["B"] == 0.0


def test_footprint_includes_air_the_sensor_saw_through():
    # floor seen only in a 1 m strip at x in [0, 1]; walls at x = 0 and x = 4; rays from the camera at
    # x = 0.5 to the far wall cross the unseen floor in between
    ys = np.linspace(0, 2, 41)
    floor = np.array([[x, y, 0.0] for x in np.linspace(0, 1, 21) for y in ys])
    wall = np.array([[4.0, y, z] for y in ys for z in np.linspace(0.3, 2.0, 18)])
    P = np.vstack([floor, wall])
    N = np.vstack([np.tile([0, 0, 1.0], (len(floor), 1)), np.tile([-1.0, 0, 0], (len(wall), 1))])
    without, o = _footprint(P, N, None)
    rays = [(np.array([0.5, 1.0, 1.4]), wall)] * MIN_RAY_HITS
    with_rays, o2 = _footprint(P, N, None, rays)
    col = lambda img, oo, x: img[int((x - oo[0]) / 0.02)].mean()
    assert col(without, o, 3.0) == 0 and col(with_rays, o2, 3.0) > 0.5


def test_footprint_always_contains_the_camera_path():
    floor = np.array([[x, y, 0.0] for x in np.linspace(0, 1, 21) for y in np.linspace(0, 1, 21)])
    N = np.tile([0, 0, 1.0], (len(floor), 1))
    P = np.vstack([floor, [[3.0, 1.0, 1.0]]])                          # a far point widens the grid
    N = np.vstack([N, [[1.0, 0, 0]]])
    cams = np.array([[0.5, 0.5, 1.4], [2.5, 0.5, 1.4]])                # walked on past the seen floor
    img, o = _footprint(P, N, None, cams=cams)
    assert img[int((2.2 - o[0]) / 0.02), int((0.5 - o[1]) / 0.02)]
