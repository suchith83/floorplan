"""ARKitScenes raw sequence -> CaptureBundle (LiDAR path, dev + eval data).

Layout: <vid>/lowres_wide/<vid>_<ts>.png, lowres_depth/ (uint16 mm), lowres_wide_intrinsics/
<vid>_<ts>.pincam ("w h fx fy cx cy"), lowres_wide.traj ("ts ax ay az tx ty tz", world->camera).
highres_depth/ holds depth rendered from the Faro laser scan (ground truth)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from fp.bundle import CaptureBundle, Frame


def is_arkitscenes(d: Path) -> bool:
    return (d / "lowres_wide.traj").exists()


def _ts(p: Path) -> float:
    return float(p.stem.split("_")[-1])


def load_traj(d: Path) -> tuple[np.ndarray, np.ndarray]:
    ts, poses = [], []
    for line in (d / "lowres_wide.traj").read_text().split("\n"):
        tok = line.split()
        if len(tok) != 7:
            continue
        R, _ = cv2.Rodrigues(np.array(tok[1:4], dtype=float))
        T_cw = np.eye(4)
        T_cw[:3, :3], T_cw[:3, 3] = R, np.array(tok[4:7], dtype=float)
        ts.append(float(tok[0]))
        poses.append(np.linalg.inv(T_cw))
    return np.array(ts), np.array(poses)


def _interp_pose(ts_all, poses, t, max_gap=0.15):
    """Camera pose at time t: linear translation + slerp rotation between bracketing traj
    samples. Poses are ~10 Hz, frames 60 Hz; nearest-pose smears walls by tens of cm."""
    i = int(np.searchsorted(ts_all, t))
    if i == 0 or i >= len(ts_all) or ts_all[i] - ts_all[i - 1] > max_gap:
        return None
    t0, t1 = ts_all[i - 1], ts_all[i]
    a = (t - t0) / (t1 - t0)
    rot = Slerp([0, 1], Rotation.from_matrix([poses[i - 1][:3, :3], poses[i][:3, :3]]))([a])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()[0]
    T[:3, 3] = (1 - a) * poses[i - 1][:3, 3] + a * poses[i][:3, 3]
    return T


def _K(pincam: Path, scale: float = 1.0) -> np.ndarray:
    _, _, fx, fy, cx, cy = map(float, pincam.read_text().split())
    return np.array([[fx * scale, 0, cx * scale], [0, fy * scale, cy * scale], [0, 0, 1]])


def load(d: Path, stride: int = 3) -> CaptureBundle:
    d = Path(d)
    ts_all, poses = load_traj(d)
    frames = []
    for rgb in sorted((d / "lowres_wide").glob("*.png"))[::stride]:
        depth = d / "lowres_depth" / rgb.name
        pincam = d / "lowres_wide_intrinsics" / (rgb.stem + ".pincam")
        T = _interp_pose(ts_all, poses, _ts(rgb))
        if T is None or not depth.exists() or not pincam.exists():
            continue
        frames.append(Frame(rgb=rgb, K=_K(pincam), T_wc=T, depth=depth, timestamp=_ts(rgb)))
    # ARKit world frame is gravity-aligned with +Y up.
    return CaptureBundle(source="lidar", frames=frames, up=np.array([0.0, 1.0, 0.0]),
                         meta={"dataset": "ARKitScenes", "video_id": d.name})


def load_rgb_only(d: Path, source: str, n_frames: int) -> CaptureBundle:
    """The same scan as camera-only input: RGB frames evenly spaced, no depth, poses, K or gravity.
    Lets the photo and video paths be measured against the same laser scan as the LiDAR path."""
    from fp.ingest.quality import image_stats
    d = Path(d)
    ts_all, _ = load_traj(d)
    rgbs = [p for p in sorted((d / "lowres_wide").glob("*.png")) if ts_all[0] <= _ts(p) <= ts_all[-1]]
    # pick the sharpest frame in each of n_frames equal time slots, as a person filming would keep
    slots = np.array_split(np.arange(len(rgbs)), n_frames)
    frames = []
    for s in slots:
        cand = [rgbs[i] for i in s[::3]]
        sharp = image_stats(cand)["sharpness"]
        best = cand[int(np.argmax(sharp))]
        frames.append(Frame(rgb=best, timestamp=_ts(best)))
    return CaptureBundle(source=source, frames=frames, up=None,
                         meta={"dataset": "ARKitScenes", "video_id": d.name, "rgb_only": True})


def load_gt_frames(d: Path, stride: int = 10) -> list[Frame]:
    """Laser-scan-rendered depth (1920x1440) with poses + intrinsics scaled from lowres."""
    d = Path(d)
    ts_all, poses = load_traj(d)
    pincams = sorted((d / "lowres_wide_intrinsics").glob("*.pincam"))
    out = []
    for p in sorted((d / "highres_depth").glob("*.png"))[::stride]:
        T = _interp_pose(ts_all, poses, _ts(p))
        if T is None:
            continue
        pin = min(pincams, key=lambda q: abs(_ts(q) - _ts(p)))
        w = float(pin.read_text().split()[0])
        out.append(Frame(rgb=p, K=_K(pin, 1920.0 / w), T_wc=T, depth=p, timestamp=_ts(p)))
    return out
