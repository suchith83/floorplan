"""Frame quality: brightness, sharpness, near-duplicates, camera rotation speed.

`select_frames` drops the frames that smear the cloud (REVIEW #1): fast rotation, blur and
frames where the camera did not move. `fp check frames|poses` prints the same numbers."""
from __future__ import annotations

import cv2
import numpy as np

from fp.bundle import Frame

MAX_ROT_DPS = 40.0      # faster turns smear walls (rotation p95 was 102 deg/s on the test scan)
MIN_SHARP_FRAC = 0.3    # sharpness below 30 % of the median = blurry
DEDUPE_M, DEDUPE_DEG = 0.02, 2.0


def image_stats(paths) -> dict[str, np.ndarray]:
    """Per image: brightness (mean grey), sharpness (variance of the Laplacian) and the mean
    absolute grey difference to the previous image (near-duplicate test; inf for the first)."""
    br, sh, df, prev = [], [], [], None
    for p in paths:
        g = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        br.append(g.mean())
        sh.append(cv2.Laplacian(g, cv2.CV_32F).var())
        df.append(np.abs(g - prev).mean() if prev is not None and prev.shape == g.shape else np.inf)
        prev = g
    return {"brightness": np.array(br), "sharpness": np.array(sh), "diff_prev": np.array(df)}


def rotation_angle_deg(R0: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """Angle between rotation matrices, element-wise over leading axes. trace(R0^T R1) = sum(R0 * R1)."""
    cos = (np.sum(R0 * R1, axis=(-2, -1)) - 1) / 2
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def rotation_speed(ts: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Angular speed (deg/s) between consecutive 4x4 poses."""
    return rotation_angle_deg(poses[:-1, :3, :3], poses[1:, :3, :3]) / np.diff(ts)


def select_frames(frames: list[Frame], max_rot_dps: float = MAX_ROT_DPS, min_sharp_frac: float = MIN_SHARP_FRAC,
                  dedupe_m: float = DEDUPE_M, dedupe_deg: float = DEDUPE_DEG) -> tuple[list[Frame], dict]:
    """Keep sharp frames taken while the camera turned slowly, and drop frames where it did not move.
    Frames without a pose are judged on sharpness only. Returns (kept frames, counts for plan.json)."""
    n = len(frames)
    sharp = image_stats([f.rgb for f in frames])["sharpness"]
    keep = sharp >= min_sharp_frac * np.median(sharp)
    stats = {"frames_in": n, "dropped_blurry": int((~keep).sum())}
    posed = n > 2 and all(f.T_wc is not None and f.timestamp is not None for f in frames)
    if posed:
        R = np.array([f.T_wc[:3, :3] for f in frames])
        t = np.array([f.timestamp for f in frames])
        # central difference over the neighbours: the speed while this frame was exposed
        speed = np.empty(n)
        speed[1:-1] = rotation_angle_deg(R[:-2], R[2:]) / (t[2:] - t[:-2])
        speed[0], speed[-1] = speed[1], speed[-2]
        fast = speed > max_rot_dps
        stats["dropped_fast"] = int((fast & keep).sum())
        keep &= ~fast
    kept, last = [], None
    for f, k in zip(frames, keep):
        if not k:
            continue
        if posed and last is not None:
            moved = np.linalg.norm(f.T_wc[:3, 3] - last.T_wc[:3, 3])
            turned = rotation_angle_deg(f.T_wc[:3, :3], last.T_wc[:3, :3])
            if moved < dedupe_m and turned < dedupe_deg:
                continue
        kept.append(f)
        last = f
    stats["dropped_duplicate"] = int(keep.sum()) - len(kept)
    stats["frames_kept"] = len(kept)
    return kept, stats
