"""Canonical capture format. Every input type (photos / video / LiDAR) is converted
into a CaptureBundle so the rest of the pipeline never branches on input type."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Frame:
    rgb: Path
    K: np.ndarray | None = None          # 3x3 intrinsics at rgb resolution
    T_wc: np.ndarray | None = None       # 4x4 camera-to-world (metres)
    depth: Path | None = None            # uint16 PNG, millimetres
    timestamp: float | None = None
    confidence: Path | None = None       # uint8 PNG 0/1/2 at depth resolution (Stray Scanner / ARKit)
    video_index: int | None = None       # frame number in the source video, for full-resolution crops


@dataclass
class CaptureBundle:
    source: str                          # "lidar" | "video" | "photos"
    frames: list[Frame]
    up: np.ndarray | None = None         # gravity-up hint in world coords, if known
    anchors: list[dict] = field(default_factory=list)  # user-typed lengths
    meta: dict = field(default_factory=dict)

    @property
    def has_depth(self) -> bool:
        return bool(self.frames) and all(f.depth is not None and f.T_wc is not None for f in self.frames)


@dataclass
class Recon:
    """Output of any reconstruction backend, in world coords (metres)."""
    points: np.ndarray                   # (N,3)
    normals: np.ndarray                  # (N,3)
    up: np.ndarray | None                # (3,) unit; None = no IMU, align estimates it
    frames: list[Frame]                  # with T_wc filled in
    colors: np.ndarray | None = None     # (N,3) RGB 0..1, display only
    meta: dict = field(default_factory=dict)
