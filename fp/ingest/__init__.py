"""Input readers. `load_capture` detects the input type, so nothing downstream branches on it."""
from __future__ import annotations

from pathlib import Path

from fp.bundle import CaptureBundle

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm"}
PHOTO_EXT = {".jpg", ".jpeg", ".png"}


def is_stray(d: Path) -> bool:
    """Stray Scanner export: odometry.csv next to depth/ (16-bit PNGs, not photos)."""
    return (d / "odometry.csv").is_file() and (d / "depth").is_dir()


def load_capture(path: Path, *, as_source: str | None = None, max_frames: int | None = None) -> CaptureBundle:
    """ARKitScenes folder -> LiDAR bundle (or RGB only with as_source='video'|'photos', to test the
    camera-only paths against the same laser scan); video file -> video bundle; photo folder -> photos."""
    from fp.ingest import arkitscenes, photos, video
    path = Path(path)
    if arkitscenes.is_arkitscenes(path):
        if as_source in ("video", "photos"):
            return arkitscenes.load_rgb_only(path, as_source, max_frames or (120 if as_source == "video" else 24))
        return arkitscenes.load(path)
    if is_stray(path):  # must come before the photo check: depth/*.png would pass as photos
        raise SystemExit(f"Stray Scanner capture: {path}\nThe Stray Scanner reader is not implemented yet.")
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return video.load(path, max_frames=max_frames or video.MAX_KEYFRAMES)
    if path.is_dir() and any(p.suffix.lower() in PHOTO_EXT for p in path.rglob("*")):
        return photos.load(path, max_frames=max_frames or photos.MAX_PHOTOS)
    raise SystemExit(f"Unrecognised capture: {path}\n"
                     "Expected an ARKitScenes folder, a video file (.mp4/.mov) or a folder of photos (.jpg).")
