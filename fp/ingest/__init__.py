"""Input readers. `detect_tier` and `load_capture` look at the input, so nothing downstream branches on it."""
from __future__ import annotations

from pathlib import Path

from fp.bundle import CaptureBundle
from fp.contract import StageError

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".3gp", ".mkv", ".webm"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}


def is_stray(d: Path) -> bool:
    """Stray Scanner export: odometry.csv next to depth/ (16-bit PNGs, not photos)."""
    return (d / "odometry.csv").is_file() and (d / "depth").is_dir()


def detect_tier(path: Path) -> str:
    """Stray or ARKitScenes folder -> lidar; video file -> video; folder of (room folders of) images -> photos.
    The LiDAR checks come first: their depth/*.png would otherwise pass as photos."""
    from fp.ingest import arkitscenes
    path = Path(path)
    if path.is_dir() and (is_stray(path) or arkitscenes.is_arkitscenes(path)):
        return "lidar"
    if path.is_file() and path.suffix.lower() in VIDEO_EXT:
        return "video"
    if path.is_dir() and any(p.suffix.lower() in PHOTO_EXT for p in path.rglob("*")):
        return "photos"
    raise SystemExit(f"Unrecognised capture: {path}\n"
                     "Expected a Stray Scanner folder (odometry.csv + depth/), a video file (.mov/.mp4) "
                     "or a folder of room folders of photos (.jpg/.heic).")


def load_capture(path: Path, *, tier: str | None = None, max_frames: int | None = None,
                 cache_dir: Path | None = None) -> CaptureBundle:
    """Capture -> bundle for the requested tier (default: the detected one). A LiDAR capture can also run
    as video or photos (its RGB alone), to compare tiers on the same rooms. Camera-tier frames are cached
    under cache_dir (default out/_cache), keyed by content."""
    from fp.ingest import arkitscenes, photos, video
    path = Path(path)
    detected = detect_tier(path)
    tier = tier or detected
    if detected == "lidar":
        if is_stray(path):
            if tier == "video":   # the capture's own RGB video, without its depth or poses
                return video.load(path / "rgb.mp4", max_frames=max_frames or video.MAX_KEYFRAMES, cache_dir=cache_dir)
            if tier == "photos":  # evenly spaced sharp stills of the same video, in one unnamed room
                return photos.from_video(path / "rgb.mp4", max_frames or photos.PHOTOS_FROM_VIDEO, cache_dir=cache_dir)
            from fp.ingest import stray
            bundle = stray.load(path)
            if not bundle.frames:
                raise StageError("ingest", f"No frames with depth in {path}.")
            return bundle
        if tier in ("video", "photos"):
            return arkitscenes.load_rgb_only(path, tier, max_frames or (120 if tier == "video" else 24))
        return arkitscenes.load(path)
    if tier != detected:
        raise SystemExit(f"--tier {tier} needs a LiDAR capture; {path} is a {detected} capture.")
    if detected == "video":
        bundle = video.load(path, max_frames=max_frames or video.MAX_KEYFRAMES, cache_dir=cache_dir)
    else:
        bundle = photos.load(path, max_frames=max_frames or photos.MAX_PHOTOS, cache_dir=cache_dir)
    if not bundle.frames:
        raise StageError("ingest", f"No usable frames in {path} (all blurry, duplicate or unreadable).")
    return bundle
