"""Folder of room folders of photos -> frames. The capture protocol is one folder per room
(photos/kitchen/IMG_0001.HEIC); the folder name labels the plan's rooms, and a doorway photo saved in
both rooms' folders (identical bytes) becomes ONE frame that belongs to both, which is what ties the
rooms together. HEIC/JPEG/PNG are decoded upright and in sRGB; EXIF focal length gives an intrinsics
prior. Near-duplicates (same room, almost the same image) are dropped.

Cached JPEGs are keyed by a hash of each photo's bytes: re-runs reuse them byte for byte, and two
captures that happen to share a folder name never collide."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np

from fp.bundle import CaptureBundle, Frame
from fp.contract import StageError
from fp.ingest import PHOTO_EXT, media
from fp.ingest.quality import MIN_SHARP_FRAC, image_stats

MAX_PHOTOS = 150
SAVE_SIDE = 1600       # longest side of the cached JPEG: enough for the recon models, 4x fewer pixels than 12 MP
JPEG_QUALITY = 95
DUPLICATE_DIFF = 1.5   # mean grey difference (0-255) to the previous photo of the room: below it, a double tap
PHOTOS_FROM_VIDEO = 24  # stills taken from a LiDAR capture's rgb.mp4 to run the photo tier on the same rooms
CACHE_VERSION = 1       # bump when the decode recipe changes


def _ignored(rel: Path) -> bool:
    return any(p.startswith(".") or p == "__MACOSX" for p in rel.parts)


def list_photos(path: Path) -> dict[str, list[Path]]:
    """{room: sorted photo paths}. Raises StageError for loose photos: the rule is one folder per room."""
    found: dict[str, list[Path]] = {}
    loose = []
    for p in sorted(path.rglob("*")):
        rel = p.relative_to(path)
        if p.suffix.lower() not in PHOTO_EXT or not p.is_file() or _ignored(rel):
            continue
        if len(rel.parts) == 1:
            loose.append(rel.name)
        else:
            found.setdefault(rel.parts[0], []).append(p)  # deeper folders count for their first-level room
    if loose:
        raise StageError("ingest", f"{len(loose)} photo(s) sit directly in {path} (e.g. {loose[0]}). Put the photos "
                         f"into one folder per room, e.g. {path.name}/kitchen/IMG_0001.HEIC, "
                         f"{path.name}/hallway/IMG_0002.HEIC; a doorway photo goes into both rooms' folders.")
    return found


def load(path: Path, max_frames: int = MAX_PHOTOS, cache_dir: Path | None = None) -> CaptureBundle:
    path = Path(path)
    rooms = list_photos(path)
    warnings: list[str] = []
    # identical bytes in several room folders = one photo shared by those rooms
    photos: dict[str, dict] = {}   # sha1 -> {"rooms": set, "first": (room, name), "jpg", "K"}
    unreadable, no_focal = [], 0
    for room in sorted(rooms):
        for p in sorted(rooms[room], key=lambda q: str(q.relative_to(path / room))):
            sha = media.sha1_file(p)
            if sha in photos:
                photos[sha]["rooms"].add(room)
                continue
            try:
                jpg, K = _cached_jpeg(p, sha, cache_dir)
            except Exception as e:  # noqa: BLE001 - one bad file must not sink the capture
                unreadable.append(f"{p.relative_to(path)} ({type(e).__name__})")
                continue
            no_focal += K is None
            photos[sha] = {"rooms": {room}, "first": (room, str(p.relative_to(path / room))), "jpg": jpg, "K": K}
    if unreadable:
        warnings.append(f"Skipped {len(unreadable)} unreadable photo(s): {', '.join(unreadable[:5])}"
                        + (" ..." if len(unreadable) > 5 else ""))
    if not photos:
        raise StageError("ingest", f"No readable photos in {path}. " + " ".join(warnings))
    order = sorted(photos, key=lambda s: photos[s]["first"])
    room_of = [photos[s]["first"][0] for s in order]
    shared = np.array([len(photos[s]["rooms"]) > 1 for s in order])
    keep = np.ones(len(order), bool)
    for room in sorted(set(room_of)):   # near-duplicates: compare with the previous photo of the same room
        ks = [k for k, r in enumerate(room_of) if r == room]
        diff = image_stats([photos[order[k]]["jpg"] for k in ks])["diff_prev"]
        for k, d in zip(ks, diff):
            keep[k] = d >= DUPLICATE_DIFF or shared[k]
    idx = np.flatnonzero(keep)
    if len(idx) > max_frames:           # evenly thin the unshared photos; shared ones are the room links
        own = idx[~shared[idx]]
        n_own = max(0, max_frames - int(shared[idx].sum()))
        own = own[np.linspace(0, len(own) - 1, n_own).astype(int)] if n_own else own[:0]
        idx = np.sort(np.concatenate([idx[shared[idx]], own]))
    used = [order[k] for k in idx]
    per_room = {r: sum(r in photos[s]["rooms"] for s in used) for r in sorted(rooms)}
    thin = [r for r, n in per_room.items() if n < 2]
    if thin:
        warnings.append(f"Room folder(s) with fewer than 2 usable photos: {', '.join(thin)}")
    if no_focal:
        warnings.append(f"{no_focal} of {len(photos)} photo(s) have no EXIF focal length; their intrinsics are estimated")
    frames = [Frame(rgb=photos[s]["jpg"], timestamp=float(i)) for i, s in enumerate(used)]
    return CaptureBundle(source="photos", frames=frames, up=None, meta={
        "folder": path.name, "photos": sum(len(v) for v in rooms.values()), "unique_photos": len(photos),
        "used": len(frames), "rooms": per_room,
        "dropped_duplicate": int((~keep).sum()), "shared_photos": int(shared[idx].sum()),
        "_frame_rooms": [photos[s]["first"][0] for s in used],
        "_frame_room_sets": [sorted(photos[s]["rooms"]) for s in used],
        "_source_ids": used,
        "_K_prior": [None if photos[s]["K"] is None else np.asarray(photos[s]["K"]).tolist() for s in used],
        "_warnings": warnings})


def _cached_jpeg(p: Path, sha: str, cache_dir: Path | None) -> tuple[Path, list | None]:
    """Decode once to an upright sRGB JPEG (longest side SAVE_SIDE) + a sidecar with the K prior.
    The sidecar is written last and acts as the 'done' marker."""
    d = Path(cache_dir or media.DEFAULT_CACHE) / "frames" / sha[:16]
    jpg, side = d / f"photo_v{CACHE_VERSION}_{SAVE_SIDE}.jpg", d / f"photo_v{CACHE_VERSION}_{SAVE_SIDE}.json"
    if side.is_file() and jpg.is_file():
        return jpg, json.loads(side.read_text())["K"]
    im, exif, stored = media.open_photo(p)
    im.thumbnail((SAVE_SIDE, SAVE_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=JPEG_QUALITY)
    media.atomic_write(jpg, buf.getvalue())
    K = media.focal_prior(exif, stored, im.size)
    K = None if K is None else K.tolist()
    media.atomic_write(side, json.dumps({"K": K, "size": list(im.size), "source": p.name}).encode())
    return jpg, K


def from_video(video: Path, n: int = PHOTOS_FROM_VIDEO, cache_dir: Path | None = None) -> CaptureBundle:
    """Photo-tier input from a LiDAR capture's own video: n sharp stills spread evenly over the clip, all in
    one unnamed room (the room-labelled photo set is built by scripts/make_camera_tiers.py)."""
    from fp.ingest import video as vid
    dec = vid.decode(Path(video), cache_dir)
    st = image_stats(dec["paths"])
    good = st["sharpness"] >= MIN_SHARP_FRAC * np.median(st["sharpness"])
    good &= st["diff_prev"] >= DUPLICATE_DIFF
    idx = vid.select_keyframes(st["sharpness"], good, n)
    frames = [Frame(rgb=dec["paths"][k], timestamp=float(i), video_index=dec["index"][k]) for i, k in enumerate(idx)]
    return CaptureBundle(source="photos", frames=frames, up=None, meta={
        "folder": Path(video).parent.name, "from_video": Path(video).name, "photos": len(dec["paths"]),
        "used": len(frames), "rooms": {"": len(frames)},
        "_frame_rooms": [""] * len(frames), "_frame_room_sets": [[] for _ in frames],
        "_source_ids": [f"{dec['sha1']}:{dec['index'][k]}" for k in idx],
        "_K_prior": [None] * len(frames), "_warnings": list(dec["warnings"])})
