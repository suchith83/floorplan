"""Video file -> keyframes. ffmpeg (the imageio-ffmpeg binary) decodes the clip, applies the phone's
rotation, tone-maps HDR (iPhone HLG / Dolby Vision, PQ) to SDR, and keeps the first frame of every
1/SAMPLE_FPS-second slot by presentation time, so variable-frame-rate clips (Stray's "60 fps" is ~46
with gaps) are sampled evenly in time. Then blurry frames and frames where the camera did not move are
dropped, and above `max_frames` the sharpest frame of each equal time slot is kept.

Decoded frames are cached by a hash of the video bytes, so a renamed copy reuses them and two
different clips with the same name never collide."""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from fp.bundle import CaptureBundle, Frame
from fp.contract import StageError
from fp.ingest import media
from fp.ingest.quality import MIN_SHARP_FRAC, image_stats

# The recon backend splits the keyframes into overlapping chunks, so this caps total work, not GPU memory;
# 240 = 80 s of walkthrough at 3 fps before thinning (provisional, to tune in work order 05).
MAX_KEYFRAMES = 240
SAMPLE_FPS = 3.0
SAVE_WIDTH = 1280
JPEG_QUALITY = 95
DUPLICATE_DIFF = 1.5  # mean grey difference to the previous frame: below it the camera did not move
# HDR -> SDR: linearise (HLG/PQ, 100-nit reference white), BT.2020 -> BT.709 primaries, Hable curve,
# back to BT.709 gamma. desat=0 keeps colours (desaturating highlights greys out sunlit walls).
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,tonemap=tonemap=hable:desat=0,"
           "zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
CACHE_VERSION = 1  # bump when the decode recipe changes, so old cached frames are not reused


def load(path: Path, max_frames: int = MAX_KEYFRAMES, cache_dir: Path | None = None) -> CaptureBundle:
    path = Path(path)
    dec = decode(path, cache_dir)
    saved, times, index = dec["paths"], dec["times"], dec["index"]
    st = image_stats(saved)
    good = (st["sharpness"] >= MIN_SHARP_FRAC * np.median(st["sharpness"])) & (st["diff_prev"] >= DUPLICATE_DIFF)
    idx = select_keyframes(st["sharpness"], good, max_frames)
    frames = [Frame(rgb=saved[k], timestamp=times[k], video_index=index[k]) for k in idx]
    p = dec["probe"]
    return CaptureBundle(source="video", frames=frames, up=None, meta={
        "file": path.name, "duration_s": round(p["duration_s"] or (times[-1] if times else 0.0), 2),
        "fps": p["fps"], "codec": p["codec"], "bit_depth": p["bit_depth"], "hdr": p["hdr"],
        "rotation_deg": p["rotation_deg"], "frames_sampled": len(saved),
        "dropped_blurry_or_still": int((~good).sum()), "keyframes": len(frames),
        "_source_ids": [f"{dec['sha1']}:{index[k]}" for k in idx],
        "_K_prior": [None] * len(frames), "_warnings": list(dec["warnings"])})


def select_keyframes(sharpness: np.ndarray, good: np.ndarray, max_frames: int) -> np.ndarray:
    """Indices of the good frames; above max_frames, the sharpest one in each of max_frames equal slots."""
    idx = np.flatnonzero(good)
    if len(idx) > max_frames:
        idx = np.array([s[np.argmax(sharpness[s])] for s in np.array_split(idx, max_frames)], dtype=int)
    return idx


def decode(path: Path, cache_dir: Path | None = None) -> dict:
    """Video -> SAMPLE_FPS upright SDR JPEGs in the cache. Returns {paths, times (s, stream PTS),
    index (source frame number), sha1, probe, warnings}. A finished decode is published by renaming its
    folder, so an interrupted run is never mistaken for a complete one."""
    if not path.is_file():
        raise StageError("ingest", f"Video not found: {path}")
    try:
        probe = media.probe_video(path)
    except ValueError as e:
        raise StageError("ingest", f"Could not read {path.name} as a video: {e}") from None
    sha = media.sha1_file(path)
    root = Path(cache_dir or media.DEFAULT_CACHE) / "frames" / sha[:16]
    final = root / f"video_v{CACHE_VERSION}_{SAMPLE_FPS:g}fps_w{SAVE_WIDTH}"
    marker = final / "index.json"
    if not marker.is_file():
        root.mkdir(parents=True, exist_ok=True)
        tmp = Path(tempfile.mkdtemp(prefix=".tmp-", dir=root))
        try:
            info = _run_ffmpeg(path, probe, tmp)
            if not info["frames"]:   # never cache a failed decode
                raise StageError("ingest", f"No decodable frames in {path.name} ({info['error']}).")
            media.atomic_write(tmp / "index.json", json.dumps(info).encode())
            try:
                os.rename(tmp, final)
            except OSError:          # another run finished the same decode first: use theirs
                if not marker.is_file():
                    raise
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    info = json.loads(marker.read_text())
    return {"paths": [final / f["file"] for f in info["frames"]], "times": [f["t"] for f in info["frames"]],
            "index": [f["n"] for f in info["frames"]], "sha1": sha, "probe": probe,
            "warnings": info.get("warnings", [])}


def _out_size(probe: dict) -> tuple[int, int]:
    w, h = probe["size"]
    if round(probe["rotation_deg"]) % 180:  # +-90: ffmpeg's autorotate swaps the axes
        w, h = h, w
    W = min(SAVE_WIDTH, w) // 2 * 2
    return W, max(2, round(h * W / w / 2) * 2)


def _run_ffmpeg(path: Path, probe: dict, out: Path) -> dict:
    """One ffmpeg pass: autorotate -> pick the first frame of each 1/SAMPLE_FPS slot -> scale -> (tonemap)
    -> rgb24 on stdout. Two showinfo filters log the picked frames' source number and PTS."""
    if not probe["size"]:
        return {"frames": [], "error": "no frame size in the stream header", "warnings": []}
    W, H = _out_size(probe)
    pick = f"select='isnan(prev_t)+gt(floor(t*{SAMPLE_FPS:g})\\,floor(prev_t*{SAMPLE_FPS:g}))'"
    vf = f"showinfo,{pick},showinfo,scale={W}:{H}:flags=area"
    if probe["hdr"]:
        vf += "," + TONEMAP
    cmd = [media.ffmpeg_exe(), "-hide_banner", "-nostats", "-loglevel", "info", "-i", str(path),
           "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = W * H * 3
    files = []
    with tempfile.TemporaryFile() as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
        while len(buf := proc.stdout.read(nbytes)) == nbytes:
            name = f"{len(files):05d}.jpg"
            rgb = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            cv2.imwrite(str(out / name), rgb[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            files.append(name)
        proc.stdout.close()
        rc = proc.wait()
        log.seek(0)
        err = log.read().decode(errors="replace")
    src = {int(m[1]): (int(m[2]), float(m[3]), m[4]) for m in
           re.finditer(r"Parsed_showinfo_0 .*?n:\s*(\d+) pts:\s*(-?\d+)\s+pts_time:(\S+).*? s:(\d+x\d+)", err)}
    by_pts = {pts: n for n, (pts, _, _) in src.items()}
    picked = [(int(m[1]), float(m[2])) for m in
              re.finditer(r"Parsed_showinfo_2 .*?n:\s*\d+ pts:\s*(-?\d+)\s+pts_time:(\S+)", err)]
    warnings = []
    if len(picked) != len(files):
        warnings.append(f"{path.name}: ffmpeg logged {len(picked)} sampled frames but wrote {len(files)}")
    first = src.get(min(src)) if src else None
    if first and _landscape(first[2]) != _landscape(f"{W}x{H}"):   # autorotate did not do what the probe said
        warnings.append(f"{path.name}: decoded frames are {first[2]}, expected the orientation of {W}x{H} "
                        f"after a {probe['rotation_deg']:g} deg rotation; frames may be squashed")
    n = min(len(picked), len(files))
    frames = [{"file": files[k], "t": round(picked[k][1], 6), "n": by_pts.get(picked[k][0], k)} for k in range(n)]
    error = None if frames else (err.strip().splitlines()[-1] if err.strip() else f"ffmpeg exit code {rc}")
    return {"frames": frames, "size": [W, H], "error": error, "warnings": warnings}


def _landscape(s: str) -> bool:
    w, h = map(int, s.split("x"))
    return w >= h
