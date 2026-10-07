"""Shared helpers for the camera tiers: content hashes, atomic cache writes, phone-photo decoding
(HEIC, EXIF rotation, Display P3 -> sRGB, EXIF focal -> intrinsics prior) and video probing with the
ffmpeg binary that ships in the imageio-ffmpeg wheel (no system ffmpeg; OpenCV's decoder ignores HDR
transfer curves, so iPhone HLG clips come out washed out)."""
from __future__ import annotations

import hashlib
import io
import math
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

DEFAULT_CACHE = Path("out/_cache")
HASH_CHUNK = 1 << 20          # 1 MiB reads: hashing a 300 MB walkthrough stays well under a second
FULL_FRAME_DIAG_MM = math.hypot(36.0, 24.0)  # 43.27 mm: the diagonal "35 mm equivalent" focal is defined on
HDR_TRANSFERS = {"smpte2084": "PQ", "arib-std-b67": "HLG"}  # ffmpeg's names for the two HDR curves
# EXIF tags (Exif IFD 0x8769)
EXIF_IFD, FOCAL_35, FOCAL_MM, FP_XRES, FP_UNIT = 0x8769, 0xA405, 0x920A, 0xA20E, 0xA210
FP_UNIT_MM = {2: 25.4, 3: 10.0, 4: 1.0, 5: 0.001}  # FocalPlaneResolutionUnit -> mm per unit (2 = inch)


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(HASH_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    """Write via a temp name + rename, so an interrupted run never leaves a half file that looks done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ---------------------------------------------------------------- photos

def _register_heif() -> None:
    import pillow_heif
    pillow_heif.register_heif_opener()  # idempotent; pillow-heif also applies the HEIF irot on decode


def open_photo(path: Path) -> tuple[Image.Image, dict, tuple[int, int]]:
    """Phone photo -> (upright 8-bit sRGB PIL image, EXIF Exif-IFD dict, stored size before rotation).
    10-bit HEICs are reduced to 8 bit by pillow-heif; an embedded ICC profile (iPhone: Display P3) is
    converted to sRGB, best effort."""
    _register_heif()
    im = Image.open(path)
    stored = im.size
    try:
        exif = dict(im.getexif().get_ifd(EXIF_IFD))
    except Exception:  # noqa: BLE001 - a broken EXIF block only costs the focal prior
        exif = {}
    icc = im.info.get("icc_profile")
    im = ImageOps.exif_transpose(im)
    if icc:
        im = _to_srgb(im, icc)
    return im.convert("RGB"), exif, stored


def _to_srgb(im: Image.Image, icc: bytes) -> Image.Image:
    try:
        from PIL import ImageCms
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        if "srgb" in ImageCms.getProfileDescription(src).lower():
            return im
        return ImageCms.profileToProfile(im.convert("RGB"), src, ImageCms.createProfile("sRGB"), outputMode="RGB")
    except Exception:  # noqa: BLE001 - unknown profile: keep the pixels as they are
        return im


def focal_prior(exif: dict, stored: tuple[int, int], size: tuple[int, int]) -> np.ndarray | None:
    """EXIF -> 3x3 K at `size` (the saved, upright image), principal point at the centre; None if unknown.
    35-mm-equivalent focal is defined on the diagonal, so it is rotation-invariant:
    f_px = f35 * diag_px / 43.27. Fallback: FocalLength (mm) x FocalPlaneXResolution (px per unit)."""
    W, H = size
    f = None
    f35 = _num(exif.get(FOCAL_35))
    if f35:
        f = f35 * math.hypot(W, H) / FULL_FRAME_DIAG_MM
    else:
        fmm, xres, unit = _num(exif.get(FOCAL_MM)), _num(exif.get(FP_XRES)), exif.get(FP_UNIT, 2)
        if fmm and xres and unit in FP_UNIT_MM:
            f = fmm * xres / FP_UNIT_MM[unit] * max(W, H) / max(stored)  # px at stored size -> saved size
    if not f:
        return None
    return np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return x if math.isfinite(x) and x > 0 else None


# ---------------------------------------------------------------- video

def ffmpeg_exe() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_video(path: Path) -> dict:
    """Parse `ffmpeg -i` stderr (the wheel has no ffprobe): codec, pix_fmt, bit depth, transfer, HDR,
    stored size, display rotation, duration, fps. Raises ValueError when there is no video stream."""
    err = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True,
                         text=True, errors="replace").stderr
    lines = err.splitlines()
    i = next((k for k, l in enumerate(lines) if re.search(r"Stream #\d+:\d+.*: Video: ", l)), None)
    if i is None:
        raise ValueError(f"no video stream ({lines[-1].strip() if lines else 'ffmpeg gave no output'})")
    parts = _split_top(lines[i].split(": Video: ", 1)[1])
    codec = parts[0].split()[0]
    pix = re.match(r"(\w+)(?:\((.*)\))?", parts[1]) if len(parts) > 1 else None
    pix_fmt = pix.group(1) if pix else "?"
    color = [c.strip() for c in (pix.group(2) or "").split(",")] if pix else []
    transfer = None
    for c in color:  # "bt2020nc/bt2020/arib-std-b67" = matrix/primaries/transfer; one name when all agree
        names = c.split("/")
        if len(names) == 3 or names[0] in HDR_TRANSFERS or names[0].startswith(("bt", "smpte", "iec")):
            transfer = names[-1]
    size = next((tuple(map(int, m.groups())) for p in parts[2:] if (m := re.match(r"(\d+)x(\d+)", p))), None)
    fps = next((float(m.group(1)) for p in parts if (m := re.match(r"([\d.]+) fps", p))), None)
    # side data / metadata of this stream: the lines indented under it, up to the next stream
    block = []
    for l in lines[i + 1:]:
        if re.search(r"Stream #\d+:\d+", l) or not l.startswith("    "):
            break
        block.append(l)
    block = "\n".join(block)
    rot = re.search(r"displaymatrix: rotation of (-?[\d.]+) degrees", block) or re.search(r"rotate\s*:\s*(-?\d+)", block)
    dur = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err)
    m = re.search(r"(\d+)(?:le|be)$", pix_fmt) or re.match(r"p0(\d\d)", pix_fmt)
    return {"codec": codec, "pix_fmt": pix_fmt, "bit_depth": int(m.group(1)) if m else 8,
            "transfer": transfer, "hdr": HDR_TRANSFERS.get(transfer or "") and transfer,
            "dolby_vision": "DOVI configuration" in block,
            "size": size, "rotation_deg": float(rot.group(1)) if rot else 0.0, "fps": fps,
            "duration_s": int(dur[1]) * 3600 + int(dur[2]) * 60 + float(dur[3]) if dur else None}


def _split_top(s: str) -> list[str]:
    """Split on ', ' outside parentheses: 'hevc (Main 10) (hvc1 / x), yuv420p10le(tv, a/b/c), 320x240'."""
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch == "(") - (ch == ")")
        if ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    return out + [cur.strip()]
