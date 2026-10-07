"""Camera-tier ingest on synthetic inputs: HEIC with EXIF rotation and focal, 10-bit HLG HEVC with a
rotation tag, the room-folder rule, shared doorway photos, and content-keyed caching."""
import math
import subprocess

import cv2
import numpy as np
import pytest
from PIL import Image

from fp.contract import StageError
from fp.ingest import load_capture, media, photos, video


def _gradient(w, h, seed=0):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w, 3), np.uint8)
    a[..., 0] = np.linspace(0, 255, w)[None]
    a[..., 1] = np.linspace(0, 255, h)[:, None]
    a[..., 2] = rng.integers(0, 255, (h, w))
    return a


def _heic(path, w=640, h=480, orientation=6, f35=26):
    import pillow_heif
    pillow_heif.register_heif_opener()
    im = Image.fromarray(_gradient(w, h))
    ex = im.getexif()
    ex[0x0112] = orientation
    ex.get_ifd(0x8769)[0xA405] = f35
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, exif=ex.tobytes())


def _jpg(path, seed=1, w=320, h=240):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_gradient(w, h, seed)).save(path, quality=90)


def test_heic_is_decoded_upright_with_a_focal_prior(tmp_path):
    cap = tmp_path / "cap"
    _heic(cap / "kitchen" / "a.HEIC")
    _jpg(cap / "kitchen" / "b.JPG")
    b = photos.load(cap, cache_dir=tmp_path / "c")
    assert b.source == "photos" and len(b.frames) == 2
    assert b.meta["_frame_rooms"] == ["kitchen", "kitchen"]
    with Image.open(b.frames[0].rgb) as im:
        assert im.size == (480, 640) and im.mode == "RGB"         # EXIF orientation 6 applied
    K = np.array(b.meta["_K_prior"][0])
    assert K[0, 0] == pytest.approx(26 * math.hypot(480, 640) / 43.27, rel=1e-3)
    assert (K[0, 2], K[1, 2]) == (240, 320)
    assert b.meta["_K_prior"][1] is None and any("EXIF focal" in w for w in b.meta["_warnings"])
    assert b.meta["_source_ids"][0] == media.sha1_file(cap / "kitchen" / "a.HEIC")


def test_10bit_hlg_hevc_with_rotation_is_tonemapped_and_upright(tmp_path):
    ff = media.ffmpeg_exe()
    raw, clip = tmp_path / "hlg.mp4", tmp_path / "hlg_rot.mov"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1",
                    "-c:v", "libx265", "-pix_fmt", "yuv420p10le",
                    "-x265-params", "log-level=error:colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc",
                    "-color_trc", "arib-std-b67", "-color_primaries", "bt2020", "-colorspace", "bt2020nc", str(raw)],
                   check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-display_rotation", "90", "-i", str(raw), "-c", "copy", str(clip)],
                   check=True)
    b = video.load(clip, cache_dir=tmp_path / "c")
    m = b.meta
    assert m["bit_depth"] == 10 and m["hdr"] == "arib-std-b67" and abs(m["rotation_deg"]) == 90
    assert m["frames_sampled"] == 3 and len(b.frames) >= 1               # 3 fps over 1 s
    assert [f.timestamp for f in b.frames] == sorted(f.timestamp for f in b.frames)
    for f in b.frames:
        g = cv2.imread(str(f.rgb), cv2.IMREAD_GRAYSCALE)
        assert g.shape == (320, 240)                                     # rotated to portrait
        assert 30 < g.mean() < 220                                       # not washed out, not black
    assert m["_K_prior"] == [None] * len(b.frames)
    assert all(s.startswith(media.sha1_file(clip) + ":") for s in m["_source_ids"])


def test_a_file_that_is_not_a_video_is_a_stage_error(tmp_path):
    bad = tmp_path / "walk.mp4"
    bad.write_bytes(b"not a video at all")
    with pytest.raises(StageError):
        video.load(bad, cache_dir=tmp_path / "c")


def test_loose_photos_in_the_capture_root_are_refused(tmp_path):
    cap = tmp_path / "cap"
    _jpg(cap / "IMG_0001.jpg")
    _jpg(cap / "kitchen" / "IMG_0002.jpg")
    with pytest.raises(StageError, match="one folder per room"):
        photos.load(cap, cache_dir=tmp_path / "c")
    only_loose = tmp_path / "cap2"
    _jpg(only_loose / "IMG_0001.jpg")
    with pytest.raises(StageError, match="one folder per room"):
        load_capture(only_loose, cache_dir=tmp_path / "c")


def test_a_doorway_photo_in_two_rooms_is_one_frame_in_both(tmp_path):
    cap = tmp_path / "cap"
    _jpg(cap / "kitchen" / "k1.jpg", seed=1)
    _jpg(cap / "kitchen" / "door.jpg", seed=9)
    _jpg(cap / "hall" / "h1.jpg", seed=2)
    (cap / "hall" / "IMG_door_copy.jpg").write_bytes((cap / "kitchen" / "door.jpg").read_bytes())
    (cap / "__MACOSX" / "kitchen").mkdir(parents=True)
    (cap / "__MACOSX" / "kitchen" / "._k1.jpg").write_bytes(b"resource fork")
    (cap / "kitchen" / ".hidden.jpg").write_bytes(b"junk")
    b = photos.load(cap, cache_dir=tmp_path / "c")
    assert len(b.frames) == 3 and b.meta["photos"] == 4
    sets = b.meta["_frame_room_sets"]
    assert sets.count(["hall", "kitchen"]) == 1
    k = sets.index(["hall", "kitchen"])
    assert b.meta["_frame_rooms"][k] == "hall"                    # first room in sorted order
    assert b.meta["rooms"] == {"hall": 2, "kitchen": 2}
    assert not b.meta["_warnings"][:-1]                             # only the no-EXIF-focal summary


def test_reruns_reuse_the_cache_byte_for_byte_and_respect_cache_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cap = tmp_path / "cap"
    _heic(cap / "living" / "a.heic")
    _jpg(cap / "living" / "b.jpeg")
    b1 = photos.load(cap, cache_dir=tmp_path / "c1")
    bytes1 = [f.rgb.read_bytes() for f in b1.frames]
    b2 = photos.load(cap, cache_dir=tmp_path / "c1")
    b3 = photos.load(cap, cache_dir=tmp_path / "c2")              # a fresh decode is identical too
    assert b1.meta["_source_ids"] == b2.meta["_source_ids"] == b3.meta["_source_ids"]
    assert [f.rgb for f in b1.frames] == [f.rgb for f in b2.frames]
    assert bytes1 == [f.rgb.read_bytes() for f in b3.frames]
    assert all(str(f.rgb).startswith(str(tmp_path / "c2")) for f in b3.frames)
    assert not (tmp_path / "out").exists()
    # same folder name, different content -> different cache entries
    other = tmp_path / "x" / "cap"
    _jpg(other / "living" / "b.jpeg", seed=5)
    b4 = photos.load(other, cache_dir=tmp_path / "c1")
    assert b4.frames[0].rgb not in [f.rgb for f in b1.frames]


def test_stray_folder_as_photos_takes_stills_from_its_video(tmp_path):
    cap = tmp_path / "cap"
    (cap / "depth").mkdir(parents=True)
    cv2.imwrite(str(cap / "depth" / "000000.png"), np.zeros((192, 256), np.uint16))
    (cap / "odometry.csv").write_text("timestamp, frame, x, y, z, qx, qy, qz, qw\n")
    vw = cv2.VideoWriter(str(cap / "rgb.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(30):
        vw.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    vw.release()
    b = load_capture(cap, tier="photos", max_frames=4, cache_dir=tmp_path / "c")
    assert b.source == "photos" and 1 <= len(b.frames) <= 4
    assert b.meta["_frame_rooms"] == [""] * len(b.frames) and b.meta["_frame_room_sets"] == [[]] * len(b.frames)
    v = load_capture(cap, tier="video", cache_dir=tmp_path / "c")
    assert v.source == "video" and v.meta["frames_sampled"] == 9    # 3 s at 3 fps
