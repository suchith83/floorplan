"""Work order 06 synthetic acceptance: paint damage onto real sample frames, then check that the live detector
finds it and places it on the right wall with a plausible extent.

    uv run python scripts/exp06_damage.py data/stray/c00a170fe1 [--out out/_exp06]

What it does: runs fp's geometry on the capture (as `fp run`), puts each defect at the observed-wall point the
most sampled frames see (as a filmed, staged defect would be), and paints it in correct perspective into every
full-resolution frame the detector sees:
  - a brown tea-stain blotch, 0.25 m across, centre 0.4 m above the floor (expected: water_stain, ~0.049 m2);
  - a jagged dark crack, 0.30 m long, 1.4 m above the floor (expected: crack).
A defect is painted only where the frame sees that wall point unoccluded, so views are consistent.
It then prints what was found, on which surface, with which extent, and writes the evidence crops."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from fp import contract
from fp.cli import _geometry
from fp.damage import project
from fp.damage.project import assess

STAIN_D, STAIN_Z = 0.25, 0.40
CRACK_L, CRACK_Z = 0.30, 1.40


def _wall_frame(wall):
    p0, p1 = np.array(wall["p0"]), np.array(wall["p1"])
    L = np.linalg.norm(p1 - p0)
    u = (p1 - p0) / L
    n = np.array([u[1], -u[0]])
    return p0, u, n, L


def _n_views(X, frames, T):
    """How many of the frames see plan point X unoccluded in the middle 80 % of the image."""
    k = 0
    for f in frames:
        Tcp = np.linalg.inv(T @ f.T_wc)
        C = Tcp[:3, :3] @ X + Tcp[:3, 3]
        if not 0.5 < C[2] < 4.0:
            continue
        h, w = cv2.imread(str(f.rgb)).shape[:2]
        u, v = (f.K @ C)[:2] / C[2]
        if not (0.1 * w < u < 0.9 * w and 0.1 * h < v < 0.9 * h):
            continue
        D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED) / 1000.0
        if abs(D[int(v * D.shape[0] / h), int(u * D.shape[1] / w)] - C[2]) < 0.10:
            k += 1
    return k


def _best_spot(plan, frames, T, z):
    """The observed-wall point at height z that the most sampled frames see: where a staged defect in a filmed
    room would be (the person films the damage)."""
    best = (-1, None, None)
    for r in plan["rooms"]:
        for w in r["walls"]:
            if not w["observed"]:
                continue
            p0, u, n, L = _wall_frame(w)
            for a in np.arange(0.4, L - 0.4, 0.25):
                k = _n_views(np.array([*(p0 + u * a - n * 0.01), z]), frames, T)
                if k > best[0]:
                    best = (k, w, a)
    return best


def _defects(stain_wall, stain_a, crack_wall, crack_a):
    rng = np.random.default_rng(6)
    p0, u, n, L = _wall_frame(stain_wall)
    on = lambda a, z: np.array([*(p0 + u * a - n * 0.005), z])     # 5 mm in front of the wall face
    t = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    r = STAIN_D / 2 * (1 + 0.12 * np.sin(3 * t + 1) + 0.06 * rng.standard_normal(len(t)))
    stain = np.array([on(stain_a + rr * np.cos(tt), STAIN_Z + rr * np.sin(tt)) for rr, tt in zip(r, t)])
    stain_c = on(stain_a, STAIN_Z)
    p0, u, n, L = _wall_frame(crack_wall)
    on = lambda a, z: np.array([*(p0 + u * a - n * 0.005), z])
    s = np.linspace(-CRACK_L / 2, CRACK_L / 2, 16)
    crack = np.array([on(crack_a + ss, CRACK_Z + 0.03 * np.sin(9 * ss) + 0.01 * rng.standard_normal()) for ss in s])
    return {"stain": stain, "crack": crack, "stain_c": stain_c, "crack_c": on(crack_a, CRACK_Z)}


def _paint(rgb, f, T, k, D):
    """Paint both defects into one upright full-resolution frame (rgb, turned k times) if it sees them."""
    h0, w0 = cv2.imread(str(f.rgb)).shape[:2]
    Hf, Wf = (rgb.shape[:2] if k % 2 == 0 else rgb.shape[1::-1])        # frame-orientation size
    img = np.ascontiguousarray(np.rot90(rgb, -k)).copy()
    Tcp = np.linalg.inv(T @ f.T_wc)
    Dm = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED) / 1000.0

    def px(X):
        C = X @ Tcp[:3, :3].T + Tcp[:3, 3]
        uv = (C @ f.K.T)[:, :2] / C[:, 2:3]
        return uv * (Wf / w0), C[:, 2]

    def visible(X):
        uv, z = px(X[None])
        if z[0] <= 0.2:
            return False
        i, j = int(uv[0, 1] * Dm.shape[0] / Hf), int(uv[0, 0] * Dm.shape[1] / Wf)
        return 0 <= i < Dm.shape[0] and 0 <= j < Dm.shape[1] and abs(Dm[i, j] - z[0]) < 0.10
    painted = []
    if visible(DEF["stain_c"]):
        uv, _ = px(DEF["stain"])
        m = np.zeros(img.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(uv).astype(np.int32)], 255)
        m = cv2.GaussianBlur(m, (0, 0), 3).astype(np.float32)[..., None] / 255
        tea = np.array([150, 105, 60], np.float32)                   # RGB of dried tea on white paint
        img[:] = (img * (1 - 0.75 * m) + tea * 0.75 * m).astype(np.uint8)
        painted.append("stain")
    if visible(DEF["crack_c"]):
        uv, _ = px(DEF["crack"])
        cv2.polylines(img, [np.round(uv).astype(np.int32)], False, (45, 40, 35), max(2, Wf // 500), cv2.LINE_AA)
        painted.append("crack")
    return np.ascontiguousarray(np.rot90(img, k)), painted


def main():
    global DEF
    ap = argparse.ArgumentParser()
    ap.add_argument("capture", type=Path)
    ap.add_argument("--out", type=Path, default=Path("out/_exp06"))
    a = ap.parse_args()
    out = a.out / a.capture.name
    out.mkdir(parents=True, exist_ok=True)
    plan = contract.empty_plan("lidar", a.capture.name, source={"path": str(a.capture), "cache": True})
    warn = lambda s, m: None
    legacy = _geometry(a.capture, out, plan, {}, "lidar", "local", True, None, warn, True)
    T = np.array(plan["frame"]["T_plan_world"])
    bundle = legacy["bundle"]
    sampled = [bundle.frames[i] for i in project.pick_frames([f.rgb for f in bundle.frames])]
    ks, stain_wall, stain_a = _best_spot(plan, sampled, T, STAIN_Z)
    kc, crack_wall, crack_a = _best_spot(plan, sampled, T, CRACK_Z)
    print(f"stain at {stain_wall['id']} {stain_a:.2f} m along ({ks} sampled frames see it); "
          f"crack at {crack_wall['id']} {crack_a:.2f} m along ({kc})")
    DEF = _defects(stain_wall, stain_a, crack_wall, crack_a)
    orig = project.detection_image
    seen = {"stain": 0, "crack": 0}

    def painted(bundle, i, T_, video):
        rgb, k = orig(bundle, i, T_, video)
        rgb, what = _paint(rgb, bundle.frames[i], T_, k, None)
        for w in what:
            seen[w] += 1
        if what:                                  # keep the painted frames for checking by eye
            cv2.imwrite(str(out / f"painted_{i:05d}_{'_'.join(what)}.jpg"), rgb[:, :, ::-1])
        return rgb, k
    project.detection_image = painted
    items, counts = assess(legacy["bundle"], plan, out, None, "local", False)   # no cache: painted images
    print(f"frames painted (sampled + confirmation): {seen}")
    print("counts", {k: v for k, v in counts.items() if k != "unconfirmed"})
    for u in counts.get("unconfirmed", []):
        print("  unconfirmed", u)
    res = []
    for it in items:
        print(f"  {it['id']} {it['class']:13s} {it['surface_id']:8s} extent {it['extent_m2']['value']:.4f} "
              f"[{it['extent_m2']['lo']:.4f}, {it['extent_m2']['hi']:.4f}] m2  views {it['_views']}  "
              f"pos {it['position']}  bbox {it['bbox_on_surface']}")
        res.append({k: v for k, v in it.items() if not k.startswith("_")})
    expect = {"water_stain": np.pi * (STAIN_D / 2) ** 2, "crack": None}
    for cls, area in expect.items():
        wall = stain_wall if cls == "water_stain" else crack_wall
        hit = [r for r in res if r["class"] == cls and r["surface_id"] == wall["id"]]
        msg = "FOUND on the right wall" if hit else "NOT found on the right wall"
        if hit and area:
            msg += f", extent {hit[0]['extent_m2']['value']:.4f} vs painted {area:.4f} m2"
        print(f"{cls}: {msg}")
    (out / "exp06.json").write_text(json.dumps({"stain_wall": stain_wall["id"], "crack_wall": crack_wall["id"], "painted_frames": seen, "items": res}, indent=1))


DEF: dict = {}

if __name__ == "__main__":
    main()
