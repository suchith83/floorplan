"""Depth + poses -> point cloud (sensor LiDAR and ground-truth laser depth).

Per frame:  pixel (u,v) + depth z  --K^-1-->  camera-frame point  --T_wc-->  world point
All frames: concatenate -> 2 cm voxel average -> statistical outlier removal -> normals.
Pixels below MIN_CONFIDENCE (ARKit's per-pixel depth confidence) are dropped before back-projection.
See HOW_IT_WORKS.md §4 for the step-by-step picture."""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from fp.bundle import CaptureBundle, Frame, Recon

# ARKit confidence: 0 low, 1 medium, 2 high. Keeping only 2 gave the thinnest walls on c00a170fe1
# (docs/STATUS.md 02, experiment D); level-1 pixels sit on edges, dark or far surfaces and glass.
MIN_CONFIDENCE = 2
MAX_DEPTH = 4.0   # m: LiDAR noise grows with range; walls are re-observed closer anyway
VOXEL = 0.01      # m: averaging cell; 1 cm gave thinner walls than 2 cm (2.2 vs 2.8 cm, STATUS 02) at 2x fuse time
PIX_STRIDE = 2    # every 2nd depth pixel: neighbouring frames overlap ~95 %, so detail is not lost


def fuse_depth(bundle: CaptureBundle, voxel: float = VOXEL, max_depth: float = MAX_DEPTH,
               min_conf: int = MIN_CONFIDENCE, debug: dict | None = None) -> Recon:
    """Sensor depth -> voxel-averaged cloud. (Open3D 0.20 TSDF returns empty clouds on
    macOS arm64, so we back-project and rely on voxel averaging + outlier removal.)"""
    rec = backproject(bundle.frames, bundle.up, pix_stride=PIX_STRIDE, max_depth=max_depth, voxel=voxel,
                      with_color=True, min_conf=min_conf, debug=debug)
    rec.meta.update({"backend": "depth-backproject", "min_confidence": min_conf, "max_depth": max_depth})
    return rec


def frame_points(f: Frame, pix_stride: int = 4, max_depth: float = 6.0, with_color: bool = False,
                 min_conf: int = MIN_CONFIDENCE):
    """One depth map -> world points (N,3) and optionally colours (N,3, RGB 0..1).
    If the frame has a confidence map, pixels below min_conf are dropped."""
    D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
    s = D.shape[1] / Image.open(f.rgb).size[0]       # K is at RGB resolution; rescale to depth
    fx, fy, cx, cy = f.K[0, 0] * s, f.K[1, 1] * s, f.K[0, 2] * s, f.K[1, 2] * s
    d = D[::pix_stride, ::pix_stride] / 1000.0       # mm -> m
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * pix_stride
    m = (d > 0.1) & (d < max_depth)
    if f.confidence is not None and min_conf > 0:
        C = cv2.imread(str(f.confidence), cv2.IMREAD_UNCHANGED)
        m &= C[::pix_stride, ::pix_stride] >= min_conf
    z = d[m]
    cam = np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], 1)   # camera frame
    world = cam @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]                        # R·p + t
    if not with_color:
        return world, None
    rgb = cv2.resize(cv2.imread(str(f.rgb)), (D.shape[1], D.shape[0]), interpolation=cv2.INTER_AREA)
    return world, rgb[::pix_stride, ::pix_stride][m][:, ::-1] / 255.0


def backproject(frames: list[Frame], up: np.ndarray, pix_stride: int = 4, max_depth: float = 6.0,
                voxel: float = 0.01, with_color: bool = False, min_conf: int = MIN_CONFIDENCE,
                debug: dict | None = None) -> Recon:
    pts, cols = [], []
    for f in frames:
        p, c = frame_points(f, pix_stride, max_depth, with_color, min_conf)
        pts.append(p)
        cols.append(c)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.concatenate(pts)))
    if with_color:
        pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols))
    if debug is not None:  # per-frame point counts let `fp check fusion` replay the accumulation
        debug["frame_counts"] = [len(p) for p in pts]
    return _finish(pcd, up, frames, voxel, {"backend": "backproject", "voxel": voxel}, debug)


def _finish(pcd, up, frames, voxel, meta, debug=None) -> Recon:
    if debug is not None:
        debug["raw"] = np.asarray(pcd.points).copy()
        debug["raw_colors"] = np.asarray(pcd.colors).copy() if pcd.has_colors() else None
    n_raw = len(pcd.points)
    pcd = pcd.voxel_down_sample(voxel)              # average all points inside each voxel cube
    n_vox = len(pcd.points)
    if debug is not None:
        debug["voxel"] = np.asarray(pcd.points).copy()
    pcd, _ = pcd.remove_statistical_outlier(20, 2.0)  # drop points far from their 20 neighbours
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
    # PCA normals have arbitrary sign; flip each to face the nearest camera that saw the area.
    P, N = np.asarray(pcd.points), np.asarray(pcd.normals)
    cams = np.array([f.T_wc[:3, 3] for f in frames])
    _, idx = cKDTree(cams).query(P)
    flip = np.einsum("ij,ij->i", N, cams[idx] - P) < 0
    N[flip] *= -1
    pcd.normals = o3d.utility.Vector3dVector(N)
    meta.update(n_raw=n_raw, n_voxel=n_vox, n_final=len(P), n_flipped=int(flip.sum()))
    return Recon(points=np.asarray(pcd.points), normals=np.asarray(pcd.normals),
                 up=None if up is None else np.asarray(up, float) / np.linalg.norm(up), frames=frames,
                 colors=np.asarray(pcd.colors) if pcd.has_colors() else None, meta=meta)


RAY_PIX_STRIDE = 8  # every 8th depth pixel (24x32 rays per frame): enough to sweep the floor plan, 1 s per run


def free_space_rays(frames: list[Frame], max_depth: float = MAX_DEPTH, min_conf: int = MIN_CONFIDENCE):
    """Per frame: (camera position, measured points) in world coords. The sensor saw through the air along
    each camera->point segment; the room footprint uses their top-down projection where the floor itself
    was never seen (fp/geometry/plan.py)."""
    out = []
    for f in frames:
        p, _ = frame_points(f, RAY_PIX_STRIDE, max_depth, False, min_conf)
        out.append((f.T_wc[:3, 3].copy(), p))
    return out
