"""Depth + poses -> point cloud (sensor LiDAR and ground-truth laser depth).

Per frame:  pixel (u,v) + depth z  --K^-1-->  camera-frame point  --T_wc-->  world point
All frames: concatenate -> 2 cm voxel average -> statistical outlier removal -> normals.
See HOW_IT_WORKS.md §4 for the step-by-step picture."""
from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from scipy.spatial import cKDTree

from fp.bundle import CaptureBundle, Frame, Recon


def fuse_depth(bundle: CaptureBundle, voxel: float = 0.02, max_depth: float = 4.0,
               debug: dict | None = None) -> Recon:
    """Sensor depth -> voxel-averaged cloud. (Open3D 0.20 TSDF returns empty clouds on
    macOS arm64, so we back-project and rely on voxel averaging + outlier removal.)
    max_depth 4 m: LiDAR noise grows with range; walls are re-observed closer anyway."""
    rec = backproject(bundle.frames, bundle.up, pix_stride=2, max_depth=max_depth, voxel=voxel,
                      with_color=True, debug=debug)
    rec.meta.update({"backend": "depth-backproject"})
    return rec


def frame_points(f: Frame, pix_stride: int = 4, max_depth: float = 6.0, with_color: bool = False):
    """One depth map -> world points (N,3) and optionally colours (N,3, RGB 0..1)."""
    D = cv2.imread(str(f.depth), cv2.IMREAD_UNCHANGED)
    s = D.shape[1] / Image.open(f.rgb).size[0]       # K is at RGB resolution; rescale to depth
    fx, fy, cx, cy = f.K[0, 0] * s, f.K[1, 1] * s, f.K[0, 2] * s, f.K[1, 2] * s
    d = D[::pix_stride, ::pix_stride] / 1000.0       # mm -> m
    v, u = np.mgrid[0:d.shape[0], 0:d.shape[1]] * pix_stride
    m = (d > 0.1) & (d < max_depth)
    z = d[m]
    cam = np.stack([(u[m] - cx) / fx * z, (v[m] - cy) / fy * z, z], 1)   # camera frame
    world = cam @ f.T_wc[:3, :3].T + f.T_wc[:3, 3]                        # R·p + t
    if not with_color:
        return world, None
    rgb = cv2.resize(cv2.imread(str(f.rgb)), (D.shape[1], D.shape[0]), interpolation=cv2.INTER_AREA)
    return world, rgb[::pix_stride, ::pix_stride][m][:, ::-1] / 255.0


def backproject(frames: list[Frame], up: np.ndarray, pix_stride: int = 4, max_depth: float = 6.0,
                voxel: float = 0.01, with_color: bool = False, debug: dict | None = None) -> Recon:
    pts, cols = [], []
    for f in frames:
        p, c = frame_points(f, pix_stride, max_depth, with_color)
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
