"""Drift correction for long handheld captures (ARKit / Stray Scanner poses).

What is left to fix: ARKit's visual-inertial odometry is locally excellent (a few mm between frames)
and its gravity is accurate (the accelerometer sees it directly), but over a 3-minute walk through a
flat its heading (yaw) and position slowly drift. A wall scanned at minute 1 and again at minute 3
then lands a few cm apart: a "double wall".

Method "posegraph" (the default), in five steps:
 1. Submaps: cut the camera path every SUBMAP_TRAVEL metres or SUBMAP_SECONDS seconds. Inside one
    submap ARKit drift is negligible, so each is fused as a rigid piece (coarse 3 cm cloud).
 2. Odometry edges between consecutive submaps: ARKit says their relative transform is identity
    (both clouds are already in world coords); the weight is how much surface they share.
 3. Loop closures: non-consecutive submaps where the camera came back (a camera of j within
    REVISIT_DIST of a camera of i). Point-to-plane ICP from identity; accept only a tight fit
    (fitness, RMSE), a geometry that pins all three directions, and a correction no bigger than
    ARKit could plausibly drift over that stretch of the walk. Every candidate and its verdict is kept.
 4. Pose graph (Open3D global_optimization, Levenberg-Marquardt with a line process that can still
    switch off a bad loop): one rigid correction per submap. Then keep only the yaw part of each
    rotation, because ARKit's gravity is better than anything ICP would add.
 5. Loop residuals before/after on the accepted loops.

Method "plane" (cheaper fallback): per submap a small yaw about its camera centroid that snaps its own
wall normals onto the global Manhattan axes. Fixes heading drift only, not position drift."""
from __future__ import annotations

import dataclasses
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from fp.bundle import Frame

# --- submaps -------------------------------------------------------------------------------------
SUBMAP_TRAVEL = 3.0     # m of camera travel per submap: ARKit drift is ~1 % of distance, so ~3 cm at most inside one
SUBMAP_SECONDS = 8.0    # s per submap, whichever comes first: standing still and turning also drifts (yaw)
MIN_SUBMAP_FRAMES = 3   # a shorter tail is merged into the previous submap (too few views to register)
REG_PIX_STRIDE = 4      # every 4th depth pixel: registration needs surfaces, not detail
REG_MAX_DEPTH = 4.0     # m, same range limit as the main fusion (fp/recon/lidar_fuse.py MAX_DEPTH)
REG_VOXEL = 0.03        # m: 3 cm cells keep ICP fast (~10-30k points per submap) and still resolve a 4 cm double wall

# --- odometry edges ------------------------------------------------------------------------------
ODOM_PRIOR = 100.0      # information added to every odometry edge (~100 matched points' worth) so the chain stays
                        # connected even when two consecutive submaps share no surface (camera turned away)

# --- loop closures -------------------------------------------------------------------------------
REVISIT_DIST = 1.5      # m: the camera of j came within 1.5 m of a camera of i, so both saw the same surfaces up close
LOOP_ICP_DISTS = (0.20, 0.10, 0.05)  # m, coarse-to-fine correspondence distance: 20 cm covers the worst drift
                                     # we expect (see LOOP_MAX_T_*), 5 cm is ~1.5 voxels for the final fit
LOOP_ICP_VOXELS = (0.08, 0.05, REG_VOXEL)  # m: coarse stages on thinned clouds (speed); the final one on the 3 cm cloud
LOOP_ICP_ITERS = 30
LOOP_MIN_FITNESS = 0.20  # >= 20 % of j's points land on i's surfaces: a revisit shares about one room of two 3 m
                         # submaps (1a83: 0.3 -> 0.2 took walls 2.45 -> 2.28 cm, doubles 14 -> 10 on the quick proxy;
                         # safety comes from the RMSE, geometry, plausibility tests and the line process)
LOOP_MAX_RMSE = 0.015    # m point-to-plane RMS of the matched points: 1.5 cm ~ LiDAR noise on a 3 cm voxel cloud
LOOP_MIN_CONSTRAINT = 0.05  # each translation direction must be pinned by >= ~5 % of the matched points (smallest
                            # eigenvalue of sum(n n^T)/N): a corridor or a single wall lets ICP slide, so reject it
LOOP_MAX_T0, LOOP_MAX_T_PER_M = 0.05, 0.02   # plausible position drift: 5 cm + 2 % of the path walked between the two
                                             # submaps (visual-inertial odometry drifts ~1 % of distance; 2x margin)
LOOP_MAX_DEG0, LOOP_MAX_DEG_PER_TURN = 1.0, 0.01  # plausible heading drift: 1 deg + 1 % of the angle the phone turned in
                                                  # between. Yaw drift comes from the gyro (scale-factor error ~0.5-1 %),
                                                  # so it grows with turning, not walking: a room scan spins 360 deg in seconds
LOOP_MAX_TILT_DEG = 1.0  # ARKit gravity is accurate to well under 1 deg; a loop that tilts more is a wrong match
LOOP_EVAL_DIST = 0.10    # m: residual before/after counts correspondences within 10 cm (covers the drift itself)

# --- pose graph ----------------------------------------------------------------------------------
EDGE_PRUNE = 0.25        # Open3D default: a loop whose line-process weight falls below 0.25 is dropped as an outlier
LOOP_PREFERENCE = 1.0    # Open3D default balance between loop edges and odometry edges

# --- plane method --------------------------------------------------------------------------------
PLANE_MAX_YAW_DEG = 3.0  # per-submap heading fix is at most 3 deg (ARKit yaw drift over a few minutes); wall normals
                         # further than this from the Manhattan axes are other geometry (angled walls, furniture)
PLANE_MIN_AREA = 0.5     # m^2 of vertical surface near the axes before a submap's yaw is trusted

# --- wall quality (plan coords) ------------------------------------------------------------------
DOUBLE_GAP = (0.04, 0.20)  # m: same surface seen twice 4-20 cm apart (< 4 cm is ordinary smear, > 20 cm is furniture)
DOUBLE_MIN_AREA = 0.25     # m^2 per face, same as a wall candidate (fp/geometry/plan.py MIN_WALL_AREA)
DOUBLE_MIN_OVERLAP = 0.5   # m: the two faces must run side by side for at least half a metre along the wall
DOUBLE_DIP = 0.5           # the histogram between the two peaks must drop below half the weaker one (two surfaces)
DOUBLE_FACE_TOL = 0.02     # m: points within 2 cm of a peak belong to that face
OVERLAP_BIN = 0.10         # m: along-wall occupancy cells used to measure the overlap


@dataclass
class DriftResult:
    frames: list[Frame]
    method: str
    submap_of_frame: np.ndarray
    submap_t: np.ndarray
    corrections: np.ndarray
    metrics: dict = field(default_factory=dict)
    loops: list[dict] = field(default_factory=list)


# ------------------------------------------------------------------------------------------------
# small geometry helpers

def _unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _rot_about(axis, angle):
    """Rodrigues: rotation by `angle` (rad) about unit `axis`."""
    k = _unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def _horizontal_basis(up):
    z = _unit(up)
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 0, 1.0])
    x = _unit(np.cross(a, z))
    return x, np.cross(z, x), z


def _yaw_of(R, up):
    """Signed heading change (rad) of R about `up`: where R sends a horizontal axis, measured around up."""
    x, y, _ = _horizontal_basis(up)
    rx = R @ x
    return float(np.arctan2(rx @ y, rx @ x))


def _tilt_deg(R, up):
    u = _unit(up)
    return float(np.degrees(np.arccos(np.clip((R @ u) @ u, -1, 1))))


def _rot_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _yaw_only(T, up, centre):
    """Keep gravity from ARKit: replace T's rotation by its rotation about `up` only, pivoting so that
    `centre` (the submap's camera centroid) still goes exactly where T sends it."""
    R = _rot_about(up, _yaw_of(T[:3, :3], up))
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3] = T[:3, :3] @ centre + T[:3, 3] - R @ centre
    return out


def _pivot(R, centre):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = centre - R @ centre
    return T


def _pcd(P, N=None):
    p = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(P, float)))
    if N is not None:
        p.normals = o3d.utility.Vector3dVector(np.asarray(N, float))
    return p


# ------------------------------------------------------------------------------------------------
# submaps

def split_submaps(t: np.ndarray, pos: np.ndarray, travel: float = SUBMAP_TRAVEL,
                  seconds: float = SUBMAP_SECONDS) -> np.ndarray:
    """Consecutive frames -> submap index, cutting at `travel` metres of path or `seconds`, whichever first."""
    n = len(t)
    idx = np.zeros(n, int)
    k, t0, d = 0, t[0] if n else 0.0, 0.0
    for i in range(1, n):
        d += float(np.linalg.norm(pos[i] - pos[i - 1]))
        if d >= travel or t[i] - t0 >= seconds:
            k, t0, d = k + 1, t[i], 0.0
        idx[i] = k
    # merge a too-short last submap into the previous one
    if k > 0 and (idx == k).sum() < MIN_SUBMAP_FRAMES:
        idx[idx == k] = k - 1
    return idx


@dataclass
class Submap:
    points: np.ndarray        # (N,3) world coords, REG_VOXEL-averaged
    normals: np.ndarray       # (N,3) unit, facing the cameras
    cams: np.ndarray          # (M,3) camera positions of its frames
    t: tuple[float, float]
    path_m: float             # cumulative camera travel at the submap's middle (for drift plausibility)
    turn_deg: float = 0.0     # cumulative |heading change| at the submap's middle (for drift plausibility)


def _build_cloud(P, cams, voxel=REG_VOXEL):
    pcd = _pcd(P).voxel_down_sample(voxel)
    if len(pcd.points) >= 10:
        pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=4 * voxel, max_nn=30))
        Q, N = np.asarray(pcd.points), np.asarray(pcd.normals).copy()
        _, j = cKDTree(cams).query(Q)
        N[np.einsum("ij,ij->i", N, cams[j] - Q) < 0] *= -1
        return Q, N
    return np.asarray(pcd.points), np.zeros((len(pcd.points), 3))


def build_submaps(frames: list[Frame], idx: np.ndarray, up=np.array([0, 1.0, 0])) -> list[Submap]:
    from fp.recon.lidar_fuse import frame_points
    with ThreadPoolExecutor() as ex:  # PNG decoding releases the GIL
        pts = list(ex.map(lambda f: frame_points(f, REG_PIX_STRIDE, REG_MAX_DEPTH)[0], frames))
    pos = np.array([f.T_wc[:3, 3] for f in frames])
    ts = np.array([f.timestamp for f in frames], float)
    path = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos, axis=0), axis=1))]
    # heading of the camera's viewing axis (OpenCV z) around up, unwrapped -> cumulative |turn| in degrees
    x, y, _ = _horizontal_basis(up)
    fwd = np.array([f.T_wc[:3, 2] for f in frames])
    head = np.unwrap(np.arctan2(fwd @ y, fwd @ x))
    turn = np.degrees(np.r_[0, np.cumsum(np.abs(np.diff(head)))])
    out = []
    for k in range(idx.max() + 1):
        sel = np.flatnonzero(idx == k)
        P = np.concatenate([pts[i] for i in sel]) if len(sel) else np.zeros((0, 3))
        Q, N = _build_cloud(P, pos[sel])
        out.append(Submap(Q, N, pos[sel], (float(ts[sel[0]]), float(ts[sel[-1]])), float(path[sel].mean()),
                          float(turn[sel].mean())))
    return out


# ------------------------------------------------------------------------------------------------
# registration

def point_to_plane_residual(src_P, dst_P, dst_N, T=np.eye(4), max_dist=LOOP_EVAL_DIST):
    """(RMS point-to-plane distance in m, fraction of source points matched, matched count, constraint)
    of src moved by T against dst. `constraint` = smallest eigenvalue of sum(n n^T)/count: how well
    the matched surfaces pin the weakest translation direction (0 = it can slide, 1/3 = a box corner)."""
    S = src_P @ T[:3, :3].T + T[:3, 3]
    d, j = cKDTree(dst_P).query(S, distance_upper_bound=max_dist)
    ok = np.isfinite(d)
    if ok.sum() < 10:
        return np.inf, float(ok.mean()) if len(ok) else 0.0, int(ok.sum()), 0.0
    n = dst_N[j[ok]]
    r = np.einsum("ij,ij->i", S[ok] - dst_P[j[ok]], n)
    lam = float(np.linalg.eigvalsh(n.T @ n / len(n))[0])
    return float(np.sqrt(np.mean(r ** 2))), float(ok.mean()), int(ok.sum()), lam


def register_loop(src: Submap, dst: Submap, path_between: float, up=np.array([0, 1.0, 0]),
                  turn_between: float = 0.0) -> dict:
    """Point-to-plane ICP of submap src onto dst from identity (coarse-to-fine), then the acceptance tests.
    Returns T (src -> dst), the numbers, and `accepted` / `reason`."""
    # ICP only needs the region both submaps could share: their boxes' intersection plus the search radius
    m = LOOP_ICP_DISTS[0]
    lo = np.maximum(src.points.min(0), dst.points.min(0)) - m
    hi = np.minimum(src.points.max(0), dst.points.max(0)) + m
    s, d = (_pcd(x.points, x.normals).crop(o3d.geometry.AxisAlignedBoundingBox(lo, hi)) for x in (src, dst))
    T = np.eye(4)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    crit = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=LOOP_ICP_ITERS)
    for vox, dist in zip(LOOP_ICP_VOXELS, LOOP_ICP_DISTS):
        sv, dv = (p.voxel_down_sample(vox) if vox > REG_VOXEL else p for p in (s, d))
        if len(sv.points) < 10 or len(dv.points) < 10:
            break
        dv.normalize_normals()
        T = o3d.pipelines.registration.registration_icp(sv, dv, dist, T, est, crit).transformation
    tilt = _tilt_deg(T[:3, :3], up)   # ICP's own tilt: large means it slid off (gravity is known to be right)
    rmse, fit, n, lam = point_to_plane_residual(src.points, dst.points, dst.normals, T, LOOP_ICP_DISTS[-1])
    t_cm = float(np.linalg.norm(T[:3, 3])) * 100
    # the translation that matters is how far the submap's own cameras move, not the offset about the origin
    c = src.cams.mean(0)
    move_cm = float(np.linalg.norm(T[:3, :3] @ c + T[:3, 3] - c)) * 100
    rot = _rot_deg(T[:3, :3])
    max_t = LOOP_MAX_T0 + LOOP_MAX_T_PER_M * path_between
    max_r = LOOP_MAX_DEG0 + LOOP_MAX_DEG_PER_TURN * turn_between
    reason = None
    if fit < LOOP_MIN_FITNESS:
        reason = f"fitness {fit:.2f} < {LOOP_MIN_FITNESS}"
    elif rmse > LOOP_MAX_RMSE:
        reason = f"rmse {rmse * 100:.1f} cm > {LOOP_MAX_RMSE * 100:.1f}"
    elif lam < LOOP_MIN_CONSTRAINT:
        reason = f"degenerate geometry (weakest direction {lam:.3f} < {LOOP_MIN_CONSTRAINT}): ICP can slide"
    elif move_cm > max_t * 100:
        reason = f"moves {move_cm:.1f} cm > plausible {max_t * 100:.1f} cm for {path_between:.0f} m walked"
    elif rot > max_r:
        reason = f"rotates {rot:.2f} deg > plausible {max_r:.2f} deg for {turn_between:.0f} deg turned"
    elif tilt > LOOP_MAX_TILT_DEG:
        reason = f"tilts {tilt:.2f} deg > {LOOP_MAX_TILT_DEG} (gravity is trusted)"
    return {"T": T, "fitness": round(fit, 3), "rmse_cm": round(rmse * 100, 2), "constraint": round(lam, 3),
            "move_cm": round(move_cm, 2), "rot_deg": round(rot, 3), "tilt_deg": round(tilt, 3),
            "t_cm": round(t_cm, 2), "path_between_m": round(path_between, 1), "turn_between_deg": round(turn_between),
            "accepted": reason is None, "reason": reason}


def loop_candidates(subs: list[Submap]) -> list[tuple[int, int]]:
    """Non-consecutive pairs whose clouds' boxes overlap and where the camera came back within REVISIT_DIST."""
    boxes = [(s.points.min(0), s.points.max(0)) if len(s.points) else None for s in subs]
    trees = [cKDTree(s.cams) for s in subs]
    out = []
    for i in range(len(subs)):
        for j in range(i + 2, len(subs)):
            if boxes[i] is None or boxes[j] is None:
                continue
            if np.any(boxes[i][1] < boxes[j][0]) or np.any(boxes[j][1] < boxes[i][0]):
                continue
            d, _ = trees[i].query(subs[j].cams, distance_upper_bound=REVISIT_DIST)
            if np.isfinite(d).any():
                out.append((i, j))
    return out


def register_candidates(subs: list[Submap], up) -> list[dict]:
    """ICP every loop candidate (threads: Open3D releases the GIL), each result tagged with its pair (i, j)."""
    def one(ij):
        i, j = ij
        r = register_loop(subs[j], subs[i], abs(subs[j].path_m - subs[i].path_m), up,
                          abs(subs[j].turn_deg - subs[i].turn_deg))
        r.update(i=i, j=j)
        return r
    with ThreadPoolExecutor() as ex:
        return list(ex.map(one, loop_candidates(subs)))


# ------------------------------------------------------------------------------------------------
# the two methods on submap clouds (testable without capture data)

def posegraph_corrections(subs: list[Submap], up=np.array([0, 1.0, 0])):
    """Submaps -> (corrections (n,4,4), loops list, metrics dict)."""
    up = _unit(up)
    n = len(subs)
    reg = o3d.pipelines.registration
    pcds = [_pcd(s.points, s.normals) for s in subs]
    pg = reg.PoseGraph()
    for _ in range(n):
        pg.nodes.append(reg.PoseGraphNode(np.eye(4)))
    for i in range(n - 1):  # odometry: ARKit says identity; weight = shared surface
        info = np.eye(6) * ODOM_PRIOR
        if len(subs[i].points) and len(subs[i + 1].points):
            info = info + reg.get_information_matrix_from_point_clouds(pcds[i + 1], pcds[i], LOOP_ICP_DISTS[-1],
                                                                      np.eye(4))
        pg.edges.append(reg.PoseGraphEdge(i + 1, i, np.eye(4), info, uncertain=False))

    loops = register_candidates(subs, up)
    for r in loops:
        i, j = r["i"], r["j"]
        if r["accepted"]:
            info = reg.get_information_matrix_from_point_clouds(pcds[j], pcds[i], LOOP_ICP_DISTS[-1], r["T"])
            pg.edges.append(reg.PoseGraphEdge(j, i, r["T"], info, uncertain=True))

    acc = [r for r in loops if r["accepted"]]
    C = np.repeat(np.eye(4)[None], n, 0)
    if acc:
        option = reg.GlobalOptimizationOption(max_correspondence_distance=LOOP_ICP_DISTS[-1],
                                              edge_prune_threshold=EDGE_PRUNE,
                                              preference_loop_closure=LOOP_PREFERENCE, reference_node=0)
        with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
            reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                                    reg.GlobalOptimizationConvergenceCriteria(), option)
        kept = {(e.source_node_id, e.target_node_id) for e in pg.edges if e.uncertain}
        for r in acc:
            if (r["j"], r["i"]) not in kept:
                r["accepted"], r["reason"] = False, "pruned by the pose graph (disagrees with the other loops)"
        if any(r["accepted"] for r in acc):
            for k in range(n):  # gravity stays ARKit's: keep only the yaw part, pivoting at the cameras
                C[k] = _yaw_only(np.asarray(pg.nodes[k].pose), up, subs[k].cams.mean(0))
    return C, loops


def plane_corrections(subs: list[Submap], up=np.array([0, 1.0, 0])):
    """Global Manhattan yaw from all vertical-surface normals, then per submap the median deviation of its own
    wall normals from those axes (within +-PLANE_MAX_YAW_DEG) is undone by a yaw about its camera centroid."""
    x, y, z = _horizontal_basis(up)

    def angles(N):
        vert = np.abs(N @ z) < 0.2
        return np.mod(np.arctan2(N[vert] @ y, N[vert] @ x), np.pi / 2)

    allang = np.concatenate([angles(s.normals) for s in subs if len(s.normals)]) if subs else np.zeros(0)
    C = np.repeat(np.eye(4)[None], len(subs), 0)
    yaws = np.zeros(len(subs))
    if len(allang) == 0:
        return C, yaws, None
    h, e = np.histogram(allang, bins=180, range=(0, np.pi / 2))   # same histogram as fp/geometry/align.py
    h = np.convolve(np.r_[h[-3:], h, h[:3]], np.ones(5) / 5, "same")[3:-3]
    g = 0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1])
    lim = np.radians(PLANE_MAX_YAW_DEG)
    # refine the 0.5 deg bin (smoothed over 2.5 deg) to the median of the normals near it
    dg = np.mod(allang - g + np.pi / 4, np.pi / 2) - np.pi / 4
    g = float(g + np.median(dg[np.abs(dg) <= lim]))
    for k, s in enumerate(subs):
        if not len(s.normals):
            continue
        dev = np.mod(angles(s.normals) - g + np.pi / 4, np.pi / 2) - np.pi / 4   # wrap to +-45 deg
        dev = dev[np.abs(dev) <= lim]
        if len(dev) * REG_VOXEL ** 2 < PLANE_MIN_AREA:
            continue
        yaws[k] = -float(np.median(dev))
        C[k] = _pivot(_rot_about(up, yaws[k]), s.cams.mean(0))
    return C, yaws, float(np.degrees(g))


def loop_residuals(subs, loops, C):
    """Median point-to-plane RMS (cm) over accepted loops, before (identity) and after the corrections."""
    b, a = [], []
    for r in loops:
        if not r["accepted"]:
            continue
        i, j = r["i"], r["j"]
        rb = point_to_plane_residual(subs[j].points, subs[i].points, subs[i].normals)[0]
        ra = point_to_plane_residual(subs[j].points, subs[i].points, subs[i].normals,
                                     np.linalg.inv(C[i]) @ C[j])[0]
        r["residual_before_cm"], r["residual_after_cm"] = round(rb * 100, 2), round(ra * 100, 2)
        b.append(rb)
        a.append(ra)
    return (float(np.median(b)) * 100 if b else float("nan"), float(np.median(a)) * 100 if a else float("nan"))


def odometry_residuals(subs, C):
    """Median point-to-plane RMS (cm) between consecutive submaps, before and after the corrections."""
    b, a = [], []
    for i in range(len(subs) - 1):
        s, d = subs[i + 1], subs[i]
        if len(s.points) < 10 or len(d.points) < 10:
            continue
        rb = point_to_plane_residual(s.points, d.points, d.normals)[0]
        ra = point_to_plane_residual(s.points, d.points, d.normals, np.linalg.inv(C[i]) @ C[i + 1])[0]
        if np.isfinite(rb) and np.isfinite(ra):
            b.append(rb)
            a.append(ra)
    return (float(np.median(b)) * 100 if b else float("nan"), float(np.median(a)) * 100 if a else float("nan"))


# ------------------------------------------------------------------------------------------------
# public API

METHODS = ("posegraph", "plane", "posegraph+plane")


def _moved(s: Submap, C) -> Submap:
    return dataclasses.replace(s, points=s.points @ C[:3, :3].T + C[:3, 3], normals=s.normals @ C[:3, :3].T,
                               cams=s.cams @ C[:3, :3].T + C[:3, 3])


def correct_drift(frames: list[Frame], up=np.array([0, 1.0, 0]), method: str = "posegraph") -> DriftResult:
    """Frames with ARKit poses -> copies with drift-corrected T_wc (inputs untouched)."""
    if method not in METHODS:
        raise ValueError(f"unknown drift method {method!r}")
    t0 = time.perf_counter()
    up = _unit(up)
    n = len(frames)
    ts = np.array([f.timestamp for f in frames], float)
    pos = np.array([f.T_wc[:3, 3] for f in frames]) if n else np.zeros((0, 3))
    idx = split_submaps(ts, pos) if n else np.zeros(0, int)
    n_sub = int(idx.max()) + 1 if n else 0
    submap_t = np.array([[ts[idx == k][0], ts[idx == k][-1]] for k in range(n_sub)]).reshape(-1, 2)
    C = np.repeat(np.eye(4)[None], n_sub, 0)
    metrics = {"n_frames": n, "n_submaps": n_sub, "n_loop_candidates": 0, "n_loop_accepted": 0,
               "loop_rmse_before_cm": float("nan"), "loop_rmse_after_cm": float("nan")}
    loops: list[dict] = []
    if n_sub < 2:
        desc = f"no correction: {n_sub} submap(s) ({SUBMAP_TRAVEL:g} m / {SUBMAP_SECONDS:g} s), nothing to close a loop with"
    else:
        tb = time.perf_counter()
        subs = build_submaps(frames, idx, up)
        metrics["seconds_submaps"] = round(time.perf_counter() - tb, 2)
        parts = []
        if method.startswith("posegraph"):
            C, loops = posegraph_corrections(subs, up)
            n_acc = sum(r["accepted"] for r in loops)
            parts.append(f"pose graph over {n_sub} submaps ({SUBMAP_TRAVEL:g} m / {SUBMAP_SECONDS:g} s), "
                         f"{n_acc} ICP loop closures accepted of {len(loops)} candidates")
            if not loops:
                parts[-1] += " (camera never revisited a place: poses kept)"
            elif not n_acc:
                parts[-1] += " (none passed the checks: poses kept)"
        else:
            # the plane method has no loops of its own; still score the same loop candidates so the
            # before/after residual is comparable with the pose graph
            loops = register_candidates(subs, up)
        if method.endswith("plane"):
            # snap the (pose-graph corrected, if any) submaps' walls onto the global Manhattan axes
            Cp, yaws, g = plane_corrections([_moved(s, c) for s, c in zip(subs, C)], up)
            C = np.einsum("kij,kjl->kil", Cp, C)
            metrics["plane_global_yaw_deg"] = round(g, 2) if g is not None else float("nan")
            metrics["plane_yaw_abs_median_deg"] = round(float(np.degrees(np.median(np.abs(yaws)))), 3)
            parts.append(f"plane-anchored yaw{'' if parts else f' over {n_sub} submaps'}: each submap's walls snapped "
                         f"to the global Manhattan axes (|yaw| <= {PLANE_MAX_YAW_DEG:g} deg)")
        desc = "; then ".join(parts)
        metrics["n_loop_candidates"] = len(loops)
        metrics["n_loop_accepted"] = int(sum(r["accepted"] for r in loops))
        before, after = loop_residuals(subs, loops, C)
        metrics["loop_rmse_before_cm"], metrics["loop_rmse_after_cm"] = round(before, 2), round(after, 2)
        # consecutive submaps share surfaces too: a correction must not tear them apart
        before, after = odometry_residuals(subs, C)
        metrics["odom_rmse_before_cm"], metrics["odom_rmse_after_cm"] = round(before, 2), round(after, 2)
        for r in loops:
            r["T"] = np.asarray(r["T"]).round(6).tolist()

    # how far each submap moved: displacement of its own cameras, and its rotation
    moves, rots = [0.0], [0.0]
    for k in range(n_sub):
        c = pos[idx == k]
        moves.append(float(np.max(np.linalg.norm(c @ C[k][:3, :3].T + C[k][:3, 3] - c, axis=1))) * 100)
        rots.append(_rot_deg(C[k][:3, :3]))
    metrics["max_correction_cm"] = round(max(moves), 2)
    metrics["max_correction_deg"] = round(max(rots), 3)
    out = []
    for f, k in zip(frames, idx):
        out.append(dataclasses.replace(f, T_wc=C[k] @ f.T_wc))
    metrics["seconds"] = round(time.perf_counter() - t0, 2)
    return DriftResult(frames=out, method=desc, submap_of_frame=idx, submap_t=submap_t, corrections=C,
                       metrics=metrics, loops=loops)


def correct_poses(result: DriftResult, t: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Apply the correction of the submap whose time range holds each timestamp (the nearest submap in time
    otherwise) to any pose array, e.g. the full unfiltered camera path. T_wc: (n,4,4) -> (n,4,4)."""
    t = np.asarray(t, float)
    T_wc = np.asarray(T_wc, float)
    if len(result.corrections) == 0 or len(t) == 0:
        return T_wc.copy()
    s = result.submap_t
    gap = np.maximum(s[None, :, 0] - t[:, None], 0) + np.maximum(t[:, None] - s[None, :, 1], 0)
    k = np.argmin(gap, axis=1)   # 0 inside a submap's range; ties go to the earlier submap
    return result.corrections[k] @ T_wc


# ------------------------------------------------------------------------------------------------
# wall quality (the ablation metric)

def _occupancy(v, lo):
    return set(np.floor((v - lo) / OVERLAP_BIN).astype(int).tolist())


def wall_quality(P: np.ndarray, N: np.ndarray, voxel: float) -> dict:
    """Plan-frame cloud -> median wall thickness (same lines and spread as the pipeline) and the double walls:
    two parallel surfaces 4-20 cm apart FACING THE SAME WAY (one physical surface seen twice by drifted
    passes; a partition wall's two faces point opposite ways and never pair up), side by side along
    the wall for >= DOUBLE_MIN_OVERLAP, each with real area."""
    from fp.geometry.plan import WALL_BAND, _wall_lines, spread  # noqa: F401
    lines = _wall_lines(P, N, None, voxel)
    th = [l["spread_m"] * 100 for l in lines]
    zlo, zhi = WALL_BAND
    band = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] > zlo) & (P[:, 2] < zhi)
    doubles = []
    for ax in (0, 1):
        for sign in (1, -1):
            m = band & (sign * N[:, ax] > 0.9)
            v, w = P[m, ax], P[m, 1 - ax]
            if len(v) < 10:
                continue
            e = np.arange(v.min() - 0.05, v.max() + 0.06, 0.01)
            h, e = np.histogram(v, e)
            hs = np.convolve(h, np.ones(3), "same")
            c = 0.5 * (e[:-1] + e[1:])
            pk = [i for i in range(1, len(hs) - 1)
                  if hs[i] >= hs[i - 1] and hs[i] > hs[i + 1] and hs[i] * voxel ** 2 >= DOUBLE_MIN_AREA]
            wlo = w.min()
            # neighbouring peaks only: three stripes of one surface are two doubles, not three
            for a, b in zip(pk[:-1], pk[1:]):
                gap = c[b] - c[a]
                if not DOUBLE_GAP[0] - 1e-9 <= gap <= DOUBLE_GAP[1] + 1e-9:
                    continue
                if hs[a:b + 1].min() > DOUBLE_DIP * min(hs[a], hs[b]):
                    continue
                fa, fb = np.abs(v - c[a]) <= DOUBLE_FACE_TOL, np.abs(v - c[b]) <= DOUBLE_FACE_TOL
                both = _occupancy(w[fa], wlo) & _occupancy(w[fb], wlo)
                ov = len(both) * OVERLAP_BIN
                if ov >= DOUBLE_MIN_OVERLAP:
                    doubles.append({"axis": ax, "coord_a": round(float(c[a]), 3), "coord_b": round(float(c[b]), 3),
                                    "gap_cm": round(float(gap) * 100, 1), "overlap_m": round(ov, 2), "faces": sign,
                                    # extent along the wall (other plan axis) where both faces are present
                                    "lo": round(float(wlo + min(both) * OVERLAP_BIN), 2),
                                    "hi": round(float(wlo + (max(both) + 1) * OVERLAP_BIN), 2)})
    return {"thickness_median_cm": float(np.median(th)) if th else float("nan"), "n_walls": len(lines),
            "double_walls": doubles}
