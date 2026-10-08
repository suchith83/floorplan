# Fix declaration (written and committed before the fix)

**Worst gate** (`eval/BENCHMARK.md` at tag `fix-before`): video wall lengths within ±3 % on `c7d28f72c6-video`,
scored against the LiDAR plan of the same capture. **0 % of 6 matched walls within; median |err| 75 %, p90 105 %**
(35× the threshold). Same capture: footprint **13.2 vs 62.64 m² (−79 %)**, 2 rooms vs 9, footprint IoU 0.18.

## Where the error enters (evidence)
Area kept at each step of `extract_rooms` (fp/geometry/plan.py) on the c7d2 video cloud. The inputs were dumped from a real
`fp run`, then the steps were replayed one at a time:

| step | c7d2 video | c7d2 LiDAR |
|---|---|---|
| footprint (floor + seen-through rays + camera path) | 84.0 m² | 73.0 m² |
| minus dilated tall-wall mask and doorway bars | 57.0 | 62.8 |
| after the 2-cell opening | 52.5 | 62.8 |
| **in a room after `split_rooms`** | **11.7** | 61.3 |

- The footprint is fine (84 m², +34 % vs the LiDAR plan). The area is lost in the **room split**.
- `tall_wall_mask` (added by 03's whole-flat stitch) treats every cell with vertical surface in ≥ 2 height bands as a
  partition wall and cuts it out. The video cloud holds 14.9 m² of such cells (LiDAR 7.2 m²) because its walls are duplicated
  and smeared (86 wall lines, median 8.6 cm thick; 62 double walls vs 15 on LiDAR).
- The free floor breaks into **195 slivers**; the median free cell is 8 cm from a "wall". `split_rooms` only seeds regions
  with a cell ≥ 0.4 m from an edge, and floor in a region with no seed is silently left out: **40.8 of 52.5 m² (78 %) is
  dropped**. On LiDAR the same step drops 2 %.
- **82 % of the video's tall cells were seen through**: camera→point rays from ≥ 3 frames (MIN_RAY_HITS, the same rule the
  footprint uses for free space) cross them, stopping 10 cm short of their own point. These are walls from a misregistered
  part of the capture, standing in air that other frames looked through.

**Root-cause hypothesis.** The stitch's wall cut assumes a drift-corrected cloud, where every tall surface is a real wall. The
LiDAR tier gets that (03's pose-graph drift correction). The camera tiers get no drift correction at all (`fp/cli.py`:
`drift_correction and tier == "lidar"`). So their misregistered copies of walls reach the cut, chop the floor into slivers,
and the split drops the slivers. Upstream of that, the MapAnything poses themselves are poor. The skeptic subagent fitted the
video camera path to the LiDAR odometry: Sim(3) scale 0.48, residual median 2.4 m. That puts the ±3 % gate out of reach for any
plan-stage fix.

## Alternatives ruled out (experiments that change one thing)
- **Ray rule on every tier:** it removes 42–75 % of the real LiDAR wall cells (rays graze along walls in 2,600 frames), and
  c7d2 LiDAR goes from 9 rooms to 8. The LiDAR reference would change, so this was rejected for the LiDAR tier.
- **Pass > hit vote** (seen through more often than seen): LiDAR still loses 44 % of its tall cells. Rejected.
- **Only cut tall cells on an axis wall line:** LiDAR goes from 9 rooms to 8. Rejected.
- **Give seedless floor to the nearest room** (`split_rooms`): video footprint −45 %, p90 142 % (worse).
- **One-room fallback when > 50 % of the floor is seedless:** a single 82 m² room. No LiDAR room overlaps it at IoU ≥ 0.2, so no
  wall can be scored at all. Rejected.

## The fix
For **predicted-depth clouds only** (the video and photo tiers, which have no drift correction), a tall cell crossed by rays
from ≥ MIN_RAY_HITS frames is free space, not a wall: a wall must block the sight lines that pass through it. Rays stop
10 cm short of their measured point, so a wall's own hits don't count. The value of the margin barely matters: 5, 10 and
20 cm keep 79, 78 and 77 m² of floor in rooms. The LiDAR tier is unchanged by construction, so the reference it is scored
against stays fixed. No new tuned threshold: MIN_RAY_HITS is the existing free-space rule. A warning reports any free floor
the split still leaves out of every room.

## Predicted numbers after the fix
From the experiment with the rule patched into a real `fp run`; the calibration refit doesn't change geometry.
- **c7d2 video, the worst gate: wall p90 |err| 105 % → 76 %**, median 75 % → 37 %, within ±3 % 0 % → 0 %. **Still FAIL**
  (35× → ~25× the threshold).
- c7d2 video footprint −79 % → **+22 %** (76.5 vs 62.6 m²), IoU 0.18 → 0.59, 3 rooms vs 9.
- Side effect, c00a video: footprint −12 % → +19 %, wall p90 63 % → 71 % (slightly worse).
- LiDAR plans and photo plans: unchanged.
