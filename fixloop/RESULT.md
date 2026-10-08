# Fix loop result

**Gate:** video wall lengths within ±3 % on `c7d28f72c6-video`, scored against the LiDAR plan of the same capture.
**Outcome:** the root cause was found, the fix shipped, and every predicted number was hit exactly. **The gate still fails**
(35× → 25.5× the threshold), as the declaration predicted, because the camera-tier poses are metres off.

Both sides are regenerated from raw inputs by `make fix-before` (988 s) and `make fix-after` (962 s). Each checks out its tag in
a temporary worktree, links `data/` and the model-output cache `out/_cache`, and runs the whole benchmark (every capture,
calibration fit, rerun). Outputs: `fixloop/before/`, `fixloop/after/`. The before tables match the committed 07
`eval/BENCHMARK.md` gate for gate.

## Before vs after vs predicted

| number | before (`fix-before`) | predicted (declaration) | after (`fix-after`) |
|---|---|---|---|
| **c7d2 video wall p90 \|err\|** (the gate) | **105 %** | **76 %** | **76 %** |
| c7d2 video wall median \|err\| | 75 % | 37 % | 37 % |
| c7d2 video walls within ±3 % | 0 % of 6 | 0 % | 0 % of 7 |
| gate result | FAIL (35×) | FAIL (~25×) | **FAIL (25.5×)** |
| c7d2 video footprint | 13.2 vs 62.64 m² (−79 %) | +22 % (76.5 m²) | 76.47 m² (+22 %) |
| c7d2 video footprint IoU / rooms | 0.18 / 2 vs 9 | 0.59 / 3 | 0.59 / 3 vs 9 |
| c00a video footprint | −12 % | +19 % | +19 % (29.02 vs 24.48 m²) |
| c00a video wall p90 | 63 % | 71 % | 71 % (worse) |
| LiDAR and photo plans | — | unchanged | unchanged (c7d2 LiDAR rooms identical; photo rows identical) |
| status counts | PASS 2, FAIL 14 | — | PASS 3, FAIL 13 |

**Not predicted:** the video calibration gate (held-out coverage of the stated 90 %) went from 78 % of 9 to **92 % of 12, PASS**.
That is not an accuracy gain. With more video walls matched, the refit **widened** the video wall interval from b = 2.41 to
**4.09** (±409 % of a wall). The interval says "we don't know" more loudly, so the truth falls inside it more often. Treat this
pass as honest bookkeeping, not as progress.

## The readable diff
`git diff fix-before fix-after --stat -- fp tests scripts Makefile`:
```
 Makefile             | 10 +++++++++-
 fp/cli.py            | 13 +++++++++++--
 fp/geometry/plan.py  | 37 ++++++++++++++++++++++++++++++++-----
 scripts/fixloop.sh   | 39 +++++++++++++++++++++++++++++++++++++++
 tests/test_stitch.py | 33 +++++++++++++++++++++++++++++++++
```
(The rest of the full `--stat` is `fixloop/FIX_DECLARATION.md` and `fixloop/before/`.)

The key hunks:
- `fp/geometry/plan.py` adds **`seen_through_walls(tall, rays, o)`**: tall-wall cells crossed by rays from ≥ `MIN_RAY_HITS`
  frames. `MIN_RAY_HITS` is the footprint's existing free-space rule. Each ray stops `RAY_STOP_SHORT` = 10 cm before its point,
  so a wall isn't counted as seen through by its own hits. `_seen_through` gains the `stop_short` argument for this.
  `extract_rooms(..., predicted_depth=False)` removes those cells from `tall` before the wall cut and the doorway bars, and
  returns `phantom_m2` and `unroomed_m2` (free floor that no room seed reached).
- `fp/cli.py` passes `predicted_depth=tier != "lidar"`. It warns how much wall surface was ignored and how much seen floor is
  in no room. `debug/rooms_split.png` draws the ignored cells in orange.
- `tests/test_stitch.py`: a wall copy inside one room that frames look through gives 1 room (it was 2); a real partition
  whose rays end on it stays as 2 rooms.

## Evidence images
- `fixloop/before/c7d28f72c6-video/rooms_split.png`: the footprint (grey) is cut by a lattice of black "tall walls", and only
  two small rooms get a colour.
- `fixloop/after/c7d28f72c6-video/rooms_split.png`: the same lattice is orange (ignored), and the footprint is three rooms.
- `fixloop/*/c7d28f72c6-video/fusion_topdown.png`: the jumbled, multiply registered video cloud, which is the remaining cause.

## Why it fell short of the gate
The fix removes the stage that **threw floor away**. It cannot correct the geometry it is handed:
- The video cloud's footprint is already 84 m² against the LiDAR plan's 62.6 m².
- The skeptic subagent fitted the video camera path to the LiDAR odometry: Sim(3) scale 0.48, median residual 2.4 m.
  MapAnything chunk merges disagree by 0.3–1.5 m.

So the footprint swung from −79 % to +22 %, and the walls are still 37 % off at the median. Passing ±3 % needs better
camera-tier poses: denser keyframes and chunk overlap, or a pose-graph and loop-closure pass like the LiDAR tier's. That means
new model passes, and it is out of scope for a plan-stage fix. A cost of the fix: on c00a video the single room now
overshoots (+19 %), where before it undershot (−12 %).

## Known limitation (found by the verifier)
On camera tiers, a **real wall that ≥ 3 frames see through** is removed: a glass partition, or a misregistered frame whose
rays cross a real wall. In a synthetic test, two rooms with a solid partition became one 18.3 m² room when room A's frames
"saw" room B's far wall. The fix recovers floor area, not room structure. On c7d2 video, R1 (61 m²) still merges most of the
rooms LiDAR splits into 9. An independent rerun reproduced every after number (p90 76 %, footprint +22 %, IoU 0.591), and
the LiDAR plans are identical before and after (ids, areas, walls, polygons).
