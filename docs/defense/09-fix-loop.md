# 09 — The fix loop: defense notes

**The numbers to remember:** on the c7d2 video, the room split **dropped 78 % of the free floor** (40.8 of 52.5 m²). **82 % of the
"walls" that cut it were seen through by other frames.** After the fix: footprint **−79 % → +22 %**, wall p90 **105 % → 76 %** (predicted
76 %). The gate (±3 %) **still fails**: the camera poses are off by metres.

**1. How did you know it was the root cause and not a symptom?**
I dumped the inputs of the room split from a real run and replayed it one step at a time, measuring the area left after each
step. The footprint was 84 m², more than the LiDAR plan. 57 m² was left after cutting tall walls, but only 11.7 m² ended up in
rooms. So the loss happens in the split, not in the reconstruction or the scoring. Then I asked why the floor was in slivers:
82 % of the tall cells had sight lines from at least 3 frames passing through them. A real wall blocks sight lines, so those
cells are wall copies from badly registered parts of the capture. The deeper cause is the camera poses. I say that openly, and
it is why the gate can't pass.

**2. What alternative did you rule out, and how?**
- **"The cloud is just bad, the split is fine."** Partly true: the skeptic subagent fitted the video camera path to the LiDAR
  odometry and got scale 0.48 with a median residual of 2.4 m. But a bad cloud doesn't explain losing 78 % of the floor. That
  loss came from a stage that silently drops floor.
- **The same ray rule on LiDAR.** It removes 42–75 % of real LiDAR wall cells, because in 2,600 frames rays graze along walls.
  So it applies only to the camera tiers, which also have no drift correction.
- **Simpler patches.** Giving the slivers to the nearest room made p90 worse (142 %). Falling back to one room gave an 82 m²
  blob that matches no LiDAR room, so not a single wall could be scored.

**3. Why was your prediction right or wrong?**
I predicted p90 76 % and footprint +22 %, and got exactly that (`fixloop/RESULT.md`). One thing I did not predict: the video calibration gate now passes (92 %), but only because the refit widened the video wall interval (b 2.41 → 4.09), so it is not an accuracy gain. The prediction came from the same rule
patched into a real run before I wrote the code, and the shipped code is deterministic on the cached model outputs. What I
did not predict is the gate passing, and it didn't: the 3 % gate needs metric-accurate poses. The c00a video got slightly
worse (p90 63 % → 71 %, footprint −12 % → +19 %). I predicted that in the declaration and kept it visible.

**4. Isn't restricting the rule to camera tiers a per-tier hack?**
No. It's a sensor model, not per-capture tuning. LiDAR walls are measured and drift-corrected (03's pose graph). Camera-tier
walls come from predicted depth and model poses with no drift correction. The rule asks the question the LiDAR tier already
answered with drift correction: is this wall consistent with the other frames?

**5. Did you tune a threshold on the evaluation capture?**
No new threshold. "Seen through by ≥ 3 frames" is the footprint's existing free-space rule (MIN_RAY_HITS). The only new constant
is the 10 cm stop-short, so a ray doesn't count its own wall. Its value doesn't matter: 5, 10 and 20 cm keep 79, 78 and 77 m²
of floor. I reported that range in the declaration.

**6. What would actually pass the gate?**
Better camera-tier poses: denser keyframes for MapAnything (chunks currently span about 20 s with 6 shared frames, and merges
disagree by 0.3–1.5 m), or a pose-graph and loop-closure pass like LiDAR gets. That means new model passes and a work order of
its own. ±3 % on a whole flat from video is very hard in any case.

**7. How can someone reproduce before and after?**
`make fix-before` and `make fix-after`. Each checks out its tag in a temporary git worktree, links in the raw captures and the
model-output cache, and reruns the whole benchmark there, so the main checkout is never touched. `fixloop/before/` and
`fixloop/after/` hold the tables and the key debug images.

**8. When does your rule get it wrong?**
A real wall that 3 or more frames see through gets removed: glass, or a misregistered frame whose rays cross a real wall. The
verifier built that case, and two rooms merged into one. It is limited to the camera tiers, which already have huge intervals.
LiDAR, the reference tier, never uses the rule.
