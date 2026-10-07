# Defense notes 03: drift correction and the stitched whole-flat plan (LiDAR)

**The numbers to remember:** on the floor-only flat scan (1a8384c3f6) drift correction takes the median
wall thickness from **4.2 cm to 1.9 cm** and the double walls from **13 to 7**. The two flat scans agree
to about **±5 cm** per wall, not the 1 cm target.

**What drift does ARKit already correct, and what's left?**
ARKit fuses the camera with the IMU, so gravity is measured directly and accurate to well under a degree.
Frame-to-frame motion is good to millimetres. What it can't measure is absolute heading and position, so those
slowly drift. On 1a8384c3f6 the walls of each 8-second piece of the walk turn steadily from +3.0° to −0.8°
over 115 s, which is heading drift you can see. A wall seen at minute 1 and again at minute 3 lands a few
cm apart: a "double wall". So I only correct heading and position, never the tilt.

**Why submaps + ICP + a pose graph?**
Within a few metres or 8 seconds ARKit is accurate, so I cut the walk into rigid pieces (submaps) and fuse
each one. Where the camera came back to a place it had seen before, I align the two pieces with
point-to-plane ICP (Open3D). The offset ICP finds is the drift accumulated in between. A pose graph spreads
all those offsets over the whole walk at once, so each piece gets one consistent correction. It's the
textbook loop-closure recipe, and every step can be checked with a number.

**How is a false loop closure rejected?**
Five independent tests, then the pose graph itself. ICP must match ≥ 20% of the points with ≤ 1.5 cm RMSE.
The geometry must pin all three directions, so a plain corridor, where ICP can slide, is rejected. The
correction must be one ARKit could plausibly have drifted (5 cm + 2% of the distance walked, 1° + 1% of the
angle turned). It must not tilt more than 1°. Finally, the pose graph's line process drops loops that
disagree with the others: on c7d28f72c6, 5 loops passed ICP but were pruned there.

**On 1a83 one submap moved 68 cm and 7°. How is that plausible?**
Most of it is the whole map turning, not a distortion. Submap 0 is held fixed, and its own heading happens to be
3° off from most of the walk, so everything else rotates toward it. At 10 m from the start, 4° of turn is already
70 cm. Alignment re-finds the wall axes afterwards, so a rigid turn changes nothing in the plan. What matters is the
relative heading between submaps: their wall angles spread 1.5° before and 0.8° after.

**What can't it fix?**
Drift bigger than the plausibility test allows. The verifier injected 10° of yaw into half of 1a83: loops that
would need that much correction were rejected (8 → 5 accepted), and only ~2° was removed. Loop RMSE still looked
fine, because it only scores the accepted loops. That's why the plan now warns when correction widens the heading
spread across submaps by more than 0.5°. Rejecting a big correction is the safe failure: a wrong loop closure
would bend the whole flat.

**What does the ablation prove?** (`out/<id>/debug/drift_ablation.png`, off left, on right, doubles in red)
On all three captures, correction lowers the loop residuals (1a83 3.3 → 1.8 cm, c7d2 3.2 → 2.1 cm,
c00a 4.2 → 1.9 cm), the double walls (1a83 13 → 7, c7d2 32 → 15) and the wall thickness. The cheaper
fallback (snap each piece's walls to the Manhattan axes) never lowered the loop residuals, so I kept the
pose graph.

**Why do the two flat scans differ, and why is the gate missed?**
They're the same flat: 69% of the wall points land within 5 cm after a 90° turn. Wall-to-wall spans in the
same room still differ by a median of 5.5 cm. One reason is coverage: the 115 s scan never looks up, while
the 215 s scan has a ceiling pass and more revisits, so different stretches of wall were seen. The
bigger reason is that drift correction trades local for global accuracy. Uncorrected, the two scans' spans
agree to 1.9 cm, because ARKit is locally excellent. But then 40% more double walls appear and rooms fail to
match. The ICP corrections are good to 1–2 cm each, and that error ends up inside the rooms. I report both
and don't tune thresholds against this gate.

**How are rooms split and kept from overlapping?**
Tall walls (seen in at least 2 of 3 height bands) are cut out of the floor footprint. Gaps of 0.5–1.3 m
between runs of wall on the same line are barred as doorways. A watershed then splits the footprint at
those narrow points. Two rooms may overlap by at most 1% of the smaller one; above that, the overlap goes
to the room whose own floor evidence covers more of it, and a warning says so. R1 is the largest room, and a
room ≤ 1.6 m wide and ≥ 2.5 m long is named "connector".

**How fast is it?**
Under 10 minutes on the Mac without frame subsampling beyond the quality filter: 1a8384c3f6 in about
1.5 min and c7d28f72c6 in about 3–5 min with drift on, which includes fusing the cloud a second time with
the original poses for the ablation. The drift step itself takes 1–12 s.
