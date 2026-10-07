# Defense notes 01: the output contract

**The one number to remember:** **90%**. Every `[lo, hi]` in `plan.json` is a stated 90% interval:
about 9 in 10 true values should fall inside it, and the benchmark checks that.

**Why intervals and not a 0–1 confidence score?**
A score like "0.64" can't be checked against a tape measure, and nobody knows what it means in
centimetres. An interval says "4.20 m, between 4.18 and 4.22", which a homeowner understands and an
evaluator can test. The brief scores calibration at every tier, and you can only calibrate something
that makes a checkable claim.

**What exactly does a "90% interval" promise?**
If you collected many measurements across rooms and checked them with a laser, about 90% of the true
values would fall inside their stated ranges. It's a claim about the whole set, not a guarantee for
one wall. If only 60% fall inside, the intervals are overconfident and I widen them. If 100% do and
the ranges are huge, they're useless. Work order 07 reports stated against observed coverage per tier.

**What if something wasn't measured, like the ceiling in the floor-only scan?**
Then `value` is null and `observed` is false, and the report says "not observed". I never invent a
number. If I fill in an assumption, for example a 2.5 m ceiling to quote paint area, the measure is
marked `observed: false`, its method says it's an assumption, and its interval is wide.

**Why stable IDs like `R2.W2` on every surface?**
Damage, concealed-damage flags and scope line items all have to point at a surface, so a scope line
reads "repaint R2.W2, 6.0 m²". The ID includes the room, so you can read it without a lookup. The
validator rejects any reference to a surface that doesn't exist, so the output can't contain a scope
line keyed to nothing.

**Where do the current intervals come from? Aren't they made up?**
They're provisional, and the plan says so (`"calibrated": false`). Each wall plane gets a position
error: 1 cm for a sharp LiDAR wall plus however thick the wall's point band is (smear), tripled if
the wall was never seen. A length is the distance between two planes, so its error is the sum of
theirs. Video and photos multiply that by 3 and 6 and add 3% or 8% for the learned scale. Work order
07 replaces these constants with values fitted on tape measurements.

**Why one schema for all three tiers?**
The brief wants the same output contract from photos, video and LiDAR. A homeowner's plan and a
scope of works shouldn't depend on which sensor made them. Only the interval widths change, and the
report shows a tier badge ("Photos, intervals ×6") so you can see that.

**What happens if a stage fails or isn't built?**
`fp run` still writes a valid `plan.json`. The missing parts are `observed: false`, and a warning
names the stage. Real errors still stop the run, so nothing is hidden. The default backend is local,
because the walk-in test runs on your machine, not on my cloud.
