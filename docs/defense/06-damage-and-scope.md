# 06 — Damage, concealed-damage flags, scope: defense notes

**The number to remember:** a 0.20 × 0.20 m stain (0.040 m²) is measured within **15 %** from face-on and 34° views
(synthetic test). On the sample data, which has no real damage, the viewpoint check cuts c00a's detections from
**4 items to 1**, and that 1 is still false (listed below).

**1. What detects damage, and why not SAM 3?**
Grounding DINO (text → boxes) finds all five classes in one pass, and SAM 2.1 (box → mask) gives the mask. Both run
on this Mac (MPS, about 1 s per full-resolution frame, about 30 s to load) and are ungated Apache-2.0 models. SAM 3
does text → mask in one model, but its weights are gated: it failed with 401 here, and an evaluator's cold run
would need Hugging Face approval first. It's still available with `--backend modal`. Thresholds are the
published defaults (box 0.35, text 0.25), not tuned.

**2. How is the extent in m² measured?**
I put the mask on the depth image as fractional coverage, so a 2-pixel crack still counts. Each covered depth pixel
adds coverage × its footprint on the wall, z²/(fx·fy·cos a), where z is depth and a is the angle between the
pixel's ray and the wall normal. A pixel seen at an angle covers more wall. Only pixels within 8 cm of the wall
plane count, so spill onto a picture frame or a cupboard doesn't. Several views are merged by the median, and the
interval is the wider of 20 % and the views' spread, widened for camera tiers because area goes with scale squared.

**3. How do you know which wall it's on?**
The mask's median 3-D point (LiDAR depth, or model depth on camera tiers) snaps to the nearest wall plane within
15 cm and inside the wall's length. Otherwise it goes to the floor or ceiling of the room it's in. At least half of
the mask must lie on that plane, otherwise it's on furniture and is dropped.

**4. What false positives did you see, and what did you do about them?**
On the sample flats, which have no damage:
- shadows on the floor called "water stain" (one is a basket's shadow that looks like a puddle);
- tile and grout edges and a ceiling joint called "crack";
- a dark rectangle (a mirror or niche) called "hole".

I didn't raise the thresholds, because that would be tuning on the sample data. Instead, every detection must be
confirmed from a second viewpoint at least 25 cm away. A real stain stays put; a reflection moves and a glint
disappears. That cut c00a from 4 items to 1, c7d2 from 5 to 1, and 1a83 from 6 to 3. Unconfirmed detections are
listed in the warnings with their crops. Static shadows survive the check because they also stay put, so the
remaining items on the sample data are false. They're reported in STATUS 06, not hidden.

**5. Why rules for concealed damage instead of a model?**
There's no dataset that pairs a visible symptom with what was later found behind the wall, so there's nothing to
learn from. Five rules, each something a surveyor would say:
- C1: wet damage on a ceiling means a leak from above.
- C2: a stain or peeling paint low on a wall means rising damp or a pipe.
- C3: a stain on a wall shared with a wet room means a plumbing leak.
- C4: a crack near a door means movement around the frame.
- C5: a stain high on a wall or on the ceiling next to a wet room means pipes in the ceiling void.

Every flag carries its rule id, the rule text and the evidence, and it's labelled "hypothesis", never a finding.

**6. How are scope quantities and their intervals computed?**
Fixed rules per surface, for example repaint wall = length × ceiling height − openings, and skirting =
perimeter − door widths. Every input has a [lo, hi], and I push the worst case through the formula: low area =
low length × low height − high openings. If the ceiling wasn't observed, I assume 2.4 m [2.1, 3.2] and mark the
item not observed.

**7. Does it pass the acceptance test on your own capture?**
That capture doesn't exist yet, so this row is "pending own capture". The command is ready:
`uv run fp run data/own/<capture>`. As a stand-in, `scripts/exp06_damage.py` paints a tea stain and a crack in
correct perspective onto real c00a frames and checks the detector finds them on the right wall. The stain was
found on its wall from 5 views: 0.055 m² against 0.049 painted (+11 %), and the interval contains the true value. The
drawn crack, clearly visible, was missed, so crack recall is the weak point I'd expect on your own capture too.
