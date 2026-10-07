# 07 — Calibrated intervals and the benchmark: defense notes

**The numbers to remember:** LiDAR wall intervals were fitted at **k = 1.42** on 16 wall-to-wall spans from the two scans of
the same flat, with held-out coverage of **94 %**. The worst gate is **video wall lengths on c7d2: p90 error 105 % vs 3 %**.
`make benchmark` regenerates everything in **500 s**. Gates: 3 pass, 13 fail, 10 can't be scored without tape.

**1. What does your 90 % interval mean?**
It's a claim about frequency: across many measurements like this one, at least 9 in 10 true values fall inside
[lo, hi]. It is not a claim that this particular wall is 90 % likely to be inside. You check it by counting: take
measurements whose truth you know, and see how often the truth lands inside. If that is far below 90 %, the intervals
are overconfident. If it is always 100 % with huge intervals, they're honest but not useful.

**2. How did you calibrate without tape?**
I used stand-ins for the truth, and each one is named in every table.
- **LiDAR:** the two whole-flat scans are the same flat. I compared distances between wall planes that both scans
  found (16 spans) and charged the whole disagreement to each scan. That is conservative, because I don't divide by √2.
- **Video and photos:** I compared them with the LiDAR plan of the same capture.

The model is the one from 01: half = k × (evidence term) + b × value. LiDAR fits k; the camera tiers fit b. I fit on
one half of the evidence and check coverage on the other half, both ways round. LiDAR held-out coverage is 88 % and
100 % (8 spans each). Repeatability only shows precision: if both scans were 2 cm too long, I couldn't see it without tape.

**3. How much ground truth do you have? Be honest about n.**
None from tape. The user's own captures and tape readings don't exist yet; the format, the matcher and the slots are
ready. The stand-in evidence is small: 16 LiDAR spans, 6 LiDAR room areas, 9 video walls, 12 photo walls, and 3–4
camera-tier rooms. With fewer than 9 rows, the 90 % rank doesn't exist, so I take the largest error and flag it. With
fewer than 3 rows, the constant stays provisional. Without tape the fit can only widen an interval, never narrow it.

**4. Why is the LiDAR plan not ground truth?**
It has its own errors: drift (walls 1.9–3.3 cm thick after correction), rooms split at different doorways between scans
(23 per-wall edges, only 1 within the gate), and a 5.5 cm median disagreement on spans. It is a good reference for the
camera tiers, because their errors are about 50 cm, but it can't score itself.

**5. Your camera-tier intervals are ±240–310 % of a wall. Isn't that useless?**
Yes, and that is the honest answer for now. The camera tiers on main are far from the LiDAR plan (c7d2 video: 13 m² vs
63 m²). A ±3 % interval there would be the "confident garbage" the brief warns about. The calibration refits itself on
every `make benchmark`, so when 09 fixes the footprint, b shrinks with the errors. Even so, held-out coverage is only
78 % (video) and 67 % (photos): two captures don't disagree in the same way, and I report that as a fail.

**6. Which gates fail, and why?**
- **Camera tiers (the worst):** the whole-flat stitch rules from 03 cut the camera-tier cloud into fragments. The same
  c7d2 video replay gave one 82 m² room before the merge and gives 2 rooms, 13 m², now. That is 09's target.
- **LiDAR repeatability:** 0 of 16 spans within 1 cm. Median 5.5 cm, with errors of both signs, so the scans are
  unrepeatable rather than biased (bias needs tape to see).
- **Openings:** 7 of 14 right by eye. Curtained windows, a low-sill window, and glazed doors called passages.
- **Calibration stress set:** 61 % coverage on per-wall edges, because a room split elsewhere is not measurement noise.

**7. Did you tune anything on the evaluation data?**
No threshold moved after I saw a result. I had to define gates the brief leaves open, and these definitions are in
D07.3 and in the code with comments:
- "within ±X %" means at least 90 % of walls are within X %;
- calibration passes at 80 % or more held-out coverage, because at n ≈ 10, one miss in ten is noise.

The fitted constants are fitted on the sample data by design. That's what calibration is, and the held-out columns
show how they do on data they weren't fitted on.

**8. What is the evidence term `a`, and why keep it?**
It's what makes one wall's interval wider than another's: the wall band's measured thickness (smear), times 3 if the
wall was never seen; jamb reveals for doors; the ceiling's patch spread. Calibration scales it but doesn't replace it,
so the intervals still widen exactly where the data is thin.
