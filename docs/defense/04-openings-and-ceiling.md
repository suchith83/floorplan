# 04 — Openings and per-room ceiling height: defense notes

**The number to remember:** a door on c7d28f72c6 measures **0.904 m** jamb to jamb (R1.O1, lintel 2.08 m), and the
synthetic test door is measured within **2 cm** of its true 0.90 m. Ceilings: **2.33–2.49 m** in most rooms, **3.09 m**
in R3; the floor-only scan reports **not observed** in every room.

**1. How do you tell an opening from a part of the wall you just didn't see?**
Ray casting. Every depth pixel is a ray from the camera to a measured point. On each wall plane I count, per 2 cm cell,
the frames whose rays ended on the plane (wall) and the frames whose rays went through it and ended more than 5 cm
behind (air). A cell no ray reached is "unobserved" — behind a wardrobe, above the camera's view — and is never
called an opening. A hole in the point cloud alone can't make that distinction.

**2. How is a width measured, and why is ±2 cm tight?**
Between the two jamb planes: each jamb is the median along-wall position of the reveal points, the surfaces inside
the wall thickness that face into the opening. Casings stand proud of the wall, so points less than 2 cm behind the
face are left out. Without reveal points I use the 95th percentile of the wall-face points next to the gap. The
LiDAR depth image is 256×192, so at 2 m one pixel is about 1 cm: each jamb carries about 1 cm of error, and a width
(two jambs) about 2 cm. That's why the gate is tight, and why I never use the gap's width in cells.

**3. What fools it?**
- *Mirror:* the sensor "sees" a reflected room behind the wall. I reflect the points seen behind a gap back across
  the plane; if most land on the room's own surfaces, it's a mirror.
- *Niche, recessed panel, mirror set in the wall:* a surface facing the room less than 40 cm behind the plane
  covers the gap, so it's a recess and is rejected.
- *Wardrobe:* it hides the wall, so cells stay unobserved (no false opening). But a wardrobe that stands *in* a
  doorway hides the door: that's a miss.
- *Glass:* LiDAR often gets no return through a window, so the window can be missed.

On c7d28f72c6 the shower-room mirror scored only 0.48 on the mirror test and was caught by the recess test
instead. The vanity mirror scored 0.30 and got through as a "window": that's the one phantom left. I didn't
lower the threshold, because real doors score up to 0.38 and that would be tuning on the test capture.

**4. What about a closed door?**
A closed leaf sits a few cm back in its frame, so it shows as a recess with a door's shape, and I keep it as a
closed door. That's how R2.W5 on c7d28f72c6 was found; the RGB frames show the handle. Limit: a leaf less than 5 cm
behind the wall face counts as wall (the hit tolerance), so that door is missed.

**5. Why do some doors have no height?**
On c00a170fe1 the camera always pointed down (−46° to −8°), so no ray ever crossed a doorway above ~1.4 m and no
lintel was seen. A door-wide gap from the floor counts as a door if it was seen through up to at least 1.2 m (above
counters and sofa backs); its height is reported as not observed, not guessed.

**6. How is the ceiling height computed per room?**
Inside the room polygon: down-facing points 2.1–3.6 m above the floor, histogrammed into layers; the largest layer
of at least 1 m² is the ceiling. Height = its median minus the median of the room's own floor (so a few cm of
drift tilt across a flat doesn't leak in). The other layers (beams, bulkheads, the 3.08 m layer in R1) go into the
warnings. I start at 2.1 m, not 1.8 m, because round 1 picked a kitchen cabinet underside at 1.86 m as the ceiling.

**7. Where does the ceiling interval come from?**
1 cm of LiDAR range bias, plus 1.645 × √(σ_ceiling²/patches + σ_floor²/patches), where patches = 25 cm squares the
layer covers. Neighbouring points share the same drift and smear, so they aren't independent; patches roughly are.
If the room's floor wasn't seen, I add 3 cm. All intervals are still provisional until 07 calibrates them.

**8. Repeatability?**
1a8384c3f6 (floor-only) gives "not observed" in every room, as it must, so the two flat scans can't be compared on
ceilings. That comparison waits for repeat captures (07). Openings: the same rooms show doors of 0.90–0.93 m on both
scans, but room ids differ between scans, so the matching is by eye.

**9. Does it meet the openings gate (≥ 85 %)?**
No. A verifier checked every opening on c7d28f72c6 by eye in 5 RGB frames. First pass: 6 of 12 correct, with 3
phantoms (a fridge front, a mirror, a toilet niche) and 4 misses, so 38 %. Two rules then fixed the worst
phantoms: a closed door must have a lintel above it, and an opening can't be wider than its own wall. After
that: 10 found, 7 correct, 1 phantom, 2 real but the wrong kind, 4 missed, so **50 %**. Those rules were found
on the same capture, so 50 % is optimistic. The misses are a tall window with a 15 cm sill, two windows behind
closed curtains, and a closed door whose leaf is flush with the wall. Widths can't be checked against ±2 cm
without tape measurements (07).
