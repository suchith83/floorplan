# Defense notes 02: LiDAR tier, Stray Scanner ingest, single room

**The one number to remember:** **2.2 cm**, the median wall thickness on the single-room capture
(c00a170fe1). That's how sharp the fused walls are. The target was ≤ 2 cm, so it's just above.
The longest wall shows two parallel stripes about 3 cm apart, which is drift (work order 03).

**Why back-projection plus voxel averaging, not TSDF fusion?**
Back-projection is one line of maths per pixel: pixel plus depth, through K⁻¹, then the pose gives a
world point. Averaging all points inside each 1 cm cube removes most of the noise. TSDF also builds a
surface, but we don't need a mesh, only wall planes and floor layers. Open3D's TSDF also returned empty
clouds on this Mac (arm64). Our method is simpler to explain and to check, and the wall thickness
says it's sharp enough.

**Why keep only confidence 2?**
ARKit marks each depth pixel low, medium or high. Medium and low pixels sit on edges, dark or far
surfaces, and glass. I measured instead of guessing: confidence ≥ 1 gave 2.81 cm walls against 2.78 cm
for confidence 2 only, with 18% more points. The difference is small because 94% of pixels are already
level 2. So the extra points add noise, not information.

**How do you find gravity and the floor?**
ARKit's world frame is already gravity-aligned (y up, from the IMU), so "up" comes from the sensor, not
from a guess. The floor is the biggest upward-facing horizontal layer 0.8–2.0 m below the median camera
height. It comes out 1.42 m below the camera, which matches a hand-held phone. The ceiling is the
biggest downward-facing layer of at least 1 m², 2.1–3.6 m above the floor. If no such layer exists, the
ceiling is "not observed", never guessed.

**Why does the single-room scan say the ceiling is "not observed"?**
The camera never pointed up: 98% of frames were pitched between −46° and −8°, so no point is
higher than 1.9 m.
Reporting 2.5 m would be confident garbage. The floor-only flat scan (1a8384c3f6) also reports
not observed, which is what we expected.

**Why Manhattan (walls on two perpendicular axes)?**
Almost all rooms in flats have right angles, and snapping walls to two axes turns noisy points into
straight walls with exact lengths. The yaw comes from a histogram of wall normals folded to 90°. Its
peak is unambiguous here (66.75°, equal to −23.25°). The cost: a slanted or curved wall gets snapped
and shows as a staircase. It's a known failure mode.

**What happens with a mirror or glass?**
LiDAR passes through glass and "sees" what's behind it. A mirror reflects it, so the points land
behind the wall. Three things limit the damage: ARKit gives those pixels low confidence (dropped), the
statistical outlier removal drops sparse floating points, and the footprint only counts a cell as
"seen through" if rays from at least 3 frames crossed it. The glass shower in this capture didn't
break the plan, but a big mirror wall still can, and we list it as a failure mode.

**The floor was missing where the camera looked at furniture. How do you still get the room?**
Every depth ray from the camera to a wall passes through air inside the room. Projected top-down,
those rays sweep the room even where the floor wasn't seen. And wherever the camera went is floor
inside the room, so a 0.3 m band around its path always counts. Together they took the main room
from 12.7 to 19.5 m² (3.4 × 6.05 m, with a kitchen-counter notch).

**How do you pick "the" room when the scan also sees the neighbours?**
I count the seconds the camera spent at least 0.3 m inside each room. The main room got 32 of 37 s. The
room seen through the doorway got 1.1 s, so it stays in the plan but is flagged "partially observed".
I use time, not frames, because the frame filter drops frames where the camera stood still.

**Did you check that depth and pose are in sync?**
Yes. I paired each depth map with the pose 1 or 2 frames earlier or later. The walls got thicker
every time (3.1–4.4 cm against 2.8 cm), so offset 0 is right.
