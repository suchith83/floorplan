# Defense notes 05: video and photo tiers (no depth, no poses)

**The one number to remember:** **0.89–1.15**, the scale between consecutive video chunks after the
merge fix (it was 0.78 → 0.22 before, which shrank the whole c00a room 2.5×). Second number: the video
tier finds the floor 1.25 m below the camera, against 1.42–1.48 m from LiDAR.

**Where does metric scale come from when there's no depth sensor?**
From the model. MapAnything predicts metric depth directly: it learned the usual size of doors, tiles,
counters and furniture from training data with real scale. Then I check it against priors: a handheld
phone is 1.0–1.8 m above the floor, ceilings are 2.3–3.2 m. If the reconstruction falls outside a prior,
I widen every interval by the relative gap, but I never rescale the values. A prior is a range, not a
measurement, so it may make us less sure but it doesn't overrule the model. For long videos each chunk
votes for its own metric scale and I take the median, so one bad chunk can't set the scale.

**Why one joint reconstruction for all the photos, not one per room?**
If each room is reconstructed alone, each gets its own coordinate frame and its own scale, and then I'd
have to stitch rooms by hand-made matching. A joint pass puts every photo in one frame, which is how the
model was trained. Rooms are tied together by a doorway photo saved in both folders. Honest result: on
our sample photos the rotations between photos come out within 10° (median), but the positions of
separate photo groups are not reliable yet (3 m RMS on c7d2). Photos must overlap, which changes the
capture protocol.

**What happens when a room shares no view with the others?**
It isn't placed. Rooms are linked by a shared doorway photo, or by at least 40 feature matches that agree
geometrically. A room with no link is left out of the plan, and the warning names it and says to retake a
doorway photo in both folders. I never drop it in a guessed spot or overlap it with another room. A wrong
room position is worse than a missing one, because nobody can see that it's wrong.

**Why can't the cache be used to cheat?**
The cache key is a hash of the input files' bytes, the model ID and every parameter. An entry can only
replay the exact images it was computed from. A new capture, such as the walk-in one, always runs the
model live, and `--no-cache` forces a live run on anything. The cache only saves the model's raw output.
Chunk merging, gravity, rooms and intervals run again every time. I checked that a live run and its replay
give byte-identical plan.json files apart from timings.

**How do you handle the walk-in iPhone files?**
Photos: HEIC is decoded with pillow-heif, EXIF rotation is applied, Display P3 is converted to sRGB, and
the EXIF 35 mm focal length becomes an intrinsics prior for the model. Video: iPhone HEVC, including
10-bit HDR and Dolby Vision, is decoded with the ffmpeg binary bundled in a Python package (no system
install). It is tone-mapped to SDR and rotated by the file's rotation tag. Tests cover a HEIC file and a
generated 10-bit HEVC clip with rotation.

**Why did the first video run come out 2.5× too small?**
Two reasons. First, Stray stores the video sideways with no rotation tag, so the model saw every frame on
its side. A real iPhone video carries the tag, so I added the tag to the derived video. Second, the chunk
merge used a least-squares point fit, whose scale shrinks when the point pairs are noisy, and that shrink
multiplied along the chain of chunks. Now the scale is the median depth ratio at the same pixels of shared
frames, which has no such bias. A unit test shows the old fit shrinking about 4% per link on clean
synthetic data.

**How do you find "up" without an IMU?**
A phone camera app turns the picture using its accelerometer, and the ingest applies that rotation, so
"image up" is close to gravity. I refine it onto the normals of horizontal surfaces (floor, table tops,
ceiling). Another image axis wins only if it has twice the horizontal support. On the c7d2 photos a long
wall almost tied with the floor (0.206 vs 0.197), and "most support wins" turned the plan into a side view.
