# Defense notes 00: capture route, repo and prototype

**The one number to remember:** the LiDAR depth map is **256 × 192** pixels in millimetres, with a
confidence of 0/1/2 per pixel; the RGB video is 1920 × 1440, so intrinsics scale by 256/1920.

**Why Stray Scanner and not your own iOS app?**
I don't have an iPhone Pro to build and test an app on, and the brief accepts a stock app plus a
protocol. Stray Scanner is free and exports raw depth, confidence, poses, intrinsics, IMU and the
video as plain files. Your own sample data was captured with it, so my reader has been run on
your real data, not just mine.

**Why not RoomPlan, Polycam or 3d Scanner App?**
RoomPlan-based apps give a finished parametric model, so there'd be nothing left for my pipeline
to do, and I couldn't measure or fix its errors. I need raw per-frame depth and poses to control
filtering, drift handling and intervals myself.

**What coordinate convention do the poses use, and how do you know?**
The pose is camera-to-world. The camera axes follow OpenCV (x right, y down, z forward), and the
world is gravity-aligned with y up. I checked by back-projecting depth into one cloud both ways:
the ARKit convention (y up, looking down −z) gives a smeared mess, while OpenCV gives sharp,
single walls.

**Why Python 3.12 and uv?**
uv installs the exact locked versions, and the right Python, in one command, which matters for
the "running in under 15 minutes on a clean machine" requirement. I stay on 3.12 because Open3D's
wheels lag new Python releases.

**You reused code from an earlier prototype. What, and how do we see that?**
The round-1 prototype's pipeline core (capture bundle, depth fusion, floor and wall geometry,
room split, damage rules, report) came over in three commits labelled "port from round-1
prototype (branch beta)". Its data, results and evaluation code didn't. Everything after those
commits is new work and shows up in the history piece by piece.

**Where is the data, and why isn't it in git?**
The captures are about 900 MB, so they live under `data/` (gitignored) and `scripts/fetch_data.sh`
fetches them. No tracked file is over 5 MB. Raw data in git would make the repo slow to clone,
and the brief asks for large files to be fetched by script.

**Is the LiDAR output ground truth?**
No. It's a reference. Ground truth is tape or laser on our own captures. I use the LiDAR output to
compare tiers and for repeatability across the two whole-flat scans, and I say so wherever I
report it.
