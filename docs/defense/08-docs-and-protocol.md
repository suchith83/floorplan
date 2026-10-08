# 08 — Capture protocol, device matrix, README, report, compliance: defense notes

**The numbers to remember:** the protocol is **one printed page** (`CAPTURE_PROTOCOL.pdf`); the report is **6 pages**
(`docs/REPORT.pdf`); the compliance matrix covers **157 of 157** atomic requirements, and of the gates **3 pass, 13 fail, 10 can't be
scored without tape, 1 not done**.

**1. Why Stray Scanner and not your own app?**
The brief allows a stock capture route. Stray Scanner is free, it saves raw LiDAR depth, a confidence map, ARKit poses and per-frame
intrinsics, and the evaluator's own sample data was recorded with it. So the LiDAR reader is tested on real files from the app the
evaluator will install. My own app would add install time and a TestFlight build without giving better data.

**2. The brief says 2–8 photos per room. What does your protocol say?**
Six corner photos per room (two opposite corners, 3 overlapping photos each) plus one photo per doorway, added to both rooms' albums.
A room with 3 or more doorways takes 2 per corner, so most rooms stay at 8 or fewer. A hallway with many doors can pass 8, and the
protocol says never to skip a doorway photo, because that photo is what joins the rooms. The code has no per-room cap. The literal
reader showed that a hard "at most 8" cap forces a choice between breaking the cap and dropping a link.

**3. How does a doorway photo join two rooms?**
The user adds it to both rooms' albums. On the laptop the same photo then sits in two room folders. The reader treats it as one frame
that belongs to both rooms, so the joint reconstruction places both rooms around it. It matches by identical bytes, or by the EXIF
capture time to the millisecond plus camera model, because an export can re-encode the file. The literal reader found that byte identity
alone was fragile, so I added the EXIF match and a test.

**4. What happens with an open-plan kitchen and living room?**
The protocol says: make two albums and treat the line between them as a doorway, with one photo on that line in both albums. In the plan
they may still come out as one room: the room split needs a neck (a doorway or a wall stub), and open plan has none. That's listed as a
known failure mode in the report.

**5. What did the literal reader find, and what did you change?**
On the first draft: 7 blocking points. Five were in the protocol: where doorway photos go, no open-plan rule, photo counts that
contradicted each other, the "floor in every photo" rule coming after the photos, and a 4-minute limit that didn't fit a big home. Two
were outside it (the GitHub repo not pushed, the dataset private). I rewrote the protocol, added the EXIF match, and had it re-read. Every
instruction is now a number or an action ("2 seconds in every doorway", "1 m in from the walls", "3 photos per corner").

**6. Your device matrix says "runs; not tested" almost everywhere. Isn't that weak?**
It's what is true. I have no iPhone. The recruiter agreed the LiDAR tier runs on her sample data, and the camera tiers were tested on video
and stills cut from those same captures, with the rotation tag and EXIF focal length a phone writes. HEIC, 10-bit HDR HEVC and rotation
are covered by unit tests on synthetic files, not by a real iPhone file. Claiming more would be the "confident garbage" the brief warns about.

**7. Can I regenerate every number?**
`make benchmark` reruns every capture and refits the intervals (~16 min). `make fix-before` and `make fix-after` rerun the fix loop at its
two git tags. The raw data and the model-output cache come from `scripts/fetch_data.sh` (a Hugging Face dataset), and the weights from
`scripts/fetch_weights.sh`. The cache is keyed by the input bytes, so it only replays for identical inputs, and `--no-cache` always runs live.

**8. What is not done?**
The head-to-head against a consumer app (no LiDAR iPhone; the substitute protocol is ready), tape ground truth, my own captures with
staged damage, and camera-tier repeatability. `COMPLIANCE.md` marks each one Not done or Partial with the reason. The harness picks the own
captures up automatically once they're in `data/own/`.
