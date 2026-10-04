# Capture guide — one phone, about 30 minutes

This is the whole capture protocol. Anyone can follow it; no app to install.
You need: an Android phone with 5 GB free, a tape measure (optional), all room lights.

## Before you start
1. Open every inside door. Open the curtains. Switch on all the lights.
2. Wipe the camera lens.
3. In the camera app, use the normal **1×** lens (not 0.5× ultra-wide). No flash.
4. Video settings: **1080p, 30 fps**. If there is a stabilisation option ("Ultra steady", "Super stable"), switch it **off**.
5. Hold the phone **sideways (landscape)** at chest height. Keep the line where the wall meets the floor in view most of the time.

## 1. Video walkthrough — do it twice (`video_1.mp4`, `video_2.mp4`)
1. Start at the front door. Hold still for 2 seconds.
2. Walk at half your normal speed. **Turn slowly: one full turn takes at least 10 seconds.**
   Fast turns blur the frames and smear the walls; this is the main cause of error we measured.
3. In each room, stand in the middle and turn one full circle slowly.
   Then walk along the walls about 1 m away, with the camera at a slight angle to the wall.
4. Between rooms, walk straight through the doorway with the camera facing forward. Pause 1 second in the doorway.
5. Do not film a mirror or a window up close, and do not film a blank wall from less than 0.5 m.
6. Visit every room, then return to the start. A flat usually takes 2–4 minutes.
7. Record the same walk again for `video_2.mp4`. Two runs let us measure repeatability.

## 2. Photos per room
1. Photo mode, 1× lens, landscape, same zoom for every photo.
2. In each room, stand in each corner and take 3 photos: the opposite corner and the two walls next to it.
   That gives about 12 photos per room. Each photo should overlap the previous one by about half.
3. In each doorway, take 2 photos: one into each room. These tie the rooms together.

## 3. Damage
For each crack, stain, mould patch, peeling paint or hole:
1. Take one photo from 1–2 m away (so the wall is recognisable) and one close-up.
2. In the video, pause on it for 2 seconds.
3. Write it in `measurements.yaml` under `defects` **before** you look at any result.

## 4. Measurements (optional, strongly recommended)
The brief asks for real tape measurements. A normal tape is fine. Use metres (3.41, not 341).
- For 2–3 rooms: **width and length**, wall to wall at about 1 m height, **two readings each**.
- One **ceiling height**, floor to ceiling in the middle of a room.
- **Door width** (the clear opening) for 2 doors.
- Optional: if you have Magicplan, scan the same rooms and write its numbers under `magicplan`.

Copy `eval/measurements.template.yaml` to `data/home/measurements.yaml` and fill it in.
Use simple room names (`kitchen`, `bedroom1`, `hall`) and the same names everywhere.

## 5. Copy the files to the laptop
Use a USB cable or Google Drive. Do not send through WhatsApp: it compresses the files.
```
data/home/
  video_1.mp4
  video_2.mp4
  photos/kitchen/*.jpg        one folder per room
  photos/bedroom1/*.jpg
  damage/*.jpg
  measurements.yaml
```
Then run (see README.md):
```bash
uv run fp run data/home/video_1.mp4
```
