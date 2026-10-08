# Capture protocol (one page)

Pick one: **LiDAR** (iPhone Pro / Pro Max only), **Video** or **Photos** (any iPhone 15 or newer).
**To run** (laptop set up per the [README](README.md)): in Terminal, in the `floorplan` folder, type `uv run fp run ` (with a
space), drag your capture (folder or video file) into the window, press Enter. Result, inside `floorplan`:
`out/<name>/report.html` (open in a browser); every number has a range. No Mac: copy files by USB-C drive (Share → Save to Files).

**Before you start:** turn on every light. Open every inside door fully. Close blinds where sun falls on the floor.
Nobody else in the rooms. Never point the phone straight at a mirror.

<img src="docs/img/walk_path.svg" width="265" alt="Walk path: hallway and three rooms, a loop in each room, pause 2 s in each doorway, end where you started">

## The walk (LiDAR and Video)

1. Start at the front door, phone at chest height. End there too, facing the same way.
2. Walk at 0.5 m per second (one slow step per second). A quarter turn takes at least 2 seconds.
3. In each room, walk one loop 1 m in from the walls, phone aimed across the room (at a wall at most 4 m away); then tilt down to
   the floor and up to the ceiling once.
4. **Each time** you pass a doorway, stop in it for **2 seconds**, phone pointing into the room ahead.
5. The whole home in **one** recording, under 4 minutes: 30 s for a small room, 60 s for a big one.
   Never stop and restart; if you must stop, start again from the beginning.

## A. LiDAR

1. Install **Stray Scanner** (free, App Store). Keep its default settings.
2. Tap record, do **the walk**, tap stop. **Files** app → On My iPhone → Stray Scanner. Long-press the newest folder → **Compress** → Share → AirDrop
   to the laptop. Double-click the `.zip` to unzip it. **Run** the unzipped folder (not the zip): it holds `rgb.mp4`,
   `odometry.csv` and `depth/`.

## B. Video

1. **Camera** app → **Video** (not Cinematic, not Slo-mo), 1× lens. 1080p or 4K, 30 or 60 fps. HDR on or off.
2. Hold the phone **sideways** for the whole clip. Do **the walk**. AirDrop the clip to the laptop (or copy by
   cable); never by WhatsApp or email: they shrink it. **Run** the video file.

## C. Photos

1. In Photos, make **one album per room**, hallway included, named in one word (`kitchen`, `hall`). **Open-plan**
   (no wall between kitchen and living room): two albums; treat the line between them as a doorway.
2. **In each room:** phone upright, 1× lens, tilted slightly down so the floor shows. Stand in one corner, take a photo,
   turn right so the next photo shows **half of the previous one**, take it: **3 photos**. Walk to the opposite corner: 3 more.
3. **Each doorway, once:** stand in it, take **1 photo** looking into either room. Add that photo to **both** rooms'
   albums (photo → ⋯ → Add to Album). This is what joins the rooms.
4. **Count:** 6 corner photos per room plus its doorway photos. A room with 3 or more doorways: 2 per corner. The front
   door is not a doorway. Never skip a doorway photo, even if an album then holds more than 8.
5. **Hand over:** on the laptop make a folder `my_home`. For each album: open it → Select → select all → Share →
   AirDrop; move the photos from `Downloads` into `my_home/<album name>/`.
   Doorway photos arrive with both albums: keep both copies. No photos loose in `my_home`.
   **Run** the `my_home` folder.
