#!/usr/bin/env bash
# Copy a small, curated set of outputs into results/ (out/ and data/ are not committed).
# Run after eval/run_public.py and the home runs.
set -euo pipefail
cd "$(dirname "$0")/.."
R=results
rm -rf "$R" && mkdir -p "$R"

copy() {  # copy <src_dir> <dest_name> <files...>
  local src=$1 dst=$R/$2; shift 2
  [ -d "$src" ] || { echo "skip $src (not run)"; return; }
  mkdir -p "$dst"
  for f in "$@"; do [ -e "$src/$f" ] && cp -R "$src/$f" "$dst/"; done
}

# the weakest result, before and after (scan 48018559, the tuning scan)
copy out/baseline/48018559 lidar-48018559-original plan.svg debug_bev.png plan.json
copy out/eval/48018559/lidar lidar-48018559-fixed plan.svg debug_bev.png plan.json report.html
copy out/eval/48018560/lidar lidar-48018560-fixed-heldout plan.svg debug_bev.png plan.json report.html
[ -f out/48018559/check/fusion_4_cleanup.png ] && cp out/48018559/check/fusion_4_cleanup.png "$R/wall_slice_unfiltered.png"
[ -f out/48018559/check-filtered/fusion_4_cleanup.png ] && cp out/48018559/check-filtered/fusion_4_cleanup.png "$R/wall_slice_filtered.png"

# camera-only on the same scan
copy out/eval/48018559/video video-rgb-only-48018559 plan.svg debug_bev.png plan.json report.html
copy out/eval/48018559/photos-24 photos24-rgb-only-48018559 plan.svg debug_bev.png plan.json

# the home capture
for run in video_1 video_2 photos; do
  copy "out/$run" "home-$run" plan.svg debug_bev.png plan.json report.html damage
done

cp eval/results.md "$R/" 2>/dev/null || true
cp eval/results_home.md "$R/" 2>/dev/null || true
du -sh "$R"; find "$R" -maxdepth 1 | sort
