#!/usr/bin/env bash
# Fetch the sample captures into data/ (data/ is gitignored; raw data is never committed).
#
# STUB. The data will be published as a Hugging Face dataset or a GitHub release (decided in
# work order 08). It will provide:
#   data/stray/c00a170fe1   single room            (Stray Scanner export, LiDAR tier)
#   data/stray/1a8384c3f6   whole flat, no ceiling (Stray Scanner export, LiDAR tier)
#   data/stray/c7d28f72c6   whole flat, ceiling    (Stray Scanner export, LiDAR tier)
#   plus our own benchmark captures (video, photo folders) and tape/laser ground truth.
#
# Until then: if you have the evaluator's zips, put them in data/zips/ and this script unpacks them.
set -euo pipefail
cd "$(dirname "$0")/.."

# zip name : capture id (plain list: macOS ships bash 3.2, which has no associative arrays)
ZIPS="single_room:c00a170fe1 single_scan_floor_only:1a8384c3f6 single_scan_with_ceiling:c7d28f72c6"

mkdir -p data/stray data/zips
for pair in $ZIPS; do
  name=${pair%%:*} id=${pair##*:}
  if [ -d "data/stray/$id" ]; then
    echo "ok       data/stray/$id"
  elif [ -f "data/zips/$name.zip" ]; then
    echo "unzip    data/zips/$name.zip -> data/stray/$id"
    unzip -q "data/zips/$name.zip" -d "data/stray/_tmp_$id"
    # the zip holds a single capture folder; move it into place under its id
    src=$(dirname "$(find "data/stray/_tmp_$id" -name odometry.csv -not -path '*/__MACOSX/*' | head -1)")
    mv "$src" "data/stray/$id" && rm -rf "data/stray/_tmp_$id"
  else
    echo "missing  data/stray/$id (no data/zips/$name.zip; remote download not set up yet)" >&2
  fi
done
