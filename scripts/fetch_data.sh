#!/usr/bin/env bash
# Fetch everything the benchmark needs into data/ and out/_cache/ (both gitignored; raw data is never committed).
#
# Source: a Hugging Face dataset (repo type "dataset"), default $FP_DATA_REPO = suchith83/floorplan-data.
# Layout of the dataset (written by scripts/publish_data.sh):
#   zips/{single_room,single_scan_floor_only,single_scan_with_ceiling}.zip   evaluator's Stray Scanner captures (LiDAR tier)
#   derived/<id>/{video,photos,selection.json}   camera-tier inputs cut from those captures (scripts/make_camera_tiers.py)
#   own/...                                      own captures, tape notes, consumer-app exports (when they exist)
#   cache/recon/*.npz + manifest.json            MapAnything outputs, replayed only for byte-identical inputs
#   cache/damage/*                               detector outputs, same rule
# Ground truth files (eval/ground_truth/*.yaml) are in git.
#
#   scripts/fetch_data.sh              # everything (~1.4 GB download)
#   scripts/fetch_data.sh --no-cache   # inputs only; every model then runs live
# A private dataset needs `hf auth login` (or HF_TOKEN) first; a public one needs nothing.
# If you already have the evaluator's zips, put them in data/zips/ and the download of the zips is skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

REPO=${FP_DATA_REPO:-suchith83/floorplan-data}
WITH_CACHE=1
[ "${1:-}" = "--no-cache" ] && WITH_CACHE=0

# zip name : capture id (plain list: macOS ships bash 3.2, which has no associative arrays)
ZIPS="single_room:c00a170fe1 single_scan_floor_only:1a8384c3f6 single_scan_with_ceiling:c7d28f72c6"

mkdir -p data/stray data/zips out/_cache

# 1. Download what is missing. huggingface_hub is a project dependency, so this needs only `uv sync`.
patterns=""
[ -d data/derived ] || patterns="derived/*"
[ -d data/own ] || patterns="$patterns own/*"
for pair in $ZIPS; do
  name=${pair%%:*} id=${pair##*:}
  [ -d "data/stray/$id" ] || [ -f "data/zips/$name.zip" ] || patterns="$patterns zips/$name.zip"
done
if [ $WITH_CACHE = 1 ] && [ ! -f out/_cache/recon/manifest.json ]; then patterns="$patterns cache/*"; fi
if [ -n "$patterns" ]; then
echo "download from datasets/$REPO:$patterns"
uv run python - "$REPO" $patterns <<'EOF'
import sys
from huggingface_hub import snapshot_download
repo, patterns = sys.argv[1], sys.argv[2:]
snapshot_download(repo_id=repo, repo_type="dataset", local_dir="data/_hf", allow_patterns=patterns)
EOF
fi

# 2. Put each part in place.
[ -d data/_hf/zips ] && mv -n data/_hf/zips/*.zip data/zips/ 2>/dev/null || true
for part in derived own; do
  if [ -d "data/_hf/$part" ]; then mkdir -p "data/$part" && cp -Rn "data/_hf/$part/." "data/$part/"; fi
done
if [ $WITH_CACHE = 1 ] && [ -d data/_hf/cache ]; then
  cp -Rn data/_hf/cache/. out/_cache/
  # drop any recon entry whose bytes don't match the manifest, so a corrupted download is never replayed
  uv run python - <<'EOF'
import hashlib, json
from pathlib import Path
c = Path("out/_cache/recon"); m = c / "manifest.json"
if m.is_file():
    want = json.loads(m.read_text())
    bad = [f for f, h in want.items() if not (c / f).is_file()
           or hashlib.sha256((c / f).read_bytes()).hexdigest() != h]
    for f in bad:
        (c / f).unlink(missing_ok=True)
    print(f"cache    out/_cache/recon: {len(want) - len(bad)} entries ok" + (f", {len(bad)} corrupted removed" if bad else ""))
EOF
fi

rm -rf data/_hf   # everything useful was moved or copied out

# 3. Unpack the Stray Scanner zips into data/stray/<id>.
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
    echo "missing  data/stray/$id (not in data/zips/ and not downloaded)" >&2
  fi
done
echo "done: data/stray/, data/derived/$( [ -d data/own ] && echo ', data/own/')$( [ $WITH_CACHE = 1 ] && echo ', out/_cache/')"
