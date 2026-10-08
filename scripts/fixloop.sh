#!/usr/bin/env bash
# Regenerate one side of the fix loop (work order 09) from raw inputs:
#   scripts/fixloop.sh fix-before fixloop/before
#   scripts/fixloop.sh fix-after  fixloop/after
# Checks the tag out in a temporary git worktree, links in the raw captures (data/, not in git) and the
# model-output cache (out/_cache: MapAnything depth and the damage detector replay from it, as in
# `make benchmark`; without it the camera tiers would need the paid Modal backend), runs the whole
# benchmark there (every capture, calibration fit, rerun), and copies the tables and the key debug
# images into DEST. The main checkout's out/ and eval/ are never touched.
set -euo pipefail
TAG=${1:?usage: fixloop.sh <tag> <dest>}
DEST=${2:?usage: fixloop.sh <tag> <dest>}
ROOT=$(git rev-parse --show-toplevel)
cd "$ROOT"
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { echo "no tag $TAG"; exit 1; }
[ -d data/stray ] && [ -d data/derived ] || { echo "need data/stray and data/derived (scripts/fetch_data.sh, scripts/make_camera_tiers.py)"; exit 1; }

WT=$(mktemp -d "${TMPDIR:-/tmp}/fp-$TAG.XXXXXX")
cleanup() { cd "$ROOT"; git worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$WT"; }
trap cleanup EXIT
git worktree add --detach "$WT" "$TAG" >/dev/null
ln -s "$ROOT/data" "$WT/data"
mkdir -p "$WT/out" "$ROOT/out/_cache"
ln -s "$ROOT/out/_cache" "$WT/out/_cache"

echo "[$TAG] $(git rev-parse --short "$TAG") in $WT"
(cd "$WT" && uv sync --frozen -q && uv run --frozen python eval/run_benchmark.py --all)

mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
cp "$WT/eval/BENCHMARK.md" "$WT/eval/CALIBRATION.md" "$WT/eval/benchmark.json" "$WT/fp/calibration.json" "$DEST/"
for cap in c7d28f72c6-video c00a170fe1-video c7d28f72c6-photos c7d28f72c6; do
  mkdir -p "$DEST/$cap"
  cp "$WT/out/$cap/plan.json" "$WT/out/$cap/plan.svg" "$DEST/$cap/"
  for img in bev.png fusion_topdown.png rooms_split.png; do
    [ -f "$WT/out/$cap/debug/$img" ] && cp "$WT/out/$cap/debug/$img" "$DEST/$cap/"
  done
done
echo "[$TAG] wrote $DEST"
