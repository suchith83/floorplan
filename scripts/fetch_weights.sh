#!/usr/bin/env bash
# Pre-download the pretrained weights into the Hugging Face cache (~/.cache/huggingface), so the first `fp run`
# on a camera tier or with damage detection doesn't stop to download. Without this they download on first use.
#   facebook/map-anything-apache       learned depth + poses for the video and photo tiers (Apache-2.0)
#   IDEA-Research/grounding-dino-tiny  damage boxes from text (Apache-2.0)
#   facebook/sam2.1-hiera-small        damage masks from boxes (Apache-2.0)
# All three are public: no login needed.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python - <<'PY'
from huggingface_hub import snapshot_download
for repo in ("facebook/map-anything-apache", "IDEA-Research/grounding-dino-tiny", "facebook/sam2.1-hiera-small"):
    print("weights ", repo, "->", snapshot_download(repo_id=repo))
PY
