"""Publish or fetch the camera-tier reconstruction cache (out/_cache/recon/*.npz) as a Hugging Face dataset.

The cache is a convenience for a cold laptop: MapAnything takes minutes on a laptop GPU and ~5 GB of weights.
A cache entry is keyed by sha1(model id + parameters + a hash of every input frame's bytes) (fp/recon/camera.py),
so a fetched entry is only ever used for exactly the inputs it was computed from; anything else runs live,
and `fp run ... --no-cache` always runs live.

    uv run python scripts/cache_sync.py push --repo <user>/floorplan-recon-cache   # needs `hf auth login`
    uv run python scripts/cache_sync.py pull --repo <user>/floorplan-recon-cache   # public, no login
    uv run python scripts/cache_sync.py list
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

CACHE = Path("out/_cache/recon")
MANIFEST = "manifest.json"   # {file: sha256}; checked after a pull so a corrupted download is never replayed


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _local(cache: Path) -> dict[str, str]:
    return {p.name: _sha256(p) for p in sorted(cache.glob("*.npz")) if not p.name.endswith(".tmp.npz")}


def push(repo: str, cache: Path) -> None:
    from huggingface_hub import HfApi
    files = _local(cache)
    if not files:
        raise SystemExit(f"Nothing to publish in {cache}.")
    (cache / MANIFEST).write_text(json.dumps(files, indent=1, sort_keys=True))
    api = HfApi()
    api.create_repo(repo, repo_type="dataset", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(cache),
                      allow_patterns=["*.npz", MANIFEST], ignore_patterns=["*.tmp.npz"],
                      commit_message=f"recon cache: {len(files)} entries")
    print(f"pushed {len(files)} entries to datasets/{repo}")


def pull(repo: str, cache: Path) -> None:
    from huggingface_hub import snapshot_download
    cache.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(cache), allow_patterns=["*.npz", MANIFEST])
    want = json.loads((cache / MANIFEST).read_text())
    bad = [f for f, h in want.items() if not (cache / f).is_file() or _sha256(cache / f) != h]
    for f in bad:
        (cache / f).unlink(missing_ok=True)
    print(f"fetched {len(want) - len(bad)} entries into {cache}" + (f"; removed {len(bad)} corrupted" if bad else ""))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["push", "pull", "list"])
    ap.add_argument("--repo", help="Hugging Face dataset id, e.g. <user>/floorplan-recon-cache")
    ap.add_argument("--cache", type=Path, default=CACHE)
    a = ap.parse_args()
    if a.cmd == "list":
        for f, h in _local(a.cache).items():
            print(f"{f}  {(a.cache / f).stat().st_size / 1e6:7.1f} MB  sha256 {h[:12]}")
        return
    if not a.repo:
        ap.error("--repo is required for push/pull")
    (push if a.cmd == "push" else pull)(a.repo, a.cache)


if __name__ == "__main__":
    main()
