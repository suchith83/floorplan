"""Check COMPLIANCE.md against the brief's requirement list (docs/brief/requirements.md).

- every requirement number in the list appears in some row's `req #` column (ranges like 97–103 count);
- every path in the `path` column exists in the repo (text in parentheses is a note, not a path).

    uv run python scripts/check_compliance.py      # exit 1 on any miss
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _req_numbers(cell: str) -> set[int]:
    out: set[int] = set()
    for a, b in re.findall(r"(\d+)\s*(?:[–-]\s*(\d+))?", cell):
        out.update(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    want = {int(m) for m in re.findall(r"^(\d+)\. ", (ROOT / "docs/brief/requirements.md").read_text(), re.M)}
    covered: set[int] = set()
    missing_paths: list[str] = []
    n_rows = n_paths = 0
    for line in (ROOT / "COMPLIANCE.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if not line.startswith("|") or len(cells) != 5 or not re.match(r"^\d", cells[0]):
            continue
        n_rows += 1
        covered |= _req_numbers(cells[0])
        for p in re.findall(r"`([^`]+)`", re.sub(r"\([^)]*\)", "", cells[2])):
            n_paths += 1
            if not (ROOT / p).exists():
                missing_paths.append(f"{p}  (row {cells[0]})")
    uncovered = sorted(want - covered)
    print(f"{n_rows} rows, {len(want)} requirements, {len(want & covered)} covered, {n_paths} paths checked")
    for r in uncovered:
        print(f"UNCOVERED requirement {r}")
    for p in missing_paths:
        print(f"MISSING path {p}")
    return 1 if uncovered or missing_paths else 0


if __name__ == "__main__":
    sys.exit(main())
