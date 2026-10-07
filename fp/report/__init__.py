"""Outputs written next to plan.json: plan.svg (the drawing) and report.html (one self-contained page)."""
from __future__ import annotations

from pathlib import Path


def write_outputs(plan: dict, out: Path) -> dict[str, Path]:
    """Write plan.svg and report.html into `out` (created if needed); return their paths.
    plan.json itself is written by the CLI."""
    from fp.report.html import write_report
    from fp.report.svg import render

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    svg = out / "plan.svg"
    svg.write_text(render(plan), encoding="utf-8")
    return {"plan.svg": svg, "report.html": write_report(plan, out)}
