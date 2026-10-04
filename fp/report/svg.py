"""plan.json -> dimensioned SVG. Solid walls = observed, dashed = inferred.
Colour = confidence (green high, amber medium, red low)."""
from __future__ import annotations

import numpy as np

PX = 120  # px per metre
M = 110   # margin px (room for side labels)


def _col(c):
    return "#2e7d32" if c >= 0.7 else "#ef8f00" if c >= 0.45 else "#c62828"


def render(plan: dict) -> str:
    allp = np.array([p for r in plan["rooms"] for p in r["polygon"]])
    lo, hi = allp.min(0), allp.max(0)
    W, H = (hi - lo) * PX + 2 * M
    tx = lambda p: (M + (p[0] - lo[0]) * PX, M + (hi[1] - p[1]) * PX)  # y up -> svg down
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H + 60:.0f}" '
           f'font-family="Helvetica,Arial" font-size="13"><rect width="100%" height="100%" fill="white"/>']
    for r in plan["rooms"]:
        if len(r["polygon"]) < 3:
            continue
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(tx, r["polygon"]))
        out.append(f'<polygon points="{pts}" fill="#f4f6f8" stroke="none"/>')
        c = np.mean([tx(p) for p in r["polygon"]], 0)
        label = f'{r.get("name", "Room")} · {r["area_m2"]} m²'
        if r.get("ceiling_h_m"):
            label += f' · h {r["ceiling_h_m"]:.2f} m'
        out.append(f'<text x="{c[0]:.0f}" y="{c[1]:.0f}" text-anchor="middle" font-weight="bold">{label}</text>')
        for w in r["walls"]:
            a, b = tx(w["p0"]), tx(w["p1"])
            dash = "" if w["observed"] else ' stroke-dasharray="8,6"'
            out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" y2="{b[1]:.1f}" '
                       f'stroke="{_col(w["confidence"])}" stroke-width="6"{dash}/>')
            # label outside the polygon: CCW polygon -> outward normal is (dy,-dx) in plan coords
            d = np.subtract(w["p1"], w["p0"]); n = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
            mid = tx(np.add(w["p0"], w["p1"]) / 2 + n * 0.15)
            anchor = "middle" if abs(n[0]) < 0.5 else ("start" if n[0] > 0 else "end")  # side labels clear the wall
            out.append(f'<text x="{mid[0]:.0f}" y="{mid[1] + 4:.0f}" text-anchor="{anchor}">'
                       f'{w["length_m"]:.2f} m <tspan fill="{_col(w["confidence"])}">({w["confidence"]:.2f})</tspan></text>')
    for c in plan.get("connections", []):  # doors: a ring on the shared border, width beside it
        x, y = tx(c["center"])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="9" fill="white" stroke="#1565c0" stroke-width="3"/>')
        out.append(f'<text x="{x + 14:.0f}" y="{y + 4:.0f}" fill="#1565c0">{c["kind"]} {c["width_m"]:.2f} m</text>')
    for d in plan.get("damage", []):  # damage pins: red ring + id, at the defect's position
        x, y = tx(d["position_m"][:2])
        out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="11" fill="#c62828" stroke="white" stroke-width="2"/>')
        out.append(f'<text x="{x:.1f}" y="{y + 4:.1f}" text-anchor="middle" fill="white" font-size="10" '
                   f'font-weight="bold">{d["id"]}</text>')
    s = plan["source"]
    out.append(f'<text x="{M}" y="{H + 22:.0f}">Source: {s["input"]} · scale: {s["scale"]}</text>')
    out.append(f'<text x="{M}" y="{H + 42:.0f}">solid = wall observed · dashed = inferred · (n) = confidence</text>')
    out.append("</svg>")
    return "\n".join(out)
