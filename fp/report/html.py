"""plan.json -> report.html: one self-contained page for a non-engineer. The plan with damage pins,
rooms and walls with confidence badges, doors, damage with evidence photos and hypotheses, and how
every number was measured. Images are embedded, so the file can be emailed on its own."""
from __future__ import annotations

import base64
import html
from pathlib import Path

from fp.report.svg import render


def _badge(c) -> str:
    if c is None:
        return "—"
    cls = "hi" if c >= 0.7 else "mid" if c >= 0.45 else "lo"
    word = {"hi": "high", "mid": "medium", "lo": "low"}[cls]
    return f'<span class="b {cls}">{c:.2f} {word}</span>'


def _img(path: Path) -> str:
    if not path.exists():
        return ""
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="evidence">'


def render_report(plan: dict, out: Path) -> str:
    e = html.escape
    s = plan["source"]
    rooms = "".join(f"<tr><td>{e(r['name'])}</td><td>{r['area_m2']}</td><td>{r['extent_x_m']} × {r['extent_y_m']}</td>"
                    f"<td>{r['ceiling_h_m'] or 'not measured'}</td><td>{len(r['walls'])}</td></tr>" for r in plan["rooms"])
    walls = "".join(f"<tr><td>{e(r['name'])} {w['id']}</td><td>{w['length_m']:.2f}</td>"
                    f"<td>{'observed' if w['observed'] else '<i>inferred</i>'}</td><td>{w['coverage']:.2f}</td>"
                    f"<td>{w['spread_cm'] if w['spread_cm'] is not None else '—'}</td><td>{_badge(w['confidence'])}</td></tr>"
                    for r in plan["rooms"] for w in r["walls"])
    conns = "".join(f"<tr><td>{c['rooms'][0]} – {c['rooms'][1]}</td><td>{c['kind']}</td><td>{c['width_m']:.2f}</td></tr>"
                    for c in plan["connections"]) or '<tr><td colspan="3">One room: no doors between rooms.</td></tr>'
    dm = plan.get("damage_meta", {})
    if plan["damage"]:
        damage = "".join(
            f"<div class='dmg'>{_img(out / d['evidence'])}<div><h3>{d['id']} · {e(d['type'])}</h3>"
            f"<p>On <b>{e(d['surface'])}</b>, {d['height_m']:.2f} m above the floor, about {d['size_m']:.2f} m across. "
            f"Seen in {d['views']} frame(s), detector score {d['score']:.2f}.</p>"
            + "".join(f"<p class='hyp'><b>Hypothesis ({c['rule']}):</b> {e(c['hypothesis'])}<br><small>Because: {e(c['evidence'])}</small></p>"
                      for c in d["concealed"]) + "</div></div>" for d in plan["damage"])
    elif dm.get("skipped"):
        damage = "<p>Damage detection was not run for this capture.</p>"
    elif "error" in dm:
        damage = f"<p>Damage detection failed: {e(dm['error'])}</p>"
    else:
        damage = "<p>No damage found on any wall, floor or ceiling.</p>"
    dnote = (f"{dm.get('detections', 0)} detections; {dm.get('no_surface', 0)} were not on a wall, floor or ceiling "
             f"and {dm.get('no_depth', 0)} had no depth, so they were dropped." if "detections" in dm else "")
    ff = s.get("frame_filter") or {}
    filt = (f" The frame filter kept {ff['frames_kept']} of {ff['frames_in']} frames (dropped: {ff['dropped_blurry']} blurry, "
            f"{ff.get('dropped_fast', 0)} fast turns, {ff['dropped_duplicate']} duplicates).") if ff else ""
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Floor plan report</title><style>
body{{font:15px/1.5 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d1f21;background:#fff}}
h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:19px;margin:28px 0 8px;border-bottom:1px solid #ddd;padding-bottom:4px}}
h3{{font-size:16px;margin:0 0 4px}} .meta{{color:#555}} table{{border-collapse:collapse;width:100%;font-size:14px}}
td,th{{text-align:left;padding:6px 8px;border-bottom:1px solid #eee}} th{{color:#555;font-weight:600}}
.plan svg{{max-width:100%;height:auto;border:1px solid #ddd;border-radius:8px}}
.b{{padding:1px 8px;border-radius:10px;font-size:12px;font-weight:600}} .hi{{background:#e6f4e8;color:#2e7d32}}
.mid{{background:#fdf1de;color:#a85f00}} .lo{{background:#fbe9e9;color:#c62828}}
.dmg{{display:flex;gap:14px;margin:12px 0;padding:12px;border:1px solid #eee;border-radius:8px}}
.dmg img{{width:220px;height:auto;border-radius:6px;object-fit:cover}} .hyp{{background:#fff7e6;padding:6px 10px;border-radius:6px}}
@media (max-width:600px){{.dmg{{flex-direction:column}}.dmg img{{width:100%}}}} footer{{margin-top:32px;color:#555;font-size:13px}}
</style></head><body>
<h1>Floor plan report</h1>
<p class="meta">Input: <b>{e(s['input'])}</b> · {s['n_frames']} frames used · scale: {e(s['scale'])} · gravity: {e(s.get('gravity', ''))} ·
floor: {e(s.get('floor', ''))} · {e(plan.get('layout', ''))}</p>
<div class="plan">{render(plan)}</div>
<h2>Rooms</h2><table><tr><th>Room</th><th>Area (m²)</th><th>Extent (m)</th><th>Ceiling (m)</th><th>Walls</th></tr>{rooms}</table>
<h2>Walls</h2><table><tr><th>Wall</th><th>Length (m)</th><th>Seen?</th><th>Coverage</th><th>Thickness (cm)</th><th>Confidence</th></tr>{walls}</table>
<h2>Doors and openings</h2><table><tr><th>Between</th><th>Kind</th><th>Width (m)</th></tr>{conns}</table>
<h2>Damage</h2>{damage}<p class="meta">{dnote}</p>
<footer><h2>How this was measured</h2>
<p>Every input becomes a 3D point cloud: LiDAR depth directly, or for photos and video, depth and camera poses predicted
by MapAnything. Walls are flat planes found in that cloud; the room outline sits on them. A wall's length is the distance
between the two walls at its ends.{filt}</p>
<p><b>Confidence</b> = input prior (LiDAR 0.9, video 0.65, photos 0.5) × (0.3 + 0.7 × how much of the two end walls was seen)
× a penalty for wall thickness above 1 cm (a smeared wall gives an uncertain position). Dashed walls were not seen: their
position comes from the edge of the floor. Treat a low score as "measure this one by hand".</p>
<p><b>Damage</b> is found by SAM 3 with text prompts (crack, water stain, mold, peeling paint, hole in wall), placed in 3D with the
depth, and kept only if it sits on a wall, floor or ceiling. A <b>hypothesis</b> is a rule about what may be hidden;
it is not a finding.</p></footer>
</body></html>"""


def write_report(plan: dict, out: Path) -> Path:
    path = out / "report.html"
    path.write_text(render_report(plan, out))
    return path
