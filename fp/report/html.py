"""plan.json (schema 1.0) -> report.html: one self-contained page a homeowner can open on a phone.
The plan (inline SVG), rooms / walls / openings / connections tables, damage with its evidence crop
(embedded as base64) and the concealed-damage hypotheses, scope line items, warnings, drift and
timings. No external resources; deterministic output (same plan + same evidence files -> same bytes).

Works on plain dicts in the shape of tests/fixtures/plan_two_rooms.json."""
from __future__ import annotations

import base64
import json
import math
from html import escape
from pathlib import Path

from fp.report.svg import half_width, render

TIER = {"lidar": "LiDAR", "video": "Video", "photos": "Photos"}
UNIT = {"m": "m", "m2": "m²", "m3": "m³"}
MIME = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}


def _e(x) -> str:
    return escape("" if x is None else str(x))


def _num(v, unit: str) -> str:
    return f"{v:.2f}" if isinstance(v, (int, float)) else _e(v)


def _scale(x) -> str:
    return f"{x:g}" if isinstance(x, (int, float)) else _e(x)


def _out(lo: float, hi: float) -> tuple[float, float]:
    """Round an interval OUTWARD to 2 decimals, so the printed range is never narrower than the stated one
    (1e-6 absorbs float noise such as 4.18 stored as 4.1799999)."""
    return math.floor(lo * 100 + 1e-6) / 100, math.ceil(hi * 100 - 1e-6) / 100


def measure(m: dict | None, *, show_unit: bool = True) -> str:
    """Measure -> 'value [lo–hi] unit' HTML, 'not observed' when value is null,
    an 'inferred' marker when observed is false. The method is the tooltip."""
    if not m:
        return '<span class="na">—</span>'
    unit = UNIT.get(m.get("unit") or "", _e(m.get("unit")))
    tip = f' title="{_e(m.get("method"))}"' if m.get("method") else ""
    v = m.get("value")
    if v is None:
        return f'<span class="no"{tip}>not observed</span>'
    lo, hi = m.get("lo"), m.get("hi")
    s = f"<b>{_num(v, unit)}</b>"
    if lo is not None and hi is not None:
        s += ' <span class="iv">[{}–{}]</span>'.format(*(_num(x, unit) for x in _out(lo, hi)))
    if show_unit and unit:
        s += f" {unit}"
    if m.get("observed") is False:
        s += ' <span class="inf">inferred / assumed</span>'
    return f"<span{tip}>{s}</span>"


def _table(head: list[str], rows: list[str], empty: str) -> str:
    if not rows:
        return f'<p class="muted">{_e(empty)}</p>'
    th = "".join(f"<th>{_e(h)}</th>" for h in head)
    return f'<div class="tw"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def _evidence(out: Path, rel) -> str:
    if rel:
        p = Path(out) / str(rel)
        if p.is_file():
            mime = MIME.get(p.suffix.lower(), "application/octet-stream")
            data = base64.b64encode(p.read_bytes()).decode("ascii")
            return f'<img class="ev" src="data:{mime};base64,{data}" alt="evidence crop {_e(rel)}">'
    return '<div class="ev missing">evidence image missing</div>'


def _hyp(c: dict) -> str:
    return (f'<div class="hyp"><b>Hypothesis (rule {_e(c.get("rule_id"))})</b>: {_e(c.get("rule_text"))}'
            f'<br><small>Evidence: {_e(c.get("evidence"))}</small></div>')


def _kv(d) -> str:
    if not isinstance(d, dict) or not d:
        return '<span class="muted">none</span>'
    return "<ul class='kv'>" + "".join(
        f"<li><b>{_e(k)}</b>: {_e(v if isinstance(v, (str, int, float, bool)) or v is None else json.dumps(v, sort_keys=True))}</li>"
        for k, v in sorted(d.items(), key=lambda kv: str(kv[0]))) + "</ul>"


def render_report(plan: dict, out: Path) -> str:
    out = Path(out)
    rooms = [r for r in (plan.get("rooms") or []) if isinstance(r, dict)]
    names = {r.get("id"): r.get("name") or r.get("id") for r in rooms}

    def room_name(rid) -> str:
        if rid is None:
            return "exterior"
        return f"{_e(names.get(rid, rid))} <span class='muted'>({_e(rid)})</span>"

    # header + badges
    iv = plan.get("intervals") or {}
    level = iv.get("level")
    lvl = f"{level * 100:.0f}% intervals" if isinstance(level, (int, float)) else "intervals"
    tier = TIER.get(plan.get("tier"), _e(plan.get("tier") or "unknown tier"))
    cal = "calibrated" if iv.get("calibrated") else "provisional"
    badges = (f'<span class="badge tier">{tier}</span>'
              f'<span class="badge">{_e(lvl)}</span>'
              f'<span class="badge">intervals ×{_scale(iv.get("scale"))}</span>'
              f'<span class="badge {"ok" if iv.get("calibrated") else "warn"}">{cal}</span>')
    iv_method = f'<p class="muted small">Interval method: {_e(iv.get("method"))}</p>' if iv.get("method") else ""

    # rooms
    room_rows = [f"<tr><td>{room_name(r.get('id'))}</td><td>{measure(r.get('floor_area'))}</td>"
                 f"<td>{measure(r.get('ceiling_height'))}</td><td>{measure(r.get('perimeter'))}</td>"
                 f"<td>{len(r.get('walls') or [])}</td></tr>" for r in rooms]
    # walls
    wall_rows = []
    for r in rooms:
        for w in r.get("walls") or []:
            seen = ('<span class="ok-t">observed</span>' if w.get("observed", True)
                    else '<span class="inf">inferred</span>')
            cov = w.get("coverage")
            cov_s = f"{cov * 100:.0f}%" if isinstance(cov, (int, float)) else "—"
            wall_rows.append(f"<tr><td>{_e(names.get(r.get('id'), r.get('id')))}</td><td>{_e(w.get('id'))}</td>"
                             f"<td>{measure(w.get('length'))}</td><td>{seen}</td><td>{cov_s}</td></tr>")
    # openings
    op_rows = []
    for r in rooms:
        for o in r.get("openings") or []:
            rr = o.get("rooms")
            if rr:
                between = " ↔ ".join(room_name(x) for x in rr)
            else:
                between = "exterior" if o.get("kind") in ("door", "window") else "—"
            op_rows.append(f"<tr><td>{_e(o.get('id'))}</td><td>{_e(o.get('kind'))}</td><td>{between}</td>"
                           f"<td>{measure(o.get('width'))}</td><td>{measure(o.get('height'))}</td>"
                           f"<td>{measure(o.get('sill'))}</td></tr>")
    # connections
    ops_by_id = {o.get("id"): o for r in rooms for o in (r.get("openings") or [])}
    conn_items = []
    for c in plan.get("connections") or []:
        rr = list(c.get("rooms") or []) + [None, None]
        oid = c.get("opening_id")
        if oid:
            o = ops_by_id.get(oid, {})
            via = f"via {_e(o.get('kind', 'opening'))} {_e(oid)}"
            if o.get("width"):
                via += f" ({measure(o.get('width'))})"
        else:
            via = '<span class="muted">open connection, no opening recorded</span>'
        conn_items.append(f"<li>{room_name(rr[0])} ↔ {room_name(rr[1])} {via}</li>")
    conns = f"<ul>{''.join(conn_items)}</ul>" if conn_items else '<p class="muted">No connections between rooms.</p>'

    # damage + concealed
    concealed = [c for c in (plan.get("concealed") or []) if isinstance(c, dict)]
    dmg_items = []
    for d in plan.get("damage") or []:
        hyps = "".join(_hyp(c) for c in concealed if c.get("damage_id") == d.get("id"))
        score = d.get("score")
        score_s = f"{score:.2f}" if isinstance(score, (int, float)) else "—"
        pos = d.get("position")
        pos_s = (f"at ({', '.join(f'{p:.2f}' for p in pos)}) m" if pos and all(isinstance(p, (int, float)) for p in pos)
                 else "position not located")
        dmg_items.append(
            f'<div class="dmg">{_evidence(out, d.get("evidence_image"))}<div class="dtxt">'
            f'<h3>{_e(d.get("id"))} · {_e(str(d.get("class", "")).replace("_", " "))}</h3>'
            f'<p>On <b>{_e(d.get("surface_id"))}</b>, {_e(pos_s)}.<br>'
            f'Extent {measure(d.get("extent_m2"))} · detector score {score_s}</p>{hyps}</div></div>')
    damage = "".join(dmg_items) or '<p class="muted">No damage reported.</p>'
    loose = [c for c in concealed if c.get("damage_id") is None]
    loose_html = ("<h3>Hypotheses not tied to a damage item</h3>" + "".join(_hyp(c) for c in loose)) if loose else ""

    # scope
    scope_rows = [f"<tr><td>{_e(s.get('id'))}</td><td>{_e(s.get('surface_id'))}</td><td>{_e(s.get('action'))}</td>"
                  f"<td>{measure(s.get('quantity'), show_unit=False)}</td><td>{_e(UNIT.get(s.get('unit') or '', s.get('unit')))}</td></tr>"
                  for s in plan.get("scope") or []]

    # warnings, drift, timings, source
    warns = "".join(f"<li><b>{_e(w.get('stage'))}</b>: {_e(w.get('message'))}</li>" for w in plan.get("warnings") or [])
    warns = f"<ul>{warns}</ul>" if warns else '<p class="muted">No warnings.</p>'
    dr = plan.get("drift") or {}
    drift = (f"<p>Method: {_e(dr.get('method'))} · {'enabled' if dr.get('enabled') else 'not enabled'}</p>"
             f"{_kv(dr.get('metrics'))}")
    tm = plan.get("timings") or {}
    t_rows = [f"<tr><td>{_e(k)}</td><td>{v:.1f}</td></tr>" if isinstance(v, (int, float)) else
              f"<tr><td>{_e(k)}</td><td>{_e(v)}</td></tr>"
              for k, v in [(k, tm[k]) for k in tm if k != "total"] + ([("total", tm["total"])] if "total" in tm else [])]

    fa = plan.get("footprint_area")
    fa_hw = half_width(fa)

    css = """
*{box-sizing:border-box}
body{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:#1d2125;background:#f6f7f9;margin:0}
.wrap{max-width:1040px;margin:0 auto;padding:20px 16px 40px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;margin:30px 0 10px;padding-bottom:4px;border-bottom:1px solid #dde1e6}
h3{font-size:15px;margin:0 0 4px}
.card{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:14px 16px}
.meta{color:#4a525b;margin:2px 0 8px}
.badge{display:inline-block;padding:2px 10px;margin:2px 6px 2px 0;border-radius:12px;font-size:12px;font-weight:600;background:#eceff3;color:#3a424b}
.badge.tier{background:#1d4f91;color:#fff}
.badge.ok{background:#e4f3e7;color:#22682f}
.badge.warn{background:#fdf0dc;color:#94560a}
.plan{background:#fff;border:1px solid #e3e6ea;border-radius:10px;padding:6px;overflow-x:auto}
.plan svg{display:block;max-width:100%;height:auto;margin:0 auto}
.tw{overflow-x:auto;-webkit-overflow-scrolling:touch;background:#fff;border:1px solid #e3e6ea;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px;min-width:520px}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid #eef0f3;vertical-align:top;white-space:nowrap}
th{color:#5a636d;font-weight:600;background:#fafbfc}
tbody tr:last-child td{border-bottom:none}
.iv{color:#5a636d}
.no{color:#7b838c;font-style:italic}
.na,.muted{color:#7b838c}
.small{font-size:13px}
.inf{display:inline-block;font-size:11px;font-style:italic;color:#94560a;background:#fdf0dc;border-radius:8px;padding:0 6px}
.ok-t{color:#22682f}
.dmg{display:flex;gap:14px;margin:10px 0;padding:12px;background:#fff;border:1px solid #e3e6ea;border-radius:10px}
.ev{width:200px;min-width:200px;height:auto;border-radius:6px;object-fit:cover}
.ev.missing{height:120px;display:flex;align-items:center;justify-content:center;background:#f0f2f4;border:1px dashed #b9c0c7;color:#6b737c;font-size:13px;text-align:center}
.dtxt{flex:1;min-width:0}
.hyp{background:#fff7e8;border-left:3px solid #e0a040;padding:6px 10px;border-radius:4px;margin-top:8px;font-size:14px}
ul.kv{margin:4px 0;padding-left:20px}
footer{margin-top:34px;color:#4a525b;font-size:14px}
@media (max-width:620px){.dmg{flex-direction:column}.ev{width:100%;min-width:0}}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Floor plan report · {_e(plan.get("capture_id"))}</title><style>{css}</style></head>
<body><div class="wrap">
<header class="card">
<h1>Floor plan report</h1>
<p class="meta">Capture <b>{_e(plan.get("capture_id"))}</b> · {_e(plan.get("device"))} · schema {_e(plan.get("schema_version"))}</p>
<div>{badges}</div>
<p class="meta">Footprint area {measure(fa)}{f" (±{fa_hw:.2f})" if fa_hw is not None else ""} · {len(rooms)} room(s)</p>
{iv_method}
</header>
<h2>Plan</h2>
<div class="plan">{render(plan)}</div>
<h2>Rooms</h2>
{_table(["Room", "Floor area", "Ceiling height", "Perimeter", "Walls"], room_rows, "No rooms reconstructed.")}
<h2>Walls</h2>
{_table(["Room", "Wall", "Length", "Seen?", "Coverage"], wall_rows, "No walls.")}
<h2>Doors, windows and openings</h2>
{_table(["Id", "Kind", "Between", "Width", "Height", "Sill"], op_rows, "No openings found.")}
<h2>Connections</h2>
{conns}
<h2>Damage</h2>
{damage}
{loose_html}
<h2>Scope of work</h2>
{_table(["Id", "Surface", "Action", "Quantity", "Unit"], scope_rows, "No scope items.")}
<h2>Warnings</h2>
{warns}
<h2>Drift correction</h2>
{drift}
<h2>Timings</h2>
{_table(["Stage", "Seconds"], t_rows, "No timings recorded.")}
<h2>Source</h2>
{_kv(plan.get("source"))}
<footer class="card">
<h2 style="margin-top:0">How to read this</h2>
<p><b>Intervals.</b> Every number comes with a range written <i>value [lo–hi]</i>, and on the plan as <i>value ±h</i>
(h is the larger distance from the value to either end of the range). It is a {_e(lvl.replace(" intervals", "") or "90%")}
interval: we state that about 9 in 10 true values fall inside [lo, hi]. {"These ranges are calibrated against reference measurements." if iv.get("calibrated") else "These ranges are provisional: they are not yet calibrated against reference measurements."}</p>
<p><b>Not observed</b> means the capture did not see it, so we give no number rather than a guess.
<b>Inferred / assumed</b> means a number we did not measure directly (for example a wall that was never seen, whose
position comes from the edge of the floor); on the plan those walls are dashed.</p>
<p><b>Hypotheses are not findings.</b> A hypothesis is a rule about what may be hidden behind a visible defect
(for example a leaking pipe behind a low water stain). It tells you where to look, not what is there.</p>
<p>The arrow on the plan is the plan's +y axis. True north is unknown: the phone capture has no compass reading.</p>
</footer>
</div></body></html>
"""


def write_report(plan: dict, out: Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "report.html"
    path.write_text(render_report(plan, out), encoding="utf-8")
    return path
