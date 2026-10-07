"""plan.json (schema 1.0) -> plan.svg, drawn the way consumer floor-plan apps (magicplan, Polycam)
draw a plan: thick dark walls, lightly filled floors, door swings, windows as a thin double line,
the length of every wall with its 90% interval half-width (`3.42 m ±0.03`), a scale bar and an
axis arrow (we do not know true north, so the arrow is the plan +y axis).

Works on plain dicts in exactly the shape of tests/fixtures/plan_two_rooms.json; it does not import
the Pydantic schema. Output is deterministic: same plan -> byte-identical SVG.

Drawing conventions (also used in the defense notes):
- A room polygon is the inside face of its walls, so each wall is drawn as a band *outside* the
  polygon. Two rooms sharing a wall have two parallel faces 10-15 cm apart; their bands meet and read
  as one thick interior wall.
- Wall labels sit outside the room on a dimension line; for a wall with another room behind it the
  label goes on the room's own inner side instead, so the two faces of an interior wall give two
  labels, one in each room, never two labels stacked in the gap.
- Door hinge: at the end of the gap nearer the wall's p0 (the start of the gap in wall direction);
  the leaf opens 90 degrees into the room that lists the opening (the owner of wall_id).
- Labels are placed greedily from a short list of candidate spots; a spot is taken only if it does
  not touch a wall, a door swing, a room label, a damage pin or an earlier label.
"""
from __future__ import annotations
import math
from html import escape

import numpy as np
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polylabel

PX_MAX = 100        # px per metre for plans up to 9 m across (fixture is 7.5 m -> 750 px wide)
PLAN_PX = 900       # larger plans shrink so their long side fits in about this many px
PX_MIN = 35         # never smaller: below this, labels no longer fit beside 1 m walls
WALL_PX = 8         # wall band thickness on screen; 8 px = 8 cm at 100 px/m, so the two faces of a
                    # 12-16 cm interior wall overlap and read as one wall
MARGIN = 95         # px around the plan for outside labels (two label rows fit in it)
HEADER = 58         # px for the title lines
FOOTER = 100        # px for the scale bar, axis arrow and legend
MIN_W = 640         # px: the legend needs this much width
FONT = "Helvetica, Arial, sans-serif"
CHAR_W = 0.58       # average glyph width as a fraction of font size (Helvetica digits ~0.556)
SNAP = 0.02         # m: wall end points closer than this are the same corner (mitre them)
PARTNER_DIST = 0.40  # m: a parallel wall of another room this close to an opening is the other face
                    # of the same physical wall, so the opening's gap is cut through it too
PROBE = 0.30        # m: probe distance behind a wall to decide if another room lies behind it
ROW1, ROW2 = 0.22, 0.48  # m from the outer wall face to the centre of the first/second label row

INK = "#262a2e"      # observed walls
INFERRED = "#8a9097"  # inferred (not seen) walls, drawn dashed on a pale band
INFERRED_BAND = "#e1e4e8"
FLOOR = "#eef1f4"
MUTED = "#5f6670"
DAMAGE = "#d32f2f"
WINDOW_GLASS = "#dcecf7"


# ----------------------------------------------------------------------------- formatting helpers
def _f(x: float) -> str:
    """Coordinate with 1 decimal, never '-0.0' (keeps the SVG byte-stable)."""
    s = f"{x:.1f}"
    return "0.0" if s == "-0.0" else s


def _val(m: dict | None):
    return None if not m else m.get("value")


def half_width(m: dict | None) -> float | None:
    """90% interval half-width max(value-lo, hi-value), or None when unknown."""
    if not m or m.get("value") is None:
        return None
    v, lo, hi = m["value"], m.get("lo"), m.get("hi")
    parts = [x for x in ((v - lo) if lo is not None else None, (hi - v) if hi is not None else None)
             if x is not None]
    return max(0.0, max(parts)) if parts else None


def fmt_len(m: dict | None) -> str:
    """'3.42 m ±0.03', '3.42 m' without an interval, '? m' when not observed."""
    v = _val(m)
    if v is None:
        return "? m"
    hw = half_width(m)
    # the half-width rounds UP to the cm, so the label never claims more precision than the interval
    return f"{v:.2f} m" + (f" ±{math.ceil(hw * 100 - 1e-6) / 100:.2f}" if hw is not None else "")


def fmt_area(m: dict | None) -> str:
    v = _val(m)
    return "? m²" if v is None else f"{v:.2f} m²"


def _text_w(s: str, size: float) -> float:
    return len(s) * size * CHAR_W


# ----------------------------------------------------------------------------- geometry helpers
def _signed_area(poly) -> float:
    a = 0.0
    for i in range(len(poly)):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % len(poly)]
        a += x0 * y1 - x1 * y0
    return a / 2


def _mitre(p, na, nb, t):
    """Outer corner point at p for two walls with outward normals na, nb and band thickness t."""
    d = 1.0 + float(np.dot(na, nb))
    if d < 0.3:  # nearly reversed walls: no sensible mitre
        return p + t * nb
    return p + t * (na + nb) / d


class _View:
    """Plan metres (y up) -> SVG px (y down)."""

    def __init__(self, lo, hi, px, ox, oy):
        self.lo, self.hi, self.px, self.ox, self.oy = lo, hi, px, ox, oy

    def __call__(self, p):
        return (self.ox + (float(p[0]) - self.lo[0]) * self.px,
                self.oy + (self.hi[1] - float(p[1])) * self.px)

    def pts(self, ps) -> str:
        return " ".join(f"{_f(x)},{_f(y)}" for x, y in (self(p) for p in ps))


def _rect(cx, cy, w, h, angle_deg):
    """Rotated rectangle (screen px) as a shapely polygon, for collision tests."""
    a = math.radians(angle_deg)
    ux, uy = math.cos(a), math.sin(a)
    vx, vy = -uy, ux
    hw, hh = w / 2, h / 2
    return Polygon([(cx + sx * hw * ux + sy * hh * vx, cy + sx * hw * uy + sy * hh * vy)
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))])


def _screen_angle(u) -> float:
    """Text angle (deg, SVG) for a label along plan direction u, kept readable: [-90, 90)."""
    a = math.degrees(math.atan2(-float(u[1]), float(u[0])))
    while a >= 90:
        a -= 180
    while a < -90:
        a += 180
    return round(a, 3) + 0.0


# ----------------------------------------------------------------------------- model building
def _build_walls(rooms):
    """Flatten walls with their geometry: unit direction u, outward normal n, outer mitre points."""
    walls = []
    for ri, r in enumerate(rooms):
        poly = r.get("polygon") or []
        ccw = _signed_area(poly) >= 0 if len(poly) >= 3 else True
        ws = []
        for wi, w in enumerate(r.get("walls") or []):
            try:
                p0 = np.asarray(w["p0"], float)[:2]
                p1 = np.asarray(w["p1"], float)[:2]
            except (KeyError, TypeError, ValueError):
                continue
            L = float(np.linalg.norm(p1 - p0))
            if L < 1e-6:
                continue
            u = (p1 - p0) / L
            n = np.array([u[1], -u[0]]) if ccw else np.array([-u[1], u[0]])
            ws.append({"room": r, "ri": ri, "wall": w, "key": (ri, wi), "p0": p0, "p1": p1,
                       "u": u, "n": n, "L": L, "cuts": [], "interior": False})
        walls.extend(ws)
    return walls


def _finish_walls(walls, t, room_polys):
    for w in walls:
        same = [o for o in walls if o["ri"] == w["ri"] and o is not w]
        prev = next((o for o in same if np.linalg.norm(o["p1"] - w["p0"]) < SNAP), None)
        nxt = next((o for o in same if np.linalg.norm(o["p0"] - w["p1"]) < SNAP), None)
        w["o0"] = _mitre(w["p0"], prev["n"], w["n"], t) if prev else w["p0"] + t * w["n"]
        w["o1"] = _mitre(w["p1"], w["n"], nxt["n"], t) if nxt else w["p1"] + t * w["n"]
        # a reflex (inward) corner at either end: outside dimension lines of the two walls would
        # cross each other there, so such walls are labelled on the room's inner side instead
        w["reflex"] = bool((prev is not None and float(np.dot(prev["n"], w["u"])) > 0.3)
                           or (nxt is not None and float(np.dot(nxt["n"], w["u"])) < -0.3))
        # another room behind this wall? probe at three points along it
        probes = [w["p0"] + w["u"] * w["L"] * f + w["n"] * PROBE for f in (0.25, 0.5, 0.75)]
        w["interior"] = any(poly is not None and poly.distance(Point(*p)) < 0.05
                            for ri, poly in room_polys if ri != w["ri"] for p in probes)


def _resolve_openings(rooms, walls):
    by_id = {}
    for w in walls:
        by_id.setdefault(w["wall"].get("id"), w)
    ops = []
    for r in rooms:
        for o in r.get("openings") or []:
            w = by_id.get(o.get("wall_id"))
            c = o.get("center")
            if w is None or not c or len(c) < 2:
                continue  # unknown wall or no position: nothing sensible to draw
            c = np.asarray(c[:2], float)
            width = _val(o.get("width"))
            sc = float(np.dot(c - w["p0"], w["u"]))
            hw = (width or 0.0) / 2
            s0, s1 = max(0.0, sc - hw), min(w["L"], sc + hw)
            op = {"op": o, "wall": w, "c": c, "sc": sc, "s0": s0, "s1": s1, "width": width,
                  "far": 0.0, "partners": []}
            ops.append(op)
            if not width or s1 <= s0:
                continue
            w["cuts"].append((s0, s1))
            for p in walls:  # the other face of the same wall (another room's wall)
                if p["ri"] == w["ri"] or abs(float(np.dot(p["u"], w["u"]))) < 0.99:
                    continue
                d = float(np.dot(c - p["p0"], p["n"]))  # signed distance from the partner line
                if abs(d) > PARTNER_DIST:
                    continue
                pc = float(np.dot(c - p["p0"], p["u"]))
                a, b = max(0.0, pc - hw), min(p["L"], pc + hw)
                if b > a:
                    p["cuts"].append((a, b))
                    op["partners"].append(p)
                    # distance from the owner's inner face to the partner's inner face
                    op["far"] = max(op["far"], abs(float(np.dot(p["p0"] - w["p0"], w["n"]))))
    return ops


def _pieces(L, cuts):
    """Solid stretches [a, b] of a wall of length L once the opening gaps are removed."""
    out, s = [], 0.0
    for a, b in sorted(cuts):
        if a > s + 1e-6:
            out.append((s, a))
        s = max(s, b)
    if s < L - 1e-6:
        out.append((s, L))
    return out


def _inner(w, s):
    return w["p0"] + w["u"] * s


def _outer(w, s, t):
    if s <= 1e-9:
        return w["o0"]
    if s >= w["L"] - 1e-9:
        return w["o1"]
    return _inner(w, s) + w["n"] * t


# ----------------------------------------------------------------------------- label placement
class _Placer:
    def __init__(self):
        self.obstacles = []

    def block(self, geom):
        self.obstacles.append(geom)

    def free(self, geom) -> bool:
        return not any(geom.intersects(o) for o in self.obstacles)

    def place(self, candidates):
        """candidates: list of (x, y, angle, w, h, extra). First free one wins, else the first."""
        if not candidates:
            return None
        chosen = candidates[0]
        for c in candidates:
            if self.free(_rect(c[0], c[1], c[3] + 4, c[4] + 2, c[2])):
                chosen = c
                break
        self.block(_rect(chosen[0], chosen[1], chosen[3] + 4, chosen[4] + 2, chosen[2]))
        return chosen


def _text(x, y, s, size=12, angle=0.0, anchor="middle", weight=None, fill=INK, halo=True, cls=None):
    attrs = [f'x="{_f(x)}"', f'y="{_f(y)}"', f'font-size="{size}"', f'text-anchor="{anchor}"',
             'dy="0.35em"', f'fill="{fill}"']
    if weight:
        attrs.append(f'font-weight="{weight}"')
    if halo:
        attrs.append('stroke="#ffffff" stroke-width="3" stroke-linejoin="round" paint-order="stroke"')
    if angle:
        attrs.append(f'transform="rotate({angle:g} {_f(x)} {_f(y)})"')
    if cls:
        attrs.append(f'class="{cls}"')
    return f"<text {' '.join(attrs)}>{escape(s)}</text>"


# ----------------------------------------------------------------------------- main entry point
def render(plan: dict) -> str:
    """plan dict (schema 1.0) -> SVG string. Never raises on missing/null optional fields."""
    rooms = [r for r in (plan.get("rooms") or []) if isinstance(r, dict)]
    walls = _build_walls(rooms)

    pts = [p[:2] for r in rooms for p in (r.get("polygon") or []) if p and len(p) >= 2]
    pts += [w["p0"] for w in walls] + [w["p1"] for w in walls]
    empty = not pts
    if empty:
        lo, hi, px = np.zeros(2), np.array([4.0, 2.0]), float(PX_MAX)
    else:
        arr = np.asarray(pts, float)
        lo, hi = arr.min(0), arr.max(0)
        span = float(max(hi[0] - lo[0], hi[1] - lo[1], 1e-6))
        px = float(min(PX_MAX, max(PX_MIN, PLAN_PX / span)))
    plan_w, plan_h = (hi - lo) * px
    W = max(MIN_W, plan_w + 2 * MARGIN)
    H = HEADER + plan_h + 2 * MARGIN + FOOTER
    view = _View(lo, hi, px, (W - plan_w) / 2, HEADER + MARGIN)
    t = WALL_PX / px  # wall band thickness in metres

    room_polys = []
    for ri, r in enumerate(rooms):
        poly = r.get("polygon") or []
        try:
            P = Polygon([tuple(p[:2]) for p in poly]) if len(poly) >= 3 else None
            if P is not None and not P.is_valid:
                P = P.buffer(0)
        except (TypeError, ValueError):
            P = None
        room_polys.append((ri, P))
    _finish_walls(walls, t, room_polys)
    ops = _resolve_openings(rooms, walls)

    cap = plan.get("capture_id") or "capture"
    tier = {"lidar": "LiDAR", "video": "Video", "photos": "Photos"}.get(plan.get("tier"), str(plan.get("tier") or "?"))
    iv = plan.get("intervals") or {}
    level = iv.get("level")
    lvl = f"{level * 100:.0f}%" if isinstance(level, (int, float)) else "90%"

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W:.0f}" height="{H:.0f}" '
           f'viewBox="0 0 {W:.0f} {H:.0f}" font-family="{FONT}">',
           f"<title>{escape(f'Floor plan of {cap}. The arrow is the plan +y axis; true north is unknown.')}</title>",
           f'<rect width="{W:.0f}" height="{H:.0f}" fill="#ffffff"/>']

    # header
    fa = plan.get("footprint_area")
    sub = [tier]
    if _val(fa) is not None:
        hw = half_width(fa)
        sub.append(f"footprint {fmt_area(fa)}" + (f" ±{hw:.2f}" if hw is not None else ""))
    sub.append(f"± = {lvl} interval" + ("" if iv.get("calibrated") else " (provisional)"))
    sub.append("+y is the plan axis, not north")
    out.append(_text(24, 22, f"Floor plan · {cap}", size=17, anchor="start", weight="bold", halo=False))
    out.append(_text(24, 42, " · ".join(sub), size=12, anchor="start", fill=MUTED, halo=False))

    placer = _Placer()

    if empty:
        cx, cy = W / 2, HEADER + MARGIN + plan_h / 2
        out.append(_text(cx, cy, "No rooms reconstructed", size=18, fill=MUTED, halo=False))

    # floors
    out.append('<g id="floors">')
    for (ri, P), r in zip(room_polys, rooms):
        if P is None:
            continue
        out.append(f'<polygon points="{view.pts(r["polygon"])}" fill="{FLOOR}" stroke="none" '
                   f'data-room="{escape(str(r.get("id", "")))}"/>')
    out.append("</g>")

    # wall bands (pieces between opening gaps)
    out.append('<g id="walls">')
    for w in walls:
        observed = bool(w["wall"].get("observed", True))
        wid = escape(str(w["wall"].get("id", "")))
        for a, b in _pieces(w["L"], w["cuts"]):
            quad = [_inner(w, a), _inner(w, b), _outer(w, b, t), _outer(w, a, t)]
            sp = [view(p) for p in quad]
            placer.block(Polygon(sp))
            if observed:
                out.append(f'<polygon points="{view.pts(quad)}" fill="{INK}" stroke="{INK}" '
                           f'stroke-width="0.6" stroke-linejoin="miter" data-wall="{wid}"/>')
            else:
                m0, m1 = view(_inner(w, a) + w["n"] * t / 2), view(_inner(w, b) + w["n"] * t / 2)
                out.append(f'<polygon points="{view.pts(quad)}" fill="{INFERRED_BAND}" stroke="none" data-wall="{wid}"/>')
                out.append(f'<line x1="{_f(m0[0])}" y1="{_f(m0[1])}" x2="{_f(m1[0])}" y2="{_f(m1[1])}" '
                           f'stroke="{INFERRED}" stroke-width="{WALL_PX}" stroke-dasharray="10,6" '
                           f'class="wall-inferred" data-wall="{wid}"/>')
    # rooms without wall records: outline the polygon thinly so the room is still visible
    for (ri, P), r in zip(room_polys, rooms):
        if P is not None and not (r.get("walls") or []):
            out.append(f'<polygon points="{view.pts(r["polygon"])}" fill="none" stroke="{INFERRED}" '
                       f'stroke-width="2" stroke-dasharray="6,4"/>')
    out.append("</g>")

    # openings
    out.append('<g id="openings">')
    for op in ops:
        o, w = op["op"], op["wall"]
        kind = o.get("kind")
        oid = escape(str(o.get("id", "")))
        if not op["width"] or op["s1"] <= op["s0"]:
            x, y = view(op["c"])
            out.append(f'<circle cx="{_f(x)}" cy="{_f(y)}" r="5" fill="#ffffff" stroke="{INK}" '
                       f'stroke-width="1.5" data-opening="{oid}"/>')
            continue
        a, b = op["s0"], op["s1"]
        ia, ib = _inner(w, a), _inner(w, b)
        depth = max(t, op["far"])  # through both faces of an interior wall
        oa, ob = ia + w["n"] * depth, ib + w["n"] * depth
        if kind == "window":
            out.append(f'<polygon points="{view.pts([ia, ib, ob, oa])}" fill="{WINDOW_GLASS}" stroke="none"/>')
            for p, q in ((ia, ib), (oa, ob)):
                (x1, y1), (x2, y2) = view(p), view(q)
                out.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{INK}" '
                           f'stroke-width="1.4" class="window" data-opening="{oid}"/>')
        # jambs: short ticks across the wall at both ends of the gap
        for p, q in ((ia, oa), (ib, ob)):
            (x1, y1), (x2, y2) = view(p), view(q)
            out.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{INK}" stroke-width="1.4"/>')
        if kind == "door":
            r_m = op["width"]
            hinge, tip = ia, ia - w["n"] * r_m
            (hx, hy), (tx_, ty_), (ex, ey) = view(hinge), view(tip), view(ib)
            cross = (tx_ - hx) * (ey - hy) - (ty_ - hy) * (ex - hx)
            sweep = 1 if cross > 0 else 0
            rp = r_m * px
            out.append(f'<line x1="{_f(hx)}" y1="{_f(hy)}" x2="{_f(tx_)}" y2="{_f(ty_)}" stroke="{INK}" '
                       f'stroke-width="2" stroke-linecap="round" class="door-leaf" data-opening="{oid}"/>')
            out.append(f'<path d="M {_f(tx_)} {_f(ty_)} A {_f(rp)} {_f(rp)} 0 0 {sweep} {_f(ex)} {_f(ey)}" '
                       f'fill="none" stroke="{INK}" stroke-width="1" class="door-arc" data-opening="{oid}"/>')
            # the swing quarter is an obstacle for labels
            placer.block(Polygon([view(p) for p in (hinge, ib, ib - w["n"] * r_m, tip)]))
        elif kind != "window":  # passage: a plain gap, threshold drawn faint
            (x1, y1), (x2, y2) = view(ia), view(ib)
            out.append(f'<line x1="{_f(x1)}" y1="{_f(y1)}" x2="{_f(x2)}" y2="{_f(y2)}" stroke="{INFERRED}" '
                       f'stroke-width="1" stroke-dasharray="3,3" class="passage" data-opening="{oid}"/>')
    out.append("</g>")

    # room labels (fixed position: pole of inaccessibility, always inside even for an L-shape)
    labels = []
    for (ri, P), r in zip(room_polys, rooms):
        if P is None or P.is_empty:
            continue
        try:
            q = polylabel(P, tolerance=0.01) if P.geom_type == "Polygon" else P.representative_point()
        except Exception:  # noqa: BLE001 - degenerate polygon: fall back to shapely's interior point
            q = P.representative_point()
        x, y = view((q.x, q.y))
        name = str(r.get("name") or r.get("id") or "Room")
        ch = r.get("ceiling_height")
        hline = f"h {_val(ch):.2f} m" if _val(ch) is not None else "h not observed"
        lines = [(name, 14, "bold", INK), (fmt_area(r.get("floor_area")), 13, None, INK), (hline, 12, None, MUTED)]
        wmax = max(_text_w(s, sz) for s, sz, _, _ in lines)
        placer.block(box(x - wmax / 2 - 4, y - 28, x + wmax / 2 + 4, y + 28))
        for k, (s, sz, wt, col) in enumerate(lines):
            labels.append(_text(x, y + (k - 1) * 17, s, size=sz, weight=wt, fill=col, cls="room-label"))

    # damage pins (fixed)
    pins = []
    for d in plan.get("damage") or []:
        pos = d.get("position")
        if not pos or len(pos) < 2 or pos[0] is None or pos[1] is None:
            continue
        x, y = view(pos[:2])
        did = escape(str(d.get("id", "")))
        cls = escape(str(d.get("class", "damage")))
        placer.block(Point(x, y).buffer(12))
        pins.append(f'<g class="damage-pin"><title>{did}: {cls} on {escape(str(d.get("surface_id", "")))}</title>'
                    f'<circle cx="{_f(x)}" cy="{_f(y)}" r="11" fill="{DAMAGE}" stroke="#ffffff" stroke-width="2"/>'
                    + _text(x, y, str(d.get("id", "")), size=10, weight="bold", fill="#ffffff", halo=False)
                    + "</g>")

    # opening labels
    for op in ops:
        o, w = op["op"], op["wall"]
        kind = str(o.get("kind") or "opening")
        s = f"{kind} {fmt_len(o.get('width'))}"
        size = 11
        tw, th = _text_w(s, size), size + 2
        ang = _screen_angle(w["u"])
        far = max(t, op["far"])
        cands = []
        along = [op["sc"]] + [op["sc"] + d * (op["s1"] - op["s0"]) / 2 for d in (0.6, -0.6)]
        if kind == "door":  # label on the side away from the swing, past the far face of the wall
            offs = [far + ROW1, far + ROW2, -(op["width"] or 0.5) - ROW1]
        else:  # windows and passages: just inside the room, then outside
            offs = [-ROW1, -ROW2, far + ROW1, far + ROW2]
        for off in offs:
            for sa in along:
                x, y = view(_inner(w, sa) + w["n"] * off)
                cands.append((x, y, ang, tw, th))
        x, y, ang, _, _ = placer.place(cands)
        labels.append(_text(x, y, s, size=size, angle=ang, fill="#1d4f91", cls="opening-label"))

    # wall length labels (+ a dimension line for walls labelled outside)
    dims = []
    for w in sorted(walls, key=lambda w: (w["interior"] or w["reflex"], -w["L"])):  # outside labels, long walls first
        s = fmt_len(w["wall"].get("length"))
        size = 12
        tw, th = _text_w(s, size), size + 2
        ang = _screen_angle(w["u"])
        fracs = (0.5, 0.62, 0.38, 0.75, 0.25, 0.85, 0.15)
        inside = w["interior"] or w["reflex"]
        offs = (-ROW1, -ROW2) if inside else (t + ROW1, t + ROW2)
        cands = []
        for off in offs:
            for fr in fracs:
                x, y = view(_inner(w, w["L"] * fr) + w["n"] * off)
                cands.append((x, y, ang, tw, th, off, fr))
        x, y, ang, _, _, off, fr = placer.place(cands)
        col = INK if w["wall"].get("observed", True) else MUTED
        if not inside:
            # dimension line with end ticks, broken where the label sits
            sl, half = w["L"] * fr, (tw / 2 + 5) / px
            dash = "" if w["wall"].get("observed", True) else ' stroke-dasharray="4,3"'
            for sa, sb in ((0.0, sl - half), (sl + half, w["L"])):
                if sb - sa <= 1e-6:
                    continue
                a, b = view(_inner(w, sa) + w["n"] * off), view(_inner(w, sb) + w["n"] * off)
                dims.append(f'<line x1="{_f(a[0])}" y1="{_f(a[1])}" x2="{_f(b[0])}" y2="{_f(b[1])}" '
                            f'stroke="{MUTED}" stroke-width="0.8"{dash}/>')
            a, b = view(w["p0"] + w["n"] * off), view(w["p1"] + w["n"] * off)
            nx, ny = view(w["p0"] + w["n"] * (off + 0.05)), view(w["p0"] + w["n"] * (off - 0.05))
            tick = (nx[0] - ny[0], nx[1] - ny[1])
            for e in (a, b):
                dims.append(f'<line x1="{_f(e[0] - tick[0])}" y1="{_f(e[1] - tick[1])}" '
                            f'x2="{_f(e[0] + tick[0])}" y2="{_f(e[1] + tick[1])}" stroke="{MUTED}" stroke-width="0.8"/>')
        labels.append(_text(x, y, s, size=size, angle=ang, fill=col, cls="wall-label"))

    out.append('<g id="dimensions">' + "".join(dims) + "</g>")
    out.append('<g id="labels">' + "".join(labels) + "</g>")
    out.append('<g id="damage">' + "".join(pins) + "</g>")
    out.extend(_footer(W, H, px, lvl))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _footer(W, H, px, lvl="90%"):
    """Scale bar (0-1-2 m, true to plan scale), +y axis arrow and legend."""
    y0 = H - FOOTER + 18
    x0 = 24
    g = ['<g id="scale-bar"><title>Scale: each segment is 1 m at plan scale</title>']
    for k in range(2):
        g.append(f'<rect x="{_f(x0 + k * px)}" y="{_f(y0)}" width="{_f(px)}" height="6" '
                 f'fill="{INK if k == 0 else "#ffffff"}" stroke="{INK}" stroke-width="1"/>')
    for k, s in enumerate(("0", "1 m", "2 m")):
        g.append(_text(x0 + k * px, y0 + 18, s, size=11, fill=INK, halo=False))
    g.append("</g>")
    # axis arrow
    ax, ay = W - 40, y0 - 2
    g.append('<g id="axis"><title>Plan +y axis. True north is unknown: the capture has no compass.</title>'
             f'<line x1="{_f(ax)}" y1="{_f(ay + 26)}" x2="{_f(ax)}" y2="{_f(ay)}" stroke="{INK}" stroke-width="2"/>'
             f'<polygon points="{_f(ax)},{_f(ay - 8)} {_f(ax - 5)},{_f(ay + 2)} {_f(ax + 5)},{_f(ay + 2)}" fill="{INK}"/>'
             + _text(ax - 10, ay + 14, "+y (plan)", size=11, anchor="end", fill=INK, halo=False)
             + "</g>")
    # legend
    ly = H - 30
    lx = 24
    g.append('<g id="legend">')
    g.append(f'<line x1="{lx}" y1="{_f(ly)}" x2="{lx + 34}" y2="{_f(ly)}" stroke="{INK}" stroke-width="{WALL_PX}"/>')
    g.append(_text(lx + 42, ly, "observed wall", size=11, anchor="start", fill=INK, halo=False))
    lx += 140
    g.append(f'<line x1="{lx}" y1="{_f(ly)}" x2="{lx + 34}" y2="{_f(ly)}" stroke="{INFERRED_BAND}" stroke-width="{WALL_PX}"/>')
    g.append(f'<line x1="{lx}" y1="{_f(ly)}" x2="{lx + 34}" y2="{_f(ly)}" stroke="{INFERRED}" '
             f'stroke-width="{WALL_PX}" stroke-dasharray="10,6"/>')
    g.append(_text(lx + 42, ly, "inferred (not seen)", size=11, anchor="start", fill=INK, halo=False))
    lx += 160
    g.append(_text(lx, ly, f"± = {lvl} interval half-width", size=11, anchor="start", fill=INK, halo=False))
    lx += 178
    g.append(f'<circle cx="{lx + 7}" cy="{_f(ly)}" r="7" fill="{DAMAGE}"/>')
    g.append(_text(lx + 20, ly, "damage", size=11, anchor="start", fill=INK, halo=False))
    g.append("</g>")
    return g
