"""Interval calibration: one interval model for every Measure, with constants fitted on evidence.

The model (the same one fp/contract.py used provisionally since work order 01, now with fitted constants):

    half = k * a + (b + m * extra_rel) * value

 - `a` is the measurement's own evidence term, computed in fp/contract.py: plane position errors from the
   wall band's thickness (smear), x3 for a wall never seen, jamb reveals for openings, the ceiling layer's
   patch spread. It is what makes one wall's interval wider than another's.
 - `b * value` is the relative term: a learned metric scale (camera tiers) makes every length wrong by the
   same factor, so its error grows with the value.
 - `extra_rel` is added when the camera tiers' learned scale disagrees with priors (fp/recon/camera.scale_check);
   m = 2 for areas (a scale error s changes an area by about 2s), else 1.
 - `k` and `b` are per tier and measurement type. Before any fit: k = 1 and b = SCALE_REL[tier] (x2 for areas),
   exactly the provisional model of D5.

Fitting (split conformal, simple on purpose): for each tier and type, every evidence row gives the factor
its interval would have needed to cover the reference value, and the constant is the ceil((n+1) x 0.9)-th
smallest such factor (the standard finite-sample 90% bound). With n < 9 that rank does not exist, so the
largest factor is used and the fit is flagged `small_n`. Two guards, because we have no tape yet:
 - a fitted constant never goes below its provisional value (with no tape we may widen, never narrow);
 - LiDAR fits k (its scale is metric, b stays 0); camera tiers fit b with k = 1 (their error is dominated by
   scale and shape, not by the plane fits).
The fitted constants live in fp/calibration.json (committed, written by eval/run_benchmark.py) with their
source; `fp run` reads them. Delete the file to go back to the provisional model.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

LEVEL = 0.9
TYPES = ("wall_length", "perimeter", "floor_area", "ceiling", "opening")
AREA_TYPES = {"floor_area"}
# Provisional relative terms (D5): the brief's own gates, 3 % video and 8 % photos; LiDAR is metric.
SCALE_REL = {"lidar": 0.0, "video": 0.03, "photos": 0.08}
FIT_PARAM = {"lidar": "k", "video": "b", "photos": "b"}   # which constant the evidence moves, per tier
PATH = Path(__file__).with_name("calibration.json")


def provisional() -> dict:
    return {t: {ty: {"k": 1.0, "b": SCALE_REL[t] * (2 if ty in AREA_TYPES else 1), "fitted": False}
                for ty in TYPES} for t in SCALE_REL}


_cache: dict | None = None


def load(path: Path | None = None) -> dict:
    """{"constants": {tier: {type: {k, b, fitted, n, source}}}, "source": str} (provisional if no file)."""
    global _cache
    if path is None and _cache is not None:
        return _cache
    p = Path(path) if path else PATH
    data = {"constants": provisional(), "source": "provisional (D5): no calibration file", "file": None}
    if p.is_file():
        fitted = json.loads(p.read_text())
        for t, types in fitted.get("constants", {}).items():
            for ty, c in types.items():
                data["constants"].setdefault(t, {})[ty] = {**data["constants"][t].get(ty, {}), **c}
        data["source"] = fitted.get("source", str(p))
        data["file"] = str(p)
    if path is None:
        _cache = data
    return data


def reset_cache() -> None:
    global _cache
    _cache = None


def const(tier: str, typ: str) -> dict:
    return load()["constants"][tier][typ]


def half(tier: str, typ: str, a: float, value: float, extra_rel: float = 0.0) -> float:
    """The interval half-width for one measurement: k * a + (b + m * extra_rel) * value."""
    c = const(tier, typ)
    m = 2 if typ in AREA_TYPES else 1
    return c["k"] * a + (c["b"] + m * extra_rel) * abs(value)


def tier_calibrated(tier: str) -> bool:
    return any(c.get("fitted") for c in load()["constants"][tier].values())


def method(tier: str) -> str:
    cs = load()["constants"][tier]
    fitted = [f"{ty} ({'k' if FIT_PARAM[tier] == 'k' else 'b'}={c[FIT_PARAM[tier]]:.3g}, n={c.get('n')})"
              for ty, c in cs.items() if c.get("fitted")]
    if not fitted:
        return "provisional (D5, fp/contract.py): not calibrated on any evidence for this tier"
    rest = [ty for ty, c in cs.items() if not c.get("fitted")]
    return (f"half = k*a + b*value; fitted on {load()['source']}: " + ", ".join(fitted)
            + (f"; provisional: {', '.join(rest)}" if rest else ""))


# --- fitting ---

def conformal_rank(n: int, level: float = LEVEL) -> int | None:
    """1-based rank of the split-conformal bound among n sorted scores; None when n is too small."""
    r = math.ceil((n + 1) * level)
    return r if r <= n else None


def quantile(scores: list[float], level: float = LEVEL) -> tuple[float, bool]:
    """(bound, small_n). small_n: n < 9, the largest score is used."""
    s = sorted(scores)
    r = conformal_rank(len(s), level)
    return (s[r - 1], False) if r else (s[-1], True)


def needed(row: dict, param: str, k: float = 1.0, b: float = 0.0) -> float:
    """The smallest k (or b) for which this row's interval covers its reference value. For areas, b is per
    unit area and already holds the factor 2 of a scale error."""
    err, a, v = abs(row["err"]), row["a"], abs(row["value"])
    m = 2 if row["type"] in AREA_TYPES else 1
    extra = m * row.get("extra_rel", 0.0) * v
    if param == "k":
        rest = err - b * v - extra
        return max(rest, 0.0) / a if a > 0 else (math.inf if rest > 0 else 0.0)
    rest = err - k * a - extra
    return max(rest, 0.0) / v if v > 0 else (math.inf if rest > 0 else 0.0)


def fit_one(rows: list[dict], tier: str, typ: str) -> dict | None:
    """Fit one constant for (tier, type) on rows; None when there are no rows."""
    rows = [r for r in rows if r["tier"] == tier and r["type"] == typ]
    if not rows:
        return None
    p = provisional()[tier][typ]
    param = FIT_PARAM[tier]
    scores = [needed(r, param, k=p["k"], b=p["b"]) for r in rows]
    q, small = quantile(scores)
    out = dict(p)
    out[param] = round(max(q, p[param]), 4)     # never narrower than provisional without tape
    out.update(fitted=True, n=len(rows), small_n=small, raw=round(q, 4), floored=q < p[param])
    return out


def coverage(rows: list[dict], consts: dict) -> dict:
    """Observed coverage of [value - half, value + half] over rows, using consts[tier][type]."""
    hit = []
    for r in rows:
        c = consts[r["tier"]][r["type"]]
        m = 2 if r["type"] in AREA_TYPES else 1
        h = c["k"] * r["a"] + (c["b"] + m * r.get("extra_rel", 0.0)) * abs(r["value"])
        hit.append(abs(r["err"]) <= h + 1e-9)
    return {"n": len(hit), "covered": int(sum(hit)), "frac": round(sum(hit) / len(hit), 3) if hit else None}


def fit(rows: list[dict]) -> dict:
    """Fit every (tier, type) that has rows; the rest stay provisional."""
    consts = provisional()
    for t in consts:
        for ty in TYPES:
            f = fit_one(rows, t, ty)
            if f:
                consts[t][ty] = f
    # perimeter follows wall length when it has no evidence of its own (same plane errors, summed)
    for t in consts:
        if consts[t]["wall_length"].get("fitted") and not consts[t]["perimeter"].get("fitted"):
            consts[t]["perimeter"] = {**consts[t]["wall_length"], "n": 0, "from": "wall_length"}
    return consts


def two_fold(rows: list[dict]) -> list[dict]:
    """Fit on one group (capture or room set), check coverage on the other, both ways, per tier and type."""
    out = []
    for t in SCALE_REL:
        for ty in TYPES:
            sub = [r for r in rows if r["tier"] == t and r["type"] == ty]
            groups = sorted({r["fold"] for r in sub})
            if len(groups) < 2:
                if sub:
                    out.append({"tier": t, "type": ty, "fit_on": groups[0] if groups else None, "test_on": None,
                                "n_fit": len(sub), "n_test": 0, "coverage": None, "note": "one fold only"})
                continue
            for g in groups:
                fit_rows = [r for r in sub if r["fold"] != g]
                test_rows = [r for r in sub if r["fold"] == g]
                consts = provisional()
                consts[t][ty] = fit_one(fit_rows, t, ty)
                cov = coverage(test_rows, consts)
                prov = coverage(test_rows, provisional())
                out.append({"tier": t, "type": ty, "fit_on": "+".join(x for x in groups if x != g), "test_on": g,
                            "n_fit": len(fit_rows), "n_test": len(test_rows), "constant": consts[t][ty][FIT_PARAM[t]],
                            "coverage": cov["frac"], "coverage_provisional": prov["frac"]})
    return out
