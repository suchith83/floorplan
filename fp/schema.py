"""The published plan.json contract, schema version "1.0".

Every tier (LiDAR, video, photos) writes one ``plan.json`` that validates against ``Property``.
Run ``python -m fp.schema`` to regenerate ``schema/plan.schema.json`` from these models.
Human-readable notes: ``docs/SCHEMA.md``.

Conventions
- Units and frame: lengths in metres, areas in m². Plan frame: z up, floor at z = 0, x/y in metres
  along the dominant wall axes.
- Every number is a ``Measure``: ``value`` plus ``lo``/``hi``, a stated 90% interval (at least 90%
  of true values should fall in [lo, hi]; ``intervals.level`` records the level), a ``method``
  string, and ``observed``.
- Never invent a number: if something was not measured, ``value`` is null and ``observed`` is
  false. An inferred or assumed value (``observed`` false with a value) is allowed, but it should
  carry a wide interval.
- Stable IDs: rooms ``R1``, walls ``R1.W1``, openings ``R1.O1``, damage ``D1``, concealed-damage
  flags ``C1``, scope items ``S1``. Surface IDs are every wall ID plus ``R1.floor`` and
  ``R1.ceiling`` per room. Damage and scope items point at surfaces by these IDs, and the
  cross-reference checks on ``Property`` keep them consistent.
- Dump with ``plan.dump()`` (= ``model_dump(mode="json", by_alias=True)``) so ``Damage.class_``
  is written as ``"class"``.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_VERSION = "1.0"
REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schema" / "plan.schema.json"
SCHEMA_ID = "urn:floorplan:plan:1.0"
SCHEMA_TITLE = "Floorplan plan.json (schema 1.0)"

Unit = Literal["m", "m2", "count"]

# Fixed-length coordinate arrays: min/max length so the JSON Schema states the arity.
Point2 = Annotated[list[float], Field(min_length=2, max_length=2, description="[x, y] in metres")]
Point3 = Annotated[list[float], Field(min_length=3, max_length=3, description="[x, y, z] in metres")]
RoomPair = Annotated[list[str], Field(min_length=2, max_length=2, description="two room ids")]
Row4 = Annotated[list[float], Field(min_length=4, max_length=4)]
Matrix4 = Annotated[list[Row4], Field(min_length=4, max_length=4)]

# ID patterns (1-based numbering).
_N = r"[1-9][0-9]*"
RoomId = Annotated[str, Field(pattern=rf"^R{_N}$")]
WallId = Annotated[str, Field(pattern=rf"^R{_N}\.W{_N}$")]
OpeningId = Annotated[str, Field(pattern=rf"^R{_N}\.O{_N}$")]
DamageId = Annotated[str, Field(pattern=rf"^D{_N}$")]
ConcealedId = Annotated[str, Field(pattern=rf"^C{_N}$")]
ScopeId = Annotated[str, Field(pattern=rf"^S{_N}$")]


class _Model(BaseModel):
    # Published contract: unknown keys are errors; NaN/inf are not valid JSON numbers (write null).
    model_config = ConfigDict(
        extra="forbid",
        allow_inf_nan=False,
        validate_by_name=True,
        validate_by_alias=True,
        serialize_by_alias=True,
    )


class Measure(_Model):
    """A number with a stated 90% interval [lo, hi]. value null = not measured (observed false)."""

    value: float | None
    lo: float | None
    hi: float | None
    unit: Unit
    method: str
    observed: bool

    @model_validator(mode="after")
    def _rules(self) -> Measure:
        if self.value is None:
            if self.observed:
                raise ValueError("value is null but observed is true (never invent a number)")
        else:
            if self.lo is None or self.hi is None:
                raise ValueError("value is set, so lo and hi are required")
            if not (self.lo <= self.value <= self.hi):
                raise ValueError(f"need lo <= value <= hi, got {self.lo} <= {self.value} <= {self.hi}")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"lo {self.lo} > hi {self.hi}")
        return self


class Frame(_Model):
    """The plan frame: z up, floor at floor_z, metres. T_plan_world maps world → plan (4×4)."""

    up: Literal["z"]
    floor_z: float
    units: Literal["m"]
    T_plan_world: Matrix4 | None


class Intervals(_Model):
    """How the intervals were made. scale = interval multiplier vs LiDAR (report badge "intervals ×N")."""

    level: float = Field(gt=0, lt=1, description="nominal coverage of every [lo, hi], e.g. 0.9")
    scale: float = Field(gt=0)
    calibrated: bool
    method: str


class Wall(_Model):
    """A wall segment p0→p1 of a room. coverage = fraction of the wall seen in the capture."""

    id: WallId
    p0: Point2
    p1: Point2
    length: Measure
    observed: bool
    coverage: float = Field(ge=0, le=1)


class Opening(_Model):
    """A door, window or passage, listed once under the room that owns wall_id."""

    id: OpeningId
    kind: Literal["door", "window", "passage"]
    wall_id: WallId
    rooms: RoomPair | None = Field(description="the two rooms it connects; null for windows/exterior doors")
    width: Measure
    height: Measure | None
    sill: Measure | None
    center: Point2


class Room(_Model):
    """One room. polygon: counter-clockwise, not closed (last point != first)."""

    id: RoomId
    name: str
    polygon: list[Point2] = Field(min_length=3)
    floor_area: Measure
    ceiling_height: Measure
    perimeter: Measure
    walls: list[Wall]
    openings: list[Opening]

    @model_validator(mode="after")
    def _polygon(self) -> Room:
        pts = self.polygon
        if pts[0] == pts[-1]:
            raise ValueError(f"{self.id}: polygon must not be closed (drop the repeated last point)")
        area2 = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(pts, pts[1:] + pts[:1]))
        if area2 <= 0:
            raise ValueError(f"{self.id}: polygon must be counter-clockwise with non-zero area")
        return self


class Connection(_Model):
    """Two rooms are adjacent. opening_id null = adjacency without a detected opening."""

    rooms: RoomPair
    opening_id: OpeningId | None


class Drift(_Model):
    """Drift correction (loop closure etc.): what ran and its numbers."""

    method: str
    enabled: bool
    metrics: dict[str, float | int | str | None]


class BBox(_Model):
    """Box on a surface. Walls: u = metres along the wall from p0, v = metres above the floor.
    Floor/ceiling: u, v = plan x, y."""

    u0: float
    v0: float
    u1: float
    v1: float

    @model_validator(mode="after")
    def _order(self) -> BBox:
        if self.u0 > self.u1 or self.v0 > self.v1:
            raise ValueError("bbox needs u0 <= u1 and v0 <= v1")
        return self


class Damage(_Model):
    """A detected damage region on one surface. JSON key "class" (Python attribute class_)."""

    id: DamageId
    class_: str = Field(alias="class")
    surface_id: str
    position: Point3 | None
    extent_m2: Measure
    bbox_on_surface: BBox
    evidence_image: str | None = Field(description="path relative to the plan.json's folder")
    score: float = Field(ge=0, le=1)


class ConcealedFlag(_Model):
    """A rule-based hypothesis about damage that cannot be seen (never a finding)."""

    id: ConcealedId
    damage_id: DamageId | None
    rule_id: str
    rule_text: str
    evidence: str
    label: Literal["hypothesis"]


class ScopeItem(_Model):
    """A repair line item on one surface. unit must equal quantity.unit."""

    id: ScopeId
    surface_id: str
    action: str
    quantity: Measure
    unit: Unit

    @model_validator(mode="after")
    def _unit(self) -> ScopeItem:
        if self.unit != self.quantity.unit:
            raise ValueError(f"{self.id}: unit {self.unit!r} != quantity.unit {self.quantity.unit!r}")
        return self


class PlanWarning(_Model):
    """Something a stage wants the reader to know (e.g. a stage not built yet, a ceiling not seen)."""

    model_config = ConfigDict(title="Warning")

    stage: str
    message: str


def _dupes(ids: list[str]) -> list[str]:
    return sorted(i for i, n in Counter(ids).items() if n > 1)


def _unit_is(m: Measure | None, unit: str, where: str, errors: list[str]) -> None:
    if m is not None and m.unit != unit:
        errors.append(f"{where}: unit must be {unit!r}, got {m.unit!r}")


class Property(_Model):
    """One capture's whole-property plan (the root of plan.json)."""

    schema_version: Literal["1.0"]
    tier: Literal["lidar", "video", "photos"]
    capture_id: str
    device: str | None
    frame: Frame
    intervals: Intervals
    source: dict[str, Any] = Field(description="free-form provenance: path, n_frames, filter stats")
    rooms: list[Room]
    connections: list[Connection]
    footprint_area: Measure
    drift: Drift
    damage: list[Damage]
    concealed: list[ConcealedFlag]
    scope: list[ScopeItem]
    timings: dict[str, float] = Field(description='seconds per stage, plus "total"')
    warnings: list[PlanWarning]
    debug: dict[str, Any] | None = Field(
        default=None,
        exclude_if=lambda v: v is None,
        description="free-form, not part of the contract",
    )

    def surface_ids(self) -> list[str]:
        """Every wall id plus R<n>.floor and R<n>.ceiling for each room."""
        out: list[str] = []
        for r in self.rooms:
            out += [w.id for w in r.walls] + [f"{r.id}.floor", f"{r.id}.ceiling"]
        return out

    def dump(self) -> dict[str, Any]:
        """The JSON-ready dict to write as plan.json."""
        return self.model_dump(mode="json", by_alias=True)

    @model_validator(mode="after")
    def _cross_refs(self) -> Property:
        errors: list[str] = []
        room_ids = [r.id for r in self.rooms]
        room_set = set(room_ids)
        errors += [f"duplicate room id {i}" for i in _dupes(room_ids)]

        wall_ids: list[str] = []
        openings: dict[str, Opening] = {}
        opening_ids: list[str] = []
        for r in self.rooms:
            own_walls = {w.id for w in r.walls}
            _unit_is(r.floor_area, "m2", f"{r.id}.floor_area", errors)
            _unit_is(r.ceiling_height, "m", f"{r.id}.ceiling_height", errors)
            _unit_is(r.perimeter, "m", f"{r.id}.perimeter", errors)
            for w in r.walls:
                wall_ids.append(w.id)
                if not w.id.startswith(r.id + "."):
                    errors.append(f"wall {w.id} is listed under room {r.id}")
                _unit_is(w.length, "m", f"{w.id}.length", errors)
            for o in r.openings:
                opening_ids.append(o.id)
                openings[o.id] = o
                if not o.id.startswith(r.id + "."):
                    errors.append(f"opening {o.id} is listed under room {r.id}")
                if o.wall_id not in own_walls:
                    errors.append(f"opening {o.id}: wall_id {o.wall_id} is not a wall of {r.id}")
                if o.rooms is not None:
                    if o.rooms[0] == o.rooms[1]:
                        errors.append(f"opening {o.id}: rooms must be two different rooms")
                    if r.id not in o.rooms:
                        errors.append(f"opening {o.id}: rooms {o.rooms} must include its owner {r.id}")
                    for rid in o.rooms:
                        if rid not in room_set:
                            errors.append(f"opening {o.id}: unknown room {rid}")
                for name in ("width", "height", "sill"):
                    _unit_is(getattr(o, name), "m", f"{o.id}.{name}", errors)
        errors += [f"duplicate wall id {i}" for i in _dupes(wall_ids)]
        errors += [f"duplicate opening id {i}" for i in _dupes(opening_ids)]

        for c in self.connections:
            for rid in c.rooms:
                if rid not in room_set:
                    errors.append(f"connection {c.rooms}: unknown room {rid}")
            if c.rooms[0] == c.rooms[1]:
                errors.append(f"connection {c.rooms}: rooms must be two different rooms")
            if c.opening_id is not None:
                o = openings.get(c.opening_id)
                if o is None:
                    errors.append(f"connection {c.rooms}: unknown opening {c.opening_id}")
                elif o.rooms is None or set(o.rooms) != set(c.rooms):
                    errors.append(f"connection {c.rooms}: opening {c.opening_id} connects {o.rooms}")

        surfaces = set(self.surface_ids())
        damage_ids = [d.id for d in self.damage]
        errors += [f"duplicate damage id {i}" for i in _dupes(damage_ids)]
        for d in self.damage:
            if d.surface_id not in surfaces:
                errors.append(f"damage {d.id}: unknown surface_id {d.surface_id}")
            _unit_is(d.extent_m2, "m2", f"{d.id}.extent_m2", errors)

        errors += [f"duplicate concealed id {i}" for i in _dupes([c.id for c in self.concealed])]
        for c in self.concealed:
            if c.damage_id is not None and c.damage_id not in damage_ids:
                errors.append(f"concealed {c.id}: unknown damage_id {c.damage_id}")

        errors += [f"duplicate scope id {i}" for i in _dupes([s.id for s in self.scope])]
        for s in self.scope:
            if s.surface_id not in surfaces:
                errors.append(f"scope {s.id}: unknown surface_id {s.surface_id}")

        _unit_is(self.footprint_area, "m2", "footprint_area", errors)
        if "total" not in self.timings:
            errors.append('timings must include "total"')

        if errors:
            raise ValueError("; ".join(errors))
        return self


# --- convenience constructors -------------------------------------------------------------------


def not_observed(unit: Unit, method: str) -> Measure:
    """A quantity we did not measure: no number, observed false."""
    return Measure(value=None, lo=None, hi=None, unit=unit, method=method, observed=False)


def measure(
    value: float, lo: float, hi: float, unit: Unit, method: str, observed: bool = True
) -> Measure:
    """A measured (or, with observed=False, inferred) quantity with its 90% interval."""
    return Measure(value=value, lo=lo, hi=hi, unit=unit, method=method, observed=observed)


# --- validation and JSON Schema export ----------------------------------------------------------


def validate(plan: dict | str | Path) -> Property:
    """Validate a plan given as a dict, a JSON string, or a path to a JSON file.

    Raises pydantic.ValidationError if the plan breaks the contract, ValueError if a string is
    neither JSON nor an existing file, FileNotFoundError for a missing Path.
    """
    if isinstance(plan, Path):
        return Property.model_validate_json(plan.read_text())
    if isinstance(plan, str):
        if plan.lstrip().startswith("{"):
            return Property.model_validate_json(plan)
        p = Path(plan)
        if p.is_file():
            return Property.model_validate_json(p.read_text())
        raise ValueError(f"not a JSON object and not an existing file: {plan[:80]!r}")
    if isinstance(plan, dict):
        return Property.model_validate(plan)
    raise TypeError(f"validate() takes a dict, JSON string or Path, got {type(plan).__name__}")


def json_schema() -> dict[str, Any]:
    """The JSON Schema for plan.json, as published in schema/plan.schema.json."""
    body = Property.model_json_schema(by_alias=True)
    body.pop("title", None)
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": SCHEMA_TITLE,
        **body,
    }


def export_json_schema(path: Path = SCHEMA_PATH) -> Path:
    """Write the JSON Schema to `path` (default: <repo>/schema/plan.schema.json)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_schema(), indent=2, ensure_ascii=False) + "\n")
    return path


if __name__ == "__main__":
    print(f"wrote {export_json_schema()}")
