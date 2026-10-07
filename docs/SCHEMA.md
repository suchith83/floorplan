# plan.json, schema 1.0

Every tier (LiDAR, video, photos) writes the same `plan.json`. The models live in `fp/schema.py`.
The machine-readable JSON Schema is `schema/plan.schema.json`. The reference instance is
`tests/fixtures/plan_two_rooms.json`.

## Entity tree

```
Property (root)
├── schema_version "1.0", tier lidar|video|photos, capture_id, device
├── frame        {up "z", floor_z, units "m", T_plan_world 4×4 | null}
├── intervals    {level 0.9, scale, calibrated, method}
├── source       free-form provenance (path, n_frames, filter stats)
├── rooms[]      Room {id, name, polygon, floor_area, ceiling_height, perimeter}
│   ├── walls[]     Wall {id, p0, p1, length, observed, coverage}
│   └── openings[]  Opening {id, kind door|window|passage, wall_id, rooms, width, height, sill, center}
├── connections[] {rooms [a, b], opening_id | null}
├── footprint_area   Measure
├── drift        {method, enabled, metrics}
├── damage[]     Damage {id, class, surface_id, position, extent_m2, bbox_on_surface, evidence_image, score}
├── concealed[]  ConcealedFlag {id, damage_id, rule_id, rule_text, evidence, label "hypothesis"}
├── scope[]      ScopeItem {id, surface_id, action, quantity, unit}
├── timings      {stage: seconds, ..., "total"}
├── warnings[]   {stage, message}
└── debug        optional, free-form, not part of the contract
```

Unknown keys are errors everywhere except `source`, `drift.metrics` and `debug`.
NaN and infinity are not allowed: write `null` instead.

## Measure

Every number is a Measure: `{value, lo, hi, unit, method, observed}`.

- `unit` is `"m"`, `"m2"` or `"count"`.
- If `value` is set, `lo` and `hi` are required and `lo ≤ value ≤ hi`.
- If `value` is null, `observed` must be false. We never invent a number.
- `observed: false` with a value is allowed: an inferred wall, an assumed ceiling height.
  Its interval should be wide.
- `method` says in words how the number was made.

Areas are m2, lengths are m. A scope item's `unit` must equal its `quantity.unit`.

## What "90% interval" promises

`[lo, hi]` is a stated 90% interval (`intervals.level` = 0.9). The promise: across our test
captures, at least 90% of true values fall inside `[lo, hi]`. The coverage table in work order 07
checks this. `intervals.calibrated` says whether that check has been done for this tier.
`intervals.scale` is how much wider the intervals are than LiDAR's; the report shows it as
"intervals ×N".

## IDs

| ID | what | rule |
|---|---|---|
| `R1` | room | unique |
| `R1.W1` | wall of R1 | unique, starts with its room id |
| `R1.O1` | opening owned by R1 | unique, starts with its room id; `wall_id` is a wall of R1 |
| `R1.floor`, `R1.ceiling` | floor and ceiling surfaces of R1 | implied by the room |
| `D1` | damage region | unique; `surface_id` is a surface id |
| `C1` | concealed-damage hypothesis | unique; `damage_id` (if set) is a damage id |
| `S1` | scope line item | unique; `surface_id` is a surface id |

Numbering starts at 1. Surface ids are every wall id plus `R<n>.floor` and `R<n>.ceiling`.
Each opening is listed once, under the room that owns its wall. A door between two rooms has
`rooms: [owner, other]`; windows and exterior doors have `rooms: null`. A connection with an
`opening_id` must name the same two rooms as that opening. A connection with `opening_id: null`
means the rooms touch but no opening was found.

A plan with zero rooms is valid. It is what a run writes before its stages exist: empty lists,
`footprint_area` with value null and observed false, and a warning saying why.

## Frame and units

Metres. z up, floor at z = 0 (`frame.floor_z`), x/y along the dominant wall axes.
Room polygons are counter-clockwise and not closed (the last point is not the first).
Damage `bbox_on_surface`: on a wall, u is metres along the wall from `p0` and v is metres above
the floor; on a floor or ceiling, u and v are plan x and y. `evidence_image` is a path relative to
the folder that holds `plan.json`.

## Using it

```python
from fp.schema import validate, measure, not_observed
plan = validate("out/c00a170fe1/plan.json")     # dict, JSON string or path; raises on error
plan.surface_ids()
data = plan.dump()                               # model_dump(mode="json", by_alias=True)
```

Damage's `class` is `class_` in Python and `"class"` in JSON. Always dump with `by_alias=True`
(`plan.dump()` does this).

## Regenerating the JSON Schema

```
uv run python -m fp.schema        # writes schema/plan.schema.json
```

`tests/test_schema.py` fails if the file on disk differs from the models, so regenerate and commit
it whenever `fp/schema.py` changes. A change that breaks old plans needs a new `schema_version`.
