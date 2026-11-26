"""Building-code style checks (spec §3.6) -- IBC-type thresholds,
fully generic (no building-specific ids anywhere -- every check iterates
whatever the loaded Building/occupancy_profiles actually declare).

Travel distance here is a *different* quantity from
dms/fields/eikonal.py's agent-speed-aware routing field: it's the pure
geometric path length through the body-clear domain (unit speed), matching
the independent reference checker's own convention (its `eik()` helper
always solves with a constant V_REF speed, so `T * V_REF` cancels back out
to plain distance in metres) -- not a time-to-exit under U_max/F_b.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import skfmm

from ..fields.eikonal import clearance_domain
from ..geometry.building import Building
from ..grid.layers import H_FINE, DynamicOverlay, FloorGrid
from ..grid.raster import rasterize_floor
from ..scenario.agent_defaults import load_agent_defaults

MM_PER_PERSON_STAIR = 7.6
MM_PER_PERSON_EXIT = 5.1
MIN_CLEAR_DOOR_M = 0.81
TRAVEL_DISTANCE_UNSPRINKLERED_M = 61.0
TRAVEL_DISTANCE_SPRINKLERED_M = 91.0
DEAD_END_UNSPRINKLERED_M = 6.1
DEAD_END_SPRINKLERED_M = 15.2
# Moved to configs/agent_defaults.yaml's
# `redundancy.single_door_max_occupancy` -- one source shared with
# dms/graph/redundancy.py's own generic single-door-room rule, not a
# separately-hardcoded duplicate of the same number.


def _geodesic_distance_m(walk: np.ndarray, targets: np.ndarray, h: float) -> np.ndarray:
    """Pure geometric path length (metres) from every walkable cell to the
    nearest target cell, via skfmm with unit speed."""
    if not targets.any():
        return np.full(walk.shape, np.inf)
    phi = np.ones(walk.shape, dtype=np.float64)
    phi[targets] = -1.0
    phi = np.ma.MaskedArray(phi, mask=~walk)
    speed = np.ones(walk.shape, dtype=np.float64)
    t = skfmm.travel_time(phi, speed, dx=h)
    return np.asarray(np.ma.filled(t, np.inf), dtype=np.float64)


def _exit_cells(floor, grid: FloorGrid) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for idx, did in enumerate(grid.door_ids):
        door = floor.doors[did]
        if door.kind == "exit" and door.default_state == "AVAILABLE":
            mask |= grid.exit_id == idx
    return mask


def _stair_cells(grid: FloorGrid, building: Building, *, enclosed_only: bool) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for idx, sid in enumerate(grid.stair_ids):
        stair = building.stairs[sid]
        if enclosed_only and not stair.enclosed:
            continue
        mask |= grid.stair_id == idx
    return mask


def max_travel_distance_to_exit(
    building: Building, floor_id: str, *, h: float = H_FINE, clearance_radius_m: float | None = None,
) -> float:
    if clearance_radius_m is None:
        clearance_radius_m = load_agent_defaults().body_classes["standard"].clearance_radius_m
    floor = building.get_floor(floor_id)
    grid = rasterize_floor(building, floor, h)
    domain = clearance_domain(grid, DynamicOverlay(), clearance_radius_m)
    targets = _exit_cells(floor, grid)
    domain = domain | targets
    dist = _geodesic_distance_m(domain, targets, h)
    finite_in_domain = np.isfinite(dist) & domain
    return float(np.nanmax(np.where(finite_in_domain, dist, np.nan)))


def max_distance_to_stair(
    building: Building, floor_id: str, *, enclosed_only: bool = False, h: float = H_FINE,
    clearance_radius_m: float | None = None,
) -> float:
    if clearance_radius_m is None:
        clearance_radius_m = load_agent_defaults().body_classes["standard"].clearance_radius_m
    floor = building.get_floor(floor_id)
    grid = rasterize_floor(building, floor, h)
    domain = clearance_domain(grid, DynamicOverlay(), clearance_radius_m)
    targets = _stair_cells(grid, building, enclosed_only=enclosed_only)
    domain = domain | targets
    dist = _geodesic_distance_m(domain, targets, h)
    non_stair = domain & ~targets
    finite_in_domain = np.isfinite(dist) & non_stair
    return float(np.nanmax(np.where(finite_in_domain, dist, np.nan)))


def enclosed_stair_separation_m(building: Building) -> dict[tuple[str, str], float]:
    """Pairwise centre distance between every pair of enclosed stairs."""
    enclosed = {sid: st for sid, st in building.stairs.items() if st.enclosed}
    centres = {}
    for sid, st in enclosed.items():
        fid = next(iter(st.landings))
        centres[sid] = st.footprints[fid].center
    out = {}
    ids = sorted(centres)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            (ax, ay), (bx, by) = centres[a], centres[b]
            out[(a, b)] = float(np.hypot(ax - bx, ay - by))
    return out


def _max_occupant_load(leaf: dict, area_m2: float) -> int:
    """Worst-case headcount for one occupancy leaf -- the max across
    whichever of count/count_range/density_per_m2 it declares (matching
    the independent reference checker's own formula), not a single-type
    dispatch like dms/scenario/population.py's sampler (which draws one
    concrete value; this wants the upper bound for a code check)."""
    load = 0
    if "count" in leaf:
        load = max(load, leaf["count"])
    if "count_range" in leaf:
        load = max(load, leaf["count_range"][1])
    if "density_per_m2" in leaf:
        load = max(load, round(leaf["density_per_m2"] * area_m2))
    return load


@dataclass(slots=True)
class RoomLoad:
    floor_id: str
    space_id: str
    use: str
    area_m2: float
    max_load: int
    n_doors: int


def room_loads(building: Building) -> list[RoomLoad]:
    out = []
    for fid, floor in building.floors.items():
        for sid, space in floor.spaces.items():
            if space.kind != "room" or not space.use:
                continue
            load = 0
            for leaf in building.occupancy_profiles.get(space.use, {}).values():
                load = max(load, _max_occupant_load(leaf, space.rect.area))
            n_doors = sum(1 for d in floor.doors.values() if sid in (d.space_a, d.space_b))
            n_doors += sum(1 for o in floor.openings.values() if sid in (o.space_a, o.space_b))
            out.append(RoomLoad(fid, sid, space.use, space.rect.area, load, n_doors))
    return out


def total_design_population(building: Building) -> int:
    """The whole building's own worst-case headcount,
    the max over occupancy periods of the summed load across every space
    with a `use` on every floor (not just one floor's worst case, like
    `run_code_check`'s own `worst_floor_load`, which exists for stair-
    capacity sizing specifically) -- this is what an assembly point must
    be sized to hold in full (spec: wind may leave any one AP as the only
    safe one, so each is sized for the whole building, not a per-AP
    share)."""
    periods = next(iter(building.occupancy_profiles.values()), {})
    worst = 0
    for period_name in periods:
        total = 0
        for floor in building.floors.values():
            for space in floor.spaces.values():
                if not space.use:
                    continue
                leaf = building.occupancy_profiles.get(space.use, {}).get(period_name)
                if leaf:
                    total += _max_occupant_load(leaf, space.rect.area)
        worst = max(worst, total)
    return worst


def narrowest_door_m(building: Building) -> float:
    widths = [
        d.width_m for floor in building.floors.values() for d in floor.doors.values()
        if d.kind in ("interior", "fire_door")
    ]
    return min(widths) if widths else float("inf")


def dead_end_corridors(building: Building) -> list[tuple[str, str, float]]:
    """(floor_id, space_id, length_m) for every stub-kind space with no
    exit on its own floor."""
    gf_id = building.floor_ids()[0]
    exit_spaces = {d.space_a for d in building.get_floor(gf_id).exit_doors}
    out = []
    for fid, floor in building.floors.items():
        for sid, space in floor.spaces.items():
            if space.kind != "stub":
                continue
            if fid == gf_id and sid in exit_spaces:
                continue
            length = max(space.rect.width, space.rect.height)
            out.append((fid, sid, length))
    return out


def stair_width_capacity_persons(building: Building) -> int:
    total_mm = sum(st.width_m for st in building.stairs.values()) * 1000.0
    return round(total_mm / MM_PER_PERSON_STAIR)


def exit_width_capacity_persons(building: Building) -> int:
    gf = building.get_floor(building.floor_ids()[0])
    total_mm = sum(d.width_m for d in gf.exit_doors if d.default_state == "AVAILABLE") * 1000.0
    return round(total_mm / MM_PER_PERSON_EXIT)


@dataclass(slots=True)
class CodeCheckReport:
    gf_max_travel_to_exit_m: float
    ff_max_travel_to_enclosed_stair_m: float
    stair_separation_m: dict[tuple[str, str], float]
    building_diagonal_m: float
    room_loads: list[RoomLoad]
    narrowest_door_m: float
    dead_end_corridors: list[tuple[str, str, float]]
    stair_capacity_persons: int
    exit_capacity_persons: int
    worst_floor_load: int


def run_code_check(building: Building) -> CodeCheckReport:
    gf_id = building.floor_ids()[0]
    ff_id = building.floor_ids()[-1]
    loads = room_loads(building)
    worst_floor_load = 0
    floor = building.get_floor(ff_id)
    for period_name in next(iter(building.occupancy_profiles.values()), {}):
        total = 0
        for space in floor.spaces.values():
            # Ambient occupancy (e.g. corridors) counts toward egress
            # capacity even though it's excluded from the room-door
            # checks below -- every space with a `use`, not just rooms.
            if not space.use:
                continue
            leaf = building.occupancy_profiles.get(space.use, {}).get(period_name)
            if leaf:
                total += _max_occupant_load(leaf, space.rect.area)
        worst_floor_load = max(worst_floor_load, total)

    x0, y0, x1, y1 = building.get_floor(gf_id).footprint.x0, building.get_floor(gf_id).footprint.y0, \
        building.get_floor(gf_id).footprint.x1, building.get_floor(gf_id).footprint.y1
    diagonal = float(np.hypot(x1 - x0, y1 - y0))

    return CodeCheckReport(
        gf_max_travel_to_exit_m=max_travel_distance_to_exit(building, gf_id),
        ff_max_travel_to_enclosed_stair_m=max_distance_to_stair(building, ff_id, enclosed_only=True),
        stair_separation_m=enclosed_stair_separation_m(building),
        building_diagonal_m=diagonal,
        room_loads=loads,
        narrowest_door_m=narrowest_door_m(building),
        dead_end_corridors=dead_end_corridors(building),
        stair_capacity_persons=stair_width_capacity_persons(building),
        exit_capacity_persons=exit_width_capacity_persons(building),
        worst_floor_load=worst_floor_load,
    )


def write_validation_report(
    report: CodeCheckReport, out_path: str | Path, *, known_exceptions: dict[str, str] | None = None,
    agent_defaults=None,
) -> Path:
    if agent_defaults is None:
        agent_defaults = load_agent_defaults()
    # Single source with dms/graph/redundancy.py's
    # own generic single-door-room rule (configs/agent_defaults.yaml
    # `redundancy:`), not a separately-hardcoded duplicate of the same number.
    max_single_door_load = agent_defaults.redundancy.single_door_max_occupancy

    known_exceptions = known_exceptions or {}
    lines = ["# Building-code validation report", ""]
    lines += [(f"- GF max travel distance to an exit: **{report.gf_max_travel_to_exit_m:.1f} m** "
               f"(limit {TRAVEL_DISTANCE_UNSPRINKLERED_M:.0f} m unsprinklered / {TRAVEL_DISTANCE_SPRINKLERED_M:.0f} m sprinklered)")]
    lines += [f"- FF max distance to nearest enclosed stair: **{report.ff_max_travel_to_enclosed_stair_m:.1f} m** (same limits)"]
    best_sep = max(report.stair_separation_m.values()) if report.stair_separation_m else float("nan")
    lines += [f"- Enclosed stair separation: best {best_sep:.1f} m (limit >= {report.building_diagonal_m / 2:.1f} m = half diagonal)"]
    lines += [f"- Narrowest door: **{report.narrowest_door_m:.2f} m** (limit >= {MIN_CLEAR_DOOR_M} m clear)"]
    lines += [f"- Stair width capacity: {report.stair_capacity_persons} persons vs worst floor load {report.worst_floor_load}"]
    lines += [f"- Exit width capacity: {report.exit_capacity_persons} persons", ""]

    over_49 = [r for r in report.room_loads if r.max_load > max_single_door_load]
    lines += [f"## Rooms with >{max_single_door_load} occupants (max over profiles)", "",
              "| Floor | Space | Use | Area m2 | Max load | Doors |", "|---|---|---|---|---|---|"]
    for r in sorted(over_49, key=lambda r: -r.max_load):
        flag = "" if r.n_doors >= 2 else "  <-- NEEDS A SECOND DOOR"
        lines.append(f"| {r.floor_id} | {r.space_id} | {r.use} | {r.area_m2:.0f} | {r.max_load}{flag} | {r.n_doors} |")
    lines.append("")

    single_door = [r for r in report.room_loads if r.n_doors == 1]
    worst_single = max((r.max_load for r in single_door), default=0)
    lines += [f"- Single-door rooms, max load: **{worst_single}** (limit <= {max_single_door_load})", ""]

    lines += ["## Dead-end corridors", ""]
    for fid, sid, length in report.dead_end_corridors:
        note = known_exceptions.get(sid)
        flag = f"  -- **known exception**: {note}" if note else (
            "  <-- exceeds sprinklered limit" if length > DEAD_END_SPRINKLERED_M else "")
        lines.append(f"- {fid}:{sid}: {length:.1f} m "
                     f"(limit {DEAD_END_UNSPRINKLERED_M} m unsprinklered / {DEAD_END_SPRINKLERED_M} m sprinklered){flag}")
    lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
