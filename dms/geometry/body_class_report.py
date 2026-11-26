"""Door/exit x body-class passability table, and
per-room x body-class reachability report. Fully generic -- no
building-specific id appears here; every door/room/class comes from
whatever the loaded Building/AgentDefaults actually declare.

Door passability is computed from the real, continuous door width
(`width_m >= 2*clearance_radius_m + door_margin_m`), never from a grid
cell count -- `tests/test_b1l_body_classes.py`
cross-checks this table against the grid-based `clearance_domain` result
for every door/exit on every floor of every fixture building, using the
same margin-inflated radius the table itself uses (`clearance_radius_m +
door_margin_m/2`) and a resolution finer than the simulation's own H_FINE
(at H_FINE alone, a door whose width is
H_FINE/2 -- a sampling artifact of the verification, not a real geometric
shortfall, confirmed by re-rasterizing at several finer resolutions and
seeing them converge). Fixing the comparison this way also surfaced one
real bug in `configs/buildings/two_storey_large.yaml` (a piece of corridor
furniture placed directly across a doorway, narrowing its effective
clearance far below its declared width) -- fixed in the YAML, not here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..fields.eikonal import EikonalField, FieldCache
from ..grid.layers import H_FINE, DynamicOverlay
from ..grid.raster import rasterize_floor
from ..scenario.agent_defaults import AgentDefaults, load_agent_defaults
from .building import Building


@dataclass(slots=True, frozen=True)
class DoorPassability:
    floor_id: str
    door_id: str
    kind: str  # door.kind, or "opening" for a doorless crossing
    width_m: float
    body_class: str
    required_m: float
    passable: bool


@dataclass(slots=True, frozen=True)
class RoomReachability:
    floor_id: str
    space_id: str
    body_class: str
    reachable: bool  # finite T to an available exit, on this floor's own base-case field


def door_passability_table(
    building: Building, *, agent_defaults: AgentDefaults | None = None,
) -> list[DoorPassability]:
    agent_defaults = agent_defaults if agent_defaults is not None else load_agent_defaults()
    rows: list[DoorPassability] = []
    for floor in building.floors.values():
        crossings = [(d.id, d.kind, d.width_m) for d in floor.doors.values()]
        crossings += [(o.id, "opening", o.width_m) for o in floor.openings.values()]
        for door_id, kind, width_m in crossings:
            for cname, bc in agent_defaults.body_classes.items():
                required = 2.0 * bc.clearance_radius_m + agent_defaults.door_margin_m
                rows.append(DoorPassability(
                    floor_id=floor.id, door_id=door_id, kind=kind, width_m=width_m,
                    body_class=cname, required_m=required, passable=width_m >= required,
                ))
    return rows


def class_reachability_report(
    building: Building, *, agent_defaults: AgentDefaults | None = None, h: float = H_FINE,
) -> list[RoomReachability]:
    """Per (floor, space, body class): does this class's own clearance
    field reach an available exit (or, for a `can_use_stairs=False` class on
    a floor above the bottom, its own refuge) from this
    space at all?

    A `can_use_stairs=True` class gets the full, N-floor-coupled field
    (every floor, bottom to top, via `FieldCache`'s own recursive coupling).
    A `can_use_stairs=False` class gets that same base-case field on the
    bottom floor, and the refuge-targeted field (`FieldCache.get_field(...,
    can_use_stairs=False)`) on every floor above it -- a floor this building
    declares no refuge for at all still reads "unreachable" there, a real,
    reportable gap, not a bug in the field itself."""
    agent_defaults = agent_defaults if agent_defaults is not None else load_agent_defaults()
    floor_ids = building.floor_ids()
    grids = {fid: rasterize_floor(building, building.get_floor(fid), h) for fid in floor_ids}
    overlay = DynamicOverlay()

    rows: list[RoomReachability] = []
    for cname, bc in agent_defaults.body_classes.items():
        # Bare class radius, matching the walker's own actual movement
        # physics (dms/scenario/evacuation.py: sample_agents) -- not the
        # margin-inclusive door_passability_table formula. Verified
        # empirically that unifying them breaks reachability building-wide
        # (every exit becomes a disconnected island at the margin-inclusive
        # radius); see sample_agents' own comment for the full reasoning.
        cache = FieldCache()
        fids_to_check = floor_ids
        fields: dict[str, EikonalField | None] = {}
        for fid in fids_to_check:
            if bc.can_use_stairs or fid == floor_ids[0]:
                fields[fid] = cache.get_field(
                    building, fid, grids, overlay=overlay, clearance_radius_m=bc.clearance_radius_m,
                    can_use_stairs=bc.can_use_stairs,
                )
            else:
                try:
                    fields[fid] = cache.get_field(
                        building, fid, grids, overlay=overlay, clearance_radius_m=bc.clearance_radius_m,
                        can_use_stairs=False,
                    )
                except ValueError:
                    fields[fid] = None  # this floor declares no refuge at all for this class
        for fid in fids_to_check:
            floor = building.get_floor(fid)
            grid = grids[fid]
            eik_field = fields[fid]
            for idx, sid in enumerate(grid.space_ids):
                space = floor.spaces[sid]
                if space.kind in ("void", "stair"):
                    continue
                cells = grid.space_id == idx
                reachable = bool((~eik_field.T.mask[cells]).any()) if eik_field is not None and cells.any() else False
                rows.append(RoomReachability(floor_id=fid, space_id=sid, body_class=cname, reachable=reachable))
    return rows


def write_body_class_report(
    building: Building, out_path: str | Path, *, agent_defaults: AgentDefaults | None = None,
) -> Path:
    agent_defaults = agent_defaults if agent_defaults is not None else load_agent_defaults()
    doors = door_passability_table(building, agent_defaults=agent_defaults)
    rooms = class_reachability_report(building, agent_defaults=agent_defaults)

    lines = ["# Body-class report", "", f"Building: **{building.id}**", "", "## Door/exit passability", "",
             "| Floor | Crossing | Kind | Width (m) | Body class | Required (m) | Passable |",
             "|---|---|---|---|---|---|---|"]
    for d in doors:
        flag = "yes" if d.passable else "NO"
        lines.append(f"| {d.floor_id} | {d.door_id} | {d.kind} | {d.width_m:.2f} | {d.body_class} | {d.required_m:.2f} | {flag} |")

    lines += ["", "## Room reachability by body class", "",
              ("(a `can_use_stairs=false` class is only checked on the bottom floor -- "
               "every upper floor needs a refuge target to be meaningfully checked)"), "",
              "| Floor | Room | Body class | Reaches an exit |", "|---|---|---|---|"]
    for r in rooms:
        flag = "yes" if r.reachable else "NO"
        lines.append(f"| {r.floor_id} | {r.space_id} | {r.body_class} | {flag} |")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
