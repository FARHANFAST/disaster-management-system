"""Generate refuges by rule, for ANY building --
hand-placed refuges are no longer the source of truth; this generator is.

Rule (stated explicitly, per review): a refuge sits inside the stair's own
protected lobby, or directly adjacent to the enclosed stair's door, within
`refuge_max_dist_m` of it (configs/agent_defaults.yaml `refuges:`) -- placed
to one side of the door's own span (never centred on it, which would sit in
the queue path), sized `width_m` x `depth_m`, and rejected if
`dms/geometry/refuge_check.py: validate_refuges` would flag it (stair
footprint overlap, door queue-zone overlap, corridor clear width -- checked
against every refuge already placed, including earlier stairs' generated
ones on the same floor, so later placements never silently conflict with
earlier ones). Only enclosed stairs get a refuge -- an open/unenclosed
stair offers no fire/smoke protection, so it isn't a realistic refuge
pairing. Every floor but the bottom one that an enclosed stair serves gets
its own refuge, generated independently.

Generation is the default source: a YAML-declared refuge already at the
generated id (`f"{floor_id}-refuge-{stair_id}"` -- the loader runs first)
is an override and is left untouched; generation never raises on a floor/
stair it can't place one for -- that's a real, reportable gap (`refuge is
None` in the returned `RefugePlacement`), not a crash.
"""

from __future__ import annotations

from dataclasses import dataclass

from .building import Building, Floor
from .primitives import Door, Rect, Refuge
from .refuge_check import validate_refuges


@dataclass(slots=True, frozen=True)
class RefugePlacement:
    stair_id: str
    floor_id: str
    refuge: Refuge | None  # None if no valid placement was found
    dist_to_door_m: float | None  # straight-line distance from the refuge's own centre to the door's midpoint
    door_id: str | None


def _door_midpoint(door: Door) -> tuple[float, float]:
    mid = (door.lo + door.hi) / 2.0
    return (mid, door.coord) if door.axis == "y" else (door.coord, mid)


def _candidate_rects(door: Door, other: Rect, *, width_m: float, depth_m: float, margin_m: float) -> list[Rect]:
    """Every geometrically valid placement beside `door`, within `other`'s
    own bounds, offset from the door's span by `margin_m` to either side --
    never centred on the door (that's the queue path), checked by
    `refuge_check.py` regardless. Ordered nearest-to-door first."""
    out: list[Rect] = []
    if door.axis == "x":
        # door on a vertical wall at x=door.coord; `other` sits on one side of it.
        if other.x1 <= door.coord + 1e-6:
            x1 = door.coord
            x0 = x1 - depth_m
        else:
            x0 = door.coord
            x1 = x0 + depth_m
        if x0 < other.x0 or x1 > other.x1:
            return out
        lo_y1 = door.lo - margin_m
        lo_y0 = lo_y1 - width_m
        if lo_y0 >= other.y0:
            out.append(Rect(x0, lo_y0, x1, lo_y1))
        hi_y0 = door.hi + margin_m
        hi_y1 = hi_y0 + width_m
        if hi_y1 <= other.y1:
            out.append(Rect(x0, hi_y0, x1, hi_y1))
    else:
        # door on a horizontal wall at y=door.coord.
        if other.y1 <= door.coord + 1e-6:
            y1 = door.coord
            y0 = y1 - depth_m
        else:
            y0 = door.coord
            y1 = y0 + depth_m
        if y0 < other.y0 or y1 > other.y1:
            return out
        lo_x1 = door.lo - margin_m
        lo_x0 = lo_x1 - width_m
        if lo_x0 >= other.x0:
            out.append(Rect(lo_x0, y0, lo_x1, y1))
        hi_x0 = door.hi + margin_m
        hi_x1 = hi_x0 + width_m
        if hi_x1 <= other.x1:
            out.append(Rect(hi_x0, y0, hi_x1, y1))
    dmx, dmy = _door_midpoint(door)
    out.sort(key=lambda r: ((r.center[0] - dmx) ** 2 + (r.center[1] - dmy) ** 2))
    return out


def _place_for_stair(building: Building, floor: Floor, stair_id: str, cfg, min_lane_width_m: float) -> RefugePlacement:
    """Tries every candidate placement, nearest-to-door first, and commits
    the first one that passes `validate_refuges` directly into
    `floor.refuges` -- so a later stair's own placement search already sees
    it. Returns a `refuge=None` result, with nothing committed, if no
    candidate passes."""
    stair = building.stairs[stair_id]
    landing_space = stair.landings[floor.id]
    candidate_doors = [
        d for d in floor.doors.values()
        if d.kind != "exit" and landing_space in (d.space_a, d.space_b)
    ]
    refuge_id = f"{floor.id}-refuge-{stair_id}"
    landing_rect = floor.spaces[landing_space].rect
    for door in sorted(candidate_doors, key=lambda d: d.id):
        other_id = door.space_b if door.space_a == landing_space else door.space_a
        other = floor.spaces.get(other_id)
        # Try the corridor/room side first (the common case), then the
        # stair's own landing side -- "inside the stair's protected lobby"
        # (per the rule) is the fallback for a door whose adjacent corridor
        # is too narrow to hold a refuge beside it at all (e.g. a corridor
        # only as deep as the door's own span, with no margin either side).
        candidate_spaces = []
        if other is not None and other.kind in ("corridor", "stub", "room"):
            candidate_spaces.append(other.rect)
        candidate_spaces.append(landing_rect)

        rects: list[Rect] = []
        for space_rect in candidate_spaces:
            rects += _candidate_rects(
                door, space_rect, width_m=cfg.width_m, depth_m=cfg.depth_m, margin_m=cfg.min_clear_margin_m,
            )
        dmx, dmy = _door_midpoint(door)
        rects.sort(key=lambda r: ((r.center[0] - dmx) ** 2 + (r.center[1] - dmy) ** 2))
        for rect in rects:
            cx, cy = rect.center
            dist = ((cx - dmx) ** 2 + (cy - dmy) ** 2) ** 0.5
            if dist > cfg.refuge_max_dist_m:
                continue
            refuge = Refuge(id=refuge_id, floor=floor.id, rect=rect, capacity=cfg.capacity, serves_stair=stair_id)
            floor.refuges[refuge_id] = refuge
            problems = [p for p in validate_refuges(building, min_lane_width_m=min_lane_width_m) if refuge_id in p]
            if problems:
                del floor.refuges[refuge_id]
                continue
            return RefugePlacement(stair_id=stair_id, floor_id=floor.id, refuge=refuge,
                                    dist_to_door_m=dist, door_id=door.id)
    return RefugePlacement(stair_id=stair_id, floor_id=floor.id, refuge=None, dist_to_door_m=None, door_id=None)


def apply_generated_refuges(building: Building, agent_defaults=None) -> list[RefugePlacement]:
    """Generates and commits a refuge for every (enclosed stair, floor it
    serves other than the bottom one) pair, in deterministic (stair id,
    floor id) order so results never depend on dict iteration order. A
    YAML-declared refuge already at the generated id is left untouched
    (that's the override); any other YAML-declared refuge is simply kept."""
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    cfg = agent_defaults.refuges
    bottom_id = building.floor_ids()[0]

    placements: list[RefugePlacement] = []
    for stair_id in sorted(building.stairs):
        stair = building.stairs[stair_id]
        if not stair.enclosed:
            continue
        for floor_id in sorted(stair.landings):
            if floor_id == bottom_id:
                continue
            floor = building.get_floor(floor_id)
            refuge_id = f"{floor_id}-refuge-{stair_id}"
            if refuge_id in floor.refuges:
                existing = floor.refuges[refuge_id]
                dmx, dmy = None, None
                door_id = None
                for d in floor.doors.values():
                    if d.kind != "exit" and stair.landings[floor_id] in (d.space_a, d.space_b):
                        dmx, dmy = _door_midpoint(d)
                        door_id = d.id
                        break
                cx, cy = existing.rect.center
                dist = ((cx - dmx) ** 2 + (cy - dmy) ** 2) ** 0.5 if dmx is not None else None
                placements.append(RefugePlacement(stair_id=stair_id, floor_id=floor_id, refuge=existing,
                                                    dist_to_door_m=dist, door_id=door_id))
                continue  # YAML already declared this exact id -- that's the override
            placements.append(_place_for_stair(building, floor, stair_id, cfg, agent_defaults.min_lane_width_m))
    return placements
