"""Geometric checks for `Building.refuges` -- a refuge
must never sit in a stair's own footprint or door queue zone, and must
never reduce the corridor it sits in below `min_lane_width_m` clear.
Checked by geometry (rect overlap / the same lane-width maths the clear-
width report already uses), never by eye.
"""

from __future__ import annotations

from .building import Building
from .corridor_width import effective_width, lanes_for_segment
from .primitives import Door, Obstacle, Rect


def _door_queue_zone(door: Door, depth: float) -> Rect:
    """A conservative buffer around a door's own span, extending `depth` to
    both sides of the wall it's on -- the area an agent occupies while
    approaching, opening and passing through it."""
    if door.axis == "x":
        return Rect(door.coord - depth, door.lo, door.coord + depth, door.hi)
    return Rect(door.lo, door.coord - depth, door.hi, door.coord + depth)


def _corridor_clear_width_ok(floor, refuge_rect: Rect, min_lane_width_m: float) -> list[str]:
    problems = []
    for sid, sp in floor.spaces.items():
        if sp.kind not in ("corridor", "stub") or not sp.rect.overlaps(refuge_rect):
            continue
        r = sp.rect
        long_axis = "x" if r.width >= r.height else "y"
        # The window checked is the refuge's own extent along the corridor's
        # long axis, not the whole corridor: lanes_for_segment evaluates one
        # segment's worst case across its *entire* long-axis span, so handing
        # it the whole corridor would combine the refuge with furniture many
        # metres away in the long direction as if they sat side by side.
        long_lo, long_hi = (refuge_rect.x0, refuge_rect.x1) if long_axis == "x" else (refuge_rect.y0, refuge_rect.y1)
        # Include the corridor's own existing furniture in that same window
        # too -- a refuge that would be fine on its own but combines with
        # nearby existing furniture to pinch the corridor is still a real
        # problem, not a false negative.
        existing = [
            o for o in floor.obstacles.values()
            if o.active_by_default and o.layer == "furniture" and o.rect.overlaps(r)
        ]
        pseudo = [Obstacle(id="__refuge_check__", floor=floor.id, rect=refuge_rect), *existing]
        if long_axis == "x":
            lanes = lanes_for_segment(r.y0, r.y1, long_lo, long_hi, "x", pseudo)
        else:
            lanes = lanes_for_segment(r.x0, r.x1, long_lo, long_hi, "y", pseudo)
        width = effective_width(lanes, min_lane_width_m)
        if width < min_lane_width_m - 1e-9:
            problems.append(
                f"leaves only {width:.2f} m clear in corridor {sid} (< {min_lane_width_m:.2f} m min lane width)"
            )
    return problems


def validate_refuges(building: Building, *, min_lane_width_m: float) -> list[str]:
    """Every problem found, each a human-readable, floor/refuge-id-prefixed
    message -- empty means every refuge passed. Generic over any building:
    iterates whatever refuges/stairs/doors the loaded Building actually
    declares, no ids hardcoded here.

    Checked against EVERY stair that serves this floor, not just the one a
    refuge declares `serves_stair` -- a refuge that only ever gets checked
    against its own declared stair can sit right next to a completely
    different stair's footprint or door with nothing to catch it (a real
    bug, found this way: a refuge declared `serves_stair: ST-N` sitting
    beside the unrelated open stair ST-C2 passed a version of this check
    that only ever looked at ST-N). The one exception: a refuge is allowed
    to overlap its OWN served stair's footprint (never a different stair's)
    -- "inside the stair's protected lobby" is an explicitly valid
    placement, per the rule this checks against; its door's queue zone is
    still checked regardless, own stair included."""
    problems: list[str] = []
    for floor in building.floors.values():
        for rid, refuge in floor.refuges.items():
            prefix = f"{floor.id}:{rid}"
            for sid, stair in building.stairs.items():
                if floor.id not in stair.landings:
                    continue
                landing_space = stair.landings[floor.id]
                stair_rect = floor.spaces[landing_space].rect
                if sid != refuge.serves_stair and refuge.rect.overlaps(stair_rect):
                    problems.append(f"{prefix} overlaps stair {sid}'s own footprint")
                for did, door in floor.doors.items():
                    if door.space_a != landing_space and door.space_b != landing_space:
                        continue
                    qz = _door_queue_zone(door, depth=door.width_m)
                    if refuge.rect.overlaps(qz):
                        problems.append(f"{prefix} overlaps stair {sid}'s door {did} queue zone")

            for msg in _corridor_clear_width_ok(floor, refuge.rect, min_lane_width_m):
                problems.append(f"{prefix} {msg}")

    return problems
