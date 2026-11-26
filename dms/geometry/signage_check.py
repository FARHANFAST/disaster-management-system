"""A sign's `points_to` exit must be a real, currently
available exit genuinely reachable from the sign's own location -- checked
with the same A* routing the walker itself would use (`dms/graph/astar.py`),
never assumed from the sign's placement alone.

`validate_sign_attributes` separately checks every sign's own
display/compliance attributes (type, mount height, viewing angle,
illuminated) and that the sign's face actually projects into walkable
space -- a sign whose `facing_deg` points straight into a wall right
behind it can't be read by anyone, regardless of how correct its route is.
"""

from __future__ import annotations

import math

from ..graph.astar import route
from ..graph.builder import NavGraph
from ..grid.layers import H_FINE
from ..grid.raster import rasterize_floor
from .building import Building

_VALID_SIGN_TYPES = frozenset({"exit_door", "directional", "stair_level"})


def _nearest_node_on_floor(graph: NavGraph, floor_id: str, x: float, y: float) -> str | None:
    """Nearest-by-distance node on `floor_id` -- a generic fallback for "the
    node a sign at (x, y) is standing at," since a sign isn't itself a
    room/corridor/door and so has no node of its own in the graph."""
    best, best_d = None, float("inf")
    for nid in graph.node_ids:
        node = graph.node(nid)
        if node.floor != floor_id:
            continue
        nx, ny = node.XY
        d = math.hypot(nx - x, ny - y)
        if d < best_d:
            best, best_d = nid, d
    return best


def _find_exit_door(building: Building, exit_id: str):
    """An exit may be declared on a different floor than the sign pointing
    to it (an upper-floor sign legitimately points toward a ground-floor
    exit, reached via a stair) -- searched across every floor, not just the
    sign's own."""
    for floor in building.floors.values():
        door = floor.doors.get(exit_id)
        if door is not None and door.kind == "exit":
            return door
    return None


def validate_signs(building: Building, graph: NavGraph) -> list[str]:
    """Every problem found, human-readable and floor/sign-id-prefixed --
    empty means every sign checked out. Generic: iterates whatever signs the
    loaded Building actually declares."""
    problems: list[str] = []
    for floor in building.floors.values():
        for sgid, sign in floor.signs.items():
            prefix = f"{floor.id}:{sgid}"
            door = _find_exit_door(building, sign.points_to)
            if door is None:
                problems.append(f"{prefix}: points_to {sign.points_to!r} is not a real exit door in this building")
                continue
            if door.default_state == "UNAVAILABLE":
                problems.append(f"{prefix}: points to exit {door.id}, which is UNAVAILABLE by default")
                continue

            goal_nid = f"X:{door.id}"
            if goal_nid not in graph.g.nodes:
                problems.append(f"{prefix}: exit {door.id} has no node in the navigation graph")
                continue

            src = _nearest_node_on_floor(graph, floor.id, sign.x, sign.y)
            if src is None:
                problems.append(f"{prefix}: no navigation-graph node found on floor {floor.id} to route from")
                continue

            r = route(graph, src, goals=[goal_nid])
            if r is None:
                problems.append(f"{prefix}: no route from the sign's location to exit {door.id} (dead end)")

    return problems


def validate_sign_attributes(building: Building, *, agent_defaults=None) -> list[str]:
    """Every problem found, human-readable and floor/sign-id-prefixed --
    empty means every sign's own display attributes check out. Generic:
    iterates whatever signs the loaded Building actually declares.

    'Faces into the walkable space it serves': a viewer must be able to
    stand somewhere in the `facing_deg` direction from the sign and look
    back at it -- stepping one `signage.trace_step_m` in that direction
    must land on a walkable cell of the sign's own floor, not a wall or
    void."""
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    cfg = agent_defaults.signage

    grids = {fid: rasterize_floor(building, floor, H_FINE) for fid, floor in building.floors.items()}

    problems: list[str] = []
    for floor in building.floors.values():
        grid = grids[floor.id]
        for sgid, sign in floor.signs.items():
            prefix = f"{floor.id}:{sgid}"
            if sign.sign_type not in _VALID_SIGN_TYPES:
                problems.append(f"{prefix}: sign_type {sign.sign_type!r} is not one of {sorted(_VALID_SIGN_TYPES)}")
            if not (0.0 <= sign.facing_deg < 360.0):
                problems.append(f"{prefix}: facing_deg {sign.facing_deg} is out of [0, 360)")
            if not (0.0 < sign.mount_height_m <= 5.0):
                problems.append(f"{prefix}: mount_height_m {sign.mount_height_m} is out of the sane (0, 5] m range")
            if not isinstance(sign.illuminated, bool):
                problems.append(f"{prefix}: illuminated must be a bool, got {sign.illuminated!r}")

            rad = math.radians(sign.facing_deg)
            vx = sign.x + math.cos(rad) * cfg.trace_step_m
            vy = sign.y + math.sin(rad) * cfg.trace_step_m
            row, col = grid.transform.local_to_cell(vx, vy)
            rows, cols = grid.shape
            in_bounds = 0 <= row < rows and 0 <= col < cols
            if not in_bounds or not grid.walkable_static[row, col]:
                problems.append(
                    f"{prefix}: facing_deg {sign.facing_deg} does not project into walkable space "
                    f"(the viewer's standing point at ({vx:.2f}, {vy:.2f}) is a wall/void)",
                )

    return problems
