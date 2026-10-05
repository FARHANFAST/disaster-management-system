"""Vector geometry -> L0 grid layers (spec §5.1), deterministic and derived
fresh from the same Building/Floor objects the loader produced — no
coordinates live here, only the rasterisation *rule*.

Wall generation: a wall occupies the single row (axis='y' wall-line) or
column (axis='x' wall-line) whose cell interval contains the line's
coordinate — i.e. exactly `GridTransform.line_to_index`, which is the same
floor() used for point-to-cell mapping. That floor() already realises the
spec's "within h/2, tie to -x/-y" rule: a coordinate that lands exactly on a
cell boundary floors down to the cell that *starts* there, so no separate
tie-break is needed. The wall's span along the line is whatever is left
after subtracting every declared door/opening/window span on that same line.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

from ..geometry.building import Building, Floor
from .layers import H_COARSE, H_FINE, FloorGrid, GridStack
from .transforms import DEFAULT_MARGIN_M, GridTransform

TOL = 1e-6


def _subtract_spans(lo: float, hi: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = [(lo, hi)]
    for blo, bhi in sorted(blocked):
        nxt = []
        for flo, fhi in free:
            if bhi <= flo or blo >= fhi:
                nxt.append((flo, fhi))
                continue
            if blo > flo:
                nxt.append((flo, blo))
            if bhi < fhi:
                nxt.append((bhi, fhi))
        free = nxt
    return free


def _crossing_spans_by_line(floor: Floor) -> dict[tuple[str, float], list[tuple[float, float]]]:
    """Every door, opening and window span, keyed by the wall line it sits on
    — these are exactly the places a wall must NOT be generated."""
    out: dict[tuple[str, float], list[tuple[float, float]]] = {}
    for d in floor.doors.values():
        out.setdefault((d.axis, d.coord), []).append((d.lo, d.hi))
    for o in floor.openings.values():
        out.setdefault((o.axis, o.coord), []).append((o.lo, o.hi))
    for w in floor.windows.values():
        out.setdefault((w.axis, w.coord), []).append((w.lo, w.hi))
    return out


def _wall_spans(floor: Floor) -> list[tuple[str, float, float, float]]:
    crossings = _crossing_spans_by_line(floor)
    spans: list[tuple[str, float, float, float]] = []

    # Interior: every pair of edge-adjacent spaces, minus the crossings on that line.
    for a, b in combinations(floor.spaces.values(), 2):
        shared = a.rect.shared_edge(b.rect, tol=TOL)
        if shared is None:
            continue
        axis, coord, lo, hi = shared
        blocked = [s for s in crossings.get((axis, coord), []) if not (s[1] <= lo or s[0] >= hi)]
        for flo, fhi in _subtract_spans(lo, hi, blocked):
            if fhi - flo > TOL:
                spans.append((axis, coord, flo, fhi))

    # Exterior: every space edge on the footprint boundary, minus exits/windows on that line.
    fp = floor.footprint
    for sp in floor.spaces.values():
        for axis, coord in (("x", sp.rect.x0), ("x", sp.rect.x1), ("y", sp.rect.y0), ("y", sp.rect.y1)):
            on_boundary = (
                axis == "x" and (abs(coord - fp.x0) <= TOL or abs(coord - fp.x1) <= TOL)
            ) or (axis == "y" and (abs(coord - fp.y0) <= TOL or abs(coord - fp.y1) <= TOL))
            if not on_boundary:
                continue
            seg = sp.rect.edge_segment(axis, coord, tol=TOL)
            if seg is None:
                continue
            lo, hi = seg
            blocked = [s for s in crossings.get((axis, coord), []) if not (s[1] <= lo or s[0] >= hi)]
            for flo, fhi in _subtract_spans(lo, hi, blocked):
                if fhi - flo > TOL:
                    spans.append((axis, coord, flo, fhi))

    return spans


def _mark_line(
    grid: np.ndarray, transform: GridTransform, axis: str, coord: float, lo: float, hi: float,
    xc: np.ndarray, yc: np.ndarray, value, *, footprint,
) -> None:
    if axis == "y":
        prefer_below = abs(coord - footprint.y1) < TOL
        row = transform.line_to_index(coord, axis="y", prefer_cell_below=prefer_below)
        if not (0 <= row < grid.shape[0]):
            return
        col_mask = (xc >= lo - TOL) & (xc < hi + TOL)
        grid[row, col_mask] = value
    else:
        prefer_below = abs(coord - footprint.x1) < TOL
        col = transform.line_to_index(coord, axis="x", prefer_cell_below=prefer_below)
        if not (0 <= col < grid.shape[1]):
            return
        row_mask = (yc >= lo - TOL) & (yc < hi + TOL)
        grid[row_mask, col] = value


def rasterize_floor(building: Building, floor: Floor, h: float, *, margin: float = DEFAULT_MARGIN_M) -> FloorGrid:
    transform = GridTransform.for_footprint(floor.footprint, h, margin=margin)
    rows, cols = transform.shape
    x0, y0 = transform.origin_local
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    X, Y = np.meshgrid(xc, yc)

    fp = floor.footprint
    outside = ~((X >= fp.x0) & (X < fp.x1) & (Y >= fp.y0) & (Y < fp.y1))

    space_id = np.full((rows, cols), -1, dtype=np.int32)
    void = np.zeros((rows, cols), dtype=bool)
    space_ids = list(floor.spaces.keys())
    for idx, sp in enumerate(floor.spaces.values()):
        r = sp.rect
        mask = (X >= r.x0) & (X < r.x1) & (Y >= r.y0) & (Y < r.y1)
        space_id[mask] = idx
        if sp.kind == "void":
            void[mask] = True

    wall = np.zeros((rows, cols), dtype=bool)
    for axis, coord, lo, hi in _wall_spans(floor):
        _mark_line(wall, transform, axis, coord, lo, hi, xc, yc, True, footprint=fp)

    door_ids = list(floor.doors.keys())
    door_id = np.full((rows, cols), -1, dtype=np.int32)
    exit_id = np.full((rows, cols), -1, dtype=np.int32)
    door_perm_open = np.zeros(len(door_ids), dtype=np.float32)
    door_perm_closed = np.zeros(len(door_ids), dtype=np.float32)
    door_default_leaf_open = np.zeros(len(door_ids), dtype=bool)
    for idx, door in enumerate(floor.doors.values()):
        _mark_line(door_id, transform, door.axis, door.coord, door.lo, door.hi, xc, yc, idx, footprint=fp)
        door_perm_open[idx] = door.gas_perm_open
        door_perm_closed[idx] = door.gas_perm_closed
        door_default_leaf_open[idx] = door.leaf == "open"
        if door.kind == "exit":
            _mark_line(exit_id, transform, door.axis, door.coord, door.lo, door.hi, xc, yc, idx, footprint=fp)

    window_ids = list(floor.windows.keys())
    window_id = np.full((rows, cols), -1, dtype=np.int32)
    window_perm_open = np.zeros(len(window_ids), dtype=np.float32)
    window_perm_closed = np.zeros(len(window_ids), dtype=np.float32)
    window_default_open = np.zeros(len(window_ids), dtype=bool)
    for idx, win in enumerate(floor.windows.values()):
        _mark_line(window_id, transform, win.axis, win.coord, win.lo, win.hi, xc, yc, idx, footprint=fp)
        window_perm_open[idx] = win.perm_open
        window_perm_closed[idx] = win.perm_closed
        window_default_open[idx] = win.is_open

    obstacle_static = np.zeros((rows, cols), dtype=bool)
    clutter_mask = np.zeros((rows, cols), dtype=bool)
    for obs in floor.obstacles.values():
        mask = (X >= obs.rect.x0) & (X < obs.rect.x1) & (Y >= obs.rect.y0) & (Y < obs.rect.y1)
        if obs.layer == "furniture":
            obstacle_static |= mask
        elif obs.layer == "clutter":
            clutter_mask |= mask

    stair_ids = list(building.stairs.keys())
    stair_id = np.full((rows, cols), -1, dtype=np.int32)
    if building.stairs:
        primary_floor_id = building.floor_ids()[0]
        for idx, stair in enumerate(building.stairs.values()):
            owning_space = stair.gf_space if floor.id == primary_floor_id else stair.ff_space
            sidx = space_ids.index(owning_space)
            stair_id[space_id == sidx] = idx

    gas_perm_static = np.ones((rows, cols), dtype=np.float32)
    gas_perm_static[wall] = 0.0
    gas_perm_static[void] = 1.0
    for idx in range(len(door_ids)):
        value = door_perm_open[idx] if door_default_leaf_open[idx] else door_perm_closed[idx]
        gas_perm_static[door_id == idx] = value
    for idx in range(len(window_ids)):
        value = window_perm_open[idx] if window_default_open[idx] else window_perm_closed[idx]
        gas_perm_static[window_id == idx] = value

    walkable_static = (space_id != -1) & (~void) & (~wall) & (~obstacle_static)

    return FloorGrid(
        building_id=building.id,
        floor_id=floor.id,
        transform=transform,
        walkable_static=walkable_static,
        wall=wall,
        space_id=space_id,
        door_id=door_id,
        exit_id=exit_id,
        stair_id=stair_id,
        obstacle_static=obstacle_static,
        clutter_mask=clutter_mask,
        void=void,
        outside=outside,
        window_id=window_id,
        gas_perm_static=gas_perm_static,
        space_ids=space_ids,
        door_ids=door_ids,
        stair_ids=stair_ids,
        window_ids=window_ids,
        door_perm_open=door_perm_open,
        door_perm_closed=door_perm_closed,
        door_default_leaf_open=door_default_leaf_open,
        window_perm_open=window_perm_open,
        window_perm_closed=window_perm_closed,
        window_default_open=window_default_open,
    )


def build_grid_stack(building: Building, floor: Floor, *, h_values: tuple[float, ...] = (H_COARSE, H_FINE)) -> GridStack:
    stack = GridStack(building_id=building.id, floor_id=floor.id)
    for h in h_values:
        stack.grids[h] = rasterize_floor(building, floor, h)
    return stack


def flood_fill(passable: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """4-connected flood fill of `passable` starting from `seeds`, via
    iterative dilation (no scipy dependency)."""
    reached = seeds & passable
    while True:
        grown = reached.copy()
        grown[1:, :] |= reached[:-1, :]
        grown[:-1, :] |= reached[1:, :]
        grown[:, 1:] |= reached[:, :-1]
        grown[:, :-1] |= reached[:, 1:]
        grown &= passable
        if np.array_equal(grown, reached):
            return reached
        reached = grown


def envelope_leak_mask(grid: FloorGrid) -> np.ndarray:
    """Cells reachable from outside with every exterior door and window
    treated as sealed — spec §7.1's watertight test. An empty-of-interior
    result means the envelope is sound; any reached cell with space_id != -1
    is a real breach (a geometry bug, not an intended opening)."""
    blocked = grid.wall.copy()
    blocked |= grid.exit_id != -1
    blocked |= grid.window_id != -1
    passable = ~blocked
    return flood_fill(passable, grid.outside.copy())
