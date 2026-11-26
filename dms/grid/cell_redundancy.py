"""Cell-level single/double/triple-failure redundancy sweep (spec
§3.3), operating directly on rasterized grid connectivity rather than the
L2 nav-graph. This is a separate module from dms/graph/redundancy.py (which
stays exactly as-is for two_storey/three_storey): a cell-level sweep can
express "cut a corridor every ~6 m", which has no graph-node equivalent,
and the acceptance numbers it must match (83/8/78/6/5/15/10/10) come from
an independent reference tool that works the same way (scipy.ndimage
connected components + a union-find across stairs/exits) -- this is this
project's own, fully generic implementation: no building-specific ids
anywhere, every category is enumerated from whatever the loaded Building
actually has.

Deliberately excludes the spec §2 auto-generated `open_between_kinds`
openings from the failure-case enumeration: there is no discrete door to
"fail" there, since it's a permanently-open structural continuity, not a
closable element -- matching the independent reference tool's own
exclusion (its door list comes only from the source YAML's explicit
doors:/openings:, never from the wall-derivation pass that makes
open_between_kinds boundaries non-walls in the first place). Recognized
here by the deterministic id shape the loader gives these openings
(`f"{floor_id}:open:{a}|{b}:{n}"`, see dms/geometry/loader.py).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from ..geometry.building import Building, Floor
from ..graph.builder import _long_axis
from .layers import H_COARSE, DynamicOverlay, FloorGrid
from .raster import rasterize_floor

STRUCT4 = ndimage.generate_binary_structure(2, 1)
AUTO_OPEN_MARKER = ":open:"
CORRIDOR_CUT_STEP_M = 6.0


def _line_cell_mask(grid: FloorGrid, axis: str, coord: float, lo: float, hi: float, footprint, *, tol: float = 1e-6) -> np.ndarray:
    """Boolean mask of the cells a wall-line crossing (door/opening/exit)
    occupies -- validated to agree exactly with grid.door_id for every
    actual door in this fixture; needed directly for openings, which (by
    design) have no door_id of their own since they carry no state."""
    transform = grid.transform
    h = transform.h
    x0, y0 = transform.origin_local
    rows, cols = grid.shape
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    mask = np.zeros(grid.shape, dtype=bool)
    if axis == "y":
        prefer_below = abs(coord - footprint.y1) < tol
        row = transform.line_to_index(coord, axis="y", prefer_cell_below=prefer_below)
        if not (0 <= row < rows):
            return mask
        col_mask = (xc >= lo - tol) & (xc < hi + tol)
        mask[row, col_mask] = True
    else:
        prefer_below = abs(coord - footprint.x1) < tol
        col = transform.line_to_index(coord, axis="x", prefer_cell_below=prefer_below)
        if not (0 <= col < cols):
            return mask
        row_mask = (yc >= lo - tol) & (yc < hi + tol)
        mask[row_mask, col] = True
    return mask


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict = {}

    def find(self, a):
        self._parent.setdefault(a, a)
        while self._parent[a] != a:
            self._parent[a] = self._parent[self._parent[a]]
            a = self._parent[a]
        return a

    def union(self, a, b) -> None:
        self._parent[self.find(a)] = self.find(b)


@dataclass(slots=True)
class CellFailureCase:
    label: str
    kind: str  # door | stair_opening | corridor | exit | stair
    cells_removed: dict[str, np.ndarray] = field(default_factory=dict)  # floor_id -> bool mask
    exits_removed: frozenset[str] = frozenset()
    stairs_removed: frozenset[str] = frozenset()


@dataclass(slots=True)
class CellFailureResult:
    case: CellFailureCase
    affected: frozenset[tuple[str, str]]  # (floor_id, space_id) room-kind spaces that lose egress


def _is_auto_opening(opening_id: str) -> bool:
    return AUTO_OPEN_MARKER in opening_id


def _base_walkable(building: Building, grids: dict[str, FloorGrid]) -> dict[str, np.ndarray]:
    overlay = DynamicOverlay()
    return {fid: grids[fid].walkable_now(overlay) for fid in building.floor_ids()}


def _egress_ok(
    building: Building, grids: dict[str, FloorGrid], walk: dict[str, np.ndarray],
    exits_open: set[str], stairs_open: set[str],
) -> set[tuple[str, str]]:
    """(floor, space) room-kind spaces whose main walkable component cannot
    reach any open exit, given per-floor walkable masks `walk` (already
    reflecting whatever cells this trial removed) and the stairs/exits
    considered open for this trial."""
    labels: dict[str, np.ndarray] = {}
    for fid, w in walk.items():
        lab, _ = ndimage.label(w, structure=STRUCT4)
        labels[fid] = lab

    uf = _UnionFind()
    OUT = ("__OUT__", 0)

    for stair_id in stairs_open:
        stair = building.stairs[stair_id]
        nodes = []
        for fid in stair.landings:
            if fid not in grids:
                continue
            grid = grids[fid]
            if stair_id not in grid.stair_ids:
                continue
            sidx = grid.stair_ids.index(stair_id)
            vals = np.unique(labels[fid][(grid.stair_id == sidx) & (labels[fid] > 0)])
            nodes += [(fid, int(v)) for v in vals]
        for a, b in itertools.pairwise(nodes):
            uf.union(a, b)

    for fid, floor in building.floors.items():
        grid = grids[fid]
        lab = labels[fid]
        for idx, door_id in enumerate(grid.door_ids):
            door = floor.doors.get(door_id)
            if door is None or door.kind != "exit" or door_id not in exits_open:
                continue
            vals = np.unique(lab[(grid.door_id == idx) & (lab > 0)])
            for v in vals:
                uf.union((fid, int(v)), OUT)

    bad: set[tuple[str, str]] = set()
    for fid, floor in building.floors.items():
        grid = grids[fid]
        lab = labels[fid]
        for sidx, sid in enumerate(grid.space_ids):
            space = floor.spaces[sid]
            if space.kind != "room":
                continue
            cells = lab[(grid.space_id == sidx) & (lab > 0)]
            if cells.size == 0:
                bad.add((fid, sid))
                continue
            main = int(np.bincount(cells).argmax())
            if uf.find((fid, main)) != uf.find(OUT):
                bad.add((fid, sid))
    return bad


def _corridor_cuts_for_floor(floor: Floor, grid: FloorGrid, *, step: float = CORRIDOR_CUT_STEP_M) -> list[CellFailureCase]:
    cases: list[CellFailureCase] = []
    transform = grid.transform
    x0, y0 = transform.origin_local
    h = transform.h
    for sidx, sid in enumerate(grid.space_ids):
        space = floor.spaces[sid]
        if space.kind != "corridor":
            continue
        axis = _long_axis(space)
        r = space.rect
        horiz = axis == "x"
        length = r.width if horiz else r.height
        n = max(1, int(length // step))
        lo_bound = r.x0 if horiz else r.y0
        for j in range(n):
            pos = lo_bound + (j + 0.5) * length / n
            m = np.zeros(grid.shape, dtype=bool)
            if horiz:
                c = int((pos - x0) // h)
                if 0 <= c < grid.shape[1]:
                    m[:, c] = True
            else:
                row = int((pos - y0) // h)
                if 0 <= row < grid.shape[0]:
                    m[row, :] = True
            m &= grid.space_id == sidx
            cases.append(CellFailureCase(
                label=f"{floor.id}:{sid}@{pos:.1f}", kind="corridor", cells_removed={floor.id: m},
            ))
    return cases


def enumerate_single_failures(building: Building, grids: dict[str, FloorGrid]) -> list[CellFailureCase]:
    cases: list[CellFailureCase] = []
    stair_space_ids: set[str] = {
        sp.id for floor in building.floors.values() for sp in floor.spaces.values() if sp.kind == "stair"
    }

    for fid, floor in building.floors.items():
        grid = grids[fid]
        fp = floor.footprint
        for idx, door_id in enumerate(grid.door_ids):
            door = floor.doors[door_id]
            if door.kind == "exit":
                continue  # exits are their own category below
            mask = grid.door_id == idx
            kind = "fire_door" if door.kind == "fire_door" else "door"
            cases.append(CellFailureCase(label=door_id, kind=kind, cells_removed={fid: mask}))
        for opening_id, opening in floor.openings.items():
            if _is_auto_opening(opening_id):
                continue
            mask = _line_cell_mask(grid, opening.axis, opening.coord, opening.lo, opening.hi, fp)
            kind = "stair_opening" if (opening.space_a in stair_space_ids or opening.space_b in stair_space_ids) else "door"
            cases.append(CellFailureCase(label=opening_id, kind=kind, cells_removed={fid: mask}))
        cases.extend(_corridor_cuts_for_floor(floor, grid))

    gf_id = building.floor_ids()[0]
    for door in building.get_floor(gf_id).exit_doors:
        if door.default_state != "AVAILABLE":
            continue
        cases.append(CellFailureCase(label=f"exit {door.id}", kind="exit", exits_removed=frozenset({door.id})))

    for stair_id, stair in building.stairs.items():
        cells_removed = {}
        for fid in stair.landings:
            grid = grids[fid]
            if stair_id in grid.stair_ids:
                sidx = grid.stair_ids.index(stair_id)
                cells_removed[fid] = grid.stair_id == sidx
        cases.append(CellFailureCase(
            label=f"stair {stair_id}", kind="stair", cells_removed=cells_removed, stairs_removed=frozenset({stair_id}),
        ))
    return cases


def _all_open_exits(building: Building) -> set[str]:
    gf_id = building.floor_ids()[0]
    return {d.id for d in building.get_floor(gf_id).exit_doors if d.default_state == "AVAILABLE"}


def _run_case(
    building: Building, grids: dict[str, FloorGrid], base_walk: dict[str, np.ndarray],
    case: CellFailureCase, *, all_exits: set[str], all_stairs: set[str], base_bad: set[tuple[str, str]],
) -> CellFailureResult:
    walk = {fid: w.copy() for fid, w in base_walk.items()}
    for fid, mask in case.cells_removed.items():
        walk[fid] = walk[fid] & ~mask
    exits_open = all_exits - case.exits_removed
    stairs_open = all_stairs - case.stairs_removed

    removed_rooms: set[tuple[str, str]] = set()
    for fid, floor in building.floors.items():
        grid = grids[fid]
        for sidx, sid in enumerate(grid.space_ids):
            if floor.spaces[sid].kind != "room":
                continue
            if not (walk[fid] & (grid.space_id == sidx)).any():
                removed_rooms.add((fid, sid))

    bad = _egress_ok(building, grids, walk, exits_open, stairs_open)
    affected = frozenset(bad - base_bad - removed_rooms)
    return CellFailureResult(case=case, affected=affected)


@dataclass(slots=True)
class CellRedundancySweep:
    base_bad: frozenset[tuple[str, str]]
    single: list[CellFailureResult]
    double_exit: list[CellFailureResult]
    double_stair: list[CellFailureResult]
    triple_stair: list[CellFailureResult]


def run_cell_redundancy_sweep(building: Building, *, h: float = H_COARSE) -> CellRedundancySweep:
    grids = {fid: rasterize_floor(building, building.get_floor(fid), h) for fid in building.floor_ids()}
    base_walk = _base_walkable(building, grids)
    all_exits = _all_open_exits(building)
    all_stairs = set(building.stairs)
    base_bad = _egress_ok(building, grids, base_walk, all_exits, all_stairs)

    def run(case: CellFailureCase) -> CellFailureResult:
        return _run_case(building, grids, base_walk, case, all_exits=all_exits, all_stairs=all_stairs, base_bad=base_bad)

    single = [run(c) for c in enumerate_single_failures(building, grids)]

    double_exit = []
    for a, b in itertools.combinations(sorted(all_exits), 2):
        case = CellFailureCase(label=f"exits {a}+{b}", kind="double_exit", exits_removed=frozenset({a, b}))
        double_exit.append(run(case))

    stair_ids = sorted(all_stairs)
    double_stair = []
    for a, b in itertools.combinations(stair_ids, 2):
        cells = {}
        for sid in (a, b):
            stair = building.stairs[sid]
            for fid in stair.landings:
                grid = grids[fid]
                if sid in grid.stair_ids:
                    m = grid.stair_id == grid.stair_ids.index(sid)
                    cells[fid] = cells.get(fid, np.zeros_like(m)) | m
        case = CellFailureCase(label=f"stairs {a}+{b}", kind="double_stair", cells_removed=cells, stairs_removed=frozenset({a, b}))
        double_stair.append(run(case))

    triple_stair = []
    for trio in itertools.combinations(stair_ids, 3):
        cells = {}
        for sid in trio:
            stair = building.stairs[sid]
            for fid in stair.landings:
                grid = grids[fid]
                if sid in grid.stair_ids:
                    m = grid.stair_id == grid.stair_ids.index(sid)
                    cells[fid] = cells.get(fid, np.zeros_like(m)) | m
        case = CellFailureCase(label="stairs " + "+".join(trio), kind="triple_stair", cells_removed=cells, stairs_removed=frozenset(trio))
        triple_stair.append(run(case))

    return CellRedundancySweep(
        base_bad=frozenset(base_bad), single=single,
        double_exit=double_exit, double_stair=double_stair, triple_stair=triple_stair,
    )
