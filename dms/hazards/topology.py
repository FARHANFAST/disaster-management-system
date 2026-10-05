"""Gas topology export (spec §6.4) — no solver yet, just the data the future
smoke/H2S transport module will need: per-floor gas permeability, vertical
couplings (VOID columns and stair landings, GF<->FF), window couplings
(indoor boundary <-> outdoor L3), and a `ConcentrationGrid` factory for
correctly shaped/originated C_in (per floor, h_s) and C_out (site grids).

`ConcentrationGrid` is defined exactly as SYSTEM_SPEC.md §4.4 gives it —
field names, types and the (slightly unusual) `cell()` method are kept
as-is, not "fixed", per PROMPT_01 §2: spec dataclasses may gain fields but
never be renamed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.building import Building
from ..geometry.site import Site
from ..grid.layers import DynamicOverlay, FloorGrid
from ..grid.transforms import GridTransform


@dataclass(slots=True)
class ConcentrationGrid:
    data: np.ndarray  # [row=y, col=x], row increases with +y
    h: float
    origin_XY: tuple[float, float]
    psi: float = 0.0
    unit: str = "ppm"  # "ppm" | "g/m2" (indoor smoke) | "mg/m3"
    z: float = 1.5
    t: float = 0.0
    species: str = "H2S"
    mw: float = 34.08

    def cell(self, p: np.ndarray) -> tuple[int, int]:
        return int(p[1] // self.h), int(p[0] // self.h)


def make_concentration_grid(
    transform: GridTransform, *, unit: str = "ppm", z: float = 1.5, species: str = "H2S", mw: float = 34.08,
) -> ConcentrationGrid:
    rows, cols = transform.shape
    return ConcentrationGrid(
        data=np.zeros((rows, cols), dtype=np.float64),
        h=transform.h, origin_XY=transform.origin_local, unit=unit, z=z, species=species, mw=mw,
    )


def c_in_factory(floor_grid: FloorGrid, **kwargs) -> ConcentrationGrid:
    """C_in: indoor concentration grid for one floor, at whatever resolution
    `floor_grid` was rasterised at (spec: h_s for the smoke/H2S solver)."""
    return make_concentration_grid(floor_grid.transform, **kwargs)


def c_out_factory(site: Site, *, h: float = 1.0, **kwargs) -> ConcentrationGrid:
    """C_out: outdoor concentration grid for the whole site, configurable h."""
    transform = GridTransform.for_footprint(site.footprint, h, margin=0.0)
    return make_concentration_grid(transform, **kwargs)


@dataclass(slots=True)
class VerticalCoupling:
    gf_cell: tuple[int, int]
    ff_cell: tuple[int, int]
    coefficient: float
    kind: str  # "void" | "stair"
    ref_id: str  # space_id (void) or stair_id (stair)


@dataclass(slots=True)
class WindowCoupling:
    floor_id: str
    window_id: str
    indoor_cell: tuple[int, int]
    outdoor_cell: tuple[int, int]
    permeability: float


def vertical_couplings(
    building: Building, gf_grid: FloorGrid, ff_grid: FloorGrid, *, overlay: DynamicOverlay | None = None,
) -> list[VerticalCoupling]:
    """GF<->FF couplings through every VOID column and every stair landing.
    GF and FF share an identical footprint (validated in Milestone 1), so
    rasterising both at the same h gives identical transforms — the same
    (row, col) index is the same (x, y) position on both floors."""
    if gf_grid.transform.shape != ff_grid.transform.shape or gf_grid.transform.h != ff_grid.transform.h:
        raise ValueError("GF and FF grids must share the same transform for vertical coupling")
    overlay = overlay or DynamicOverlay()

    out: list[VerticalCoupling] = []

    # A space's own rect still "claims" its perimeter wall cells by centre-
    # containment (the wall overlay is a separate pass — see raster.py), so
    # `void`/`stair_id` alone over-include those wall cells. A wall cell
    # isn't a place gas meaningfully couples vertically through, so exclude
    # wall (and, defensively, outside) cells on *either* floor.
    valid = ~gf_grid.wall & ~gf_grid.outside & ~ff_grid.wall & ~ff_grid.outside

    void_rows, void_cols = np.nonzero(ff_grid.void & valid)
    for row, col in zip(void_rows.tolist(), void_cols.tolist()):
        out.append(VerticalCoupling(gf_cell=(row, col), ff_cell=(row, col), coefficient=1.0, kind="void", ref_id="void"))

    for stair in building.stairs.values():
        if stair.enclosed:
            fire_door_id = f"{stair.id}-fd-{ff_grid.floor_id}"
            if fire_door_id in ff_grid.door_ids:
                fd_idx = ff_grid.door_ids.index(fire_door_id)
                leaf = overlay.leaf_states.get(fire_door_id)
                is_open = ff_grid.door_default_leaf_open[fd_idx] if leaf is None else (leaf == "open")
                coeff = float(ff_grid.door_perm_open[fd_idx] if is_open else ff_grid.door_perm_closed[fd_idx])
            else:
                coeff = stair.permeability_enclosed_closed
        else:
            coeff = stair.permeability_open

        sidx = ff_grid.stair_ids.index(stair.id)
        rows, cols = np.nonzero((ff_grid.stair_id == sidx) & valid)
        for row, col in zip(rows.tolist(), cols.tolist()):
            out.append(VerticalCoupling(gf_cell=(row, col), ff_cell=(row, col), coefficient=coeff, kind="stair", ref_id=stair.id))

    return out


def window_couplings(
    building: Building, floor_id: str, floor_grid: FloorGrid, site_transform: GridTransform,
    *, overlay: DynamicOverlay | None = None,
) -> list[WindowCoupling]:
    """Every window cell <-> the outdoor C_out cell it sits in front of."""
    overlay = overlay or DynamicOverlay()
    out: list[WindowCoupling] = []
    x0, y0 = floor_grid.transform.origin_local
    h = floor_grid.transform.h

    for idx, wid in enumerate(floor_grid.window_ids):
        rows, cols = np.nonzero(floor_grid.window_id == idx)
        if rows.size == 0:
            continue
        is_open = overlay.window_open.get(wid, bool(floor_grid.window_default_open[idx]))
        perm = float(floor_grid.window_perm_open[idx] if is_open else floor_grid.window_perm_closed[idx])
        for row, col in zip(rows.tolist(), cols.tolist()):
            local_x = (col + 0.5) * h + x0
            local_y = (row + 0.5) * h + y0
            site_x, site_y = building.to_site_xy(local_x, local_y)
            out_row, out_col = site_transform.local_to_cell(site_x, site_y)
            out.append(WindowCoupling(
                floor_id=floor_id, window_id=wid, indoor_cell=(row, col),
                outdoor_cell=(out_row, out_col), permeability=perm,
            ))
    return out


@dataclass(slots=True)
class GasTopology:
    building_id: str
    gas_perm_now: dict[str, np.ndarray]  # floor_id -> float32 array
    vertical: list[VerticalCoupling]
    windows: list[WindowCoupling]
    c_in: dict[str, ConcentrationGrid] = field(default_factory=dict)
    c_out: ConcentrationGrid | None = None


def export_gas_topology(
    site: Site, building: Building, grids: dict[str, FloorGrid], *,
    overlay_by_floor: dict[str, DynamicOverlay] | None = None, site_h: float = 1.0,
) -> GasTopology:
    """grids: {floor_id: FloorGrid} — both floors' grids, rasterised at the
    SAME h (spec says h_s for the smoke solver; any shared h works here)."""
    overlay_by_floor = overlay_by_floor or {}
    gf_id, ff_id = building.floor_ids()
    gf_grid, ff_grid = grids[gf_id], grids[ff_id]

    gas_perm = {fid: g.gas_perm_now(overlay_by_floor.get(fid, DynamicOverlay())) for fid, g in grids.items()}
    vertical = vertical_couplings(building, gf_grid, ff_grid, overlay=overlay_by_floor.get(ff_id))

    site_transform = GridTransform.for_footprint(site.footprint, site_h, margin=0.0)
    c_out = make_concentration_grid(site_transform)
    windows: list[WindowCoupling] = []
    for fid, grid in grids.items():
        windows += window_couplings(building, fid, grid, site_transform, overlay=overlay_by_floor.get(fid))

    c_in = {fid: c_in_factory(g) for fid, g in grids.items()}

    return GasTopology(building_id=building.id, gas_perm_now=gas_perm, vertical=vertical, windows=windows, c_in=c_in, c_out=c_out)
