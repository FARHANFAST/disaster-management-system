"""Cell <-> local (L1) <-> site (L3) coordinate transforms (spec §1.1),
extended with the 2 m grid margin (spec §5.1).

A floor grid covers the building footprint plus a 2 m margin of `outside`
cells on every side, so the grid origin is offset from the footprint origin
by -margin on both axes. Cell (row, col) center:
    p = ((col + 1/2) h + x0, (row + 1/2) h + y0)
with the canonical rule that row increases with +y (spec §1.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..geometry.building import Building
from ..geometry.primitives import Rect

DEFAULT_MARGIN_M = 2.0


@dataclass(frozen=True, slots=True)
class GridTransform:
    h: float
    origin_local: tuple[float, float]  # (x0, y0) of cell (0,0)'s lower corner
    width_cells: int   # columns
    height_cells: int  # rows

    @classmethod
    def for_footprint(cls, footprint: Rect, h: float, *, margin: float = DEFAULT_MARGIN_M) -> GridTransform:
        x0 = footprint.x0 - margin
        y0 = footprint.y0 - margin
        x1 = footprint.x1 + margin
        y1 = footprint.y1 + margin
        width_cells = round((x1 - x0) / h)
        height_cells = round((y1 - y0) / h)
        return cls(h=h, origin_local=(x0, y0), width_cells=width_cells, height_cells=height_cells)

    @property
    def shape(self) -> tuple[int, int]:
        """(rows, cols), i.e. numpy array shape."""
        return self.height_cells, self.width_cells

    def cell_to_local(self, row: int, col: int) -> tuple[float, float]:
        x0, y0 = self.origin_local
        return (col + 0.5) * self.h + x0, (row + 0.5) * self.h + y0

    def local_to_cell(self, x: float, y: float) -> tuple[int, int]:
        x0, y0 = self.origin_local
        row = math.floor((y - y0) / self.h)
        col = math.floor((x - x0) / self.h)
        return row, col

    def line_to_index(self, coord: float, *, axis: str, prefer_cell_below: bool = False) -> int:
        """The single row (axis='y') or column (axis='x') whose cell interval
        contains the given wall-line coordinate — see dms/grid/raster.py.

        `prefer_cell_below`: a Rect's bounds are half-open [lo, hi) (see
        Rect.contains_point), so `coord` sitting exactly at a space's
        *upper* edge (hi) is, by that convention, already outside the space
        — the default floor() would hand back the cell starting at `coord`,
        which is one cell past the edge. Set this when `coord` is such an
        upper bound, to get the cell ending at `coord` instead.
        """
        x0, y0 = self.origin_local
        origin = y0 if axis == "y" else x0
        idx = math.floor((coord - origin) / self.h)
        if prefer_cell_below and abs((idx * self.h + origin) - coord) < 1e-9:
            idx -= 1
        return idx

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height_cells and 0 <= col < self.width_cells


def cell_to_site(transform: GridTransform, building: Building, row: int, col: int) -> tuple[float, float]:
    x, y = transform.cell_to_local(row, col)
    return building.to_site_xy(x, y)


def site_to_cell(transform: GridTransform, building: Building, X: float, Y: float) -> tuple[int, int]:
    x, y = building.from_site_xy(X, Y)
    return transform.local_to_cell(x, y)
