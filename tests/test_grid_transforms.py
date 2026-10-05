"""§7.4: transforms round-trip (cell <-> local <-> L3) and follow the
row-up convention (row increases with +y)."""

from __future__ import annotations

import itertools

import pytest

from dms.grid.layers import H_COARSE, H_FINE
from dms.grid.transforms import GridTransform, cell_to_site, site_to_cell

H_VALUES = (H_COARSE, H_FINE)


def test_for_footprint_adds_margin_both_sides(gf):
    t = GridTransform.for_footprint(gf.footprint, H_COARSE, margin=2.0)
    assert t.origin_local == (gf.footprint.x0 - 2.0, gf.footprint.y0 - 2.0)
    assert t.width_cells == round((gf.footprint.width + 4.0) / H_COARSE)
    assert t.height_cells == round((gf.footprint.height + 4.0) / H_COARSE)


@pytest.mark.parametrize("h", H_VALUES)
def test_cell_to_local_to_cell_round_trips(gf, h):
    t = GridTransform.for_footprint(gf.footprint, h)
    rows, cols = t.shape
    sample_rows = range(0, rows, max(1, rows // 23))
    sample_cols = range(0, cols, max(1, cols // 23))
    for row, col in itertools.product(sample_rows, sample_cols):
        x, y = t.cell_to_local(row, col)
        row2, col2 = t.local_to_cell(x, y)
        assert (row2, col2) == (row, col)


def test_row_increases_with_plus_y(gf):
    t = GridTransform.for_footprint(gf.footprint, H_COARSE)
    x0, y0 = t.cell_to_local(10, 5)
    x1, y1 = t.cell_to_local(11, 5)
    assert y1 > y0 and x1 == x0


def test_col_increases_with_plus_x(gf):
    t = GridTransform.for_footprint(gf.footprint, H_COARSE)
    x0, y0 = t.cell_to_local(5, 10)
    x1, y1 = t.cell_to_local(5, 11)
    assert x1 > x0 and y1 == y0


def test_line_to_index_matches_local_to_cell_floor(gf):
    """A coordinate exactly on a wall line must resolve to the same row/col
    whether you ask via line_to_index (wall generation) or local_to_cell
    (point lookup) — otherwise a wall could be drawn one cell off from
    where a door on the same line is punched."""
    t = GridTransform.for_footprint(gf.footprint, H_COARSE)
    for coord in (0.0, 10.0, 13.0, 27.0, 30.0, 40.0):
        row_expected, _ = t.local_to_cell(0.0, coord)
        assert t.line_to_index(coord, axis="y") == row_expected
    for coord in (0.0, 8.0, 11.0, 28.5, 31.5, 49.0, 52.0, 60.0):
        _, col_expected = t.local_to_cell(coord, 0.0)
        assert t.line_to_index(coord, axis="x") == col_expected


@pytest.mark.parametrize("row,col", [(0, 0), (10, 20), (43, 63), (87, 127)])
def test_site_round_trip_through_building_transform(building, gf, row, col):
    t = GridTransform.for_footprint(gf.footprint, H_COARSE)
    if not t.in_bounds(row, col):
        pytest.skip("sample cell out of bounds for this footprint")
    X, Y = cell_to_site(t, building, row, col)
    row2, col2 = site_to_cell(t, building, X, Y)
    assert (row2, col2) == (row, col)


def test_building_local_site_round_trip_with_rotation():
    from dms.geometry.building import Building

    b = Building(id="rot", origin=(10.0, -5.0), rotation_deg=37.0, wall_thickness_m=0.2)
    for x, y in [(0.0, 0.0), (12.3, 4.5), (-3.0, 8.0), (30.0, 30.0)]:
        X, Y = b.to_site_xy(x, y)
        x2, y2 = b.from_site_xy(X, Y)
        assert x2 == pytest.approx(x, abs=1e-9)
        assert y2 == pytest.approx(y, abs=1e-9)
