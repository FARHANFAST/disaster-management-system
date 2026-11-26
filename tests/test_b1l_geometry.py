"""spec §3.1: geometry acceptance checks for the converted B1-L
fixture -- validation, watertightness at both resolutions, exact per-floor
tiled area, the half-open door-cell rule, and no default-on obstacle
splitting a room or cutting a corridor below 1.0 m clear."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dms.geometry.corridor_width import effective_width, lanes_for_segment
from dms.geometry.loader import load_site
from dms.geometry.validate import ValidationReport, check_stair_alignment, validate_site
from dms.grid.layers import H_COARSE, H_FINE
from dms.grid.raster import envelope_leak_mask, flood_fill, rasterize_floor
from dms.scenario.agent_defaults import load_agent_defaults

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"
H_VALUES = (H_COARSE, H_FINE)


@pytest.fixture(scope="module")
def site_b1l():
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="module")
def building_b1l(site_b1l):
    return site_b1l.get_building("B1")


def test_validate_site_passes(site_b1l):
    report = validate_site(site_b1l)
    assert report.ok, report.to_markdown()


@pytest.mark.parametrize("h", H_VALUES)
def test_every_floor_envelope_is_watertight(building_b1l, h):
    for floor in building_b1l.floors.values():
        grid = rasterize_floor(building_b1l, floor, h)
        leak = envelope_leak_mask(grid)
        breached = leak & (grid.space_id != -1)
        assert not breached.any(), f"{floor.id}: {breached.sum()} interior cells reached from outside at h={h}"


def test_spaces_tile_exactly_5184_m2_per_floor(building_b1l):
    for floor in building_b1l.floors.values():
        area = sum(sp.rect.area for sp in floor.spaces.values())
        assert area == pytest.approx(5184.0), f"{floor.id}: {area} m2"
        assert floor.footprint.area == pytest.approx(5184.0)


def test_stair_footprints_align_across_floors(building_b1l):
    report = ValidationReport()
    check_stair_alignment(building_b1l, report)
    assert report.ok, report.to_markdown()


def test_one_meter_door_is_five_cells_at_h_fine(building_b1l):
    """Half-open rule, same convention as space fill: a 1.0 m door at
    h=0.2 occupies exactly 5 cells."""
    gf = building_b1l.get_floor("GF")
    grid = rasterize_floor(building_b1l, gf, H_FINE)
    door = gf.doors["GF:G-02-d1"]
    assert door.width_m == pytest.approx(1.0)
    idx = grid.door_ids.index(door.id)
    n_cells = int((grid.door_id == idx).sum())
    # the door spans one axis fully (5 cells) and the wall's own thickness
    # on the other axis -- check the count along the door's own span axis.
    if door.axis == "x":
        cells_along_span = int(np.any(grid.door_id == idx, axis=1).sum())
    else:
        cells_along_span = int(np.any(grid.door_id == idx, axis=0).sum())
    assert cells_along_span == 5, f"{n_cells} total door cells, {cells_along_span} along the span axis"


def test_no_default_on_obstacle_splits_a_room(building_b1l):
    """Reuses the existing, scipy-free flood_fill: a room's walkable cells
    must form exactly one connected component under the default overlay."""
    from dms.grid.layers import DynamicOverlay

    overlay = DynamicOverlay()
    for floor in building_b1l.floors.values():
        grid = rasterize_floor(building_b1l, floor, H_COARSE)
        walkable = grid.walkable_now(overlay)
        for idx, sid in enumerate(grid.space_ids):
            space = floor.spaces[sid]
            if space.kind != "room":
                continue
            room_mask = (grid.space_id == idx) & walkable
            if not room_mask.any():
                continue
            seed = np.zeros_like(room_mask)
            rows, cols = np.nonzero(room_mask)
            seed[rows[0], cols[0]] = True
            reached = flood_fill(room_mask, seed)
            assert np.array_equal(reached, room_mask), f"{floor.id}:{sid} is split by a default-on obstacle"


def test_no_default_on_obstacle_cuts_a_corridor_below_minimum(building_b1l):
    """Direct re-check of spec §4's validator against B1-L's own
    corridor/obstacle placement (validate_site already asserts this; this
    test pins the specific minimum-lane invariant for every corridor/stub
    segment explicitly)."""
    from dms.geometry.validate import _default_active_obstacles, _iter_corridor_segments

    min_lane_width_m = load_agent_defaults().min_lane_width_m
    for floor, space, axis, short_lo, short_hi, lo, hi in _iter_corridor_segments(building_b1l):
        obstacles = _default_active_obstacles(floor, space.id)
        clear = effective_width(lanes_for_segment(short_lo, short_hi, lo, hi, axis, obstacles), min_lane_width_m)
        assert clear >= min_lane_width_m, f"{floor.id}:{space.id} [{lo:.1f},{hi:.1f}]: clear width {clear:.2f} m"
