"""§7.1 watertight test: with all exterior doors and windows closed, a flood
fill from `outside` never reaches an interior cell — at both h_c and h_s.

Also covers the dynamic overlay (§2 principle 3): door/leaf/clutter/obstacle
state changes must mask walkable_now / gas_perm_now / speed_F_now without
ever re-rasterising the static layers.
"""

from __future__ import annotations

import numpy as np
import pytest

from dms.geometry.primitives import Rect
from dms.grid.layers import F_B_DEFAULT, H_COARSE, H_FINE, U_MAX_DEFAULT, DynamicOverlay
from dms.grid.raster import (
    build_grid_stack,
    envelope_leak_mask,
    flood_fill,
    rasterize_floor,
)

H_VALUES = (H_COARSE, H_FINE)


@pytest.mark.parametrize("h", H_VALUES)
def test_gf_envelope_is_watertight(building, gf, h):
    grid = rasterize_floor(building, gf, h)
    leak = envelope_leak_mask(grid)
    breached = leak & (grid.space_id != -1)
    assert not breached.any(), f"{breached.sum()} interior cells reached from outside at h={h}"


@pytest.mark.parametrize("h", H_VALUES)
def test_ff_envelope_is_watertight(building, ff, h):
    grid = rasterize_floor(building, ff, h)
    leak = envelope_leak_mask(grid)
    breached = leak & (grid.space_id != -1)
    assert not breached.any(), f"{breached.sum()} interior cells reached from outside at h={h}"


def test_with_exits_open_outside_does_reach_the_interior(building, gf):
    """Sanity counterpart: the sealed test above must not be vacuously true
    because the margin is disconnected from the building altogether."""
    grid = rasterize_floor(building, gf, H_COARSE)
    passable = ~grid.wall
    reached = flood_fill(passable, grid.outside.copy())
    assert (reached & (grid.space_id != -1)).any()


def test_a_fabricated_gap_in_the_envelope_is_caught():
    """Negative control: punch an undeclared hole in an exterior wall and
    confirm the watertight check actually detects it (not just a check
    that always happens to pass on well-formed data)."""
    from dms.geometry.building import Building, Floor, Space

    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    building = Building(id="LEAKY", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)
    building.floors["GF"] = floor

    grid = rasterize_floor(building, floor, H_COARSE)
    # No doors/windows were declared, so the envelope should be fully sealed.
    assert not (envelope_leak_mask(grid) & (grid.space_id != -1)).any()

    # Now fabricate a gap directly in the wall layer (simulating a geometry bug).
    row = grid.transform.line_to_index(0.0, axis="y")
    grid.wall[row, 5:7] = False
    leak = envelope_leak_mask(grid)
    assert (leak & (grid.space_id != -1)).any(), "a hand-punched gap must be detected"


# --------------------------------------------------------------------- overlay

def test_overlay_never_mutates_static_layers(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    before_walkable = grid.walkable_static.copy()
    before_wall = grid.wall.copy()

    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"}, clutter_on=True)
    grid.walkable_now(overlay)
    grid.gas_perm_now(overlay)
    grid.speed_F_now(overlay)

    assert np.array_equal(grid.walkable_static, before_walkable)
    assert np.array_equal(grid.wall, before_wall)


def test_unavailable_door_removes_exactly_its_own_cells(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    idx = grid.door_ids.index("X-S")
    door_cells = grid.door_id == idx

    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    walkable_now = grid.walkable_now(overlay)

    assert not walkable_now[door_cells].any()
    assert np.array_equal(walkable_now | door_cells, grid.walkable_static | door_cells)


def test_clutter_toggle_blocks_only_when_on(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    assert grid.clutter_mask.any(), "fixture YAML declares clutter slots"

    off = grid.walkable_now(DynamicOverlay(clutter_on=False))
    on = grid.walkable_now(DynamicOverlay(clutter_on=True))

    assert np.array_equal(off, grid.walkable_static)
    assert not on[grid.clutter_mask].any()
    assert on.sum() < off.sum()


def test_extra_obstacle_event_blocks_its_footprint(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    rect = Rect(3.0, 3.0, 5.0, 5.0)  # inside G-01, clear of its door/furniture
    overlay = DynamicOverlay(extra_obstacles=[rect])
    walkable_now = grid.walkable_now(overlay)

    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    rows, cols = grid.shape
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    X, Y = np.meshgrid(xc, yc)
    mask = (X >= rect.x0) & (X < rect.x1) & (Y >= rect.y0) & (Y < rect.y1)

    assert mask.any()
    assert not walkable_now[mask & grid.walkable_static].any()


def test_leaf_state_changes_gas_perm_without_touching_walkability(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    fire_door_id = next(d for d in grid.door_ids if d.startswith("ST-W-fd"))
    idx = grid.door_ids.index(fire_door_id)
    cells = grid.door_id == idx

    closed = grid.gas_perm_now(DynamicOverlay(leaf_states={fire_door_id: "closed"}))
    opened = grid.gas_perm_now(DynamicOverlay(leaf_states={fire_door_id: "open"}))

    assert not np.array_equal(closed[cells], opened[cells])
    assert (opened[cells] > closed[cells]).all()

    walk_closed = grid.walkable_now(DynamicOverlay(leaf_states={fire_door_id: "closed"}))
    walk_open = grid.walkable_now(DynamicOverlay(leaf_states={fire_door_id: "open"}))
    assert np.array_equal(walk_closed, walk_open), "leaf state must not affect agent walkability"


def test_speed_f_now_matches_walkability(gf, building):
    grid = rasterize_floor(building, gf, H_COARSE)
    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    F = grid.speed_F_now(overlay)
    walkable = grid.walkable_now(overlay)

    assert np.allclose(F[walkable], U_MAX_DEFAULT)
    assert np.allclose(F[~walkable], F_B_DEFAULT)


def test_grid_stack_has_both_resolutions(building, gf):
    stack = build_grid_stack(building, gf)
    assert stack.coarse.transform.h == H_COARSE
    assert stack.fine.transform.h == H_FINE
    assert stack.coarse.shape != stack.fine.shape
