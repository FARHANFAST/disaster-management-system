"""Milestone 4: Eikonal travel-time fields (N1-N2), multi-floor stair
coupling, caching, and the direction field."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dms.fields.eikonal import (
    FieldCache,
    _one_sided_gradient,
    compute_gf_field,
    direction_field,
    domain_mask,
    resolve_exit_set,
    stairs_blocked_by_scenario,
)
from dms.geometry.primitives import Rect
from dms.grid.layers import DynamicOverlay

# --------------------------------------------------------------------- GF field


def test_t_is_finite_everywhere_in_domain_all_exits(building, gf, gf_grid):
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    assert np.isfinite(field.T.data[field.domain]).all()


def test_t_is_zero_exactly_at_exit_cells(building, gf, gf_grid):
    """exit_set="all" means every currently-AVAILABLE exit — X-SE defaults
    to UNAVAILABLE (it's a locked service door), so it's correctly excluded
    from the target set and should NOT read T=0."""
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    available_exit_ids = {d.id for d in gf.exit_doors if d.default_state != "UNAVAILABLE"}
    for idx, (did, door) in enumerate(gf.doors.items()):
        if door.kind != "exit":
            continue
        cells = gf_grid.door_id == idx
        if did in available_exit_ids:
            assert (field.T.data[cells] == 0.0).all()
        else:
            assert (field.T.data[cells] > 0.0).all()


def test_t_is_larger_in_a_dead_end_than_at_its_door(building, gf, gf_grid):
    """G-08 (toilets) is a declared dead-end off a single door."""
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    g08_idx = gf_grid.space_ids.index("G-08")
    interior = (gf_grid.space_id == g08_idx) & gf_grid.walkable_static
    door_idx = list(gf.doors.keys()).index("G-08-d1")
    door_cells = gf_grid.door_id == door_idx

    assert field.T.data[interior].max() > field.T.data[door_cells].max()
    assert field.T.data[interior].min() >= field.T.data[door_cells].min() - 1e-6


def test_gradient_descent_from_a_room_reaches_t_zero_monotonically(building, gf, gf_grid):
    """Following -grad(T) (N1) from a free cell must walk steadily downhill
    to an exit — the direct behavioural proof that T encodes a real
    shortest-time field, not just a plausible-looking heatmap."""
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    ex, ey = direction_field(field.T, field.domain, gf_grid.transform.h)

    g16_idx = gf_grid.space_ids.index("G-16")
    rows, cols = np.nonzero((gf_grid.space_id == g16_idx) & gf_grid.walkable_static)
    row, col = int(rows[0]), int(cols[0])

    prev_t = field.T.data[row, col]
    for _ in range(3000):
        if prev_t <= 1e-6:
            break
        dx, dy = ex[row, col], ey[row, col]
        assert not math.isnan(dx)
        # Single-axis steps only (dominant component of -grad T) — a
        # simultaneous diagonal step can cut through a wall's corner even
        # when both orthogonal neighbours individually look fine.
        if abs(dx) >= abs(dy):
            candidates = [(row, col + (1 if dx > 0 else -1)), (row + (1 if dy > 0 else -1), col)]
        else:
            candidates = [(row + (1 if dy > 0 else -1), col), (row, col + (1 if dx > 0 else -1))]

        moved = False
        for nrow, ncol in candidates:
            nrow = min(max(nrow, 0), field.T.shape[0] - 1)
            ncol = min(max(ncol, 0), field.T.shape[1] - 1)
            if (nrow, ncol) == (row, col) or not field.domain[nrow, ncol]:
                continue
            nt = field.T.data[nrow, ncol]
            assert nt <= prev_t + 1e-3, "T must not increase when walking -grad(T)"
            row, col, prev_t = nrow, ncol, nt
            moved = True
            break
        if not moved:
            break
    else:
        pytest.fail("gradient descent did not reach T~0 within the step budget")
    assert prev_t <= 1.0


def test_gradient_magnitude_matches_inverse_speed_on_free_cells(building, gf, gf_grid):
    """|grad T| ~= 1/F (N1) away from walls/doors/obstacles, where the
    one-sided-difference truncation error is largest."""
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    h = gf_grid.transform.h
    gx, gy = _one_sided_gradient(np.asarray(field.T.filled(0.0)), field.domain, h)
    mag = np.hypot(gx, gy)

    free = gf_grid.walkable_static & (gf_grid.door_id == -1) & (gf_grid.exit_id == -1) & ~gf_grid.obstacle_static
    for _ in range(2):  # erode 2 cells from any non-free neighbour
        shifted = free.copy()
        shifted[1:, :] &= free[:-1, :]
        shifted[:-1, :] &= free[1:, :]
        shifted[:, 1:] &= free[:, :-1]
        shifted[:, :-1] &= free[:, 1:]
        free = shifted

    ratio = mag[free] * field.speed[free]
    assert np.mean((ratio > 0.9) & (ratio < 1.1)) > 0.95
    assert ratio.mean() == pytest.approx(1.0, abs=0.1)


def test_unavailable_exit_is_excluded_from_targets(building, gf, gf_grid):
    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    field_blocked = compute_gf_field(building, gf, gf_grid, overlay=overlay)
    field_open = compute_gf_field(building, gf, gf_grid)
    x_s_idx = list(gf.doors.keys()).index("X-S")
    x_s_cells = gf_grid.door_id == x_s_idx
    assert (field_blocked.T.data[x_s_cells] > 0).all()
    assert (field_open.T.data[x_s_cells] == 0).all()


def test_exit_set_resolution():
    class _FakeBuilding:
        def __init__(self):
            self.exit_sets = {"west": ["X-W", "X-NW"]}

    assert resolve_exit_set(_FakeBuilding(), "all") is None
    assert resolve_exit_set(_FakeBuilding(), "west") == ["X-W", "X-NW"]
    assert resolve_exit_set(_FakeBuilding(), "X-S") == ["X-S"]  # single-exit fallback


def test_named_exit_set_targets_only_its_own_exits(building, gf, gf_grid):
    field = compute_gf_field(building, gf, gf_grid, exit_set="south")
    x_s_idx = list(gf.doors.keys()).index("X-S")
    x_n_idx = list(gf.doors.keys()).index("X-N")
    assert (field.T.data[gf_grid.door_id == x_s_idx] == 0).all()
    assert (field.T.data[gf_grid.door_id == x_n_idx] > 0).all()


def test_blocking_every_exit_in_the_set_raises(building, gf, gf_grid):
    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    with pytest.raises(ValueError):
        compute_gf_field(building, gf, gf_grid, exit_set="south", overlay=overlay)


def test_determinism_same_inputs_give_identical_fields(building, gf, gf_grid):
    a = compute_gf_field(building, gf, gf_grid, exit_set="all")
    b = compute_gf_field(building, gf, gf_grid, exit_set="all")
    assert np.array_equal(a.T.data, b.T.data)
    assert np.array_equal(a.domain, b.domain)


# --------------------------------------------------------------------- FF multi-floor coupling


def test_ff_t_is_finite_everywhere_in_domain(building, ff, ff_grid, gf_grid):
    cache = FieldCache()
    field = cache.get_ff(building, ff, ff_grid, gf_grid)
    assert np.isfinite(field.T.data[field.domain]).all()


def test_ff_dead_end_room_is_slower_than_its_door(building, ff, ff_grid, gf_grid):
    cache = FieldCache()
    field = cache.get_ff(building, ff, ff_grid, gf_grid)
    f10_idx = ff_grid.space_ids.index("F-10")
    interior = (ff_grid.space_id == f10_idx) & ff_grid.walkable_static
    door_idx = list(ff.doors.keys()).index("F-10-d1")
    door_cells = ff_grid.door_id == door_idx
    assert field.T.data[interior].max() > field.T.data[door_cells].max()


def test_excluding_a_stair_never_decreases_ff_travel_time(building, ff, ff_grid, gf_grid):
    cache = FieldCache()
    baseline = cache.get_ff(building, ff, ff_grid, gf_grid)
    detour = cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=frozenset({"ST-C"}))
    domain = baseline.domain
    assert (detour.T.data[domain] >= baseline.T.data[domain] - 1e-6).all()
    stc_idx = ff_grid.stair_ids.index("ST-C")
    near_stc = (ff_grid.stair_id == stc_idx) & domain
    assert detour.T.data[near_stc].mean() > baseline.T.data[near_stc].mean()


def test_two_stairs_lost_still_solves_via_the_third(building, ff, ff_grid, gf_grid):
    cache = FieldCache()
    field = cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=frozenset({"ST-C", "ST-E"}))
    assert np.isfinite(field.T.data[field.domain]).all()


def test_excluding_every_stair_raises(building, ff, ff_grid, gf_grid):
    cache = FieldCache()
    with pytest.raises(ValueError):
        cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=frozenset({"ST-C", "ST-E", "ST-W"}))


def test_st_w_direct_discharge_beats_the_ring_route(building, ff, ff_grid, gf_grid):
    """ST-W discharges straight to X-NW (T_GF(discharge)=0); ST-E and ST-C
    must first cross the GF ring to an exit. For FF cells near ST-W, the
    ST-W-via path should be the one actually selected by the min-combination
    — check indirectly: excluding ST-W measurably raises T near ST-W,
    by more than the ST-W stair-crossing time alone would explain if the
    ring detour were free."""
    cache = FieldCache()
    baseline = cache.get_ff(building, ff, ff_grid, gf_grid)
    without_stw = cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=frozenset({"ST-W"}))
    stw_idx = ff_grid.stair_ids.index("ST-W")
    near_stw = (ff_grid.stair_id == stw_idx) & baseline.domain
    assert (without_stw.T.data[near_stw] > baseline.T.data[near_stw] + 1.0).all()


def test_stairs_blocked_by_scenario_resolves_stair_c_lost(site, building):
    blocked = stairs_blocked_by_scenario(site, building, "stair_C_lost")
    assert blocked == frozenset({"ST-C"})


def test_stairs_blocked_by_scenario_ignores_non_stair_link_events(site, building):
    # block_ringS_mid targets an opening id, not a stair — must resolve empty.
    blocked = stairs_blocked_by_scenario(site, building, "block_ringS_mid")
    assert blocked == frozenset()


# --------------------------------------------------------------------- caching / architecture


def test_cache_hit_does_not_recompute(building, gf, gf_grid):
    cache = FieldCache()
    cache.get_gf(building, gf, gf_grid)
    calls_after_first = cache.calls
    cache.get_gf(building, gf, gf_grid)
    assert cache.calls == calls_after_first


def test_different_overlay_is_a_cache_miss(building, gf, gf_grid):
    cache = FieldCache()
    cache.get_gf(building, gf, gf_grid)
    cache.get_gf(building, gf, gf_grid, overlay=DynamicOverlay(door_states={"X-SE": "UNAVAILABLE"}))
    assert cache.calls == 2


def test_leaf_state_change_is_not_a_cache_miss(building, gf, gf_grid):
    """Leaf state only affects gas permeability, not F(x) — must not trigger
    a pointless recompute."""
    cache = FieldCache()
    cache.get_gf(building, gf, gf_grid, overlay=DynamicOverlay())
    cache.get_gf(building, gf, gf_grid, overlay=DynamicOverlay(leaf_states={"ST-W-fd-GF": "closed"}))
    assert cache.calls == 1


def test_invalidate_clears_only_the_named_building(building, gf, gf_grid):
    cache = FieldCache()
    cache.get_gf(building, gf, gf_grid)
    cache.invalidate(building.id)
    assert cache._gf == {}
    cache.get_gf(building, gf, gf_grid)
    assert cache.calls == 2


def test_field_computation_never_rasterises(building, gf, gf_grid, ff, ff_grid, monkeypatch):
    """Spec §2 design principle 3: a door/leaf/clutter state change must
    never re-rasterise. eikonal.py only ever consumes an already-built
    FloorGrid, so this holds architecturally — proven by making the
    rasteriser explode if anything in this module calls it."""
    def _boom(*args, **kwargs):
        raise AssertionError("compute_gf_field/compute_ff_field must not rasterise")

    monkeypatch.setattr("dms.grid.raster.rasterize_floor", _boom)

    cache = FieldCache()
    cache.get_gf(building, gf, gf_grid, overlay=DynamicOverlay(door_states={"X-SE": "UNAVAILABLE"}))
    cache.get_ff(building, ff, ff_grid, gf_grid)


# --------------------------------------------------------------------- direction field


def test_direction_field_is_unit_length_in_domain(building, gf, gf_grid):
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    ex, ey = direction_field(field.T, field.domain, gf_grid.transform.h)
    mag = np.hypot(ex, ey)
    # exclude cells with ~zero gradient (exactly at an exit / a local flat spot)
    nonzero = field.domain & (mag > 1e-6)
    np.testing.assert_allclose(mag[nonzero], 1.0, atol=1e-6)


def test_direction_field_is_nan_outside_domain(building, gf, gf_grid):
    field = compute_gf_field(building, gf, gf_grid, exit_set="all")
    ex, ey = direction_field(field.T, field.domain, gf_grid.transform.h)
    assert np.isnan(ex[~field.domain]).all()
    assert np.isnan(ey[~field.domain]).all()


def test_domain_mask_excludes_wall_void_outside(building, gf, gf_grid):
    domain = domain_mask(gf_grid)
    assert not domain[gf_grid.wall].any()
    assert not domain[gf_grid.void].any()
    assert not domain[gf_grid.outside].any()
    assert domain[gf_grid.obstacle_static].all()  # obstacles are IN the domain (slow, not excluded)


# --------------------------------------------------------------------- extra obstacle / clutter


def test_clutter_toggle_raises_t_through_the_blocked_corridor(building, gf, gf_grid):
    assert gf_grid.clutter_mask.any()
    off = compute_gf_field(building, gf, gf_grid, overlay=DynamicOverlay(clutter_on=False))
    on = compute_gf_field(building, gf, gf_grid, overlay=DynamicOverlay(clutter_on=True))
    assert (on.T.data[gf_grid.clutter_mask] >= off.T.data[gf_grid.clutter_mask] - 1e-6).all()
    assert on.T.data[gf_grid.clutter_mask].mean() > off.T.data[gf_grid.clutter_mask].mean()


def test_ad_hoc_obstacle_raises_t_locally(building, gf, gf_grid):
    rect = Rect(13.0, 15.5, 19.0, 16.5)  # sits on G-16's own desk-row footprint
    baseline = compute_gf_field(building, gf, gf_grid)
    blocked = compute_gf_field(building, gf, gf_grid, overlay=DynamicOverlay(extra_obstacles=[rect]))
    assert blocked.T.data[baseline.domain].sum() >= baseline.T.data[baseline.domain].sum()
