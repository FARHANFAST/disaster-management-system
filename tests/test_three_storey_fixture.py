"""Irregular 3-storey fixture: proves the N-floor
generalization work (Stair.landings subsets, the recursive Eikonal field,
redundancy.py's generic reference-room/double-failure handling) against a
building that is genuinely irregular, not a scaled repeat of
configs/buildings/two_storey.yaml -- see three_storey_large.yaml's own
header comment for the specific topology features it exercises.

Build time (load + rasterize every floor + build
the nav graph + solve every floor's recursive Eikonal field) is budgeted
and asserted separately from the full single-failure redundancy sweep,
since the sweep is not held to the same tight bound.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from dms.fields.eikonal import FieldCache
from dms.geometry.loader import load_site
from dms.geometry.site import Site
from dms.geometry.validate import validate_site
from dms.graph.astar import route
from dms.graph.builder import NavGraph, build_navigation_graph
from dms.graph.redundancy import (
    check_ff_ring_edge_disjoint_paths,
    run_exit_pair_sweep,
    run_named_double_failures,
    run_single_failure_sweep,
)
from dms.grid.layers import H_COARSE, H_FINE, FloorGrid
from dms.grid.raster import envelope_leak_mask, rasterize_floor

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "three_storey_large.yaml"

BUILD_BUDGET_S = 10.0
REDUNDANCY_BUDGET_S = 60.0


@pytest.fixture(scope="module")
def site3() -> Site:
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="module")
def building3(site3: Site):
    return site3.get_building("B2")


@pytest.fixture(scope="module")
def grids3(building3) -> dict[str, FloorGrid]:
    return {
        fid: rasterize_floor(building3, building3.get_floor(fid), H_COARSE)
        for fid in building3.floor_ids()
    }


@pytest.fixture(scope="module")
def nav3(site3: Site) -> NavGraph:
    return build_navigation_graph(site3, "B2")


# --------------------------------------------------------------------- topology shape


def test_building_has_three_floors_in_order(building3):
    assert building3.floor_ids() == ["L1", "L2", "L3"]


def test_stair_risers_sum_exactly_to_floor_to_floor_height(building3):
    """spec §2: same generic riser-sum invariant as two_storey's own
    test, re-checked here since this fixture's stairs are distinct objects."""
    for stair in building3.stairs.values():
        assert sum(stair.rise_per_flight) == pytest.approx(stair.floor_to_floor_m)


def test_five_stairs_with_the_required_span_pattern(building3):
    stairs = building3.stairs
    assert set(stairs) == {"ST-W", "ST-E", "ST-C", "ST-SUB", "ST-FIRE"}
    # 2 enclosed end stairs + 1 main open stair spanning every floor
    for sid in ("ST-W", "ST-E", "ST-C"):
        assert list(stairs[sid].landings) == ["L1", "L2", "L3"]
    assert stairs["ST-W"].enclosed and stairs["ST-E"].enclosed
    assert not stairs["ST-C"].enclosed
    # a stair spanning only floors 1-2
    assert list(stairs["ST-SUB"].landings) == ["L1", "L2"]
    # the external fire stair: its own direct exterior discharge, every floor
    assert list(stairs["ST-FIRE"].landings) == ["L1", "L2", "L3"]
    assert stairs["ST-FIRE"].discharge_exit_id == "X-E-STFIRE"


def test_l3_room_layout_differs_from_l2_not_just_renamed(building3):
    l2, l3 = building3.get_floor("L2"), building3.get_floor("L3")
    # L2 has ST-SUB as a stair-kind space; L3 has no such space at all --
    # that slot is two ordinary offices instead, a genuine layout change.
    assert "ST-SUB" in l2.spaces
    assert "ST-SUB" not in l3.spaces
    assert "L3-M1" in l3.spaces and "L3-M2" in l3.spaces
    assert l3.spaces["L3-M1"].kind == "room" and l3.spaces["L3-M2"].kind == "room"
    room_kinds_l2 = {sid for sid, sp in l2.spaces.items() if sp.kind == "room"}
    room_kinds_l3 = {sid for sid, sp in l3.spaces.items() if sp.kind == "room"}
    assert room_kinds_l2 != room_kinds_l3


def test_l_shaped_corridor_exists(building3):
    """corr-A (east-west) and corr-B (north-south) meet at a right-angle
    corner via a corridor_opening -- the same pattern the two-storey
    building's ring corners already use."""
    l1 = building3.get_floor("L1")
    assert "corr-A" in l1.spaces and "corr-B" in l1.spaces
    corr_a, corr_b = l1.spaces["corr-A"], l1.spaces["corr-B"]
    # corr-A runs along x (long axis x), corr-B runs along y (long axis y)
    assert corr_a.rect.width > corr_a.rect.height
    assert corr_b.rect.height > corr_b.rect.width
    junction = l1.openings["corr-L-junction"]
    assert {junction.space_a, junction.space_b} == {"corr-A", "corr-B"}


def test_corridor_widths_within_spec_range(building3):
    for floor in building3.floors.values():
        for sp in floor.spaces.values():
            if sp.kind != "corridor":
                continue
            short = min(sp.rect.width, sp.rect.height)
            assert 1.8 <= short <= 4.5, f"{floor.id}/{sp.id} short width {short}"


def test_eight_exits_on_the_ground_floor_only(building3):
    l1 = building3.get_floor("L1")
    assert len(l1.exit_doors) == 8
    for fid in ("L2", "L3"):
        assert building3.get_floor(fid).exit_doors == []


def test_two_restricted_hazard_rooms_on_different_floors(building3):
    hazard_spaces = []
    for floor in building3.floors.values():
        for hz in floor.hazards.values():
            hazard_spaces.append((floor.id, hz.space_id))
    assert len(hazard_spaces) == 2
    floors_with_hazards = {fid for fid, _ in hazard_spaces}
    assert len(floors_with_hazards) == 2
    for fid, space_id in hazard_spaces:
        assert building3.get_floor(fid).spaces[space_id].restricted


# --------------------------------------------------------------------- validation / watertightness


def test_validate_site_passes():
    report = validate_site(load_site(CONFIG_PATH))
    assert report.ok, report.to_markdown()


@pytest.mark.parametrize("h", (H_COARSE, H_FINE))
def test_every_floor_envelope_is_watertight(building3, h):
    for floor in building3.floors.values():
        grid = rasterize_floor(building3, floor, h)
        leak = envelope_leak_mask(grid)
        breached = leak & (grid.space_id != -1)
        assert not breached.any(), f"{floor.id}: {breached.sum()} interior cells reached from outside at h={h}"


# --------------------------------------------------------------------- build budget + N-floor Eikonal recursion


def test_build_completes_within_budget(site3):
    """Load + rasterize every floor + build the nav graph + solve every
    floor's recursive Eikonal field, measured separately from the
    redundancy sweep (plan's own timing split)."""
    t0 = time.perf_counter()
    building = site3.get_building("B2")
    grids = {fid: rasterize_floor(building, building.get_floor(fid), H_COARSE) for fid in building.floor_ids()}
    build_navigation_graph(site3, "B2")
    cache = FieldCache()
    for fid in building.floor_ids():
        field = cache.get_field(building, fid, grids)
        assert field.domain.any()
    elapsed = time.perf_counter() - t0
    assert elapsed < BUILD_BUDGET_S, f"build took {elapsed:.2f}s, budget {BUILD_BUDGET_S}s"


def test_recursive_field_is_finite_on_every_floor(building3, grids3):
    import numpy as np

    cache = FieldCache()
    for fid in building3.floor_ids():
        field = cache.get_field(building3, fid, grids3)
        assert np.isfinite(field.T.data[field.domain]).all(), f"{fid}: non-finite travel time in domain"


def test_subset_stair_only_couples_l1_l2(building3, grids3):
    """ST-SUB must never participate in L3's field coupling (it has no
    landing there) -- exercised indirectly: L3's field must still solve
    successfully using only ST-W/ST-C/ST-E/ST-FIRE."""
    cache = FieldCache()
    field_l3 = cache.get_field(building3, "L3", grids3)
    assert field_l3.domain.any()
    import numpy as np
    assert np.isfinite(field_l3.T.data[field_l3.domain]).all()


# --------------------------------------------------------------------- routing + determinism


def test_a_star_routes_resolve_on_every_floor(nav3):
    sample_rooms = ["L1:R:S2", "L2:R:N1", "L3:R:N1", "L2:R:N2", "L1:R:S4"]
    for rid in sample_rooms:
        r = route(nav3, rid)
        assert r is not None and r.cost > 0.0, rid


def test_determinism_same_site_rebuilt_gives_identical_routes():
    costs = []
    for _ in range(2):
        site = load_site(CONFIG_PATH)
        nav = build_navigation_graph(site, "B2")
        costs.append(route(nav, "L3:R:N1").cost)
    assert costs[0] == costs[1]


# --------------------------------------------------------------------- redundancy sweep (separate time budget)


def test_redundancy_sweep_completes_within_generous_budget(nav3, building3):
    t0 = time.perf_counter()
    single = run_single_failure_sweep(nav3, building3)
    named = run_named_double_failures(nav3, building3)
    pairs = run_exit_pair_sweep(nav3, building3)
    ring = check_ff_ring_edge_disjoint_paths(nav3)
    elapsed = time.perf_counter() - t0

    assert elapsed < REDUNDANCY_BUDGET_S, f"redundancy sweep took {elapsed:.2f}s, budget {REDUNDANCY_BUDGET_S}s"
    assert len(single) > 0
    # this building defines no "two_stairs_lost"/"block_main_exit" scenarios,
    # so the default specs should resolve to nothing rather than crash
    assert named == []
    assert len(pairs) == 28  # C(8,2) exit pairs
    assert isinstance(ring, dict)  # no floor here is named "FF"; an empty dict is correct, not a bug
