"""spec §3.3: cell-level redundancy sweep, cross-checked against the
independent reference checker's own redundancy.py run against this exact
building -- category counts, affected-room counts, and the precise 35-room
dead-end identity list all confirmed to match exactly (verified once by
running the reference tool directly; pinned here as literal expected
values, not re-run every test session)."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.grid.cell_redundancy import run_cell_redundancy_sweep

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"

# From a direct run of the independent reference tool's redundancy.py
# against two_storey_large.yaml (matches spec §3.3's table exactly).
EXPECTED_DEAD_ENDS = frozenset({
    ("FF", "F-01"), ("FF", "F-04"), ("FF", "F-05"), ("FF", "F-07"), ("FF", "F-09"), ("FF", "F-11"),
    ("FF", "F-12"), ("FF", "F-14"), ("FF", "F-15"), ("FF", "F-17"), ("FF", "F-20"),
    ("GF", "G-01"), ("GF", "G-02"), ("GF", "G-03"), ("GF", "G-04"), ("GF", "G-06"), ("GF", "G-07"),
    ("GF", "G-08"), ("GF", "G-09"), ("GF", "G-10"), ("GF", "G-11"), ("GF", "G-12"), ("GF", "G-14"),
    ("GF", "G-15"), ("GF", "G-17"), ("GF", "G-18"), ("GF", "G-19"), ("GF", "G-20"), ("GF", "G-22"),
    ("GF", "G-23"), ("GF", "G-24"), ("GF", "G-26"), ("GF", "G-28"), ("GF", "G-30"), ("GF", "G-31"),
})


@pytest.fixture(scope="module")
def building_b1l():
    return load_site(CONFIG_PATH).get_building("B1")


@pytest.fixture(scope="module")
def sweep(building_b1l):
    return run_cell_redundancy_sweep(building_b1l)


@pytest.mark.slow
def test_baseline_nothing_disconnected(sweep):
    assert sweep.base_bad == frozenset()


@pytest.mark.slow
def test_door_and_fire_door_categories_total_83_with_35_dead_ends(sweep):
    door_like = [r for r in sweep.single if r.case.kind in ("door", "fire_door")]
    assert len(door_like) == 83
    affected = frozenset().union(*(r.affected for r in door_like))
    assert affected == EXPECTED_DEAD_ENDS


@pytest.mark.slow
def test_stair_opening_category_is_8_with_no_dead_ends(sweep):
    cases = [r for r in sweep.single if r.case.kind == "stair_opening"]
    assert len(cases) == 8
    assert all(not r.affected for r in cases)


@pytest.mark.slow
def test_corridor_cut_category_is_78_with_no_dead_ends(sweep):
    cases = [r for r in sweep.single if r.case.kind == "corridor"]
    assert len(cases) == 78
    assert all(not r.affected for r in cases)


@pytest.mark.slow
def test_exit_category_is_6_with_no_dead_ends(sweep):
    cases = [r for r in sweep.single if r.case.kind == "exit"]
    assert len(cases) == 6  # the 6 AVAILABLE exits; X-SE (LOCKED) is excluded
    assert all(not r.affected for r in cases)


@pytest.mark.slow
def test_stair_category_is_5_with_no_dead_ends(sweep):
    cases = [r for r in sweep.single if r.case.kind == "stair"]
    assert len(cases) == 5
    assert all(not r.affected for r in cases)


@pytest.mark.slow
def test_double_exit_pairs_are_15_with_no_dead_ends(sweep):
    assert len(sweep.double_exit) == 15  # C(6, 2)
    assert all(not r.affected for r in sweep.double_exit)


@pytest.mark.slow
def test_double_stair_pairs_are_10_with_no_dead_ends(sweep):
    assert len(sweep.double_stair) == 10  # C(5, 2)
    assert all(not r.affected for r in sweep.double_stair)


@pytest.mark.slow
def test_triple_stair_combos_are_10_with_no_dead_ends(sweep):
    assert len(sweep.triple_stair) == 10  # C(5, 3)
    assert all(not r.affected for r in sweep.triple_stair)


@pytest.mark.slow
def test_sweep_completes_within_a_generous_budget(building_b1l):
    import time

    t0 = time.perf_counter()
    run_cell_redundancy_sweep(building_b1l)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"cell redundancy sweep took {elapsed:.1f}s"
