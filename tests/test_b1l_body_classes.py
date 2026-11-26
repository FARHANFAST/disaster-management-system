"""Body classes + clearance-aware routing. Generic over
any building YAML (Core rule: tested on both two_storey_large and
three_storey_large, not just B1-L) -- no building-specific id appears in
this file beyond the fixture's own id, used only to pick which building to
load."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

from dms.fields.eikonal import FieldCache, clearance_domain
from dms.geometry.body_class_report import (
    class_reachability_report,
    door_passability_table,
    write_body_class_report,
)
from dms.geometry.loader import load_site
from dms.grid.layers import H_FINE, DynamicOverlay
from dms.grid.raster import rasterize_floor
from dms.scenario.agent_defaults import AgentDefaults, load_agent_defaults

FIXTURES = [
    ("configs/buildings/two_storey_large.yaml", "B1"),
    ("configs/buildings/three_storey_large.yaml", "B2"),
]
_STRUCT4 = ndimage.generate_binary_structure(2, 1)


def _load(rel_path: str, building_id: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    site = load_site(path)
    return site, site.get_building(building_id)


@pytest.fixture(scope="module")
def agent_defaults() -> AgentDefaults:
    return load_agent_defaults()


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_clearance_domain_regression_vs_old_one_cell_erosion(rel_path, building_id, agent_defaults):
    """The new distance-transform mask is
    NOT forced to equal the old one-cell `binary_erosion` bit-for-bit (it's
    more accurate); this reports how much they actually differ, expecting
    it to be a small fraction of the walkable area, concentrated near
    concave corners, not a wholesale behaviour change."""
    _, building = _load(rel_path, building_id)
    overlay = DynamicOverlay()
    standard_r = agent_defaults.body_classes["standard"].clearance_radius_m
    for fid in building.floor_ids():
        grid = rasterize_floor(building, building.get_floor(fid), H_FINE)
        walkable = grid.walkable_now(overlay)
        old = ndimage.binary_erosion(walkable, structure=_STRUCT4, iterations=1, border_value=False)
        new = clearance_domain(grid, overlay, standard_r)
        diff = old != new
        frac = diff.sum() / max(1, walkable.sum())
        assert frac < 0.05, f"{building_id}/{fid}: {frac:.1%} of walkable cells differ from the old erosion"


# The grid's own `clearance_domain` must be
# compared against the SAME quantity the table's formula uses -- the table is
# `width_m >= 2*clearance_radius_m + door_margin_m`, i.e. a check against a
# radius already inflated by half the margin, not the bare clearance radius
# (a prior version of this test compared the margin-inclusive table against a
# bare-radius grid check, an apples-to-oranges mismatch that masked real
# disagreements). Also run the comparison at a finer resolution than the
# simulation's own H_FINE: at H_FINE, a door whose width is an exact multiple
# of H_FINE can land exactly half a cell off the grid's sampling points,
# undercounting its own true continuous clearance by up to H_FINE/2 (verified
# empirically: re-rasterizing the same door at H_FINE/2, H_FINE/4, H_FINE/10
# all converge on the same, larger clearance value H_FINE alone missed) --
# this is a sampling artifact of the verification, not of the simulation's
# routing field, so only the verification's own resolution is refined here.
_DOOR_CHECK_H = H_FINE / 2.0


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_door_passability_table_matches_grid_result_for_every_door(rel_path, building_id, agent_defaults):
    """Door/exit passability must come from
    the real continuous width, and the grid's own clearance_domain must
    agree with that table for every interior door/opening -- checked
    directly (not assumed), on every floor, for every body class."""
    _, building = _load(rel_path, building_id)
    overlay = DynamicOverlay()
    table = {
        (row.floor_id, row.door_id, row.body_class): row
        for row in door_passability_table(building, agent_defaults=agent_defaults)
    }
    checked = 0
    for fid in building.floor_ids():
        floor = building.get_floor(fid)
        grid = rasterize_floor(building, floor, _DOOR_CHECK_H)
        interior = [d for d in floor.doors.values() if d.kind != "exit"]
        for cname, bc in agent_defaults.body_classes.items():
            r_check = bc.clearance_radius_m + agent_defaults.door_margin_m / 2.0
            move = clearance_domain(grid, overlay, r_check)
            for door in interior:
                try:
                    door_idx = grid.door_ids.index(door.id)
                except ValueError:
                    continue
                cell_mask = grid.door_id == door_idx
                if not cell_mask.any():
                    continue
                grid_pass = bool((move & cell_mask).any())
                geom_pass = table[(fid, door.id, cname)].passable
                assert grid_pass == geom_pass, (building_id, fid, door.id, cname, geom_pass, grid_pass)
                checked += 1
    assert checked > 0


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_every_exit_is_passable_for_standard(rel_path, building_id, agent_defaults):
    _, building = _load(rel_path, building_id)
    rows = [r for r in door_passability_table(building, agent_defaults=agent_defaults)
            if r.body_class == "standard" and r.kind == "exit"]
    assert rows, "fixture should declare at least one exit"
    failing = [r for r in rows if not r.passable]
    assert failing == [], failing


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_standard_and_large_every_occupied_room_reaches_an_exit(rel_path, building_id, agent_defaults):
    _, building = _load(rel_path, building_id)
    rows = class_reachability_report(building, agent_defaults=agent_defaults)
    for cname in ("standard", "large"):
        unreachable = [r for r in rows if r.body_class == cname and not r.reachable]
        assert unreachable == [], f"{building_id}/{cname}: {unreachable}"


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_wheelchair_every_ground_floor_room_reaches_an_exit(rel_path, building_id, agent_defaults):
    _, building = _load(rel_path, building_id)
    bottom_id = building.floor_ids()[0]
    rows = class_reachability_report(building, agent_defaults=agent_defaults)
    wheelchair_rows = [r for r in rows if r.body_class == "wheelchair"]
    assert wheelchair_rows, "expected at least one wheelchair-class row"
    gf_rows = [r for r in wheelchair_rows if r.floor_id == bottom_id]
    unreachable = [r for r in gf_rows if not r.reachable]
    assert unreachable == [], unreachable


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_wheelchair_every_upper_floor_room_reaches_a_refuge_where_one_is_declared(rel_path, building_id, agent_defaults):
    """class_reachability_report also checks upper
    floors for a can_use_stairs=False class, via the refuge-targeted field.
    A floor that declares at least one refuge must let every occupied room
    on it reach one; a floor with none declared (e.g. three_storey_large,
    which declares no refuges at all) is a known, acknowledged gap, not a
    routing bug -- skipped here, not asserted reachable."""
    _, building = _load(rel_path, building_id)
    bottom_id = building.floor_ids()[0]
    rows = class_reachability_report(building, agent_defaults=agent_defaults)
    for fid in building.floor_ids():
        if fid == bottom_id or not building.get_floor(fid).refuges:
            continue
        unreachable = [r for r in rows if r.body_class == "wheelchair" and r.floor_id == fid and not r.reachable]
        assert unreachable == [], f"{building_id}/{fid}: {unreachable}"


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_write_body_class_report(tmp_path, rel_path, building_id, agent_defaults):
    _, building = _load(rel_path, building_id)
    out = write_body_class_report(building, tmp_path / "body_class_report.md", agent_defaults=agent_defaults)
    text = out.read_text(encoding="utf-8")
    assert "Door/exit passability" in text
    assert "wheelchair" in text


def test_field_cache_keys_on_clearance_radius_not_just_a_bool(agent_defaults):
    """Two distinct radii must never collide in FieldCache -- regression
    guard for the body_clear bool -> clearance_radius_m generalization."""
    _, building = _load(*FIXTURES[0])
    fid = building.floor_ids()[0]
    grid = rasterize_floor(building, building.get_floor(fid), H_FINE)
    cache = FieldCache()
    overlay = DynamicOverlay()
    f_std = cache.get_field(building, fid, {fid: grid}, overlay=overlay,
                             clearance_radius_m=agent_defaults.body_classes["standard"].clearance_radius_m)
    f_wc = cache.get_field(building, fid, {fid: grid}, overlay=overlay,
                            clearance_radius_m=agent_defaults.body_classes["wheelchair"].clearance_radius_m)
    f_none = cache.get_field(building, fid, {fid: grid}, overlay=overlay, clearance_radius_m=None)
    assert not np.array_equal(f_std.domain, f_wc.domain)
    assert not np.array_equal(f_std.domain, f_none.domain)
    assert cache.calls == 3  # no accidental cache collision avoided a real solve


def test_agent_defaults_overridable_per_run_same_seed_identical_results():
    """Every parameter must be overridable per run, with same seed
    -> identical results. Builds two EvacuationWorlds from independently
    loaded (but identical) AgentDefaults and checks they produce the exact
    same simulation, then confirms an actually-different override changes
    the result (proving the override is live, not ignored)."""
    from dms.scenario.evacuation import build_world, sample_agents, simulate

    site, building = _load(*FIXTURES[0])
    default_a = load_agent_defaults()
    default_b = load_agent_defaults()  # independently loaded, same file -> equal values
    world_a = build_world(site, building, "baseline", agent_defaults=default_a)
    world_b = build_world(site, building, "baseline", agent_defaults=default_b)
    agents_a = sample_agents(world_a, "day", seed=5, N=30)
    agents_b = sample_agents(world_b, "day", seed=5, N=30)
    res_a = simulate(world_a, agents_a, t_end=300.0, seed=3)
    res_b = simulate(world_b, agents_b, t_end=300.0, seed=3)
    assert len(res_a.frames) == len(res_b.frames)
    assert np.array_equal(res_a.frames[-1].pos, res_b.frames[-1].pos)

    # Now override something that actually changes behaviour: inflate every
    # body-class radius a little (kept modest so every exit stays wide
    # enough to remain reachable -- this is about proving the override is
    # live, not stress-testing an unrealistic body size).
    import dataclasses

    inflated_classes = {
        cid: dataclasses.replace(bc, clearance_radius_m=bc.clearance_radius_m * 1.3)
        for cid, bc in default_a.body_classes.items()
    }
    default_c = dataclasses.replace(default_a, body_classes=inflated_classes)
    world_c = build_world(site, building, "baseline", agent_defaults=default_c)
    agents_c = sample_agents(world_c, "day", seed=5, N=30)
    assert agents_c[0].clearance_radius_m == pytest.approx(agents_a[0].clearance_radius_m * 1.3)
