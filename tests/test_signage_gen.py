"""Generated signage. Parametrized over both
two_storey_large (B1) and three_storey_large (B2) -- the
"every junction has a sign within R_vis" acceptance criterion must now
pass on both, not just B1."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.geometry.signage_gen import (
    _junction_and_turn_candidates,
    apply_generated_signs,
)
from dms.graph.builder import build_navigation_graph
from dms.scenario.agent_defaults import load_agent_defaults

FIXTURES = [
    ("configs/buildings/two_storey_large.yaml", "B1"),
    ("configs/buildings/three_storey_large.yaml", "B2"),
]


def _load(rel_path: str, building_id: str):
    path = Path(__file__).resolve().parents[1] / rel_path
    site = load_site(path)
    return site, site.get_building(building_id)


@pytest.fixture(scope="module")
def agent_defaults():
    return load_agent_defaults()


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_every_junction_has_a_sign_within_r_vis(rel_path, building_id, agent_defaults):
    site, building = _load(rel_path, building_id)
    apply_generated_signs(site, building, agent_defaults)
    nav = build_navigation_graph(site, building.id, use_outdoor_network=False)
    junctions = [
        c for c in _junction_and_turn_candidates(nav, building, turn_cos_threshold=agent_defaults.signage.turn_cos_threshold)
        if c.reason == "junction"
    ]
    assert junctions, f"{building_id}: expected at least one corridor junction"

    r_vis = agent_defaults.perception.R_vis_m
    uncovered = []
    for j in junctions:
        signs_here = building.get_floor(j.floor_id).signs.values()
        best = min(
            (math.hypot(s.x - j.x, s.y - j.y) for s in signs_here),
            default=float("inf"),
        )
        if best > r_vis:
            uncovered.append((j.floor_id, round(j.x, 1), round(j.y, 1), round(best, 2)))
    assert uncovered == [], f"{building_id}: junctions with no sign within R_vis={r_vis}m: {uncovered}"


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_sign_count_per_floor_and_building(rel_path, building_id, agent_defaults):
    """Not a strict pass/fail threshold (there's no single "correct" count)
    -- just confirms every floor with real circulation gets at least one
    sign, and reports the breakdown (visible in -v output / a failure
    message) for the written report."""
    site, building = _load(rel_path, building_id)
    signs = apply_generated_signs(site, building, agent_defaults)
    assert signs, f"{building_id}: expected at least one generated sign"
    per_floor = {fid: len(building.get_floor(fid).signs) for fid in building.floor_ids()}
    assert all(n > 0 for n in per_floor.values()), f"{building_id}: a floor got zero signs: {per_floor}"


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_every_generated_sign_passes_signage_check(rel_path, building_id, agent_defaults):
    from dms.geometry.signage_check import validate_signs

    site, building = _load(rel_path, building_id)
    apply_generated_signs(site, building, agent_defaults)
    nav = build_navigation_graph(site, building.id, use_outdoor_network=False)
    problems = validate_signs(building, nav)
    assert problems == []


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_every_generated_sign_has_valid_attributes(rel_path, building_id, agent_defaults):
    """Type/facing_deg/mount_height_m/illuminated are all
    well-formed, and every sign's face projects into walkable space --
    not just that its route is correct (test_every_generated_sign_passes_
    signage_check, above)."""
    from dms.geometry.signage_check import validate_sign_attributes

    site, building = _load(rel_path, building_id)
    apply_generated_signs(site, building, agent_defaults)
    problems = validate_sign_attributes(building, agent_defaults=agent_defaults)
    assert problems == []


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_signage_overrides_remove_obscure_and_randomise(rel_path, building_id, agent_defaults):
    """A scenario's optional `signage:` block can remove or
    obscure specific signs, or a random fraction of them, for robustness
    testing -- explicit ids are always honoured, random fractions sample
    reproducibly from whatever's left, and a scenario with no `signage:`
    block at all removes/obscures nothing."""
    from dms.scenario.presets import signage_overrides_from_scenario

    site, building = _load(rel_path, building_id)
    apply_generated_signs(site, building, agent_defaults)
    all_ids = {sgid for floor in building.floors.values() for sgid in floor.signs}
    one_id = next(iter(all_ids))

    removed, obscured = signage_overrides_from_scenario({}, building)
    assert removed == frozenset() and obscured == frozenset()

    removed, obscured = signage_overrides_from_scenario(
        {"signage": {"remove": [one_id]}}, building,
    )
    assert removed == frozenset({one_id})
    assert obscured == frozenset()

    preset = {"signage": {"random_remove_fraction": 0.1, "random_obscure_fraction": 0.1}}
    a = signage_overrides_from_scenario(preset, building, seed=7)
    b = signage_overrides_from_scenario(preset, building, seed=7)
    assert a == b
    removed_r, obscured_r = a
    assert removed_r and obscured_r
    assert removed_r.isdisjoint(obscured_r)
    assert removed_r <= all_ids and obscured_r <= all_ids
