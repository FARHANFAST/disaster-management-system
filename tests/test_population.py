from __future__ import annotations

from collections import Counter

import pytest

from dms.geometry.building import Building
from dms.scenario.agent_defaults import load_agent_defaults
from dms.scenario.population import (
    PopulationError,
    _headcount,
    effective_speed,
    sample_population,
)

_AGENT_DEFAULTS = load_agent_defaults()
_BODY_CLASSES = _AGENT_DEFAULTS.body_classes
DEFAULT_PERSONAS = _AGENT_DEFAULTS.personas


def test_determinism_same_seed_same_result(building):
    a = sample_population(building, "day", seed=42)
    b = sample_population(building, "day", seed=42)
    assert a == b


def test_different_seed_can_differ(building):
    a = sample_population(building, "day", seed=1)
    b = sample_population(building, "day", seed=2)
    # Same total room/preset structure, but at least one coordinate or
    # persona assignment differs between independent seeds.
    assert a != b


def test_restricted_rooms_only_sample_operator(building):
    agents = sample_population(building, "day", seed=7)
    restricted_space_ids = {
        sp.id
        for floor in building.floors.values()
        for sp in floor.spaces.values()
        if sp.restricted
    }
    assert restricted_space_ids, "fixture should have at least one restricted room"
    restricted_agents = [a for a in agents if a.space_id in restricted_space_ids]
    assert restricted_agents, "restricted rooms should produce at least one agent across presets"
    assert all(a.persona == "operator" for a in restricted_agents)


def test_no_agent_overlaps_another_in_same_room(building):
    """Placement spacing uses each agent's
    own body-class radius (the sum of both agents' own radii), not a
    single shared constant."""
    agents = sample_population(building, "day", seed=11)
    by_room: dict[tuple[str, str], list] = {}
    for a in agents:
        by_room.setdefault((a.floor, a.space_id), []).append(a)
    for room_agents in by_room.values():
        for i, a in enumerate(room_agents):
            ra = _BODY_CLASSES[a.body_class].clearance_radius_m
            for b in room_agents[i + 1 :]:
                rb = _BODY_CLASSES[b.body_class].clearance_radius_m
                dist2 = (a.x - b.x) ** 2 + (a.y - b.y) ** 2
                assert dist2 >= (ra + rb - 1e-6) ** 2, (a, b)


def test_no_agent_overlaps_obstacle(building):
    agents = sample_population(building, "day", seed=11)
    for a in agents:
        radius = _BODY_CLASSES[a.body_class].clearance_radius_m
        floor = building.get_floor(a.floor)
        for obs in floor.obstacles.values():
            if obs.space_id != a.space_id or not obs.active_by_default:
                continue
            r = obs.rect
            # agent disc center must not fall inside the obstacle rect
            # expanded by the agent radius on every side
            if (r.x0 - radius) < a.x < (r.x1 + radius) and (r.y0 - radius) < a.y < (r.y1 + radius):
                # only a true failure if the disc actually overlaps the rect
                closest_x = min(max(a.x, r.x0), r.x1)
                closest_y = min(max(a.y, r.y0), r.y1)
                dist2 = (a.x - closest_x) ** 2 + (a.y - closest_y) ** 2
                assert dist2 >= radius * radius - 1e-6, (a, obs.id)


def test_n_rescale_matches_requested_total(building):
    agents = sample_population(building, "day", seed=5, N=200)
    assert agents  # non-empty
    # Rounding (fractional carry) keeps the total within a few agents of N.
    assert abs(len(agents) - 200) <= 5


def test_persona_mix_weights_respected_statistically(building):
    agents = sample_population(building, "day", seed=3, N=4000)
    non_operator = [a for a in agents if a.persona != "operator"]
    counts = Counter(a.persona for a in non_operator)
    total = sum(counts.values())
    assert total > 500  # enough samples for a meaningful proportion check
    for persona_id, entry in building.persona_mix.items():
        observed_frac = counts.get(persona_id, 0) / total
        assert observed_frac == pytest.approx(entry["share"], abs=0.05)


def test_unknown_persona_mix_id_raises():
    b = Building(id="X", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)
    b.persona_mix = {"not_a_real_persona": {"share": 1.0, "v0": None}}
    with pytest.raises(PopulationError):
        sample_population(b, "day", seed=1)


def test_persona_id_lookup_is_case_insensitive():
    """B1-L's own source YAML spells its persona_mix
    ids with a capital gender suffix (adult_M/elderly_F/...), which doesn't
    match DEFAULT_PERSONAS' fixed lowercase spelling by bare `==`, even
    though they're the same identity -- sample_population must not reject a
    building over that, and effective_speed's catalog fallback must resolve
    it too."""
    b = Building(id="X", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)
    b.persona_mix = {"Adult_M": {"share": 1.0, "v0": None}}
    sample_population(b, "day", seed=1)  # must not raise
    assert effective_speed("Adult_M", b.persona_mix) == DEFAULT_PERSONAS["adult_m"].speed_mps


def test_unoccupied_space_kinds_are_skipped(building):
    agents = sample_population(building, "day", seed=9)
    stair_or_void_ids = {
        sp.id
        for floor in building.floors.values()
        for sp in floor.spaces.values()
        if sp.kind in ("stair", "void")
    }
    assert not any(a.space_id in stair_or_void_ids for a in agents)


def test_count_range_headcount_is_within_bounds_and_deterministic():
    """spec §2 regression: count_range is already generic in _headcount
    (locked in here directly, not just exercised indirectly through a full
    sample_population call) -- same seed/leaf always gives the same count,
    and it never falls outside [lo, hi]."""
    import random

    leaf = {"count_range": [2, 5]}
    for seed in range(20):
        n = _headcount(leaf, area_m2=0.0, rng=random.Random(seed))
        assert 2 <= n <= 5
    assert _headcount(leaf, area_m2=0.0, rng=random.Random(7)) == _headcount(leaf, area_m2=0.0, rng=random.Random(7))


def test_default_personas_cover_persona_mix(building):
    assert set(building.persona_mix).issubset(DEFAULT_PERSONAS)
