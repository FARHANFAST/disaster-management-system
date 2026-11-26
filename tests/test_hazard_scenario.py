"""The hazard scenario interface -- which hazard sources are
active, their release profile, and wind, resolved from a scenario's
`hazard:` block and validated. No dispersion physics is exercised here
-- only that the data resolves to something internally
consistent, is reproducible by seed, and that a declared range actually
varies across seeds."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.scenario.agent_defaults import load_agent_defaults
from dms.scenario.hazard_scenario import (
    resolve_hazard_scenario,
    validate_hazard_scenario,
)

FIXTURES = [
    ("configs/buildings/two_storey_large.yaml", "h4_release_day", "train"),
    ("configs/buildings/two_storey_large.yaml", "h4_release_windy", "test"),
    ("configs/buildings/three_storey_large.yaml", "h3_release_day", "train"),
    ("configs/buildings/three_storey_large.yaml", "h3_release_windy", "test"),
]


@pytest.fixture(scope="module")
def agent_defaults():
    return load_agent_defaults()


def _load(rel_path: str):
    return load_site(Path(__file__).resolve().parents[1] / rel_path)


@pytest.mark.parametrize("rel_path,scenario_name,expected_split", FIXTURES)
def test_hazard_scenario_resolves_and_validates(rel_path, scenario_name, expected_split, agent_defaults):
    site = _load(rel_path)
    preset = site.scenarios[scenario_name]
    hs = resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=1)
    assert hs is not None
    assert hs.split == expected_split
    assert hs.releases, f"{scenario_name}: expected at least one active hazard source"
    problems = validate_hazard_scenario(site, hs)
    assert problems == [], f"{scenario_name}: {problems}"


@pytest.mark.parametrize("rel_path,scenario_name", [(f[0], f[1]) for f in FIXTURES if not f[1].endswith("_day")])
def test_hazard_scenario_same_seed_same_values(rel_path, scenario_name, agent_defaults):
    site = _load(rel_path)
    preset = site.scenarios[scenario_name]
    a = resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=42)
    b = resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=42)
    assert a == b


@pytest.mark.parametrize("rel_path,scenario_name", [(f[0], f[1]) for f in FIXTURES if not f[1].endswith("_day")])
def test_hazard_scenario_different_seed_varies_a_randomised_field(rel_path, scenario_name, agent_defaults):
    site = _load(rel_path)
    preset = site.scenarios[scenario_name]
    samples = [resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=s) for s in range(8)]
    wind_from = {round(hs.wind_from_deg, 6) for hs in samples}
    assert len(wind_from) > 1, f"{scenario_name}: wind_from_deg never varied across 8 seeds"


def test_scenario_without_a_hazard_block_resolves_to_none(agent_defaults):
    site = _load("configs/buildings/two_storey_large.yaml")
    preset = site.scenarios["baseline"]
    assert resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=1) is None


def test_active_sources_all_expands_to_every_site_hazard(agent_defaults):
    site = _load("configs/buildings/three_storey_large.yaml")
    preset = site.scenarios["h3_release_windy"]
    hs = resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=3)
    assert {r.source_id for r in hs.releases} == set(site.hazards)


def test_unknown_source_id_fails_validation(agent_defaults):
    site = _load("configs/buildings/two_storey_large.yaml")
    preset = {"hazard": {"active_sources": ["NOT-A-REAL-SOURCE"]}}
    hs = resolve_hazard_scenario(site, preset, agent_defaults=agent_defaults, seed=1)
    problems = validate_hazard_scenario(site, hs)
    assert any("not a hazard source" in p for p in problems)
