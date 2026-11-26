"""Outdoor gate/walkway redundancy -- "close any one
gate or any one walkway segment, and every exit still reaches at least 2
assembly points." Generic over any building that declares gates/walkways;
no id hardcoded here."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.graph.builder import build_navigation_graph
from dms.graph.redundancy import check_exit_ap_redundancy_after_gate_or_walkway_failure

FIXTURES = [
    ("configs/buildings/two_storey_large.yaml", "B1"),
    ("configs/buildings/three_storey_large.yaml", None),
]


@pytest.mark.parametrize("rel_path,building_id", FIXTURES)
def test_every_exit_reaches_at_least_two_aps_after_any_gate_or_walkway_failure(rel_path, building_id):
    path = Path(__file__).resolve().parents[1] / rel_path
    site = load_site(path)
    building = site.get_building(building_id) if building_id else site.get_building(site.building_ids()[0])
    nav = build_navigation_graph(site, building.id)

    results = check_exit_ap_redundancy_after_gate_or_walkway_failure(nav)
    assert results, f"{building.id}: expected at least one gate/walkway-segment failure case"
    below_target = {label: n for label, n in results.items() if n < 2}
    assert below_target == {}, f"{building.id}: gate/walkway failures leaving <2 reachable APs: {below_target}"
