"""The committed out/redundancy.md
reports must never silently drift from what the code actually produces --
a fresh regeneration must match the committed file byte-for-byte. Generic
over any (config, out_dir, building_id) triple the repo commits a report
for; no room id named here, only the building/report identifiers already
established by the commit layout."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.graph.builder import build_navigation_graph
from dms.graph.redundancy import (
    DEFAULT_DOUBLE_FAILURE_SPECS,
    _double_failure_specs_from_scenarios,
    check_ff_ring_edge_disjoint_paths,
    run_exit_pair_sweep,
    run_named_double_failures,
    run_single_failure_sweep,
    write_redundancy_report,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

COMMITTED_REPORTS = [
    ("configs/buildings/two_storey_large.yaml", "B1", "out/redundancy.md"),
    ("configs/buildings/three_storey_large.yaml", None, "out/three_storey/redundancy.md"),
]


@pytest.mark.parametrize("config_rel,building_id,report_rel", COMMITTED_REPORTS)
def test_committed_redundancy_report_matches_a_fresh_regeneration(tmp_path, config_rel, building_id, report_rel):
    committed_path = REPO_ROOT / report_rel
    assert committed_path.exists(), f"{report_rel} is expected to be committed"
    committed_text = committed_path.read_text(encoding="utf-8")

    site = load_site(REPO_ROOT / config_rel)
    building = site.get_building(building_id) if building_id else site.get_building(site.building_ids()[0])
    nav = build_navigation_graph(site, building.id)

    specs = _double_failure_specs_from_scenarios(site) or list(DEFAULT_DOUBLE_FAILURE_SPECS)
    single = run_single_failure_sweep(nav, building)
    named = run_named_double_failures(nav, building, specs)
    pairs = run_exit_pair_sweep(nav, building)
    ring = check_ff_ring_edge_disjoint_paths(nav)

    fresh_path = write_redundancy_report(nav, building, tmp_path / "redundancy.md", single, named, pairs, ring)
    fresh_text = fresh_path.read_text(encoding="utf-8")

    assert fresh_text == committed_text, (
        f"{report_rel} is stale -- regenerate it (python -m dms.graph.redundancy "
        f"--config {config_rel} --out-dir <its out dir>) and commit the result"
    )
