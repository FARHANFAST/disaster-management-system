"""spec §3.6: building-code style checks against the independent
reference checker's own numbers. Distances come from an independently
rasterized geodesic solve, so a small (~2%) discretization difference
between two independent rasterizers is expected and tolerated; every
exact, closed-form number (capacities, counts, widths) must match bit-for-
bit."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.code_check import (
    run_code_check,
    write_validation_report,
)
from dms.geometry.loader import load_site
from dms.scenario.agent_defaults import load_agent_defaults

MAX_SINGLE_DOOR_LOAD = load_agent_defaults().redundancy.single_door_max_occupancy

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"

EXPECTED_OVER_49 = {
    ("FF", "F-10"): (269, 2),
    ("GF", "G-05"): (96, 3),
    ("GF", "G-25"): (87, 3),
}


@pytest.fixture(scope="module")
def building_b1l():
    return load_site(CONFIG_PATH).get_building("B1")


@pytest.fixture(scope="module")
def report(building_b1l):
    return run_code_check(building_b1l)


def test_gf_travel_distance_to_exit(report):
    assert report.gf_max_travel_to_exit_m == pytest.approx(52.7, rel=0.02)


def test_ff_travel_distance_to_enclosed_stair(report):
    assert report.ff_max_travel_to_enclosed_stair_m == pytest.approx(48.5, rel=0.02)


def test_enclosed_stair_separation_vs_half_diagonal(report):
    best = max(report.stair_separation_m.values())
    assert best == pytest.approx(86.9, abs=0.1)
    assert best >= report.building_diagonal_m / 2  # the code-compliance condition itself
    assert report.building_diagonal_m == pytest.approx(110.1, abs=0.1)


def test_rooms_over_49_occupants_match_reference_exactly(report):
    by_key = {(r.floor_id, r.space_id): (r.max_load, r.n_doors) for r in report.room_loads}
    for key, expected in EXPECTED_OVER_49.items():
        assert by_key[key] == expected, key
    over_49 = {(r.floor_id, r.space_id) for r in report.room_loads if r.max_load > MAX_SINGLE_DOOR_LOAD}
    assert over_49 == set(EXPECTED_OVER_49)
    for (floor_id, space_id), (_, n_doors) in EXPECTED_OVER_49.items():
        assert n_doors >= 2, f"{floor_id}:{space_id} has {n_doors} door(s) but >49 occupants"


def test_single_door_rooms_max_load(report):
    single_door = [r for r in report.room_loads if r.n_doors == 1]
    assert max(r.max_load for r in single_door) == 39
    assert all(r.max_load <= MAX_SINGLE_DOOR_LOAD for r in single_door)


def test_ff_design_load_by_profile_worst_case(report):
    assert report.worst_floor_load == 451


def test_stair_and_exit_width_capacity(report):
    assert report.stair_capacity_persons == 1132
    assert report.exit_capacity_persons == 2000
    assert report.stair_capacity_persons >= report.worst_floor_load


def test_narrowest_door(report):
    # Realism pass: several low-occupancy service
    # rooms (store/toilets/plant/HVAC) were given narrower, more realistic
    # doors (1.00 -> 0.85 m clear); 0.85 m is the new narrowest door
    # building-wide, still above the code minimum.
    assert report.narrowest_door_m == pytest.approx(0.85, abs=0.001)
    from dms.geometry.code_check import MIN_CLEAR_DOOR_M
    assert report.narrowest_door_m >= MIN_CLEAR_DOOR_M


def test_fn_stub_is_the_one_dead_end_and_exceeds_even_the_sprinklered_limit(report):
    """FN-STUB is a deliberate trap-test dead-end by design, "slightly
    over the sprinklered limit" -- 15.5 m vs 15.2 m
    sprinklered (and well over the 6.1 m unsprinklered limit too). Flagged
    as a known, intentional exception in validation_b1l.md, not a bug to
    silently pass."""
    from dms.geometry.code_check import DEAD_END_SPRINKLERED_M, DEAD_END_UNSPRINKLERED_M

    assert len(report.dead_end_corridors) == 1
    fid, sid, length = report.dead_end_corridors[0]
    assert (fid, sid) == ("FF", "FN-STUB")
    assert length == pytest.approx(15.5, abs=0.1)
    assert length > DEAD_END_UNSPRINKLERED_M
    assert length > DEAD_END_SPRINKLERED_M


def test_write_validation_report(tmp_path, report):
    out = tmp_path / "validation_b1l.md"
    path = write_validation_report(
        report, out, known_exceptions={"FN-STUB": "deliberate dead-end lounge corridor, trap-test fixture"},
    )
    text = path.read_text(encoding="utf-8")
    assert "Building-code validation report" in text
    assert "FN-STUB" in text
    assert "known exception" in text
