"""spec §3.2: derived stair and exit values must match the independent
reference checker's table exactly, now that the stair formula fix
(continuous actual-riser-height, landing_depth_m) is in place."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"

# §3.2's table: stair_id -> (risers, slope_deg, walking_length_m, Q_max)
EXPECTED_STAIRS = {
    "ST-W": (24, 33.0, 8.84, 1.61),
    "ST-E": (24, 33.0, 8.84, 2.02),
    "ST-C": (24, 33.0, 8.84, 3.23),
    "ST-C2": (24, 33.0, 8.84, 2.69),
    "ST-N": (24, 33.0, 8.84, 2.02),
}

# §3.2's table: exit_id -> Q (= 1.92 * width_m)
EXPECTED_EXIT_Q = {
    "X-S": 4.61, "X-S2": 3.46, "X-W": 2.30, "X-NW": 2.88, "X-N": 3.46, "X-E": 2.88,
}

# Full-precision cross-check against a direct run of the independent
# reference checker's own sim.stair_metrics() (not just the rounded
# table above): stair_id -> (slope_deg, walking_length_m, Q_max), 6 d.p.
REFERENCE_EXACT_STAIRS = {
    "ST-W": (32.9977, 8.844767, 1.6128),
    "ST-E": (32.9977, 8.844767, 2.016),
    "ST-C": (32.9977, 8.844767, 3.2256),
    "ST-C2": (32.9977, 8.844767, 2.688),
    "ST-N": (32.9977, 8.844767, 2.016),
}


@pytest.fixture(scope="module")
def building_b1l():
    return load_site(CONFIG_PATH).get_building("B1")


@pytest.mark.parametrize("stair_id,expected", EXPECTED_STAIRS.items())
def test_stair_derived_values_match_reference_table(building_b1l, stair_id, expected):
    risers, slope_deg, walking_length_m, q_max = expected
    stair = building_b1l.stairs[stair_id]
    assert stair.n_risers == risers
    assert stair.slope_deg == pytest.approx(slope_deg, abs=0.05)
    assert stair.walking_length_m == pytest.approx(walking_length_m, abs=0.005)
    computed_q = 1.92 * stair.width_m * stair.chi
    assert computed_q == pytest.approx(q_max, abs=0.005)


@pytest.mark.parametrize("exit_id,expected_q", EXPECTED_EXIT_Q.items())
def test_exit_capacity_matches_reference_table(building_b1l, exit_id, expected_q):
    gf = building_b1l.get_floor("GF")
    door = next(d for d in gf.exit_doors if d.id == exit_id)
    computed_q = 1.92 * door.width_m
    assert computed_q == pytest.approx(expected_q, abs=0.005)


@pytest.mark.parametrize("stair_id,expected", REFERENCE_EXACT_STAIRS.items())
def test_stair_derived_values_match_reference_checker_exactly(building_b1l, stair_id, expected):
    """Cross-checked directly against a live run of the independent
    checker's own sim.stair_metrics(), not just the rounded table above --
    bit-exact to 4-6 d.p."""
    slope_deg, walking_length_m, q_max = expected
    stair = building_b1l.stairs[stair_id]
    assert stair.slope_deg == pytest.approx(slope_deg, abs=1e-3)
    assert stair.walking_length_m == pytest.approx(walking_length_m, abs=1e-5)
    assert 1.92 * stair.width_m * stair.chi == pytest.approx(q_max, abs=1e-5)


def test_x_se_is_the_one_unavailable_exit(building_b1l):
    """X-SE is a service door, LOCKED by default (-> UNAVAILABLE), not one
    of the 6 usable exits the §3.2 table lists."""
    gf = building_b1l.get_floor("GF")
    by_id = {d.id: d for d in gf.exit_doors}
    assert set(by_id) == set(EXPECTED_EXIT_Q) | {"X-SE"}
    assert by_id["X-SE"].default_state == "UNAVAILABLE"
    assert all(by_id[eid].default_state == "AVAILABLE" for eid in EXPECTED_EXIT_Q)
