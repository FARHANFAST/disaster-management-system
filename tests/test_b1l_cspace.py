"""spec §3.4: body-clear reachability -- Eikonal fields solved on
configuration space (walkable cells at least one "standard" body-class
clearance radius from any wall/obstacle, via the distance-
transform clearance_domain, so a route never passes through a gap a body
can't fit through), for every one of B1-L's 11 scenarios, on both the
belief and truth knowledge states where they differ (spec §5's
knowledge split)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dms.fields.eikonal import FieldCache
from dms.geometry.loader import load_site
from dms.grid.layers import H_FINE
from dms.grid.raster import rasterize_floor
from dms.scenario.agent_defaults import load_agent_defaults
from dms.scenario.presets import overlays_from_scenario

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"
MAX_UNREACHABLE_FRAC = 0.02  # <=2% of a room's body-clear cells may be unreachable corners
STANDARD_RADIUS_M = load_agent_defaults().body_classes["standard"].clearance_radius_m


@pytest.fixture(scope="module")
def site_b1l():
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="module")
def building_b1l(site_b1l):
    return site_b1l.get_building("B1")


@pytest.fixture(scope="module")
def grids_fine(building_b1l):
    return {fid: rasterize_floor(building_b1l, building_b1l.get_floor(fid), H_FINE) for fid in building_b1l.floor_ids()}


def _body_clear_problems(building, grids, overlay) -> list[tuple[str, str, str]]:
    """(floor, space, reason) for every non-void space whose body-clear
    coverage is insufficient under this overlay."""
    cache = FieldCache()
    fields = {
        fid: cache.get_field(building, fid, grids, overlay=overlay, clearance_radius_m=STANDARD_RADIUS_M)
        for fid in building.floor_ids()
    }
    problems = []
    for fid, floor in building.floors.items():
        grid = grids[fid]
        field = fields[fid]
        for idx, sid in enumerate(grid.space_ids):
            space = floor.spaces[sid]
            if space.kind == "void":
                continue
            if space.kind == "stair" and sid in overlay.blocked_stairs:
                continue  # deliberately closed off by this scenario -- zero cells is correct, not a problem
            cells = (grid.space_id == idx) & field.domain
            if not cells.any():
                problems.append((fid, sid, "no body-clear cells"))
                continue
            finite = np.isfinite(field.T.data[cells])
            if not finite.any():
                problems.append((fid, sid, "no route to any open exit"))
            elif (~finite).mean() > MAX_UNREACHABLE_FRAC:
                problems.append((fid, sid, f"{(~finite).mean():.0%} of room cut off"))
    return problems


@pytest.mark.slow
@pytest.mark.parametrize("scenario_name", [
    "baseline", "main_exit_blocked", "north_exit_blocked_unknown", "main_stair_lost", "compound",
    "slide23_A", "slide23_B", "slide23_C", "slide23_D", "two_stairs_lost", "three_stairs_lost",
])
def test_zero_body_clear_problems_in_every_scenario(building_b1l, grids_fine, scenario_name):
    """Every one of B1-L's 11 scenarios: 0 problems, on both belief and
    truth (they coincide except for the 2 scenarios with a BLOCKED_UNKNOWN
    exit, where both must still independently have 0 problems)."""
    site = load_site(CONFIG_PATH)
    scenario_preset = site.scenarios[scenario_name]
    belief_overlay, truth_overlay = overlays_from_scenario(scenario_preset, building_b1l)

    belief_problems = _body_clear_problems(building_b1l, grids_fine, belief_overlay)
    truth_problems = _body_clear_problems(building_b1l, grids_fine, truth_overlay)

    assert belief_problems == [], f"{scenario_name} (belief): {belief_problems}"
    assert truth_problems == [], f"{scenario_name} (truth): {truth_problems}"


@pytest.mark.slow
def test_belief_routes_toward_unknown_blocked_exit_truth_routes_away(building_b1l, grids_fine):
    """The specific divergence this test confirms directly: in
    north_exit_blocked_unknown, belief T on N-STUB is finite and decreases
    toward X-N; truth T there leads away from X-N (increases toward it)."""
    site = load_site(CONFIG_PATH)
    preset = site.scenarios["north_exit_blocked_unknown"]
    belief_overlay, truth_overlay = overlays_from_scenario(preset, building_b1l)

    from dms.fields.eikonal import compute_floor_field

    gf = building_b1l.get_floor("GF")
    grid = grids_fine["GF"]
    belief_field = compute_floor_field(building_b1l, gf, grid, overlay=belief_overlay, clearance_radius_m=STANDARD_RADIUS_M)
    truth_field = compute_floor_field(building_b1l, gf, grid, overlay=truth_overlay, clearance_radius_m=STANDARD_RADIUS_M)

    sidx = grid.space_ids.index("N-STUB")
    cells = grid.space_id == sidx
    y0 = grid.transform.origin_local[1]
    h = grid.transform.h

    def y_sorted_T(field):
        rows, cols = np.nonzero(cells & field.domain)
        ys = y0 + (rows + 0.5) * h
        order = np.argsort(ys)
        return np.asarray(field.T.data[rows, cols])[order]

    belief_T = y_sorted_T(belief_field)
    truth_T = y_sorted_T(truth_field)
    assert np.isfinite(belief_T).all()
    assert np.isfinite(truth_T).all()
    # X-N is on the north wall (high y): belief decreases toward it, truth increases
    assert belief_T[-1] < belief_T[0]
    assert truth_T[-1] > truth_T[0]


def test_trolleys_are_against_the_wall_and_each_leave_one_lane_clear(building_b1l):
    """The actual placed trolleys (C-E, C-X3) sit against a wall, not
    mid-corridor -- each must leave exactly one lane >= 1.0 m clear, not
    the mid-corridor "two gaps under 0.75 m" case that was already
    identified and fixed as a bug in this fixture."""
    from dms.geometry.corridor_width import effective_width, lanes_for_segment
    from dms.geometry.validate import _iter_corridor_segments
    from dms.scenario.agent_defaults import load_agent_defaults

    min_lane_width_m = load_agent_defaults().min_lane_width_m
    gf = building_b1l.get_floor("GF")
    trolleys = {o.space_id: o for o in gf.obstacles.values() if o.type == "trolley"}
    assert set(trolleys) == {"C-E", "C-X3"}

    checked = set()
    for floor, space, axis, short_lo, short_hi, lo, hi in _iter_corridor_segments(building_b1l):
        if floor.id != "GF" or space.id not in trolleys:
            continue
        trolley = trolleys[space.id]
        tc = trolley.rect.center[0] if axis == "x" else trolley.rect.center[1]
        if not (lo <= tc < hi):
            continue
        lanes = lanes_for_segment(short_lo, short_hi, lo, hi, axis, [trolley])
        wide_lanes = [w for w in lanes if w >= min_lane_width_m]
        assert len(wide_lanes) == 1, f"{space.id}: expected exactly one clear lane, got {lanes}"
        assert effective_width(lanes, min_lane_width_m) >= min_lane_width_m
        checked.add(space.id)
    assert checked == {"C-E", "C-X3"}
