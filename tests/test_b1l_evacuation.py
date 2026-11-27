"""spec §4/§3.5: the vectorized capacity-enforced evacuation walker.
Exercises dms/scenario/evacuation.py against B1-L's own baseline/compound
scenarios. The 25-run full stress test lives in test_b1l_stress.py
(@pytest.mark.slow); these are the fast, per-mechanism checks."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dms.geometry.loader import load_site
from dms.graph.builder import F_S_LIT
from dms.scenario.evacuation import (
    DONE,
    IN_REFUGE,
    MOVE,
    Agent,
    build_world,
    sample_agents,
    simulate,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"


@pytest.fixture(scope="module")
def building():
    return load_site(CONFIG_PATH).get_building("B1")


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG_PATH)


def test_exit_capacities_match_f_s_w_eff(site, building):
    world = build_world(site, building, "baseline")
    for e in world.exits:
        expected = F_S_LIT * e.door.width_m
        assert e.Q == pytest.approx(expected)


def test_stair_capacities_match_f_s_w_chi(site, building):
    world = build_world(site, building, "baseline")
    for s in world.stairs:
        stair = building.stairs[s.id]
        assert s.Q == pytest.approx(F_S_LIT * stair.width_m * stair.chi)


def test_compound_blocks_x_s_and_st_c_and_marks_x_n_unknown(site, building):
    world = build_world(site, building, "compound")
    by_id = {e.id: e for e in world.exits}
    assert by_id["X-S"].open_true is False
    assert by_id["X-N"].unknown is True
    assert by_id["X-N"].open_true is False  # BLOCKED_UNKNOWN is a physical closure, just not a known one
    assert "ST-C" in world.blocked_stair_ids


def test_baseline_small_population_all_arrive(site, building):
    world = build_world(site, building, "baseline")
    agents = sample_agents(world, "day", seed=7, N=80)
    res = simulate(world, agents, t_end=1500.0, seed=3)
    assert res.n_stuck == 0
    assert (res.frames[-1].state == DONE).all()


def test_exit_token_bucket_never_exceeds_q_in_any_20_person_window(site, building):
    """For every exit, the peak flow
    over any 20-person window is <= Q = 1.92 * w_eff."""
    world = build_world(site, building, "baseline")
    agents = sample_agents(world, "day", seed=7)
    res = simulate(world, agents, t_end=1500.0, seed=5)
    for k, e in enumerate(world.exits):
        used = np.where(res.exit_used == k)[0]
        if len(used) < 21:
            continue
        times = np.sort(res.t_exit[used])
        window = times[20:] - times[:-20]  # time to pass 20 agents, sliding
        window = window[window > 0]
        if len(window) == 0:
            continue
        peak_rate = (20 / window).max()
        assert peak_rate <= e.Q + 1e-6, f"{e.id}: peak {peak_rate:.3f} p/s > Q {e.Q:.3f} p/s"


def test_same_seed_gives_identical_frames(site, building):
    world = build_world(site, building, "baseline")
    agents = sample_agents(world, "day", seed=7, N=60)
    res1 = simulate(world, agents, t_end=600.0, seed=11)
    res2 = simulate(world, agents, t_end=600.0, seed=11)
    assert len(res1.frames) == len(res2.frames)
    for f1, f2 in zip(res1.frames, res2.frames):
        assert f1.t == f2.t
        assert np.array_equal(f1.pos, f2.pos)
        assert np.array_equal(f1.state, f2.state)
        assert np.array_equal(f1.know, f2.know)
    assert np.array_equal(res1.exit_used, res2.exit_used)


def test_discovery_and_reroute_in_north_exit_blocked_unknown(site, building):
    world = build_world(site, building, "north_exit_blocked_unknown")
    agents = sample_agents(world, "day", seed=7)
    res = simulate(world, agents, t_end=1500.0, seed=9)
    assert res.n_stuck - res.n_awaiting_refuge == 0
    assert res.discovered.sum() > 100
    assert len(res.events) == 1
    assert res.events[0]["exit"] == "X-N"


def test_compound_full_population_zero_stuck_and_reroutes(site, building):
    world = build_world(site, building, "compound")
    agents = sample_agents(world, "day", seed=7)
    res = simulate(world, agents, t_end=1500.0, seed=1)
    assert res.n_stuck - res.n_awaiting_refuge == 0
    assert res.discovered.sum() > 100
    by_id = {e.id: k for k, e in enumerate(world.exits)}
    assert int((res.exit_used == by_id["X-S"]).sum()) == 0
    assert int((res.exit_used == by_id["X-N"]).sum()) == 0
    stair_names = [s.id for s in world.stairs]
    assert int((res.stair_used == stair_names.index("ST-C")).sum()) == 0


def test_agents_never_cross_floors_in_separation_queries(site, building):
    """Density/separation must be
    computed strictly per floor. Smoke-checks this by asserting GF and FF
    agents placed at the exact same local (x, y) never influence each
    other's position (a shared-tree bug would push them apart)."""
    world = build_world(site, building, "baseline")
    from dms.scenario.evacuation import Agent

    standard = world.agent_defaults.body_classes["standard"]
    x, y = 10.0, 5.0
    agents = [
        Agent(floor_idx=0, x=x, y=y, persona="adult_m", v0=1.2, room="dummy", t_pre=9999.0,
              body_class="standard", clearance_radius_m=standard.clearance_radius_m,
              can_use_stairs=standard.can_use_stairs, stair_speed_factor=standard.stair_speed_factor),
        Agent(floor_idx=1, x=x, y=y, persona="adult_m", v0=1.2, room="dummy", t_pre=9999.0,
              body_class="standard", clearance_radius_m=standard.clearance_radius_m,
              can_use_stairs=standard.can_use_stairs, stair_speed_factor=standard.stair_speed_factor),
    ]
    res = simulate(world, agents, t_end=1.0, seed=1)
    fr = res.frames[0]
    assert np.allclose(fr.pos[0], fr.pos[1])  # no separation push since they're on different floors


def test_wheelchair_agent_on_ff_reaches_a_refuge_and_is_safe_not_stuck(building):
    """B1-L's FF declares 3 refuges (one per
    enclosed stair) -- a wheelchair-class agent placed on FF must target
    the nearest one, arrive, and be reported as `in_refuge` (safe, awaiting
    assisted evacuation), counted in neither `n_stuck` nor
    `n_awaiting_refuge`. B1-L's own persona_mix declares no body_class_mix
    override (every real sampled agent today is "standard"), so this
    scenario is built by hand here, not by sampling the building's own
    population."""
    site = load_site(CONFIG_PATH)
    world = build_world(site, building, "baseline")
    wc = world.agent_defaults.body_classes["wheelchair"]
    std = world.agent_defaults.body_classes["standard"]
    ff = building.get_floor("FF")
    room = next(sp for sp in ff.spaces.values() if sp.kind == "room" and sp.use)
    x, y = room.rect.center

    agents = [
        Agent(floor_idx=1, x=x, y=y, persona="mobility_impaired", v0=0.5, room=room.id, t_pre=0.0,
              body_class="wheelchair", clearance_radius_m=wc.clearance_radius_m,
              can_use_stairs=wc.can_use_stairs, stair_speed_factor=wc.stair_speed_factor),
        Agent(floor_idx=1, x=x + 1.0, y=y, persona="adult_m", v0=1.2, room=room.id, t_pre=0.0,
              body_class="standard", clearance_radius_m=std.clearance_radius_m,
              can_use_stairs=std.can_use_stairs, stair_speed_factor=std.stair_speed_factor),
    ]
    res = simulate(world, agents, t_end=1500.0, seed=1)

    assert res.n_in_refuge == 1
    assert res.in_refuge.tolist() == [True, False]
    assert res.n_awaiting_refuge == 0, "reached a refuge -- must not also read as awaiting one"
    assert res.n_stuck == 0, "in_refuge is safe, not stuck"
    assert res.frames[-1].state[0] == IN_REFUGE
    assert res.frames[-1].state[1] == DONE  # the standard-class agent is unaffected


def test_wheelchair_agent_on_a_floor_with_no_refuge_is_reported_as_awaiting_refuge_not_hidden():
    """Generality check: a floor with no refuge at all
    must still behave consistently -- the agent
    genuinely can't reach anything on that floor, still counts honestly in
    n_stuck (never silently excluded), and is separately, visibly reported
    via n_awaiting_refuge/awaiting_refuge, not silently treated as
    in_refuge. Refuges are generated automatically by `load_site` for
    every ENCLOSED stair on every upper floor of any building --
    two_storey.yaml no longer serves as a "no refuge"
    example, so this builds a minimal 2-floor building by hand with a
    single OPEN (unenclosed) stair, which deliberately never gets a refuge
    generated for it (an open stair offers no fire/smoke protection)."""
    from dms.geometry.building import Building, Floor, Space
    from dms.geometry.primitives import Door, Rect, Stair
    from dms.geometry.site import AssemblyPoint, Site

    gf = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    gf.spaces["R1"] = Space(id="R1", floor="GF", name="R1", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0), use="office")
    gf.spaces["ST-A"] = Space(id="ST-A", floor="GF", name="ST-A", kind="stair", rect=Rect(10.0, 0.0, 14.0, 10.0))
    gf.doors["R1-ST-A"] = Door(id="R1-ST-A", floor="GF", axis="x", coord=10.0, lo=4.0, hi=6.0,
                                 kind="interior", space_a="R1", space_b="ST-A")
    gf.doors["X-1"] = Door(id="X-1", floor="GF", axis="x", coord=0.0, lo=4.0, hi=5.5,
                             kind="exit", default_state="AVAILABLE", space_a="R1", space_b="outside")

    ff = Floor(id="FF", z=4.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    ff.spaces["R2"] = Space(id="R2", floor="FF", name="R2", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0), use="office")
    ff.spaces["ST-A"] = Space(id="ST-A", floor="FF", name="ST-A", kind="stair", rect=Rect(10.0, 0.0, 14.0, 10.0))
    ff.doors["R2-ST-A"] = Door(id="R2-ST-A", floor="FF", axis="x", coord=10.0, lo=4.0, hi=6.0,
                                 kind="interior", space_a="R2", space_b="ST-A")

    building = Building(id="NO-REFUGE-FIXTURE", origin=(30.0, 30.0), rotation_deg=0.0, wall_thickness_m=0.2)
    building.floors["GF"] = gf
    building.floors["FF"] = ff
    building.stairs["ST-A"] = Stair(
        id="ST-A", landings={"GF": "ST-A", "FF": "ST-A"}, width_m=1.2, enclosed=False,
        riser_m=0.17, tread_m=0.28, flights=2, floor_to_floor_m=4.0,
        footprints={"GF": Rect(10.0, 0.0, 14.0, 10.0), "FF": Rect(10.0, 0.0, 14.0, 10.0)},
    )

    site = Site(name="no-refuge-fixture-site", width_m=100.0, height_m=100.0)
    site.buildings["NO-REFUGE-FIXTURE"] = building
    site.assembly_points["AP-1"] = AssemblyPoint(id="AP-1", x=20.0, y=35.0, diameter_m=15.0)

    from dms.geometry.refuge_gen import apply_generated_refuges
    placements = apply_generated_refuges(building)
    assert placements == [], "an open/unenclosed stair must never even be attempted for a generated refuge"
    assert ff.refuges == {}

    world = build_world(site, building, "baseline")
    wc = world.agent_defaults.body_classes["wheelchair"]

    agents = [
        Agent(floor_idx=1, x=5.0, y=5.0, persona="mobility_impaired", v0=0.5, room="R2", t_pre=0.0,
              body_class="wheelchair", clearance_radius_m=wc.clearance_radius_m,
              can_use_stairs=wc.can_use_stairs, stair_speed_factor=wc.stair_speed_factor),
    ]
    res = simulate(world, agents, t_end=300.0, seed=1)

    assert res.n_stuck == 1, "no refuge exists -- must still count as stuck, never silently excluded"
    assert res.n_awaiting_refuge == 1
    assert res.n_in_refuge == 0
    assert res.frames[-1].state[0] == MOVE  # never reaches DONE or IN_REFUGE without a refuge


def test_stair_capture_is_distance_based_not_exact_cell_only(building):
    """Found via stress testing that right next
    to a stair, the belief/truth T field is nearly flat (the landing is
    where it bottoms out), so a walker can orbit the stair's own footprint
    under the 24-heading-walk + jitter without ever truncating to a cell
    stair_id itself marks -- stair capture must work like the exit gate's
    own distance-based capture, not require an exact cell hit."""
    site = load_site(CONFIG_PATH)
    world = build_world(site, building, "baseline")
    fk = world.floor_kinematics(world.agent_defaults.body_classes["standard"].clearance_radius_m)["FF"]

    stair_cells = (fk.grid.stair_id >= 0) & fk.move
    assert stair_cells.any()
    # every movable stair cell is its own nearest stair cell, at distance 0
    assert np.all(fk.stair_capture_dist_m[stair_cells] == 0.0)
    assert np.array_equal(fk.stair_capture_id[stair_cells], fk.grid.stair_id[stair_cells])

    # a cell just outside the footprint, within the configured capture
    # radius, must resolve to that same nearby stair, not -1.
    rows, cols = np.nonzero(stair_cells)
    r0, c0 = rows[0], cols[0]
    h = fk.grid.transform.h
    capture_cells = round(world.agent_defaults.kinematics.stair_capture_m / h)
    for dr, dc in ((capture_cells, 0), (0, capture_cells), (-capture_cells, 0), (0, -capture_cells)):
        rr, cc = r0 + dr, c0 + dc
        if not (0 <= rr < fk.move.shape[0] and 0 <= cc < fk.move.shape[1]):
            continue
        if fk.grid.stair_id[rr, cc] >= 0:
            continue  # still on a (different) stair cell, not a useful probe point
        if fk.stair_capture_dist_m[rr, cc] <= world.agent_defaults.kinematics.stair_capture_m:
            assert fk.stair_capture_id[rr, cc] == fk.grid.stair_id[r0, c0]


def test_stress_regression_meeting_404_main_stair_lost_no_longer_stalls(site, building):
    """The exact case the stair-capture fix above was written for: before
    the fix, this combination left one mobility_aid-class agent orbiting
    ST-W's footprint forever (confirmed via direct investigation, not
    assumed) -- a real, reproducible stall, not a rare random fluke."""
    world = build_world(site, building, "main_stair_lost")
    agents = sample_agents(world, "meeting", seed=404, N=None)
    res = simulate(world, agents, t_end=1500.0, seed=404)
    other_stuck = res.n_stuck - res.n_awaiting_refuge
    assert other_stuck == 0, f"{res.n_stuck} stuck, {res.n_awaiting_refuge} awaiting refuge, {other_stuck} other"
