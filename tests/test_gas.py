"""Gas dispersion model (dms/gas/*) and gas-aware A* on the graph-RL
navigation graph (dms/rl/graph_astar.py, dms/gas/routing.py)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dms.gas.alarm import EMERGENCY, NORMAL, WARNING, AlarmConfig, AlarmSystem, Detector
from dms.gas.dispersion import (
    PuffModel,
    ReleaseSource,
    mg_m3_to_ppm,
    sigma_y,
    sigma_z,
    steady_plume_mg_m3,
    virtual_distance,
)
from dms.gas.indoor import BoxModel, MapVentilationState, Ventilation, zone_ach
from dms.gas.meteo import Weather, WeatherSchedule, stability_class
from dms.gas.routing import GasAwarePlanner, RoutingParams
from dms.gas.scenario import load_scenario
from dms.gas.simulate import DEFAULT_SCENARIO, A_SAFE, Simulation, quantize
from dms.gas.zones import DOOR, EXIT, ZONE, NodeZones
from dms.rl.graph_astar import GraphAStar
from dms.rl.graph_env import GraphEvacEnv
from dms.rl.graph_q import graph_walker_optimum

DOMAIN = (0.0, 0.0, 160.0, 140.0)


# ------------------------------------------------------------------ physics


def test_ppm_conversion_matches_reference():
    # design doc: 1 ppm ≈ 1.39 mg/m³ at 25 °C, 1 atm
    assert mg_m3_to_ppm(1.39) == pytest.approx(1.0, rel=0.01)
    assert mg_m3_to_ppm(1.0, temperature_c=40.0) > mg_m3_to_ppm(1.0)  # warmer air, more ppm per mg


@pytest.mark.parametrize("terrain", ["urban", "rural"])
@pytest.mark.parametrize("cls", list("ABCDEF"))
def test_briggs_sigmas_increase_and_invert(cls, terrain):
    x = np.geomspace(1, 5000, 50)
    sy, sz = sigma_y(x, cls, terrain), sigma_z(x, cls, terrain)
    assert np.all(np.diff(sy) > 0) and np.all(np.diff(sz) > 0)
    assert virtual_distance(sy, cls, terrain, 0) == pytest.approx(x, rel=1e-3)
    assert virtual_distance(sz, cls, terrain, 1) == pytest.approx(x, rel=1e-3)


def test_stability_table():
    assert stability_class(1.5, "strong") == "A"
    assert stability_class(5.0, "night_clear") == "D"
    assert stability_class(2.5, "night_clear") == "F"


@pytest.mark.parametrize("cls", ["B", "D", "F"])
def test_puffs_converge_to_steady_plume_under_constant_wind(cls):
    """Design doc: the steady plume is the verification check — under constant
    wind the puff sum must agree with it."""
    w = Weather(wind_speed=2.5, wind_from_deg=135, stability=cls)
    src = ReleaseSource("s", 140, 40, height_m=1.0, rate_gps=50.0)
    pm = PuffModel([src], domain=DOMAIN)
    for i in range(300):
        pm.step(float(i), 1.0, w)
    d = np.array([20.0, 40.0, 80.0, 120.0])
    ex, ey = -math.sqrt(0.5), math.sqrt(0.5)  # downwind = north-west
    for off in (0.0, 4.0):  # centreline and 4 m crosswind
        px, py = 140 + d * ex + off * -ey, 40 + d * ey + off * ex
        ratio = pm.concentration_mg_m3(px, py) / steady_plume_mg_m3(px, py, src, w)
        assert np.all(np.abs(ratio - 1) < 0.08), ratio


def test_plume_follows_the_wind():
    for from_deg, (dx, dy) in [(270, (1, 0)), (0, (0, -1)), (90, (-1, 0))]:
        pm = PuffModel([ReleaseSource("s", 80, 70, rate_gps=10.0)], domain=DOMAIN)
        for i in range(60):
            pm.step(float(i), 1.0, Weather(wind_speed=3.0, wind_from_deg=from_deg))
        down = pm.concentration_mg_m3([80 + 30 * dx], [70 + 30 * dy])[0]
        up = pm.concentration_mg_m3([80 - 30 * dx], [70 - 30 * dy])[0]
        assert down > 1.0 and up < 1e-6 * down


def test_mass_is_conserved_dry_and_washed_out_in_rain():
    src = ReleaseSource("s", 80, 70, rate_gps=10.0, duration_s=20.0)
    big = (-1e4, -1e4, 1e4, 1e4)  # nothing leaves the domain
    dry = PuffModel([src], domain=big)
    for i in range(40):
        dry.step(float(i), 1.0, Weather(wind_speed=2.0))
    assert dry.released_mg == pytest.approx(10.0 * 20.0 * 1e3)
    assert dry.airborne_mg == pytest.approx(dry.released_mg)

    src2 = ReleaseSource("s", 80, 70, rate_gps=10.0, duration_s=20.0)
    wet = PuffModel([src2], domain=big)
    for i in range(40):
        wet.step(float(i), 1.0, Weather(wind_speed=2.0, precipitation_mmh=10.0))
    assert wet.deposited_mg > 0
    assert wet.airborne_mg + wet.deposited_mg == pytest.approx(wet.released_mg)


def test_instantaneous_release_is_one_puff_that_drifts():
    src = ReleaseSource("tank", 40, 70, kind="instantaneous", mass_kg=5.0, start_s=2.0)
    pm = PuffModel([src], domain=DOMAIN)
    for i in range(20):
        pm.step(float(i), 1.0, Weather(wind_speed=2.0, wind_from_deg=270))
    assert pm.n_puffs == 1
    assert pm.x[0] == pytest.approx(40 + 2.0 * 18, abs=1e-6)  # released at 2 s, 18 s of drift east


def test_weather_schedule_veers_the_short_way():
    sched = WeatherSchedule.from_config([{"t": 0, "wind_from_deg": 350, "stability": "D"},
                                         {"t": 10, "wind_from_deg": 10, "stability": "F"}])
    assert min(sched.at(5).wind_from_deg, 360 - sched.at(5).wind_from_deg) == pytest.approx(0.0, abs=1e-9)  # via north
    assert sched.at(5).stability == "D" and sched.at(10).stability == "F"


def test_box_model_lags_and_stays_below_outdoor():
    box = BoxModel(1)
    c_out = np.array([100.0])
    ach = np.array([2.0])
    for _ in range(600):
        box.step(1.0, c_out, ach)
    expected = 100.0 * (1 - math.exp(-2.0 / 3600 * 600))
    assert box.c[0] == pytest.approx(expected, rel=1e-9)
    assert box.c[0] < 100.0


def test_ventilation_state_sets_ach():
    v = Ventilation()
    by_exit = {7: [0]}
    on = zone_ach(3, v, MapVentilationState(hvac_on=True), by_exit)
    off = zone_ach(3, v, MapVentilationState(hvac_on=False, sealed_zones=frozenset({2}), open_exits=frozenset({7})), by_exit)
    assert on.tolist() == [v.ach_hvac_on] * 3
    assert off[1] == v.ach_hvac_off and off[2] == v.ach_sealed
    assert off[0] == v.ach_hvac_off + v.ach_door_open


def test_alarm_latches_only_after_sustained_reading():
    alarm = AlarmSystem([Detector("a", 0, 0), Detector("b", 0, 0)], AlarmConfig(alarm_ppm=30, warning_ppm=10, hold_s=5))
    assert alarm.update(1, 1.0, np.array([12.0, 0.0])) == WARNING
    for t in range(2, 6):  # 4 s above 30 ppm: not yet
        assert alarm.update(t, 1.0, np.array([40.0, 0.0])) == WARNING
    assert alarm.update(6, 1.0, np.array([5.0, 0.0])) == NORMAL  # dipped: the hold timer resets
    for t in range(7, 12):
        state = alarm.update(t, 1.0, np.array([0.0, 35.0]))
    assert state == EMERGENCY and alarm.emergency_t == 11 and alarm.trigger_detector == "b"
    assert alarm.update(12, 1.0, np.zeros(2)) == EMERGENCY  # latched


# ------------------------------------------------------------------ graph + A*


@pytest.fixture(scope="module")
def genv(site):
    return GraphEvacEnv(site, "B1")


@pytest.fixture(scope="module")
def zones(genv, site, building):
    return NodeZones.from_graph_env(genv, site, building)


def test_astar_matches_the_graph_walker_optimum(genv):
    """Without gas costs A* is exactly the best graph+walker route."""
    opt = graph_walker_optimum(genv)
    astar = GraphAStar(genv)
    rng = np.random.default_rng(0)
    for s in rng.choice(genv.start_states, size=400, replace=False):
        route = astar.plan(int(s))
        assert route is not None
        assert route.time_s == pytest.approx(opt[s], abs=1e-9)
        assert genv.terminal[route.nodes[-1]]


def test_astar_route_replays_through_the_env(genv):
    astar = GraphAStar(genv)
    s = int(genv.start_states[1234])
    route = astar.plan(s)
    t = 0.0
    for v in route.nodes[1:]:
        slot = int(np.flatnonzero(genv.neighbours[genv.node(s)] == v)[0])
        s, r, _ = genv.step(s, slot)
        t -= r
    assert s == route.cells[-1] and t == pytest.approx(route.time_s)


def test_every_graph_node_gets_a_gas_rule(genv, zones):
    kinds = set(zones.node_kind.tolist())
    assert kinds == {ZONE, DOOR, EXIT}
    assert all(zones.door_zones[k] for k in np.flatnonzero(zones.node_kind == DOOR))
    assert len(zones.exit_ap) == int(genv.terminal.sum())
    assert dict(zip([genv.nav.node(genv.node_ids[k]).ref_id for k in zones.exit_nodes], zones.exit_ap)) == {
        "X-S": "AP-S", "X-N": "AP-N", "X-W": "AP-W", "X-E": "AP-E", "X-NW": "AP-W"}


def test_planner_avoids_gassed_nodes(genv, zones):
    planner = GasAwarePlanner(genv, RoutingParams(lookahead=False))
    blind = GasAwarePlanner(genv, gas_aware=False)
    start = int(genv.cells_of_node[genv.node_ids.index("GF:R:G-03")][0])
    clean = np.zeros(genv.n_nodes)
    fastest = blind.plan(start, 0.0, clean, None).route
    first_exit = fastest.nodes[-1]

    gassy = clean.copy()
    gassy[first_exit] = 50.0  # its usual exit is in the plume
    plan = planner.plan(start, 0.0, gassy, None)
    assert plan.tier == 0 and plan.route.nodes[-1] != first_exit
    assert max(gassy[k] for k in plan.route.nodes[1:]) < planner.params.avoid_ppm
    assert plan.route.time_s >= fastest.time_s

    gassy[zones.exit_nodes] = 50.0  # every exit gassed: nothing gas-free, least-exposure fallback
    assert planner.plan(start, 0.0, gassy, None, max_tier=0) is None
    fallback = planner.plan(start, 0.0, gassy, None)
    assert fallback is not None and fallback.tier == 1


# ------------------------------------------------------------------ end to end


@pytest.fixture(scope="module")
def runs():
    sc = load_scenario(DEFAULT_SCENARIO)
    env = sc.make_env()
    return Simulation(sc, env).run(), Simulation(sc, env, gas_aware=False).run()


def test_scenario_alarm_and_response(runs):
    aware, _ = runs
    s = aware.summary()
    assert s["alarm_t"] is not None and s["alarm_t"] < 30
    assert s["hvac_off_t"] == s["alarm_t"]
    assert not any(a.t_alert is not None and a.t_alert < s["alarm_t"] and a.peak < 10 for a in aware.agents)


def test_gas_aware_agents_avoid_the_gas(runs):
    aware, blind = runs
    p = aware.sc.routing
    assert all(a.state == A_SAFE for a in aware.agents)
    for a in aware.agents:
        assert all(ppm < p.avoid_ppm for _, _, ppm in a.entered), a.room  # never stepped into a gassed node
        assert a.exit_ppm < p.avoid_ppm, a.room  # left through a clean exit
        assert a.peak < p.avoid_ppm
    assert sum(a.dose for a in aware.agents) < 0.01 * sum(b.dose for b in blind.agents)
    assert any(b.exit_ppm >= p.avoid_ppm for b in blind.agents)  # the comparison actually had gas in the way


def test_runs_are_deterministic(runs):
    aware, _ = runs
    again = Simulation(aware.sc, aware.env).run()
    assert [a.trace_s for a in again.agents] == [a.trace_s for a in aware.agents]
    assert np.array_equal(again.rec_node_ppm[-1], aware.rec_node_ppm[-1])


def test_quantize_roundtrip():
    ppm = np.array([0.0, 0.05, 0.1, 1.0, 10.0, 30.0, 500.0, 5000.0])
    q = quantize(ppm)
    assert q[0] == 0 and q[1] == 0 and q[2] == 1 and q[-1] == 255
    back = 0.1 * 10 ** ((q[3:7].astype(float) - 1) / 254 * math.log10(2000 / 0.1))
    assert back == pytest.approx(ppm[3:7], rel=0.03)
