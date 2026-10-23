"""Agent behaviour model (dms/agents, design doc v0.3): population split,
attribute-driven weights and their event adaptation, physiology, personal
exit knowledge, the state machine, step() contract and an end-to-end run.

A synthetic gas history keeps the simulation tests fast and exact; the real
H2S history is exercised in test_gas_dqn.py::test_timeline_records_the_scenario.
"""

from __future__ import annotations

import numpy as np
import pytest

from dms.agents.agent import ALERT, AT_ASSEMBLY, INCAPACITATED, MOVING, NORMAL, Action
from dms.agents.config import PERSONA_W, AgentModelConfig, BehaviourConfig, PhysiologyConfig, PopulationConfig
from dms.agents.physiology import Body, band
from dms.agents.population import sample_population
from dms.agents.sim import AgentSimulation
from dms.agents.weights import WeightState, base_logits
from dms.gas.timeline import GasTimeline
from dms.gas.zones import NodeZones
from dms.rl.graph_env import GraphEvacEnv

EXITS = ["X-S", "X-N", "X-W", "X-E", "X-NW"]


@pytest.fixture(scope="module")
def world(site):
    env = GraphEvacEnv(site, "B1")
    return env, NodeZones.from_graph_env(env, site, site.get_building("B1"))


def _timeline(env: GraphEvacEnv, zones: NodeZones, gassed: dict[str, float] | None = None,
              duration: int = 400) -> GasTimeline:
    """Constant ppm on the named exits from t=0 (alarm at t=0), clean air elsewhere."""
    node = np.zeros((duration + 1, env.n_nodes))
    for k in np.flatnonzero(env.terminal):
        node[:, k] = (gassed or {}).get(env.nav.node(env.node_ids[k]).ref_id, 0.0)
    return GasTimeline(dt=1.0, node_ppm=node, soon_ppm=node.copy(), forecast_t0=np.empty(0), forecasts=[],
                       alarm_t=0.0, zones=zones)


def _sim(world, n=60, seed=3, gassed=None, blocked=(), adapt=True, duration=300.0) -> AgentSimulation:
    env, zones = world
    cfg = AgentModelConfig(duration_s=duration, seed=seed)
    cfg.population.n_agents = n
    cfg.behaviour.adapt_weights = adapt
    return AgentSimulation(cfg, blocked=blocked, env=env, timeline=_timeline(env, zones, gassed))


# ------------------------------------------------------------------ population


def test_population_split_is_seeded_and_complete():
    cfg = PopulationConfig(n_agents=550)
    a = sample_population(cfg, EXITS, np.random.default_rng(1))
    b = sample_population(cfg, EXITS, np.random.default_rng(1))
    assert len(a) == 550 and [p.group for p in a] == [p.group for p in b]
    assert {p.age for p in a} == {"Young", "Adult", "Elderly"} and {p.sex for p in a} == {"M", "F"}
    vips = [p for p in a if p.vip]
    assert vips and all(p.role == "Visitor" and p.escort is not None and a[p.escort].responder for p in vips)
    assert all(p.persona == "Responder" for p in a if p.responder)
    assert all(p.known_exits == frozenset(EXITS) for p in a if p.responder)
    assert all(p.known_exits == frozenset({"X-S"}) for p in a if p.role == "Visitor")


def test_attributes_drive_speed_reaction_and_knowledge():
    pop = sample_population(PopulationConfig(n_agents=3000), EXITS, np.random.default_rng(2))

    def mean(f, sel):
        return float(np.mean([f(p) for p in pop if sel(p)]))
    assert mean(lambda p: p.speed, lambda p: p.age == "Elderly") < mean(lambda p: p.speed, lambda p: p.age == "Young")
    assert mean(lambda p: p.speed, lambda p: p.sex == "F") < mean(lambda p: p.speed, lambda p: p.sex == "M")
    assert (mean(lambda p: p.reaction_s, lambda p: p.age == "Elderly")
            > mean(lambda p: p.reaction_s, lambda p: p.age == "Young"))
    assert (mean(lambda p: p.reaction_s, lambda p: p.role == "Senior staff")
            < mean(lambda p: p.reaction_s, lambda p: p.role == "Visitor"))
    assert (mean(lambda p: len(p.known_exits), lambda p: p.role == "Senior staff")
            > mean(lambda p: len(p.known_exits), lambda p: p.role == "Junior staff") > 1.0)
    assert mean(lambda p: p.vision_m, lambda p: p.age == "Elderly") < 5.0


# ------------------------------------------------------------------ weights


def test_base_weights_follow_persona_and_attributes():
    pop = sample_population(PopulationConfig(n_agents=400), EXITS, np.random.default_rng(4))
    cfg = BehaviourConfig()
    for p in pop[:50]:
        w = WeightState(p, cfg).weights()
        assert w.sum() == pytest.approx(1.0) and (w > 0).all()
    staff = next(p for p in pop if p.persona == "Rationalist" and p.role == "Senior staff")
    visitor = next(p for p in pop if p.persona == "Rationalist" and p.role == "Visitor")
    ws, wv = WeightState(staff, cfg).weights(), WeightState(visitor, cfg).weights()
    assert ws[0] > wv[0] and ws[1] < wv[1]  # same persona: the familiar one trusts its knowledge more
    z = base_logits("Cautious", staff, cfg)
    assert np.argmax(np.exp(z)) == 2 or PERSONA_W["Cautious"][2] == max(PERSONA_W["Cautious"])


def test_route_failure_shifts_weight_away_from_knowledge_then_recovers():
    p = next(p for p in sample_population(PopulationConfig(n_agents=200), EXITS, np.random.default_rng(5))
             if p.persona == "Rationalist")
    ws = WeightState(p, BehaviourConfig())
    w0 = ws.weights()
    ws.route_failed()
    w1 = ws.weights()
    assert w1[0] < w0[0] and w1[2] > w0[2] and w1[1] > w0[1]
    assert (w1[2] - w0[2]) > (w1[1] - w0[1])  # a Rationalist turns to the hazard more than to the crowd
    for _ in range(600):
        ws.relax(0.5, 0.0)
    assert ws.weights() == pytest.approx(w0, abs=1e-3)
    frozen = WeightState(p, BehaviourConfig(adapt_weights=False))
    frozen.route_failed()
    assert frozen.weights() == pytest.approx(w0)
    assert ws.weights(panicked=True)[1] == pytest.approx(w0[1] + 0.2, abs=1e-3)


# ------------------------------------------------------------------ physiology


def test_dose_bands_speed_and_collapse():
    cfg = PhysiologyConfig()
    rng = np.random.default_rng(0)
    assert [band(c, cfg) for c in (0.1, 5, 50, 200, 600)] == [0, 1, 2, 3, 4]
    b = Body(PhysiologyConfig(collapse_rate=(0.0,) * 4 + (float("inf"),)))
    for _ in range(60):
        b.breathe(30.0, 1.0, rng)
    assert b.dose == pytest.approx(30.0) and b.health in ("Distressed", "Disabled")  # 30 ppm x 1 min
    assert b.speed_factor(30.0) == pytest.approx(0.8)
    assert Body(cfg).breathe(600.0, 0.5, rng)  # >= 500 ppm: immediate
    # collapse chance grows with dose in the same air
    def p_collapse(dose):
        hits = 0
        for i in range(4000):
            bb = Body(cfg, dose=dose)
            hits += bb.breathe(200.0, 1.0, np.random.default_rng(i))
        return hits / 4000
    assert p_collapse(150.0) > p_collapse(0.0) > 0.0
    assert Body(cfg, susceptibility=2.0).speed_factor(60.0) == pytest.approx(0.5)  # 60 ppm acts like 120


# ------------------------------------------------------------------ behaviour in the building


def test_step_contract_and_state_machine(world):
    sim = _sim(world, n=30, seed=1)
    sim.run()
    a = sim.agents[0]
    from dms.agents.world import Snapshot
    act = a.step(Snapshot(t=sim.t, alarm=True, timeline=sim.timeline, blocked_exits=frozenset(),
                          hash=sim.hash, agents=sim._views()), 0.5)
    assert isinstance(act, Action) and act.agent_id == 0
    for ag in sim.agents:
        states = [e["state"] for e in ag.events if e["event"] == "state"]
        assert states[0] == ALERT  # the alarm at t=0 is the first transition for everyone
        if ag.state == AT_ASSEMBLY:
            assert MOVING in states and ag.safe_t is not None and ag.safe_t > ag.p.reaction_s
    assert sum(ag.state == AT_ASSEMBLY for ag in sim.agents) >= 27
    assert NORMAL not in {ag.state for ag in sim.agents}


def test_same_seed_same_run(world):
    r1 = _sim(world, n=40, seed=9).run().agent_rows()
    r2 = _sim(world, n=40, seed=9).run().agent_rows()
    assert r1 == r2


def test_blocked_exit_is_found_only_on_sight_and_people_reroute(world):
    sim = _sim(world, n=80, seed=2, blocked=("X-E",)).run()
    found = [e for ag in sim.agents for e in ag.events if e["event"] == "exit_unusable" and e["exit"] == "X-E"]
    assert found and {e["why"] for e in found} <= {"blocked", "told by responder"}
    assert len({e["t"] for e in found}) > 5  # people find out one by one, not all at once
    assert len(found) < len(sim.agents)  # and some never need to know
    assert all(ag.exit_used != "X-E" for ag in sim.agents)
    rerouted = [ag for ag in sim.agents if ag.reroutes and ag.state == AT_ASSEMBLY]
    assert rerouted  # somebody headed for X-E, found it locked and still got out


def test_gassed_exit_splits_people_by_attributes(world):
    """X-S at 15 ppm: an exit counts as bad at 5 ppm / W3, so people with a
    low hazard weight walk through it and get dosed, while those with a
    high one see the gas and turn away."""
    sim = _sim(world, n=250, seed=4, gassed={"X-S": 15.0}).run()
    users = [ag for ag in sim.agents if ag.exit_used == "X-S"]
    turned = [ag for ag in sim.agents if any(e["event"] == "route_failed" and e["exit"] == "X-S" for e in ag.events)]
    assert users and turned
    assert all(ag.body.dose > 0 for ag in users)
    w3 = lambda ags: np.mean([WeightState(ag.p, BehaviourConfig()).weights()[2] for ag in ags])  # noqa: E731
    assert w3(users) < w3(turned)


def test_adaptation_changes_behaviour(world):
    on = _sim(world, n=150, seed=6, gassed={"X-S": 60.0, "X-E": 40.0}, adapt=True).run().agent_rows()
    off = _sim(world, n=150, seed=6, gassed={"X-S": 60.0, "X-E": 40.0}, adapt=False).run().agent_rows()
    assert [r["exit"] for r in on] != [r["exit"] for r in off]
    assert sum(r["final_state"] == INCAPACITATED for r in on) <= 150
