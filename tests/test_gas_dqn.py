"""Gas-aware DQN (dms/rl/gas_dqn.py): the state carries the gas readings,
the reward is the gas-aware A* edge cost with avoid / incapacitation
penalties, and a short training run learns to leave a gassed exit alone.

A synthetic gas history (one exit flooded from the start) keeps these fast
and exact; dms/gas/timeline.py is checked against the real model in one test.
"""

from __future__ import annotations

import numpy as np
import pytest

from dms.gas.scenario import load_scenario
from dms.gas.simulate import DEFAULT_SCENARIO
from dms.gas.timeline import GasTimeline
from dms.rl.agent_profile import sample_population
from dms.rl.dqn import DQNConfig
from dms.rl.gas_dqn import GasDQN, GasRewardConfig, enc
from dms.rl.graph_env import GraphEvacEnv


@pytest.fixture(scope="module")
def genv(site) -> GraphEvacEnv:
    return GraphEvacEnv(site, "B1")


def _exit(env: GraphEvacEnv, ref: str) -> int:
    return next(k for k in np.flatnonzero(env.terminal) if env.nav.node(env.node_ids[k]).ref_id == ref)


def flooded(env: GraphEvacEnv, ref: str, ppm: float, duration_s: int = 600) -> GasTimeline:
    """Constant `ppm` on exit `ref` from t=0, clean air everywhere else."""
    node = np.zeros((duration_s + 1, env.n_nodes))
    node[:, _exit(env, ref)] = ppm
    return GasTimeline(dt=1.0, node_ppm=node, soon_ppm=node.copy(), forecast_t0=np.empty(0), forecasts=[],
                       alarm_t=0.0, zones=None)


def _healthy(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    while True:
        p = sample_population(1, rng)[0]
        if p["healthStatus"] == "Healthy":
            return p


def _cell_next_to(env: GraphEvacEnv, exit_node: int) -> tuple[int, int]:
    """(cell, slot) for a neighbour of `exit_node` whose leg into it exists."""
    for u in range(env.n_nodes):
        hits = np.flatnonzero(env.neighbours[u] == exit_node)
        if hits.size and not env.terminal[u]:
            s = int(env.cells_of_node[u][0])
            if env.walker.leg(u, exit_node, s) is not None:
                return s, int(hits[0])
    raise AssertionError("exit has no reachable neighbour")


def test_state_has_gas_features_in_range(genv):
    tl = flooded(genv, "X-S", 90.0)
    dqn = GasDQN(genv, [tl], DQNConfig(episodes=1, device="cpu"))
    s, slot = _cell_next_to(genv, _exit(genv, "X-S"))
    x = dqn._x(genv.node(s), dqn.new_episode(_healthy(), tl, 0.0))
    assert x.shape == (dqn.n_x,) and x.dtype == np.float32
    gas = x[dqn.n_attr:]
    assert gas.min() >= 0.0 and gas.max() <= 1.0
    # the X-S slot reads 90 ppm now and soon; the agent's own node is clean
    assert gas[2 + slot] == pytest.approx(enc(90.0)) and gas[2 + genv.max_deg + slot] == pytest.approx(enc(90.0))
    assert gas[0] == 0.0


def test_reward_matches_astar_cost_plus_avoid_penalty(genv):
    tl = flooded(genv, "X-S", 90.0)
    g = GasRewardConfig()
    dqn = GasDQN(genv, [tl], DQNConfig(episodes=1, device="cpu"), g)
    s, slot = _cell_next_to(genv, _exit(genv, "X-S"))
    p = _healthy()
    ep = dqn.new_episode(p, tl, 5.0)
    leg = genv.walker.leg(genv.node(s), _exit(genv, "X-S"), s)
    _, r, done = dqn._step(s, slot, ep)
    tau = leg.time_s * p["factor"]
    expected = tau * (1 + g.w_C * 0.5 * (0.0 + 90.0) / g.C_ref) + g.avoid_penalty_s
    assert done and not ep.incap
    assert -r == pytest.approx(expected)
    assert ep.t == pytest.approx(tau) and ep.t_abs == pytest.approx(5.0 + tau)
    assert ep.peak == 90.0 and ep.dose == pytest.approx(45.0**2 * tau / 60.0)


def test_incapacitation_ends_the_episode_as_a_failure(genv):
    tl = flooded(genv, "X-S", 600.0)
    dqn = GasDQN(genv, [tl], DQNConfig(episodes=1, device="cpu"))
    s, slot = _cell_next_to(genv, _exit(genv, "X-S"))
    ep = dqn.new_episode(_healthy(), tl, 0.0)
    s2, r, done = dqn._step(s, slot, ep)
    assert done and ep.incap and not dqn._reached(s2, ep)
    assert -r > GasRewardConfig().incap_penalty_s


def test_training_learns_to_avoid_a_flooded_exit(genv):
    """X-E flooded at 90 ppm: after a short run, G-15 (whose fastest exit is
    X-E) leaves by another exit and stays out of the gas."""
    from dms.gas.simulate import spawn_cell

    tl = flooded(genv, "X-E", 90.0)
    cfg = DQNConfig(episodes=1500, batch_size=256, train_start=500, train_every=8, target_sync=500,
                    seed=0, device="cpu")
    dqn = GasDQN(genv, [tl], cfg)
    log = dqn.train()
    assert np.mean(log.reached_exit[-200:]) > 0.6
    s = spawn_cell(genv, "GF", "G-15")
    ro = dqn.rollout(s, dqn.new_episode(_healthy(), tl, 0.0))
    assert ro.reached_exit, ro.failure
    assert ro.nodes[-1] != _exit(genv, "X-E") and ro.peak_ppm < 10.0


def test_timeline_records_the_scenario(genv):
    sc = load_scenario(DEFAULT_SCENARIO)
    sc.duration_s = 60.0
    env = sc.make_env()
    tl = GasTimeline.from_scenario(sc, env)
    assert tl.node_ppm.shape == (61, env.n_nodes) and tl.duration_s == 60.0
    assert tl.alarm_t < 60.0  # the default leak trips the alarm within seconds
    assert (tl.soon_ppm >= tl.node_ppm - 1e-12).all()
    assert tl.forecast_at(tl.forecast_t0[0]) is tl.forecasts[0]
    assert np.array_equal(tl.now(1e6), tl.node_ppm[-1])  # holds the last frame
    assert tl.now(30.0).max() > 10.0
