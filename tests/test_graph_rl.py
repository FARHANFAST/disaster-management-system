"""Hierarchical evacuee (dms/rl/graph_env.py, graph_q.py): graph clean-up,
the grid walker's physical validity, and that graph Q-learning lands close
to both references."""

from __future__ import annotations

import numpy as np
import pytest

from dms.rl.evaluate import optimal_times
from dms.rl.graph_env import GraphEvacEnv
from dms.rl.graph_q import (
    GraphQConfig,
    evaluate_graph_policy,
    graph_walker_optimum,
    greedy_graph_rollout,
    train_graph,
)


@pytest.fixture(scope="module")
def genv(site):
    return GraphEvacEnv(site, "B1")


@pytest.fixture(scope="module")
def trained(genv):
    Q, _ = train_graph(genv, GraphQConfig(episodes=20_000, seed=0))
    return Q


def test_every_cell_belongs_to_one_kept_node(genv):
    assert genv.node_of_state.shape == (genv.cell_env.n_states,)
    assert sum(len(c) for c in genv.cells_of_node) == genv.cell_env.n_states
    assert all(len(c) > 0 for c in genv.cells_of_node)


def test_graph_cleanup(genv):
    kinds = {genv.nav.node(n).kind_detail for n in genv.node_ids}
    assert "opening" not in kinds and "assembly_point" not in kinds
    assert "X:X-SE" not in genv.node_ids  # locked by default
    exits = {genv.label(k) for k in np.flatnonzero(genv.terminal)}
    assert exits == {"X:X-S", "X:X-N", "X:X-W", "X:X-E", "X:X-NW"}
    floor = lambda k: genv.cell_env.cell_floor[genv.cells_of_node[k][0]]
    vertical = {(genv.label(u), genv.label(int(v))) for u in range(genv.n_nodes)
                for v in genv.neighbours[u][genv.valid[u]] if floor(u) != floor(int(v)) and u < v}
    assert len(vertical) == 3  # ST-W, ST-E, ST-C


def test_walker_legs_are_physical_and_stay_in_their_two_nodes(genv):
    ce = genv.cell_env
    rng = np.random.default_rng(0)
    for u in rng.choice(genv.n_nodes, size=40, replace=False):
        u = int(u)
        for v in genv.neighbours[u][genv.valid[u]]:
            v = int(v)
            s = int(rng.choice(genv.cells_of_node[u]))
            path = genv.walker.path(u, v, s)
            leg = genv.walker.leg(u, v, s)
            assert path[0] == s and path[-1] == leg.arrival
            assert genv.node(path[-1]) == v
            walked = 0.0
            for a, b in zip(path, path[1:]):
                acts = np.flatnonzero(ce.next_state[a] == b)
                assert acts.size, "walker made a move the cell model forbids"
                walked -= ce.reward[a, acts[0]]
            assert walked == pytest.approx(leg.time_s)
            if ce.cell_floor[s] == ce.cell_floor[path[-1]]:
                assert {genv.node(c) for c in path} <= {u, v}


def test_graph_walker_optimum_is_bounded_by_cell_optimum(genv):
    t_cell = optimal_times(genv.cell_env)
    t_graph = graph_walker_optimum(genv)
    assert np.isfinite(t_graph).all()
    assert (t_graph >= t_cell - 1e-6).all()


def test_graph_q_learning_evacuates_near_optimally(genv, trained):
    e = evaluate_graph_policy(genv, trained, optimal_times(genv.cell_env))
    assert e.success_rate >= 0.99
    assert e.mean_excess <= 0.06


def test_ff_spawn_takes_stairs_down(site, genv, trained):
    ce = genv.cell_env
    room = site.get_building("B1").get_floor("FF").spaces["F-10"]
    ro = greedy_graph_rollout(genv, trained, ce.nearest_state(*room.rect.center, "FF"), with_cells=True)
    assert ro.reached_exit
    assert ce.state_floor(ro.cells[-1]) == "GF"
    assert any(genv.nav.node(genv.label(k)).kind_detail == "stair_landing" for k in ro.nodes)


def test_closed_stair_is_not_in_the_graph(site):
    env = GraphEvacEnv(site, "B1", closed_stairs=["ST-C"])
    floor = lambda k: env.cell_env.cell_floor[env.cells_of_node[k][0]]
    landings = [k for k in range(env.n_nodes) if env.label(k).startswith("ST-C:")]
    for k in landings:
        assert all(floor(int(v)) == floor(k) for v in env.neighbours[k][env.valid[k]])
