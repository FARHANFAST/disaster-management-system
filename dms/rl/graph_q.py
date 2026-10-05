"""Tabular Q-learning over the navigation graph (GraphEvacEnv).

Same update as dms/rl/q_learning.py, but a state is a graph node and an
action is "go to neighbour k"; the reward is minus the walker's real walking
time for that leg.

One difference from the cell-level learner: the node is an abstraction of
the agent's true position (where in the room it stands), so the same
(node, action) can cost a different time depending on where the agent
entered the node. The target is therefore noisy and alpha defaults to 0.3
(averaging) instead of 1.0 — Q(node, a) tracks the *typical* time over the
positions the agent tends to be in, which is all a graph-level policy can
know. (Sweep on B1: alpha 0.3 beat 0.1 and 1.0 on mean excess time.)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from .graph_env import GraphEvacEnv


@dataclass(slots=True)
class GraphQConfig:
    episodes: int = 30_000
    alpha: float = 0.3
    gamma: float = 1.0
    eps_start: float = 0.5
    eps_end: float = 0.02
    eps_decay_frac: float = 0.8
    max_hops: int = 200
    seed: int = 0


@dataclass(slots=True)
class GraphTrainLog:
    episode_time: list[float] = field(default_factory=list)
    episode_hops: list[int] = field(default_factory=list)
    reached_exit: list[bool] = field(default_factory=list)


def train_graph(env: GraphEvacEnv, cfg: GraphQConfig = GraphQConfig(), *,
                progress_every: int = 0) -> tuple[np.ndarray, GraphTrainLog]:
    rng = np.random.default_rng(cfg.seed)
    Q = -rng.uniform(0.0, 1e-6, size=env.neighbours.shape)
    Q[~env.valid] = -np.inf
    Q[env.terminal] = 0.0
    valid_slots = [np.flatnonzero(row).tolist() for row in env.valid]
    log = GraphTrainLog()
    span = max(1, int(cfg.episodes * cfg.eps_decay_frac))

    for ep in range(cfg.episodes):
        eps = cfg.eps_start + min(1.0, ep / span) * (cfg.eps_end - cfg.eps_start)
        s = int(rng.choice(env.start_states))
        t, hops, done = 0.0, 0, False
        while hops < cfg.max_hops:
            u = env.node(s)
            slots = valid_slots[u]
            a = slots[int(rng.integers(len(slots)))] if rng.random() < eps else int(Q[u].argmax())
            s2, r, done = env.step(s, a)
            target = r if done else r + cfg.gamma * Q[env.node(s2)].max()
            Q[u, a] += cfg.alpha * (target - Q[u, a])
            t -= r
            hops += 1
            s = s2
            if done:
                break
        log.episode_time.append(t)
        log.episode_hops.append(hops)
        log.reached_exit.append(done)
        if progress_every and (ep + 1) % progress_every == 0:
            w = slice(-progress_every, None)
            print(f"  ep {ep + 1:6d}  eps={eps:.3f}  success={np.mean(log.reached_exit[w]):6.1%}  "
                  f"mean hops={np.mean(log.episode_hops[w]):5.1f}  mean time={np.mean(log.episode_time[w]):6.1f}s")
    return Q, log


@dataclass(slots=True)
class GraphRollout:
    nodes: list[int]   # graph decisions, start node first
    cells: list[int]   # every cell the walker stepped on
    time_s: float
    reached_exit: bool
    reason: str = ""


def greedy_graph_rollout(env: GraphEvacEnv, Q: np.ndarray, start: int, *, with_cells: bool = False,
                         max_hops: int = 200) -> GraphRollout:
    """Follow argmax Q over nodes from cell `start`; the walker executes each
    hop. A revisited node means a loop (the greedy policy depends only on
    the node), so the rollout stops there."""
    s, t = start, 0.0
    nodes = [env.node(s)]
    cells = [s]
    seen = {nodes[0]}
    for _ in range(max_hops):
        u = env.node(s)
        if env.terminal[u]:
            return GraphRollout(nodes, cells, t, True)
        a = int(Q[u].argmax())
        v = int(env.neighbours[u, a])
        if with_cells:
            cells.extend(env.walker.path(u, v, s)[1:])
        s, r, _done = env.step(s, a)
        t -= r
        if v in seen:
            return GraphRollout(nodes + [v], cells, t, False, "loop")
        seen.add(v)
        nodes.append(v)
    return GraphRollout(nodes, cells, t, bool(env.terminal[env.node(s)]), "max_hops")


@dataclass(slots=True)
class GraphEvalSummary:
    n_starts: int
    success_rate: float
    within_5pct: float      # realised time within 5% of the cell-level optimum
    mean_excess: float      # mean (t_graph / t_cell_opt - 1) over successful starts
    p90_excess: float
    max_excess: float


def evaluate_graph_policy(env: GraphEvacEnv, Q: np.ndarray, t_cell_opt: np.ndarray) -> GraphEvalSummary:
    """Run the greedy graph policy + walker from every non-exit cell and
    compare its real walking time to the exact cell-level optimum (the best
    any policy could do with the same physics)."""
    starts = [int(s) for s in env.start_states if np.isfinite(t_cell_opt[s]) and t_cell_opt[s] > 0]
    excess = []
    for s in starts:
        ro = greedy_graph_rollout(env, Q, s)
        if ro.reached_exit:
            excess.append(ro.time_s / t_cell_opt[s] - 1.0)
    ex = np.asarray(excess)
    return GraphEvalSummary(
        n_starts=len(starts),
        success_rate=len(ex) / len(starts),
        within_5pct=float((ex <= 0.05).sum()) / len(starts),
        mean_excess=float(ex.mean()) if ex.size else float("nan"),
        p90_excess=float(np.percentile(ex, 90)) if ex.size else float("nan"),
        max_excess=float(ex.max()) if ex.size else float("nan"),
    )


def graph_walker_optimum(env: GraphEvacEnv) -> np.ndarray:
    """Best possible time from every cell for *any* graph policy executed by
    this walker — decisions allowed to depend on the exact cell, not just the
    node. Backward Dijkstra over walker legs (cell -> arrival cell).

    Splits a graph policy's gap to the cell optimum into
      cell optimum -> this  : cost of the design (walking node to node)
      this -> graph RL      : cost of deciding from the node alone
    """
    n = env.cell_env.n_states
    rev: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for s in range(n):
        u = env.node(s)
        if env.terminal[u]:
            continue
        for a in np.flatnonzero(env.valid[u]):
            s2, r, _ = env.step(s, int(a))
            if s2 != s:
                rev[s2].append((s, -r))
    dist = np.full(n, np.inf)
    heap: list[tuple[float, int]] = []
    for s in range(n):
        if env.terminal[env.node(s)]:
            dist[s] = 0.0
            heap.append((0.0, s))
    heapq.heapify(heap)
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist[s]:
            continue
        for p, c in rev[s]:
            if d + c < dist[p]:
                dist[p] = d + c
                heapq.heappush(heap, (d + c, p))
    return dist
