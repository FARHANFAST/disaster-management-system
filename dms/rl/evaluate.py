"""Grading a learned Q-table.

Two references:
  * `optimal_times` — exact shortest evacuation time on the *same* 8-connected
    grid MDP (Dijkstra backwards from the exit cells). This is the true
    optimum Q-learning can converge to, so it's what the agent is graded on.
  * the Eikonal field T (any-angle, continuous) — reported for context only:
    an 8-connected path is up to 1/cos(22.5 deg) ~= 8.2% longer than an
    any-angle one, so even a perfect grid agent sits a few % above T.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np

from .env import GridEvacEnv
from .q_learning import greedy_rollout


def optimal_times(env: GridEvacEnv) -> np.ndarray:
    """Shortest time (s) from every state to any exit cell, via Dijkstra on
    the reversed transition graph. inf where no exit is reachable."""
    preds: list[list[tuple[int, float]]] = [[] for _ in range(env.n_states)]
    for s in range(env.n_states):
        for a in range(env.next_state.shape[1]):
            s2 = int(env.next_state[s, a])
            if s2 != s:
                preds[s2].append((s, -float(env.reward[s, a])))

    dist = np.full(env.n_states, np.inf)
    heap: list[tuple[float, int]] = []
    for s in np.flatnonzero(env.terminal):
        dist[s] = 0.0
        heap.append((0.0, int(s)))
    heapq.heapify(heap)
    while heap:
        d, s = heapq.heappop(heap)
        if d > dist[s]:
            continue
        for p, cost in preds[s]:
            nd = d + cost
            if nd < dist[p]:
                dist[p] = nd
                heapq.heappush(heap, (nd, p))
    return dist


@dataclass(slots=True)
class EvalSummary:
    n_starts: int
    success_rate: float  # greedy policy reaches an exit
    optimal_rate: float  # ...and within 1% of the grid optimum
    mean_excess: float  # mean (t_greedy / t_opt - 1) over successful starts
    max_excess: float
    value_mae_s: float  # mean |V(s) - (-t_opt(s))| over non-exit states


def evaluate_all_states(env: GridEvacEnv, Q: np.ndarray, t_opt: np.ndarray) -> EvalSummary:
    starts = [int(s) for s in env.start_states if np.isfinite(t_opt[s])]
    ok, near_opt, excess = 0, 0, []
    for s in starts:
        ro = greedy_rollout(env, Q, s)
        if not ro.reached_exit:
            continue
        ok += 1
        ratio = ro.time_s / t_opt[s] - 1.0
        excess.append(ratio)
        near_opt += ratio <= 0.01
    V = Q.max(axis=1)
    finite = np.isfinite(t_opt) & ~env.terminal
    return EvalSummary(
        n_starts=len(starts),
        success_rate=ok / len(starts),
        optimal_rate=near_opt / len(starts),
        mean_excess=float(np.mean(excess)) if excess else float("nan"),
        max_excess=float(np.max(excess)) if excess else float("nan"),
        value_mae_s=float(np.mean(np.abs(V[finite] + t_opt[finite]))),
    )
