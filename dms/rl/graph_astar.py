"""A* on the hierarchical evacuation graph (GraphEvacEnv) — the same graph
the graph-RL agent decides on, with the same GridWalker legs as edges.

A search state is the cell the agent stands on when it takes a graph
decision (its start cell, then each leg's arrival cell), because a leg's
walking time depends on where in the node the agent is. Expanding cell s in
node u tries every neighbour v: the walker leg u -> v from s gives the time
and the arrival cell. So the planned time is exactly the time the agent will
walk (GraphEvacEnv.step), and with no extra costs A* returns the best
graph+walker route — graph_walker_optimum, from one start instead of all.

Heuristic: straight-line distance to the nearest allowed exit cell / v0,
plus the cheapest stair trip when no allowed exit is on the agent's floor.
The walker moves 8-connected at v0, so this never overestimates. Extra
costs (gas, crowding, ...) plug in through `edge_cost`, which must return
at least the leg's walking time (or inf to forbid the edge) for the
heuristic to stay admissible.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from .graph_env import GraphEvacEnv

# edge_cost(u, v, t_before, leg_time) -> cost; t_before is the walking time
# already spent on the route when the leg starts (so time-aware costs can
# look up when the agent will actually be there).
EdgeCost = Callable[[int, int, float, float], float]


@dataclass(slots=True)
class GraphRoute:
    nodes: list[int]  # graph decisions, start node first
    cells: list[int]  # start cell, then each leg's arrival cell
    times: list[float]  # walking time at each entry of `cells`
    cost: float

    @property
    def time_s(self) -> float:
        return self.times[-1]

    @property
    def first_hop(self) -> int | None:
        return self.nodes[1] if len(self.nodes) > 1 else None


class GraphAStar:
    """Reusable planner for one GraphEvacEnv: precomputes exit-cell
    positions and the cheapest stair trip once."""

    def __init__(self, env: GraphEvacEnv):
        self.env = env
        ce = env.cell_env
        self.v0 = ce.v0
        self.cell_xy = (ce.cells[:, ::-1] + 0.5) * ce.h  # (x, y) up to a constant origin shift
        self.min_stair_s = min(ce.stair_cost_s.values(), default=0.0)
        self.exit_nodes = [int(k) for k in np.flatnonzero(env.terminal)]

    def _heuristic(self, goals: list[int]) -> Callable[[int], float]:
        ce = self.env.cell_env
        by_floor: dict[int, np.ndarray] = {}
        for k in goals:
            cells = self.env.cells_of_node[k]
            for f in np.unique(ce.cell_floor[cells]):
                pts = self.cell_xy[cells[ce.cell_floor[cells] == f]]
                by_floor[int(f)] = np.vstack([by_floor[int(f)], pts]) if int(f) in by_floor else pts
        all_pts = np.vstack(list(by_floor.values()))
        cache: dict[int, float] = {}

        def h(s: int) -> float:
            if s not in cache:
                f = int(ce.cell_floor[s])
                pts = by_floor.get(f)
                if pts is None:
                    cache[s] = float(np.hypot(*(all_pts - self.cell_xy[s]).T).min()) / self.v0 + self.min_stair_s
                else:
                    cache[s] = float(np.hypot(*(pts - self.cell_xy[s]).T).min()) / self.v0
            return cache[s]

        return h

    def plan(self, start: int, *, edge_cost: EdgeCost | None = None, goals: Iterable[int] | None = None,
             max_expansions: int = 200_000) -> GraphRoute | None:
        """Cheapest route from cell `start` to any goal exit node (default:
        every exit). None if no goal is reachable."""
        env = self.env
        goal_list = self.exit_nodes if goals is None else [int(g) for g in goals]
        if not goal_list:
            return None
        goal_set = set(goal_list)
        if env.node(start) in goal_set:
            return GraphRoute([env.node(start)], [start], [0.0], 0.0)
        h = self._heuristic(goal_list)

        tie = itertools.count()
        g_cost = {start: 0.0}
        walk = {start: 0.0}
        parent: dict[int, int] = {}
        closed: set[int] = set()
        heap = [(h(start), next(tie), start)]
        while heap and len(closed) < max_expansions:
            _f, _, s = heapq.heappop(heap)
            if s in closed:
                continue
            closed.add(s)
            u = env.node(s)
            if u in goal_set:
                return self._reconstruct(parent, walk, g_cost, start, s)
            if env.terminal[u]:
                continue  # an exit that isn't an allowed goal is a dead end
            for v in env.neighbours[u][env.valid[u]]:
                v = int(v)
                leg = env.walker.leg(u, v, s)
                if leg is None or leg.arrival in closed:
                    continue
                c = leg.time_s if edge_cost is None else edge_cost(u, v, walk[s], leg.time_s)
                if math.isinf(c):
                    continue
                ng = g_cost[s] + c
                if ng < g_cost.get(leg.arrival, math.inf) - 1e-12:
                    g_cost[leg.arrival] = ng
                    walk[leg.arrival] = walk[s] + leg.time_s
                    parent[leg.arrival] = s
                    heapq.heappush(heap, (ng + h(leg.arrival), next(tie), leg.arrival))
        return None

    def _reconstruct(self, parent: dict[int, int], walk: dict[int, float], g_cost: dict[int, float],
                     start: int, end: int) -> GraphRoute:
        cells = [end]
        while cells[-1] != start:
            cells.append(parent[cells[-1]])
        cells.reverse()
        return GraphRoute(nodes=[self.env.node(s) for s in cells], cells=cells,
                          times=[walk[s] for s in cells], cost=g_cost[end])

    def route_cells(self, route: GraphRoute) -> list[int]:
        """Every cell the walker steps on along `route` (for drawing)."""
        out = [route.cells[0]]
        for (s, u), v in zip(zip(route.cells, route.nodes), route.nodes[1:]):
            out.extend(self.env.walker.path(u, v, s)[1:])
        return out
