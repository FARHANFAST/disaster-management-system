"""A* over the NavGraph using the (N5) edge cost — spec §6.2. Goals default
to every assembly point (an exit is only ever an intermediate node, never a
goal itself, per spec [Y])."""

from __future__ import annotations

import heapq
import itertools
import math
from dataclasses import dataclass

from .builder import NavGraph
from .cost import CostParams, link_cost

V_MAX = 1.65  # m/s — spec U_max; an upper bound on achievable speed anywhere,
# which is what keeps h(n) = dist/v_max admissible.


@dataclass(slots=True)
class Route:
    nodes: list[str]
    links: list[tuple[str, str]]
    cost: float
    eta_s: float


def _dist(graph: NavGraph, a: str, b: str) -> float:
    ax, ay = graph.node(a).XY
    bx, by = graph.node(b).XY
    return math.hypot(bx - ax, by - ay)


def _reconstruct(graph: NavGraph, came_from: dict[str, str], src: str, goal: str, cost: float) -> Route:
    path = [goal]
    while path[-1] != src:
        path.append(came_from[path[-1]])
    path.reverse()
    links = list(itertools.pairwise(path))
    return Route(nodes=path, links=links, cost=cost, eta_s=cost)


def route(
    graph: NavGraph,
    src: str,
    goals: list[str] | None = None,
    *,
    params: CostParams | None = None,
    v_max: float = V_MAX,
) -> Route | None:
    """Lowest-cost path from `src` to the nearest reachable goal. `goals`
    defaults to every assembly point node. Returns None if unreachable."""
    if goals is None:
        goals = graph.assembly_point_nodes()
    if not goals:
        return None
    goal_set = set(goals)
    if src in goal_set:
        return Route(nodes=[src], links=[], cost=0.0, eta_s=0.0)

    def h(n: str) -> float:
        return min(_dist(graph, n, g) for g in goals) / v_max

    g_score: dict[str, float] = {src: 0.0}
    came_from: dict[str, str] = {}
    visited: set[str] = set()
    open_heap: list[tuple[float, float, str]] = [(h(src), 0.0, src)]

    while open_heap:
        _f, g, node = heapq.heappop(open_heap)
        if node in visited:
            continue
        visited.add(node)
        if node in goal_set:
            return _reconstruct(graph, came_from, src, node, g)
        for nb in graph.g.neighbors(node):
            if nb in visited:
                continue
            link = graph.link(node, nb)
            c = link_cost(link, params=params)
            if math.isinf(c):
                continue
            ng = g + c
            if ng < g_score.get(nb, math.inf) - 1e-12:
                g_score[nb] = ng
                came_from[nb] = node
                heapq.heappush(open_heap, (ng + h(nb), ng, nb))

    return None


def nearest_assembly_route(graph: NavGraph, src: str, *, params: CostParams | None = None) -> Route | None:
    return route(graph, src, params=params)
