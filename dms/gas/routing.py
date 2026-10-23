"""Gas-aware route planning on the evacuation graph: A* (the default
planner) with gas costs from the gas model.

Edge cost (spec N5's gas term, same w_C and C_ref as dms/graph/cost.py):

    cost(u -> v) = leg_time · (1 + w_C · C_leg / C_ref),   C_leg = (C_u + C_v) / 2

and a node the agent would ENTER at C_v >= the tier's limit is forbidden.
With `lookahead`, C is not just today's reading: it is the worst of the
current value and the 5-minute forecast over the window the agent will
actually be there (its ETA on the route to ETA + `linger_s`), so a corridor
the plume is about to reach is avoided before it gets there.

Planning goes through tiers, stopping at the first that finds an exit:
  tier 0  gas-free: no node at or above `avoid_ppm`         -> evacuate
  tier 1  no node at or above `hard_ppm` (IDLH, 100 ppm)    -> evacuate, exposed
  tier 2  no limit; least-exposure route                    -> evacuate, exposed
The caller decides what to do when tier 0 fails (e.g. shelter in a clean
room instead, see simulate.py). `gas_aware=False` gives the gas-blind
baseline: plain fastest route, no costs, no limits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..graph.cost import C_REF, W_C
from ..rl.graph_astar import GraphAStar, GraphRoute
from ..rl.graph_env import GraphEvacEnv
from .model import Forecast


@dataclass(slots=True)
class RoutingParams:
    avoid_ppm: float = 10.0  # "a place with gas": the spec's ALERT/MINOR level
    hard_ppm: float = 100.0  # NIOSH IDLH for H2S
    w_C: float = W_C
    C_ref: float = C_REF
    lookahead: bool = True
    linger_s: float = 15.0


@dataclass(slots=True)
class Plan:
    route: GraphRoute
    tier: int  # 0 gas-free, 1 below IDLH, 2 least exposure (-1: gas-blind planner)
    worst_ppm: float  # worst expected C on any node the route enters


class GasAwarePlanner:
    algorithm = "astar"

    def __init__(self, env: GraphEvacEnv, params: RoutingParams | None = None, *, gas_aware: bool = True,
                 dwell_s: dict[int, float] | None = None):
        """`dwell_s` overrides `linger_s` per node — e.g. an exit's outdoor
        walk to its assembly point, which is all spent in that exit's air."""
        self.env = env
        self.params = params or RoutingParams()
        self.gas_aware = gas_aware
        self.dwell = np.full(env.n_nodes, self.params.linger_s)
        for k, d in (dwell_s or {}).items():
            self.dwell[k] = d
        self.astar = GraphAStar(env)

    def expected_ppm(self, k: int, t_abs: float, node_ppm: np.ndarray, forecast: Forecast | None) -> float:
        c = float(node_ppm[k])
        if self.params.lookahead and forecast is not None:
            c = max(c, forecast.node_max(k, t_abs, t_abs + self.dwell[k]))
        return c

    def plan(self, start: int, t_now: float, node_ppm: np.ndarray, forecast: Forecast | None,
             *, max_tier: int = 2) -> Plan | None:
        if not self.gas_aware:
            route = self.astar.plan(start)
            return None if route is None else Plan(route, -1, self._worst(route, t_now, node_ppm, forecast))

        p = self.params
        start_node = self.env.node(start)
        for tier, limit in enumerate((p.avoid_ppm, p.hard_ppm, math.inf)):
            if tier > max_tier:
                break

            def edge_cost(u: int, v: int, t_before: float, leg_time: float, limit=limit) -> float:
                c_v = self.expected_ppm(v, t_now + t_before + leg_time, node_ppm, forecast)
                if c_v >= limit and v != start_node:
                    return math.inf
                c_u = self.expected_ppm(u, t_now + t_before, node_ppm, forecast)
                return leg_time * (1.0 + p.w_C * 0.5 * (c_u + c_v) / p.C_ref)

            route = self.astar.plan(start, edge_cost=edge_cost)
            if route is not None:
                return Plan(route, tier, self._worst(route, t_now, node_ppm, forecast))
        return None

    def _worst(self, route: GraphRoute, t_now: float, node_ppm: np.ndarray, forecast: Forecast | None) -> float:
        return max((self.expected_ppm(k, t_now + t, node_ppm, forecast)
                    for k, t in zip(route.nodes[1:], route.times[1:])), default=0.0)
