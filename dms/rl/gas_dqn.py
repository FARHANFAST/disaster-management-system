"""Gas-aware DQN: the route policy of dms/rl/dqn.py, trained inside an H2S
release from the gas model (dms/gas/), so it learns to route around the plume.

Same network, replay buffer, double-Q target and masking as the base DQN; the
episode hooks change:

State (beside the node one-hot), all gas values log-scaled to [0, 1] by
log(1 + C) / log(1 + 500 ppm):
    agent attribute features (dms/rl/agent_profile.py)
    gas at the agent's node: now, and expected soon (forecast, next 60 s)
    gas at each neighbour slot: now and soon (0 for empty slots)
    time since the alarm / 300 s
The neighbour readings are what an evacuee can see or is told; with time
since the alarm the state is Markov for one fixed release, and the gas inputs
let one policy carry over to the wind variants it trains on.

Transition: the graph env's walker leg, timed by the agent's own speed
(attribute `factor`), on the gas history from dms/gas/timeline.py starting
when this agent starts moving (alarm + a sampled pre-movement delay).

Reward = -cost, the gas-aware A* edge cost (dms/gas/routing.py, spec N5) so
both planners optimise the same thing:
    leg_time · (1 + w_C · susceptibility · C_leg / C_ref)
  + avoid_penalty_s       entering a node at or above avoid_ppm (10 ppm)
  + incap_penalty_s       C · susceptibility >= 500 ppm: incapacitated, episode over
Susceptibility comes from the agent's health status (Irritated / Distressed
agents are more affected), so attributes now change which route is best.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import torch

from ..gas.routing import GasAwarePlanner, RoutingParams
from ..gas.timeline import GasTimeline
from ..graph.cost import C_REF, W_C
from .agent_profile import profile_features, sample_population
from .dqn import DQN, DQNConfig, Episode
from .graph_env import GraphEvacEnv

PPM_SCALE = math.log1p(500.0)
SUSCEPTIBILITY = {"Healthy": 1.0, "Irritated": 1.5, "Distressed": 2.0}


def enc(c: np.ndarray | float) -> np.ndarray:
    return np.minimum(np.log1p(np.asarray(c, dtype=np.float64)) / PPM_SCALE, 1.0)


@dataclass(slots=True)
class GasRewardConfig:
    w_C: float = W_C
    C_ref: float = C_REF
    avoid_ppm: float = 10.0
    avoid_penalty_s: float = 120.0
    incap_ppm: float = 500.0
    incap_penalty_s: float = 1000.0
    start_after_alarm_s: tuple[float, float] = (0.0, 40.0)  # sampled pre-movement window
    time_scale_s: float = 300.0


@dataclass(slots=True)
class GasEpisode(Episode):
    tl: GasTimeline | None = None
    t_abs: float = 0.0  # scenario clock
    factor: float = 1.0  # V0 / own speed
    sus: float = 1.0
    cost: float = 0.0
    peak: float = 0.0
    dose: float = 0.0  # ppm²·min, L = Σ C² dt / 60
    incap: bool = False
    nodes: list[int] = field(default_factory=list)


@dataclass(slots=True)
class GasRollout:
    nodes: list[int]
    time_s: float
    reached_exit: bool
    incapacitated: bool
    peak_ppm: float
    dose: float
    cost: float
    failure: str | None = None


def profile_physics(p: dict) -> tuple[float, float]:
    """(walk-time factor, gas susceptibility) for profile `p`."""
    return float(p["factor"]), SUSCEPTIBILITY[p["healthStatus"]]


class GasDQN(DQN):
    def __init__(self, env: GraphEvacEnv, timelines: list[GasTimeline], cfg: DQNConfig = DQNConfig(),
                 gas: GasRewardConfig | None = None):
        if not timelines:
            raise ValueError("need at least one gas timeline")
        self.timelines = timelines
        self.gas = gas or GasRewardConfig()
        super().__init__(env, cfg)

    # ------------------------------------------------------------- hooks

    def _n_extra(self) -> int:
        return self.n_attr + 2 + 2 * self.env.max_deg + 1

    def new_episode(self, profile: dict, tl: GasTimeline, t_abs: float) -> GasEpisode:
        factor, sus = profile_physics(profile)
        return GasEpisode(attr=profile_features(profile), tl=tl, t_abs=t_abs, factor=factor, sus=sus)

    def _begin_episode(self) -> tuple[int, GasEpisode]:
        p = sample_population(1, self.rng)[0]
        tl = self.timelines[int(self.rng.integers(len(self.timelines)))]
        t_abs = tl.alarm_t + float(self.rng.uniform(*self.gas.start_after_alarm_s))
        return int(self.rng.choice(self.env.start_states)), self.new_episode(p, tl, t_abs)

    def _x(self, u: int, ep: GasEpisode) -> np.ndarray:
        tl, env = ep.tl, self.env
        now, soon = tl.now(ep.t_abs), tl.soon(ep.t_abs)
        nb, ok = env.neighbours[u], env.valid[u]
        since = np.clip((ep.t_abs - tl.alarm_t) / self.gas.time_scale_s, 0.0, 1.0)
        return np.concatenate([
            ep.attr,
            enc([now[u], soon[u]]),
            np.where(ok, enc(now[nb]), 0.0),
            np.where(ok, enc(soon[nb]), 0.0),
            [since],
        ]).astype(np.float32)

    def _step(self, s: int, a: int, ep: GasEpisode) -> tuple[int, float, bool]:
        env, g, tl = self.env, self.gas, ep.tl
        u = env.node(s)
        v = int(env.neighbours[u, a])
        leg = env.walker.leg(u, v, s)
        if leg is None:  # shouldn't happen on a valid graph: stand still at a large cost
            ep.t += 60.0
            ep.t_abs += 60.0
            ep.cost += 60.0
            return s, -60.0, False
        tau = leg.time_s * ep.factor
        c_u = float(tl.now(ep.t_abs)[u])
        c_v = float(tl.now(ep.t_abs + tau)[v])
        c_leg = 0.5 * (c_u + c_v)
        cost = tau * (1.0 + g.w_C * ep.sus * c_leg / g.C_ref)
        if c_v >= g.avoid_ppm:
            cost += g.avoid_penalty_s
        done = bool(env.terminal[v])
        if max(c_u, c_v) * ep.sus >= g.incap_ppm:
            ep.incap = True
            cost += g.incap_penalty_s
            done = True
        ep.dose += c_leg * c_leg * tau / 60.0
        ep.peak = max(ep.peak, c_u, c_v)
        ep.t += tau
        ep.t_abs += tau
        ep.cost += cost
        return leg.arrival, -cost, done

    def _reached(self, s: int, ep: GasEpisode) -> bool:
        return bool(self.env.terminal[self.env.node(s)]) and not ep.incap

    # ------------------------------------------------------------- policies

    @torch.no_grad()
    def greedy_slot(self, s: int, ep: GasEpisode) -> int:
        u = self.env.node(s)
        q = self._q_valid(self.online, np.asarray([u]), self._x(u, ep)[None])
        return int(q.argmax().item())

    def rollout(self, start: int, ep: GasEpisode, choose: Callable[[int, GasEpisode], int | None] | None = None,
                *, max_hops: int | None = None) -> GasRollout:
        """Walk `choose` (default: the greedy DQN) through the gas history.
        `choose(s, ep)` returns a neighbour slot, or None when it has no route."""
        choose = choose or self.greedy_slot
        env, s = self.env, start
        ep.nodes = [env.node(s)]
        failure = "max_hops"
        for _ in range(max_hops or self.cfg.max_hops):
            if env.terminal[env.node(s)] or ep.incap:
                failure = None
                break
            a = choose(s, ep)
            if a is None:
                failure = "no_route"
                break
            s, _, _ = self._step(s, a, ep)
            ep.nodes.append(env.node(s))
        reached = self._reached(s, ep)
        if ep.incap:
            failure = "incapacitated"
        return GasRollout(ep.nodes, ep.t, reached, ep.incap, ep.peak, ep.dose, ep.cost,
                          None if reached else failure)


def astar_chooser(env: GraphEvacEnv, routing: RoutingParams, *, gas_aware: bool,
                  dwell_s: dict[int, float] | None = None) -> Callable[[int, GasEpisode], int | None]:
    """A* re-planned at every node on the live reading and the forecast —
    the default planner of dms/gas/simulate.py, as a slot chooser."""
    planner = GasAwarePlanner(env, routing, gas_aware=gas_aware, dwell_s=dwell_s)

    def choose(s: int, ep: GasEpisode) -> int | None:
        plan = planner.plan(s, ep.t_abs, ep.tl.now(ep.t_abs), ep.tl.forecast_at(ep.t_abs))
        if plan is None or plan.route.first_hop is None:
            return None
        u = env.node(s)
        return int(np.flatnonzero(env.neighbours[u] == plan.route.first_hop)[0])

    return choose


def egress_dwell(env: GraphEvacEnv, tl: GasTimeline) -> dict[int, float]:
    """Seconds spent on each exit's outdoor walk to its assembly point."""
    return {int(k): float(np.hypot(*(pts[-1] - pts[0]))) / env.cell_env.v0
            for k, pts in zip(tl.zones.exit_nodes, tl.zones.egress_xy)}
