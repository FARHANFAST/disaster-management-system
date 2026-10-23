"""End-to-end H2S scenario: gas model + alarm + evacuees on the graph-RL
navigation graph, routed by gas-aware A*. Writes a replay page.

    python -m dms.gas.simulate                                   # default scenario
    python -m dms.gas.simulate --scenario configs/scenarios/h2s_tank_leak.yaml
    python -m dms.gas.simulate --no-blind                        # skip the gas-blind comparison run

Simulation core (Gas Dispersion design doc, "Integration"), each second:
  1-3. gas model: weather, puffs, rooms, detectors, alarm (dms/gas/model.py)
  4.   the alarm state is broadcast; on EMERGENCY the facility shuts HVAC down
  5.   agents step at `agent_hz`, reading the latest gas values; exit doors
       they open and rooms they seal are written to the map and change
       infiltration from the next gas step
  6.   the forecast refreshes inside the gas model
  7.   the frame is recorded for replay

Agents (one per spawn room):
  NORMAL      at their desk until the EMERGENCY broadcast, or until their own
              node reads >= self_alert_ppm (they smell it)
  ALERTED     pre-movement delay (seeded uniform draw), then
  EVACUATING  at every graph node they re-plan with A* from where they stand
              and walk the first hop with the grid walker (same legs as the
              graph-RL env). No gas-free route and their room is still clean
              -> SHELTERING; otherwise the least-exposed route (tier 1/2)
  SHELTERING  door shut (room sealed, lower ACH); re-plan every few seconds
              and leave as soon as a gas-free route opens
  OUTSIDE     through an exit, walking the egress path to its assembly point
              (exposed to the outdoor plume on the way)
  SAFE        at the assembly point
  INCAPACITATED  local concentration >= 500 ppm (spec band B4)
Walking speed is scaled by the spec's health factor φ for the local band.
Exposure: peak ppm and toxic load L = Σ C² dt/60 [ppm²·min] (spec X1, n=2).

The comparison run puts the same people (same spawns, same pre-movement
draws) on the gas-blind planner: the fastest route with no gas costs.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..rl.env import STAIRS
from ..rl.graph_env import GraphEvacEnv
from ..rl.live_page import NODE_KINDS, _categories
from .alarm import EMERGENCY, NORMAL, WARNING
from .indoor import MapVentilationState
from .model import GasModel
from .routing import GasAwarePlanner, Plan
from .scenario import REPO_ROOT, Scenario, load_scenario
from .zones import NodeZones

DEFAULT_SCENARIO = REPO_ROOT / "configs" / "scenarios" / "h2s_tank_leak.yaml"
TEMPLATE = Path(__file__).resolve().parent / "web" / "gas_template.html"

BAND_EDGES = (0.5, 10.0, 100.0, 500.0)  # ppm, SYSTEM_SPEC §2.4
PHI = (1.00, 1.00, 0.80, 0.40, 0.00)  # walking-speed factor per band
INCAPACITATION_PPM = 500.0
SOON_S = 60.0  # replay flags a node when gas is due there within this many seconds

A_NORMAL, A_ALERTED, A_EVAC, A_SHELTER, A_OUTSIDE, A_SAFE, A_INCAP = (
    "NORMAL", "ALERTED", "EVACUATING", "SHELTERING", "OUTSIDE", "SAFE", "INCAPACITATED")
AGENT_STATES = (A_NORMAL, A_ALERTED, A_EVAC, A_SHELTER, A_OUTSIDE, A_SAFE, A_INCAP)


def band(c_ppm: float) -> int:
    return int(np.searchsorted(BAND_EDGES, c_ppm, side="right"))


@dataclass(slots=True)
class Agent:
    i: int
    floor: str
    room: str
    name: str
    start: int
    premove_s: float
    s: int = -1
    state: str = A_NORMAL
    go_t: float = math.inf
    path: list[int] | None = None
    path_idx: int = 0
    progress: float = 0.0
    next_replan: float = 0.0
    sealed_zone: int | None = None
    plan: Plan | None = None
    c_now: float = 0.0  # ppm where the agent stands, latest agent step
    dose: float = 0.0  # ppm²·min, accumulated every agent step (10 Hz)
    peak: float = 0.0
    rec_ppm: list[float] = field(default_factory=list)  # per gas second
    rec_dose: list[float] = field(default_factory=list)
    t_alert: float | None = None
    t_out: float | None = None
    t_safe: float | None = None
    exit_name: str | None = None
    exit_ppm: float | None = None  # worst air on the egress path when they stepped out
    ap: str | None = None
    egress: np.ndarray | None = None  # outdoor path, site metres
    egress_len: float = 0.0
    egress_dist: float = 0.0
    out_t: list[float] = field(default_factory=list)  # outdoor trace, sampled
    out_xy: list[tuple[float, float]] = field(default_factory=list)
    worst_tier: int = -1
    reroutes: int = 0
    trace_t: list[float] = field(default_factory=list)
    trace_s: list[int] = field(default_factory=list)
    states: list[tuple[float, str]] = field(default_factory=list)
    plans: list[tuple[float, int, list[int]]] = field(default_factory=list)
    entered: list[tuple[float, int, float]] = field(default_factory=list)  # (t, node, ppm on entry)
    _last_nodes: list[int] = field(default_factory=list)


def spawn_cell(env: GraphEvacEnv, floor_id: str, room: str) -> int:
    """Walkable cell of `room` nearest its centre."""
    ce = env.cell_env
    cells = env.cells_of_node[env.node_ids.index(f"{floor_id}:R:{room}")]
    xy = np.array([ce.state_xy(int(s)) for s in cells])
    return int(cells[np.argmin(np.hypot(*(xy - xy.mean(axis=0)).T))])


class Simulation:
    def __init__(self, sc: Scenario, env: GraphEvacEnv | None = None, *, gas_aware: bool = True):
        self.sc = sc
        self.env = env = env or sc.make_env()
        self.building = sc.site.get_building(sc.building_id)
        self.zones = NodeZones.from_graph_env(env, sc.site, self.building, ceiling_m=sc.ceiling_m)
        self._exit_index = {int(k): i for i, k in enumerate(self.zones.exit_nodes)}
        self.gas = GasModel(sc.site, self.zones, sc.make_sources(), sc.weather, sc.make_detectors(env), sc.gas_cfg)
        egress_walk_s = {int(k): float(np.hypot(*(pts[-1] - pts[0]))) / env.cell_env.v0
                         for k, pts in zip(self.zones.exit_nodes, self.zones.egress_xy)}
        self.planner = GasAwarePlanner(env, sc.routing, gas_aware=gas_aware, dwell_s=egress_walk_s)
        self.gas_aware = gas_aware
        self.hvac_on = True
        self.exit_open_until: dict[int, float] = {}
        self.t = 0.0

        rng = np.random.default_rng(sc.seed)
        lo, hi = sc.premovement_s
        self.agents: list[Agent] = []
        for i, sp in enumerate(sc.spawns):
            s = spawn_cell(env, sp.floor, sp.room)
            name = self.building.get_floor(sp.floor).spaces[sp.room].name
            ag = Agent(i=i, floor=sp.floor, room=sp.room, name=name, start=s, s=s,
                       premove_s=float(rng.uniform(lo, hi)))
            ag.trace_t.append(0.0)
            ag.trace_s.append(s)
            ag.states.append((0.0, A_NORMAL))
            self.agents.append(ag)

        # per-second record
        self.rec_t: list[float] = []
        self.rec_alarm: list[str] = []
        self.rec_wind: list[tuple[float, float]] = []
        self.rec_hvac: list[bool] = []
        self.rec_node_ppm: list[np.ndarray] = []
        self.rec_det_ppm: list[np.ndarray] = []
        self.rec_field: list[np.ndarray] = []
        self.rec_forecast: list[tuple[float, np.ndarray]] = []
        self.rec_node_soon: list[np.ndarray] = []  # worst of now and the forecast over the next SOON_S
        self.decisions: list[dict] = []  # why each agent took the route it took (gas-aware run)
        self.hvac_off_t: float | None = None

    # ------------------------------------------------------------- map state

    def map_state(self) -> MapVentilationState:
        return MapVentilationState(
            hvac_on=self.hvac_on,
            open_exits=frozenset(k for k, until in self.exit_open_until.items() if until > self.t),
            sealed_zones=frozenset(a.sealed_zone for a in self.agents if a.sealed_zone is not None),
        )

    # ------------------------------------------------------------- run

    def run(self) -> Simulation:
        sc = self.sc
        n_steps = int(round(sc.duration_s / sc.gas_dt_s))
        sub = max(1, int(round(sc.agent_hz * sc.gas_dt_s)))
        dt_a = sc.gas_dt_s / sub
        last_fc = None
        for _ in range(n_steps):
            self.gas.step(sc.gas_dt_s, self.map_state())  # steps 1-3 (+6)
            if self.gas.alarm.state == EMERGENCY and self.hvac_on and sc.hvac_shutdown_on_alarm:
                self.hvac_on = False  # step 4: facility response, written to the map
                self.hvac_off_t = self.gas.t
            t_gas = self.gas.t
            for j in range(sub):  # step 5: agents move through [t_gas, t_gas + dt) on the values at t_gas
                self.t = t_gas + j * dt_a
                for ag in self.agents:
                    self._agent_step(ag, dt_a)
            self.t = t_gas
            self.rec_t.append(t_gas)  # step 7
            self.rec_alarm.append(self.gas.alarm.state)
            self.rec_wind.append((self.gas.weather.wind_speed, self.gas.weather.wind_from_deg))
            self.rec_hvac.append(self.hvac_on)
            self.rec_node_ppm.append(self.gas.node_ppm.copy())
            self.rec_det_ppm.append(self.gas.detector_ppm.copy())
            self.rec_field.append(self.gas.field_ppm.copy())
            self.rec_node_soon.append(self._node_soon(t_gas))
            for ag in self.agents:
                ag.rec_ppm.append(ag.c_now)
                ag.rec_dose.append(ag.dose)
            if self.gas.forecast is not None and self.gas.forecast is not last_fc:
                last_fc = self.gas.forecast
                self.rec_forecast.append((last_fc.t0, last_fc.field_max.copy()))
        return self

    # ------------------------------------------------------------- agents

    def _set_state(self, ag: Agent, state: str) -> None:
        if ag.state != state:
            ag.state = state
            ag.states.append((self.t, state))

    def _node_soon(self, t: float) -> np.ndarray:
        fc, now = self.gas.forecast, self.gas.node_ppm
        if fc is None:
            return now.copy()
        T = fc.node_ppm.shape[0]
        i0 = int(np.clip(np.floor((t - fc.t0) / fc.dt), 0, T - 1))
        i1 = int(np.clip(np.ceil((t + SOON_S - fc.t0) / fc.dt), i0 + 1, T))
        return np.maximum(now, fc.node_ppm[i0:i1].max(axis=0))

    def _agent_step(self, ag: Agent, dt: float) -> None:
        if ag.state == A_SAFE:
            ag.c_now = 0.0  # at the assembly point: out of the modelled area
            return
        if ag.state == A_INCAP:
            return
        env = self.env
        if ag.state == A_OUTSIDE:
            self._walk_outside(ag, dt)
            return
        c = float(self.gas.node_ppm[env.node(ag.s)])
        ag.c_now = c
        ag.dose += c * c * dt / 60.0
        ag.peak = max(ag.peak, c)
        if c >= INCAPACITATION_PPM:
            ag.path = None
            ag.sealed_zone = None
            self._set_state(ag, A_INCAP)
            return

        if ag.state == A_NORMAL:
            if self.gas.alarm.state == EMERGENCY or c >= self.sc.self_alert_ppm:
                ag.t_alert = self.t
                ag.go_t = self.t + ag.premove_s
                self._set_state(ag, A_ALERTED)
            return
        if ag.state == A_ALERTED:
            if self.t < ag.go_t:
                return
            self._set_state(ag, A_EVAC)
        if ag.state == A_SHELTER:
            if self.t < ag.next_replan:
                return
            self._decide(ag)
            if ag.state == A_SHELTER:
                return

        phi = PHI[band(c)]
        budget = dt * phi
        while budget > 0 and ag.state == A_EVAC:
            if ag.path is None:
                self._decide(ag)
                if ag.path is None:
                    return
            nxt = ag.path[ag.path_idx + 1]
            cost = self._step_cost(ag.path[ag.path_idx], nxt)
            need = cost - ag.progress
            if budget < need:
                ag.progress += budget
                return
            budget -= need
            ag.progress = 0.0
            t_in = self.t + (dt * phi - budget) / phi  # when within this sub-step the cell was reached
            self._enter_cell(ag, nxt, t_in)
            ag.path_idx += 1
            if ag.path_idx == len(ag.path) - 1:  # arrived in the target node: next decision
                ag.path = None
                if env.terminal[env.node(ag.s)]:
                    self._exit(ag, t_in)

    def _step_cost(self, a: int, b: int) -> float:
        ce = self.env.cell_env
        if ce.cell_floor[a] != ce.cell_floor[b]:
            return float(-ce.reward[a, STAIRS])
        r0, c0 = ce.cells[a]
        r1, c1 = ce.cells[b]
        return ce.h * (math.sqrt(2.0) if (r0 != r1 and c0 != c1) else 1.0) / ce.v0

    def _enter_cell(self, ag: Agent, s: int, t: float) -> None:
        env = self.env
        u_old, u_new = env.node(ag.s), env.node(s)
        ag.s = s
        ag.trace_t.append(t)
        ag.trace_s.append(s)
        if u_new != u_old:
            ag.entered.append((t, u_new, float(self.gas.node_ppm[u_new])))

    def _exit(self, ag: Agent, t: float) -> None:
        k = self.env.node(ag.s)
        i = self._exit_index[k]
        ag.t_out = t
        ag.exit_name = self.env.nav.node(self.env.node_ids[k]).ref_id
        ag.exit_ppm = float(self.gas.node_ppm[k])
        ag.ap = self.zones.exit_ap[i]
        ag.egress = self.zones.egress_xy[i]
        ag.egress_len = float(np.hypot(*(ag.egress[-1] - ag.egress[0])))
        ag.egress_dist = 0.0
        ag.out_t.append(t)
        ag.out_xy.append(tuple(map(float, ag.egress[0])))
        self.exit_open_until[k] = t + self.sc.exit_door_hold_open_s
        self._set_state(ag, A_OUTSIDE)

    def _walk_outside(self, ag: Agent, dt: float) -> None:
        a, b = ag.egress[0], ag.egress[-1]
        f = ag.egress_dist / ag.egress_len if ag.egress_len > 0 else 1.0
        x, y = a + f * (b - a)
        c = float(self.gas.point_ppm(x, y)[0])
        ag.c_now = c
        ag.dose += c * c * dt / 60.0
        ag.peak = max(ag.peak, c)
        if c >= INCAPACITATION_PPM:
            self._set_state(ag, A_INCAP)
            ag.out_t.append(self.t)
            ag.out_xy.append((float(x), float(y)))
            return
        v = self.env.cell_env.v0 * PHI[band(c)]
        t_end = self.t + dt
        if ag.egress_dist + v * dt >= ag.egress_len:
            ag.t_safe = self.t + (ag.egress_len - ag.egress_dist) / v
            ag.egress_dist = ag.egress_len
            ag.out_t.append(ag.t_safe)
            ag.out_xy.append(tuple(map(float, b)))
            self._set_state(ag, A_SAFE)
            return
        ag.egress_dist += v * dt
        if t_end - ag.out_t[-1] >= 1.0 - 1e-9:  # straight line: a sample a second is exact enough
            f = ag.egress_dist / ag.egress_len
            ag.out_t.append(t_end)
            ag.out_xy.append(tuple(map(float, a + f * (b - a))))

    def _decide(self, ag: Agent) -> None:
        env, gas = self.env, self.gas
        u = env.node(ag.s)
        plan = self.planner.plan(ag.s, self.t, gas.node_ppm, gas.forecast, max_tier=0)
        if plan is None:
            z = int(self.zones.zone_of_node[u])
            here = self.planner.expected_ppm(u, self.t, gas.node_ppm, gas.forecast)
            if z >= 0 and self.zones.zone_is_room[z] and here < self.sc.routing.avoid_ppm:
                if ag.state != A_SHELTER:
                    self._explain(ag, None, "shelter")
                ag.sealed_zone = z  # shut the door and wait
                ag.next_replan = self.t + self.sc.shelter_replan_s
                ag.path = None
                self._set_state(ag, A_SHELTER)
                return
            plan = self.planner.plan(ag.s, self.t, gas.node_ppm, gas.forecast)
        if plan is None or plan.route.first_hop is None:
            ag.next_replan = self.t + self.sc.shelter_replan_s
            ag.path = None
            self._set_state(ag, A_SHELTER)  # nowhere to go: hold position
            return

        ag.sealed_zone = None
        self._set_state(ag, A_EVAC)
        ag.worst_tier = max(ag.worst_tier, plan.tier)
        nodes = plan.route.nodes
        last = ag._last_nodes
        if not (last and len(nodes) <= len(last) and last[-len(nodes):] == nodes):
            rerouted = False
            if last and u in last:
                i = last.index(u)
                if i + 1 < len(last) and last[i + 1] != nodes[1]:
                    ag.reroutes += 1
                    rerouted = True
            if self.gas_aware:
                self._explain(ag, plan, "reroute" if rerouted else "plan" if not last else "update")
            ag.plans.append((self.t, plan.tier, self.planner.astar.route_cells(plan.route)))
            ag._last_nodes = list(nodes)
        ag.plan = plan
        ag.path = env.walker.path(u, nodes[1], ag.s)
        ag.path_idx = 0
        ag.progress = 0.0
        if len(ag.path) < 2:  # already standing in the target node
            ag.path = None
            self._enter_cell(ag, ag.s, self.t)

    def _node_name(self, k: int) -> str:
        nd = self.env.nav.node(self.env.node_ids[k])
        return nd.ref_id or nd.space_id or nd.id

    def _explain(self, ag: Agent, plan: Plan | None, kind: str) -> None:
        """Record why A* chose `plan` (or chose to shelter): compare it with
        the plain fastest route from the same cell and name the first node on
        that route the agent would reach while it is gassed."""
        gas, p = self.gas, self.sc.routing
        fastest = self.planner.astar.plan(ag.s)
        if fastest is None:
            return
        block = None
        for k, tt in zip(fastest.nodes[1:], fastest.times[1:]):
            c = self.planner.expected_ppm(k, self.t + tt, gas.node_ppm, gas.forecast)
            if c >= p.avoid_ppm:
                block = {"node": int(k), "name": self._node_name(k), "eta": round(tt, 1), "ppm": round(c, 1),
                         "now": round(float(gas.node_ppm[k]), 1)}
                break
        same = plan is not None and plan.route.nodes == fastest.nodes
        if kind == "update" and same:
            return  # nothing worth explaining: still on the fastest route
        self.decisions.append({
            "t": round(self.t, 2), "agent": ag.i, "kind": kind if not same else "fastest",
            "tier": None if plan is None else plan.tier,
            "fastestExit": self._node_name(fastest.nodes[-1]),
            "chosenExit": None if plan is None else self._node_name(plan.route.nodes[-1]),
            "extra": None if plan is None else round(plan.route.time_s - fastest.time_s, 1),
            "block": block, "fastestCells": self.planner.astar.route_cells(fastest),
        })

    # ------------------------------------------------------------- summary

    def summary(self) -> dict:
        return {
            "alarm_t": self.gas.alarm.emergency_t, "alarm_detector": self.gas.alarm.trigger_detector,
            "hvac_off_t": self.hvac_off_t, "safe": sum(a.state == A_SAFE for a in self.agents),
            "outside": sum(a.state == A_OUTSIDE for a in self.agents),
            "sheltering": sum(a.state == A_SHELTER for a in self.agents),
            "incapacitated": sum(a.state == A_INCAP for a in self.agents),
            "mean_dose": float(np.mean([a.dose for a in self.agents])),
            "max_peak": float(max(a.peak for a in self.agents)),
            "released_kg": self.gas.puffs.released_mg / 1e6,
        }


# ------------------------------------------------------------------ replay page

Q_MIN, Q_MAX = 0.1, 2000.0  # ppm range of the uint8 log encoding


def quantize(ppm: np.ndarray) -> np.ndarray:
    """0 below Q_MIN, else 1..255 log-spaced over [Q_MIN, Q_MAX]."""
    ppm = np.asarray(ppm, float)
    q = np.log10(np.maximum(ppm, Q_MIN) / Q_MIN) / math.log10(Q_MAX / Q_MIN) * 254 + 1
    return np.where(ppm < Q_MIN, 0, np.clip(np.round(q), 1, 255)).astype(np.uint8)


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _run_payload(sim: Simulation) -> dict:
    agents = []
    for a in sim.agents:
        agents.append({
            "i": a.i, "room": a.room, "name": a.name, "floor": a.floor, "start": a.start,
            "traceT": [round(t, 2) for t in a.trace_t], "traceS": a.trace_s,
            "states": [[round(t, 2), AGENT_STATES.index(s)] for t, s in a.states],
            "plans": [[round(t, 2), tier, cells] for t, tier, cells in a.plans],
            "outT": [round(t, 2) for t in a.out_t], "outXY": [[round(x, 2), round(y, 2)] for x, y in a.out_xy],
            "tOut": a.t_out, "tSafe": a.t_safe, "exit": a.exit_name, "ap": a.ap, "exitPpm": a.exit_ppm,
            "tAlert": a.t_alert,
            "peak": a.peak, "dose": a.dose, "reroutes": a.reroutes, "worstTier": a.worst_tier,
            "final": AGENT_STATES.index(a.state),
        })
    return {
        "nodePpm": _b64(quantize(np.array(sim.rec_node_ppm))),
        "nodeSoon": _b64(quantize(np.array(sim.rec_node_soon))),
        "agentPpm": _b64(np.array([a.rec_ppm for a in sim.agents], dtype=np.float32)),
        "agentDose": _b64(np.array([a.rec_dose for a in sim.agents], dtype=np.float32)),
        "decisions": sim.decisions,
        "detPpm": _b64(np.array(sim.rec_det_ppm, dtype=np.float32)),
        "hvac": [int(h) for h in sim.rec_hvac],
        "agents": agents, "summary": sim.summary(),
    }


def build_payload(aware: Simulation, blind: Simulation | None) -> dict:
    env, sc, b = aware.env, aware.sc, aware.building
    if abs(b.rotation_deg) > 1e-9:
        raise NotImplementedError("replay page draws axis-aligned buildings only")
    ce = env.cell_env
    floors = []
    for f, fid in enumerate(ce.floor_ids):
        labels = []
        for sp in b.get_floor(fid).spaces.values():
            if sp.kind in ("room", "stair"):
                r, c = ce.grid.transform.local_to_cell(*sp.rect.center)
                labels.append({"id": sp.id, "r": r, "c": c})
        floors.append({"id": fid, "cat": _b64(_categories(env, f)), "labels": labels})
    nodes = []
    for nid in env.node_ids:
        nd = env.nav.node(nid)
        name = nd.ref_id if nd.kind_detail in ("interior", "fire_door", "exit", "stair_landing") else nd.space_id
        nodes.append({"id": nid, "kind": NODE_KINDS[nd.kind_detail], "name": name or nid})

    gas = aware.gas
    site = sc.site
    fp = b.get_floor(ce.floor_ids[0]).footprint
    every = 2  # field frames every 2 s (the page blends between them)
    field_frames = np.array(aware.rec_field[every - 1::every])
    return {
        "scenario": sc.name, "duration": sc.duration_s, "dt": sc.gas_dt_s,
        "rows": ce.grid.shape[0], "cols": ce.grid.shape[1], "h": ce.h, "v0": ce.v0,
        "gridOrigin": list(ce.grid.transform.origin_local),
        "nCells": ce.n_states, "nNodes": env.n_nodes,
        "cellFloor": _b64(ce.cell_floor.astype(np.uint8)),
        "cellR": _b64(ce.cells[:, 0].astype(np.int16)), "cellC": _b64(ce.cells[:, 1].astype(np.int16)),
        "nodeOf": _b64(env.node_of_state.astype(np.int32)),
        "stairCost": _b64((-ce.reward[:, STAIRS]).astype(np.float32)),
        "nodes": nodes, "floors": floors,
        "building": {"id": b.id, "origin": list(b.origin), "w": fp.width, "hgt": fp.height},
        "site": {
            "w": site.width_m, "hgt": site.height_m,
            "aps": [{"id": a.id, "x": a.x, "y": a.y, "d": a.diameter_m} for a in site.assembly_points.values()],
            "obstacles": [{"id": o.id, "x0": o.rect.x0, "y0": o.rect.y0, "x1": o.rect.x1, "y1": o.rect.y1}
                          for o in site.obstacles.values() if o.active_by_default],
            "sources": [{"id": s.id, "x": s.x, "y": s.y, "rate": s.rate_gps} for s in gas.puffs.sources],
            "detectors": [{"id": d.id, "x": d.x, "y": d.y, "indoor": d.node is not None} for d in gas.detectors],
            "exits": [{"node": int(k), "name": env.nav.node(env.node_ids[k]).ref_id, "x": float(x), "y": float(y),
                       "ap": ap, "path": [[float(p[0]), float(p[1])] for p in (pts[0], pts[-1])]}
                      for k, (x, y), ap, pts in zip(aware.zones.exit_nodes, aware.zones.exit_xy,
                                                     aware.zones.exit_ap, aware.zones.egress_xy)],
        },
        "field": {"rows": gas.field_shape[0], "cols": gas.field_shape[1], "h": sc.gas_cfg.field_h,
                  "x0": gas.field_origin[0], "y0": gas.field_origin[1], "every": every,
                  "n": len(field_frames), "data": _b64(quantize(field_frames))},
        "forecast": [{"t": t0, "data": _b64(quantize(fm))} for t0, fm in aware.rec_forecast],
        "t": aware.rec_t,
        "alarm": [(NORMAL, WARNING, EMERGENCY).index(s) for s in aware.rec_alarm],
        "wind": [[round(u, 2), round(d, 1)] for u, d in aware.rec_wind],
        "runs": {"aware": _run_payload(aware), **({"blind": _run_payload(blind)} if blind else {})},
        "thresholds": {"avoid": sc.routing.avoid_ppm, "hard": sc.routing.hard_ppm,
                       "alarm": sc.gas_cfg.alarm.alarm_ppm, "warning": sc.gas_cfg.alarm.warning_ppm,
                       "qMin": Q_MIN, "qMax": Q_MAX},
        "releaseRate": sum(s.rate_gps for s in gas.puffs.sources),
        "detectorIds": [d.id for d in gas.detectors],
    }


def write_page(payload: dict, out: Path) -> Path:
    html = TEMPLATE.read_text().replace('"__DATA__"', json.dumps(payload, separators=(",", ":")))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out


# ------------------------------------------------------------------ CLI


def _route_text(sim: Simulation, a: Agent) -> str:
    if not a.trace_s:
        return "-"
    parts: list[str] = []
    for s in a.trace_s:
        nd = sim.env.nav.node(sim.env.node_ids[sim.env.node(s)])
        if nd.kind_detail in ("interior", "exit", "stair_landing"):
            if not parts or parts[-1] != nd.ref_id:
                parts.append(nd.ref_id)
    return " > ".join(parts[-4:]) if parts else "-"


def print_report(aware: Simulation, blind: Simulation | None) -> None:
    s = aware.summary()
    print(f"\nAlarm: EMERGENCY at t={s['alarm_t']}s (detector {s['alarm_detector']}); HVAC off at t={s['hvac_off_t']}s; "
          f"{s['released_kg']:.1f} kg released")
    hdr = (f"{'agent':10s} {'state':13s} {'out at':>7s} {'exit':5s} {'path ppm':>8s} {'peak':>7s} {'dose':>9s} "
           f"{'reroutes':>8s}")
    if blind:
        hdr += f"   | blind: {'state':13s} {'exit':5s} {'path ppm':>8s} {'peak':>7s} {'dose':>9s}"
    print(hdr)
    fmt = lambda v: f"{v:.1f}" if v is not None else "-"  # noqa: E731
    for i, a in enumerate(aware.agents):
        line = (f"{a.floor}:{a.room:6s} {a.state:13s} {(f'{a.t_out:.0f}s' if a.t_out else '-'):>7s} "
                f"{a.exit_name or '-':5s} {fmt(a.exit_ppm):>8s} {a.peak:7.1f} {a.dose:9.1f} {a.reroutes:8d}")
        if blind:
            b = blind.agents[i]
            line += (f"   |        {b.state:13s} {b.exit_name or '-':5s} {fmt(b.exit_ppm):>8s} "
                     f"{b.peak:7.1f} {b.dose:9.1f}")
        print(line)
    print("path ppm: worst air on the exit's egress path when they stepped out; peak: ppm; "
          "dose: ppm²·min (L = Σ C² dt / 60)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--out", type=Path, default=Path("out/gas_sim.html"))
    ap.add_argument("--no-blind", action="store_true", help="skip the gas-blind comparison run")
    args = ap.parse_args()

    sc = load_scenario(args.scenario)
    env = sc.make_env()
    print(f"Scenario {sc.name}: {sc.duration_s:.0f} s, {len(sc.spawns)} agents, graph {env.n_nodes} nodes "
          f"({int(env.terminal.sum())} exits), planner: A*")
    t0 = time.time()
    aware = Simulation(sc, env, gas_aware=True).run()
    print(f"Gas-aware run: {time.time() - t0:.1f}s")
    blind = None
    if sc.compare_gas_blind and not args.no_blind:
        t0 = time.time()
        blind = Simulation(sc, env, gas_aware=False).run()
        print(f"Gas-blind run: {time.time() - t0:.1f}s")
    print_report(aware, blind)
    path = write_page(build_payload(aware, blind), args.out)
    print(f"\nWrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
