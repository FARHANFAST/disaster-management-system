"""One evacuee: Perception -> Decision -> Action, behind `step(snapshot, dt)`
(design doc v0.3 §4-§8).

Perception is local. The agent reads the air where it stands, the gas and
the doors it can see within its vision radius, and the people inside its
sensing bubble. It plans over the exits *it knows*. Doors it cannot see are
assumed open; a blocked or gassed exit is discovered only on sight, or from
a responder in range.

Decision, at each graph node: score every neighbouring node v

    S(v) = W1 K(v) + W2 F(v) + W3 H(v) - backtrack penalty
    K    knowledge: progress toward the nearest *known* usable exit, blended
         with exit signage by familiarity: fam K_known + (1 - fam) K_sign
    F    social: share of visible moving people heading to v (VIP: the escort)
    H    hazard: 1 - min(1, C_seen(v) / 100 ppm), C unseen = 0
    pick v ~ softmax(S / T),  T = T0 + T_stress * stress

Action: the walker path into v, walked at the agent's own speed x health
factor (x panic boost). There are no collisions yet: this movement is a
labelled stand-in for the social-force orchestrator (SYSTEM_SPEC §2.1).
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

from ..rl.env import STAIRS
from .config import BehaviourConfig, PhysiologyConfig
from .physiology import Body, band
from .population import Profile
from .weights import WeightState
from .world import GraphInfo, Snapshot

NORMAL, ALERT, DECIDING, MOVING, PANICKED, TRAPPED, AT_ASSEMBLY, INCAPACITATED = (
    "NORMAL", "ALERT", "DECIDING", "MOVING_TO_ASSEMBLY", "PANICKED", "TRAPPED", "AT_ASSEMBLY", "INCAPACITATED")
STATES = (NORMAL, ALERT, DECIDING, MOVING, PANICKED, TRAPPED, AT_ASSEMBLY, INCAPACITATED)
ACTIVE = (DECIDING, MOVING, PANICKED, TRAPPED)  # people visibly on the move
FINAL = (AT_ASSEMBLY, INCAPACITATED)


@dataclass(slots=True)
class Action:
    agent_id: int
    state: str
    cell: int
    target_node: int | None
    speed: float
    arrived: bool
    incapacitated: bool


class Agent:
    def __init__(self, profile: Profile, start_cell: int, gi: GraphInfo, beh: BehaviourConfig,
                 phys: PhysiologyConfig, seed: int, egress_m: dict[int, float]):
        self.p = profile
        self.gi = gi
        self.cfg = beh
        self.rng = np.random.default_rng([seed, profile.agent_id])  # independent of update order
        self.body = Body(phys, susceptibility=profile.susceptibility, health=profile.health0)
        self.w = WeightState(profile, beh)
        self.state = NORMAL
        self.s = start_cell
        self.node = gi.env.node(start_cell)
        self.prev_node: int | None = None
        self.path: list[int] | None = None
        self.path_idx = 0
        self.progress = 0.0
        self.target_node: int | None = None
        self.target_exit: int | None = None
        self.known = {gi.exit_of_name[e] for e in profile.known_exits if e in gi.exit_of_name}
        self.blocked_seen: set[int] = set()  # exits found locked: permanent
        self.gassed: dict[int, float] = {}  # exit -> when it was last believed gassed (fades)
        self.bad: set[int] = set()  # exits this agent currently believes unusable
        self.stress = 0.0
        self.t_state = 0.0
        self.decide_t = math.inf
        self.alert_t: float | None = None
        self.safe_t: float | None = None
        self.last_move_t = 0.0
        self.panic_since: float | None = None  # 3+ panicked neighbours since
        self.calm_since: float | None = None
        self.panic_calm_since: float | None = None
        self.smelt_at: float | None = None
        self.air: deque[tuple[float, float]] = deque()
        self.c_now = 0.0
        self.outside_left: float | None = None  # metres of egress walk left
        self.egress_m = egress_m
        self._others: list[int] = []
        self._n_panicked = 0
        self.exit_used: str | None = None
        self.reroutes = 0
        self.panics = 0
        self.persona_shifts = 0
        self.route: list[int] = [self.node]
        self.events: list[dict] = []

    # ------------------------------------------------------------- helpers

    @property
    def xy(self) -> np.ndarray:
        return self.gi.cell_xy[self.s]

    @property
    def floor(self) -> int:
        return int(self.gi.env.cell_env.cell_floor[self.s])

    @property
    def done(self) -> bool:
        return self.state in FINAL

    def _log(self, t: float, event: str, **kw) -> None:
        w = self.w.weights(self.state == PANICKED)
        self.events.append(dict(t=round(t, 2), agent=self.p.agent_id, event=event, state=self.state,
                                persona=self.w.persona, node=self.gi.env.node_ids[self.node],
                                ppm=round(self.c_now, 2), dose=round(self.body.dose, 3),
                                stress=round(self.stress, 3), health=self.body.health,
                                W1=round(float(w[0]), 3), W2=round(float(w[1]), 3), W3=round(float(w[2]), 3), **kw))

    def _set(self, t: float, state: str, rule: str, **kw) -> None:
        if state != self.state:
            old = self.state
            self.state = state
            self.t_state = t
            self._log(t, "state", from_state=old, rule=rule, **kw)

    def _usable(self) -> list[int]:
        """Known exits believed usable; if every one seems gassed, the known
        ones that are not locked (people still leave by the least-bad door)."""
        ok = [k for k in self.known if k not in self.bad]
        return ok or [k for k in self.known if k not in self.blocked_seen]

    def _bad_ppm(self) -> float:
        if self.p.responder:
            return self.cfg.responder_avoid_ppm
        return self.cfg.avoid_base_ppm / max(self.w.weights()[2], 1e-3)

    # ------------------------------------------------------------- the step

    def step(self, snap: Snapshot, dt: float) -> Action:
        if not self.done:
            self._perceive_body(snap, dt)
        if not self.done:
            self._perceive_world(snap)
            self._transition(snap, dt)
        return Action(self.p.agent_id, self.state, self.s, self.target_node, self._speed(),
                      self.state == AT_ASSEMBLY, self.state == INCAPACITATED)

    def _perceive_body(self, snap: Snapshot, dt: float) -> None:
        t = snap.t
        k = self.node
        c = snap.ppm(k)
        self.c_now = c
        if self.body.breathe(c, dt, self.rng):
            self.path = None
            self._set(t, INCAPACITATED, "T13 collapse")
            return
        b = band(c * self.p.susceptibility, self.body.cfg)
        n_pan = sum(1 for j in snap.hash.near(self.p.agent_id, self.p.vision_m) if snap.agents[j].state == PANICKED)
        self._n_panicked = n_pan
        cfg = self.cfg
        self.stress = float(np.clip(self.stress + dt * (cfg.stress_band_rate[b]
                                                         + cfg.stress_per_panicked * min(n_pan, cfg.panic_neighbours)
                                                         - cfg.stress_decay * self.stress), 0.0, 1.0))
        self.w.relax(dt, self.stress)
        self.air.append((t, c))
        while self.air and self.air[0][0] < t - cfg.spike_window_s:
            self.air.popleft()
        if c >= cfg.smell_ppm and (self.smelt_at is None or c - self.smelt_at >= cfg.smell_rise_ppm):
            self.smelt_at = c
            self.w.smelt_gas()
            self._log(t, "smelt_gas")

    def _perceive_world(self, snap: Snapshot) -> None:
        """Exits seen (learnt, or found blocked / gassed), crowd cues, responder broadcast."""
        t, gi, me = snap.t, self.gi, self.p
        if self.outside_left is not None:
            return
        thr = self._bad_ppm()
        for k in [k for k, t0 in self.gassed.items() if t - t0 > self.cfg.gas_memory_s]:
            del self.gassed[k]
        if me.responder:  # radio: the current forecast at every exit, refreshed every tick
            for k in gi.exit_nodes:
                if snap.soon(k) >= thr:
                    self._mark_bad(t, k, "forecast (radio)")
                elif snap.soon(k) < 0.5 * thr:  # hysteresis: no flip-flop around the threshold
                    self.gassed.pop(k, None)
        for k in gi.exit_nodes:
            if gi.node_distance(self.xy, self.floor, k) <= me.vision_m:
                if k not in self.known:
                    self.known.add(k)
                    self._log(t, "learnt_exit", exit=gi.exit_name[k], how="seen")
                if k in snap.blocked_exits:
                    self._mark_bad(t, k, "blocked", blocked=True)
                elif snap.ppm(k) >= thr:
                    self._mark_bad(t, k, "gassed (seen)")
                elif k in self.gassed and not me.responder:
                    del self.gassed[k]  # seen with clean air: belief corrected
                    self._log(t, "exit_clear", exit=gi.exit_name[k])
        self.bad = self.blocked_seen | set(self.gassed)
        others = snap.hash.near(me.agent_id, me.vision_m)
        heading: dict[int, int] = {}
        for j in others:
            a = snap.agents[j]
            if a.state in (MOVING, PANICKED) and a.target_exit is not None:
                heading[a.target_exit] = heading.get(a.target_exit, 0) + 1
            if a.p.responder and not me.responder and a.state in ACTIVE and self.state != NORMAL \
                    and np.hypot(*(a.xy - self.xy)) <= a.p.vision_m:
                for k in a.bad - self.bad:  # directed reroute broadcast (EVAC-SIM slide 25)
                    self._mark_bad(t, k, "told by responder", blocked=k in snap.blocked_exits, hearsay=True)
        for k, cnt in heading.items():
            if k not in self.known and cnt > 2:
                self.known.add(k)
                self._log(t, "learnt_exit", exit=gi.exit_name[k], how="followed others")
        self._others = others

    def _mark_bad(self, t: float, k: int, why: str, *, blocked: bool = False, hearsay: bool = False) -> None:
        new = k not in self.bad
        if blocked:
            self.blocked_seen.add(k)
        elif not (hearsay and k in self.gassed):  # a repeated rumour does not keep a belief alive
            self.gassed[k] = t
        self.bad.add(k)
        self.known.add(k)
        if not new:
            return
        was_target = k == self.target_exit
        self._log(t, "exit_unusable", exit=self.gi.exit_name[k], why=why, was_target=was_target)
        if was_target and self.state in ACTIVE:
            self.w.route_failed()
            self.reroutes += 1
            self.stress = min(1.0, self.stress + self.cfg.stress_fail / self.reroutes)
            self._log(t, "route_failed", exit=self.gi.exit_name[k])
            self.path = None  # re-decide at once

    # ------------------------------------------------------------- state machine

    def _transition(self, snap: Snapshot, dt: float) -> None:
        t, cfg, me = snap.t, self.cfg, self.p
        others = self._others
        if self.state == NORMAL:
            n_alert = sum(1 for j in others if snap.agents[j].state != NORMAL)
            need = 1 if self.w.persona == "Follower" else cfg.alert_neighbours
            resp = any(snap.agents[j].p.responder and snap.agents[j].state in ACTIVE for j in others)
            rule = ("T1 alarm" if snap.alarm else "T1 smelt gas" if self.c_now >= cfg.smell_ppm
                    else "T1 neighbours alert" if n_alert >= need else "T1 responder" if resp else None)
            if rule:
                self.alert_t = t
                self.decide_t = t + me.reaction_s
                self._set(t, ALERT, rule)
            return
        if self.state == ALERT:
            if t >= self.decide_t:
                self._set(t, DECIDING, "T2 reaction time over")
            else:
                return
        if self.state == TRAPPED:
            if t - self.t_state >= cfg.trapped_wait_s:
                self._set(t, DECIDING, "T9 re-evaluate")
            else:
                return
        if self.state == DECIDING:
            self.last_move_t = t
            if self._decide(snap):
                self._set(t, MOVING, "T3 route chosen", next=self.gi.env.node_ids[self.target_node],
                          exit=self.gi.exit_name.get(self.target_exit, "-"))
            else:
                self._set(t, TRAPPED, "T6 no open path")
            if self.state != MOVING:
                return
        self._panic_and_persona(snap)
        self._move(snap, dt)

    def _panic_and_persona(self, snap: Snapshot) -> None:
        t, cfg, me = snap.t, self.cfg, self.p
        spike = bool(self.air) and self.c_now - min(c for _, c in self.air) >= cfg.spike_ppm
        if self.state == MOVING and (self.stress >= me.panic_threshold or spike):
            self.panics += 1
            self.panic_calm_since = None
            self._set(t, PANICKED, "T5 gas spike" if spike else "T5 stress over threshold")
        elif self.state == PANICKED:
            if self.stress < me.panic_threshold - cfg.panic_recover_margin:
                self.panic_calm_since = self.panic_calm_since if self.panic_calm_since is not None else t
                if t - self.panic_calm_since >= cfg.panic_recover_s:
                    self._set(t, MOVING, "T9 calmed down")
            else:
                self.panic_calm_since = None
        if me.responder:
            return
        if self._n_panicked >= cfg.panic_neighbours:
            self.calm_since = None
            self.panic_since = self.panic_since if self.panic_since is not None else t
            if self.w.persona == "Rationalist" and t - self.panic_since >= cfg.persona_shift_hold_s:
                self.w.set_persona("Follower")
                self.persona_shifts += 1
                self._log(t, "persona_shift", to="Follower")
        else:
            self.panic_since = None
            if self.w.persona != me.persona:
                self.calm_since = self.calm_since if self.calm_since is not None else t
                if t - self.calm_since >= cfg.persona_revert_s:
                    self.w.set_persona(me.persona)
                    self._log(t, "persona_shift", to=me.persona)

    # ------------------------------------------------------------- decision

    def _scores(self, snap: Snapshot, u: int, cands: list[int]) -> tuple[np.ndarray, np.ndarray]:
        gi, me, cfg = self.gi, self.p, self.cfg
        usable = self._usable()
        d_known = gi.dist_to(usable)
        signed = [k for k in gi.exit_nodes if k not in self.bad] or [k for k in gi.exit_nodes
                                                                      if k not in self.blocked_seen]
        d_sign = gi.dist_to(signed) if cfg.signage else None

        def prog(d: np.ndarray, v: int) -> float:
            if not np.isfinite(d[u]):
                return 0.5
            if not np.isfinite(d[v]):
                return 0.0
            return float(np.clip(0.5 + 0.5 * (d[u] - d[v]) / gi.edge_s[(u, v)], 0.0, 1.0))

        fam = 1.0 if me.responder else me.familiarity
        moving = [snap.agents[j] for j in self._others if snap.agents[j].state in (MOVING, PANICKED)]
        escort = snap.agents[me.escort] if me.vip and me.escort is not None else None
        escort_seen = escort is not None and not escort.done and escort.floor == self.floor \
            and np.hypot(*(escort.xy - self.xy)) <= max(me.vision_m, escort.p.vision_m)
        K, F, H = [], [], []
        for v in cands:
            K.append(fam * prog(d_known, v) + (1 - fam) * (prog(d_sign, v) if d_sign is not None else 0.5))
            if escort_seen:
                F.append(1.0 if v in (escort.target_node, escort.node) else 0.0)
            else:
                F.append(sum(1 for a in moving if a.target_node == v) / len(moving) if moving else 0.0)
            seen = gi.node_distance(self.xy, self.floor, v) <= me.vision_m or v == u
            c = snap.soon(v) if me.responder else (snap.ppm(v) if seen else 0.0)
            H.append(1.0 - min(1.0, c / cfg.hazard_ref_ppm))
        w = self.w.weights(self.state == PANICKED)
        terms = np.array([K, F, H])
        S = w @ terms - cfg.backtrack_penalty * np.array([v == self.prev_node for v in cands], dtype=float)
        return S, terms

    def _decide(self, snap: Snapshot) -> bool:
        t, gi, u = snap.t, self.gi, self.node
        cands = []
        for v in gi.nbrs[u]:
            if v in self.bad:
                continue
            if v in snap.blocked_exits:  # standing next to it: it is in plain sight
                self._mark_bad(t, v, "blocked", blocked=True)
                continue
            if gi.env.walker.leg(u, v, self.s) is None:
                continue
            cands.append(v)
        if not cands:
            self.target_node = None
            return False
        S, terms = self._scores(snap, u, cands)
        T = self.cfg.temp0 + self.cfg.temp_stress * self.stress
        p = np.exp((S - S.max()) / T)
        p /= p.sum()
        v = cands[int(self.rng.choice(len(cands), p=p))]
        # crowd cue: most of the people I can see go somewhere my knowledge would not
        moving = [snap.agents[j] for j in self._others if snap.agents[j].state in (MOVING, PANICKED)]
        if len(moving) >= self.cfg.crowd_min_seen:
            best_k = cands[int(np.argmax(terms[0]))]
            tgt = [a.target_node for a in moving]
            m = max(set(tgt), key=tgt.count)
            if m in cands and m != best_k and tgt.count(m) / len(tgt) >= self.cfg.crowd_majority:
                self.w.crowd_elsewhere()
        usable = self._usable()
        self.target_exit = min(usable, key=lambda k: gi.dist[k][v]) if usable else None
        self.target_node = v
        self.path = gi.env.walker.path(u, v, self.s)
        self.path_idx = 0
        self.progress = 0.0
        return True

    # ------------------------------------------------------------- action

    def _speed(self) -> float:
        if self.state not in (MOVING, PANICKED):
            return 0.0
        v = self.p.speed * self.body.speed_factor(self.c_now)
        if self.state == PANICKED:
            v *= self.cfg.panic_speed
        return min(v, self.cfg.max_speed)

    def _step_cost(self, a: int, b: int) -> float:
        ce = self.gi.env.cell_env
        if ce.cell_floor[a] != ce.cell_floor[b]:
            return float(-ce.reward[a, STAIRS])
        r0, c0 = ce.cells[a]
        r1, c1 = ce.cells[b]
        return ce.h * (math.sqrt(2.0) if (r0 != r1 and c0 != c1) else 1.0) / ce.v0

    def _move(self, snap: Snapshot, dt: float) -> None:
        t, gi = snap.t, self.gi
        speed = self._speed()
        if self.outside_left is not None:  # walking the egress path to the assembly point
            self.outside_left -= dt * speed
            if self.outside_left <= 0:
                self.safe_t = t + dt
                self.outside_left = None
                self._set(t + dt, AT_ASSEMBLY, "T7 reached assembly point", exit=self.exit_used)
            return
        budget = dt * speed / gi.env.cell_env.v0  # in v0-seconds of walking
        guard = 0
        if budget > 0:
            self.last_move_t = t  # walking, however slowly, is progress (crowd jams come with the SFM)
        while budget > 0 and self.state in (MOVING, PANICKED) and self.outside_left is None and guard < 50:
            guard += 1
            if self.path is None or self.path_idx >= len(self.path) - 1:
                if not self._decide(snap):
                    self._set(t, TRAPPED, "T6 no open path")
                    return
                if len(self.path) < 2:
                    self._arrive(t, self.target_node)
                    continue
            need = self._step_cost(self.path[self.path_idx], self.path[self.path_idx + 1]) - self.progress
            if budget < need:
                self.progress += budget
                break
            budget -= need
            self.progress = 0.0
            self.path_idx += 1
            self.s = self.path[self.path_idx]
            self.last_move_t = t
            new = gi.env.node(self.s)
            if new != self.node:
                self._arrive(t, new)
        if t - self.last_move_t >= self.cfg.no_progress_s and self.state in (MOVING, PANICKED):
            self._set(t, TRAPPED, "T6 no progress")

    def _arrive(self, t: float, node: int) -> None:
        gi = self.gi
        self.prev_node, self.node = self.node, node
        self.route.append(node)
        self.last_move_t = t
        self.path = None
        if gi.env.terminal[node]:
            self.exit_used = gi.exit_name[node]
            self.outside_left = self.egress_m.get(node, 0.0)
            self._log(t, "left_building", exit=self.exit_used)
