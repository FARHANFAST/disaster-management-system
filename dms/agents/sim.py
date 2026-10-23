"""Run the agent behaviour model: 500-600 heterogeneous agents in B1 under
the H2S scenario of dms/gas (design doc v0.3 §9).

    python -m dms.agents.sim                          # 550 agents, default leak
    python -m dms.agents.sim --agents 600 --block X-N # an exit locked (found only on arrival)
    python -m dms.agents.sim --compare                # also run with weight adaptation off

Each tick (dt = 0.5 s) the orchestrator freezes a read-only view of every
agent, so update order cannot bias the result; then every agent calls
step(snapshot, dt). Outputs: out/agents_telemetry.csv (one row per event /
state change), out/agents_summary.csv (one row per agent) and a printed
breakdown by age, sex, role/VIP and persona.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..gas.scenario import load_scenario
from ..gas.simulate import DEFAULT_SCENARIO
from ..gas.timeline import GasTimeline
from ..rl.graph_env import GraphEvacEnv
from .agent import AT_ASSEMBLY, INCAPACITATED, Agent
from .config import AgentModelConfig
from .population import Profile, sample_population
from .world import GraphInfo, Snapshot, SpatialHash


@dataclass(slots=True, frozen=True)
class AgentView:
    """What other agents may read about an agent during one tick."""
    p: Profile
    state: str
    node: int
    target_node: int | None
    target_exit: int | None
    xy: np.ndarray
    floor: int
    bad: frozenset[int]
    done: bool


class AgentSimulation:
    def __init__(self, cfg: AgentModelConfig | None = None, *, scenario: Path = DEFAULT_SCENARIO,
                 blocked: tuple[str, ...] = (), env: GraphEvacEnv | None = None,
                 timeline: GasTimeline | None = None):
        self.cfg = cfg = cfg or AgentModelConfig()
        sc = load_scenario(scenario)
        sc.duration_s = cfg.duration_s
        self.env = env = env or sc.make_env()
        self.gi = gi = GraphInfo(env)
        self.timeline = tl = timeline or GasTimeline.from_scenario(sc, env)
        self.blocked = frozenset(gi.exit_of_name[x] for x in blocked)
        egress_m = {int(k): float(np.hypot(*(pts[-1] - pts[0])))
                    for k, pts in zip(tl.zones.exit_nodes, tl.zones.egress_xy)}
        rng = np.random.default_rng(cfg.seed)
        self.population = pop = sample_population(cfg.population, [gi.exit_name[k] for k in gi.exit_nodes], rng)
        sizes = np.array([len(env.cells_of_node[k]) for k in gi.rooms], dtype=float)
        room_of = [gi.rooms[int(rng.choice(len(gi.rooms), p=sizes / sizes.sum()))] for _ in pop]
        for p in pop:  # a VIP guest is with its escort
            if p.vip and p.escort is not None:
                room_of[p.agent_id] = room_of[p.escort]
        self.agents = [Agent(p, int(rng.choice(env.cells_of_node[room_of[p.agent_id]])), gi,
                             cfg.behaviour, cfg.physiology, cfg.seed, egress_m) for p in pop]
        self.hash = SpatialHash(max(p.vision_m for p in pop))
        self.t = 0.0

    def _views(self) -> list[AgentView]:
        return [AgentView(a.p, a.state, a.node, a.target_node, a.target_exit, a.xy.copy(), a.floor,
                          frozenset(a.bad), a.done) for a in self.agents]

    def run(self) -> AgentSimulation:
        cfg, dt = self.cfg, self.cfg.behaviour.dt
        n_steps = int(round(cfg.duration_s / dt))
        for _ in range(n_steps):
            views = self._views()
            xy = np.array([v.xy for v in views])
            active = np.array([not v.done and self.agents[i].outside_left is None for i, v in enumerate(views)])
            self.hash.rebuild(xy, np.array([v.floor for v in views]), active)
            snap = Snapshot(t=self.t, alarm=self.t >= self.timeline.alarm_t, timeline=self.timeline,
                            blocked_exits=self.blocked, hash=self.hash, agents=views)
            for a in self.agents:
                a.step(snap, dt)
            self.t += dt
            if all(a.done for a in self.agents):
                break
        return self

    # ------------------------------------------------------------- outputs

    def telemetry(self) -> list[dict]:
        return sorted((e for a in self.agents for e in a.events), key=lambda e: (e["t"], e["agent"]))

    def agent_rows(self) -> list[dict]:
        alarm = self.timeline.alarm_t
        rows = []
        for a in self.agents:
            p = a.p
            rows.append(dict(
                agent=p.agent_id, age=p.age, sex=p.sex, role=p.role, vip=p.vip, responder=p.responder,
                persona=p.persona, persona_end=a.w.persona, speed=round(p.speed, 3),
                reaction_s=round(p.reaction_s, 1), familiarity=round(p.familiarity, 2), drills=p.drills,
                known_exits_start="|".join(sorted(p.known_exits)), final_state=a.state, exit=a.exit_used or "",
                t_alert=a.alert_t, t_safe=a.safe_t,
                time_to_safety=round(a.safe_t - alarm, 1) if a.safe_t is not None else None,
                dose=round(a.body.dose, 3), peak_ppm=round(a.body.peak, 2), health=a.body.health,
                reroutes=a.reroutes, panics=a.panics, persona_shifts=a.persona_shifts))
        return rows

    def write(self, out_dir: Path) -> tuple[Path, Path]:
        out_dir.mkdir(parents=True, exist_ok=True)
        tel, summ = out_dir / "agents_telemetry.csv", out_dir / "agents_summary.csv"
        events = self.telemetry()
        keys = sorted({k for e in events for k in e}, key=lambda k: (k not in ("t", "agent", "event", "state"), k))
        with tel.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(events)
        rows = self.agent_rows()
        with summ.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        return tel, summ


def group_table(rows: list[dict], key) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = []
    for g, rs in sorted(groups.items()):
        tts = np.array([r["time_to_safety"] for r in rs if r["time_to_safety"] is not None], dtype=float)
        exits: dict[str, int] = {}
        for r in rs:
            if r["exit"]:
                exits[r["exit"]] = exits.get(r["exit"], 0) + 1
        out.append(dict(
            group=g, n=len(rs), safe=sum(r["final_state"] == AT_ASSEMBLY for r in rs) / len(rs),
            incap=sum(r["final_state"] == INCAPACITATED for r in rs),
            stuck=sum(r["final_state"] not in (AT_ASSEMBLY, INCAPACITATED) for r in rs),
            tts_mean=float(tts.mean()) if tts.size else float("nan"),
            tts_p95=float(np.percentile(tts, 95)) if tts.size else float("nan"),
            dose=float(np.mean([r["dose"] for r in rs])), peak=float(np.mean([r["peak_ppm"] for r in rs])),
            reroutes=float(np.mean([r["reroutes"] for r in rs])), panics=sum(r["panics"] > 0 for r in rs),
            exits=" ".join(f"{k}:{v}" for k, v in sorted(exits.items(), key=lambda kv: -kv[1]))))
    return out


def print_tables(rows: list[dict], title: str) -> None:
    print(f"\n### {title}")
    splits = (("age", lambda r: r["age"]), ("sex", lambda r: r["sex"]),
              ("role / VIP", lambda r: "VIP" if r["vip"] else ("Responder" if r["responder"] else r["role"])),
              ("persona (start)", lambda r: r["persona"]), ("all", lambda r: "all"))
    for name, key in splits:
        print(f"\n{name:16s} {'n':>4s} {'safe':>6s} {'incap':>5s} {'stuck':>5s} {'TTS mean':>8s} {'p95':>6s} "
              f"{'dose':>7s} {'peak':>6s} {'rerout':>6s} {'panic':>5s}  exits used")
        for g in group_table(rows, key):
            print(f"{g['group']:16s} {g['n']:4d} {g['safe']:6.1%} {g['incap']:5d} {g['stuck']:5d} "
                  f"{g['tts_mean']:8.1f} {g['tts_p95']:6.1f} {g['dose']:7.2f} {g['peak']:6.1f} "
                  f"{g['reroutes']:6.2f} {g['panics']:5d}  {g['exits']}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--agents", type=int, default=550)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--block", action="append", default=[], metavar="EXIT_ID",
                    help="exit that is locked; agents only find out when they see it")
    ap.add_argument("--no-adapt", action="store_true", help="freeze W1..W3 at their base values")
    ap.add_argument("--compare", action="store_true", help="also run the same people with adaptation off")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    args = ap.parse_args()

    cfg = AgentModelConfig(duration_s=args.duration, seed=args.seed)
    cfg.population.n_agents = args.agents
    cfg.behaviour.adapt_weights = not args.no_adapt
    t0 = time.time()
    sim = AgentSimulation(cfg, scenario=args.scenario, blocked=tuple(args.block))
    print(f"{args.agents} agents, {sim.env.n_nodes}-node graph, gas history {sim.timeline.duration_s:.0f} s "
          f"(alarm {sim.timeline.alarm_t:.0f} s), blocked {args.block or '-'}; set up in {time.time() - t0:.1f}s")
    t0 = time.time()
    sim.run()
    print(f"Simulated {sim.t:.0f} s in {time.time() - t0:.1f}s")
    rows = sim.agent_rows()
    print_tables(rows, "adaptive weights" if cfg.behaviour.adapt_weights else "weights fixed")
    tel, summ = sim.write(args.out_dir)
    print(f"\nWrote {tel} ({len(sim.telemetry())} events) and {summ}")
    if args.compare:
        cfg2 = AgentModelConfig(duration_s=args.duration, seed=args.seed)
        cfg2.population.n_agents = args.agents
        cfg2.behaviour.adapt_weights = False
        sim2 = AgentSimulation(cfg2, scenario=args.scenario, blocked=tuple(args.block), env=sim.env,
                               timeline=sim.timeline).run()
        print_tables(sim2.agent_rows(), "same people, weights fixed (ablation)")


if __name__ == "__main__":
    main()
