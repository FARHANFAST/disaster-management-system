"""Train the gas-aware DQN (dms/rl/gas_dqn.py) inside an H2S scenario and
grade it against the planners of dms/gas/simulate.py on the same gas history.

    python -m dms.rl.train_gas_dqn                                  # default H2S tank leak
    python -m dms.rl.train_gas_dqn --wind-offset 0 --wind-offset -20 --wind-offset 20
    python -m dms.rl.train_gas_dqn --episodes 12000 --device cuda

Graded against, on the same people, start times and gas:
  * gas-aware A*  — re-planned at every node on the live reading + forecast
  * gas-blind A*  — plain fastest route (what an untrained, gas-blind policy does)
Metrics per run: reached an exit, walking time, peak ppm, toxic load
L = Σ C² dt / 60 [ppm²·min] and the shared cost the DQN is trained on.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..gas.scenario import load_scenario
from ..gas.simulate import DEFAULT_SCENARIO, spawn_cell
from ..gas.timeline import GasTimeline
from .agent_profile import sample_population
from .dqn import DQNConfig
from .gas_dqn import GasDQN, GasRewardConfig, GasRollout, astar_chooser, egress_dwell
from .train_dqn import route_label

_DEFAULTS = DQNConfig()


def grade(dqn: GasDQN, sc, tl: GasTimeline, profile: dict, t_start: float, starts: dict[str, int]) -> dict:
    """Roll out DQN, gas-aware A* and gas-blind A* from every start."""
    dwell = egress_dwell(dqn.env, tl)
    policies = {
        "dqn": None,
        "astar_gas": astar_chooser(dqn.env, sc.routing, gas_aware=True, dwell_s=dwell),
        "astar_blind": astar_chooser(dqn.env, sc.routing, gas_aware=False),
    }
    out: dict[str, dict[str, GasRollout]] = {k: {} for k in policies}
    for name, s in starts.items():
        for k, choose in policies.items():
            out[k][name] = dqn.rollout(s, dqn.new_episode(profile, tl, t_start), choose)
    return out


def summarise(runs: dict[str, GasRollout]) -> dict:
    r = list(runs.values())
    ok = [x for x in r if x.reached_exit]
    return dict(n=len(r), reached=len(ok) / len(r), incap=sum(x.incapacitated for x in r),
                mean_time=float(np.mean([x.time_s for x in ok])) if ok else float("nan"),
                mean_peak=float(np.mean([x.peak_ppm for x in r])), max_peak=float(max(x.peak_ppm for x in r)),
                mean_dose=float(np.mean([x.dose for x in r])), mean_cost=float(np.mean([x.cost for x in r])),
                gassed_nodes=sum(x.peak_ppm >= 10.0 for x in r))


def exit_of(dqn: GasDQN, ro: GasRollout) -> str:
    k = ro.nodes[-1]
    return dqn.env.nav.node(dqn.env.node_ids[k]).ref_id if ro.reached_exit else (ro.failure or "-").upper()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", type=Path, default=DEFAULT_SCENARIO)
    ap.add_argument("--episodes", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    ap.add_argument("--wind-offset", type=float, action="append", default=None, metavar="DEG",
                    help="train on the scenario with every wind keyframe rotated by DEG (repeatable; default 0)")
    ap.add_argument("--start-after", type=float, default=15.0,
                    help="grading: seconds after the alarm the evacuees start moving")
    ap.add_argument("--n-random", type=int, default=150, help="grading: extra random start cells")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    args = ap.parse_args()

    sc = load_scenario(args.scenario)
    env = sc.make_env()
    offsets = args.wind_offset or [0.0]
    t0 = time.time()
    timelines = []
    for off in offsets:
        timelines.append(GasTimeline.from_scenario(sc, env, wind_offset_deg=off,
                                                   zones=timelines[0].zones if timelines else None))
    print(f"Scenario {sc.name}: graph {env.n_nodes} nodes ({int(env.terminal.sum())} exits); "
          f"{len(timelines)} gas histories (wind offsets {offsets}) in {time.time() - t0:.1f}s; "
          f"alarm at {[tl.alarm_t for tl in timelines]} s")

    cfg = DQNConfig(episodes=args.episodes, batch_size=args.batch_size, device=args.device, seed=args.seed)
    dqn = GasDQN(env, timelines, cfg, GasRewardConfig(avoid_ppm=sc.routing.avoid_ppm))
    dev = f"{dqn.device} ({torch.cuda.get_device_name(0)})" if str(dqn.device).startswith("cuda") else str(dqn.device)
    print(f"Device: {dev}; state: {env.n_nodes} one-hot node + {dqn.n_attr} attributes + "
          f"{dqn.n_x - dqn.n_attr} gas/time features -> {env.n_nodes + dqn.n_x}")
    t0 = time.time()
    log = dqn.train(progress_every=max(1, cfg.episodes // 10))
    print(f"Trained in {time.time() - t0:.1f}s ({sum(log.episode_hops):,} decisions, "
          f"{len(log.losses):,} gradient steps)")

    profile = sample_population(1, np.random.default_rng(cfg.seed))[0]
    rooms = {f"{sp.floor}:{sp.room}": spawn_cell(env, sp.floor, sp.room) for sp in sc.spawns}
    rng = np.random.default_rng(1234)
    rand = {f"cell{int(s)}": int(s) for s in rng.choice(env.start_states, size=args.n_random, replace=False)}

    for tl in timelines:
        t_start = tl.alarm_t + args.start_after
        print(f"\n=== wind offset {tl.wind_offset_deg:+.0f} deg, evacuees move at t={t_start:.0f}s "
              f"(alarm {tl.alarm_t:.0f}s + {args.start_after:.0f}s); reference agent "
              f"{profile['age']} {profile['gender']} {profile['healthStatus']}, {profile['speed']:.2f} m/s")
        res = grade(dqn, sc, tl, profile, t_start, rooms)
        print(f"{'room':9s} {'DQN route':38s} {'exit':>5s} {'time':>6s} {'peak':>6s} {'dose':>8s} | "
              f"{'A*gas':>5s} {'peak':>6s} {'dose':>8s} | {'blind':>5s} {'peak':>6s} {'dose':>8s}")
        for name in rooms:
            d, a, b = res["dqn"][name], res["astar_gas"][name], res["astar_blind"][name]
            print(f"{name:9s} {route_label(env, d.nodes)[-38:]:38s} {exit_of(dqn, d):>5s} {d.time_s:6.1f} "
                  f"{d.peak_ppm:6.1f} {d.dose:8.1f} | {exit_of(dqn, a):>5s} {a.peak_ppm:6.1f} {a.dose:8.1f} | "
                  f"{exit_of(dqn, b):>5s} {b.peak_ppm:6.1f} {b.dose:8.1f}")
        res_r = grade(dqn, sc, tl, profile, t_start, rand)
        print(f"\n{'policy':12s} {'starts':>6s} {'reached':>8s} {'incap':>5s} {'time':>7s} {'peak':>7s} "
              f"{'max':>7s} {'dose':>9s} {'cost':>8s} {'peak>=10ppm':>11s}")
        for label, r in (("rooms", res), (f"{args.n_random} random", res_r)):
            for k in ("dqn", "astar_gas", "astar_blind"):
                m = summarise(r[k])
                print(f"{k:12s} {m['n']:6d} {m['reached']:8.1%} {m['incap']:5d} {m['mean_time']:7.1f} "
                      f"{m['mean_peak']:7.1f} {m['max_peak']:7.1f} {m['mean_dose']:9.1f} {m['mean_cost']:8.1f} "
                      f"{m['gassed_nodes']:11d}   ({label})")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "gas_dqn_policy.pt"
    torch.save({"state_dict": dqn.online.state_dict(), "dqn_config": cfg, "gas_config": dqn.gas,
                "scenario": str(args.scenario), "wind_offsets": offsets,
                "n_nodes": env.n_nodes, "n_x": dqn.n_x, "max_deg": env.max_deg, "hidden": cfg.hidden}, ckpt)
    print(f"\nSaved {ckpt}")


if __name__ == "__main__":
    main()
