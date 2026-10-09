"""Train the deep Q-network route policy (dms/rl/dqn.py) on the navigation
graph + grid walker, and grade it against the same references as the tabular
trainer (dms/rl/train_graph.py).

    python -m dms.rl.train_dqn                       # CPU or RTX 4050, auto-detected
    python -m dms.rl.train_dqn --device cuda         # force GPU
    python -m dms.rl.train_dqn --close-stair ST-C
    python -m dms.rl.train_dqn --block X-S

Graded from every walkable cell against two references:
  * cell optimum          — best possible with the same physics (dms/rl/evaluate.py)
  * graph+walker optimum  — best possible for any graph policy with this walker
A single shared MLP conditions on the agent's attribute features, so a trained
DQN is directly comparable to (and interchangeable with) the tabular policy.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from ..geometry.loader import load_site
from ..grid.layers import DynamicOverlay
from .agent_profile import profile_features, sample_population
from .dqn import DQN, DQNConfig
from .evaluate import optimal_times
from .graph_env import GraphEvacEnv
from .graph_q import graph_walker_optimum
from .train_graph import spawn_cells

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
_DEFAULTS = DQNConfig()


def route_label(env: GraphEvacEnv, nodes: list[int]) -> str:
    parts: list[str] = []
    for k in nodes:
        nid = env.label(k)
        kind = env.nav.node(nid).kind_detail
        if kind in ("interior", "exit", "stair_landing"):
            name = env.nav.node(nid).ref_id
        else:
            continue
        if not parts or parts[-1] != name:
            parts.append(name)
    return " > ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=_DEFAULTS.episodes)
    ap.add_argument("--batch-size", type=int, default=_DEFAULTS.batch_size)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--seed", type=int, default=_DEFAULTS.seed)
    ap.add_argument("--block", action="append", default=[], metavar="EXIT_ID")
    ap.add_argument("--close-stair", action="append", default=[], metavar="STAIR_ID")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    args = ap.parse_args()

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    overlay = DynamicOverlay(door_states={x: "UNAVAILABLE" for x in args.block})
    env = GraphEvacEnv(site, "B1", overlay=overlay, closed_stairs=args.close_stair)
    print(f"Graph: {env.n_nodes} nodes, {int(env.valid.sum()) // 2} edges, {int(env.terminal.sum())} exits, "
          f"max {env.max_deg} neighbours; grid: {env.cell_env.n_states} cells")
    if args.block or args.close_stair:
        print(f"Scenario: blocked exits {args.block or '-'}, closed stairs {args.close_stair or '-'}")

    t0 = time.time()
    t_cell = optimal_times(env.cell_env)
    t_graph = graph_walker_optimum(env)
    print(f"References computed in {time.time() - t0:.1f}s")

    cfg = DQNConfig(episodes=args.episodes, batch_size=args.batch_size, device=args.device, seed=args.seed)
    dqn = DQN(env, cfg)
    if str(dqn.device).startswith("cuda"):
        print(f"Device: {dqn.device} ({torch.cuda.get_device_name(0)}) — note: at this problem size "
              f"CPU is usually as fast; GPU pays off once crowd/smoke features scale the state.")
    else:
        print(f"Device: {dqn.device}")
    print(f"State dim: {env.n_nodes} one-hot node + {dqn.n_attr} attribute features -> "
          f"{env.n_nodes + dqn.n_attr}; training {cfg}")
    t0 = time.time()
    log = dqn.train(progress_every=max(1, cfg.episodes // 5))
    print(f"Trained in {time.time() - t0:.1f}s ({sum(log.episode_hops):,} graph decisions, "
          f"{len(log.losses):,} gradient steps)")

    attr = profile_features(sample_population(1, np.random.default_rng(cfg.seed))[0])
    e = dqn.evaluate(t_cell, attr)
    print("\nGreedy DQN policy + walker, from every walkable start cell:")
    print(f"  reaches an exit:                {e.success_rate:7.2%}  ({e.n_starts} starts)")
    print(f"  within 5% of cell optimum:      {e.within_5pct:7.2%}")
    print(f"  excess over cell optimum:       mean {e.mean_excess:+.2%}, p90 {e.p90_excess:+.1%}, max {e.max_excess:+.0%}")
    starts_all = [int(s) for s in env.start_states if t_cell[s] > 0]
    design = t_graph[starts_all] / t_cell[starts_all] - 1
    print(f"  ...of which the design costs:   mean {design.mean():+.2%} (best graph policy with this walker)")

    starts = spawn_cells(env, building, t_cell)
    print(f"\n{'room':6s} {'route (doors > stairs > exit)':44s} {'DQN':>9s} {'best graph':>10s} {'cell opt':>9s}")
    for room, s in starts.items():
        ro = dqn.greedy_rollout(s, attr)
        rl = f"{ro.time_s:8.1f}s" if ro.reached_exit else f"{'STUCK':>9s}"
        print(f"{room:6s} {route_label(env, ro.nodes):44s} {rl} {t_graph[s]:9.1f}s {t_cell[s]:8.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = args.out_dir / "dqn_policy.pt"
    torch.save({
        "state_dict": dqn.online.state_dict(),
        "dqn_config": cfg,
        "n_nodes": env.n_nodes, "n_attr": dqn.n_attr, "max_deg": env.max_deg, "hidden": cfg.hidden,
    }, ckpt)
    np.savez_compressed(args.out_dir / "rl_dqn_v.npz",
                        V=-dqn.q_values_all(attr), neighbours=env.neighbours,
                        node_ids=np.array(env.node_ids))
    print(f"\nSaved {ckpt} and {args.out_dir / 'rl_dqn_v.npz'}")


if __name__ == "__main__":
    main()