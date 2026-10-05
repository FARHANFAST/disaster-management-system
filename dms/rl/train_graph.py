"""Train the hierarchical evacuee: Q-learning on the navigation graph decides
which node to head for next, the grid walker walks there cell by cell.

    python -m dms.rl.train_graph                     # ~2 s training
    python -m dms.rl.train_graph --close-stair ST-C
    python -m dms.rl.train_graph --block X-S

Graded from every walkable cell against two references:
  * cell optimum          — best possible with the same physics (dms/rl/evaluate.py)
  * graph+walker optimum  — best possible for any graph policy with this walker
Writes out/rl_graph.png: per-node learned values, the graph, and the ten
hard-spawn agents' routes (decision points marked).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..geometry.loader import load_site
from ..grid.layers import DynamicOverlay
from .evaluate import optimal_times
from .graph_env import GraphEvacEnv
from .graph_q import GraphQConfig, evaluate_graph_policy, graph_walker_optimum, greedy_graph_rollout, train_graph
from .live_page import SPAWN_ROOMS

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
_DEFAULTS = GraphQConfig()


def spawn_cells(env: GraphEvacEnv, building, t_cell_opt: np.ndarray) -> dict[str, int]:
    """The same ten hard spawns as the live page: each room's slowest cell."""
    ce = env.cell_env
    out = {}
    for floor_id, room in SPAWN_ROOMS:
        f = ce.floor_ids.index(floor_id)
        g = ce.grids[f]
        si = g.space_ids.index(room)
        on = np.flatnonzero((ce.cell_floor == f) & (g.space_id[ce.cells[:, 0], ce.cells[:, 1]] == si))
        out[room] = int(on[np.argmax(t_cell_opt[on])])
    return out


def route_label(env: GraphEvacEnv, nodes: list[int]) -> str:
    """Doors, stairs and the exit along a node route, e.g. 'F-10-d1 > ST-C > X-S'."""
    parts: list[str] = []
    for k in nodes:
        nid = env.label(k)
        kind = env.nav.node(nid).kind_detail
        if kind in ("interior", "exit"):
            name = env.nav.node(nid).ref_id
        elif kind == "stair_landing":
            name = env.nav.node(nid).ref_id
        else:
            continue
        if not parts or parts[-1] != name:
            parts.append(name)
    return " > ".join(parts)


def plot(env: GraphEvacEnv, Q: np.ndarray, starts: dict[str, int], log, path: Path) -> Path:
    ce = env.cell_env
    x0, y0 = ce.grid.transform.origin_local
    rows, cols = ce.grid.shape
    extent = (x0, x0 + cols * ce.h, y0, y0 + rows * ce.h)
    node_xy = np.array([np.mean([ce.state_xy(int(s)) for s in cs], axis=0) for cs in env.cells_of_node])
    node_floor = np.array([ce.cell_floor[cs[0]] for cs in env.cells_of_node])
    V = -Q.max(axis=1)
    vmax = np.percentile(V[np.isfinite(V)], 98)
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    rollouts = {room: greedy_graph_rollout(env, Q, s, with_cells=True) for room, s in starts.items()}

    fig = plt.figure(figsize=(17, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1.1])
    for f, fid in enumerate(ce.floor_ids):
        ax = fig.add_subplot(gs[0, f])
        val = np.full(ce.grid.shape, np.nan)
        on = ce.cell_floor == f
        val[ce.cells[on, 0], ce.cells[on, 1]] = V[env.node_of_state[on]]
        ax.imshow(ce.grids[f].wall, cmap="Greys", extent=extent, origin="lower", interpolation="nearest", alpha=0.9)
        im = ax.imshow(val, cmap="viridis", vmin=0, vmax=vmax, extent=extent, origin="lower", interpolation="nearest")
        # the graph itself
        for u in range(env.n_nodes):
            for v in env.neighbours[u][env.valid[u]]:
                if u < v and node_floor[u] == f and node_floor[v] == f:
                    ax.plot(*node_xy[[u, v]].T, "-", color="white", lw=0.6, alpha=0.45, zorder=3)
        sel = node_floor == f
        ax.scatter(node_xy[sel, 0], node_xy[sel, 1], s=7, c="white", alpha=0.6, zorder=3)
        # agents: walker path + the cell where each graph decision was taken
        for i, (room, ro) in enumerate(rollouts.items()):
            seg = [s for s in ro.cells if ce.cell_floor[s] == f]
            if not seg:
                continue
            xy = np.array([ce.state_xy(s) for s in seg])
            ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[i], lw=2, zorder=5)
            if ce.cell_floor[ro.cells[0]] == f:
                ax.plot(*xy[0], "o", color=colors[i], ms=8, mec="white", zorder=6, label=room)
        ax.set_title(f"{fid} — graph RL value per node [s to exit], graph, walker paths")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, fraction=0.035)
        ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax_c = fig.add_subplot(gs[1, :])
    w = max(1, len(log.episode_time) // 100)
    k = np.ones(w) / w
    ep = np.arange(len(log.episode_time))
    ax_c.plot(ep[w - 1:], np.convolve(log.episode_time, k, mode="valid"), color="#3d7ea6")
    ax_c.set_ylabel("episode time, s (moving avg)")
    ax_c.set_xlabel("episode")
    ax2 = ax_c.twinx()
    ax2.plot(ep[w - 1:], np.convolve(np.asarray(log.reached_exit, float), k, mode="valid"), color="#c45c26")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("reached exit (orange)")
    ax_c.set_title("Learning curve (episodes start from random cells)")
    fig.suptitle("B1 — Q-learning on the navigation graph + grid walker", fontsize=14)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=_DEFAULTS.episodes)
    ap.add_argument("--alpha", type=float, default=_DEFAULTS.alpha)
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

    cfg = GraphQConfig(episodes=args.episodes, alpha=args.alpha, seed=args.seed)
    print(f"Training: {cfg}")
    t0 = time.time()
    Q, log = train_graph(env, cfg, progress_every=max(1, cfg.episodes // 5))
    print(f"Trained in {time.time() - t0:.1f}s ({sum(log.episode_hops):,} graph decisions)")

    e = evaluate_graph_policy(env, Q, t_cell)
    print("\nGreedy graph policy + walker, from every walkable start cell:")
    print(f"  reaches an exit:                {e.success_rate:7.2%}  ({e.n_starts} starts)")
    print(f"  within 5% of cell optimum:      {e.within_5pct:7.2%}")
    print(f"  excess over cell optimum:       mean {e.mean_excess:+.2%}, p90 {e.p90_excess:+.1%}, max {e.max_excess:+.0%}")
    starts_all = [int(s) for s in env.start_states if t_cell[s] > 0]
    design = t_graph[starts_all] / t_cell[starts_all] - 1
    print(f"  ...of which the design costs:   mean {design.mean():+.2%} (best graph policy with this walker)")

    starts = spawn_cells(env, building, t_cell)
    print(f"\n{'room':6s} {'route (doors > stairs > exit)':44s} {'graph RL':>9s} {'best graph':>10s} {'cell opt':>9s}")
    for room, s in starts.items():
        ro = greedy_graph_rollout(env, Q, s)
        rl = f"{ro.time_s:8.1f}s" if ro.reached_exit else f"{'STUCK':>9s}"
        print(f"{room:6s} {route_label(env, ro.nodes):44s} {rl} {t_graph[s]:9.1f}s {t_cell[s]:8.1f}s")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "rl_graph_q.npz", Q=Q, neighbours=env.neighbours,
                        node_ids=np.array(env.node_ids))
    fig_path = plot(env, Q, starts, log, args.out_dir / "rl_graph.png")
    print(f"\nSaved {args.out_dir / 'rl_graph_q.npz'} and {fig_path}")


if __name__ == "__main__":
    main()
