"""Train a tabular Q-learning evacuee on all of B1 (GF + FF linked by stairs)
and grade it against the exact grid optimum.

    python -m dms.rl.train                         # 200k episodes, ~30 s
    python -m dms.rl.train --episodes 50000 --seed 1
    python -m dms.rl.train --close-stair ST-C      # scenario stair_C_lost
    python -m dms.rl.train --block X-S             # main exit unavailable

Writes out/rl_q.npz (Q-table + state->(floor, cell) map) and out/rl.png
(learned value maps + greedy paths vs optimal, per floor, and the learning
curve). Static building only — no hazards/crowds yet.
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
from .env import STAIRS, GridEvacEnv
from .evaluate import evaluate_all_states, optimal_times
from .q_learning import QConfig, greedy_rollout, train

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
_DEFAULTS = QConfig(episodes=200_000)
DEMO_ROOMS = [("GF", "G-01"), ("GF", "G-11"), ("GF", "LOBBY"), ("FF", "F-01"), ("FF", "F-07"), ("FF", "F-10")]


def _optimal_path(env: GridEvacEnv, t_opt: np.ndarray, start: int) -> list[int]:
    """Exact-optimum descent — the path a perfect agent takes."""
    path = [start]
    s = start
    while not env.terminal[s]:
        s = int(min(env.next_state[s], key=lambda s2: (t_opt[s2] if s2 != s else np.inf)))
        path.append(s)
    return path


def describe_route(env: GridEvacEnv, states: list[int]) -> str:
    """e.g. 'ST-C > X-S' — stairs taken, then the exit reached."""
    parts = []
    for a, b in zip(states, states[1:]):
        if env.cell_floor[a] != env.cell_floor[b]:
            g = env.grids[env.cell_floor[a]]
            r, c = env.cells[a]
            parts.append(g.stair_ids[g.stair_id[r, c]])
    last = states[-1]
    if env.terminal[last]:
        g = env.grids[env.cell_floor[last]]
        r, c = env.cells[last]
        parts.append(g.door_ids[g.exit_id[r, c]])
    else:
        parts.append("STUCK")
    return " > ".join(parts)


def _value_map(env: GridEvacEnv, per_state: np.ndarray, f: int) -> np.ndarray:
    out = np.full(env.grid.shape, np.nan)
    on = env.cell_floor == f
    vals = np.where(np.isfinite(per_state[on]), per_state[on], np.nan)
    out[env.cells[on, 0], env.cells[on, 1]] = vals
    return out


def plot(env: GridEvacEnv, Q: np.ndarray, t_opt: np.ndarray, starts: dict[str, int], log, path: Path) -> Path:
    x0, y0 = env.grid.transform.origin_local
    rows, cols = env.grid.shape
    extent = (x0, x0 + cols * env.h, y0, y0 + rows * env.h)
    n_f = len(env.floor_ids)

    fig = plt.figure(figsize=(8 * n_f, 13))
    gs = fig.add_gridspec(3, n_f, height_ratios=[3, 3, 1.3])
    learned_V = -Q.max(axis=1)
    finite = t_opt[np.isfinite(t_opt)]
    vmax = np.percentile(finite, 99) if finite.size else 1.0
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    paths = {
        "learned": {room: greedy_rollout(env, Q, s).states for room, s in starts.items()},
        "optimal": {room: _optimal_path(env, t_opt, s) for room, s in starts.items()},
    }

    for row, (kind, values) in enumerate((("learned", learned_V), ("optimal", t_opt))):
        for f, fid in enumerate(env.floor_ids):
            ax = fig.add_subplot(gs[row, f])
            grid = env.grids[f]
            ax.imshow(grid.wall, cmap="Greys", extent=extent, origin="lower", interpolation="nearest", alpha=0.9)
            im = ax.imshow(_value_map(env, values, f), cmap="viridis", vmin=0, vmax=vmax, extent=extent,
                           origin="lower", interpolation="nearest")
            on = env.cell_floor == f
            for mask, marker, color, label in ((env.terminal & on, "s", "#c45c26", "exit cells"),
                                               (env.valid[:, STAIRS] & on, ".", "#ffffff", "stair (STAIRS valid)")):
                if mask.any():
                    xy = np.array([env.state_xy(s) for s in np.flatnonzero(mask)])
                    ax.scatter(xy[:, 0], xy[:, 1], marker=marker, s=18 if marker == "s" else 4, c=color,
                               label=label, zorder=4, alpha=1.0 if marker == "s" else 0.35)
            for i, (room, states) in enumerate(paths[kind].items()):
                seg = [s for s in states if env.cell_floor[s] == f]
                if not seg:
                    continue
                xy = np.array([env.state_xy(s) for s in seg])
                ax.plot(xy[:, 0], xy[:, 1], "-", color=colors[i], lw=2, zorder=5)
                if env.cell_floor[states[0]] == f:
                    ax.plot(*xy[0], "o", color=colors[i], ms=7, mec="white", zorder=6, label=room)
                else:  # arrived by stairs
                    ax.plot(*xy[0], "*", color=colors[i], ms=12, mec="white", zorder=6)
            title = "Learned  -max_a Q" if kind == "learned" else "Optimal (Dijkstra)"
            ax.set_title(f"{fid} — {title} [s to exit]")
            ax.set_aspect("equal")
            fig.colorbar(im, ax=ax, fraction=0.035)
            if row == 0:
                ax.legend(loc="upper left", fontsize=7, ncol=2)

    ax_c = fig.add_subplot(gs[2, :])
    window = max(1, len(log.episode_steps) // 100)
    kernel = np.ones(window) / window
    ep = np.arange(len(log.episode_steps))
    ax_c.plot(ep[window - 1:], np.convolve(log.episode_steps, kernel, mode="valid"), color="#3d7ea6")
    ax_c.set_ylabel("steps / episode (moving avg)")
    ax_c.set_xlabel("episode")
    ax2 = ax_c.twinx()
    ax2.plot(ep[window - 1:], np.convolve(np.asarray(log.reached_exit, float), kernel, mode="valid"), color="#c45c26")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("reached exit (orange)")
    ax_c.set_title("Learning curve")
    fig.suptitle("B1 — tabular Q-learning evacuee, floors linked by stairs (★ = arrived by stairs)", fontsize=14)
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
    ap.add_argument("--block", action="append", default=[], metavar="EXIT_ID", help="make an exit UNAVAILABLE")
    ap.add_argument("--close-stair", action="append", default=[], metavar="STAIR_ID", help="remove a stair")
    ap.add_argument("--out-dir", type=Path, default=Path("out"))
    args = ap.parse_args()

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    overlay = DynamicOverlay(door_states={x: "UNAVAILABLE" for x in args.block})
    env = GridEvacEnv.for_building(building, overlay=overlay, closed_stairs=args.close_stair)
    t_opt = optimal_times(env)
    per_floor = ", ".join(f"{fid} {int((env.cell_floor == i).sum())}" for i, fid in enumerate(env.floor_ids))
    print(f"States: {env.n_states} ({per_floor}), {int(env.terminal.sum())} exit cells, "
          f"stairs {', '.join(f'{k} {v:.1f}s' for k, v in env.stair_cost_s.items())}; "
          f"{int(np.isinf(t_opt).sum())} states with no route out")
    if args.block or args.close_stair:
        print(f"Scenario: blocked exits {args.block or '-'}, closed stairs {args.close_stair or '-'}")

    cfg = QConfig(episodes=args.episodes, alpha=args.alpha, seed=args.seed)
    print(f"Training: {cfg}")
    t0 = time.time()
    Q, log = train(env, cfg, progress_every=max(1, cfg.episodes // 10))
    print(f"Trained in {time.time() - t0:.1f}s ({sum(log.episode_steps):,} steps)")

    summary = evaluate_all_states(env, Q, t_opt)
    print("\nGreedy policy from every walkable start cell:")
    print(f"  reaches an exit:          {summary.success_rate:7.2%}  ({summary.n_starts} starts)")
    print(f"  within 1% of optimum:     {summary.optimal_rate:7.2%}")
    print(f"  mean / max excess time:   {summary.mean_excess:+7.2%} / {summary.max_excess:+.2%}")
    print(f"  value error |V + t_opt|:  {summary.value_mae_s:7.2f} s (mean)")

    starts: dict[str, int] = {}
    print(f"\n{'room':7s} {'learned route':16s} {'learned':>8s} {'optimal':>8s}   (seconds)")
    for fid, room in DEMO_ROOMS:
        s = env.nearest_state(*building.get_floor(fid).spaces[room].rect.center, fid)
        starts[room] = s
        ro = greedy_rollout(env, Q, s)
        learned = f"{ro.time_s:8.1f}" if ro.reached_exit else f"{'-':>8s}"
        print(f"{room:7s} {describe_route(env, ro.states):16s} {learned} {t_opt[s]:8.1f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    q_path = args.out_dir / "rl_q.npz"
    np.savez_compressed(q_path, Q=Q, cells=env.cells, cell_floor=env.cell_floor,
                        floor_ids=np.array(env.floor_ids), terminal=env.terminal, h=env.h)
    fig_path = plot(env, Q, t_opt, starts, log, args.out_dir / "rl.png")
    print(f"\nSaved {q_path} and {fig_path}")


if __name__ == "__main__":
    main()
