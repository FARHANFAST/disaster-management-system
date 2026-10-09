"""Export the hierarchical evacuation environment (graph RL + grid walker)
into a self-contained HTML page that runs graph Q-learning live in the
browser (dms/rl/web/live_template.html).

    python -m dms.rl.live_page            # -> out/rl_live.html

The page gets the exact graph (nodes, neighbours, exits), the exact
cell->node map, and every walker leg precomputed by GridWalker (time, next
cell, arrival cell for each start cell of each directed edge), so the agents
you watch walk the same paths the Python trainer and tests use — only the
Q-learning loop is re-implemented in JS. Agents are sampled with a full
attribute set (demographics, cognitive profile, physiological state) but only
the movement attributes — walking speed, mobility, reaction time — change the
simulation; the rest are carried along for the future Agent/Decision module.
Each agent's real walk time is its leg time scaled by v0 / (its own speed).
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

from ..geometry.loader import load_site
from .agent_profile import MAX_SPEED, sample_population
from .env import STAIRS
from .evaluate import optimal_times
from .graph_env import GraphEvacEnv
from .graph_q import graph_walker_optimum

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
TEMPLATE = Path(__file__).resolve().parent / "web" / "live_template.html"

# Hard spawns used by the Python trainer (dms/rl/train_graph.py): each room's
# slowest cell. The live page ignores this list and samples its own population.
SPAWN_ROOMS = [
    ("FF", "F-09"), ("FF", "F-08"), ("FF", "F-10"), ("FF", "F-02"), ("FF", "F-01"),
    ("FF", "F-12"), ("FF", "F-07"), ("GF", "G-03"), ("GF", "G-15"), ("GF", "G-06"),
]

# Cell categories for drawing the plan (one byte per cell).
OUTSIDE, WALL, FLOOR, FURNITURE, DOOR, EXIT, STAIR, VOID, WINDOW, LOCKED_EXIT = range(10)
# Node kinds for the page (drawing + route labels).
NODE_KINDS = {"room": 0, "corridor": 1, "stub": 1, "interior": 2, "fire_door": 2, "stair_landing": 3, "exit": 4}


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


def _categories(genv: GraphEvacEnv, f: int) -> np.ndarray:
    ce = genv.cell_env
    g = ce.grids[f]
    walk = ce.walkable[f]
    cat = np.full(g.shape, OUTSIDE, dtype=np.uint8)
    cat[g.space_id != -1] = FLOOR
    cat[g.void] = VOID
    cat[g.obstacle_static] = FURNITURE
    cat[g.wall] = WALL
    cat[g.window_id != -1] = WINDOW
    cat[(g.door_id != -1) & walk] = DOOR
    cat[(g.stair_id != -1) & walk] = STAIR
    cat[(g.exit_id != -1) & walk] = EXIT
    cat[(g.exit_id != -1) & ~walk] = LOCKED_EXIT
    return cat


def sample_agents(genv: GraphEvacEnv, building, t_cell: np.ndarray, t_graph: np.ndarray,
                  n: int, seed: int) -> list[dict]:
    """`n` agents at distinct random walkable cells across every room of both
    floors (so bigger rooms get proportionally more agents), each carrying its
    sampled demographics + attributes plus per-cell optimum references."""
    rng = np.random.default_rng(seed)
    ce = genv.cell_env
    candidates: list[tuple[str, str, int]] = []
    for f, fid in enumerate(ce.floor_ids):
        g = ce.grids[f]
        walk = ce.walkable[f]
        for sp in building.get_floor(fid).spaces.values():
            if sp.kind != "room" or sp.id not in g.space_ids:
                continue
            si = g.space_ids.index(sp.id)
            on = ((ce.cell_floor == f) & walk[ce.cells[:, 0], ce.cells[:, 1]]
                  & (g.space_id[ce.cells[:, 0], ce.cells[:, 1]] == si))
            candidates.extend((fid, sp.id, int(s)) for s in np.flatnonzero(on))
    if len(candidates) < n:
        raise ValueError(f"need {n} distinct spawn cells but only {len(candidates)} walkable room cells")
    pop = sample_population(n, np.random.default_rng(seed + 1))
    agents = []
    for p, i in zip(pop, rng.choice(len(candidates), size=n, replace=False)):
        fid, room, s = candidates[int(i)]
        agents.append({"room": room, "name": building.get_floor(fid).spaces[room].name,
                       "floor": fid, "start": s, "cellOpt": float(t_cell[s]),
                       "graphOpt": float(t_graph[s]), **p})
    return agents


def _legs(genv: GraphEvacEnv) -> dict:
    """Flat walker-leg tables. Edge (u, slot) owns len(cells_of_node[u])
    consecutive entries, in cells_of_node[u] order; vertical (stair) edges own
    none and use the per-cell stair arrays instead."""
    ce = genv.cell_env
    n, deg = genv.neighbours.shape
    offset = np.full(n * deg, -1, dtype=np.int32)
    vertical = np.zeros(n * deg, dtype=np.uint8)
    times, nexts, arrivals = [], [], []
    pos = 0
    for u in range(n):
        cells_u = genv.cells_of_node[u]
        for k in np.flatnonzero(genv.valid[u]):
            v = int(genv.neighbours[u, k])
            e = u * deg + int(k)
            if genv.walker._vertical(u, v):
                vertical[e] = 1
                continue
            table = genv.walker._table(u, v)
            offset[e] = pos
            for s in cells_u:
                t, nxt, arr = table[int(s)]
                times.append(t)
                nexts.append(nxt)
                arrivals.append(arr)
            pos += len(cells_u)
    return {
        "legOffset": _b64(offset), "legVertical": _b64(vertical),
        "legTime": _b64(np.asarray(times, dtype=np.float32)),
        "legNext": _b64(np.asarray(nexts, dtype=np.int32)),
        "legArrival": _b64(np.asarray(arrivals, dtype=np.int32)),
        "stairNext": _b64(ce.next_state[:, STAIRS].astype(np.int32)),
        "stairCost": _b64((-ce.reward[:, STAIRS]).astype(np.float32)),
    }


def build_payload(genv: GraphEvacEnv, building, n_agents: int = 100, seed: int = 7) -> dict:
    ce = genv.cell_env
    t_cell = optimal_times(ce)
    t_graph = graph_walker_optimum(genv)

    floors = []
    for f, fid in enumerate(ce.floor_ids):
        floor = building.get_floor(fid)
        labels = []
        for sp in floor.spaces.values():
            if sp.kind in ("room", "stair"):
                r, c = ce.grid.transform.local_to_cell(*sp.rect.center)
                labels.append({"id": sp.id, "r": r, "c": c})
        floors.append({"id": fid, "cat": _b64(_categories(genv, f)), "labels": labels})

    nodes = []
    for k, nid in enumerate(genv.node_ids):
        nd = genv.nav.node(nid)
        name = nd.ref_id if nd.kind_detail in ("interior", "fire_door", "exit", "stair_landing") else nd.space_id
        nodes.append({"id": nid, "kind": NODE_KINDS[nd.kind_detail], "name": name or nid})

    agents = sample_agents(genv, building, t_cell, t_graph, n_agents, seed)

    pos_in_node = np.empty(ce.n_states, dtype=np.int32)
    for cells in genv.cells_of_node:
        pos_in_node[cells] = np.arange(len(cells), dtype=np.int32)

    rows, cols = ce.grid.shape
    return {
        "rows": rows, "cols": cols, "h": ce.h, "v0": ce.v0,
        "nCells": ce.n_states, "nNodes": genv.n_nodes, "maxDeg": genv.max_deg,
        "cellFloor": _b64(ce.cell_floor.astype(np.uint8)),
        "cellR": _b64(ce.cells[:, 0].astype(np.int16)),
        "cellC": _b64(ce.cells[:, 1].astype(np.int16)),
        "nodeOf": _b64(genv.node_of_state.astype(np.int32)),
        "posInNode": _b64(pos_in_node),
        "neighbours": _b64(genv.neighbours.astype(np.int32)),
        "terminal": _b64(genv.terminal.astype(np.uint8)),
        "nodes": nodes,
        **_legs(genv),
        "floors": floors,
        "agents": agents,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("out/rl_live.html"))
    ap.add_argument("--agents", type=int, default=100, help="number of evacuees on the page")
    ap.add_argument("--seed", type=int, default=7, help="RNG seed for spawn cells and demographics")
    args = ap.parse_args()

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    genv = GraphEvacEnv(site, "B1")
    payload = build_payload(genv, building, args.agents, args.seed)
    html = TEMPLATE.read_text(encoding="utf-8").replace('"__DATA__"', json.dumps(payload, separators=(",", ":")))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {genv.n_nodes} nodes, "
          f"{payload['nCells']} cells, {len(payload['agents'])} agents)")
    for a in payload["agents"]:
        print(f"  {a['floor']} {a['room']:5s} {a['age']:7s} {a['gender']} {a['role']:12s} "
              f"mob={a['mobility']:8s} react={a['reaction_s']:5.1f}s speed={a['speed']:4.2f} m/s  "
              f"{a['persona']:10s} fam={a['familiarity']:.2f} health={a['healthStatus']:9s}  "
              f"best graph {a['graphOpt']:5.1f}s  cell optimum {a['cellOpt']:5.1f}s")


if __name__ == "__main__":
    main()
