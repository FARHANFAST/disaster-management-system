"""The building as the agent module sees it, and the read-only snapshot the
orchestrator hands every agent each tick.

GraphInfo (static, built once): per graph node its centroid, floor and kind,
the exits, and a travel-time distance from every node to every exit over the
navigation graph (centroid distance at walking speed; stair edges use the
stair time). An agent combines the rows of the exits *it knows* — nobody
plans over the whole building.

Snapshot (per tick): time, alarm flag, the gas history (dms/gas/timeline.py),
which exits are physically blocked (unknown to agents until seen), and a
spatial hash of everyone's position for the sensing bubble.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np

from ..gas.timeline import GasTimeline
from ..rl.graph_env import GraphEvacEnv


class GraphInfo:
    def __init__(self, env: GraphEvacEnv):
        self.env = env
        ce = env.cell_env
        n = env.n_nodes
        xy = np.array([ce.state_xy(int(s)) for s in range(ce.n_states)])
        self.cell_xy = xy
        self.node_cells_xy = [xy[c] for c in env.cells_of_node]
        self.centroid = np.array([c.mean(axis=0) for c in self.node_cells_xy])
        self.floor = np.array([int(ce.cell_floor[c[0]]) for c in env.cells_of_node])
        self.kind = [env.nav.node(nid).kind_detail for nid in env.node_ids]
        self.exit_nodes = [int(k) for k in np.flatnonzero(env.terminal)]
        self.exit_name = {k: env.nav.node(env.node_ids[k]).ref_id for k in self.exit_nodes}
        self.exit_of_name = {v: k for k, v in self.exit_name.items()}
        self.rooms = [k for k in range(n) if self.kind[k] == "room"]
        self.nbrs = [env.neighbours[u][env.valid[u]].astype(int).tolist() for u in range(n)]
        # edge travel times (s at the walker's speed)
        stair_s = min(ce.stair_cost_s.values(), default=10.0)
        self.edge_s: dict[tuple[int, int], float] = {}
        for u in range(n):
            for v in self.nbrs[u]:
                if self.floor[u] != self.floor[v]:
                    t = stair_s
                else:
                    t = float(np.hypot(*(self.centroid[u] - self.centroid[v]))) / ce.v0
                self.edge_s[(u, v)] = max(t, 0.1)
        self.dist = {k: self._dijkstra(k) for k in self.exit_nodes}  # exit -> (n,) seconds

    def _dijkstra(self, goal: int) -> np.ndarray:
        d = np.full(self.env.n_nodes, np.inf)
        d[goal] = 0.0
        heap = [(0.0, goal)]
        while heap:
            du, u = heapq.heappop(heap)
            if du > d[u]:
                continue
            for v in self.nbrs[u]:
                nd = du + self.edge_s[(v, u)]
                if nd < d[v]:
                    d[v] = nd
                    heapq.heappush(heap, (nd, v))
        return d

    def dist_to(self, exits: list[int]) -> np.ndarray:
        """Time from every node to the nearest of `exits` (inf if none)."""
        if not exits:
            return np.full(self.env.n_nodes, np.inf)
        return np.min([self.dist[k] for k in exits], axis=0)

    def node_distance(self, xy: np.ndarray, floor: int, k: int) -> float:
        """Distance (m) from a point to the nearest cell of node k; inf on another floor."""
        if self.floor[k] != floor:
            return float("inf")
        c = self.node_cells_xy[k]
        return float(np.sqrt(((c - xy) ** 2).sum(axis=1).min()))


class SpatialHash:
    """Uniform grid over (floor, x, y), cell = the largest sensing radius."""

    def __init__(self, cell: float):
        self.cell = cell
        self.buckets: dict[tuple[int, int, int], list[int]] = {}
        self.xy = np.empty((0, 2))
        self.floor = np.empty(0, dtype=int)

    def rebuild(self, xy: np.ndarray, floor: np.ndarray, active: np.ndarray) -> None:
        self.xy, self.floor = xy, floor
        self.buckets = {}
        keys = np.floor(xy / self.cell).astype(int)
        for i in np.flatnonzero(active):
            self.buckets.setdefault((int(floor[i]), int(keys[i, 0]), int(keys[i, 1])), []).append(int(i))

    def near(self, i: int, radius: float) -> list[int]:
        f = int(self.floor[i])
        cx, cy = np.floor(self.xy[i] / self.cell).astype(int)
        span = int(np.ceil(radius / self.cell))
        out = []
        for dx in range(-span, span + 1):
            for dy in range(-span, span + 1):
                for j in self.buckets.get((f, cx + dx, cy + dy), ()):
                    if j != i and np.hypot(*(self.xy[j] - self.xy[i])) <= radius:
                        out.append(j)
        return out


@dataclass(slots=True)
class Snapshot:
    t: float
    alarm: bool
    timeline: GasTimeline
    blocked_exits: frozenset[int]  # physically unusable; agents learn it only on sight
    hash: SpatialHash
    agents: list = field(default_factory=list)  # read-only view of every Agent

    def ppm(self, k: int) -> float:
        return float(self.timeline.now(self.t)[k])

    def soon(self, k: int) -> float:
        return float(self.timeline.soon(self.t)[k])
