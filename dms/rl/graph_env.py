"""Hierarchical evacuation environment: RL decides on the L2 navigation
graph, a grid walker does the physical moving on the L1 cell grid.

    decision (RL)   "I'm in node u (room F-10) -> go to neighbour v (door F-10-d1)"
    execution       GridWalker walks cell by cell from where the agent actually
                    stands to the nearest cell of v, never leaving u's and v's
                    own cells — so it exits a room through the door, never
                    through a wall, and steps around furniture.
    reward          -(seconds that walk actually took)

State  = the graph node the agent's cell belongs to (`node_of`).
Action = a slot in that node's neighbour list (padded to max degree).
Done   = the agent arrives in an exit node.

Graph clean-up relative to dms/graph/builder.py's NavGraph:
  * opening nodes (wall gaps with no cells of their own) are contracted into
    direct edges between their two sides — passing an opening is not a decision;
  * nodes with no walkable cell under the current overlay are dropped
    (assembly points, the default-locked X-SE, any door made UNAVAILABLE).

The walker uses the exact move model of GridEvacEnv (same cells, 8-connected,
no corner cutting, same step times, STAIRS for vertical edges), so a graph
policy's realised time is directly comparable to the cell-level optimum.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from ..geometry.site import Site
from ..graph.builder import build_navigation_graph
from ..grid.layers import H_COARSE, DynamicOverlay
from .env import MOVES, STAIRS, GridEvacEnv


@dataclass(slots=True)
class Leg:
    time_s: float
    arrival: int  # cell state where the agent enters the target node


class GridWalker:
    """Shortest cell path from any cell of node u into node v, restricted to
    u's and v's cells. Legs are computed on first use and cached per (u, v)."""

    def __init__(self, cell_env: GridEvacEnv, node_of_state: np.ndarray, cells_of_node: list[np.ndarray]):
        self.env = cell_env
        self.node_of_state = node_of_state
        self.cells_of_node = cells_of_node
        # predecessor lists over walking moves only (STAIRS handled per vertical edge)
        self.preds: list[list[tuple[int, float]]] = [[] for _ in range(cell_env.n_states)]
        for s in range(cell_env.n_states):
            for a in range(len(MOVES)):
                s2 = int(cell_env.next_state[s, a])
                if s2 != s:
                    self.preds[s2].append((s, -float(cell_env.reward[s, a])))
        # per directed edge: state -> (time to reach v, next state on the path, arrival state)
        self._legs: dict[tuple[int, int], dict[int, tuple[float, int, int]]] = {}
        self.fallbacks = 0  # legs that needed the whole floor (u∪v alone was disconnected)

    def _vertical(self, u: int, v: int) -> bool:
        a, b = self.cells_of_node[u][0], self.cells_of_node[v][0]
        return self.env.cell_floor[a] != self.env.cell_floor[b]

    def _solve(self, u: int, v: int, region: set[int] | None) -> dict[int, tuple[float, int, int]]:
        """Backward Dijkstra from v's cells over `region` (None = whole floor)."""
        floor = self.env.cell_floor[self.cells_of_node[v][0]]
        out: dict[int, tuple[float, int, int]] = {}
        heap: list[tuple[float, int, int, int]] = []
        for s in self.cells_of_node[v]:
            s = int(s)
            out[s] = (0.0, s, s)
            heap.append((0.0, s, s, s))
        heapq.heapify(heap)
        while heap:
            d, s, _nxt, arr = heapq.heappop(heap)
            if d > out[s][0]:
                continue
            for p, cost in self.preds[s]:
                if region is not None and p not in region:
                    continue
                if region is None and self.env.cell_floor[p] != floor:
                    continue
                nd = d + cost
                if p not in out or nd < out[p][0] - 1e-12:
                    out[p] = (nd, s, arr)
                    heapq.heappush(heap, (nd, p, s, arr))
        return out

    def _table(self, u: int, v: int) -> dict[int, tuple[float, int, int]]:
        key = (u, v)
        if key not in self._legs:
            region = set(self.cells_of_node[u].tolist()) | set(self.cells_of_node[v].tolist())
            table = self._solve(u, v, region)
            if any(int(s) not in table for s in self.cells_of_node[u]):
                self.fallbacks += 1
                table = {**self._solve(u, v, None), **table}
            self._legs[key] = table
        return self._legs[key]

    def leg(self, u: int, v: int, s: int) -> Leg | None:
        """Walk from cell `s` (in node u) into node v. None if impossible."""
        if self._vertical(u, v):
            s2 = int(self.env.next_state[s, STAIRS])
            if s2 == s:
                return None
            return Leg(-float(self.env.reward[s, STAIRS]), s2)
        entry = self._table(u, v).get(s)
        return None if entry is None else Leg(entry[0], entry[2])

    def path(self, u: int, v: int, s: int) -> list[int]:
        """Cells walked on the leg u -> v starting at `s` (inclusive of both ends)."""
        if self._vertical(u, v):
            return [s, int(self.env.next_state[s, STAIRS])]
        table = self._table(u, v)
        out = [s]
        while table[out[-1]][1] != out[-1]:
            out.append(table[out[-1]][1])
        return out


class GraphEvacEnv:
    def __init__(self, site: Site, building_id: str = "B1", *, overlay: DynamicOverlay | None = None,
                 closed_stairs: Iterable[str] = ()):
        building = site.get_building(building_id)
        closed_stairs = set(closed_stairs)
        self.cell_env = GridEvacEnv.for_building(building, overlay=overlay, closed_stairs=closed_stairs)
        nav = build_navigation_graph(site, building_id, use_outdoor_network=False)
        self.nav = nav
        ce = self.cell_env

        # nav node index for every cell state
        nav_of_state = np.empty(ce.n_states, dtype=np.int64)
        for f, fid in enumerate(ce.floor_ids):
            on = ce.cell_floor == f
            nav_of_state[on] = nav.node_of[(fid, H_COARSE)][ce.cells[on, 0], ce.cells[on, 1]]
        if (nav_of_state < 0).any():
            raise ValueError("some walkable cells belong to no graph node")

        kept = sorted(set(nav_of_state.tolist()))
        self.node_ids = [nav.node_ids[i] for i in kept]
        compact = {nav_i: k for k, nav_i in enumerate(kept)}
        self.node_of_state = np.array([compact[i] for i in nav_of_state], dtype=np.int64)
        self.n_nodes = len(self.node_ids)
        self.cells_of_node = [np.flatnonzero(self.node_of_state == k) for k in range(self.n_nodes)]

        # edges between kept nodes, with opening nodes contracted away
        kept_ids = set(self.node_ids)
        index = {nid: k for k, nid in enumerate(self.node_ids)}
        stair_of_landing = {nid: nav.node(nid).ref_id for nid in self.node_ids
                            if nav.node(nid).kind_detail == "stair_landing"}
        adj: list[set[int]] = [set() for _ in range(self.n_nodes)]

        def connect(a: str, b: str) -> None:
            if a in kept_ids and b in kept_ids and a != b:
                if a in stair_of_landing and b in stair_of_landing and stair_of_landing[a] in closed_stairs:
                    return
                adj[index[a]].add(index[b])
                adj[index[b]].add(index[a])

        for a, b in nav.g.edges:
            connect(a, b)
        for nid in nav.node_ids:
            if nav.node(nid).kind_detail == "opening":
                for a, b in itertools.combinations(nav.g.neighbors(nid), 2):
                    connect(a, b)

        self.max_deg = max(len(n) for n in adj)
        self.neighbours = np.full((self.n_nodes, self.max_deg), -1, dtype=np.int64)
        for k, ns in enumerate(adj):
            self.neighbours[k, :len(ns)] = sorted(ns)
        self.valid = self.neighbours >= 0
        self.terminal = np.array([nav.node(n).kind_detail == "exit" for n in self.node_ids])
        self.walker = GridWalker(ce, self.node_of_state, self.cells_of_node)
        self.start_states = np.flatnonzero(~self.terminal[self.node_of_state])

    # ------------------------------------------------------------- stepping

    def node(self, s: int) -> int:
        return int(self.node_of_state[s])

    def step(self, s: int, slot: int) -> tuple[int, float, bool]:
        """Agent at cell `s` picks neighbour `slot` of its node. Returns
        (arrival cell, reward = -walk seconds, done). An unwalkable leg
        (shouldn't happen on a valid graph) leaves the agent in place at a
        large cost so the learner abandons it."""
        u = self.node(s)
        v = int(self.neighbours[u, slot])
        leg = self.walker.leg(u, v, s)
        if leg is None:
            return s, -60.0, False
        return leg.arrival, -leg.time_s, bool(self.terminal[v])

    def label(self, k: int) -> str:
        return self.node_ids[k]
