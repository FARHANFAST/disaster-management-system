"""Grid evacuation environment for tabular RL — one agent, a static
("perfect") building: no hazards, crowds or door failures yet. Covers one
floor (GridEvacEnv(grid)) or a whole building with its floors linked by
stairs (GridEvacEnv.for_building).

State  = a walkable cell on one of the floors (flattened to 0..n_states-1;
         `cell_floor[s]` says which floor).
Action = one of 8 moves (N, NE, E, SE, S, SW, W, NW), plus STAIRS: valid
         only inside a stairwell, it moves the agent to the same cell on
         the other floor (every stair's GF and FF footprints coincide).
Reward = -(seconds the action takes): a move at walking speed `v0`, a stair
         trip at chi * v0 over the stair's walking length (N4s: the 0.70
         stair factor). An undiscounted return is exactly minus the
         evacuation time, so a learned V(s) = max_a Q(s, a) is directly
         comparable to shortest-path times.
Done   = the agent stands on an exit cell. Only GF has exits on B1, so an
         FF agent must learn to find and take a stair.

Invalid actions — a move into a wall / furniture / outside cell, or STAIRS
outside a stairwell — have `valid[s, a]` False and the learner masks them
out (the agent can see the wall in front of it). If stepped anyway they
leave the agent in place and cost one straight tick. Diagonal moves are only
allowed when both orthogonal neighbours are walkable, so the agent can never
squeeze between two wall cells that touch at a corner.

The whole transition model is precomputed into two (n_states, N_ACTIONS)
arrays, so a step is two array lookups — that's what keeps pure-Python
Q-learning fast enough on ~16k states.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass, replace

import numpy as np

from ..geometry.building import Building, Floor
from ..grid.layers import H_COARSE, DynamicOverlay, FloorGrid
from ..grid.raster import rasterize_floor

# (d_row, d_col); row increases with +y (spec §1.1), so "N" is +1 row.
MOVES: tuple[tuple[int, int], ...] = (
    (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1),
)
STAIRS = len(MOVES)  # action index of "take the stairs to the other floor"
ACTION_NAMES = ("N", "NE", "E", "SE", "S", "SW", "W", "NW", "STAIRS")
N_ACTIONS = len(ACTION_NAMES)
V0 = 1.2  # m/s — same free walking speed the demo walkers use


@dataclass(slots=True)
class StepResult:
    state: int
    reward: float
    done: bool


def _apply_door_defaults(floor: Floor | None, overlay: DynamicOverlay) -> DynamicOverlay:
    """Doors' YAML `default_state` (e.g. X-SE is a service door locked by
    default). FloorGrid.walkable_now only knows the overlay's explicit
    states, while the Eikonal fields fall back to the default — the two must
    agree on which exits exist. Explicit overlay states win."""
    if floor is None:
        return overlay
    defaults = {d.id: d.default_state for d in floor.doors.values()}
    return replace(overlay, door_states={**defaults, **overlay.door_states})


class GridEvacEnv:
    def __init__(self, grid: FloorGrid, *, floor: Floor | None = None, overlay: DynamicOverlay | None = None,
                 v0: float = V0):
        """Single floor. Pass `floor` so doors' YAML default states apply."""
        self._setup([(grid, floor)], building=None, overlay=overlay, v0=v0, closed_stairs=())

    @classmethod
    def for_building(cls, building: Building, *, h: float = H_COARSE, overlay: DynamicOverlay | None = None,
                     v0: float = V0, closed_stairs: Iterable[str] = ()) -> GridEvacEnv:
        """Every floor of `building`, linked by its stairs. `closed_stairs`
        removes a stair's STAIRS action (e.g. scenario `stair_C_lost`)."""
        env = cls.__new__(cls)
        floors = [building.get_floor(fid) for fid in building.floor_ids()]
        env._setup([(rasterize_floor(building, f, h), f) for f in floors],
                   building=building, overlay=overlay, v0=v0, closed_stairs=closed_stairs)
        return env

    def _setup(self, layers: list[tuple[FloorGrid, Floor | None]], *, building: Building | None,
               overlay: DynamicOverlay | None, v0: float, closed_stairs: Iterable[str]) -> None:
        self.grids = [g for g, _ in layers]
        self.grid = self.grids[0]
        self.floor_ids = [g.floor_id for g in self.grids]
        self.v0 = v0
        self.h = self.grid.transform.h
        if any(g.transform != self.grid.transform for g in self.grids):
            raise ValueError("all floors must share one grid transform (STAIRS maps a cell to the same cell)")
        overlay = overlay or DynamicOverlay()

        self.walkable = np.stack([g.walkable_now(_apply_door_defaults(f, overlay)) for g, f in layers])
        fl, rows, cols = np.nonzero(self.walkable)
        self.cell_floor = fl
        self.cells = np.stack([rows, cols], axis=1)  # (n_states, 2) of (row, col)
        self.n_states = len(fl)
        self.state_of = np.full(self.walkable.shape, -1, dtype=np.int64)  # [floor, row, col]
        self.state_of[fl, rows, cols] = np.arange(self.n_states)

        exit_cells = np.stack([g.exit_id != -1 for g in self.grids]) & self.walkable
        self.terminal = exit_cells[fl, rows, cols]
        if not self.terminal.any():
            raise ValueError(f"floors {self.floor_ids} have no available exit cells")

        self.next_state, self.reward = self._build_moves()
        self.stair_cost_s: dict[str, float] = {}
        if building is not None:
            self._add_stairs(building, set(closed_stairs))
        self.valid = self.next_state != np.arange(self.n_states)[:, None]
        self.start_states = np.flatnonzero(~self.terminal)
        self.state: int = -1

    def _build_moves(self) -> tuple[np.ndarray, np.ndarray]:
        _, rows, cols = self.walkable.shape
        nxt = np.tile(np.arange(self.n_states)[:, None], (1, N_ACTIONS))
        rew = np.full((self.n_states, N_ACTIONS), -self.h / self.v0)  # a bump costs one straight tick

        for s, (f, (r, c)) in enumerate(zip(self.cell_floor, self.cells)):
            walk = self.walkable[f]

            def free(rr: int, cc: int) -> bool:
                return 0 <= rr < rows and 0 <= cc < cols and bool(walk[rr, cc])

            for a, (dr, dc) in enumerate(MOVES):
                diagonal = dr != 0 and dc != 0
                ok = free(r + dr, c + dc)
                if diagonal:
                    ok = ok and free(r + dr, c) and free(r, c + dc)
                if ok:
                    nxt[s, a] = self.state_of[f, r + dr, c + dc]
                    rew[s, a] = -self.h * (math.sqrt(2.0) if diagonal else 1.0) / self.v0
        return nxt, rew

    def _add_stairs(self, building: Building, closed: set[str]) -> None:
        """STAIRS links each stairwell cell to the same cell on the other
        floor (two-floor building: GF <-> FF, as in Stair's own model)."""
        if len(self.grids) != 2:
            raise ValueError(f"stairs link exactly two floors; got {self.floor_ids}")
        unknown = closed - set(building.stairs)
        if unknown:
            raise ValueError(f"unknown stair ids {sorted(unknown)}")
        lower, upper = self.grids
        for stair in building.stairs.values():
            if stair.id in closed:
                continue
            cost = stair.walking_length_m / (stair.chi * self.v0)
            self.stair_cost_s[stair.id] = cost
            both = ((lower.stair_id == lower.stair_ids.index(stair.id))
                    & (upper.stair_id == upper.stair_ids.index(stair.id))
                    & self.walkable[0] & self.walkable[1])
            for r, c in np.argwhere(both):
                lo, hi = self.state_of[0, r, c], self.state_of[1, r, c]
                self.next_state[lo, STAIRS], self.reward[lo, STAIRS] = hi, -cost
                self.next_state[hi, STAIRS], self.reward[hi, STAIRS] = lo, -cost

    # ------------------------------------------------------------- gym-style API

    def reset(self, rng: np.random.Generator, start: int | None = None) -> int:
        self.state = int(rng.choice(self.start_states)) if start is None else start
        return self.state

    def step(self, action: int) -> StepResult:
        s2 = int(self.next_state[self.state, action])
        r = float(self.reward[self.state, action])
        self.state = s2
        return StepResult(state=s2, reward=r, done=bool(self.terminal[s2]))

    # ------------------------------------------------------------- helpers

    def _floor_index(self, floor_id: str | None) -> int:
        return 0 if floor_id is None else self.floor_ids.index(floor_id)

    def cell_state(self, row: int, col: int, floor_id: str | None = None) -> int:
        return int(self.state_of[self._floor_index(floor_id), row, col])

    def state_floor(self, s: int) -> str:
        return self.floor_ids[int(self.cell_floor[s])]

    def state_xy(self, s: int) -> tuple[float, float]:
        r, c = self.cells[s]
        return self.grid.transform.cell_to_local(int(r), int(c))

    def nearest_state(self, x: float, y: float, floor_id: str | None = None) -> int:
        """Walkable cell closest to a local-frame point on a floor (default:
        the first) — e.g. a room centre that happens to fall on furniture."""
        f = self._floor_index(floor_id)
        on_floor = np.flatnonzero(self.cell_floor == f)
        centres = (self.cells[on_floor, ::-1] + 0.5) * self.h + np.asarray(self.grid.transform.origin_local)
        return int(on_floor[np.argmin(np.hypot(centres[:, 0] - x, centres[:, 1] - y))])
