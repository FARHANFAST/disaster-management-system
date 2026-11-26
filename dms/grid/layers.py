"""FloorGrid / GridStack: static rasterised layers plus the dynamic overlay
(door state, leaf state, clutter, ad-hoc obstacles) that masks them into
walkable_now / gas_perm_now / speed_F_now without ever re-rasterising
(spec §5.2, §2 design principle 3).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.primitives import Rect
from .transforms import GridTransform

H_COARSE = 0.5  # h_c: occupancy / CA grid (spec §1.1)
H_FINE = 0.2  # h_s: smoke / Eikonal grid (spec §1.1)

U_MAX_DEFAULT = 1.65  # m/s, spec §3 parameter registry (U_max, [M])
F_B_DEFAULT = 0.001  # m/s, spec (N2) blocked-front speed, [M] Eq. 6


@dataclass(slots=True)
class FloorGrid:
    building_id: str
    floor_id: str
    transform: GridTransform

    walkable_static: np.ndarray  # bool [row, col] — includes furniture, excludes toggleable obstacle groups
    wall: np.ndarray  # bool
    space_id: np.ndarray  # int32, -1 outside any space
    door_id: np.ndarray  # int32, -1 if not a door/exit/fire-door cell
    exit_id: np.ndarray  # int32, -1 if not an exit cell (subset of door_id)
    stair_id: np.ndarray  # int32, -1 if not a stair landing cell
    refuge_id: np.ndarray  # int32, -1 if not a refuge cell — an AREA mask, not a wall line
    obstacle_static: np.ndarray  # bool — furniture only (already baked into walkable_static)
    obstacle_group_id: np.ndarray  # int32, -1 if not a toggleable (non-furniture) obstacle cell
    void: np.ndarray  # bool
    outside: np.ndarray  # bool — the 2 m margin ring
    window_id: np.ndarray  # int32, -1 if not a window cell
    gas_perm_static: np.ndarray  # float32, using each door/window's *default* leaf/open state

    space_ids: list[str]
    door_ids: list[str]
    stair_ids: list[str]
    refuge_ids: list[str]
    window_ids: list[str]
    obstacle_group_ids: list[str]  # spec §4: independently toggleable groups (clutter, maintenance, ...)
    obstacle_group_default_on: np.ndarray  # bool, parallel to obstacle_group_ids

    # per-door / per-window gas permeability pairs, parallel to door_ids / window_ids
    door_default_state: list[str]  # AVAILABLE | PARTIAL | UNAVAILABLE | BLOCKED_UNKNOWN, parallel to door_ids
    door_perm_open: np.ndarray
    door_perm_closed: np.ndarray
    door_default_leaf_open: np.ndarray  # bool
    window_perm_open: np.ndarray
    window_perm_closed: np.ndarray
    window_default_open: np.ndarray  # bool

    @property
    def shape(self) -> tuple[int, int]:
        return self.transform.shape

    def door_index(self, door_id: str) -> int:
        return self.door_ids.index(door_id)

    # ----------------------------------------------------------------- dynamic

    def _obstacle_mask(self, rects: list[Rect]) -> np.ndarray:
        mask = np.zeros(self.shape, dtype=bool)
        if not rects:
            return mask
        x0, y0 = self.transform.origin_local
        h = self.transform.h
        rows, cols = self.shape
        xc = (np.arange(cols) + 0.5) * h + x0
        yc = (np.arange(rows) + 0.5) * h + y0
        X, Y = np.meshgrid(xc, yc)
        for r in rects:
            mask |= (X >= r.x0) & (X < r.x1) & (Y >= r.y0) & (Y < r.y1)
        return mask

    def walkable_now(self, overlay: DynamicOverlay) -> np.ndarray:
        """Spec §5: BLOCKED_UNKNOWN is physically blocked here
        unconditionally -- a locked door is locked whether or not anyone
        knows it yet. This is the only place that's true: fields/cost code
        keep comparing against the literal "UNAVAILABLE" they always did,
        and the knowledge split is produced by which overlay a caller
        hands them (an uninformed walker's own overlay simply never
        mentions a BLOCKED_UNKNOWN door, so it reads as available there).
        Falls back to the door's own default_state when the overlay
        doesn't override it -- previously only the overlay was ever
        checked, so a door whose *default* state was UNAVAILABLE (e.g.
        X-SE) was wrongly walkable under an empty/baseline overlay."""
        walkable = self.walkable_static.copy()
        for idx, did in enumerate(self.door_ids):
            state = overlay.door_states.get(did, self.door_default_state[idx])
            if state in ("UNAVAILABLE", "BLOCKED_UNKNOWN"):
                walkable &= ~(self.door_id == idx)
        for idx, gname in enumerate(self.obstacle_group_ids):
            on = overlay.active_groups.get(gname, bool(self.obstacle_group_default_on[idx]))
            if on:
                walkable &= ~(self.obstacle_group_id == idx)
        if overlay.extra_obstacles:
            walkable &= ~self._obstacle_mask(overlay.extra_obstacles)
        for name in overlay.blocked_stairs:
            if name in self.stair_ids:
                walkable &= self.stair_id != self.stair_ids.index(name)
        return walkable

    def gas_perm_now(self, overlay: DynamicOverlay) -> np.ndarray:
        perm = self.gas_perm_static.copy()
        for idx, did in enumerate(self.door_ids):
            leaf = overlay.leaf_states.get(did)
            is_open = self.door_default_leaf_open[idx] if leaf is None else (leaf == "open")
            value = self.door_perm_open[idx] if is_open else self.door_perm_closed[idx]
            perm = np.where(self.door_id == idx, value, perm)
        for idx, wid in enumerate(self.window_ids):
            is_open = overlay.window_open.get(wid, bool(self.window_default_open[idx]))
            value = self.window_perm_open[idx] if is_open else self.window_perm_closed[idx]
            perm = np.where(self.window_id == idx, value, perm)
        return perm.astype(np.float32)

    def speed_F_now(
        self, overlay: DynamicOverlay, *, u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT
    ) -> np.ndarray:
        """Eikonal speed field (N2): U_max on currently-walkable cells, F_b elsewhere.
        Dynamic gas/smoke blocking (C >= C_hi) is not modelled yet (later milestone)."""
        walkable = self.walkable_now(overlay)
        return np.where(walkable, u_max, f_b).astype(np.float32)


@dataclass(slots=True)
class DynamicOverlay:
    """Door state, leaf state, per-group obstacle toggles, and ad-hoc added
    obstacles — an overlay applied by masking, never by re-rasterising
    (spec §2 principle 3)."""

    door_states: dict[str, str] = field(default_factory=dict)  # door_id -> AVAILABLE|PARTIAL|UNAVAILABLE
    leaf_states: dict[str, str] = field(default_factory=dict)  # door_id -> "open"|"closed"
    window_open: dict[str, bool] = field(default_factory=dict)  # window_id -> bool
    active_groups: dict[str, bool] = field(default_factory=dict)  # obstacle group -> on/off (spec §4)
    extra_obstacles: list[Rect] = field(default_factory=list)  # AddObstacle events
    blocked_stairs: frozenset[str] = frozenset()  # spec §2: SetStairState UNAVAILABLE -- footprint unwalkable on every floor it serves


@dataclass(slots=True)
class GridStack:
    building_id: str
    floor_id: str
    grids: dict[float, FloorGrid] = field(default_factory=dict)

    def at(self, h: float) -> FloorGrid:
        return self.grids[h]

    @property
    def coarse(self) -> FloorGrid:
        return self.grids[H_COARSE]

    @property
    def fine(self) -> FloorGrid:
        return self.grids[H_FINE]
