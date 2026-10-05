"""FloorGrid / GridStack: static rasterised layers plus the dynamic overlay
(door state, leaf state, clutter, ad-hoc obstacles) that masks them into
walkable_now / gas_perm_now / speed_F_now without ever re-rasterising
(spec §5.2, PROMPT_01 §2 design principle 3).
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

    walkable_static: np.ndarray  # bool [row, col] — includes furniture, excludes clutter
    wall: np.ndarray  # bool
    space_id: np.ndarray  # int32, -1 outside any space
    door_id: np.ndarray  # int32, -1 if not a door/exit/fire-door cell
    exit_id: np.ndarray  # int32, -1 if not an exit cell (subset of door_id)
    stair_id: np.ndarray  # int32, -1 if not a stair landing cell
    obstacle_static: np.ndarray  # bool — furniture only (already baked into walkable_static)
    clutter_mask: np.ndarray  # bool — candidate clutter cells, toggled by the overlay
    void: np.ndarray  # bool
    outside: np.ndarray  # bool — the 2 m margin ring
    window_id: np.ndarray  # int32, -1 if not a window cell
    gas_perm_static: np.ndarray  # float32, using each door/window's *default* leaf/open state

    space_ids: list[str]
    door_ids: list[str]
    stair_ids: list[str]
    window_ids: list[str]

    # per-door / per-window gas permeability pairs, parallel to door_ids / window_ids
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
        walkable = self.walkable_static.copy()
        for idx, did in enumerate(self.door_ids):
            if overlay.door_states.get(did) == "UNAVAILABLE":
                walkable &= ~(self.door_id == idx)
        if overlay.clutter_on:
            walkable &= ~self.clutter_mask
        if overlay.extra_obstacles:
            walkable &= ~self._obstacle_mask(overlay.extra_obstacles)
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
    """Door state, leaf state, clutter on/off, and ad-hoc added obstacles —
    an overlay applied by masking, never by re-rasterising (spec §2 principle 3)."""

    door_states: dict[str, str] = field(default_factory=dict)  # door_id -> AVAILABLE|PARTIAL|UNAVAILABLE
    leaf_states: dict[str, str] = field(default_factory=dict)  # door_id -> "open"|"closed"
    window_open: dict[str, bool] = field(default_factory=dict)  # window_id -> bool
    clutter_on: bool = False
    extra_obstacles: list[Rect] = field(default_factory=list)  # AddObstacle events


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
