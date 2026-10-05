"""Static Eikonal travel-time fields T^eik_{b,k} (spec N1-N2, §1.6),
PROMPT_01 §6.3: per-floor FMM fields with multi-floor stair coupling.

Masking (spec §6.3): walls/void/outside are excluded from the solve domain
entirely (numpy.ma — never traversable, no matter how slow). Obstacles,
clutter (when toggled) and UNAVAILABLE doors get the blocked-front speed
F_b — slow, not impossible, exactly `FloorGrid.speed_F_now`. Free cells get
U_max. This module never rasterises — it only ever consumes an already-built
FloorGrid, so a door/leaf/clutter change can never trigger a re-rasterise
(spec §2 design principle 3; verified in tests by monkeypatching
rasterize_floor to raise if called).

Fields are cached by (building, floor, exit_set, overlay key) — see
FieldCache — so a dirty event only recomputes the affected building/floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import skfmm

from ..geometry.building import Building, Floor
from ..geometry.site import Site
from ..grid.layers import F_B_DEFAULT, U_MAX_DEFAULT, DynamicOverlay, FloorGrid

V_REF = 1.19  # m/s, spec §3 parameter registry — same reference speed cost.py uses


@dataclass(slots=True)
class EikonalField:
    floor_id: str
    exit_set: str
    T: np.ma.MaskedArray  # seconds; masked outside the solve domain
    domain: np.ndarray  # bool, True where solved
    speed: np.ndarray  # the F(x) actually used (for |grad T| ~= 1/F checks)


def domain_mask(grid: FloorGrid) -> np.ndarray:
    """Cells participating in the FMM solve at all — wall/void/outside are
    hard-excluded (never traversable), independent of F_b."""
    return (grid.space_id != -1) & ~grid.void & ~grid.wall


def _exit_target_mask(floor: Floor, grid: FloorGrid, overlay: DynamicOverlay, exit_ids: list[str] | None) -> np.ndarray:
    targets = np.zeros(grid.shape, dtype=bool)
    for idx, (did, door) in enumerate(floor.doors.items()):
        if door.kind != "exit":
            continue
        if exit_ids is not None and did not in exit_ids:
            continue
        state = overlay.door_states.get(did, door.default_state)
        if state == "UNAVAILABLE":
            continue
        targets |= grid.exit_id == idx
    return targets


def resolve_exit_set(building: Building, exit_set: str) -> list[str] | None:
    """None means "every currently-available exit". Falls back to treating
    `exit_set` as a literal exit id if it's not a named set, so a single
    exit can always be selected without needing a YAML entry for it."""
    if exit_set == "all":
        return None
    if exit_set in building.exit_sets:
        return building.exit_sets[exit_set]
    return [exit_set]


def _overlay_key(overlay: DynamicOverlay) -> tuple:
    """Only the fields that actually change F(x)/the domain matter here —
    leaf/window state only affects gas permeability, not walkability or speed."""
    return (
        tuple(sorted(overlay.door_states.items())),
        overlay.clutter_on,
        tuple(sorted((r.x0, r.y0, r.x1, r.y1) for r in overlay.extra_obstacles)),
    )


def compute_gf_field(
    building: Building, floor: Floor, grid: FloorGrid, *,
    exit_set: str = "all", overlay: DynamicOverlay | None = None,
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT,
) -> EikonalField:
    """Ground-floor field: targets are the chosen exit set's cells, T=0 there (N1)."""
    overlay = overlay or DynamicOverlay()
    exit_ids = resolve_exit_set(building, exit_set)
    targets = _exit_target_mask(floor, grid, overlay, exit_ids)
    if not targets.any():
        raise ValueError(f"exit set {exit_set!r} has no AVAILABLE exit on floor {floor.id}")

    domain = domain_mask(grid)
    speed = grid.speed_F_now(overlay, u_max=u_max, f_b=f_b)

    phi = np.ones(grid.shape, dtype=np.float64)
    phi[targets] = -1.0
    phi = np.ma.MaskedArray(phi, mask=~domain)

    T = skfmm.travel_time(phi, speed, dx=grid.transform.h)
    T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
    T.data[targets] = 0.0  # exact N1 boundary condition (FMM's sub-cell interpolation gives ~h/2, not 0)

    return EikonalField(floor_id=floor.id, exit_set=exit_set, T=T, domain=domain, speed=speed)


def compute_ff_field(
    building: Building, ff_floor: Floor, ff_grid: FloorGrid, gf_grid: FloorGrid, gf_field: EikonalField, *,
    overlay: DynamicOverlay | None = None, excluded_stairs: frozenset[str] = frozenset(),
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT,
) -> EikonalField:
    """Upper-floor field (PROMPT_01 §6.3): skfmm can't seed different initial
    times per target, so solve one field per stair (seeded at its FF landing)
    and combine:

        T_FF(x) = min_s [ T_FF,s(x) + t_stair,s + T_GF(discharge_s) ]

    t_stair,s = L_stair / v_stair, with v_stair = chi_s * V_REF (N4s, no
    crowding/health reduction modelled yet — rho_s=0, phi=1). T_GF(discharge_s)
    is 0 only when the stair has a direct discharge exit that's currently
    available (ST-W -> X-NW); otherwise it's looked up from the already-solved
    GF field at that stair's GF landing cells (gf_grid.stair_id uses the same
    stair indexing as ff_grid.stair_id — both built from building.stairs in
    dms/grid/raster.py).
    """
    overlay = overlay or DynamicOverlay()
    domain = domain_mask(ff_grid)
    speed = ff_grid.speed_F_now(overlay, u_max=u_max, f_b=f_b)

    combined: np.ndarray | None = None
    for sidx, stair_id in enumerate(ff_grid.stair_ids):
        if stair_id in excluded_stairs:
            continue
        stair = building.stairs[stair_id]
        landing_ff = (ff_grid.stair_id == sidx) & domain
        if not landing_ff.any():
            continue

        t_stair = stair.walking_length_m / (stair.chi * V_REF)

        use_direct_discharge = stair.discharge_exit_id is not None and \
            overlay.door_states.get(stair.discharge_exit_id) != "UNAVAILABLE"
        if use_direct_discharge:
            t_gf_discharge = 0.0
        else:
            landing_gf = (gf_grid.stair_id == sidx) & gf_field.domain & (~gf_field.T.mask)
            values = gf_field.T.data[landing_gf]
            if values.size == 0:
                continue  # this stair's GF landing is unreachable under the current overlay; skip it
            t_gf_discharge = float(values.min())

        phi = np.ones(ff_grid.shape, dtype=np.float64)
        phi[landing_ff] = -1.0
        phi = np.ma.MaskedArray(phi, mask=~domain)
        T_s = skfmm.travel_time(phi, speed, dx=ff_grid.transform.h)
        candidate = np.asarray(T_s.data, dtype=np.float64) + t_stair + t_gf_discharge

        combined = candidate if combined is None else np.minimum(combined, candidate)

    if combined is None:
        raise ValueError("every stair is excluded or unreachable; FF has no path to the ground floor")

    T = np.ma.MaskedArray(combined, mask=~domain)
    return EikonalField(floor_id=ff_floor.id, exit_set=gf_field.exit_set, T=T, domain=domain, speed=speed)


@dataclass(slots=True)
class FieldCache:
    """Caches EikonalFields by (building, floor, exit_set, overlay key,
    excluded_stairs) — PROMPT_01 §6.3. `calls` counts actual FMM solves, for
    tests asserting a no-op re-request doesn't recompute."""

    _gf: dict[tuple, EikonalField] = field(default_factory=dict)
    _ff: dict[tuple, EikonalField] = field(default_factory=dict)
    calls: int = 0

    def get_gf(self, building: Building, floor: Floor, grid: FloorGrid, *, exit_set: str = "all",
               overlay: DynamicOverlay | None = None, **kw) -> EikonalField:
        overlay = overlay or DynamicOverlay()
        key = (building.id, floor.id, exit_set, _overlay_key(overlay))
        cached = self._gf.get(key)
        if cached is None:
            self.calls += 1
            cached = compute_gf_field(building, floor, grid, exit_set=exit_set, overlay=overlay, **kw)
            self._gf[key] = cached
        return cached

    def get_ff(self, building: Building, ff_floor: Floor, ff_grid: FloorGrid, gf_grid: FloorGrid, *,
               exit_set: str = "all", overlay: DynamicOverlay | None = None,
               excluded_stairs: frozenset[str] = frozenset(), **kw) -> EikonalField:
        overlay = overlay or DynamicOverlay()
        gf_floor = building.get_floor(building.floor_ids()[0])
        gf_field = self.get_gf(building, gf_floor, gf_grid, exit_set=exit_set, overlay=overlay, **kw)
        key = (building.id, ff_floor.id, exit_set, _overlay_key(overlay), tuple(sorted(excluded_stairs)))
        cached = self._ff.get(key)
        if cached is None:
            self.calls += 1
            cached = compute_ff_field(building, ff_floor, ff_grid, gf_grid, gf_field,
                                       overlay=overlay, excluded_stairs=excluded_stairs, **kw)
            self._ff[key] = cached
        return cached

    def invalidate(self, building_id: str | None = None) -> None:
        if building_id is None:
            self._gf.clear()
            self._ff.clear()
            return
        self._gf = {k: v for k, v in self._gf.items() if k[0] != building_id}
        self._ff = {k: v for k, v in self._ff.items() if k[0] != building_id}


def stairs_blocked_by_scenario(site: Site, building: Building, scenario_name: str) -> frozenset[str]:
    """Stair ids a scenario's events take UNAVAILABLE — lets the field layer
    exclude a stair from the FF combination without needing a NavGraph."""
    preset = site.scenarios[scenario_name]
    blocked = set()
    for ev in preset.get("events", []):
        if ev.get("type") == "SetLinkState" and ev.get("state") == "UNAVAILABLE" and ev.get("id") in building.stairs:
            blocked.add(ev["id"])
    return frozenset(blocked)


def direction_field(
    T: np.ma.MaskedArray, domain: np.ndarray, h: float, *, degraded: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """e = -grad(T)/|grad(T)| (N1), central differences with one-sided
    differences next to walls/domain edges (spec §6.3).

    `degraded`: cells that are in-domain but F_b-slow under the *current*
    overlay — pass `field.speed < some_small_threshold` (NOT a static layer
    like grid.clutter_mask: clutter cells are only actually F_b when
    clutter_on=True, and using the static layer unconditionally would wrongly
    exclude perfectly normal-speed cells as neighbours whenever clutter is
    off). A cell wedged between a wall (out of domain) and a degraded cell
    has no clean neighbour on either side; falling back to a one-sided
    difference against the degraded neighbour divides by F_b's near-zero
    speed and produces a wildly oversized gradient component (T jumps by
    hundreds of seconds across one cell) that drowns out the correct
    direction from the other axis. Excluding degraded cells as *differencing
    neighbours* (they can still receive a direction themselves) avoids that;
    the lost accuracy is minor since those neighbours were barely
    informative anyway.
    """
    data = np.asarray(T.filled(0.0), dtype=np.float64)
    neighbor_ok = domain if degraded is None else (domain & ~degraded)
    gx, gy = _one_sided_gradient(data, domain, h, neighbor_ok)
    mag = np.hypot(gx, gy)
    with np.errstate(invalid="ignore", divide="ignore"):
        ex = np.where(mag > 1e-12, -gx / mag, 0.0)
        ey = np.where(mag > 1e-12, -gy / mag, 0.0)
    ex = np.where(domain, ex, np.nan)
    ey = np.where(domain, ey, np.nan)
    return ex, ey


def _one_sided_gradient(
    T: np.ndarray, domain: np.ndarray, h: float, neighbor_ok: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    neighbor_ok = domain if neighbor_ok is None else neighbor_ok
    gx = np.zeros_like(T)
    gy = np.zeros_like(T)

    left_ok = np.zeros_like(domain)
    left_ok[:, 1:] = neighbor_ok[:, :-1]
    right_ok = np.zeros_like(domain)
    right_ok[:, :-1] = neighbor_ok[:, 1:]
    T_left = np.roll(T, 1, axis=1)
    T_right = np.roll(T, -1, axis=1)

    both_x = domain & left_ok & right_ok
    only_right_x = domain & right_ok & ~left_ok
    only_left_x = domain & left_ok & ~right_ok
    gx[both_x] = (T_right[both_x] - T_left[both_x]) / (2 * h)
    gx[only_right_x] = (T_right[only_right_x] - T[only_right_x]) / h
    gx[only_left_x] = (T[only_left_x] - T_left[only_left_x]) / h

    up_ok = np.zeros_like(domain)
    up_ok[1:, :] = neighbor_ok[:-1, :]
    down_ok = np.zeros_like(domain)
    down_ok[:-1, :] = neighbor_ok[1:, :]
    T_up = np.roll(T, 1, axis=0)
    T_down = np.roll(T, -1, axis=0)

    both_y = domain & up_ok & down_ok
    only_down_y = domain & down_ok & ~up_ok
    only_up_y = domain & up_ok & ~down_ok
    gy[both_y] = (T_down[both_y] - T_up[both_y]) / (2 * h)
    gy[only_down_y] = (T_down[only_down_y] - T[only_down_y]) / h
    gy[only_up_y] = (T[only_up_y] - T_up[only_up_y]) / h

    return gx, gy
