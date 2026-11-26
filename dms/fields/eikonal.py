"""Static Eikonal travel-time fields T^eik_{b,k} (spec N1-N2, §1.6, §6.3):
per-floor FMM fields with multi-floor stair coupling.

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
from scipy import ndimage

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


def clearance_domain(grid: FloorGrid, overlay: DynamicOverlay, clearance_radius_m: float) -> np.ndarray:
    """"Configuration space" for a body of radius `clearance_radius_m`
    (spec §3.4, generalized from a
    fixed one-cell erosion to a real-metre distance transform): cells a
    body of this radius can actually occupy, i.e. `walkable_now` cells
    whose Euclidean distance to the nearest non-walkable cell is at least
    `clearance_radius_m` (`scipy.ndimage.distance_transform_edt` with
    `sampling=grid.transform.h`, so the threshold is metres, not a cell
    count -- a cell-count erosion is radius-correct only at the one `h` it
    happened to be tuned for). Reflects the *current* dynamic overlay
    (closed doors, toggled obstacles). Deliberately not forced to
    reproduce the old one-cell erosion bit-for-bit (more accurate instead,
    verified against it by the regression test, which reports where they
    differ rather than asserting equality)."""
    walkable = grid.walkable_now(overlay)
    dist = ndimage.distance_transform_edt(walkable, sampling=grid.transform.h)
    return walkable & (dist >= clearance_radius_m)


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


def _refuge_target_mask(floor: Floor, grid: FloorGrid, refuge_ids: list[str] | None = None) -> np.ndarray:
    targets = np.zeros(grid.shape, dtype=bool)
    for idx, rid in enumerate(grid.refuge_ids):
        if refuge_ids is not None and rid not in refuge_ids:
            continue
        targets |= grid.refuge_id == idx
    return targets


def compute_refuge_field(
    building: Building, floor: Floor, grid: FloorGrid, *,
    overlay: DynamicOverlay | None = None, refuge_ids: list[str] | None = None,
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT, clearance_radius_m: float | None = None,
) -> EikonalField:
    """The field a `can_use_stairs=False` body class uses
    on a floor it can't leave by stairs -- a local-only solve (no lower-floor
    stair coupling at all) seeded at this floor's refuge cells instead of its
    exits, otherwise identical in structure to `compute_floor_field`'s own
    base case (same clearance-domain target-restoration rule, same `T=0` exact
    boundary condition fixup)."""
    overlay = overlay or DynamicOverlay()
    domain = clearance_domain(grid, overlay, clearance_radius_m) if clearance_radius_m is not None else domain_mask(grid)
    speed = grid.speed_F_now(overlay, u_max=u_max, f_b=f_b)

    targets = _refuge_target_mask(floor, grid, refuge_ids)
    if not targets.any():
        raise ValueError(f"floor {floor.id} has no refuge cells to target")
    if clearance_radius_m is not None:
        domain = domain | targets

    phi = np.ones(grid.shape, dtype=np.float64)
    phi[targets] = -1.0
    phi = np.ma.MaskedArray(phi, mask=~domain)

    T = skfmm.travel_time(phi, speed, dx=grid.transform.h)
    T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
    T.data[targets] = 0.0

    return EikonalField(floor_id=floor.id, exit_set="__refuge__", T=T, domain=domain, speed=speed)


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
        tuple(sorted(overlay.active_groups.items())),
        tuple(sorted((r.x0, r.y0, r.x1, r.y1) for r in overlay.extra_obstacles)),
    )


def compute_floor_field(
    building: Building, floor: Floor, grid: FloorGrid, *,
    lower_floor: Floor | None = None, lower_grid: FloorGrid | None = None,
    lower_field: EikonalField | None = None,
    exit_set: str = "all", overlay: DynamicOverlay | None = None,
    excluded_stairs: frozenset[str] = frozenset(),
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT,
    clearance_radius_m: float | None = None,
) -> EikonalField:
    """One recursive step of spec §6.3's floor coupling, generalized to
    N floors (spec §8).

    `lower_field=None` is the base case: this floor is solved directly
    against its own exit set, T=0 there (N1) -- this is any building's
    bottom floor, not necessarily named "GF".

    Otherwise (`lower_floor`/`lower_grid`/`lower_field` all given), this
    floor's field is built the same way the old two-tier `compute_ff_field`
    built FF's: skfmm can't seed different initial times per target, so
    solve one field per connecting stair (seeded at its landing on *this*
    floor) and combine:

        T(x) = min_s [ T_s(x) + t_stair,s + T_lower(discharge_s) ]

    t_stair,s = L_stair / v_stair, with v_stair = chi_s * V_REF (N4s, no
    crowding/health reduction modelled yet -- rho_s=0, phi=1). A stair only
    participates in *this* recursive step if it lands on both this floor
    and `lower_floor` (`Stair.landings`) -- this is what lets a stair that
    serves only a subset of floors drop out cleanly, instead of the old
    code's hardcoded assumption that every stair connects FF straight to
    GF. T_lower(discharge_s) is 0 only when the stair has a direct
    discharge exit that's currently available; otherwise it's looked up
    from the already-solved lower field at that stair's landing cells on
    `lower_floor` (every floor's `grid.stair_id` indexes into the same
    global `building.stairs` list -- see dms/grid/raster.py -- so `sidx`
    is comparable across any two floors' grids).

    `clearance_radius_m` not None (spec §3.4, now generalized to any radius)
    solves on `clearance_domain(grid, overlay, clearance_radius_m)`
    instead of `domain_mask` -- the distance transform can exclude a target
    exit's own cells when they sit against the building envelope (the
    envelope is a non-walkable neighbour too), so the base case restores
    exactly the cells it's about to seed as T=0 targets after applying the
    clearance threshold, same as the independent reference checker's own
    `wgf = mv0 | tgt`. The recursive case (coupling to a lower floor) does
    *not* restore anything -- an upper floor has no exit targets of its own
    here, matching the reference exactly.
    """
    overlay = overlay or DynamicOverlay()
    domain = clearance_domain(grid, overlay, clearance_radius_m) if clearance_radius_m is not None else domain_mask(grid)
    speed = grid.speed_F_now(overlay, u_max=u_max, f_b=f_b)

    if lower_field is None:
        exit_ids = resolve_exit_set(building, exit_set)
        targets = _exit_target_mask(floor, grid, overlay, exit_ids)
        if not targets.any():
            raise ValueError(f"exit set {exit_set!r} has no AVAILABLE exit on floor {floor.id}")
        if clearance_radius_m is not None:
            domain = domain | targets

        phi = np.ones(grid.shape, dtype=np.float64)
        phi[targets] = -1.0
        phi = np.ma.MaskedArray(phi, mask=~domain)

        T = skfmm.travel_time(phi, speed, dx=grid.transform.h)
        T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
        T.data[targets] = 0.0  # exact N1 boundary condition (FMM's sub-cell interpolation gives ~h/2, not 0)

        return EikonalField(floor_id=floor.id, exit_set=exit_set, T=T, domain=domain, speed=speed)

    assert lower_floor is not None and lower_grid is not None
    combined: np.ndarray | None = None
    for sidx, stair_id in enumerate(grid.stair_ids):
        if stair_id in excluded_stairs:
            continue
        stair = building.stairs[stair_id]
        if lower_floor.id not in stair.landings:
            continue  # doesn't connect down to the floor directly below
        landing_here = (grid.stair_id == sidx) & domain
        if not landing_here.any():
            continue

        t_stair = stair.walking_length_m / (stair.chi * V_REF)

        use_direct_discharge = stair.discharge_exit_id is not None and \
            overlay.door_states.get(stair.discharge_exit_id) != "UNAVAILABLE"
        if use_direct_discharge:
            t_lower_discharge = 0.0
        else:
            landing_lower = (lower_grid.stair_id == sidx) & lower_field.domain & (~lower_field.T.mask)
            values = lower_field.T.data[landing_lower]
            if values.size == 0:
                continue  # this stair's landing on the lower floor is unreachable under the current overlay; skip it
            t_lower_discharge = float(values.min())

        phi = np.ones(grid.shape, dtype=np.float64)
        phi[landing_here] = -1.0
        phi = np.ma.MaskedArray(phi, mask=~domain)
        T_s = skfmm.travel_time(phi, speed, dx=grid.transform.h)
        candidate = np.asarray(T_s.data, dtype=np.float64) + t_stair + t_lower_discharge

        combined = candidate if combined is None else np.minimum(combined, candidate)

    if combined is None:
        raise ValueError(f"every stair connecting {floor.id} down to {lower_floor.id} is excluded or unreachable")

    T = np.ma.MaskedArray(combined, mask=~domain)
    return EikonalField(floor_id=floor.id, exit_set=lower_field.exit_set, T=T, domain=domain, speed=speed)


def compute_gf_field(
    building: Building, floor: Floor, grid: FloorGrid, *,
    exit_set: str = "all", overlay: DynamicOverlay | None = None,
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT,
) -> EikonalField:
    """Back-compat alias: a bottom-floor solve is `compute_floor_field`'s
    base case (`lower_field=None`). Kept under its original name/signature
    since it's imported directly by tests/viz/the ghost walker."""
    return compute_floor_field(building, floor, grid, exit_set=exit_set, overlay=overlay, u_max=u_max, f_b=f_b)


def compute_ff_field(
    building: Building, ff_floor: Floor, ff_grid: FloorGrid, gf_grid: FloorGrid, gf_field: EikonalField, *,
    overlay: DynamicOverlay | None = None, excluded_stairs: frozenset[str] = frozenset(),
    u_max: float = U_MAX_DEFAULT, f_b: float = F_B_DEFAULT,
) -> EikonalField:
    """Back-compat alias for the two-floor case: `gf_field`'s floor is
    `lower_floor`/`lower_grid`/`lower_field` in `compute_floor_field`'s
    general recursive step. Kept under its original name/signature since
    it's imported directly by tests/viz/the ghost walker."""
    gf_floor = building.get_floor(gf_field.floor_id)
    return compute_floor_field(
        building, ff_floor, ff_grid, lower_floor=gf_floor, lower_grid=gf_grid, lower_field=gf_field,
        overlay=overlay, excluded_stairs=excluded_stairs, u_max=u_max, f_b=f_b,
    )


@dataclass(slots=True)
class FieldCache:
    """Caches EikonalFields by (building, floor, exit_set, overlay key,
    excluded_stairs) — spec §6.3, generalized to N floors (spec
    §8): one bucket per floor id, `get_field` recurses bottom-up through
    `building.floor_ids()`, memoizing every floor it touches along the way
    (so solving floor 3 for the first time also populates floors 1 and 2's
    cache entries). `calls` counts actual FMM solves, for tests asserting a
    no-op re-request doesn't recompute.

    `get_gf`/`get_ff` are back-compat wrappers over `get_field`, kept under
    their original two-floor-shaped signatures since they're called
    directly by tests/viz/the ghost walker.
    """

    _fields: dict[str, dict[tuple, EikonalField]] = field(default_factory=dict)
    calls: int = 0

    def get_field(
        self, building: Building, floor_id: str, grids: dict[str, FloorGrid], *,
        exit_set: str = "all", overlay: DynamicOverlay | None = None,
        excluded_stairs: frozenset[str] = frozenset(), clearance_radius_m: float | None = None,
        can_use_stairs: bool = True, refuge_ids: list[str] | None = None, **kw,
    ) -> EikonalField:
        overlay = overlay or DynamicOverlay()
        floor_ids = building.floor_ids()
        idx = floor_ids.index(floor_id)
        # clearance_radius_m is part of the key -- spec §3.4's clearance
        # field (generalized to any radius) is a
        # different result from the ordinary one, and a different radius is a
        # different result again; none may ever collide with another in the
        # cache. This also means "cache one mask/field per distinct
        # radius, built lazily" -- every distinct radius a caller asks for
        # gets its own cache bucket automatically, with no special-casing.
        # can_use_stairs=False is also part of the key:
        # on any floor above the bottom it selects an entirely different
        # field (refuge-targeted, no stair coupling), never comparable to the
        # can_use_stairs=True field at the same radius.
        key = (
            building.id, floor_id, exit_set, _overlay_key(overlay), tuple(sorted(excluded_stairs)),
            clearance_radius_m, can_use_stairs, tuple(refuge_ids) if refuge_ids is not None else None,
        )
        bucket = self._fields.setdefault(floor_id, {})
        cached = bucket.get(key)
        if cached is not None:
            return cached

        floor = building.get_floor(floor_id)
        grid = grids[floor_id]
        if idx == 0:
            self.calls += 1
            cached = compute_floor_field(
                building, floor, grid, exit_set=exit_set, overlay=overlay,
                clearance_radius_m=clearance_radius_m, **kw,
            )
        elif not can_use_stairs:
            # This class never descends a stair, so this
            # floor's own refuges are its only target -- no lower-floor
            # coupling at all, not even to compute a field we'd discard.
            self.calls += 1
            cached = compute_refuge_field(
                building, floor, grid, overlay=overlay, refuge_ids=refuge_ids,
                clearance_radius_m=clearance_radius_m, **kw,
            )
        else:
            lower_id = floor_ids[idx - 1]
            # excluded_stairs only gates floor_id's own coupling step, same
            # as today's compute_ff_field never passed it down to the GF solve.
            lower_field = self.get_field(
                building, lower_id, grids, exit_set=exit_set, overlay=overlay,
                clearance_radius_m=clearance_radius_m, **kw,
            )
            self.calls += 1
            cached = compute_floor_field(
                building, floor, grid, lower_floor=building.get_floor(lower_id), lower_grid=grids[lower_id],
                lower_field=lower_field, exit_set=exit_set, overlay=overlay, excluded_stairs=excluded_stairs,
                clearance_radius_m=clearance_radius_m, **kw,
            )
        bucket[key] = cached
        return cached

    def get_gf(self, building: Building, floor: Floor, grid: FloorGrid, *, exit_set: str = "all",
               overlay: DynamicOverlay | None = None, **kw) -> EikonalField:
        return self.get_field(building, floor.id, {floor.id: grid}, exit_set=exit_set, overlay=overlay, **kw)

    def get_ff(self, building: Building, ff_floor: Floor, ff_grid: FloorGrid, gf_grid: FloorGrid, *,
               exit_set: str = "all", overlay: DynamicOverlay | None = None,
               excluded_stairs: frozenset[str] = frozenset(), **kw) -> EikonalField:
        gf_floor = building.get_floor(building.floor_ids()[0])
        grids = {gf_floor.id: gf_grid, ff_floor.id: ff_grid}
        return self.get_field(building, ff_floor.id, grids, exit_set=exit_set, overlay=overlay,
                               excluded_stairs=excluded_stairs, **kw)

    def invalidate(self, building_id: str | None = None) -> None:
        if building_id is None:
            self._fields.clear()
            return
        for floor_id, bucket in self._fields.items():
            self._fields[floor_id] = {k: v for k, v in bucket.items() if k[0] != building_id}


def stairs_blocked_by_scenario(site: Site, building: Building, scenario_name: str) -> frozenset[str]:
    """Stair ids a scenario's events take UNAVAILABLE — lets the field layer
    exclude a stair from the floor-coupling combination without needing a
    NavGraph. Recognizes both SetLinkState (legacy, graph-link-only) and the
    stricter SetStairState (spec §2, also blocks the stair's footprint
    in the grid overlay — see DynamicOverlay.blocked_stairs) identically for
    this purpose: either way, the stair drops out of the coupling step."""
    preset = site.scenarios[scenario_name]
    blocked = set()
    for ev in preset.get("events", []):
        if (
            ev.get("type") in ("SetLinkState", "SetStairState")
            and ev.get("state") == "UNAVAILABLE"
            and ev.get("id") in building.stairs
        ):
            blocked.add(ev["id"])
    return frozenset(blocked)


def direction_field(
    T: np.ma.MaskedArray, domain: np.ndarray, h: float, *, degraded: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """e = -grad(T)/|grad(T)| (N1), central differences with one-sided
    differences next to walls/domain edges (spec §6.3).

    `degraded`: cells that are in-domain but F_b-slow under the *current*
    overlay — pass `field.speed < some_small_threshold` (NOT a static layer
    like grid.obstacle_group_id: a toggleable obstacle cell is only actually
    F_b when its group is active in the overlay, and using the static layer
    unconditionally would wrongly exclude perfectly normal-speed cells as
    neighbours whenever that group is off). A cell wedged between a wall
    (out of domain) and a degraded cell
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
