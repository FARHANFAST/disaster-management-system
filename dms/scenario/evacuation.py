"""Vectorized, capacity-enforced evacuation walker (spec §4; made
fully config-driven and body-class-aware).

Hundreds of agents, numpy-vectorized across every agent at once (never a
per-agent Python loop): pre-movement delay, exits as token-bucket gates,
stairs as FIFO queues, a belief/truth Eikonal-field switch on BLOCKED_UNKNOWN
discovery. Locomotion is a light kinematic model -- desired heading chosen by
sampling T at N candidate headings ahead and picking the lowest, plus
density slow-down and pairwise disc separation -- explicitly *not* the SFM of
SYSTEM_SPEC.md (no such model exists yet; this is a visual/behavioral check
of the geometry, fields and capacity constraints, same scope as the existing
ghost walker and demo_simulation.py).

Builds entirely on already-generic dms/ primitives (FieldCache,
clearance_domain, overlays_from_scenario, sample_population,
effective_speed) -- no building-specific id/coordinate appears in this
module; every exit/stair/AP/floor this module touches comes from whatever
the loaded Building/Site actually declares. No numeric behaviour/physics
constant is hardcoded here either: every speed,
radius, margin, capacity and timing constant comes from an `AgentDefaults`
(configs/agent_defaults.yaml, dms/scenario/agent_defaults.py), overridable
per run/per call for later RL-training randomisation.

Headless by construction (RL readiness):
this module never imports matplotlib; `run_evacuation` is the clean,
documented entry point a trainer or batch runner can call directly.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

from ..fields.eikonal import FieldCache, clearance_domain
from ..fields.outdoor import (
    DISCHARGE_MARGIN_M,
    SiteGrid,
    compute_ap_gather_field,
    exit_discharge_site_xy,
    exit_outward_normal_site,
    rasterize_site,
)
from ..geometry.building import Building
from ..geometry.primitives import Door
from ..geometry.site import Site
from ..graph.builder import F_S_LIT
from ..grid.layers import H_FINE, DynamicOverlay, FloorGrid
from ..grid.raster import rasterize_floor
from .agent_defaults import AgentDefaults, BodyClass, load_agent_defaults
from .population import PlacedAgent, effective_speed, sample_population
from .presets import (
    blocked_stairs_from_events,
    door_states_from_events,
    overlays_from_scenario,
)

# agent state codes, matching the 7-state machine design (PRE/MOVE/QUEUE/STAIR/DISC/OUT/DONE),
# plus an 8th terminal state, IN_REFUGE -- a can_use_stairs=False
# agent that reached its floor's refuge. Safe, awaiting assisted evacuation, never "stuck".
PRE, MOVE, QUEUE, STAIR, DISC, OUT, DONE, IN_REFUGE = range(8)
STATE_NAMES = ("PRE", "MOVE", "QUEUE", "STAIR", "DISC", "OUT", "DONE", "IN_REFUGE")


# --------------------------------------------------------------------- world


@dataclass(slots=True)
class FloorKinematics:
    floor_id: str
    grid: FloorGrid
    move: np.ndarray  # bool [row, col], ground-truth walkable at this clearance radius
    belief_T: np.ndarray  # float [row, col], filled(inf) outside its own domain
    truth_T: np.ndarray  # float [row, col], filled(inf)
    stair_capture_id: np.ndarray  # int [row, col], the nearest movable stair cell's global stair index, or -1 if none exists at all
    stair_capture_dist_m: np.ndarray  # float [row, col], distance (m) to that nearest movable stair cell
    refuge_capture_dist_m: np.ndarray  # float [row, col], distance (m) to the nearest movable refuge cell, inf if this
    # floor has none at all -- same "near enough" capture pattern as stair_capture_dist_m


@dataclass(slots=True)
class StairInfo:
    id: str
    idx: int  # index into every floor grid's stair_ids (global, consistent across floors)
    width_m: float
    chi: float
    walking_length_m: float
    upper_floor: str
    lower_floor: str
    Q: float  # f_s * w * chi, persons/s


@dataclass(slots=True)
class ExitInfo:
    id: str
    floor_id: str
    seg: tuple[tuple[float, float], tuple[float, float]]  # local (x, y) endpoints
    mid: tuple[float, float]
    door: Door
    Q: float  # f_s * w_eff, persons/s
    open_true: bool  # AVAILABLE/PARTIAL under the truth overlay
    unknown: bool  # BLOCKED_UNKNOWN per this scenario's own events


@dataclass(slots=True)
class EvacuationWorld:
    building: Building
    site: Site
    scenario: str
    agent_defaults: AgentDefaults
    floor_order: list[str]  # bottom to top
    grids: dict[str, FloorGrid]
    belief_overlay: DynamicOverlay
    truth_overlay: DynamicOverlay
    blocked_stair_ids: frozenset[str]
    stairs: list[StairInfo]
    exits: list[ExitInfo]  # bottom floor's exits, fixed order
    site_grid: SiteGrid
    ap_gather_T: np.ndarray  # filled(inf), site frame
    ap_ids: list[str]
    ap_xy: np.ndarray  # (n_ap, 2), site frame
    ap_r: np.ndarray  # (n_ap,) m
    parking_on: bool
    _cache: FieldCache = field(default_factory=FieldCache)
    _radius_floors: dict[tuple[float, bool], dict[str, FloorKinematics]] = field(default_factory=dict)

    def floor_kinematics(self, clearance_radius_m: float, can_use_stairs: bool = True) -> dict[str, FloorKinematics]:
        """Lazily builds and caches the belief/truth clearance fields for
        one (body-class radius, can_use_stairs) pair -- caches one mask and
        field per distinct radius, built
        lazily; can_use_stairs=False picks
        an entirely different field on any floor above the bottom one
        (refuge-targeted, no stair coupling), so it's part of the cache key
        too, never comparable to the can_use_stairs=True field at the same
        radius."""
        key = (clearance_radius_m, can_use_stairs)
        cached = self._radius_floors.get(key)
        if cached is not None:
            return cached
        belief_fields: dict[str, object] = {}
        truth_fields: dict[str, object] = {}
        for fid in self.floor_order:
            for overlay, dest in ((self.belief_overlay, belief_fields), (self.truth_overlay, truth_fields)):
                try:
                    dest[fid] = self._cache.get_field(
                        self.building, fid, self.grids, overlay=overlay,
                        excluded_stairs=self.blocked_stair_ids, clearance_radius_m=clearance_radius_m,
                        can_use_stairs=can_use_stairs,
                    )
                except ValueError:
                    # A can_use_stairs=False class on a floor
                    # this building declares no refuge for at all -- leave this
                    # floor fully unreachable for it (T=inf everywhere) rather
                    # than crash; such a building/class combination is a known,
                    # reportable gap (the room/body-class reachability report
                    # already flags it), not a simulation error.
                    dest[fid] = None

        floors: dict[str, FloorKinematics] = {}
        for fid in self.floor_order:
            grid = self.grids[fid]
            move = clearance_domain(grid, self.truth_overlay, clearance_radius_m)
            stair_capture_id, stair_capture_dist_m = _stair_capture_fields(grid, move)
            refuge_capture_dist_m = _refuge_capture_dist(grid, move)
            bf, tf = belief_fields[fid], truth_fields[fid]
            belief_T = np.asarray(bf.T.filled(np.inf), dtype=np.float64) if bf is not None else np.full(grid.shape, np.inf)
            truth_T = np.asarray(tf.T.filled(np.inf), dtype=np.float64) if tf is not None else np.full(grid.shape, np.inf)
            floors[fid] = FloorKinematics(
                floor_id=fid, grid=grid, move=move, belief_T=belief_T, truth_T=truth_T,
                stair_capture_id=stair_capture_id, stair_capture_dist_m=stair_capture_dist_m,
                refuge_capture_dist_m=refuge_capture_dist_m,
            )
        self._radius_floors[key] = floors
        return floors


def _stair_capture_fields(grid: FloorGrid, move: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For every cell: the nearest movable stair cell's global stair index,
    and the (metre) distance to it -- a distance-based "near enough" stair
    capture, the same shape as the exit gate's own distance-to-segment
    check, instead of requiring an agent's position to truncate to the
    *exact* stair cell. Found necessary during follow-up
    stress testing: right next to a stair, the belief/truth T field is
    nearly flat (the stair's own landing is where T bottoms out), so an
    exact-cell-only capture can leave an agent orbiting the stair's
    footprint via the 24-heading walk + jitter without ever landing on a
    cell truncation-equal to it."""
    stair_cells = (grid.stair_id >= 0) & move
    if not stair_cells.any():
        return np.full(grid.shape, -1, dtype=np.int64), np.full(grid.shape, np.inf)
    dist, (nearest_r, nearest_c) = ndimage.distance_transform_edt(
        ~stair_cells, sampling=grid.transform.h, return_indices=True,
    )
    stair_capture_id = grid.stair_id[nearest_r, nearest_c]
    return stair_capture_id.astype(np.int64), dist


def _refuge_capture_dist(grid: FloorGrid, move: np.ndarray) -> np.ndarray:
    """Distance (m) to the nearest movable refuge cell, same "near enough"
    pattern as `_stair_capture_fields` -- inf everywhere
    if this floor has no refuge cells at all."""
    refuge_cells = (grid.refuge_id >= 0) & move
    if not refuge_cells.any():
        return np.full(grid.shape, np.inf)
    return ndimage.distance_transform_edt(~refuge_cells, sampling=grid.transform.h)


def _door_segment(door: Door) -> tuple[tuple[float, float], tuple[float, float]]:
    if door.axis == "y":
        return (door.lo, door.coord), (door.hi, door.coord)
    return (door.coord, door.lo), (door.coord, door.hi)


def build_world(
    site: Site, building: Building, scenario: str, *,
    h: float = H_FINE, agent_defaults: AgentDefaults | None = None,
) -> EvacuationWorld:
    """Builds every static per-scenario artifact the walker needs: the raw
    per-floor grids and overlays (clearance fields are solved lazily per
    distinct body-class radius -- see `EvacuationWorld.floor_kinematics`),
    stair metrics (already-derived Stair properties, spec §0/§2 --
    never recomputed here), exit segments/capacities, and the outdoor
    AP-gather field. `agent_defaults` defaults to
    `configs/agent_defaults.yaml`; pass a different one to override every
    body/physics parameter for one run (e.g. RL-training randomisation)."""
    agent_defaults = agent_defaults if agent_defaults is not None else load_agent_defaults()
    preset = site.scenarios[scenario] if scenario != "baseline" else {"events": []}
    belief_overlay, truth_overlay = overlays_from_scenario(preset, building)
    blocked_stair_ids = blocked_stairs_from_events(preset, building)
    door_states = door_states_from_events(preset)
    unknown_exit_ids = frozenset(did for did, state in door_states.items() if state == "BLOCKED_UNKNOWN")

    floor_order = building.floor_ids()
    grids = {fid: rasterize_floor(building, building.get_floor(fid), h) for fid in floor_order}

    stairs: list[StairInfo] = []
    bottom_grid = grids[floor_order[0]]
    for idx, stair_id in enumerate(bottom_grid.stair_ids):
        stair = building.stairs[stair_id]
        served = [fid for fid in floor_order if fid in stair.landings]
        for lower, upper in itertools.pairwise(served):
            stairs.append(StairInfo(
                id=stair_id, idx=idx, width_m=stair.width_m, chi=stair.chi,
                walking_length_m=stair.walking_length_m, upper_floor=upper, lower_floor=lower,
                Q=F_S_LIT * stair.width_m * stair.chi,
            ))

    bottom_floor = building.get_floor(floor_order[0])
    exits: list[ExitInfo] = []
    for door in bottom_floor.exit_doors:
        state = truth_overlay.door_states.get(door.id, door.default_state)
        seg = _door_segment(door)
        mid = ((seg[0][0] + seg[1][0]) / 2.0, (seg[0][1] + seg[1][1]) / 2.0)
        # SYSTEM_SPEC.md N5: PARTIAL state's capacity multiplier is 1/omega_p (omega_p=2.0 -> half).
        partial_capacity = door.width_m / agent_defaults.partial_capacity_omega_p if state == "PARTIAL" else door.width_m
        exits.append(ExitInfo(
            id=door.id, floor_id=floor_order[0], seg=seg, mid=mid, door=door,
            Q=F_S_LIT * partial_capacity,
            open_true=state in ("AVAILABLE", "PARTIAL"),
            unknown=door.id in unknown_exit_ids,
        ))

    active_groups = {ev["group"]: bool(ev["on"]) for ev in preset.get("events", []) if ev.get("type") == "ToggleLayer"}
    parking_on = active_groups.get("parking", False)
    site_grid = rasterize_site(site, parking_on=parking_on, agent_defaults=agent_defaults)
    ap_ids = list(site.assembly_points.keys())
    ap_xy = np.array([[site.assembly_points[a].x, site.assembly_points[a].y] for a in ap_ids])
    ap_r = np.array([site.assembly_points[a].diameter_m / 2.0 for a in ap_ids])
    ap_gather_T = np.asarray(
        compute_ap_gather_field(
            site, site_grid, target_frac=agent_defaults.kinematics.ap_target_frac, agent_defaults=agent_defaults,
        ).filled(np.inf),
        dtype=np.float64,
    )

    return EvacuationWorld(
        building=building, site=site, scenario=scenario, agent_defaults=agent_defaults,
        floor_order=floor_order, grids=grids, belief_overlay=belief_overlay, truth_overlay=truth_overlay,
        blocked_stair_ids=blocked_stair_ids, stairs=stairs, exits=exits, site_grid=site_grid,
        ap_gather_T=ap_gather_T, ap_ids=ap_ids, ap_xy=ap_xy, ap_r=ap_r, parking_on=parking_on,
    )


# --------------------------------------------------------------------- population


@dataclass(slots=True)
class Agent:
    floor_idx: int  # index into world.floor_order; len(floor_order) == outdoor site in simulate()'s own numbering
    x: float
    y: float
    persona: str
    v0: float
    room: str
    t_pre: float
    body_class: str
    clearance_radius_m: float
    can_use_stairs: bool
    stair_speed_factor: float


def _snap_into_move_mask(fk: FloorKinematics, space_ids: list[str], space_id: str, x: float, y: float) -> tuple[float, float]:
    """dms/scenario/population.py samples a generic room-interior margin
    (the agent's own body-class radius from the declared Space rect) that
    doesn't know about the walker's own, stricter clearance ("configuration
    space") domain -- a point legitimately inside a room by that margin can
    still fall just outside the clearance threshold next to a wall (this is
    exactly where the reference's own
    sampler instead draws candidates directly from its eroded mask).
    Rather than entangle the general-purpose population sampler with
    Eikonal/grid concerns it has no other use for, snap the rare
    out-of-mask placement to the nearest still-walkable cell in the same
    room -- generic over any building/room, no id/coordinate involved
    beyond the one the caller already placed."""
    tr = fk.grid.transform
    x0, y0 = tr.origin_local
    row = int((y - y0) / tr.h)
    col = int((x - x0) / tr.h)
    rows, cols = fk.move.shape
    if 0 <= row < rows and 0 <= col < cols and fk.move[row, col]:
        return x, y
    sidx = space_ids.index(space_id)
    mask = fk.move & (fk.grid.space_id == sidx)
    if not mask.any():
        return x, y  # no clear cell in this room at all (vanishingly small room) -- leave as-is
    cand_rows, cand_cols = np.nonzero(mask)
    cx, cy = tr.cell_to_local(cand_rows, cand_cols)
    d2 = (cx - x) ** 2 + (cy - y) ** 2
    j = int(np.argmin(d2))
    return float(cx[j]), float(cy[j])


def sample_agents(
    world: EvacuationWorld, profile: str, *, seed: int, N: int | None = None,
) -> list[Agent]:
    """Wraps dms/scenario/population.py's already-generic sampler (which
    now also samples each agent's body class) with a
    persona speed (`effective_speed`) and a pre-movement delay
    (`t_pre = exp(Normal(mu, sigma))`, both config)."""
    rng = np.random.default_rng(seed)
    body_classes = world.agent_defaults.body_classes
    placed: list[PlacedAgent] = sample_population(world.building, profile, seed=seed, N=N, body_classes=body_classes)
    pm = world.agent_defaults.pre_movement
    agents: list[Agent] = []
    for p in placed:
        v0 = effective_speed(p.persona, world.building.persona_mix)
        t_pre = float(np.exp(rng.normal(pm.t_pre_mu, pm.t_pre_sigma)))
        bc: BodyClass = body_classes[p.body_class]
        # A follow-up review asked to confirm the table, the
        # grid check and the walker all use "the" one door-passability rule
        # (width >= 2r + margin_m). Tried unifying all three on the margin-
        # inclusive (effective) radius and verified empirically: at that
        # radius, every one of B1-L's exits becomes a connectivity island
        # disconnected from its own floor's interior (clearance_domain at
        # r+margin/2 erodes B1-L's own corridors enough that no exit's
        # approach clears it anywhere) -- skfmm's own "no zero contour"
        # error on literally every exit, not a near-miss on a couple of
        # doors. door_margin_m is a DESIGN-TIME compliance buffer (is this
        # door provably, conservatively wide enough), not a physical
        # requirement a real body needs merely to fit through a gap -- so
        # the walker's own movement stays on the bare class radius (can
        # this body occupy this cell at all, full stop), same as every
        # other field solve. The table and the grid check
        # that verifies it both still use the SAME margin-inclusive rule as
        # each other (confirmed in tests/test_b1l_body_classes.py and
        # tests/test_clearance_fixture.py) -- that agreement, not forcing
        # the walker's own physics to match a design margin, is what
        # "exactly this rule" means here.
        fk = world.floor_kinematics(bc.clearance_radius_m, bc.can_use_stairs)[p.floor]
        x, y = _snap_into_move_mask(fk, fk.grid.space_ids, p.space_id, p.x, p.y)
        agents.append(Agent(
            floor_idx=world.floor_order.index(p.floor), x=x, y=y, persona=p.persona,
            v0=v0, room=p.space_id, t_pre=t_pre, body_class=p.body_class,
            clearance_radius_m=bc.clearance_radius_m, can_use_stairs=bc.can_use_stairs,
            stair_speed_factor=bc.stair_speed_factor,
        ))
    return agents


# --------------------------------------------------------------------- simulate


@dataclass(slots=True)
class Frame:
    t: float
    pos: np.ndarray  # (n, 2)
    floor: np.ndarray  # (n,) int, 0..len(floor_order)-1, or len(floor_order) for outdoor
    state: np.ndarray  # (n,) int
    know: np.ndarray  # (n,) bool


@dataclass(slots=True)
class SimResult:
    frames: list[Frame]
    exit_used: np.ndarray  # (n,) int, -1 if never
    t_exit: np.ndarray  # (n,) float, nan if never
    ap_used: np.ndarray  # (n,) int, -1 if never
    t_ap: np.ndarray  # (n,) float, nan if never
    discovered: np.ndarray  # (n,) bool, discovered an unknown-blocked exit at least once
    stair_used: np.ndarray  # (n,) int, -1 if never used a stair
    events: list[dict]
    t_end: float
    n_stuck: int  # agents neither DONE nor IN_REFUGE by t_end (includes awaiting_refuge -- never hidden from this count)
    awaiting_refuge: np.ndarray  # (n,) bool: not DONE/IN_REFUGE, can_use_stairs=False, not on the bottom floor --
    # never reached a refuge (none declared on this floor, or didn't arrive in time)
    n_awaiting_refuge: int
    in_refuge: np.ndarray  # (n,) bool: can_use_stairs=False agent that reached a refuge -- safe, not stuck
    n_in_refuge: int


def simulate(world: EvacuationWorld, agents: list[Agent], *, t_end: float | None = None, seed: int = 1) -> SimResult:
    kin = world.agent_defaults.kinematics
    perc = world.agent_defaults.perception
    sim_cfg = world.agent_defaults.simulation
    if t_end is None:
        t_end = sim_cfg.t_end_s
    rng = np.random.default_rng(seed)
    n = len(agents)
    n_floors = len(world.floor_order)
    SITE = n_floors  # outdoor pseudo-floor index

    pos = np.array([[a.x, a.y] for a in agents], dtype=np.float64)
    floor = np.array([a.floor_idx for a in agents], dtype=np.int64)
    v0 = np.array([a.v0 for a in agents], dtype=np.float64)
    t_pre = np.array([a.t_pre for a in agents], dtype=np.float64)
    radius = np.array([a.clearance_radius_m for a in agents], dtype=np.float64)
    can_use_stairs = np.array([a.can_use_stairs for a in agents], dtype=bool)
    stair_speed_factor = np.array([a.stair_speed_factor for a in agents], dtype=np.float64)
    distinct_radii = sorted(set(radius.tolist()))
    # Each body class has one fixed can_use_stairs value, so in practice radius
    # determines it uniquely -- but derive it from the actual agents, not that
    # assumption, so a future config where two classes share a radius fails
    # loudly (mixed True/False for the same r) rather than silently picking one.
    radius_cus: dict[float, bool] = {}
    for r in distinct_radii:
        vals = set(can_use_stairs[radius == r].tolist())
        if len(vals) > 1:
            raise ValueError(f"radius {r} is shared by both can_use_stairs=True and =False agents -- not supported")
        radius_cus[r] = vals.pop()

    state = np.full(n, PRE, dtype=np.int64)
    know = np.zeros(n, dtype=bool)
    disc_until = np.zeros(n)
    stair_idx = np.full(n, -1, dtype=np.int64)  # index into world.stairs
    stair_release = np.zeros(n)
    stair_arrive = np.zeros(n)
    exit_used = np.full(n, -1, dtype=np.int64)
    t_exit = np.full(n, np.nan)
    ap_used = np.full(n, -1, dtype=np.int64)
    t_ap = np.full(n, np.nan)
    discovered_once = np.zeros(n, dtype=bool)
    stair_used = np.full(n, -1, dtype=np.int64)

    stair_free = np.zeros(len(world.stairs))
    unknown_exits = [e for e in world.exits if e.unknown]
    true_exits_idx = [i for i, e in enumerate(world.exits) if e.open_true]
    ex_Q = np.array([e.Q for e in world.exits])
    ex_tokens = np.full(len(world.exits), sim_cfg.exit_token_burst)
    ex_seg = np.array([e.seg for e in world.exits]) if world.exits else np.zeros((0, 2, 2))

    ang = np.linspace(0, 2 * np.pi, kin.n_headings, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], axis=1)

    events: list[dict] = []
    rec: list[Frame] = []

    def movable_floor(fidx: int, r: float | None, P: np.ndarray) -> np.ndarray:
        assert r is not None  # every indoor floor has a real class radius; only SITE omits one
        fk = world.floor_kinematics(r, radius_cus[r])[world.floor_order[fidx]]
        tr = fk.grid.transform
        x0, y0 = tr.origin_local
        rows, cols = fk.move.shape
        row = np.floor((P[:, 1] - y0) / tr.h).astype(np.int64)
        col = np.floor((P[:, 0] - x0) / tr.h).astype(np.int64)
        ok = (row >= 0) & (col >= 0) & (row < rows) & (col < cols)
        out = np.zeros(len(P), dtype=bool)
        out[ok] = fk.move[row[ok], col[ok]]
        return out

    def movable_site(P: np.ndarray) -> np.ndarray:
        tr = world.site_grid.transform
        x0, y0 = tr.origin_local
        rows, cols = world.site_grid.walkable.shape
        row = np.floor((P[:, 1] - y0) / tr.h).astype(np.int64)
        col = np.floor((P[:, 0] - x0) / tr.h).astype(np.int64)
        ok = (row >= 0) & (col >= 0) & (row < rows) & (col < cols)
        out = np.zeros(len(P), dtype=bool)
        out[ok] = world.site_grid.walkable[row[ok], col[ok]]
        return out

    def movable(fidx: int, r: float | None, P: np.ndarray) -> np.ndarray:
        return movable_site(P) if fidx == SITE else movable_floor(fidx, r, P)

    def T_lookup(fidx: int, r: float, P: np.ndarray, kn: np.ndarray) -> np.ndarray:
        # always called for an indoor floor with a real class radius -- the outdoor site uses
        # T_lookup_site instead, which needs no radius at all (see the None sentinel note above).
        fk = world.floor_kinematics(r, radius_cus[r])[world.floor_order[fidx]]
        tr = fk.grid.transform
        x0, y0 = tr.origin_local
        rows, cols = fk.move.shape
        row = np.clip(np.floor((P[:, 1] - y0) / tr.h).astype(np.int64), 0, rows - 1)
        col = np.clip(np.floor((P[:, 0] - x0) / tr.h).astype(np.int64), 0, cols - 1)
        Tb = fk.belief_T[row, col]
        Tt = fk.truth_T[row, col]
        return np.where(kn, Tt, Tb)

    def T_lookup_site(P: np.ndarray) -> np.ndarray:
        tr = world.site_grid.transform
        x0, y0 = tr.origin_local
        rows, cols = world.site_grid.walkable.shape
        row = np.clip(np.floor((P[:, 1] - y0) / tr.h).astype(np.int64), 0, rows - 1)
        col = np.clip(np.floor((P[:, 0] - x0) / tr.h).astype(np.int64), 0, cols - 1)
        return world.ap_gather_T[row, col]

    t = 0.0
    step = 0
    while t < t_end:
        state[(state == PRE) & (t >= t_pre)] = MOVE
        state[(state == DISC) & (t >= disc_until)] = MOVE

        arr = np.where((state == STAIR) & (t >= stair_arrive))[0]
        for i in arr:
            si = world.stairs[stair_idx[i]]
            lower_fidx = world.floor_order.index(si.lower_floor)
            fk = world.floor_kinematics(radius[i])[si.lower_floor]
            land_mask = (fk.grid.stair_id == si.idx) & fk.move
            fp = np.argwhere(land_mask)
            T_here = fk.truth_T if know[i] else fk.belief_T
            T0 = T_here[fp[:, 0], fp[:, 1]]
            keep = max(1, len(fp) // kin.stair_arrival_best_frac)
            cand = fp[np.argsort(T0)[:keep]]
            land_row, land_col = cand[rng.integers(len(cand))]
            lx, ly = fk.grid.transform.cell_to_local(int(land_row), int(land_col))
            pos[i] = (lx, ly)
            floor[i] = lower_fidx
            state[i] = MOVE

        state[(state == QUEUE) & (t >= stair_release)] = STAIR

        new = pos.copy()
        active = np.where(state == MOVE)[0]
        for fidx in range(n_floors):
            idx_floor = active[floor[active] == fidx]
            if len(idx_floor) == 0:
                continue
            for r in distinct_radii:
                idx = idx_floor[radius[idx_floor] == r]
                if len(idx) == 0:
                    continue
                P = pos[idx]
                kn = know[idx]
                cand = P[:, None, :] + kin.look_indoor_m * dirs[None, :, :]
                flat = cand.reshape(-1, 2)
                mid_pts = (P[:, None, :] + 0.5 * kin.look_indoor_m * dirs[None, :, :]).reshape(-1, 2)
                Tc = T_lookup(fidx, r, flat, np.repeat(kn, kin.n_headings))
                ok = movable(fidx, r, flat) & movable(fidx, r, mid_pts)
                Tc = np.where(ok, Tc, np.inf).reshape(len(idx), kin.n_headings)
                Tc = Tc + rng.normal(0, kin.tie_break_eps_s, Tc.shape)
                best = np.argmin(Tc, axis=1)
                d = dirs[best] + rng.normal(0, kin.heading_jitter_sd, (len(idx), 2))
                d /= np.linalg.norm(d, axis=1, keepdims=True)
                new[idx] = P + d * (v0[idx] * kin.dt_s)[:, None]

        sidx = active[floor[active] == SITE]
        if len(sidx):
            P = pos[sidx]
            cand = P[:, None, :] + kin.look_outdoor_m * dirs[None, :, :]
            Tc = T_lookup_site(cand.reshape(-1, 2)).reshape(len(sidx), kin.n_headings)
            ok = movable(SITE, None, cand.reshape(-1, 2)).reshape(len(sidx), kin.n_headings)
            Tc = np.where(ok, Tc, np.inf)
            Tc = Tc + rng.normal(0, kin.tie_break_eps_s, Tc.shape)
            best = np.argmin(Tc, axis=1)
            new[sidx] = P + dirs[best] * (v0[sidx] * kin.dt_s)[:, None]

        # density slow-down + pairwise separation, strictly per floor (indoor floors and outdoor
        # site queried separately -- agents on different
        # floors must never interact).
        moving = np.where(np.isin(state, (MOVE, QUEUE, DISC, PRE)))[0]
        for fidx in range(n_floors + 1):
            grp = moving[floor[moving] == fidx]
            if len(grp) < 2:
                continue
            Q = new[grp]
            tree = cKDTree(Q)
            cnt = tree.query_ball_point(Q, kin.density_radius_m, return_length=True) - 1
            rho = cnt / (np.pi * kin.density_radius_m ** 2)
            fac = np.clip(1 - rho / kin.rho_max_per_m2, kin.density_slowdown_min_factor, kin.density_slowdown_max_factor)
            grp_active = np.isin(grp, active)
            ia = grp[grp_active]
            new[ia] = pos[ia] + (new[ia] - pos[ia]) * fac[grp_active, None]
            Q = new[grp]
            tree = cKDTree(Q)
            grp_radius = radius[grp]
            max_r = float(grp_radius.max())
            pairs = tree.query_pairs(2 * max_r, output_type="ndarray") if max_r > 0 else np.zeros((0, 2), dtype=np.int64)
            if len(pairs):
                a, b = pairs[:, 0], pairs[:, 1]
                req = grp_radius[a] + grp_radius[b]
                dv = Q[a] - Q[b]
                dist = np.linalg.norm(dv, axis=1) + 1e-9
                overlap = dist < req
                if overlap.any():
                    a, b, dist, req = a[overlap], b[overlap], dist[overlap], req[overlap]
                    push = ((req - dist) / 2)[:, None] * dv[overlap] / dist[:, None] * kin.separation_push
                    dp = np.zeros((len(grp), 2))
                    np.add.at(dp, a, push)
                    np.add.at(dp, b, -push)
                    still = ~grp_active
                    dp[still] *= kin.still_push_damping
                    new[grp] += dp

        # wall check per floor: slide on x or y if the full step is blocked, else stay put.
        upd_mask = np.isin(state, (MOVE, QUEUE, DISC, PRE))
        for fidx in range(n_floors + 1):
            for r in (distinct_radii if fidx < n_floors else [None]):
                idx = np.where(upd_mask & (floor == fidx) & (radius == r if fidx < n_floors else True))[0]
                if len(idx) == 0:
                    continue
                ok = movable(fidx, r, new[idx])
                bad = idx[~ok]
                if len(bad):
                    tx = np.c_[new[bad, 0], pos[bad, 1]]
                    ty = np.c_[pos[bad, 0], new[bad, 1]]
                    okx = movable(fidx, r, tx)
                    oky = movable(fidx, r, ty)
                    new[bad] = np.where(okx[:, None], tx, np.where(oky[:, None], ty, pos[bad]))
        pos[upd_mask] = new[upd_mask]

        # discovery of unknown-blocked exits
        if unknown_exits:
            g0 = np.where((state == MOVE) & (floor == 0) & ~know)[0]
            for e in unknown_exits:
                dd = np.hypot(*(pos[g0] - np.array(e.mid)).T)
                hit = g0[dd < perc.R_discover_m]
                if len(hit):
                    state[hit] = DISC
                    know[hit] = True
                    disc_until[hit] = t + perc.t_discover_s
                    for i in hit:
                        if not discovered_once.any():
                            events.append({"t": round(t, 1), "kind": "first_discovery", "exit": e.id})
                        discovered_once[i] = True

        # exit token-bucket gates
        if len(world.exits):
            ex_tokens = np.minimum(ex_tokens + ex_Q * kin.dt_s, sim_cfg.exit_token_burst)
            g0 = np.where((state == MOVE) & (floor == 0))[0]
            if len(g0):
                for k in true_exits_idx:
                    (ax_, ay_), (bx_, by_) = ex_seg[k]
                    P = pos[g0]
                    seg = np.array([bx_ - ax_, by_ - ay_])
                    seg_len2 = seg @ seg
                    u = np.clip(((P[:, 0] - ax_) * seg[0] + (P[:, 1] - ay_) * seg[1]) / seg_len2, 0, 1)
                    cp = np.c_[ax_ + u * seg[0], ay_ + u * seg[1]]
                    dd = np.hypot(*(P - cp).T)
                    near = np.where(dd < kin.exit_capture_m)[0]
                    take = int(ex_tokens[k])
                    out = g0[near[np.argsort(dd[near])][:take]]
                    if len(out):
                        ex_tokens[k] -= len(out)
                        exit_used[out] = k
                        t_exit[out] = t
                        floor[out] = SITE
                        state[out] = MOVE
                        door = world.exits[k].door
                        nx_, ny_ = exit_outward_normal_site(world.building, world.exits[k].floor_id, door)
                        tx_, ty_ = -ny_, nx_
                        base_x, base_y = exit_discharge_site_xy(
                            world.building, world.exits[k].floor_id, door, margin=DISCHARGE_MARGIN_M,
                        )
                        jitter = rng.uniform(-kin.exit_jitter_m, kin.exit_jitter_m, len(out))
                        pos[out] = np.c_[base_x + tx_ * jitter, base_y + ty_ * jitter]
                        g0 = np.setdiff1d(g0, out)

        # stair entry (any floor other than the bottom one) -- a body class with
        # can_use_stairs=False never enters the queue;
        # it targets a refuge instead (see refuge data below).
        for fidx in range(1, n_floors):
            fmoving = np.where((state == MOVE) & (floor == fidx) & can_use_stairs)[0]
            if len(fmoving) == 0:
                continue
            fk_by_r = {r: world.floor_kinematics(r)[world.floor_order[fidx]] for r in set(radius[fmoving].tolist())}
            sk = np.full(len(fmoving), -1, dtype=np.int64)
            for j, i in enumerate(fmoving):
                fk = fk_by_r[radius[i]]
                tr = fk.grid.transform
                x0, y0 = tr.origin_local
                row = int((pos[i, 1] - y0) / tr.h)
                col = int((pos[i, 0] - x0) / tr.h)
                if (0 <= row < fk.move.shape[0] and 0 <= col < fk.move.shape[1]
                        and fk.stair_capture_dist_m[row, col] <= kin.stair_capture_m):
                    sk[j] = fk.stair_capture_id[row, col]
            ent = fmoving[sk >= 0]
            for i, gsidx in zip(ent, sk[sk >= 0]):
                fk = fk_by_r[radius[i]]
                stair_name = fk.grid.stair_ids[gsidx]
                if stair_name in world.blocked_stair_ids:
                    continue
                si_list = [j for j, s in enumerate(world.stairs) if s.idx == gsidx and s.upper_floor == world.floor_order[fidx]]
                if not si_list:
                    continue
                si = si_list[0]
                stair = world.stairs[si]
                start = max(t, stair_free[si])
                stair_free[si] = start + 1.0 / stair.Q
                stair_idx[i] = si
                stair_release[i] = start
                stair_arrive[i] = start + stair.walking_length_m / (stair.chi * v0[i] * stair_speed_factor[i])
                stair_used[i] = si
                state[i] = QUEUE

        # refuge arrival (any floor other than the bottom one) --
        # the mirror image of stair entry above: only a can_use_stairs=False agent
        # ever targets a refuge (its own field is refuge-seeded, see
        # EvacuationWorld.floor_kinematics), and arrival here is terminal (safe,
        # awaiting assisted evacuation), never re-entering MOVE.
        for fidx in range(1, n_floors):
            rmoving = np.where((state == MOVE) & (floor == fidx) & ~can_use_stairs)[0]
            if len(rmoving) == 0:
                continue
            fk_by_r = {r: world.floor_kinematics(r, False)[world.floor_order[fidx]] for r in set(radius[rmoving].tolist())}
            for i in rmoving:
                fk = fk_by_r[radius[i]]
                tr = fk.grid.transform
                x0, y0 = tr.origin_local
                row = int((pos[i, 1] - y0) / tr.h)
                col = int((pos[i, 0] - x0) / tr.h)
                if (0 <= row < fk.move.shape[0] and 0 <= col < fk.move.shape[1]
                        and fk.refuge_capture_dist_m[row, col] <= kin.refuge_capture_m):
                    state[i] = IN_REFUGE

        # assembly point arrival
        s2 = np.where((state == MOVE) & (floor == SITE))[0]
        if len(s2):
            d = np.hypot(pos[s2, None, 0] - world.ap_xy[None, :, 0], pos[s2, None, 1] - world.ap_xy[None, :, 1])
            inside = d < (kin.ap_capture_frac * world.ap_r)[None, :]
            got = inside.any(axis=1)
            ii = s2[got]
            ap_used[ii] = inside[got].argmax(axis=1)
            t_ap[ii] = t
            state[ii] = DONE

        if step % kin.rec_every_steps == 0:
            rec.append(Frame(t=round(t, 2), pos=pos.copy(), floor=floor.copy(), state=state.copy(), know=know.copy()))
        if np.isin(state, (DONE, IN_REFUGE)).all():
            break
        t += kin.dt_s
        step += 1

    rec.append(Frame(t=round(t, 2), pos=pos.copy(), floor=floor.copy(), state=state.copy(), know=know.copy()))
    # IN_REFUGE is a terminal, safe state -- never counted
    # as stuck, and (since a refuge-arrival is how a can_use_stairs=False
    # agent now leaves MOVE on an upper floor) never counted as awaiting_refuge
    # either, which is specifically "not done AND has no refuge to go to yet."
    not_done = ~np.isin(state, (DONE, IN_REFUGE))
    n_stuck = int(not_done.sum())
    # A can_use_stairs=False agent not on the bottom floor and not in a
    # refuge has no legitimate way down (either this building/floor declares
    # no refuge at all, or it never reached one in time) -- still genuinely
    # "stuck" (not DONE, still counted in n_stuck above), but reported
    # separately so a real bug elsewhere never hides behind this known,
    # acknowledged gap, and so this known gap never silently passes a
    # stuck-count check either.
    awaiting_refuge = not_done & ~can_use_stairs & (floor != 0)
    n_awaiting_refuge = int(awaiting_refuge.sum())
    in_refuge = state == IN_REFUGE
    n_in_refuge = int(in_refuge.sum())
    return SimResult(
        frames=rec, exit_used=exit_used, t_exit=t_exit, ap_used=ap_used, t_ap=t_ap,
        discovered=discovered_once, stair_used=stair_used, events=events, t_end=t, n_stuck=n_stuck,
        awaiting_refuge=awaiting_refuge, n_awaiting_refuge=n_awaiting_refuge,
        in_refuge=in_refuge, n_in_refuge=n_in_refuge,
    )


def summarize(world: EvacuationWorld, agents: list[Agent], res: SimResult) -> list[str]:
    n = len(agents)
    last_exit = np.nanmax(res.t_exit) if np.isfinite(np.nanmax(res.t_exit, initial=-1)) else float("nan")
    last_ap = np.nanmax(res.t_ap) if np.isfinite(np.nanmax(res.t_ap, initial=-1)) else float("nan")
    lines = [
        (f"scenario={world.scenario}  agents={n}  "
         f"evacuated(at AP)={int((~np.isnan(res.t_ap)).sum())}  "
         f"last out of building={last_exit:.1f}s  last at AP={last_ap:.1f}s"),
    ]
    cnt = {e.id: int((res.exit_used == k).sum()) for k, e in enumerate(world.exits)}
    lines.append("per exit: " + ", ".join(f"{k}={v}" for k, v in cnt.items()))
    stair_names = [s.id for s in world.stairs]
    st = {name: int((res.stair_used == k).sum()) for k, name in enumerate(stair_names)}
    lines.append("per stair (upper-floor agents): " + ", ".join(f"{k}={v}" for k, v in st.items()))
    if any(e.unknown for e in world.exits):
        lines.append(f"agents who walked to the unknown-blocked exit and re-routed: {int(res.discovered.sum())}")
    if res.n_in_refuge:
        lines.append(f"in_refuge (safe, awaiting assisted evacuation)={res.n_in_refuge}")
    if res.n_stuck:
        other_stuck = res.n_stuck - res.n_awaiting_refuge
        lines.append(
            f"stuck={res.n_stuck} (awaiting_refuge={res.n_awaiting_refuge} -- "
            f"can_use_stairs=False agents off the bottom floor that never reached a refuge; "
            f"other_stuck={other_stuck})",
        )
    return lines


# --------------------------------------------------------------------- RL-ready entry point


@dataclass(slots=True)
class EvacuationRun:
    world: EvacuationWorld
    agents: list[Agent]
    result: SimResult


def run_evacuation(
    building_yaml_path, scenario: str, seed: int, *,
    profile: str = "day", N: int | None = None, t_end: float | None = None,
    agent_defaults: AgentDefaults | None = None, building_id: str | None = None,
) -> EvacuationRun:
    """Clean, documented, headless entry point:
    building YAML + scenario + seed (+ optional config
    overrides) in, a full run result out. Never imports matplotlib --
    rendering is a separate, later step (dms/viz/evacuation_video.py)."""
    from ..geometry.loader import load_site

    site = load_site(building_yaml_path)
    building = site.get_building(building_id) if building_id is not None else next(iter(site.buildings.values()))
    world = build_world(site, building, scenario, agent_defaults=agent_defaults)
    agents = sample_agents(world, profile, seed=seed, N=N)
    result = simulate(world, agents, t_end=t_end, seed=seed)
    return EvacuationRun(world=world, agents=agents, result=result)
