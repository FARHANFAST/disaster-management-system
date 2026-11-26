"""L3 outdoor site network (spec §3.9): a
rasterised ground grid plus FMM-based exit<->assembly-point distances.
"Outdoor links from every exit to every AP, with lengths from an outdoor
FMM (not Euclidean, so obstacles count)" — replaces the straight-line
distance the graph builder used as a Milestone 3 placeholder.

No agents, no dispersion here — this is purely the static distance network,
same spirit as dms/fields/eikonal.py but for the one L3 ground plane instead
of per-floor grids.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import skfmm

from ..geometry.building import Building
from ..geometry.primitives import Door, Rect
from ..geometry.site import Site
from ..grid.transforms import GridTransform

DEFAULT_H_SITE = 1.0  # m — site is large (160x140 m); 1 m keeps the FMM solve cheap
DISCHARGE_MARGIN_M = 1.0  # how far outside its own wall an exit's outdoor sample point sits


@dataclass(slots=True)
class SiteGrid:
    transform: GridTransform
    walkable: np.ndarray  # bool — open ground, not inside any building footprint, obstacle, or fence
    building_mask: np.ndarray  # bool — True inside a building footprint
    walkway_mask: np.ndarray = None  # bool — True on a walkway (speed boost, not a
    # walkability restriction -- open ground off a walkway stays walkable, just slower/less preferred)
    fence_mask: np.ndarray = None  # bool — True where a fence physically blocks (already
    # excluded from `walkable` too; kept separately for rendering/diagnostics)


def _point_to_segments_dist(X: np.ndarray, Y: np.ndarray, points: tuple[tuple[float, float], ...]) -> np.ndarray:
    """Per-cell minimum distance to any segment of the polyline `points`
    (vectorized: one pass per segment, not per cell) -- shared by fence and
    walkway rasterization."""
    best = np.full(X.shape, np.inf)
    for (ax, ay), (bx, by) in itertools.pairwise(points):
        abx, aby = bx - ax, by - ay
        seg_len2 = abx * abx + aby * aby
        if seg_len2 <= 1e-12:
            d = np.hypot(X - ax, Y - ay)
        else:
            t = np.clip(((X - ax) * abx + (Y - ay) * aby) / seg_len2, 0.0, 1.0)
            px, py = ax + t * abx, ay + t * aby
            d = np.hypot(X - px, Y - py)
        best = np.minimum(best, d)
    return best


def _discharge_point_local(door: Door, footprint: Rect, *, margin: float = DISCHARGE_MARGIN_M) -> tuple[float, float]:
    """The point `margin` metres outside the building, directly in front of
    an exit door — the exit's own position sits exactly on the wall line,
    which is inside the (excluded) building footprint, so FMM can't seed
    there directly."""
    mid = (door.lo + door.hi) / 2.0
    if door.axis == "y":
        x = mid
        y = door.coord - margin if abs(door.coord - footprint.y0) <= 1e-6 else door.coord + margin
    else:
        y = mid
        x = door.coord - margin if abs(door.coord - footprint.x0) <= 1e-6 else door.coord + margin
    return x, y


def exit_outward_normal_site(building: Building, floor_id: str, door: Door) -> tuple[float, float]:
    """Unit vector pointing away from the building, in SITE (L3) coordinates
    — the local footprint-edge normal rotated by the building's own psi
    (direction only, so translation doesn't apply)."""
    footprint = building.get_floor(floor_id).footprint
    if door.axis == "y":
        nx, ny = (0.0, -1.0) if abs(door.coord - footprint.y0) <= 1e-6 else (0.0, 1.0)
    else:
        nx, ny = (-1.0, 0.0) if abs(door.coord - footprint.x0) <= 1e-6 else (1.0, 0.0)
    psi = math.radians(building.rotation_deg)
    c, s = math.cos(psi), math.sin(psi)
    return c * nx - s * ny, s * nx + c * ny


def exit_discharge_site_xy(building: Building, floor_id: str, door: Door, *, margin: float = DISCHARGE_MARGIN_M) -> tuple[float, float]:
    footprint = building.get_floor(floor_id).footprint
    x, y = _discharge_point_local(door, footprint, margin=margin)
    return building.to_site_xy(x, y)


def rasterize_site(
    site: Site, *, h: float = DEFAULT_H_SITE, parking_on: bool = False, agent_defaults=None,
) -> SiteGrid:
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()

    transform = GridTransform.for_footprint(site.footprint, h, margin=0.0)
    rows, cols = transform.shape
    x0, y0 = transform.origin_local
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    X, Y = np.meshgrid(xc, yc)

    building_mask = np.zeros((rows, cols), dtype=bool)
    for building in site.buildings.values():
        psi = math.radians(building.rotation_deg)
        c, s = math.cos(psi), math.sin(psi)
        ox, oy = building.origin
        dx, dy = X - ox, Y - oy
        lx = c * dx + s * dy
        ly = -s * dx + c * dy
        fp = building.get_floor(building.floor_ids()[0]).footprint
        building_mask |= (lx >= fp.x0) & (lx < fp.x1) & (ly >= fp.y0) & (ly < fp.y1)

    obstacle_mask = np.zeros((rows, cols), dtype=bool)
    for obs in site.obstacles.values():
        if obs.layer == "parking" and not parking_on:
            continue
        r = obs.rect
        obstacle_mask |= (X >= r.x0) & (X < r.x1) & (Y >= r.y0) & (Y < r.y1)

    # Fences block, except through their own (available)
    # gates -- a gate's opening is a disc of radius width_m/2 centred on the
    # gate's own point, carved out of the fence-blocked mask regardless of
    # which exact segment it's nearest to (gates sit exactly on a straight
    # fence segment in practice).
    fence_mask = np.zeros((rows, cols), dtype=bool)
    thickness = agent_defaults.site.fence_thickness_m
    for fence in site.fences.values():
        if len(fence.points) < 2:
            continue
        fence_mask |= _point_to_segments_dist(X, Y, fence.points) <= thickness / 2.0
    for gate in site.gates.values():
        if gate.default_state == "UNAVAILABLE":
            continue
        open_disc = np.hypot(X - gate.x, Y - gate.y) <= gate.width_m / 2.0
        fence_mask &= ~open_disc

    walkway_mask = np.zeros((rows, cols), dtype=bool)
    for wk in site.walkways.values():
        if len(wk.points) < 2:
            continue
        walkway_mask |= _point_to_segments_dist(X, Y, wk.points) <= wk.width_m / 2.0

    walkable = ~building_mask & ~obstacle_mask & ~fence_mask
    return SiteGrid(
        transform=transform, walkable=walkable, building_mask=building_mask,
        walkway_mask=walkway_mask, fence_mask=fence_mask,
    )


def site_speed_field(site_grid: SiteGrid, agent_defaults=None) -> np.ndarray:
    """FMM speed array for outdoor fields -- uniform
    speed=1 m/s (metres == seconds at unit speed) except on a walkway
    cell, which is `walkway_speed_factor` times cheaper to traverse, so an
    FMM solve naturally prefers routing along walkways over open ground
    without making open ground impassable."""
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    speed = np.ones(site_grid.transform.shape, dtype=np.float64)
    speed[site_grid.walkway_mask] = agent_defaults.site.walkway_speed_factor
    return speed


def compute_ap_distance_fields(site: Site, site_grid: SiteGrid, agent_defaults=None) -> dict[str, np.ma.MaskedArray]:
    """One FMM field per assembly point: travel time (seconds) at the
    walkway-aware outdoor speed (`site_speed_field`) from that AP to every
    walkable outdoor cell -- no longer plain distance once a walkway speed
    boost exists, but kept under its original name/shape since it's
    consumed as "cost" either way (`build_outdoor_network`'s own exit<->AP
    matrix)."""
    domain = site_grid.walkable
    speed = site_speed_field(site_grid, agent_defaults)
    fields: dict[str, np.ma.MaskedArray] = {}
    for ap_id, ap in site.assembly_points.items():
        row, col = site_grid.transform.local_to_cell(ap.x, ap.y)
        if not site_grid.transform.in_bounds(row, col) or not domain[row, col]:
            raise ValueError(f"assembly point {ap_id} is not on walkable outdoor ground")
        phi = np.ones(site_grid.transform.shape, dtype=np.float64)
        phi[row, col] = -1.0
        phi = np.ma.MaskedArray(phi, mask=~domain)
        T = skfmm.travel_time(phi, speed, dx=site_grid.transform.h)
        T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
        T.data[row, col] = 0.0
        fields[ap_id] = T
    return fields


def compute_nearest_ap_field(site: Site, site_grid: SiteGrid, agent_defaults=None) -> np.ma.MaskedArray:
    """Distance to the *nearest* assembly point, all 4 seeded simultaneously
    in one FMM solve — the field an evacuating agent would actually follow
    outdoors (spec N1's -grad(T) direction, same idea as the indoor fields'
    "nearest exit", just at site scale). Individual per-AP fields (see
    `compute_ap_distance_fields`) are for the exit<->AP distance *matrix*;
    this combined field is for navigation."""
    domain = site_grid.walkable
    speed = site_speed_field(site_grid, agent_defaults)
    phi = np.ones(site_grid.transform.shape, dtype=np.float64)
    seed_cells = []
    for ap_id, ap in site.assembly_points.items():
        row, col = site_grid.transform.local_to_cell(ap.x, ap.y)
        if not site_grid.transform.in_bounds(row, col) or not domain[row, col]:
            raise ValueError(f"assembly point {ap_id} is not on walkable outdoor ground")
        phi[row, col] = -1.0
        seed_cells.append((row, col))
    phi = np.ma.MaskedArray(phi, mask=~domain)
    T = skfmm.travel_time(phi, speed, dx=site_grid.transform.h)
    T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
    for row, col in seed_cells:
        T.data[row, col] = 0.0
    return T


def compute_ap_gather_field(
    site: Site, site_grid: SiteGrid, *, target_frac: float = 0.5, agent_defaults=None,
) -> np.ma.MaskedArray:
    """Distance to the nearest assembly point's *inner gather disc* (radius
    `target_frac * ap.diameter_m/2`, not the single centre point) -- an
    evacuation walker should head toward
    "inside the AP area", not one exact pixel, and feel free to approach
    from anywhere around its edge. Every AP's disc is seeded at once, same
    single-combined-solve shape as `compute_nearest_ap_field`. Uses
    the walkway-aware speed field, so the walker this feeds
    (dms/scenario/evacuation.py) naturally prefers walkways over open
    ground, and fences (with their gates) now shape `site_grid.walkable`
    itself -- the walker's own outdoor "clearance field" in both senses."""
    domain = site_grid.walkable
    speed = site_speed_field(site_grid, agent_defaults)
    tr = site_grid.transform
    rows, cols = tr.shape
    x0, y0 = tr.origin_local
    xc = (np.arange(cols) + 0.5) * tr.h + x0
    yc = (np.arange(rows) + 0.5) * tr.h + y0
    X, Y = np.meshgrid(xc, yc)

    targets = np.zeros((rows, cols), dtype=bool)
    for ap in site.assembly_points.values():
        r = target_frac * ap.diameter_m / 2.0
        targets |= (X - ap.x) ** 2 + (Y - ap.y) ** 2 <= r ** 2
    targets &= domain
    if not targets.any():
        raise ValueError("no assembly point's gather disc overlaps walkable outdoor ground")

    phi = np.ones((rows, cols), dtype=np.float64)
    phi[targets] = -1.0
    phi = np.ma.MaskedArray(phi, mask=~domain)
    T = skfmm.travel_time(phi, speed, dx=tr.h)
    T = np.ma.MaskedArray(np.asarray(T.data, dtype=np.float64), mask=~domain)
    T.data[targets] = 0.0
    return T


@dataclass(slots=True)
class OutdoorNetwork:
    site_grid: SiteGrid
    ap_fields: dict[str, np.ma.MaskedArray]
    nearest_ap_field: np.ma.MaskedArray
    exit_ap_distance_m: dict[tuple[str, str], float]  # (exit_id, ap_id) -> metres


def build_outdoor_network(
    site: Site, building: Building, *, h: float = DEFAULT_H_SITE, parking_on: bool = False, agent_defaults=None,
) -> OutdoorNetwork:
    site_grid = rasterize_site(site, h=h, parking_on=parking_on, agent_defaults=agent_defaults)
    ap_fields = compute_ap_distance_fields(site, site_grid, agent_defaults)
    nearest_ap_field = compute_nearest_ap_field(site, site_grid, agent_defaults)

    distances: dict[tuple[str, str], float] = {}
    for floor in building.floors.values():
        for door in floor.exit_doors:
            local_discharge = _discharge_point_local(door, floor.footprint)
            site_x, site_y = building.to_site_xy(*local_discharge)
            row, col = site_grid.transform.local_to_cell(site_x, site_y)
            if not site_grid.transform.in_bounds(row, col) or not site_grid.walkable[row, col]:
                raise ValueError(f"exit {door.id}'s discharge point is not on walkable outdoor ground")
            for ap_id, field in ap_fields.items():
                if field.mask[row, col]:
                    raise ValueError(f"exit {door.id} cannot reach assembly point {ap_id} around site obstacles")
                distances[(door.id, ap_id)] = float(field.data[row, col]) + DISCHARGE_MARGIN_M
    return OutdoorNetwork(site_grid=site_grid, ap_fields=ap_fields, nearest_ap_field=nearest_ap_field, exit_ap_distance_m=distances)
