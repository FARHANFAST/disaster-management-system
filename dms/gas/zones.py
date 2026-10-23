"""Where the gas model meets the graph the evacuees decide on.

Every node of GraphEvacEnv (the graph the graph-RL agent and the A* planner
use) gets a concentration:

  * zone nodes — rooms, corridor segments, stubs, stair landings — are one
    well-mixed box each (dms/gas/indoor.py). A room is exactly one box, as
    the design doc specifies; a corridor is split into its graph segments so
    a 60 m ring corridor is not averaged across the whole building;
  * door nodes (interior and fire doors) have no air volume of their own and
    read the worst of the zones they join;
  * exit nodes read the OUTDOOR plume along their egress path: the straight
    walk from the doorway to the exit's assembly point (the nearest one whose
    path does not cut through the building). An exit counts as gassed when
    any point on that walk is — stepping out into clean air and then walking
    through the plume is not a safe exit.

Positions are site (L3) metres; a zone's receptor height is its floor's
level plus the 1.5 m breathing zone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..geometry.building import Building
from ..geometry.site import Site
from ..rl.graph_env import GraphEvacEnv
from .dispersion import RECEPTOR_Z

ZONE, DOOR, EXIT = 0, 1, 2
_ZONE_KINDS = ("room", "corridor", "stub", "stair_landing")
_DOOR_KINDS = ("interior", "fire_door")
EGRESS_SPACING_M = 2.0


def _crosses_footprint(a: tuple[float, float], b: tuple[float, float], building: Building, inset: float = 0.5) -> bool:
    """Does segment a-b pass through the building's (inset) footprint?
    Sampled — exits sit on the wall, so the inset keeps the doorway itself
    from counting."""
    fp = next(iter(building.floors.values())).footprint
    x0, y0, x1, y1 = fp.x0 + inset, fp.y0 + inset, fp.x1 - inset, fp.y1 - inset
    for f in np.linspace(0.0, 1.0, 200):
        lx, ly = building.from_site_xy(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]))
        if x0 < lx < x1 and y0 < ly < y1:
            return True
    return False


def egress_path(site: Site, building: Building, exit_xy: tuple[float, float]) -> tuple[str, np.ndarray]:
    """(assembly point id, sample points from the doorway to it)."""
    best = None
    for ap in site.assembly_points.values():
        d = math.hypot(ap.x - exit_xy[0], ap.y - exit_xy[1])
        if not _crosses_footprint(exit_xy, (ap.x, ap.y), building) and (best is None or d < best[0]):
            best = (d, ap)
    if best is None:
        raise ValueError(f"exit at {exit_xy}: no assembly point reachable without crossing the building")
    d, ap = best
    n = max(2, int(math.ceil(d / EGRESS_SPACING_M)) + 1)
    f = np.linspace(0.0, 1.0, n)[:, None]
    return ap.id, np.asarray(exit_xy) + f * (np.array([ap.x, ap.y]) - np.asarray(exit_xy))


@dataclass(slots=True)
class NodeZones:
    node_kind: np.ndarray  # per graph node: ZONE | DOOR | EXIT
    zone_of_node: np.ndarray  # zone index, -1 for door / exit nodes
    zone_node: np.ndarray  # graph node of each zone
    zone_xy: np.ndarray  # (n_zones, 2) site metres
    zone_z: np.ndarray  # receptor height, m
    zone_volume: np.ndarray  # m³
    zone_is_room: np.ndarray  # bool — where an agent may shelter in place
    door_zones: list[list[int]]  # per graph node: zones a door node joins (empty otherwise)
    exit_nodes: np.ndarray  # graph node indices of exits
    exit_xy: np.ndarray  # (n_exits, 2) site metres
    zones_by_exit: dict[int, list[int]]  # exit node -> zones it opens onto
    exit_ap: list[str]  # assembly point each exit's egress path leads to
    egress_xy: list[np.ndarray]  # per exit: (n_i, 2) sample points, doorway first

    @property
    def n_zones(self) -> int:
        return self.zone_node.size

    @classmethod
    def from_graph_env(cls, env: GraphEvacEnv, site: Site, building: Building, *,
                       ceiling_m: float = 3.0) -> NodeZones:
        ce = env.cell_env
        n = env.n_nodes
        kind = np.empty(n, dtype=np.int8)
        for k, nid in enumerate(env.node_ids):
            kd = env.nav.node(nid).kind_detail
            kind[k] = ZONE if kd in _ZONE_KINDS else DOOR if kd in _DOOR_KINDS else EXIT if kd == "exit" else -1
            if kind[k] < 0:
                raise ValueError(f"node {nid}: no gas rule for kind {kd!r}")

        zone_node = np.flatnonzero(kind == ZONE)
        zone_of_node = np.full(n, -1, dtype=np.int64)
        zone_of_node[zone_node] = np.arange(zone_node.size)

        def centroid(k: int) -> tuple[float, float]:
            local = np.mean([ce.state_xy(int(s)) for s in env.cells_of_node[k]], axis=0)
            return building.to_site_xy(float(local[0]), float(local[1]))

        floor_z = {fid: building.get_floor(fid).z for fid in building.floor_ids()}
        zone_xy = np.array([centroid(int(k)) for k in zone_node]).reshape(-1, 2)
        zone_z = np.array([floor_z[ce.floor_ids[ce.cell_floor[env.cells_of_node[k][0]]]] + RECEPTOR_Z
                           for k in zone_node])
        zone_volume = np.array([len(env.cells_of_node[k]) * ce.h**2 * ceiling_m for k in zone_node])
        zone_is_room = np.array([env.nav.node(env.node_ids[k]).kind_detail == "room" for k in zone_node])

        def zone_neighbours(k: int) -> list[int]:
            return sorted({int(zone_of_node[v]) for v in env.neighbours[k][env.valid[k]] if zone_of_node[v] >= 0})

        door_zones = [zone_neighbours(k) if kind[k] == DOOR else [] for k in range(n)]
        exit_nodes = np.flatnonzero(kind == EXIT)
        exit_xy = np.array([env.nav.node(env.node_ids[k]).XY for k in exit_nodes]).reshape(-1, 2)
        zones_by_exit = {int(k): zone_neighbours(int(k)) for k in exit_nodes}
        egress = [egress_path(site, building, (float(x), float(y))) for x, y in exit_xy]
        return cls(kind, zone_of_node, zone_node, zone_xy, zone_z, zone_volume, zone_is_room,
                   door_zones, exit_nodes, exit_xy, zones_by_exit,
                   [ap for ap, _ in egress], [pts for _, pts in egress])

    def node_ppm(self, zone_ppm: np.ndarray, exit_ppm: np.ndarray) -> np.ndarray:
        """Concentration on every graph node from zone (indoor) and exit
        (outdoor) values. Works on a leading time axis too: (..., n_zones)."""
        zone_ppm = np.asarray(zone_ppm)
        out = np.zeros(zone_ppm.shape[:-1] + (self.node_kind.size,))
        out[..., self.zone_node] = zone_ppm
        out[..., self.exit_nodes] = exit_ppm
        for k, zs in enumerate(self.door_zones):
            if zs:
                out[..., k] = zone_ppm[..., zs].max(axis=-1)
        return out
