"""Build the L2 navigation graph G=(N,E) from a Site's compiled geometry
(spec §6.1). Node ids follow the spec's convention:
`GF:R:G-16` (room), `GF:C:ringS-03` (corridor segment), `GF:D:G-16-d2`
(door), `GF:O:<id>` (opening), `GF:FD:<id>` (fire door), `ST-W:GF` (stair
landing), `X:X-S` (exit), `AP:AP-N` (assembly point).

All Node.XY are stored in SITE (L3) metres — building.to_site_xy is a rigid
transform (rotation + translation), so straight-line distances in this frame
equal straight-line distances in the building's own local frame, which is
exactly what the A* heuristic needs, and it lets exit/AP nodes compare
directly against indoor nodes without a second coordinate system.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from ..geometry.building import Building, Floor, Space
from ..geometry.corridor_width import effective_width, lanes_for_segment
from ..geometry.primitives import Door, Opening, Stair
from ..geometry.site import Site
from ..grid.layers import FloorGrid
from ..grid.raster import build_grid_stack

RHO_CAP_DEFAULT = 2.0  # ped/m^2 — M3 project default [U]; not in the spec registry, verify before production use
MIN_SEGMENT_M = 1.5
MAX_SEGMENT_M = 6.0
F_S_LIT = 1.92  # ped/(m*s), spec §2.2 door/link discharge, lit default


@dataclass(slots=True)
class Node:
    id: str
    floor: str  # "GF" | "FF" | "SITE" (assembly points)
    kind: str  # platform | transition | vertical | hazard_ctrl | terminal
    area_m2: float
    width_m: float
    capacity: int
    XY: tuple[float, float]  # site (L3) metres
    building: str | None = None
    kind_detail: str = ""
    space_id: str | None = None
    ref_id: str | None = None  # underlying door/opening/stair/exit/AP id


@dataclass(slots=True)
class Link:
    u: str
    v: str
    length_m: float
    width_m: float
    slope_deg: float = 0.0
    oneway: bool = False
    state: str = "AVAILABLE"  # AVAILABLE | PARTIAL | UNAVAILABLE | BLOCKED_UNKNOWN
    shutter_closed: bool = False
    rho: float = 0.0
    C: float = 0.0
    temp: float = 20.0
    d_haz: float = float("inf")
    chi: float = 1.0  # capacity/speed factor; 1.0 level, 0.70 stairs (spec §6.1)

    @property
    def Q_max(self) -> float:
        """Spec: Q_max,l = f_s * w_eff,l * chi_l. PARTIAL (spec §5:
        e.g. one leaf of a double door jammed) halves the effective width."""
        partial_factor = 0.5 if self.state == "PARTIAL" else 1.0
        return F_S_LIT * self.width_m * self.chi * partial_factor


@dataclass(slots=True)
class NavGraph:
    building_id: str
    g: nx.Graph
    node_ids: list[str]
    door_node_id: dict[str, str] = field(default_factory=dict)
    opening_node_id: dict[str, str] = field(default_factory=dict)
    stair_landing_nodes: dict[str, dict[str, str]] = field(default_factory=dict)
    gate_node_id: dict[str, str] = field(default_factory=dict)  # gate id -> node id
    walkway_segment_nodes: dict[str, list[str]] = field(default_factory=dict)  # walkway id -> vertex node ids, in order
    node_of: dict[tuple[str, float], np.ndarray] = field(default_factory=dict)

    def node(self, node_id: str) -> Node:
        return self.g.nodes[node_id]["obj"]

    def link(self, u: str, v: str) -> Link:
        return self.g.edges[u, v]["obj"]

    def assembly_point_nodes(self) -> list[str]:
        return [n for n in self.node_ids if self.node(n).kind_detail == "assembly_point"]

    def room_nodes(self) -> list[str]:
        return [n for n in self.node_ids if self.node(n).kind_detail == "room"]


def _capacity(area_m2: float) -> int:
    return max(1, int(area_m2 * RHO_CAP_DEFAULT))


def _long_axis(space: Space) -> str:
    r = space.rect
    return "x" if r.width >= r.height else "y"


def _crossing_pos(crossing: Door | Opening, axis: str) -> float:
    """Project a door/opening span onto a corridor's long axis: the span's
    own coordinate if it sits on the corridor's end wall (a junction), else
    the midpoint of its span (a side doorway)."""
    return crossing.coord if crossing.axis == axis else (crossing.lo + crossing.hi) / 2.0


def _crossing_midpoint(crossing: Door | Opening) -> tuple[float, float]:
    mid = (crossing.lo + crossing.hi) / 2.0
    return (mid, crossing.coord) if crossing.axis == "y" else (crossing.coord, mid)


def _corridor_cut_points(floor: Floor, space: Space, axis: str) -> list[float]:
    lo_bound, hi_bound = (space.rect.x0, space.rect.x1) if axis == "x" else (space.rect.y0, space.rect.y1)
    cuts = {lo_bound, hi_bound}
    for d in floor.doors.values():
        if space.id in (d.space_a, d.space_b):
            cuts.add(min(max(_crossing_pos(d, axis), lo_bound), hi_bound))
    for o in floor.openings.values():
        if space.id in (o.space_a, o.space_b):
            cuts.add(min(max(_crossing_pos(o, axis), lo_bound), hi_bound))
    return sorted(cuts)


def _split_long(pieces: list[tuple[float, float]], *, max_len: float) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for lo, hi in pieces:
        length = hi - lo
        if length > max_len + 1e-9:
            n = math.ceil(length / max_len)
            step = length / n
            out.extend((lo + k * step, lo + (k + 1) * step) for k in range(n))
        else:
            out.append((lo, hi))
    return out


def _segment_lengths(
    cuts: list[float], *, min_len: float = MIN_SEGMENT_M, max_len: float = MAX_SEGMENT_M
) -> list[tuple[float, float]]:
    raw = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] - cuts[i] > 1e-6]
    merged = _split_long(raw, max_len=max_len)

    changed = True
    while changed and len(merged) > 1:
        changed = False
        for i, (lo, hi) in enumerate(merged):
            if hi - lo < min_len:
                if i == 0:
                    _nlo, nhi = merged[1]
                    merged[1] = (lo, nhi)
                    del merged[0]
                else:
                    plo, _phi = merged[i - 1]
                    merged[i - 1] = (plo, hi)
                    del merged[i]
                # a merge can push the absorbing neighbour back over max_len
                # (e.g. two 6 m pieces either side of a 0.25 m sliver), so
                # re-split before continuing the merge pass.
                merged = _split_long(merged, max_len=max_len)
                changed = True
                break
    return merged


def _segment_for_position(segs: list[tuple[float, float, str]], pos: float) -> str:
    """Half-open [lo, hi) — consistent with Rect containment elsewhere. A
    closed interval would resolve a position sitting exactly on the shared
    boundary between two segments to the *earlier* one every time, which can
    orphan the later segment when the crossing is physically inside its
    footprint rather than the earlier segment's (observed: a ring corridor's
    end-cap segment left with degree 1 because a corner opening at its own
    lower bound kept attaching to the previous segment instead)."""
    for lo, hi, nid in segs:
        if pos >= lo - 1e-6 and pos < hi - 1e-6:
            return nid
    return segs[0][2] if pos < segs[0][0] else segs[-1][2]


def _stair_for_space(building: Building, space_id: str) -> Stair | None:
    for stair in building.stairs.values():
        if space_id in stair.landings.values():
            return stair
    return None


class _Builder:
    """Internal construction state — scoped to one build_navigation_graph call."""

    def __init__(
        self, site: Site, building: Building, min_lane_width_m: float,
        outdoor_distances: dict[tuple[str, str], float] | None = None,
    ):
        self.site = site
        self.building = building
        self.outdoor_distances = outdoor_distances or {}
        self.min_lane_width_m = min_lane_width_m
        self.g = nx.Graph()
        self.nodes: dict[str, Node] = {}
        self.door_node_id: dict[str, str] = {}
        self.opening_node_id: dict[str, str] = {}
        self.room_or_stair_node: dict[tuple[str, str], str] = {}
        self.segments_by_space: dict[tuple[str, str], list[tuple[float, float, str]]] = {}
        self.long_axis_by_space: dict[tuple[str, str], str] = {}
        self.stair_landing_nodes: dict[str, dict[str, str]] = {}
        self.exit_node_ids: list[tuple[str, str, float]] = []  # (node_id, door_id, width_m),
        # filled by build_crossings, consumed by build_outdoor_network_graph to connect every exit into the
        # walkway/gate network (or, as a fallback, straight to every AP if the site declares no outdoor network data)
        self.gate_node_id: dict[str, str] = {}  # gate id -> its own node id
        self.walkway_segment_nodes: dict[str, list[str]] = {}  # walkway id -> its vertex node ids, in order

    def xy(self, local_x: float, local_y: float) -> tuple[float, float]:
        return self.building.to_site_xy(local_x, local_y)

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise ValueError(f"duplicate graph node id {node.id!r}")
        self.nodes[node.id] = node
        self.g.add_node(node.id, obj=node)

    def add_edge(self, a: str, b: str, link: Link) -> None:
        self.g.add_edge(a, b, obj=link)

    def dist(self, a: str, b: str) -> float:
        ax, ay = self.nodes[a].XY
        bx, by = self.nodes[b].XY
        return math.hypot(bx - ax, by - ay)

    def resolve_side(self, space_id: str, floor_id: str, crossing: Door | Opening) -> str:
        key = (floor_id, space_id)
        if key in self.room_or_stair_node:
            return self.room_or_stair_node[key]
        segs = self.segments_by_space[key]
        axis = self.long_axis_by_space[key]
        return _segment_for_position(segs, _crossing_pos(crossing, axis))

    # ------------------------------------------------------------- platforms

    def build_assembly_points(self) -> None:
        for ap_id, ap in self.site.assembly_points.items():
            nid = f"AP:{ap_id}"
            area = math.pi * (ap.diameter_m / 2.0) ** 2
            self.add_node(Node(
                id=nid, floor="SITE", kind="terminal", kind_detail="assembly_point",
                area_m2=area, width_m=ap.diameter_m, capacity=_capacity(area),
                XY=(ap.x, ap.y), ref_id=ap_id,
            ))

    def build_rooms_and_stairs(self, floor: Floor) -> None:
        for space in floor.spaces.values():
            if space.kind == "room":
                cx, cy = space.rect.center
                nid = f"{floor.id}:R:{space.id}"
                self.add_node(Node(
                    id=nid, floor=floor.id, kind="platform", kind_detail="room",
                    area_m2=space.rect.area, width_m=min(space.rect.width, space.rect.height),
                    capacity=_capacity(space.rect.area), XY=self.xy(cx, cy),
                    building=self.building.id, space_id=space.id,
                ))
                self.room_or_stair_node[(floor.id, space.id)] = nid
            elif space.kind == "stair":
                stair = _stair_for_space(self.building, space.id)
                assert stair is not None, f"stair space {space.id} has no owning Stair"
                cx, cy = space.rect.center
                nid = f"{stair.id}:{floor.id}"
                self.add_node(Node(
                    id=nid, floor=floor.id, kind="vertical", kind_detail="stair_landing",
                    area_m2=space.rect.area, width_m=stair.width_m,
                    capacity=_capacity(space.rect.area), XY=self.xy(cx, cy),
                    building=self.building.id, space_id=space.id, ref_id=stair.id,
                ))
                self.room_or_stair_node[(floor.id, space.id)] = nid
                self.stair_landing_nodes.setdefault(stair.id, {})[floor.id] = nid

    def build_corridor_segments(self, floor: Floor) -> None:
        for space in floor.spaces.values():
            if space.kind not in ("corridor", "stub"):
                continue
            axis = _long_axis(space)
            cuts = _corridor_cut_points(floor, space, axis)
            segs = _segment_lengths(cuts)

            if axis == "x":
                short_lo, short_hi = space.rect.y0, space.rect.y1
            else:
                short_lo, short_hi = space.rect.x0, space.rect.x1
            short_mid = (short_lo + short_hi) / 2.0

            # Link.width_m/Q_max reflect the *static* (default-state) clear
            # width (spec §4) -- default-on obstacles (furniture, and
            # any obstacle group that's active_by_default) narrow it;
            # default-off groups (toggleable clutter/maintenance) don't,
            # same principle as walkable_static only ever reflecting
            # furniture, never a toggleable group.
            default_active = [
                obs for obs in floor.obstacles.values()
                if obs.space_id == space.id and (obs.layer == "furniture" or obs.active_by_default)
            ]

            seg_nodes: list[tuple[float, float, str]] = []
            seg_widths: dict[str, float] = {}
            for i, (lo, hi) in enumerate(segs):
                mid = (lo + hi) / 2.0
                local_xy = (mid, short_mid) if axis == "x" else (short_mid, mid)
                nid = f"{floor.id}:C:{space.id}-{i:02d}"
                lanes = lanes_for_segment(short_lo, short_hi, lo, hi, axis, default_active)
                clear_width = effective_width(lanes, self.min_lane_width_m)
                self.add_node(Node(
                    id=nid, floor=floor.id, kind="platform", kind_detail=space.kind,
                    area_m2=(hi - lo) * clear_width, width_m=clear_width, capacity=_capacity((hi - lo) * clear_width),
                    XY=self.xy(*local_xy), building=self.building.id, space_id=space.id,
                ))
                seg_nodes.append((lo, hi, nid))
                seg_widths[nid] = clear_width

            self.segments_by_space[(floor.id, space.id)] = seg_nodes
            self.long_axis_by_space[(floor.id, space.id)] = axis

            for (_, _, n1), (_, _, n2) in itertools.pairwise(seg_nodes):
                link_width = min(seg_widths[n1], seg_widths[n2])
                self.add_edge(n1, n2, Link(u=n1, v=n2, length_m=self.dist(n1, n2), width_m=link_width))

    # ------------------------------------------------------------- crossings

    def build_crossings(self, floor: Floor) -> None:
        for door in floor.doors.values():
            xy = self.xy(*_crossing_midpoint(door))
            if door.kind == "exit":
                nid = f"X:{door.id}"
                self.add_node(Node(
                    id=nid, floor=floor.id, kind="terminal", kind_detail="exit",
                    area_m2=door.width_m * self.building.wall_thickness_m, width_m=door.width_m,
                    capacity=_capacity(door.width_m * self.building.wall_thickness_m),
                    XY=xy, building=self.building.id, ref_id=door.id,
                ))
                self.door_node_id[door.id] = nid
                inside = self.resolve_side(door.space_a, floor.id, door)
                self.add_edge(nid, inside, Link(u=nid, v=inside, length_m=max(self.dist(nid, inside), 0.1), width_m=door.width_m))
                # No more a single shortcut edge straight from
                # here to every AP -- build_outdoor_network_graph() connects
                # every exit into the real walkway/gate network instead, so
                # a gate or walkway segment is a real, individually-failable
                # part of the route, not invisible to the redundancy sweep.
                self.exit_node_ids.append((nid, door.id, door.width_m))
                continue

            is_fire = door.kind == "fire_door"
            prefix = "FD" if is_fire else "D"
            nid = f"{floor.id}:{prefix}:{door.id}"
            self.add_node(Node(
                id=nid, floor=floor.id,
                kind="hazard_ctrl" if is_fire else "transition",
                kind_detail=door.kind,
                area_m2=door.width_m * self.building.wall_thickness_m, width_m=door.width_m,
                capacity=_capacity(door.width_m * self.building.wall_thickness_m),
                XY=xy, building=self.building.id, ref_id=door.id,
            ))
            self.door_node_id[door.id] = nid
            for space_id in (door.space_a, door.space_b):
                target = self.resolve_side(space_id, floor.id, door)
                self.add_edge(nid, target, Link(u=nid, v=target, length_m=max(self.dist(nid, target), 0.1), width_m=door.width_m))

        for opening in floor.openings.values():
            xy = self.xy(*_crossing_midpoint(opening))
            nid = f"{floor.id}:O:{opening.id}"
            self.add_node(Node(
                id=nid, floor=floor.id, kind="transition", kind_detail="opening",
                area_m2=opening.width_m * self.building.wall_thickness_m, width_m=opening.width_m,
                capacity=_capacity(opening.width_m * self.building.wall_thickness_m),
                XY=xy, building=self.building.id, ref_id=opening.id,
            ))
            self.opening_node_id[opening.id] = nid
            for space_id in (opening.space_a, opening.space_b):
                target = self.resolve_side(space_id, floor.id, opening)
                self.add_edge(nid, target, Link(u=nid, v=target, length_m=max(self.dist(nid, target), 0.1), width_m=opening.width_m))

    # ------------------------------------------------------------- vertical

    def build_vertical_links(self) -> None:
        """Connects every pair of *consecutive* floors a stair actually
        serves (spec §8) -- `stair_landing_nodes` is already keyed by
        floor id generically; this no longer assumes exactly 2 floors or
        that they're floor_ids()[0]/[1]. A stair spanning a subset (e.g.
        only floors 1-2 of a taller building) connects just those floors
        it has landings on, in building floor order."""
        for stair in self.building.stairs.values():
            landings = self.stair_landing_nodes.get(stair.id, {})
            served = [fid for fid in self.building.floor_ids() if fid in landings]
            for lower_id, upper_id in itertools.pairwise(served):
                u, v = landings[lower_id], landings[upper_id]
                self.add_edge(u, v, Link(
                    u=u, v=v, length_m=stair.walking_length_m, width_m=stair.width_m,
                    slope_deg=stair.slope_deg, chi=stair.chi,
                ))

    # ------------------------------------------------------------- outdoor network

    def _nearest_node(self, from_id: str, candidates: list[str]) -> str | None:
        if not candidates:
            return None
        fx, fy = self.nodes[from_id].XY
        best, best_d = None, float("inf")
        for cid in candidates:
            cx, cy = self.nodes[cid].XY
            d = math.hypot(cx - fx, cy - fy)
            if d < best_d:
                best, best_d = cid, d
        return best

    def build_outdoor_network_graph(self) -> None:
        """Real nodes for every walkway vertex and gate, connected in
        sequence (each walkway segment its own failable Link, each gate its
        own failable node) and cross-linked to the nearest such node from
        every exit and AP -- replaces the old single exit<->AP shortcut
        edge per pair, so a gate or walkway segment is now a real part of
        the route a single-failure sweep can actually remove. Falls back to
        the original direct exit<->AP edges (straight-line, or the
        supplied `outdoor_distances` matrix) when the site declares no
        walkways/gates at all, so a building without outdoor network site data
        still produces a connected graph."""
        for wk in self.site.walkways.values():
            vids = []
            for i, (x, y) in enumerate(wk.points):
                vid = f"WK:{wk.id}:{i:02d}"
                self.add_node(Node(
                    id=vid, floor="SITE", kind="platform", kind_detail="walkway",
                    area_m2=0.0, width_m=wk.width_m, capacity=0, XY=(x, y), ref_id=wk.id,
                ))
                vids.append(vid)
            self.walkway_segment_nodes[wk.id] = vids
            for u, v in itertools.pairwise(vids):
                self.add_edge(u, v, Link(u=u, v=v, length_m=self.dist(u, v), width_m=wk.width_m))

        all_walkway_nodes = [nid for vids in self.walkway_segment_nodes.values() for nid in vids]
        for gate in self.site.gates.values():
            gid = f"GT:{gate.id}"
            self.add_node(Node(
                id=gid, floor="SITE", kind="transition", kind_detail="gate",
                area_m2=0.0, width_m=gate.width_m, capacity=0, XY=(gate.x, gate.y), ref_id=gate.id,
            ))
            self.gate_node_id[gate.id] = gid
            nearest = self._nearest_node(gid, all_walkway_nodes)
            if nearest is not None:
                self.add_edge(gid, nearest, Link(u=gid, v=nearest, length_m=self.dist(gid, nearest), width_m=gate.width_m))

        all_outdoor_nodes = all_walkway_nodes + list(self.gate_node_id.values())

        if not all_outdoor_nodes:
            for nid, door_id, width_m in self.exit_node_ids:
                for ap_id in self.site.assembly_points:
                    ap_nid = f"AP:{ap_id}"
                    length = self.outdoor_distances.get((door_id, ap_id), self.dist(nid, ap_nid))
                    self.add_edge(nid, ap_nid, Link(u=nid, v=ap_nid, length_m=length, width_m=width_m))
            return

        # Connect to the nearest vertex of EVERY walkway (plus the nearest
        # gate), not just the single globally-nearest node -- two separate
        # walkways that happen to both end near the same exit/AP (built
        # deliberately, for redundancy) must both actually connect to it;
        # picking only the one globally-nearest point would silently starve
        # every walkway but one of any real connection there.
        def _connect_to_outdoor_network(from_id: str, own_width_m: float) -> None:
            targets: set[str] = set()
            for vids in self.walkway_segment_nodes.values():
                nearest = self._nearest_node(from_id, vids)
                if nearest is not None:
                    targets.add(nearest)
            nearest_gate = self._nearest_node(from_id, list(self.gate_node_id.values()))
            if nearest_gate is not None:
                targets.add(nearest_gate)
            for t in targets:
                width_m = min(own_width_m, self.nodes[t].width_m)
                self.add_edge(from_id, t, Link(u=from_id, v=t, length_m=self.dist(from_id, t), width_m=width_m))

        for nid, _door_id, width_m in self.exit_node_ids:
            _connect_to_outdoor_network(nid, width_m)
        for ap_id in self.site.assembly_points:
            ap_nid = f"AP:{ap_id}"
            _connect_to_outdoor_network(ap_nid, self.nodes[ap_nid].width_m)


def _compute_node_of(floor: Floor, grid: FloorGrid, b: _Builder, node_ids: list[str]) -> np.ndarray:
    rows, cols = grid.shape
    out = np.full((rows, cols), -1, dtype=np.int32)
    id_to_idx = {nid: i for i, nid in enumerate(node_ids)}
    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0

    for space_idx, (sid, space) in enumerate(floor.spaces.items()):
        if space.kind == "void":
            continue
        # Restrict to walkable_static: a space's rect still "claims" its wall
        # and furniture cells by center-containment, but node_of must only
        # cover walkable cells (spec §5.2).
        mask = (grid.space_id == space_idx) & grid.walkable_static
        key = (floor.id, sid)
        if key in b.room_or_stair_node:
            out[mask] = id_to_idx[b.room_or_stair_node[key]]
        elif key in b.segments_by_space:
            axis = b.long_axis_by_space[key]
            segs = b.segments_by_space[key]
            rows_idx, cols_idx = np.nonzero(mask)
            if rows_idx.size == 0:
                continue
            pos = xc[cols_idx] if axis == "x" else yc[rows_idx]
            assigned = np.zeros(pos.shape, dtype=bool)
            for lo, hi, nid in segs:
                sel = (~assigned) & (pos >= lo - 1e-6) & (pos <= hi + 1e-6)
                if sel.any():
                    out[rows_idx[sel], cols_idx[sel]] = id_to_idx[nid]
                    assigned |= sel
            if not assigned.all():
                leftover = ~assigned
                for ridx, cidx, p in zip(rows_idx[leftover], cols_idx[leftover], pos[leftover]):
                    best = min(segs, key=lambda s: min(abs(p - s[0]), abs(p - s[1])))
                    out[ridx, cidx] = id_to_idx[best[2]]

    for idx, (did, door) in enumerate(floor.doors.items()):
        nid = b.door_node_id.get(did)
        if nid is not None:
            out[grid.door_id == idx] = id_to_idx[nid]

    return out


def build_navigation_graph(
    site: Site, building_id: str, *, compute_node_of: bool = True,
    outdoor_distances: dict[tuple[str, str], float] | None = None, use_outdoor_network: bool = True,
    min_lane_width_m: float | None = None,
) -> NavGraph:
    """`outdoor_distances` lets a caller supply a precomputed exit<->AP
    distance matrix (e.g. to reuse one OutdoorNetwork across several graph
    builds); otherwise it's computed fresh via dms/fields/outdoor.py unless
    `use_outdoor_network=False` (falls back to straight-line distance —
    useful for fast tests that don't care about outdoor obstacle routing).
    `min_lane_width_m` (config, not a code constant)
    defaults to configs/agent_defaults.yaml's value when not overridden."""
    building = site.get_building(building_id)
    if outdoor_distances is None and use_outdoor_network:
        from ..fields.outdoor import build_outdoor_network
        outdoor_distances = build_outdoor_network(site, building).exit_ap_distance_m
    if min_lane_width_m is None:
        from ..scenario.agent_defaults import load_agent_defaults
        min_lane_width_m = load_agent_defaults().min_lane_width_m
    b = _Builder(site, building, min_lane_width_m, outdoor_distances)

    b.build_assembly_points()
    for floor in building.floors.values():
        b.build_rooms_and_stairs(floor)
        b.build_corridor_segments(floor)
    for floor in building.floors.values():
        b.build_crossings(floor)
    b.build_vertical_links()
    b.build_outdoor_network_graph()

    node_ids = list(b.nodes.keys())
    nav = NavGraph(
        building_id=building.id, g=b.g, node_ids=node_ids,
        door_node_id=b.door_node_id, opening_node_id=b.opening_node_id,
        stair_landing_nodes=b.stair_landing_nodes,
        gate_node_id=b.gate_node_id, walkway_segment_nodes=b.walkway_segment_nodes,
    )

    if compute_node_of:
        for floor in building.floors.values():
            stack = build_grid_stack(building, floor)
            for h, grid in stack.grids.items():
                nav.node_of[(floor.id, h)] = _compute_node_of(floor, grid, b, node_ids)

    return nav
