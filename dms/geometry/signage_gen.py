"""Generate exit signs by rule, from geometry +
routing, for ANY building -- hand-placed signs are no longer the source of
truth; this generator is.

Rule (stated explicitly, per review): a sign is generated at every exit,
every stair door, every corridor junction (a circulation node with 3+
*circulation* connections), and every turn (a circulation node with
exactly 2 circulation connections whose two neighbour directions are not
roughly collinear -- `signage.turn_cos_threshold`, configs/agent_defaults.
yaml). "Circulation connection" deliberately excludes a plain doorway into
a room: the nav graph (`dms/graph/builder.py`) cuts every corridor segment
at each door/opening along it, including ordinary room doors, so a
corridor segment node's raw graph degree also counts every room door
opening onto it -- counting those would put a sign at nearly every room
door (found by review). `_is_circulation_continuation_node` is the fix:
only a neighbour that is itself another corridor/stub segment, or a
door/opening/fire-door whose BOTH sides are corridor/stub space, counts
toward a node's junction/turn degree or candidacy. A plain door between a
corridor and a room never counts, on either end.

Along a straight corridor run with no junction/turn/exit/stair sign for a
long stretch, `_spacing_fill_candidates` adds extra directional signs so
no point is more than `signage.sign_max_spacing_m` from a visible sign
(IFC 1013 / NFPA 101 7.10 practice), de-duplicating anything within
`signage.sign_min_spacing_m` of an existing sign. Candidate positions come
from the building's own navigation graph (`dms/graph/builder.py`),
generically -- no id is ever hardcoded here.

Each sign's direction is the next hop of the truth-domain route for the
standard body class, in the baseline scenario: the local gradient
(`dms/fields/eikonal.py: direction_field`) of that class's own truth T
field at the sign's position, toward the nearest available exit. On an
upper floor this field is already the stair-coupled one (spec §6.3),
so the gradient there already points toward whichever stair leads to the
lowest overall time -- "upper floors route via stairs" falls out of the
existing field, not a special case here. A sign's `points_to` (which exit
its route eventually reaches, for `signage_check.py`'s own reachability
check and for reporting) is found by literally tracing that gradient,
floor to floor via stair capture, to an exit cell.

Generation is the default source: a YAML-declared sign already at the
generated id (the loader runs first) is an override and is left untouched;
any other YAML-declared sign is simply kept alongside the generated set.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

import numpy as np

from ..fields.eikonal import direction_field
from ..graph.builder import NavGraph, Node, build_navigation_graph
from ..grid.layers import FloorGrid
from .building import Building
from .primitives import Door, Sign
from .site import Site

# A node whose OWN kind_detail is one of these always continues the route
# (another stretch of corridor/stub). A door/opening/fire-door node only
# counts as continuing the route if it connects two circulation spaces --
# never if one side is a room (an ordinary room doorway must not count as
# a junction/turn connection on either end: see module docstring).
_ALWAYS_CIRCULATION = frozenset({"corridor", "stub"})
_CONDITIONAL_CIRCULATION = frozenset({"opening", "interior", "fire_door"})

_SIGN_TYPE_OF_REASON = {
    "exit": "exit_door", "stair": "stair_level", "junction": "directional",
    "turn": "directional", "spacing": "directional",
}


@dataclass(slots=True, frozen=True)
class SignCandidate:
    floor_id: str
    x: float
    y: float
    reason: str  # exit | stair | junction | turn
    node_id: str


def _door_heading_deg(door: Door, inside_rect) -> float:
    """The direction a body standing inside `inside_rect` (the door's own
    `space_a`) walks to cross this door -- used for exit/stair signs, where
    the local gradient is unreliable right at the target cell (T bottoms
    out there). Determined from which side of the wall `space_a` itself
    sits on, not from a candidate position, which can sit exactly on the
    wall line (an exit candidate's own position, ambiguous by `<`/`>`
    comparison). 0=+x/east, counterclockwise."""
    if door.axis == "x":
        return 0.0 if inside_rect.x1 <= door.coord + 1e-6 else 180.0
    return 90.0 if inside_rect.y1 <= door.coord + 1e-6 else 270.0


def _is_circulation_continuation_node(building: Building, node: Node) -> bool:
    """True if this graph node is a point the ROUTE continues through --
    another corridor/stub segment, or a door/opening/fire-door crossing
    whose both sides are corridor/stub space. False for a room/stair/exit
    node, and false for a plain doorway into a room even though the room
    side's own node exists in the graph -- the one check that keeps
    ordinary room doors out of the junction/turn rule entirely (see module
    docstring)."""
    if node.kind_detail in _ALWAYS_CIRCULATION:
        return True
    if node.kind_detail not in _CONDITIONAL_CIRCULATION:
        return False
    floor = building.floors.get(node.floor)
    if floor is None or node.ref_id is None:
        return False
    crossing = floor.doors.get(node.ref_id) or floor.openings.get(node.ref_id)
    if crossing is None:
        return False
    sa = floor.spaces.get(crossing.space_a)
    sb = floor.spaces.get(crossing.space_b)
    return sa is not None and sb is not None and sa.kind in _ALWAYS_CIRCULATION and sb.kind in _ALWAYS_CIRCULATION


def _junction_and_turn_candidates(nav: NavGraph, building: Building, *, turn_cos_threshold: float) -> list[SignCandidate]:
    """Node positions come back from the nav graph in SITE (L3) frame
    (`building.to_site_xy`) -- converted back to building-local
    (L1) frame here via `from_site_xy`, since every grid/field this module
    samples is in local frame."""
    def local_xy(node) -> tuple[float, float]:
        return building.from_site_xy(*node.XY)

    out: list[SignCandidate] = []
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.floor not in building.floors:
            continue  # a SITE-floor node (assembly point) -- not a building-floor candidate
        if node.kind_detail == "exit":
            x, y = local_xy(node)
            out.append(SignCandidate(floor_id=node.floor, x=x, y=y, reason="exit", node_id=nid))
            continue
        if node.kind_detail == "stair_landing":
            x, y = local_xy(node)
            out.append(SignCandidate(floor_id=node.floor, x=x, y=y, reason="stair", node_id=nid))
            continue
        if not _is_circulation_continuation_node(building, node):
            continue  # a room/stair/exit boundary, or a doorway onto a room -- never a route split itself
        neighbors = [n for n in nav.g.neighbors(nid) if _is_circulation_continuation_node(building, nav.node(n))]
        x, y = local_xy(node)
        if len(neighbors) >= 3:
            out.append(SignCandidate(floor_id=node.floor, x=x, y=y, reason="junction", node_id=nid))
        elif len(neighbors) == 2:
            n1, n2 = neighbors
            p0 = np.array((x, y))
            p1 = np.array(local_xy(nav.node(n1)))
            p2 = np.array(local_xy(nav.node(n2)))
            v1 = p0 - p1
            v2 = p2 - p0
            n1_norm, n2_norm = np.linalg.norm(v1), np.linalg.norm(v2)
            if n1_norm < 1e-9 or n2_norm < 1e-9:
                continue
            cos_sim = float(np.dot(v1, v2) / (n1_norm * n2_norm))
            if cos_sim < turn_cos_threshold:
                out.append(SignCandidate(floor_id=node.floor, x=x, y=y, reason="turn", node_id=nid))
    return out


def _backbone_graph(
    nav: NavGraph, building: Building, floor_id: str,
) -> tuple[dict[str, Node], dict[str, list[tuple[str, float]]]]:
    """The subgraph `_spacing_fill_candidates` walks: every true circulation
    node on `floor_id` (corridor/stub segments and circulation-continuation
    crossings), plus every exit/stair-landing node on it (a corridor's own
    end -- already signed, and a legitimate anchor to measure spacing
    from). Edges are the real nav-graph edges between two such nodes, so a
    room branching off the backbone (which has no node in this subgraph at
    all) can never attach to it."""
    nodes: dict[str, Node] = {}
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.floor != floor_id:
            continue
        if node.kind_detail in ("exit", "stair_landing") or _is_circulation_continuation_node(building, node):
            nodes[nid] = node
    edges: dict[str, list[tuple[str, float]]] = {nid: [] for nid in nodes}
    for u in nodes:
        for v in nav.g.neighbors(u):
            if v in nodes:
                edges[u].append((v, nav.link(u, v).length_m))
    return nodes, edges


def _dijkstra_from(sources: set[str], edges: dict[str, list[tuple[str, float]]]) -> dict[str, float]:
    dist: dict[str, float] = {s: 0.0 for s in sources}
    pq: list[tuple[float, str]] = [(0.0, s) for s in sources]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, float("inf")):
            continue
        for v, w in edges.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist


def _spacing_fill_candidates(
    nav: NavGraph, building: Building, floor_id: str, anchor_node_ids: set[str],
    *, max_spacing_m: float, min_spacing_m: float,
) -> list[SignCandidate]:
    """Extra `reason="spacing"` candidates so no backbone node on this
    floor is farther than `max_spacing_m` (graph distance) from a sign --
    the already-placed exit/stair/junction/turn signs are the initial
    anchors. Repeatedly finds the single farthest-from-any-sign node and
    anchors a new sign there (skipping it, without ever revisiting it, if
    it's already within `min_spacing_m` straight-line distance of an
    existing sign -- e.g. the far side of a ring corridor, graph-far but
    visually close), which strictly shrinks the worst distance each time
    and always terminates within `len(nodes)` iterations."""
    nodes, edges = _backbone_graph(nav, building, floor_id)
    anchors = {nid for nid in anchor_node_ids if nid in nodes}
    if not anchors or not nodes:
        return []

    def local_xy(node: Node) -> tuple[float, float]:
        return building.from_site_xy(*node.XY)

    sign_positions = [local_xy(nodes[nid]) for nid in anchors]
    out: list[SignCandidate] = []
    for _ in range(len(nodes)):
        dist = _dijkstra_from(anchors, edges)
        worst_nid, worst_d = None, max_spacing_m
        for nid in nodes:
            d = dist.get(nid, float("inf"))
            if d > worst_d:
                worst_d, worst_nid = d, nid
        if worst_nid is None:
            break
        x, y = local_xy(nodes[worst_nid])
        anchors.add(worst_nid)  # either a real new sign or a visually-covered node -- never revisited either way
        if any(math.hypot(x - px, y - py) < min_spacing_m for px, py in sign_positions):
            continue
        sign_positions.append((x, y))
        out.append(SignCandidate(floor_id=floor_id, x=x, y=y, reason="spacing", node_id=worst_nid))
    return out


def _cell_of(grid: FloorGrid, x: float, y: float) -> tuple[int, int]:
    return grid.transform.local_to_cell(x, y)


def _in_bounds(grid: FloorGrid, row: int, col: int) -> bool:
    rows, cols = grid.shape
    return 0 <= row < rows and 0 <= col < cols


def _nearby_exit_id(fk, row: int, col: int) -> int | None:
    """An exit cell within one cell of (row, col) -- not just the exact
    cell: a `trace_step_m` stride (deliberately bigger than the grid's own
    `h`, or not perfectly aligned to it) can otherwise step clean over a
    door's own 1-cell-thick rasterized band without ever landing on it."""
    rows, cols = fk.grid.shape
    r0, r1 = max(0, row - 1), min(rows, row + 2)
    c0, c1 = max(0, col - 1), min(cols, col + 2)
    block = fk.grid.exit_id[r0:r1, c0:c1]
    hit = block[block != -1]
    return int(hit[0]) if hit.size else None


def _nearby_stair_capture(fk, row: int, col: int, stair_capture_m: float) -> int | None:
    rows, cols = fk.grid.shape
    r0, r1 = max(0, row - 1), min(rows, row + 2)
    c0, c1 = max(0, col - 1), min(cols, col + 2)
    block_d = fk.stair_capture_dist_m[r0:r1, c0:c1]
    if not (block_d <= stair_capture_m).any():
        return None
    block_id = fk.stair_capture_id[r0:r1, c0:c1]
    idx = np.unravel_index(np.argmin(block_d), block_d.shape)
    gsidx = int(block_id[idx])
    return gsidx if gsidx >= 0 else None


_TRACE_HEADINGS = 16  # [U] candidate directions sampled per trace step -- a fixed geometric fan, not a tunable physics constant


def _trace_to_exit(world, dir_fields: dict, floor_id: str, x: float, y: float, *, trace_step_m: float,
                    trace_max_steps: int, stair_capture_m: float) -> str | None:
    """Walks from (floor_id, x, y) toward the nearest exit, floor to floor
    via stair capture: at each step, samples `_TRACE_HEADINGS` candidate
    directions and moves toward whichever walkable one has the lowest T --
    the same shape of lookahead `dms/scenario/evacuation.py`'s own walker
    uses, not a pure single-direction gradient step (which can clip a wall
    near a corridor boundary, or stall entirely at a concave corner where
    the one-sided-difference gradient briefly disagrees with the true best
    direction). Returns the reached exit's door id, or None if the trace
    never resolves (should not happen for a position the field actually
    reaches)."""
    fidx = world.floor_order.index(floor_id)
    cx, cy = x, y
    angles = [2.0 * math.pi * k / _TRACE_HEADINGS for k in range(_TRACE_HEADINGS)]

    for _ in range(trace_max_steps):
        fid = world.floor_order[fidx]
        fk = dir_fields[fid][2]
        row, col = _cell_of(fk.grid, cx, cy)
        if not _in_bounds(fk.grid, row, col) or not fk.move[row, col]:
            return None
        if fidx == 0:
            eidx = _nearby_exit_id(fk, row, col)
            if eidx is not None:
                return fk.grid.door_ids[eidx]
        else:
            gsidx = _nearby_stair_capture(fk, row, col, stair_capture_m)
            if gsidx is not None:
                fidx -= 1
                lower_fid = world.floor_order[fidx]
                lower_fk = dir_fields[lower_fid][2]
                landing = (lower_fk.grid.stair_id == gsidx) & lower_fk.move
                rows, cols = np.nonzero(landing)
                if len(rows) == 0:
                    return None
                cx, cy = lower_fk.grid.transform.cell_to_local(int(rows[0]), int(cols[0]))
                continue

        best_xy = None
        best_T = float(fk.truth_T[row, col])
        for a in angles:
            px, py = cx + math.cos(a) * trace_step_m, cy + math.sin(a) * trace_step_m
            pr, pc = _cell_of(fk.grid, px, py)
            if not _in_bounds(fk.grid, pr, pc) or not fk.move[pr, pc]:
                continue
            t = float(fk.truth_T[pr, pc])
            if t < best_T:
                best_T = t
                best_xy = (px, py)
        if best_xy is None:
            return None  # every direction is either blocked or worse than staying -- a genuine local minimum
        cx, cy = best_xy
    return None


def _process_candidate(
    cand: SignCandidate, world, dir_fields: dict, nav: NavGraph, building: Building, cfg, stair_capture_m: float,
) -> tuple[SignCandidate, float, str] | None:
    """Resolves one candidate position into (final candidate, heading_deg,
    points_to), or None if no usable direction/route exists there. Shared
    by the rule-based anchor candidates (exit/stair/junction/turn) and the
    spacing-fill candidates (reason="spacing") -- a spacing-fill point is
    just another circulation-node position on the same floor, resolved the
    identical way."""
    ex, ey, fk = dir_fields[cand.floor_id]
    cand_x, cand_y = cand.x, cand.y
    heading: float | None = None

    if cand.reason == "exit":
        # An exit's own cell sits exactly on the wall line and isn't
        # part of clearance_domain's walkable mask (the Eikonal field's
        # own domain restores it as a T=0 target, but FloorKinematics.move
        # doesn't -- see EvacuationWorld.floor_kinematics) -- nudge one
        # cell inward, off the wall, so the position is genuinely
        # walkable. T is also exactly 0 at the exit itself, so its
        # gradient there is unreliable regardless -- the door's own
        # geometry gives an unambiguous "which way is out" instead.
        node = nav.node(cand.node_id)
        floor = building.get_floor(cand.floor_id)
        door = floor.doors.get(node.ref_id)
        if door is None or door.space_a not in floor.spaces:
            return None
        heading = _door_heading_deg(door, floor.spaces[door.space_a].rect)
        inward = math.radians(heading + 180.0)
        nudge = fk.grid.transform.h * 1.5
        cand_x += math.cos(inward) * nudge
        cand_y += math.sin(inward) * nudge

    row, col = _cell_of(fk.grid, cand_x, cand_y)
    if not _in_bounds(fk.grid, row, col) or not fk.move[row, col]:
        return None

    if heading is None:
        dx, dy = float(ex[row, col]), float(ey[row, col])
        gradient_ok = math.isfinite(dx) and math.isfinite(dy) and not (dx == 0.0 and dy == 0.0)
        if not gradient_ok:
            return None  # no usable direction here (e.g. a stair landing's own near-flat T minimum)
        heading = math.degrees(math.atan2(dy, dx))

    points_to = _trace_to_exit(
        world, dir_fields, cand.floor_id, cand_x, cand_y,
        trace_step_m=cfg.trace_step_m, trace_max_steps=cfg.trace_max_steps, stair_capture_m=stair_capture_m,
    )
    if points_to is None:
        return None
    final = SignCandidate(floor_id=cand.floor_id, x=cand_x, y=cand_y, reason=cand.reason, node_id=cand.node_id)
    return final, heading, points_to


def generate_signs(site: Site, building: Building, agent_defaults=None) -> list[Sign]:
    """Every generated sign, in deterministic order (floor id, then x, y,
    rule-based signs before spacing-fill ones) -- callers decide whether/
    how to commit them (see `apply_generated_signs`)."""
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    from ..scenario.evacuation import build_world

    cfg = agent_defaults.signage
    radius = agent_defaults.body_classes["standard"].clearance_radius_m
    stair_capture_m = agent_defaults.kinematics.stair_capture_m

    world = build_world(site, building, "baseline", agent_defaults=agent_defaults)
    nav = build_navigation_graph(site, building.id, use_outdoor_network=False)

    dir_fields: dict[str, tuple] = {}
    for fid in world.floor_order:
        fk = world.floor_kinematics(radius)[fid]
        domain = fk.move
        T = np.ma.MaskedArray(fk.truth_T, mask=~domain)
        ex, ey = direction_field(T, domain, fk.grid.transform.h)
        dir_fields[fid] = (ex, ey, fk)

    candidates = _junction_and_turn_candidates(nav, building, turn_cos_threshold=cfg.turn_cos_threshold)

    signs: list[Sign] = []
    counters: dict[str, int] = {}
    anchors_by_floor: dict[str, set[str]] = {fid: set() for fid in world.floor_order}

    def place(cand: SignCandidate) -> None:
        result = _process_candidate(cand, world, dir_fields, nav, building, cfg, stair_capture_m)
        if result is None:
            return
        final, heading, points_to = result
        counters[final.floor_id] = counters.get(final.floor_id, 0) + 1
        sign_id = f"{final.floor_id}-sign-{final.reason}-{counters[final.floor_id]:02d}"
        signs.append(Sign(
            id=sign_id, floor=final.floor_id, x=final.x, y=final.y, points_to=points_to,
            heading_deg=heading, reason=final.reason,
            sign_type=_SIGN_TYPE_OF_REASON.get(final.reason, "directional"),
            facing_deg=(heading + 180.0) % 360.0,
            mount_height_m=cfg.mount_height_m, illuminated=cfg.illuminated,
        ))
        anchors_by_floor[final.floor_id].add(final.node_id)

    for cand in sorted(candidates, key=lambda c: (c.floor_id, c.reason, round(c.x, 3), round(c.y, 3))):
        place(cand)

    for floor_id in world.floor_order:
        spacing_candidates = _spacing_fill_candidates(
            nav, building, floor_id, anchors_by_floor[floor_id],
            max_spacing_m=cfg.sign_max_spacing_m, min_spacing_m=cfg.sign_min_spacing_m,
        )
        for cand in spacing_candidates:
            place(cand)

    return signs


def apply_generated_signs(site: Site, building: Building, agent_defaults=None) -> list[Sign]:
    """Commits every generated sign into `floor.signs`, keyed by its own
    generated id -- a YAML-declared sign already at that id (the loader
    runs first) is an override and is left untouched; any other YAML-
    declared sign is simply kept alongside the generated set."""
    signs = generate_signs(site, building, agent_defaults)
    for sign in signs:
        floor = building.get_floor(sign.floor)
        if sign.id in floor.signs:
            continue  # YAML already declared this exact id -- that's the override
        floor.signs[sign.id] = sign
    return signs
