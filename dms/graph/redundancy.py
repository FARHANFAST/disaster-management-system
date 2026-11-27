"""Connectivity and redundancy analysis (spec §7.2).

Every door, opening, stair and exit is taken UNAVAILABLE one at a time (a
"single failure"); the only rooms allowed to become unreachable are the
declared `designed_dead_ends` (spec: rooms whose *only* door is the failed
element). A handful of named double failures and an exit-pair sweep follow,
plus an edge-disjoint-paths check on the FF ring. Writes redundancy.md into
--out-dir (default out/reports/).

This module never mutates the live NavGraph — every trial operates on a
`G.copy()` with nodes/edges removed, and `route()` runs against a thin
NavGraph view over that copy (reusing the original's lookup dicts, since
removing nodes never invalidates id->id string mappings).
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..geometry.building import Building
from ..geometry.code_check import RoomLoad, room_loads
from .astar import Route, route
from .builder import NavGraph

REFERENCE_ROOMS = ("GF:R:G-16", "GF:R:G-11", "FF:R:F-07")  # two_storey.yaml-specific; see default_reference_rooms() for the N-floor-general runtime pick every function below actually defaults to
FF_RING_SPACE_IDS = ("ring-S", "ring-N", "ring-W", "ring-E")

# spec §8: the two named double-failure cases used to be two hardcoded
# `if "ST-C" in ... and "ST-E" in ...` string-match blocks inside
# run_named_double_failures. They're data now: (label, element_ids) pairs,
# overridable via the `specs` parameter -- main() below builds its specs
# from the building's own scenario presets instead of using this literal
# default, which exists only as the fallback for callers (tests included)
# that don't pass their own.
DEFAULT_DOUBLE_FAILURE_SPECS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("two_stairs_lost (ST-C + ST-E)", ("ST-C", "ST-E")),
    ("block_main_exit + block_ringS_mid", ("X-S", "LOBBY-op-ringS")),
)


def default_reference_rooms(nav: NavGraph, building: Building, *, n_per_floor: int = 1) -> tuple[str, ...]:
    """N-floor-general stand-in for the old hardcoded REFERENCE_ROOMS: the
    `n_per_floor` highest-capacity room(s) on each floor, in
    `building.floor_ids()` order, ties broken by node id for determinism.
    A 3-storey building gets its own 3 reference rooms without this module
    needing to know their names."""
    picks: list[str] = []
    for floor_id in building.floor_ids():
        candidates = [nid for nid in nav.room_nodes() if nav.node(nid).floor == floor_id]
        if not candidates:
            continue
        candidates.sort(key=lambda nid: (-nav.node(nid).capacity, nid))
        picks.extend(candidates[:n_per_floor])
    return tuple(picks)


def _navgraph_view(nav: NavGraph, g: nx.Graph) -> NavGraph:
    node_ids = [n for n in nav.node_ids if g.has_node(n)]
    return NavGraph(building_id=nav.building_id, g=g, node_ids=node_ids,
                     door_node_id=nav.door_node_id, opening_node_id=nav.opening_node_id,
                     stair_landing_nodes=nav.stair_landing_nodes)


def _disconnected_from_any_ap(g: nx.Graph, ap_nodes: set[str], candidates: set[str]) -> frozenset[str]:
    reachable: set[str] = set()
    for comp in nx.connected_components(g):
        if comp & ap_nodes:
            reachable |= comp
    return frozenset(candidates & g.nodes - reachable)


# --------------------------------------------------------------------- failure cases

@dataclass(slots=True)
class FailureCase:
    kind: str  # door | fire_door | exit | opening | stair | corridor_segment
    element_id: str
    label: str
    nodes_removed: tuple[str, ...] = ()
    edges_removed: tuple[tuple[str, str], ...] = ()


def enumerate_single_failures(nav: NavGraph) -> list[FailureCase]:
    cases: list[FailureCase] = []
    for did, node_id in nav.door_node_id.items():
        node = nav.node(node_id)
        kind = "exit" if node.kind_detail == "exit" else ("fire_door" if node.kind == "hazard_ctrl" else "door")
        cases.append(FailureCase(kind=kind, element_id=did, label=did, nodes_removed=(node_id,)))
    for oid, node_id in nav.opening_node_id.items():
        cases.append(FailureCase(kind="opening", element_id=oid, label=oid, nodes_removed=(node_id,)))
    for stair_id, landings in nav.stair_landing_nodes.items():
        if len(landings) == 2:
            u, v = landings.values()
            cases.append(FailureCase(kind="stair", element_id=stair_id, label=stair_id, edges_removed=((u, v),)))
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.kind == "platform" and node.kind_detail in ("corridor", "stub"):
            cases.append(FailureCase(kind="corridor_segment", element_id=nid, label=nid, nodes_removed=(nid,)))
    # A gate or a walkway segment is now a real,
    # individually-failable part of the outdoor route (dms/graph/builder.py:
    # build_outdoor_network_graph), not invisible to this sweep.
    for gate_id, node_id in nav.gate_node_id.items():
        cases.append(FailureCase(kind="gate", element_id=gate_id, label=gate_id, nodes_removed=(node_id,)))
    for walkway_id, vids in nav.walkway_segment_nodes.items():
        for i, (u, v) in enumerate(itertools.pairwise(vids)):
            label = f"{walkway_id}-seg{i:02d}"
            cases.append(FailureCase(kind="walkway_segment", element_id=label, label=label, edges_removed=((u, v),)))
    return cases


def apply_failure(nav: NavGraph, case: FailureCase) -> nx.Graph:
    g = nav.g.copy()
    for nid in case.nodes_removed:
        if g.has_node(nid):
            g.remove_node(nid)
    for u, v in case.edges_removed:
        if g.has_edge(u, v):
            g.remove_edge(u, v)
    return g


# --------------------------------------------------------------------- single-failure sweep

@dataclass(slots=True)
class SingleFailureResult:
    case: FailureCase
    disconnected_rooms: frozenset[tuple[str, str]]  # (floor_id, space_id) pairs, not node ids
    reference_routes: dict[str, Route | None]


def _room_space_ids(nav: NavGraph) -> dict[str, tuple[str, str]]:
    """node_id -> (floor_id, space_id), restricted to room-kind nodes.
    A bare space id is not assumed globally unique -- B1-L's
    corridor/stair ids genuinely repeat across floors (room ids happen not
    to, but this must not rely on that coincidence), so every disconnected-
    room result is tracked by floor+space throughout, never a bare id."""
    return {
        nid: (nav.node(nid).floor, nav.node(nid).space_id)
        for nid in nav.node_ids if nav.node(nid).kind_detail == "room"
    }


def _expand_declared_dead_ends(building: Building, declared: list[str]) -> set[tuple[str, str]]:
    """A `"FLOOR:SPACE"` entry maps to exactly that pair; a bare `"SPACE"`
    entry (today's two_storey/three_storey format) expands to
    `(floor_id, "SPACE")` for every floor that actually has a space with
    that id -- so an old building's bare list keeps meaning "this id, on
    whichever floor(s) it exists," never stripped, never assumed unique."""
    expanded: set[tuple[str, str]] = set()
    for entry in declared:
        if ":" in entry:
            floor_id, space_id = entry.split(":", 1)
            expanded.add((floor_id, space_id))
        else:
            for floor_id, floor in building.floors.items():
                if entry in floor.spaces:
                    expanded.add((floor_id, entry))
    return expanded


def _fmt_room(pair: tuple[str, str]) -> str:
    return f"{pair[0]}:{pair[1]}"


def _fmt_rooms(pairs) -> str:
    return ", ".join(_fmt_room(p) for p in sorted(pairs))


# --------------------------------------------------------------------- generic single-door-room rule
#
# A single-door room is
# acceptable if occupancy <= single_door_max_occupancy AND common_path <=
# common_path_max_m -- both from configs/agent_defaults.yaml's
# `redundancy:` section, never a by-name exception list. Replaces the old
# "is this disconnection in the declared designed_dead_ends list" gate in
# write_redundancy_report specifically (enumerate_single_failures/
# run_single_failure_sweep/_expand_declared_dead_ends themselves are
# untouched -- designed_dead_ends remains valid cross-reference data, just
# no longer the thing that decides pass/fail here).

def common_path_of_travel_m(nav: NavGraph, room_node: str, single_results: list[SingleFailureResult]) -> float:
    """The distance from `room_node` to the farthest point along its route
    to safety whose failure would disconnect it -- i.e. the length of the
    single-path segment before redundancy begins (the standard "common
    path of travel" definition), computed directly from the single-failure
    sweep's own results: a room's worst-case distance (via the INTACT
    graph's own edge lengths) to any element that, when failed, disconnects
    it. 0.0 if nothing in `single_results` disconnects this room (it's
    never on anyone's single path -- always redundant)."""
    worst = 0.0
    if not nav.g.has_node(room_node):
        return worst
    room_floor = nav.node(room_node).floor
    room_space = nav.node(room_node).space_id
    for r in single_results:
        if (room_floor, room_space) not in r.disconnected_rooms:
            continue
        element_nodes = list(r.case.nodes_removed) + [n for uv in r.case.edges_removed for n in uv]
        for nid in element_nodes:
            if not nav.g.has_node(nid):
                continue
            try:
                d = nx.shortest_path_length(nav.g, room_node, nid, weight=lambda u, v, dd: dd["obj"].length_m)
            except nx.NetworkXNoPath:
                continue
            worst = max(worst, d)
    return worst


@dataclass(slots=True, frozen=True)
class RoomRedundancyStatus:
    floor_id: str
    space_id: str
    use: str
    max_occupancy: int
    n_doors: int
    common_path_m: float
    acceptable: bool  # single door AND occupancy/common-path both within config limits


def classify_room_redundancy(
    building: Building, nav: NavGraph, single_results: list[SingleFailureResult],
    rooms: set[tuple[str, str]], *, single_door_max_occupancy: int, common_path_max_m: float,
) -> dict[tuple[str, str], RoomRedundancyStatus]:
    """One status per (floor, space) room in `rooms` -- generic, no id
    special-cased: a room with >1 door is never "acceptable" by this rule
    regardless of occupancy/common-path (it shouldn't have been
    disconnectable by a single failure at all; its disconnection is a real
    finding, not a single-door-room exception)."""
    loads_by_room: dict[tuple[str, str], RoomLoad] = {
        (rl.floor_id, rl.space_id): rl for rl in room_loads(building)
    }
    out: dict[tuple[str, str], RoomRedundancyStatus] = {}
    for fid, sid in rooms:
        rl = loads_by_room.get((fid, sid))
        use = rl.use if rl else ""
        max_occ = rl.max_load if rl else 0
        n_doors = rl.n_doors if rl else 0
        room_node = f"{fid}:R:{sid}"
        cp = common_path_of_travel_m(nav, room_node, single_results)
        acceptable = (n_doors == 1 and max_occ <= single_door_max_occupancy and cp <= common_path_max_m)
        out[(fid, sid)] = RoomRedundancyStatus(
            floor_id=fid, space_id=sid, use=use, max_occupancy=max_occ, n_doors=n_doors,
            common_path_m=cp, acceptable=acceptable,
        )
    return out


def run_single_failure_sweep(
    nav: NavGraph, building: Building, *, reference_rooms: tuple[str, ...] | None = None,
) -> list[SingleFailureResult]:
    reference_rooms = reference_rooms if reference_rooms is not None else default_reference_rooms(nav, building)
    ap_nodes = set(nav.assembly_point_nodes())
    room_space_of = _room_space_ids(nav)
    room_nodes = set(room_space_of)

    results = []
    for case in enumerate_single_failures(nav):
        g = apply_failure(nav, case)
        disconnected_nodes = _disconnected_from_any_ap(g, ap_nodes, room_nodes)
        disconnected_rooms = frozenset(room_space_of[n] for n in disconnected_nodes)

        view = _navgraph_view(nav, g)
        ref_routes = {}
        for rid in reference_rooms:
            ref_routes[rid] = route(view, rid) if g.has_node(rid) else None

        results.append(SingleFailureResult(case=case, disconnected_rooms=disconnected_rooms, reference_routes=ref_routes))
    return results


# --------------------------------------------------------------------- double failures

@dataclass(slots=True)
class DoubleFailureResult:
    label: str
    cases: tuple[FailureCase, ...]
    disconnected_rooms: frozenset[tuple[str, str]]  # (floor_id, space_id) pairs
    reference_routes: dict[str, Route | None]


def _combined_failure_graph(nav: NavGraph, cases: list[FailureCase]) -> nx.Graph:
    g = nav.g.copy()
    for case in cases:
        for nid in case.nodes_removed:
            if g.has_node(nid):
                g.remove_node(nid)
        for u, v in case.edges_removed:
            if g.has_edge(u, v):
                g.remove_edge(u, v)
    return g


def _run_combined(
    nav: NavGraph, label: str, cases: list[FailureCase], *, reference_rooms: tuple[str, ...],
) -> DoubleFailureResult:
    ap_nodes = set(nav.assembly_point_nodes())
    room_space_of = _room_space_ids(nav)
    room_nodes = set(room_space_of)

    g = _combined_failure_graph(nav, cases)
    disconnected_nodes = _disconnected_from_any_ap(g, ap_nodes, room_nodes)
    disconnected_rooms = frozenset(room_space_of[n] for n in disconnected_nodes)

    view = _navgraph_view(nav, g)
    ref_routes = {rid: (route(view, rid) if g.has_node(rid) else None) for rid in reference_rooms}
    return DoubleFailureResult(label=label, cases=tuple(cases), disconnected_rooms=disconnected_rooms, reference_routes=ref_routes)


def run_named_double_failures(
    nav: NavGraph, building: Building,
    specs: list[tuple[str, tuple[str, ...]]] | None = None, *,
    reference_rooms: tuple[str, ...] | None = None,
) -> list[DoubleFailureResult]:
    """`specs`: (label, element_ids) pairs -- which elements each named
    double failure removes, and the label it's reported under. Defaults to
    `DEFAULT_DOUBLE_FAILURE_SPECS` (this building's two historically-named
    cases); `main()` instead builds its specs from the building's own
    scenario presets, so the labels/ids are scenario data, not a hardcoded
    string match against known element ids. A spec whose element ids don't
    all exist as failable elements on this building (e.g. a 3-storey
    fixture with no "ST-C") is silently skipped, not an error."""
    specs = specs if specs is not None else DEFAULT_DOUBLE_FAILURE_SPECS
    reference_rooms = reference_rooms if reference_rooms is not None else default_reference_rooms(nav, building)
    all_cases = {c.element_id: c for c in enumerate_single_failures(nav)}

    results = []
    for label, element_ids in specs:
        cases = [all_cases[eid] for eid in element_ids if eid in all_cases]
        if len(cases) != len(element_ids):
            continue
        results.append(_run_combined(nav, label, cases, reference_rooms=reference_rooms))
    return results


def run_exit_pair_sweep(
    nav: NavGraph, building: Building, *, reference_rooms: tuple[str, ...] | None = None,
) -> list[DoubleFailureResult]:
    reference_rooms = reference_rooms if reference_rooms is not None else default_reference_rooms(nav, building)
    exit_cases = [c for c in enumerate_single_failures(nav) if c.kind == "exit"]
    results = []
    for i in range(len(exit_cases)):
        for j in range(i + 1, len(exit_cases)):
            a, b = exit_cases[i], exit_cases[j]
            results.append(_run_combined(nav, f"{a.element_id} + {b.element_id}", [a, b], reference_rooms=reference_rooms))
    return results


# --------------------------------------------------------------------- outdoor gate/walkway redundancy

def check_exit_ap_redundancy_after_gate_or_walkway_failure(nav: NavGraph) -> dict[str, int]:
    """For every gate and every walkway-segment single failure: the
    minimum, over all exits, of how many distinct assembly points that
    exit can still reach -- the requirement is that closing any one gate or
    any one walkway segment still leaves every exit reaching at least 2
    assembly points. Empty dict if the site declares no gates/walkways at all
    (a building without outdoor site data -- nothing to check, not a 0)."""
    cases = [c for c in enumerate_single_failures(nav) if c.kind in ("gate", "walkway_segment")]
    exit_nodes = [nid for nid in nav.node_ids if nav.node(nid).kind_detail == "exit"]
    ap_nodes = set(nav.assembly_point_nodes())
    results: dict[str, int] = {}
    for case in cases:
        g = apply_failure(nav, case)
        worst = min(
            (len(nx.node_connected_component(g, ex) & ap_nodes) if g.has_node(ex) else 0)
            for ex in exit_nodes
        ) if exit_nodes else 0
        results[case.label] = worst
    return results


# --------------------------------------------------------------------- edge-disjoint paths

def check_ff_ring_edge_disjoint_paths(nav: NavGraph, *, min_paths: int = 2) -> dict[str, int]:
    """spec §7.2: from every FF ring segment, at least `min_paths` edge-
    disjoint paths to an assembly point (networkx.edge_disjoint_paths), via
    a virtual sink tied to every AP node."""
    g = nav.g.copy()
    g.add_node("__SINK__")
    for ap in nav.assembly_point_nodes():
        g.add_edge(ap, "__SINK__")

    counts: dict[str, int] = {}
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.floor != "FF" or node.kind_detail != "corridor" or node.space_id not in FF_RING_SPACE_IDS:
            continue
        try:
            paths = list(nx.edge_disjoint_paths(g, nid, "__SINK__", cutoff=min_paths))
            counts[nid] = len(paths)
        except nx.NetworkXNoPath:
            counts[nid] = 0
    return counts


# --------------------------------------------------------------------- report

def _route_summary(before: Route | None, after: Route | None) -> str:
    if before is None:
        return "n/a"
    if after is None:
        return "**UNREACHABLE**"
    if before.cost <= 1e-9:
        return f"{after.cost:.1f}s"
    pct = (after.cost - before.cost) / before.cost * 100.0
    return f"{after.cost:.1f}s ({pct:+.0f}%)"


def write_redundancy_report(
    nav: NavGraph, building: Building, out_path: str | Path,
    single: list[SingleFailureResult], named_double: list[DoubleFailureResult],
    exit_pairs: list[DoubleFailureResult], ring_paths: dict[str, int], *,
    reference_rooms: tuple[str, ...] | None = None, agent_defaults=None,
) -> Path:
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    rd_cfg = agent_defaults.redundancy

    reference_rooms = reference_rooms if reference_rooms is not None else default_reference_rooms(nav, building)
    reference_labels = [nav.node(rid).space_id or rid for rid in reference_rooms]
    baseline_routes = {rid: route(nav, rid) for rid in reference_rooms}
    declared_dead_ends = _expand_declared_dead_ends(building, building.designed_dead_ends)
    all_affected = frozenset().union(*(r.disconnected_rooms for r in single)) if single else frozenset()
    status_by_room = classify_room_redundancy(
        building, nav, single, all_affected,
        single_door_max_occupancy=rd_cfg.single_door_max_occupancy, common_path_max_m=rd_cfg.common_path_max_m,
    )
    acceptable_rooms = {k for k, v in status_by_room.items() if v.acceptable}
    unacceptable_rooms = all_affected - acceptable_rooms

    lines = ["# Redundancy report", "", f"Building: **{building.id}**", ""]

    lines += ["## Single-failure sweep", ""]
    lines += [f"- {len(single)} elements tested (doors, fire doors, exits, openings, stairs, corridor segments, gates, walkway segments)."]
    lines += [f"- Rooms disconnected by *some* single failure: {_fmt_rooms(all_affected) or 'none'}"]
    acceptance_rule_line = (
        "- Acceptance rule (configs/agent_defaults.yaml `redundancy:`, generic -- no room id ever named here): "
        f"a disconnected room is acceptable iff it has exactly 1 door AND occupancy <= "
        f"{rd_cfg.single_door_max_occupancy} AND common path of travel <= {rd_cfg.common_path_max_m:.0f} m."
    )
    lines += [acceptance_rule_line]
    lines += [f"- Declared `designed_dead_ends` (reference only, not the acceptance rule): {_fmt_rooms(declared_dead_ends)}"]
    match = "YES" if not unacceptable_rooms else "NO -- see unexpected rows below"
    lines += [f"- Every disconnected room passes the rule: **{match}**", ""]

    if all_affected:
        lines += ["### Disconnected-room detail", "",
                  "| Room | Use | Max occupancy | Doors | Common path (m) | Acceptable |",
                  "|---|---|---|---|---|---|"]
        for fid, sid in sorted(all_affected):
            st = status_by_room[(fid, sid)]
            flag = "yes" if st.acceptable else "**NO**"
            lines.append(f"| {fid}:{sid} | {st.use} | {st.max_occupancy} | {st.n_doors} | {st.common_path_m:.1f} | {flag} |")
        lines.append("")

    if unacceptable_rooms:
        lines += ["### Unexpected disconnections (fail the generic single-door-room rule)", "",
                  "| Failed element | Kind | Rooms disconnected |", "|---|---|---|"]
        for r in single:
            extra = _fmt_rooms(r.disconnected_rooms & unacceptable_rooms)
            if extra:
                lines.append(f"| {r.case.label} | {r.case.kind} | {extra} |")
        lines.append("")

    header_cells = " | ".join(reference_labels)
    lines += ["### Reference-room cost impact (worst 10 by cost increase)", "",
              f"| Failed element | Kind | {header_cells} |",
              "|---|---|" + "---|" * len(reference_rooms)]
    scored = []
    for r in single:
        worst = 0.0
        for rid in reference_rooms:
            b, a = baseline_routes[rid], r.reference_routes[rid]
            if b and a:
                worst = max(worst, (a.cost - b.cost) / max(b.cost, 1e-9))
            elif b and not a:
                worst = max(worst, 1e9)
        scored.append((worst, r))
    scored.sort(key=lambda t: -t[0])
    for _, r in scored[:10]:
        cells = [_route_summary(baseline_routes[rid], r.reference_routes[rid]) for rid in reference_rooms]
        lines.append(f"| {r.case.label} | {r.case.kind} | " + " | ".join(cells) + " |")
    lines.append("")

    lines += ["## Double-failure scenarios", ""]
    for r in named_double:
        lines += [f"### {r.label}", "",
                  f"- Rooms disconnected: {_fmt_rooms(r.disconnected_rooms) or 'none'}"]
        for rid in reference_rooms:
            lines.append(f"- {rid}: {_route_summary(baseline_routes[rid], r.reference_routes[rid])}")
        lines.append("")

    n_bad_pairs = sum(1 for r in exit_pairs if r.disconnected_rooms)
    lines += ["### Every exit pair", "", f"- {len(exit_pairs)} pairs tested.",
              f"- Pairs that disconnect at least one room: {n_bad_pairs}", ""]
    disconnecting_pairs = [r for r in exit_pairs if r.disconnected_rooms]
    if disconnecting_pairs:
        lines += ["| Exit pair | Rooms disconnected |", "|---|---|"]
        for r in disconnecting_pairs:
            lines.append(f"| {r.label} | {_fmt_rooms(r.disconnected_rooms)} |")
        lines.append("")

    lines += ["## FF ring edge-disjoint paths to an assembly point", "",
              "| Ring segment node | Edge-disjoint paths found (cutoff=2) |", "|---|---|"]
    below_target = []
    for nid, n in sorted(ring_paths.items()):
        flag = "" if n >= 2 else "  <-- BELOW TARGET"
        lines.append(f"| {nid} | {n}{flag} |")
        if n < 2:
            below_target.append(nid)
    lines.append("")
    if below_target:
        note = (
            "Segments below target are not necessarily a redundancy defect: a ring segment whose "
            "*only* other neighbour is a single-door dead-end room has no second useful path by "
            "construction (a detour into that room just leads back out the same door). Verify each "
            "one before treating it as a finding:"
        )
        lines += [note, ""]
        for nid in below_target:
            lines.append(f"- `{nid}`")
        lines.append("")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="building YAML (default: configs/buildings/two_storey.yaml)")
    parser.add_argument("--out-dir", type=Path, default=None, help="output directory (default: out/reports/)")
    parser.add_argument(
        "--with-images", action="store_true",
        help="also render the named double-failure scenarios' comparison PNGs (dms.viz.plot_routes)",
    )
    return parser.parse_args()


def _scenario_element_ids(site, scenario_name: str) -> list[str]:
    preset = site.scenarios.get(scenario_name, {})
    return [ev["id"] for ev in preset.get("events", []) if "id" in ev]


def _double_failure_specs_from_scenarios(site) -> list[tuple[str, tuple[str, ...]]]:
    """Build named double-failure specs from the building's own scenario
    presets (spec §8) instead of hardcoding which elements make a
    "named double failure" -- a scenario this building doesn't define is
    silently dropped rather than crashing, so a building without a
    "two_stairs_lost" preset (e.g. the 3-storey fixture) just gets fewer
    named double-failure rows in the report, not a traceback."""
    specs: list[tuple[str, tuple[str, ...]]] = []
    ids = _scenario_element_ids(site, "two_stairs_lost")
    if ids:
        specs.append(("two_stairs_lost", tuple(ids)))
    combo_names = ("block_main_exit", "block_ringS_mid")
    combo_ids = [eid for name in combo_names for eid in _scenario_element_ids(site, name)]
    if len(combo_ids) == len(combo_names):
        specs.append((" + ".join(combo_names), tuple(combo_ids)))
    return specs


def main() -> None:
    from ..geometry.loader import load_site
    from .builder import build_navigation_graph

    args = _parse_args()
    config = args.config or Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
    out_dir = args.out_dir or Path(__file__).resolve().parents[2] / "out" / "reports"

    site = load_site(config)
    building = site.get_building(site.building_ids()[0])
    nav = build_navigation_graph(site, building.id)

    specs = _double_failure_specs_from_scenarios(site) or list(DEFAULT_DOUBLE_FAILURE_SPECS)

    single = run_single_failure_sweep(nav, building)
    named = run_named_double_failures(nav, building, specs)
    pairs = run_exit_pair_sweep(nav, building)
    ring = check_ff_ring_edge_disjoint_paths(nav)

    out = write_redundancy_report(nav, building, out_dir / "redundancy.md", single, named, pairs, ring)
    print(f"wrote {out}")

    if args.with_images:
        from ..viz import plot_routes

        plot_routes.main()


if __name__ == "__main__":
    main()
