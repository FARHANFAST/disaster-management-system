"""Connectivity and redundancy analysis (spec §7.2, PROMPT_01 deliverable 6).

Every door, opening, stair and exit is taken UNAVAILABLE one at a time (a
"single failure"); the only rooms allowed to become unreachable are the
declared `designed_dead_ends` (spec: rooms whose *only* door is the failed
element). A handful of named double failures and an exit-pair sweep follow,
plus an edge-disjoint-paths check on the FF ring. Writes out/redundancy.md.

This module never mutates the live NavGraph — every trial operates on a
`G.copy()` with nodes/edges removed, and `route()` runs against a thin
NavGraph view over that copy (reusing the original's lookup dicts, since
removing nodes never invalidates id->id string mappings).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from ..geometry.building import Building
from .astar import Route, route
from .builder import NavGraph

REFERENCE_ROOMS = ("GF:R:G-16", "GF:R:G-11", "FF:R:F-07")
FF_RING_SPACE_IDS = ("ring-S", "ring-N", "ring-W", "ring-E")


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
    disconnected_rooms: frozenset[str]  # space ids (e.g. "G-08"), not node ids
    reference_routes: dict[str, Route | None]


def _room_space_ids(nav: NavGraph) -> dict[str, str]:
    """node_id -> space_id, restricted to room-kind nodes."""
    return {nid: nav.node(nid).space_id for nid in nav.node_ids if nav.node(nid).kind_detail == "room"}


def run_single_failure_sweep(nav: NavGraph, building: Building) -> list[SingleFailureResult]:
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
        for rid in REFERENCE_ROOMS:
            ref_routes[rid] = route(view, rid) if g.has_node(rid) else None

        results.append(SingleFailureResult(case=case, disconnected_rooms=disconnected_rooms, reference_routes=ref_routes))
    return results


# --------------------------------------------------------------------- double failures

@dataclass(slots=True)
class DoubleFailureResult:
    label: str
    cases: tuple[FailureCase, ...]
    disconnected_rooms: frozenset[str]
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


def _run_combined(nav: NavGraph, label: str, cases: list[FailureCase]) -> DoubleFailureResult:
    ap_nodes = set(nav.assembly_point_nodes())
    room_space_of = _room_space_ids(nav)
    room_nodes = set(room_space_of)

    g = _combined_failure_graph(nav, cases)
    disconnected_nodes = _disconnected_from_any_ap(g, ap_nodes, room_nodes)
    disconnected_rooms = frozenset(room_space_of[n] for n in disconnected_nodes)

    view = _navgraph_view(nav, g)
    ref_routes = {rid: (route(view, rid) if g.has_node(rid) else None) for rid in REFERENCE_ROOMS}
    return DoubleFailureResult(label=label, cases=tuple(cases), disconnected_rooms=disconnected_rooms, reference_routes=ref_routes)


def run_named_double_failures(nav: NavGraph, building: Building) -> list[DoubleFailureResult]:
    stair_case = {c.element_id: c for c in enumerate_single_failures(nav) if c.kind == "stair"}
    door_case = {c.element_id: c for c in enumerate_single_failures(nav) if c.kind in ("door", "exit", "fire_door")}
    opening_case = {c.element_id: c for c in enumerate_single_failures(nav) if c.kind == "opening"}

    results = []
    if "ST-C" in stair_case and "ST-E" in stair_case:
        results.append(_run_combined(nav, "two_stairs_lost (ST-C + ST-E)", [stair_case["ST-C"], stair_case["ST-E"]]))
    if "X-S" in door_case and "LOBBY-op-ringS" in opening_case:
        results.append(_run_combined(nav, "block_main_exit + block_ringS_mid",
                                      [door_case["X-S"], opening_case["LOBBY-op-ringS"]]))
    return results


def run_exit_pair_sweep(nav: NavGraph, building: Building) -> list[DoubleFailureResult]:
    exit_cases = [c for c in enumerate_single_failures(nav) if c.kind == "exit"]
    results = []
    for i in range(len(exit_cases)):
        for j in range(i + 1, len(exit_cases)):
            a, b = exit_cases[i], exit_cases[j]
            results.append(_run_combined(nav, f"{a.element_id} + {b.element_id}", [a, b]))
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
    exit_pairs: list[DoubleFailureResult], ring_paths: dict[str, int],
) -> Path:
    baseline_routes = {rid: route(nav, rid) for rid in REFERENCE_ROOMS}
    declared_dead_ends = set(building.designed_dead_ends)
    all_affected = frozenset().union(*(r.disconnected_rooms for r in single)) if single else frozenset()

    lines = ["# Redundancy report", "", f"Building: **{building.id}**", ""]

    lines += ["## Single-failure sweep", ""]
    lines += [f"- {len(single)} elements tested (doors, fire doors, exits, openings, stairs, corridor segments)."]
    lines += [f"- Rooms disconnected by *some* single failure: {', '.join(sorted(all_affected)) or 'none'}"]
    lines += [f"- Declared `designed_dead_ends`: {', '.join(sorted(declared_dead_ends))}"]
    match = "YES" if all_affected == declared_dead_ends else "NO -- see unexpected rows below"
    lines += [f"- Matches declared set exactly: **{match}**", ""]

    unexpected = [r for r in single if r.disconnected_rooms - declared_dead_ends]
    if unexpected:
        lines += ["### Unexpected disconnections (not in designed_dead_ends)", "",
                  "| Failed element | Kind | Rooms disconnected |", "|---|---|---|"]
        for r in unexpected:
            extra = ", ".join(sorted(r.disconnected_rooms - declared_dead_ends))
            lines.append(f"| {r.case.label} | {r.case.kind} | {extra} |")
        lines.append("")

    lines += ["### Reference-room cost impact (worst 10 by cost increase)", "",
              "| Failed element | Kind | G-16 | G-11 | F-07 |", "|---|---|---|---|---|"]
    scored = []
    for r in single:
        worst = 0.0
        for rid in REFERENCE_ROOMS:
            b, a = baseline_routes[rid], r.reference_routes[rid]
            if b and a:
                worst = max(worst, (a.cost - b.cost) / max(b.cost, 1e-9))
            elif b and not a:
                worst = max(worst, 1e9)
        scored.append((worst, r))
    scored.sort(key=lambda t: -t[0])
    for _, r in scored[:10]:
        cells = [_route_summary(baseline_routes[rid], r.reference_routes[rid]) for rid in REFERENCE_ROOMS]
        lines.append(f"| {r.case.label} | {r.case.kind} | {cells[0]} | {cells[1]} | {cells[2]} |")
    lines.append("")

    lines += ["## Double-failure scenarios", ""]
    for r in named_double:
        lines += [f"### {r.label}", "",
                  f"- Rooms disconnected: {', '.join(sorted(r.disconnected_rooms)) or 'none'}"]
        for rid in REFERENCE_ROOMS:
            lines.append(f"- {rid}: {_route_summary(baseline_routes[rid], r.reference_routes[rid])}")
        lines.append("")

    n_bad_pairs = sum(1 for r in exit_pairs if r.disconnected_rooms)
    lines += ["### Every exit pair", "", f"- {len(exit_pairs)} pairs tested.",
              f"- Pairs that disconnect at least one room: {n_bad_pairs}", ""]
    disconnecting_pairs = [r for r in exit_pairs if r.disconnected_rooms]
    if disconnecting_pairs:
        lines += ["| Exit pair | Rooms disconnected |", "|---|---|"]
        for r in disconnecting_pairs:
            lines.append(f"| {r.label} | {', '.join(sorted(r.disconnected_rooms))} |")
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
