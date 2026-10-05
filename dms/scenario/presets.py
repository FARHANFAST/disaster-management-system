"""Apply scenario presets (spec §3.8, §6.5) to a built NavGraph.

Preset event lists live in the YAML (`site.scenarios`) as pure data — this
module is the interpreter. For Milestone 3 (routing), only the events that
affect link *state* are wired up: SetDoorState, SetExitState and
SetLinkState (which also resolves a stair id to its vertical link). Layer
toggles (clutter/parking) and hazard activation don't have a cost-model
hook yet (no gas/agents), so they're reported as unhandled rather than
silently accepted or raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..geometry.site import Site
from ..graph.builder import NavGraph

_LINK_STATE_EVENTS = ("SetDoorState", "SetExitState", "SetLinkState")


@dataclass(slots=True)
class ScenarioResult:
    name: str
    applied: list[dict] = field(default_factory=list)
    unhandled: list[dict] = field(default_factory=list)


def reset_states(graph: NavGraph) -> None:
    """Reset every link back to AVAILABLE, undoing any previously applied scenario."""
    for _u, _v, data in graph.g.edges(data=True):
        data["obj"].state = "AVAILABLE"


def _set_edges_through(graph: NavGraph, node_id: str, state: str) -> bool:
    changed = False
    for nb in list(graph.g.neighbors(node_id)):
        graph.link(node_id, nb).state = state
        changed = True
    return changed


def apply_event(graph: NavGraph, event: dict) -> bool:
    """Apply one scenario event in place. Returns True if it changed something."""
    etype = event.get("type")
    if etype not in _LINK_STATE_EVENTS:
        return False

    state = event.get("state")
    target_id = event.get("id")
    if state is None or target_id is None:
        return False

    if target_id in graph.stair_landing_nodes:
        landings = graph.stair_landing_nodes[target_id]
        if len(landings) != 2:
            return False
        u, v = landings.values()
        if not graph.g.has_edge(u, v):
            return False
        graph.link(u, v).state = state
        return True

    node_id = graph.door_node_id.get(target_id) or graph.opening_node_id.get(target_id)
    if node_id is None:
        return False
    return _set_edges_through(graph, node_id, state)


def apply_scenario(graph: NavGraph, site: Site, name: str, *, reset_first: bool = True) -> ScenarioResult:
    """Apply a named preset from `site.scenarios` onto `graph`, in place."""
    if reset_first:
        reset_states(graph)
    preset = site.scenarios[name]
    result = ScenarioResult(name=name)
    for event in preset.get("events", []):
        if apply_event(graph, event):
            result.applied.append(event)
        else:
            result.unhandled.append(event)
    return result
