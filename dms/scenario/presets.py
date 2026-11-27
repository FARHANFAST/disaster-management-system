"""Apply scenario presets (spec §3.8, §6.5) to a built NavGraph.

Preset event lists live in the YAML (`site.scenarios`) as pure data — this
module is the interpreter. For Milestone 3 (routing), only the events that
affect link *state* are wired up: SetDoorState, SetExitState, SetLinkState,
and SetStairState (which also resolves a stair id to its vertical link --
spec §2: a stricter sibling of SetLinkState that, on the grid/overlay
side, also makes the stair's footprint physically unwalkable on every floor
it serves; see dms/grid/layers.py: DynamicOverlay.blocked_stairs). Layer
toggles (clutter/parking) and hazard activation don't have a cost-model
hook yet (no gas/agents), so they're reported as unhandled rather than
silently accepted or raising.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.building import Building
from ..geometry.site import Site
from ..graph.builder import NavGraph
from ..grid.layers import DynamicOverlay

_LINK_STATE_EVENTS = ("SetDoorState", "SetExitState", "SetLinkState", "SetStairState")


@dataclass(slots=True)
class ScenarioResult:
    name: str
    applied: list[dict] = field(default_factory=list)
    unhandled: list[dict] = field(default_factory=list)


def door_states_from_events(preset: dict) -> dict[str, str]:
    """SetDoorState/SetExitState events -> {id: state}, for building a grid
    overlay's `door_states` generically from a scenario's own event list
    (not a hardcoded per-scenario branch)."""
    return {
        ev["id"]: ev["state"] for ev in preset.get("events", [])
        if ev.get("type") in ("SetDoorState", "SetExitState") and "id" in ev and "state" in ev
    }


def active_groups_from_events(preset: dict) -> dict[str, bool]:
    """ToggleLayer events -> {group: on}, for a grid overlay's
    `active_groups` -- generic over whatever group names a scenario
    actually toggles (clutter/maintenance/parking, or any future one)."""
    return {
        ev["group"]: bool(ev["on"]) for ev in preset.get("events", [])
        if ev.get("type") == "ToggleLayer" and "group" in ev and "on" in ev
    }


def blocked_stairs_from_events(preset: dict, building) -> frozenset[str]:
    """Stair ids a scenario's SetStairState events take UNAVAILABLE --
    for a grid overlay's `blocked_stairs` (footprint blocking).
    Deliberately narrower than dms/fields/eikonal.py's
    `stairs_blocked_by_scenario`, which also treats the legacy SetLinkState
    the same way, but only for the floor-coupling-exclusion purpose:
    SetLinkState alone never blocks a stair's footprint."""
    return frozenset(
        ev["id"] for ev in preset.get("events", [])
        if ev.get("type") == "SetStairState" and ev.get("state") == "UNAVAILABLE" and ev.get("id") in building.stairs
    )


def signage_overrides_from_scenario(preset: dict, building: Building, *, seed: int | None = None) -> tuple[frozenset[str], frozenset[str]]:
    """(removed_sign_ids, obscured_sign_ids) for this scenario's optional
    `signage:` block -- robustness-testing data: which
    generated signs this scenario treats as missing or hard to read.
    Explicit `remove`/`obscure` id lists are always honoured;
    `random_remove_fraction`/`random_obscure_fraction` additionally sample
    from whichever signs neither list already names, seeded so the same
    seed always removes/obscures the same signs. Not consumed by the
    walker's own routing (it doesn't perceive signs at all, see
    dms/geometry/signage_gen.py's own docstring) -- this is for rendering
    and for a later perception-aware agent to read."""
    sg = preset.get("signage")
    if not sg:
        return frozenset(), frozenset()

    all_ids = sorted(sgid for floor in building.floors.values() for sgid in floor.signs)
    removed = set(sg.get("remove", []))
    obscured = set(sg.get("obscure", []))

    rng = np.random.default_rng(seed)
    remaining = [i for i in all_ids if i not in removed and i not in obscured]
    frac_r = float(sg.get("random_remove_fraction", 0.0))
    if frac_r > 0.0 and remaining:
        n = min(round(len(remaining) * frac_r), len(remaining))
        picked = rng.choice(remaining, size=n, replace=False)
        removed.update(picked.tolist())
        remaining = [i for i in remaining if i not in removed]
    frac_o = float(sg.get("random_obscure_fraction", 0.0))
    if frac_o > 0.0 and remaining:
        n = min(round(len(remaining) * frac_o), len(remaining))
        picked = rng.choice(remaining, size=n, replace=False)
        obscured.update(picked.tolist())

    return frozenset(removed), frozenset(obscured)


def overlays_from_scenario(preset: dict, building: Building) -> tuple[DynamicOverlay, DynamicOverlay]:
    """(belief_overlay, truth_overlay) for one scenario preset -- this
    module's belief/truth knowledge split: truth explicitly marks every
    BLOCKED_UNKNOWN exit as UNAVAILABLE (informed: known-blocked); belief
    drops those entries entirely, falling back to the door's own declared
    default_state, matching the established ghost-walker convention
    (dms/viz/walker_trace.py's own belief_overlay construction). Both share
    the same door_states/active_groups/blocked_stairs for every other kind
    of event (SetDoorState/SetExitState/ToggleLayer/SetStairState) -- there
    is no knowledge split for those, only for BLOCKED_UNKNOWN exits."""
    door_states = door_states_from_events(preset)
    active_groups = active_groups_from_events(preset)
    blocked_stairs = blocked_stairs_from_events(preset, building)

    belief_states = {k: v for k, v in door_states.items() if v != "BLOCKED_UNKNOWN"}
    truth_states = {k: ("UNAVAILABLE" if v == "BLOCKED_UNKNOWN" else v) for k, v in door_states.items()}

    belief = DynamicOverlay(door_states=belief_states, active_groups=active_groups, blocked_stairs=blocked_stairs)
    truth = DynamicOverlay(door_states=truth_states, active_groups=active_groups, blocked_stairs=blocked_stairs)
    return belief, truth


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
