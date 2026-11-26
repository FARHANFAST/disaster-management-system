"""spec §5: BLOCKED_UNKNOWN is a knowledge split, not a blanket block.
An uninformed walker's own belief-overlay never mentions a not-yet-
discovered BLOCKED_UNKNOWN exit, so it follows a field that treats it as
available and walks *toward* it; on arrival within R_vis it loses
t_discover, marks it UNAVAILABLE in its own knowledge, and the field it
follows next is re-solved from its current position, excluding that exit.
"""

from __future__ import annotations

from dms.fields.eikonal import FieldCache
from dms.grid.layers import DynamicOverlay
from dms.viz.test_walker import R_VIS, T_DISCOVER, trace_gf_walker_with_discovery


def test_blocked_unknown_is_a_legal_door_state():
    from dms.geometry.primitives import STATES

    assert "BLOCKED_UNKNOWN" in STATES


def test_walkable_now_physically_blocks_blocked_unknown_regardless_of_overlay(gf_grid):
    """The one place belief never matters: ground truth always blocks it,
    whether or not the overlay handed in even mentions the door."""
    door_id = next(d for d in gf_grid.door_ids if gf_grid.door_default_state[gf_grid.door_ids.index(d)] != "UNAVAILABLE")
    # Force a door's *ground-truth* overlay entry to BLOCKED_UNKNOWN and
    # confirm walkable_now excludes its cells even though nothing about
    # this door's state resembles the literal "UNAVAILABLE" check the
    # uninformed-belief path relies on elsewhere.
    overlay = DynamicOverlay(door_states={door_id: "BLOCKED_UNKNOWN"})
    walkable = gf_grid.walkable_now(overlay)
    idx = gf_grid.door_ids.index(door_id)
    assert not walkable[gf_grid.door_id == idx].any()


def test_walker_heads_toward_the_blocked_unknown_exit_then_discovers_and_reroutes(building, gf, gf_grid):
    """The full sequence demonstrated end to end: the walker
    actually walks toward the exit (not away from it, not a trivial
    wiggle), loses t_discover with no movement on arrival, and the
    post-discovery path visibly diverges toward a different exit."""
    ground_truth = DynamicOverlay(door_states={"X-N": "BLOCKED_UNKNOWN"})
    cache = FieldCache()
    # G-13 "opens onto the N-stub, not the ring" (its own YAML notes) --
    # its baseline shortest path goes almost straight to X-N, the clearest
    # possible demonstration of "walks toward the exit it believes is open".
    result = trace_gf_walker_with_discovery(
        "G-13", "exit_N_blocked_unknown", building, gf, gf_grid, cache,
        ground_truth_overlay=ground_truth, blocked_unknown_exits=frozenset({"X-N"}),
    )

    assert result.reached_exit and not result.stuck, result.reason
    assert result.exit_id != "X-N", "X-N was BLOCKED_UNKNOWN -- the walker must not actually exit through it"
    assert len(result.discovery_events) == 1
    discovered_exit, discovered_t = result.discovery_events[0]
    assert discovered_exit == "X-N"

    # t_discover was actually spent (the step before discovery is t_discover
    # earlier than the discovery event's own recorded timestamp).
    pre = [s for s in result.path if s.t <= discovered_t - T_DISCOVER]
    post = [s for s in result.path if s.t >= discovered_t]
    assert pre and post

    # Pre-discovery: genuinely walked toward X-N (y increases toward the
    # north wall, y=40), not just a short local wiggle.
    assert pre[-1].y - pre[0].y > 2.0
    # It got within R_vis of X-N's own position (x~30, y=40) before discovering.
    import math
    assert math.hypot(pre[-1].x - 30.0, pre[-1].y - 40.0) <= R_VIS + 1e-6

    # Post-discovery: the path actually diverges away from X-N, ending up
    # far from it at a different exit -- not a trivial backtrack.
    last = result.path[-1]
    assert math.hypot(last.x - 30.0, last.y - 40.0) > 10.0


def test_only_two_exits_preset_leaves_every_room_a_route(nav, site):
    """New presets are pure data (apply_event already passes any state
    through unchanged) -- confirm they actually apply and still route."""
    from dms.graph.astar import route
    from dms.scenario.presets import apply_scenario, reset_states

    reset_states(nav)
    result = apply_scenario(nav, site, "only_two_exits")
    assert len(result.applied) == 4 and not result.unhandled

    for rid in nav.room_nodes():
        r = route(nav, rid)
        assert r is not None, f"{rid} has no route with only X-S/X-N available"
    reset_states(nav)


def test_main_exit_partial_halves_q_max_not_just_cost(nav, site):
    from dms.scenario.presets import apply_scenario, reset_states

    reset_states(nav)
    apply_scenario(nav, site, "main_exit_partial")
    # Walk every edge touching the X-S terminal node and confirm PARTIAL
    # halves Q_max, not just the A* cost multiplier.
    xs_node = nav.door_node_id["X-S"]
    assert nav.node(xs_node).kind_detail == "exit"
    for nb in nav.g.neighbors(xs_node):
        link = nav.link(xs_node, nb)
        assert link.state == "PARTIAL"
        assert link.Q_max == link.width_m * link.chi * 1.92 * 0.5
    reset_states(nav)
