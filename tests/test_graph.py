"""Milestone 3: L2 graph construction, (N5) edge cost, A* routing and
scenario presets."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from dms.graph.astar import V_MAX, route
from dms.graph.builder import Link
from dms.graph.cost import CostParams, link_cost
from dms.scenario.presets import apply_scenario, reset_states

# --------------------------------------------------------------------- builder structure


def test_graph_is_a_single_connected_component(nav):
    assert nx.is_connected(nav.g)


def test_every_room_and_stair_landing_has_degree_at_least_one(nav):
    for nid in nav.node_ids:
        assert nav.g.degree(nid) >= 1, f"{nid} is isolated"


def test_node_kind_vocabulary_matches_spec(nav):
    seen = {nav.node(n).kind for n in nav.node_ids}
    assert seen <= {"platform", "transition", "vertical", "hazard_ctrl", "terminal"}
    detail_by_kind = {}
    for n in nav.node_ids:
        node = nav.node(n)
        detail_by_kind.setdefault(node.kind, set()).add(node.kind_detail)
    assert detail_by_kind["vertical"] == {"stair_landing"}
    assert detail_by_kind["hazard_ctrl"] == {"fire_door"}
    assert detail_by_kind["terminal"] <= {"exit", "assembly_point"}


def test_room_count_matches_floor_geometry(nav, gf, ff):
    expected = sum(1 for sp in gf.spaces.values() if sp.kind == "room")
    expected += sum(1 for sp in ff.spaces.values() if sp.kind == "room")
    assert len(nav.room_nodes()) == expected


def test_stair_landings_are_vertically_linked(nav, building):
    for stair in building.stairs.values():
        landings = nav.stair_landing_nodes[stair.id]
        gf_id, ff_id = building.floor_ids()
        u, v = landings[gf_id], landings[ff_id]
        assert nav.g.has_edge(u, v)
        link = nav.link(u, v)
        assert link.length_m == pytest.approx(stair.walking_length_m)
        assert link.chi == pytest.approx(stair.chi)
        assert link.slope_deg == pytest.approx(stair.slope_deg)


def test_exits_connect_to_every_assembly_point(nav, gf):
    aps = set(nav.assembly_point_nodes())
    for door in gf.exit_doors:
        exit_node = nav.door_node_id[door.id]
        reachable_aps = {nb for nb in nav.g.neighbors(exit_node) if nb in aps}
        assert reachable_aps == aps


def test_corridor_segments_respect_length_bounds(nav):
    from dms.graph.builder import MAX_SEGMENT_M

    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.kind == "platform" and node.kind_detail in ("corridor", "stub"):
            seg_len = node.area_m2 / node.width_m
            # the merge rule only guarantees >= min_len when a merge partner
            # existed; a lone short stub (nothing to merge into) is allowed
            # to stay short, but nothing may exceed the split threshold.
            assert seg_len <= MAX_SEGMENT_M + 1e-6


def test_q_max_scales_with_width_and_chi():
    level = Link(u="a", v="b", length_m=5.0, width_m=3.0, chi=1.0)
    stair = Link(u="a", v="b", length_m=5.0, width_m=1.5, chi=0.70)
    assert level.Q_max == pytest.approx(1.92 * 3.0 * 1.0)
    assert stair.Q_max < level.Q_max


# --------------------------------------------------------------------- (N5) cost


def test_unavailable_link_has_infinite_cost():
    link = Link(u="a", v="b", length_m=10.0, width_m=1.5, state="UNAVAILABLE")
    assert math.isinf(link_cost(link))


def test_shuttered_link_has_infinite_cost_even_if_available():
    link = Link(u="a", v="b", length_m=10.0, width_m=1.5, state="AVAILABLE", shutter_closed=True)
    assert math.isinf(link_cost(link))


def test_partial_link_costs_omega_p_times_available():
    available = Link(u="a", v="b", length_m=10.0, width_m=1.5, state="AVAILABLE")
    partial = Link(u="a", v="b", length_m=10.0, width_m=1.5, state="PARTIAL")
    params = CostParams()
    assert link_cost(partial, params=params) == pytest.approx(link_cost(available, params=params) * params.omega_p)


def test_cost_increases_monotonically_with_rho():
    base = Link(u="a", v="b", length_m=10.0, width_m=1.5, rho=0.0)
    crowded = Link(u="a", v="b", length_m=10.0, width_m=1.5, rho=5.0)
    jammed = Link(u="a", v="b", length_m=10.0, width_m=1.5, rho=10.0)
    assert link_cost(base) < link_cost(crowded) < link_cost(jammed)


def test_cost_increases_monotonically_with_gas_concentration():
    clean = Link(u="a", v="b", length_m=10.0, width_m=1.5, C=0.0)
    alarm = Link(u="a", v="b", length_m=10.0, width_m=1.5, C=30.0)
    assert link_cost(clean) < link_cost(alarm)


def test_cost_increases_as_hazard_gets_closer():
    params = CostParams()
    far = Link(u="a", v="b", length_m=10.0, width_m=1.5, d_haz=params.R_H - 10.0)  # inside R_H, barely
    near = Link(u="a", v="b", length_m=10.0, width_m=1.5, d_haz=50.0)
    untouched = Link(u="a", v="b", length_m=10.0, width_m=1.5, d_haz=math.inf)
    assert link_cost(untouched) < link_cost(far, params=params) < link_cost(near, params=params)


def test_hazard_term_clips_at_zero_beyond_r_h():
    params = CostParams()
    just_outside = Link(u="a", v="b", length_m=10.0, width_m=1.5, d_haz=params.R_H + 1.0)
    far_outside = Link(u="a", v="b", length_m=10.0, width_m=1.5, d_haz=params.R_H * 10)
    assert link_cost(just_outside, params=params) == pytest.approx(link_cost(far_outside, params=params))


def test_stair_chi_increases_cost_relative_to_level_link_of_same_length():
    level = Link(u="a", v="b", length_m=8.9, width_m=1.5, chi=1.0)
    stair = Link(u="a", v="b", length_m=8.9, width_m=1.5, chi=0.70)
    assert link_cost(stair) > link_cost(level)


def test_zero_chi_is_treated_as_impassable():
    link = Link(u="a", v="b", length_m=10.0, width_m=1.5, chi=0.0)
    assert math.isinf(link_cost(link))


# --------------------------------------------------------------------- A*


def test_baseline_every_room_and_stair_landing_reaches_an_assembly_point(nav):
    reset_states(nav)
    unreachable = []
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.kind in ("platform", "vertical") and route(nav, nid) is None:
            unreachable.append(nid)
    assert unreachable == []


def test_route_to_self_is_free(nav):
    reset_states(nav)
    ap = nav.assembly_point_nodes()[0]
    r = route(nav, ap)
    assert r.cost == 0.0
    assert r.nodes == [ap]


def test_route_cost_equals_sum_of_link_costs(nav):
    reset_states(nav)
    r = route(nav, "GF:R:G-16")
    assert r is not None
    total = sum(link_cost(nav.link(u, v)) for u, v in r.links)
    assert r.cost == pytest.approx(total)


def test_heuristic_is_admissible_along_every_found_route(nav):
    """h(n) = dist/v_max must never exceed the true remaining cost, or A*
    could return a suboptimal route."""
    reset_states(nav)
    for nid in ("GF:R:G-01", "GF:R:LOBBY", "ST-C:FF", "FF:R:F-10"):
        r = route(nav, nid)
        assert r is not None
        for node_on_path in r.nodes:
            ax, ay = nav.node(node_on_path).XY
            best_ap_dist = min(
                math.hypot(ax - nav.node(ap).XY[0], ay - nav.node(ap).XY[1]) for ap in nav.assembly_point_nodes()
            )
            h = best_ap_dist / V_MAX
            # remaining true cost from this node to the goal actually used:
            idx = r.nodes.index(node_on_path)
            tail_cost = sum(link_cost(nav.link(u, v)) for u, v in r.links[idx:])
            assert h <= tail_cost + 1e-6


# --------------------------------------------------------------------- scenario presets


def test_baseline_preset_locks_service_exit(nav, site):
    reset_states(nav)
    apply_scenario(nav, site, "baseline")
    x_se = nav.door_node_id["X-SE"]
    assert all(nav.link(x_se, nb).state == "UNAVAILABLE" for nb in nav.g.neighbors(x_se))


def test_block_main_exit_raises_lobby_cost_and_avoids_x_s(nav, site):
    reset_states(nav)
    baseline_cost = route(nav, "GF:R:LOBBY").cost

    reset_states(nav)
    apply_scenario(nav, site, "block_main_exit")
    r = route(nav, "GF:R:LOBBY")

    assert r is not None
    assert "X:X-S" not in r.nodes
    assert r.cost > baseline_cost


def test_stair_c_lost_reroutes_ff_via_another_stair(nav, site):
    reset_states(nav)
    baseline = route(nav, "ST-C:FF")
    assert baseline is not None
    assert "ST-C:GF" in baseline.nodes

    reset_states(nav)
    apply_scenario(nav, site, "stair_C_lost")
    rerouted = route(nav, "ST-C:FF")

    assert rerouted is not None
    assert "ST-C:GF" not in rerouted.nodes  # the vertical link itself is unavailable
    used_stairs = {n.split(":")[0] for n in rerouted.nodes if n.endswith((":GF", ":FF"))}
    assert used_stairs & {"ST-W", "ST-E"}
    assert rerouted.cost > baseline.cost


def test_two_stairs_lost_ff_still_egresses_via_st_w(nav, site):
    reset_states(nav)
    apply_scenario(nav, site, "two_stairs_lost")

    for room_nid in ("FF:R:F-10", "FF:R:F-12", "FF:R:F-07"):
        r = route(nav, room_nid)
        assert r is not None, f"{room_nid} must still reach an assembly point via ST-W"
        assert "ST-W:FF" in r.nodes and "ST-W:GF" in r.nodes
        assert "ST-C:GF" not in r.nodes
        assert "ST-E:GF" not in r.nodes


def test_block_ring_s_mid_detours_lobby_around_the_blocked_opening(nav, site):
    reset_states(nav)
    baseline = route(nav, "GF:R:LOBBY")

    reset_states(nav)
    apply_scenario(nav, site, "block_ringS_mid")
    rerouted = route(nav, "GF:R:LOBBY")

    assert rerouted is not None
    assert "GF:O:LOBBY-op-ringS" not in rerouted.nodes
    assert rerouted.cost >= baseline.cost


def test_unhandled_events_are_reported_not_silently_dropped(nav, site):
    reset_states(nav)
    result = apply_scenario(nav, site, "slide23_C")  # ToggleLayer events: no cost-model hook yet
    assert result.unhandled
    assert all(ev["type"] == "ToggleLayer" for ev in result.unhandled)


def test_reset_states_clears_a_previously_applied_scenario(nav, site):
    apply_scenario(nav, site, "block_main_exit")
    x_s = nav.door_node_id["X-S"]
    assert any(nav.link(x_s, nb).state == "UNAVAILABLE" for nb in nav.g.neighbors(x_s))

    reset_states(nav)
    assert all(nav.link(x_s, nb).state == "AVAILABLE" for nb in nav.g.neighbors(x_s))
