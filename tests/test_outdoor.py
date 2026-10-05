"""Milestone 6: outdoor L3 site network (spec §3.9) — rasterisation and
FMM-based (obstacle-aware) exit<->assembly-point distances."""

from __future__ import annotations

import math

import pytest

from dms.fields.eikonal import direction_field
from dms.fields.outdoor import (
    build_outdoor_network,
    compute_ap_distance_fields,
    compute_nearest_ap_field,
    exit_discharge_site_xy,
    exit_outward_normal_site,
    rasterize_site,
)
from dms.graph.astar import route
from dms.graph.builder import build_navigation_graph


def test_rasterize_site_shape_matches_footprint(site):
    grid = rasterize_site(site, h=1.0)
    assert grid.transform.shape == (round(site.height_m), round(site.width_m))


def test_building_footprint_is_not_walkable(site, building):
    grid = rasterize_site(site, h=1.0)
    cx, cy = building.to_site_xy(30.0, 20.0)  # middle of the building, local coords
    row, col = grid.transform.local_to_cell(cx, cy)
    assert grid.building_mask[row, col]
    assert not grid.walkable[row, col]


def test_open_ground_away_from_everything_is_walkable(site):
    grid = rasterize_site(site, h=1.0)
    row, col = grid.transform.local_to_cell(5.0, 5.0)  # far SW corner, clear of the building/outbuilding
    assert grid.walkable[row, col]


def test_parking_toggle_blocks_only_when_on(site):
    off = rasterize_site(site, h=1.0, parking_on=False)
    on = rasterize_site(site, h=1.0, parking_on=True)
    assert off.walkable.sum() > on.walkable.sum()
    assert (off.building_mask == on.building_mask).all()  # parking never affects the building mask


def test_ap_distance_field_is_zero_at_its_own_ap_and_finite_elsewhere(site):
    grid = rasterize_site(site, h=1.0)
    fields = compute_ap_distance_fields(site, grid)
    for ap_id, ap in site.assembly_points.items():
        row, col = grid.transform.local_to_cell(ap.x, ap.y)
        assert fields[ap_id].data[row, col] == 0.0
        assert not fields[ap_id].mask[~grid.building_mask & grid.walkable].any()


def test_exit_discharge_point_is_outside_the_building_footprint(building, gf):
    for door in gf.exit_doors:
        site_x, site_y = exit_discharge_site_xy(building, "GF", door)
        local_x, local_y = building.from_site_xy(site_x, site_y)
        assert not gf.footprint.contains_point(local_x, local_y)


def test_outdoor_network_same_side_exit_is_much_closer_than_opposite_side(site, building):
    net = build_outdoor_network(site, building)
    d = net.exit_ap_distance_m
    # X-S sits on the south face, close to AP-S; AP-N is on the far side of
    # the building, which the route must go around.
    assert d[("X-S", "AP-S")] < d[("X-S", "AP-N")]


def test_outdoor_distance_exceeds_euclidean_when_the_building_is_in_the_way(site, building):
    """spec: "not Euclidean, so obstacles count" — a cross-building pair
    must detour around it, so FMM distance > straight-line distance."""
    net = build_outdoor_network(site, building)
    gf = building.get_floor("GF")
    x_e = next(d for d in gf.exit_doors if d.id == "X-E")
    ap_w = site.assembly_points["AP-W"]
    ex, ey = exit_discharge_site_xy(building, "GF", x_e)
    euclidean = math.hypot(ap_w.x - ex, ap_w.y - ey)
    assert net.exit_ap_distance_m[("X-E", "AP-W")] > euclidean * 1.1


def test_outdoor_network_has_every_exit_ap_pair(site, building, gf):
    net = build_outdoor_network(site, building)
    expected = {(d.id, ap_id) for d in gf.exit_doors for ap_id in site.assembly_points}
    assert set(net.exit_ap_distance_m) == expected


def test_graph_exit_links_use_outdoor_distance_not_euclidean(site):
    with_outdoor = build_navigation_graph(site, "B1", compute_node_of=False, use_outdoor_network=True)
    without = build_navigation_graph(site, "B1", compute_node_of=False, use_outdoor_network=False)

    link_out = with_outdoor.link("X:X-E", "AP:AP-W")
    link_eucl = without.link("X:X-E", "AP:AP-W")
    assert link_out.length_m > link_eucl.length_m * 1.1


def test_graph_still_fully_routes_with_outdoor_network(nav):
    for nid in nav.room_nodes():
        assert route(nav, nid) is not None


def test_supplying_a_precomputed_outdoor_network_is_equivalent_to_computing_it(site, building):
    net = build_outdoor_network(site, building)
    a = build_navigation_graph(site, "B1", compute_node_of=False, outdoor_distances=net.exit_ap_distance_m)
    b = build_navigation_graph(site, "B1", compute_node_of=False, use_outdoor_network=True)
    assert a.link("X:X-S", "AP:AP-S").length_m == pytest.approx(b.link("X:X-S", "AP:AP-S").length_m)


# --------------------------------------------------------------------- exit-normal direction check


def test_individual_ap_fields_can_legitimately_point_sideways_not_just_outward(site, building, gf):
    """Negative control: this is *why* the real check below uses the
    combined nearest-AP field instead of each per-AP field individually —
    e.g. exiting via the south door toward the north AP, the fastest route
    goes sideways around the building first, not straight out the south
    wall, so that particular dot product is legitimately negative."""
    net = build_outdoor_network(site, building)
    door = next(d for d in gf.exit_doors if d.id == "X-S")
    nx, ny = exit_outward_normal_site(building, "GF", door)
    ex, ey = exit_discharge_site_xy(building, "GF", door)
    row, col = net.site_grid.transform.local_to_cell(ex, ey)
    ex_field, ey_field = direction_field(net.ap_fields["AP-N"], net.site_grid.walkable, net.site_grid.transform.h)
    dot = ex_field[row, col] * nx + ey_field[row, col] * ny
    assert dot < 0


def test_every_exit_is_steered_outward_by_the_nearest_ap_field(site, building, gf):
    """spec-adjacent sanity check: at every exterior exit, -grad(T) of the
    *combined* nearest-assembly-point field (all 4 APs seeded at once, the
    field an evacuating agent actually follows) has a positive dot product
    with the exit's own outward wall normal — confirming agents are
    steered away from the wall they just came through, never back into it."""
    net = build_outdoor_network(site, building)
    ex_field, ey_field = direction_field(net.nearest_ap_field, net.site_grid.walkable, net.site_grid.transform.h)

    assert len(gf.exit_doors) == 6
    for door in gf.exit_doors:
        nx, ny = exit_outward_normal_site(building, "GF", door)
        ex, ey = exit_discharge_site_xy(building, "GF", door)
        row, col = net.site_grid.transform.local_to_cell(ex, ey)
        assert net.site_grid.walkable[row, col]
        dx, dy = ex_field[row, col], ey_field[row, col]
        assert not math.isnan(dx)
        dot = dx * nx + dy * ny
        assert dot > 0.5, f"{door.id}: outward dot product {dot:.3f} is not steering away from the wall"


def test_nearest_ap_field_is_zero_at_every_assembly_point(site, building):
    net = build_outdoor_network(site, building)
    for ap in site.assembly_points.values():
        row, col = net.site_grid.transform.local_to_cell(ap.x, ap.y)
        assert net.nearest_ap_field.data[row, col] == 0.0


def test_nearest_ap_field_never_exceeds_any_individual_ap_field(site, building):
    """Mathematically the 4-seed combined field is the pointwise min of the
    4 single-seed fields, but FMM is a numerical PDE solve, not an exact
    min: the narrow-band heap visits cells in a different order with 4
    seeds active at once than with 1, so tiny floating-point disagreements
    (<0.05 m, far below the 1 m grid spacing) are expected discretisation
    noise rather than a construction bug — verified empirically to be
    scattered 40-70 cells from any seed, never near one."""
    net = build_outdoor_network(site, building)
    domain = net.site_grid.walkable
    for field in net.ap_fields.values():
        assert (net.nearest_ap_field.data[domain] <= field.data[domain] + 0.05).all()


def test_compute_nearest_ap_field_matches_outdoor_network(site, building):
    net = build_outdoor_network(site, building)
    standalone = compute_nearest_ap_field(site, net.site_grid)
    assert (standalone.data == net.nearest_ap_field.data).all()
