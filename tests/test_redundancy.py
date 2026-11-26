"""Milestone 5: connectivity and redundancy analysis (spec §7.2)."""

from __future__ import annotations

from dms.graph.redundancy import (
    REFERENCE_ROOMS,
    _expand_declared_dead_ends,
    apply_failure,
    check_ff_ring_edge_disjoint_paths,
    enumerate_single_failures,
    run_exit_pair_sweep,
    run_named_double_failures,
    run_single_failure_sweep,
)


def test_baseline_every_room_reaches_an_assembly_point(nav):
    import networkx as nx

    assert nx.is_connected(nav.g)


def test_single_failure_sweep_covers_every_controllable_element(nav):
    cases = enumerate_single_failures(nav)
    kinds = {c.kind for c in cases}
    assert kinds == {"door", "fire_door", "exit", "opening", "stair", "corridor_segment"}
    # every door/opening/stair id in the model is actually exercised
    assert {c.element_id for c in cases if c.kind in ("door", "fire_door", "exit")} == set(nav.door_node_id)
    assert {c.element_id for c in cases if c.kind == "opening"} == set(nav.opening_node_id)
    assert {c.element_id for c in cases if c.kind == "stair"} == set(nav.stair_landing_nodes)


def test_single_failure_never_disconnects_a_non_dead_end_room(nav, building):
    results = run_single_failure_sweep(nav, building)
    declared = _expand_declared_dead_ends(building, building.designed_dead_ends)
    for r in results:
        unexpected = r.disconnected_rooms - declared
        assert not unexpected, f"{r.case.label} ({r.case.kind}) unexpectedly disconnected {sorted(unexpected)}"


def test_single_failure_affected_set_matches_designed_dead_ends_exactly(nav, building):
    """spec §7.2: 'the test asserts that exactly the declared set is affected.'"""
    results = run_single_failure_sweep(nav, building)
    all_affected = frozenset().union(*(r.disconnected_rooms for r in results))
    assert all_affected == _expand_declared_dead_ends(building, building.designed_dead_ends)


def test_each_dead_end_room_is_disconnected_by_failing_its_own_sole_door(nav, building, gf, ff):
    """Every declared dead-end room must actually go unreachable when *its*
    specific door fails — not just appear in the aggregate union above.
    Tracked as a (floor, space) pair throughout (spec §2) since a bare
    id is not assumed unique across floors."""
    results = {r.case.element_id: r for r in run_single_failure_sweep(nav, building) if r.case.kind == "door"}
    for space_id in building.designed_dead_ends:
        floor = gf if space_id in gf.spaces else ff
        doors = [d for d in floor.doors.values() if d.space_a == space_id or d.space_b == space_id]
        assert len(doors) == 1, f"{space_id} is declared a dead end but has {len(doors)} doors"
        door_id = doors[0].id
        assert (floor.id, space_id) in results[door_id].disconnected_rooms, (
            f"failing {space_id}'s only door ({door_id}) should disconnect it"
        )


def test_reference_room_routes_exist_in_baseline(nav):
    from dms.graph.astar import route

    for rid in REFERENCE_ROOMS:
        assert route(nav, rid) is not None


def test_apply_failure_removes_exactly_the_targeted_element(nav):
    cases = {c.element_id: c for c in enumerate_single_failures(nav)}
    case = cases["X-S"]
    g = apply_failure(nav, case)
    assert g.number_of_nodes() == nav.g.number_of_nodes() - 1
    assert not g.has_node("X:X-S")
    # the original graph must be untouched
    assert nav.g.has_node("X:X-S")


def test_two_stairs_lost_disconnects_nothing(nav, building):
    results = run_named_double_failures(nav, building)
    r = next(r for r in results if r.label.startswith("two_stairs_lost"))
    assert r.disconnected_rooms == frozenset()


def test_block_main_exit_plus_ring_s_disconnects_nothing(nav, building):
    results = run_named_double_failures(nav, building)
    r = next(r for r in results if "block_main_exit" in r.label)
    assert r.disconnected_rooms == frozenset()


def test_every_exit_pair_leaves_the_building_connected(nav, building):
    results = run_exit_pair_sweep(nav, building)
    assert len(results) == 15  # C(6,2)
    for r in results:
        assert r.disconnected_rooms == frozenset(), f"{r.label} disconnected {sorted(r.disconnected_rooms)}"


def test_ff_ring_segments_have_at_least_two_disjoint_paths_except_near_a_dead_end(nav):
    """ring-W-00's only other neighbour is F-01's door, and F-01 is itself a
    single-door dead end (no onward path) — so 1 is the structurally correct
    answer there, not a bug. Every other ring segment must hit the target."""
    counts = check_ff_ring_edge_disjoint_paths(nav)
    assert counts, "expected at least one FF ring segment node"
    below_target = {nid: n for nid, n in counts.items() if n < 2}
    assert below_target == {"FF:C:ring-W-00": 1}


def _repeated_room_id_fixture_doc() -> dict:
    """Two floors, each with a room literally named "A" (a single-door dead
    end) and a corridor "C" with its own exit -- same bare space ids on both
    floors, distinct door ids (door ids are a separate, pre-existing
    assumption not in scope here) -- to prove disconnected_rooms tracks
    (floor, space) pairs and never conflates the two floors' "A"."""
    def floor(fid: str, z: float) -> dict:
        return {
            "id": fid, "z": z,
            "footprint": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
            "spaces": [
                {
                    "id": "A", "kind": "room", "rect": {"x0": 0.0, "y0": 0.0, "x1": 5.0, "y1": 10.0},
                    "doors": [{"id": f"A-d1-{fid}", "wall": "x=5", "lo": 4.0, "hi": 6.0}],
                },
                {"id": "C", "kind": "corridor", "rect": {"x0": 5.0, "y0": 0.0, "x1": 10.0, "y1": 10.0}},
            ],
            "exits": [{"id": f"C-exit-{fid}", "space": "C", "wall": "x=10", "lo": 4.0, "hi": 6.0}],
        }

    return {
        "site": {
            "name": "x", "width_m": 40.0, "height_m": 40.0,
            "assembly_points": [{"id": "AP1", "x": 25.0, "y": 5.0, "diameter_m": 10.0}],
        },
        "buildings": [
            {
                "id": "BX",
                "origin": {"x": 0.0, "y": 0.0},
                "designed_dead_ends": ["A"],
                "floors": [floor("GF", 0.0), floor("FF", 4.0)],
            }
        ],
    }


def test_repeated_bare_room_id_across_floors_is_not_conflated(tmp_path):
    import yaml

    from dms.geometry.loader import load_site
    from dms.graph.builder import build_navigation_graph

    p = tmp_path / "repeated.yaml"
    p.write_text(yaml.safe_dump(_repeated_room_id_fixture_doc()), encoding="utf-8")
    site = load_site(p)
    building = site.get_building("BX")
    nav = build_navigation_graph(site, "BX")

    declared = _expand_declared_dead_ends(building, building.designed_dead_ends)
    assert declared == {("GF", "A"), ("FF", "A")}

    results = {r.case.element_id: r for r in run_single_failure_sweep(nav, building) if r.case.kind == "door"}
    assert results["A-d1-GF"].disconnected_rooms == {("GF", "A")}
    assert results["A-d1-FF"].disconnected_rooms == {("FF", "A")}
    # failing one floor's door must never disconnect the other floor's "A"
    assert ("FF", "A") not in results["A-d1-GF"].disconnected_rooms
    assert ("GF", "A") not in results["A-d1-FF"].disconnected_rooms


def test_write_redundancy_report(tmp_path, nav, building):
    from dms.graph.redundancy import write_redundancy_report

    single = run_single_failure_sweep(nav, building)
    named = run_named_double_failures(nav, building)
    pairs = run_exit_pair_sweep(nav, building)
    ring = check_ff_ring_edge_disjoint_paths(nav)

    out = tmp_path / "redundancy.md"
    path = write_redundancy_report(nav, building, out, single, named, pairs, ring)
    text = path.read_text(encoding="utf-8")
    assert "Redundancy report" in text
    assert "Single-failure sweep" in text
    assert "two_stairs_lost" in text
    assert "ring-W-00" in text
