"""§7.1: every door/opening lies on exactly one shared boundary between two
spaces, or between a space and outside; every exit lies on the envelope."""

from __future__ import annotations

from dms.geometry.building import Building, Floor, Space
from dms.geometry.primitives import Door, Opening, Rect
from dms.geometry.validate import (
    ValidationReport,
    check_doors_on_shared_boundary,
    check_exits_on_envelope,
)


def _tiny_building() -> Building:
    return Building(id="TEST", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)


def _two_room_floor() -> Floor:
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 20.0, 10.0))
    return floor


def test_baseline_doors_and_openings_resolve_cleanly(building, gf, ff):
    report = ValidationReport()
    check_doors_on_shared_boundary(building, gf, report)
    check_doors_on_shared_boundary(building, ff, report)
    check_exits_on_envelope(building, gf, report)
    check_exits_on_envelope(building, ff, report)
    assert report.ok, [i.message for i in report.errors]


def test_correct_interior_door_passes():
    floor = _two_room_floor()
    floor.doors["d1"] = Door(id="d1", floor="GF", axis="x", coord=10.0, lo=2.0, hi=4.0, space_a="A", space_b="B")

    report = ValidationReport()
    check_doors_on_shared_boundary(_tiny_building(), floor, report)

    assert report.ok


def test_door_with_wrong_claimed_neighbour_is_flagged():
    floor = _two_room_floor()
    # The door is physically on the A/B boundary, but claims a neighbour that
    # doesn't actually touch that span there.
    floor.doors["d1"] = Door(id="d1", floor="GF", axis="x", coord=10.0, lo=2.0, hi=4.0, space_a="A", space_b="GHOST")

    report = ValidationReport()
    check_doors_on_shared_boundary(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "DOOR_BOUNDARY_MISMATCH" for i in report.errors)


def test_opening_not_on_a_real_shared_edge_is_flagged():
    floor = _two_room_floor()
    # x=5 isn't a boundary of either space.
    floor.openings["o1"] = Opening(id="o1", floor="GF", axis="x", coord=5.0, lo=2.0, hi=4.0, space_a="A", space_b="B")

    report = ValidationReport()
    check_doors_on_shared_boundary(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "OPENING_BOUNDARY_MISMATCH" for i in report.errors)


def test_exit_not_on_envelope_is_flagged():
    floor = _two_room_floor()
    # x=10 is an interior wall between A and B, not the exterior envelope.
    floor.doors["x1"] = Door(
        id="x1", floor="GF", axis="x", coord=10.0, lo=2.0, hi=4.0, kind="exit", space_a="A", space_b="outside"
    )

    report = ValidationReport()
    check_doors_on_shared_boundary(_tiny_building(), floor, report)
    check_exits_on_envelope(_tiny_building(), floor, report)

    assert not report.ok
    codes = {i.code for i in report.errors}
    assert "EXIT_NOT_ON_ENVELOPE" in codes


def test_exit_on_envelope_passes():
    floor = _two_room_floor()
    floor.doors["x1"] = Door(
        id="x1", floor="GF", axis="x", coord=0.0, lo=2.0, hi=4.0, kind="exit", space_a="A", space_b="outside"
    )

    report = ValidationReport()
    check_doors_on_shared_boundary(_tiny_building(), floor, report)
    check_exits_on_envelope(_tiny_building(), floor, report)

    assert report.ok
