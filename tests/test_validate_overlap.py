"""§7.1: no two spaces overlap on a floor."""

from __future__ import annotations

from dms.geometry.building import Building, Floor, Space
from dms.geometry.primitives import Rect
from dms.geometry.validate import ValidationReport, check_no_overlaps


def _tiny_building() -> Building:
    return Building(id="TEST", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)


def test_baseline_floors_have_no_overlaps(building, gf, ff):
    report = ValidationReport()
    check_no_overlaps(building, gf, report)
    check_no_overlaps(building, ff, report)
    assert report.ok, [i.message for i in report.errors]


def test_overlapping_spaces_are_flagged():
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 20.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(5.0, 0.0, 15.0, 10.0))

    report = ValidationReport()
    check_no_overlaps(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "SPACE_OVERLAP" for i in report.errors)


def test_edge_adjacent_spaces_are_not_flagged():
    """Touching along a shared edge (the normal case) must not count as overlap."""
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 20.0, 10.0))

    report = ValidationReport()
    check_no_overlaps(_tiny_building(), floor, report)

    assert report.ok
