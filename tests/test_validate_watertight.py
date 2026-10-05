"""§7.1 envelope watertight (vector-level stand-in for the future grid flood
fill): spaces must exactly tile the footprint, and declared exits/windows on
the same wall line must not overlap."""

from __future__ import annotations

from dms.geometry.building import Building, Floor, Space
from dms.geometry.primitives import Door, Rect
from dms.geometry.validate import (
    ValidationReport,
    check_tiling,
    check_watertight_envelope,
)


def _tiny_building() -> Building:
    return Building(id="TEST", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)


def test_baseline_floors_are_watertight(building, gf, ff):
    report = ValidationReport()
    check_watertight_envelope(building, gf, report)
    check_watertight_envelope(building, ff, report)
    assert report.ok, [i.message for i in report.errors]


def test_full_tiling_passes():
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 20.0, 10.0))

    report = ValidationReport()
    check_tiling(_tiny_building(), floor, report)

    assert report.ok


def test_gap_in_tiling_is_flagged():
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 9.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 20.0, 10.0))

    report = ValidationReport()
    check_tiling(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "FLOOR_NOT_TILED" for i in report.errors)


def test_space_extending_past_footprint_is_flagged():
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 25.0, 10.0))

    report = ValidationReport()
    check_tiling(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "FLOOR_NOT_TILED" for i in report.errors)


def test_overlapping_exits_on_same_wall_are_flagged():
    floor = Floor(id="GF", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 10.0))
    floor.spaces["A"] = Space(id="A", floor="GF", name="A", kind="room", rect=Rect(0.0, 0.0, 10.0, 10.0))
    floor.spaces["B"] = Space(id="B", floor="GF", name="B", kind="room", rect=Rect(10.0, 0.0, 20.0, 10.0))
    floor.doors["x1"] = Door(id="x1", floor="GF", axis="y", coord=0.0, lo=1.0, hi=3.0, kind="exit", space_a="A", space_b="outside")
    floor.doors["x2"] = Door(id="x2", floor="GF", axis="y", coord=0.0, lo=2.0, hi=4.0, kind="exit", space_a="A", space_b="outside")

    report = ValidationReport()
    check_watertight_envelope(_tiny_building(), floor, report)

    assert not report.ok
    assert any(i.code == "ENVELOPE_BREACH_OVERLAP" for i in report.errors)
