"""§7.1: stair footprints coincide on GF and FF."""

from __future__ import annotations

from dataclasses import replace

from dms.geometry.building import Building
from dms.geometry.primitives import Rect
from dms.geometry.validate import ValidationReport, check_stair_alignment


def test_baseline_stairs_are_aligned(building):
    report = ValidationReport()
    check_stair_alignment(building, report)
    assert report.ok, [i.message for i in report.errors]


def test_baseline_stair_footprints_are_identical_rects(building):
    for stair in building.stairs.values():
        assert stair.footprint_gf == stair.footprint_ff


def test_misaligned_stair_is_flagged(building):
    broken = Building(
        id="TEST", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2,
        stairs={
            sid: replace(stair, footprint_ff=Rect(0.0, 0.0, 1.0, 1.0)) if sid == "ST-W" else stair
            for sid, stair in building.stairs.items()
        },
    )

    report = ValidationReport()
    check_stair_alignment(broken, report)

    assert not report.ok
    errors = [i for i in report.errors if i.code == "STAIR_FOOTPRINT_MISALIGNED"]
    assert len(errors) == 1
    assert "ST-W" in errors[0].message
