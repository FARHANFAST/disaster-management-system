"""Integration: validate_site must run clean on the authoritative two_storey
layout (spec §7.1: "run on every load")."""

from __future__ import annotations

from dms.geometry.validate import validate_site


def test_two_storey_baseline_validates_clean(site):
    report = validate_site(site)
    assert report.ok, report.to_markdown()
    assert report.warnings == []
