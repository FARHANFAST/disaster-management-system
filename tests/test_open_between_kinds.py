"""spec §2: `open_between_kinds` auto-generates an Opening between any
two same-floor spaces whose kinds match a declared pair and who share a
boundary -- replacing the need to list every corridor/stub junction as an
explicit `opening:`/`corridor_opening:` entry."""

from __future__ import annotations

import yaml

from dms.geometry.loader import load_site
from dms.geometry.validate import validate_site


def _synthetic_doc(open_between_kinds: list) -> dict:
    return {
        "site": {"name": "x", "width_m": 20.0, "height_m": 20.0},
        "buildings": [
            {
                "id": "BX",
                "origin": {"x": 0.0, "y": 0.0},
                "open_between_kinds": open_between_kinds,
                "floors": [
                    {
                        "id": "GF",
                        "z": 0.0,
                        "footprint": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
                        "spaces": [
                            {"id": "A", "kind": "corridor", "rect": {"x0": 0.0, "y0": 0.0, "x1": 5.0, "y1": 10.0}},
                            {"id": "B", "kind": "corridor", "rect": {"x0": 5.0, "y0": 0.0, "x1": 10.0, "y1": 10.0}},
                        ],
                    },
                ],
            }
        ],
    }


def _load(tmp_path, open_between_kinds):
    p = tmp_path / "doc.yaml"
    p.write_text(yaml.safe_dump(_synthetic_doc(open_between_kinds)), encoding="utf-8")
    return load_site(p)


def test_auto_opening_generated_for_declared_kind_pair(tmp_path):
    site = _load(tmp_path, [["corridor", "corridor"]])
    floor = site.get_building("BX").get_floor("GF")
    autos = [o for o in floor.openings.values() if {o.space_a, o.space_b} == {"A", "B"}]
    assert len(autos) == 1
    o = autos[0]
    assert o.axis == "x" and o.lo == 0.0 and o.hi == 10.0  # the full shared edge, nothing already covering it

    report = validate_site(site)
    assert report.ok, report.to_markdown()


def test_no_auto_opening_without_declared_kind_pair(tmp_path):
    site = _load(tmp_path, [])  # no open_between_kinds at all -- the two corridors stay walled apart
    floor = site.get_building("BX").get_floor("GF")
    autos = [o for o in floor.openings.values() if {o.space_a, o.space_b} == {"A", "B"}]
    assert autos == []
