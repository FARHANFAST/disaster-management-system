"""spec §2: an obstacle instance may override its catalog type's own
size via `size: [L, W]` -- e.g. a maintenance barrier sized to the exact
corridor it blocks, rather than needing a new catalog entry per instance."""

from __future__ import annotations

import pytest
import yaml

from dms.geometry.loader import load_site


def _synthetic_doc(obstacles: list[dict]) -> dict:
    return {
        "site": {"name": "x", "width_m": 20.0, "height_m": 20.0},
        "buildings": [
            {
                "id": "BX",
                "origin": {"x": 0.0, "y": 0.0},
                "obstacle_types": {"barrier": {"width_m": 3.0, "height_m": 0.2}},
                "floors": [
                    {
                        "id": "GF",
                        "z": 0.0,
                        "footprint": {"x0": 0.0, "y0": 0.0, "x1": 20.0, "y1": 20.0},
                        "spaces": [
                            {
                                "id": "C",
                                "kind": "corridor",
                                "rect": {"x0": 0.0, "y0": 0.0, "x1": 20.0, "y1": 20.0},
                                "obstacles": obstacles,
                            },
                        ],
                    },
                ],
            }
        ],
    }


def _load(tmp_path, obstacles):
    p = tmp_path / "doc.yaml"
    p.write_text(yaml.safe_dump(_synthetic_doc(obstacles)), encoding="utf-8")
    site = load_site(p)
    return site.get_building("BX").get_floor("GF")


def test_default_catalog_size_used_without_override(tmp_path):
    floor = _load(tmp_path, [{"id": "o1", "type": "barrier", "x": 10.0, "y": 10.0}])
    r = floor.obstacles["o1"].rect
    assert r.width == pytest.approx(3.0) and r.height == pytest.approx(0.2)


def test_size_override_replaces_catalog_size(tmp_path):
    floor = _load(tmp_path, [
        {"id": "o1", "type": "barrier", "x": 10.0, "y": 10.0},
        {"id": "o2", "type": "barrier", "x": 10.0, "y": 15.0, "size": [2.0, 0.2]},
    ])
    default_rect = floor.obstacles["o1"].rect
    overridden_rect = floor.obstacles["o2"].rect
    assert default_rect.width == pytest.approx(3.0)
    assert overridden_rect.width == pytest.approx(2.0)
    assert overridden_rect.height == pytest.approx(0.2)
    assert floor.obstacles["o2"].type == "barrier"  # type kept for display/grouping despite the size override
