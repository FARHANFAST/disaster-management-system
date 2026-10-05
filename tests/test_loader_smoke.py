"""Basic sanity that the YAML loads into the expected shape before the
geometric checks (in the other test files) go any deeper."""

from __future__ import annotations

from dms.geometry.loader import LoaderError, load_site


def test_site_and_building_load(site, building):
    assert site.name
    assert site.building_ids() == ["B1"]
    assert building.id == "B1"
    assert building.origin == (50.0, 50.0)


def test_both_floors_present(building, gf, ff):
    assert building.floor_ids() == ["GF", "FF"]
    assert gf.z == 0.0
    assert ff.z == 4.0


def test_gf_has_six_exits_one_locked(gf):
    exits = {d.id: d for d in gf.exit_doors}
    assert set(exits) == {"X-S", "X-N", "X-W", "X-E", "X-NW", "X-SE"}
    assert exits["X-SE"].default_state == "UNAVAILABLE"
    assert all(d.default_state == "AVAILABLE" for k, d in exits.items() if k != "X-SE")


def test_ff_has_no_exterior_exits(ff):
    assert ff.exit_doors == []


def test_three_stairs_present(building):
    assert set(building.stairs) == {"ST-W", "ST-E", "ST-C"}
    for stair in building.stairs.values():
        assert stair.n_risers > 0
        assert stair.walking_length_m > 0


def test_windows_generated_for_rooms_only(gf):
    assert gf.windows, "window rule should have produced at least one window"
    for w in gf.windows.values():
        assert gf.spaces[w.space_id].kind == "room"


def test_designed_dead_ends_are_known_spaces(building):
    for sid in building.designed_dead_ends:
        floor = building.get_floor("GF") if sid in building.get_floor("GF").spaces else building.get_floor("FF")
        assert sid in floor.spaces


def test_orphan_door_raises(tmp_path, site):
    # A door with no matching neighbour must fail loudly at load time, not be
    # silently dropped (PROMPT_01 §9: "do not silently fix the layout").
    import yaml

    broken = {
        "site": {"name": "x", "width_m": 10.0, "height_m": 10.0},
        "buildings": [
            {
                "id": "BX",
                "origin": {"x": 0.0, "y": 0.0},
                "floors": [
                    {
                        "id": "GF",
                        "z": 0.0,
                        "footprint": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
                        "spaces": [
                            {
                                "id": "A",
                                "kind": "room",
                                "rect": {"x0": 0.0, "y0": 0.0, "x1": 5.0, "y1": 10.0},
                                "doors": [{"id": "A-d1", "wall": "x=5", "lo": 50.0, "hi": 52.0}],
                            },
                            {
                                "id": "B",
                                "kind": "room",
                                "rect": {"x0": 5.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
                            },
                        ],
                    },
                    {
                        "id": "FF",
                        "z": 4.0,
                        "footprint": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0},
                        "spaces": [
                            {"id": "A", "kind": "room", "rect": {"x0": 0.0, "y0": 0.0, "x1": 10.0, "y1": 10.0}},
                        ],
                    },
                ],
            }
        ],
    }
    p = tmp_path / "broken.yaml"
    p.write_text(yaml.safe_dump(broken), encoding="utf-8")
    try:
        load_site(p)
        assert False, "expected LoaderError"
    except LoaderError:
        pass
