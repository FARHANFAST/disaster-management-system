"""A small, hand-built test-only fixture
building (never configs/buildings/, never B1-L) with controlled door/gap
widths, specifically to exercise all four clearance-shading buckets and
cross-check the door-passability table against the grid -- not just assert
0 mismatches on a building that happens not to have any narrow doors.

Layout (one floor, one straight row of rooms, all local metres):
    R1 --0.6m door-- R2 --0.82m door-- R3 --0.95m door-- R4 --1.5m exit--> outside
R4 also has two furniture blocks leaving a 0.7m clear gap across its width.

Required widths at this building's own agent_defaults (margin=0.10 m):
    standard (r=0.23): 0.56 m   large (r=0.28): 0.66 m
    mobility_aid (r=0.35): 0.80 m   wheelchair (r=0.40): 0.90 m
So: 0.6m door -> standard only; 0.82m door -> not wheelchair (mobility_aid
boundary, passable -- deliberately 0.82 m, not exactly 0.80 m, so the
comparison doesn't land on a binary-float boundary where `hi - lo` rounds to
a hair under 0.80); 0.95m door -> all classes; 0.7m furniture gap -> not
mobility_aid (passable up to large, standard).

Both the bucket shading and the door-passability cross-check below use the
SAME margin-inflated radius (`clearance_radius_m + door_margin_m/2`) that
`door_passability_table` and `dms/viz/progress_body_classes.py`'s own area
shading use -- comparing a margin-inclusive table against a bare-radius grid
check is an apples-to-oranges mismatch that masked real disagreements (see
the module docstring above). The grid-vs-table cross-check also
runs at `H / 2`, not `H`: a door whose width is an exact multiple of the
grid step can undercount its own clearance by up to half a cell, a sampling
artifact of the verification, not a real geometric shortfall (confirmed
against configs/buildings/two_storey_large.yaml, where the same artifact
appeared and was checked away by halving the resolution, not by changing
the geometry)."""

from __future__ import annotations

import pytest

from dms.fields.eikonal import clearance_domain
from dms.geometry.body_class_report import door_passability_table
from dms.geometry.building import Building, Floor, Space
from dms.geometry.primitives import Door, Obstacle, Rect
from dms.grid.layers import DynamicOverlay
from dms.grid.raster import rasterize_floor
from dms.scenario.agent_defaults import load_agent_defaults
from dms.viz.progress_body_classes import render_body_class_diagnostic

H = 0.1  # fine grid so a 0.6-0.95 m door/gap isn't swallowed by quantization


def _fixture_building() -> Building:
    rooms = {
        "R1": Rect(0.0, 0.0, 4.0, 6.0),
        "R2": Rect(4.0, 0.0, 8.0, 6.0),
        "R3": Rect(8.0, 0.0, 12.0, 6.0),
        "R4": Rect(12.0, 0.0, 20.0, 6.0),
    }
    spaces = {
        sid: Space(id=sid, floor="L1", name=sid, kind="room", rect=r, use="office")
        for sid, r in rooms.items()
    }

    doors = {
        "D-0.6": Door(id="D-0.6", floor="L1", axis="x", coord=4.0, lo=2.7, hi=3.3,
                       kind="interior", space_a="R1", space_b="R2"),
        "D-0.8": Door(id="D-0.8", floor="L1", axis="x", coord=8.0, lo=2.59, hi=3.41,
                       kind="interior", space_a="R2", space_b="R3"),
        "D-0.95": Door(id="D-0.95", floor="L1", axis="x", coord=12.0, lo=2.525, hi=3.475,
                        kind="interior", space_a="R3", space_b="R4"),
        "X-1": Door(id="X-1", floor="L1", axis="x", coord=20.0, lo=2.25, hi=3.75,
                     kind="exit", default_state="AVAILABLE", space_a="R4", space_b="outside"),
    }

    # Two furniture blocks across R4's width (y), 1 m deep (x), leaving a
    # 0.7 m clear gap in the middle -- a corridor pinch with no declared door.
    obstacles = {
        "FURN-N": Obstacle(id="FURN-N", floor="L1", rect=Rect(15.5, 3.35, 16.5, 6.0), space_id="R4"),
        "FURN-S": Obstacle(id="FURN-S", floor="L1", rect=Rect(15.5, 0.0, 16.5, 2.65), space_id="R4"),
    }

    floor = Floor(id="L1", z=0.0, footprint=Rect(0.0, 0.0, 20.0, 6.0), spaces=spaces, doors=doors,
                  obstacles=obstacles)
    building = Building(id="CLEARANCE-FIXTURE", origin=(0.0, 0.0), rotation_deg=0.0, wall_thickness_m=0.2)
    building.floors["L1"] = floor
    return building


@pytest.fixture(scope="module")
def building() -> Building:
    return _fixture_building()


@pytest.fixture(scope="module")
def agent_defaults():
    return load_agent_defaults()


def test_four_clearance_buckets_present(building, agent_defaults):
    """The four doors/gap are deliberately sized to land in all four
    narrowest-passing-class buckets -- verified directly, not assumed.

    Rasterized at H/2, not H, for the same reason as the door-passability
    cross-check below: at H alone, the 0.7 m furniture gap's own quantization
    can undercount its true centre clearance by up to H/2, enough to drop it
    a whole bucket."""
    floor = building.get_floor("L1")
    grid = rasterize_floor(building, floor, H / 2.0)
    overlay = DynamicOverlay()
    classes = sorted(agent_defaults.body_classes.items(), key=lambda kv: kv[1].clearance_radius_m)
    margin = agent_defaults.door_margin_m

    def bucket_at(x: float, y: float) -> str:
        row, col = grid.transform.local_to_cell(x, y)
        for cid, bc in reversed(classes):  # widest-reach first
            if clearance_domain(grid, overlay, bc.clearance_radius_m + margin / 2.0)[row, col]:
                return cid
        return "none"

    assert bucket_at(4.0, 3.0) == "standard"       # the 0.6 m door
    assert bucket_at(8.0, 3.0) == "mobility_aid"    # the 0.82 m door (passable up to mobility_aid)
    assert bucket_at(12.0, 3.0) == "wheelchair"     # the 0.95 m door (passable even for wheelchair)
    assert bucket_at(16.0, 3.0) == "large"          # the 0.7 m furniture gap (passable up to large)


def test_door_passability_table_matches_grid_for_every_door(building, agent_defaults):
    """Interior doors only -- an exit's own door_id cells sit on the facade
    boundary and aren't part of the indoor walkable mask `clearance_domain`
    operates on, the same reason `tests/test_b1l_body_classes.py`'s
    equivalent check excludes exits too (see `test_every_exit_is_passable`
    below for the exit's own, table-only check)."""
    floor = building.get_floor("L1")
    grid = rasterize_floor(building, floor, H / 2.0)
    overlay = DynamicOverlay()
    table = {(r.door_id, r.body_class): r.passable for r in door_passability_table(building, agent_defaults=agent_defaults)}

    checked = 0
    interior = [d for d in floor.doors.values() if d.kind != "exit"]
    for door in interior:
        door_idx = grid.door_ids.index(door.id)
        cell_mask = grid.door_id == door_idx
        assert cell_mask.any(), door.id
        for cid, bc in agent_defaults.body_classes.items():
            r_check = bc.clearance_radius_m + agent_defaults.door_margin_m / 2.0
            move = clearance_domain(grid, overlay, r_check)
            grid_pass = bool((move & cell_mask).any())
            geom_pass = table[(door.id, cid)]
            assert grid_pass == geom_pass, (door.id, cid, geom_pass, grid_pass)
            checked += 1
    assert checked == len(interior) * len(agent_defaults.body_classes)


def test_every_exit_is_passable(building, agent_defaults):
    rows = [r for r in door_passability_table(building, agent_defaults=agent_defaults) if r.kind == "exit"]
    assert rows, "fixture should declare at least one exit"
    assert all(r.passable for r in rows), [r for r in rows if not r.passable]


def test_expected_passability_values(building, agent_defaults):
    rows = {(r.door_id, r.body_class): r.passable for r in door_passability_table(building, agent_defaults=agent_defaults)}
    assert rows[("D-0.6", "standard")] is True
    assert rows[("D-0.6", "large")] is False
    assert rows[("D-0.8", "mobility_aid")] is True
    assert rows[("D-0.8", "wheelchair")] is False
    assert rows[("D-0.95", "wheelchair")] is True
    assert rows[("X-1", "wheelchair")] is True  # the exit stays wide


def test_render_clearance_fixture_png(building):
    out = render_body_class_diagnostic(building, out_name="clearance_fixture.png")
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.name == "clearance_fixture.png"
