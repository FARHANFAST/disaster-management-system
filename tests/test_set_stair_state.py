"""spec §2: SetStairState is a stricter sibling of the existing
SetLinkState -- it must make a stair's footprint physically unwalkable on
every floor it serves (DynamicOverlay.blocked_stairs), in addition to
excluding it from the vertical Eikonal coupling (same as SetLinkState
already does via excluded_stairs/stairs_blocked_by_scenario)."""

from __future__ import annotations

from dms.fields.eikonal import stairs_blocked_by_scenario
from dms.grid.layers import DynamicOverlay
from dms.scenario.presets import blocked_stairs_from_events, door_states_from_events


def test_blocked_stairs_from_events_recognizes_set_stair_state(building):
    preset = {"events": [{"type": "SetStairState", "id": "ST-W", "state": "UNAVAILABLE"}]}
    assert blocked_stairs_from_events(preset, building) == frozenset({"ST-W"})


def test_blocked_stairs_from_events_ignores_set_link_state(building):
    """SetLinkState never blocks a footprint -- only SetStairState does."""
    preset = {"events": [{"type": "SetLinkState", "id": "ST-W", "state": "UNAVAILABLE"}]}
    assert blocked_stairs_from_events(preset, building) == frozenset()


def test_stairs_blocked_by_scenario_recognizes_both_event_types(site, building):
    site.scenarios["_test_link"] = {"events": [{"type": "SetLinkState", "id": "ST-W", "state": "UNAVAILABLE"}]}
    site.scenarios["_test_stair"] = {"events": [{"type": "SetStairState", "id": "ST-W", "state": "UNAVAILABLE"}]}
    try:
        assert stairs_blocked_by_scenario(site, building, "_test_link") == frozenset({"ST-W"})
        assert stairs_blocked_by_scenario(site, building, "_test_stair") == frozenset({"ST-W"})
    finally:
        del site.scenarios["_test_link"]
        del site.scenarios["_test_stair"]


def test_walkable_now_blocks_stair_footprint_on_every_floor(gf, ff, gf_grid, ff_grid):
    overlay = DynamicOverlay(blocked_stairs=frozenset({"ST-W"}))
    gf_walkable = gf_grid.walkable_now(overlay)
    ff_walkable = ff_grid.walkable_now(overlay)
    gf_idx = gf_grid.stair_ids.index("ST-W")
    ff_idx = ff_grid.stair_ids.index("ST-W")
    assert not gf_walkable[gf_grid.stair_id == gf_idx].any()
    assert not ff_walkable[ff_grid.stair_id == ff_idx].any()

    # baseline overlay (no blocked_stairs) leaves the stair walkable, as before
    baseline = DynamicOverlay()
    assert gf_grid.walkable_now(baseline)[gf_grid.stair_id == gf_idx].any()


def test_door_states_from_events_unaffected(building):
    preset = {"events": [{"type": "SetExitState", "id": "X-S", "state": "UNAVAILABLE"}]}
    assert door_states_from_events(preset) == {"X-S": "UNAVAILABLE"}
