"""Tabular Q-learning evacuee (dms/rl): environment transition rules, the
Dijkstra reference, and that Q-learning converges to that reference."""

from __future__ import annotations

import math

import numpy as np
import pytest

from dms.grid.layers import H_COARSE, DynamicOverlay
from dms.grid.raster import rasterize_floor
from dms.rl.env import MOVES, STAIRS, GridEvacEnv
from dms.rl.evaluate import evaluate_all_states, optimal_times
from dms.rl.q_learning import QConfig, greedy_rollout, train


class _Transform:
    def __init__(self, h: float, shape: tuple[int, int]):
        self.h = h
        self.origin_local = (0.0, 0.0)
        self.shape = shape

    def cell_to_local(self, row: int, col: int) -> tuple[float, float]:
        return (col + 0.5) * self.h, (row + 0.5) * self.h


class _ToyGrid:
    """Minimal FloorGrid stand-in: just the fields GridEvacEnv reads."""

    def __init__(self, walkable: np.ndarray, exits: np.ndarray, h: float = 0.5):
        self.floor_id = "TOY"
        self.transform = _Transform(h, walkable.shape)
        self._walkable = walkable
        self.exit_id = np.where(exits, 0, -1)

    def walkable_now(self, overlay: DynamicOverlay) -> np.ndarray:
        return self._walkable.copy()


def _toy(layout: list[str]) -> GridEvacEnv:
    """'#' wall, '.' floor, 'X' exit. layout[0] is the *top* row (highest y)."""
    rows = layout[::-1]
    walkable = np.array([[ch != "#" for ch in r] for r in rows])
    exits = np.array([[ch == "X" for ch in r] for r in rows])
    return GridEvacEnv(_ToyGrid(walkable, exits))


@pytest.fixture(scope="module")
def gf_env(site):
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    return GridEvacEnv(rasterize_floor(building, gf, H_COARSE), floor=gf)


def test_no_corner_cutting_between_diagonal_walls():
    env = _toy([
        "...",
        ".#.",
        "#..",
        "X..",
    ])
    # cell (row=2, col=0) in array terms is '.' left of the '#'; NE of it is (3,1),
    # but its E neighbour (2,1) is a wall, so the diagonal must be blocked.
    s = env.cell_state(2, 0)
    ne = MOVES.index((1, 1))
    assert not env.valid[s, ne]
    assert env.next_state[s, ne] == s


def test_rewards_are_negative_seconds():
    env = _toy(["X..."])
    s = env.cell_state(0, 2)
    w = MOVES.index((0, -1))
    assert env.valid[s, w]
    assert env.reward[s, w] == pytest.approx(-0.5 / env.v0)


def test_optimal_times_matches_hand_computed_corridor():
    env = _toy(["X....."])
    t = optimal_times(env)
    for col in range(6):
        assert t[env.cell_state(0, col)] == pytest.approx(col * 0.5 / env.v0)


def test_q_learning_converges_on_toy_room():
    env = _toy([
        "########",
        "#......#",
        "#..##..#",
        "#......X",
        "########",
    ])
    t_opt = optimal_times(env)
    Q, log = train(env, QConfig(episodes=3_000, seed=0))
    summary = evaluate_all_states(env, Q, t_opt)
    assert summary.success_rate == 1.0
    assert summary.optimal_rate == 1.0
    assert log.reached_exit[-1]


def test_gf_every_walkable_cell_has_a_route(gf_env):
    t = optimal_times(gf_env)
    assert np.isfinite(t).all()
    exits = {gf_env.grid.door_ids[gf_env.grid.exit_id[r, c]] for r, c in gf_env.cells[gf_env.terminal]}
    assert exits == {"X-S", "X-N", "X-W", "X-E", "X-NW"}  # X-SE is locked by default


def test_overlay_can_unlock_a_default_locked_exit(site):
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    grid = rasterize_floor(building, gf, H_COARSE)
    locked = GridEvacEnv(grid, floor=gf)
    unlocked = GridEvacEnv(grid, floor=gf, overlay=DynamicOverlay(door_states={"X-SE": "AVAILABLE"}))
    assert unlocked.terminal.sum() > locked.terminal.sum()


def test_gf_optimum_is_within_octile_bound_of_eikonal(site, gf_env):
    """An 8-connected path can be at most 1/cos(22.5 deg) longer than the
    any-angle Eikonal one, so grid optimum must sit in [~T, 1.082 T]. Slack
    is absolute (1 m), not relative: the two disagree by up to ~1 cell-pair
    near the exits (where the no-corner-cutting rule and FMM seeding differ),
    which is a big *ratio* only on short trips (observed: LOBBY by X-S)."""
    from dms.fields.eikonal import FieldCache
    from dms.grid.layers import U_MAX_DEFAULT

    building = site.get_building("B1")
    gf = building.get_floor("GF")
    eik = FieldCache().get_gf(building, gf, gf_env.grid)
    t_grid = optimal_times(gf_env) * gf_env.v0  # metres
    rr, cc = gf_env.cells[:, 0], gf_env.cells[:, 1]
    t_eik = np.asarray(eik.T.data)[rr, cc] * U_MAX_DEFAULT
    far = t_eik > 5.0
    octile = 1 / math.cos(math.radians(22.5))
    assert (t_grid[far] - octile * t_eik[far]).max() < 1.0
    assert (t_grid[far] / t_eik[far]).min() > 0.95


def test_gf_short_training_evacuates_demo_rooms(site, gf_env):
    gf = site.get_building("B1").get_floor("GF")
    Q, _ = train(gf_env, QConfig(episodes=30_000, seed=0))
    t_opt = optimal_times(gf_env)
    for room in ("G-01", "G-11", "LOBBY"):
        s = gf_env.nearest_state(*gf.spaces[room].rect.center)
        ro = greedy_rollout(gf_env, Q, s)
        assert ro.reached_exit, room
        assert ro.time_s <= t_opt[s] * 1.10, room


# ------------------------------------------------------------- two floors + stairs


@pytest.fixture(scope="module")
def building_env(site):
    return GridEvacEnv.for_building(site.get_building("B1"))


def test_building_only_gf_has_exits_and_every_ff_cell_gets_out(building_env):
    env = building_env
    assert set(env.cell_floor[env.terminal].tolist()) == {env.floor_ids.index("GF")}
    t = optimal_times(env)
    assert np.isfinite(t).all()  # FF can only get out by stairs, and always can


def test_stairs_action_links_same_cell_on_other_floor(site, building_env):
    env = building_env
    stair = site.get_building("B1").stairs["ST-E"]
    s = env.nearest_state(*stair.footprint_gf.center, "GF")
    s2 = int(env.next_state[s, STAIRS])
    assert env.valid[s, STAIRS]
    assert env.state_floor(s2) == "FF"
    assert tuple(env.cells[s2]) == tuple(env.cells[s])
    assert env.reward[s, STAIRS] == pytest.approx(-stair.walking_length_m / (stair.chi * env.v0))
    assert int(env.next_state[s2, STAIRS]) == s  # and back up/down again


def test_stairs_invalid_outside_stairwells(site, building_env):
    env = building_env
    room = site.get_building("B1").get_floor("FF").spaces["F-07"]
    assert not env.valid[env.nearest_state(*room.rect.center, "FF"), STAIRS]


def test_closing_a_stair_never_makes_ff_faster(site, building_env):
    closed = GridEvacEnv.for_building(site.get_building("B1"), closed_stairs=["ST-C"])
    t_all, t_closed = optimal_times(building_env), optimal_times(closed)
    ff = building_env.cell_floor == building_env.floor_ids.index("FF")
    assert (t_closed[ff] >= t_all[ff] - 1e-9).all()
    assert (t_closed[ff] > t_all[ff] + 1e-9).any()
    assert "ST-C" not in closed.stair_cost_s
    assert closed.valid[:, STAIRS].sum() < building_env.valid[:, STAIRS].sum()


def test_q_learning_takes_ff_agents_downstairs(site, building_env):
    env = building_env
    ff = site.get_building("B1").get_floor("FF")
    Q, _ = train(env, QConfig(episodes=80_000, seed=0))
    t_opt = optimal_times(env)
    for room in ("F-01", "F-07", "F-10"):
        s = env.nearest_state(*ff.spaces[room].rect.center, "FF")
        ro = greedy_rollout(env, Q, s)
        assert ro.reached_exit, room
        assert env.state_floor(ro.states[-1]) == "GF", room
        assert ro.time_s <= t_opt[s] * 1.10, room
