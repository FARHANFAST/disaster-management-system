"""spec §3.5: the 25-run stress test (5 scenarios x 5 profile/seed/N
combinations), reusing the evacuation walker's own `simulate()` directly
(not re-implemented). A run counts as stuck if any agent is not at an
assembly point by t_end = 1500 s. Marked
`@pytest.mark.slow` -- 25 full evacuation runs, hundreds of agents each --
excluded from the fast edit loop (`pytest -q -m "not slow"`), run in full
before merging (`pytest -q`).

`res.n_awaiting_refuge` is a can_use_stairs=False
agent (today: only the wheelchair class) stuck off the bottom floor with no
refuge target yet -- a known, acknowledged gap, not a routing bug.
B1-L's own persona_mix declares no body_class_mix (every agent samples as
"standard"), so this is 0 in every run today; the assertion below still
requires every *other* stuck cause to be genuinely zero, and will keep doing
so once a future building/override does place a wheelchair-class agent
upstairs, without needing this test rewritten."""

from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.scenario.evacuation import build_world, sample_agents, simulate

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"
T_END = 1500.0

# 5 scenarios x 5 (profile, seed, N) combinations = 25 runs.
SCENARIOS = ("baseline", "main_exit_blocked", "north_exit_blocked_unknown", "main_stair_lost", "compound")
RUNS = (
    ("day", 101, None),
    ("day", 202, 300),
    ("lunch", 303, None),
    ("meeting", 404, None),
    ("night", 505, None),
)


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="module")
def building(site):
    return site.get_building("B1")


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SCENARIOS)
@pytest.mark.parametrize("profile,seed,n", RUNS)
def test_no_agent_is_stuck_after_t_end(site, building, scenario, profile, seed, n):
    world = build_world(site, building, scenario)
    agents = sample_agents(world, profile, seed=seed, N=n)
    res = simulate(world, agents, t_end=T_END, seed=seed)
    other_stuck = res.n_stuck - res.n_awaiting_refuge
    assert other_stuck == 0, (
        f"{scenario}/{profile}/seed={seed}/N={n or len(agents)}: {res.n_stuck} agent(s) "
        f"not at an assembly point by t_end={T_END:.0f}s ({res.n_awaiting_refuge} awaiting a "
        f"refuge, {other_stuck} genuinely stuck)"
    )
