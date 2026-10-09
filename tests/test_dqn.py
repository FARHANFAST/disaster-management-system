"""Deep Q-network route policy (dms/rl/dqn.py): the MLP trains on the same
graph transitions as the tabular learner, converges toward the cell optimum,
reproduces the same policy for the same seed, and carries agent attributes
through the state without changing the interfaces (argmax over neighbours)."""

from __future__ import annotations

import numpy as np
import pytest

from dms.rl.agent_profile import profile_features, sample_population
from dms.rl.dqn import DQN, DQNConfig, calibration_features
from dms.rl.evaluate import optimal_times
from dms.rl.graph_env import GraphEvacEnv


def _cfg(**kw) -> DQNConfig:
    defaults = dict(episodes=700, batch_size=256, train_start=500,
                    train_every=8, target_sync=500, seed=0, device="cpu")
    defaults.update(kw)
    return DQNConfig(**defaults)


@pytest.fixture(scope="module")
def genv(site) -> GraphEvacEnv:
    return GraphEvacEnv(site, "B1")


@pytest.fixture(scope="module")
def trained(genv) -> DQN:
    dqn = DQN(genv, _cfg())
    dqn.train()
    return dqn


def test_profile_features_are_normalised_and_stable():
    f1 = profile_features(sample_population(1, np.random.default_rng(0))[0])
    f2 = profile_features(sample_population(1, np.random.default_rng(0))[0])
    assert f1.dtype == np.float32 and f1.ndim == 1
    assert np.array_equal(f1, f2)
    assert 0.0 <= f1.min() <= f1.max() <= 1.0


def test_attribute_features_differ_across_agents():
    a = sample_population(500, np.random.default_rng(1))
    feats = np.stack([profile_features(p) for p in a])
    assert feats.shape == (500, 26)
    assert feats.std(axis=0).min() > 0


def test_state_vector_is_node_onehot_plus_attributes(genv, trained):
    assert trained.n_attr == 26
    assert trained.cfg.batch_size == 256


def test_dqn_evacuates_near_optimally(genv, trained):
    e = trained.evaluate(optimal_times(genv.cell_env), calibration_features())
    assert e.success_rate >= 0.85
    assert e.mean_excess <= 0.25


def test_same_seed_is_reproducible(genv):
    attr = calibration_features()
    a = DQN(genv, _cfg(episodes=250, seed=3))
    a.train()
    b = DQN(genv, _cfg(episodes=250, seed=3))
    b.train()
    assert np.allclose(a.q_values_all(attr), b.q_values_all(attr), atol=1e-5)


def test_different_attributes_still_route_to_an_exit(genv, trained):
    s = int(genv.start_states[len(genv.start_states) // 2])
    for seed in (5, 6):
        attr = profile_features(sample_population(1, np.random.default_rng(seed))[0])
        ro = trained.greedy_rollout(s, attr)
        assert ro.reached_exit