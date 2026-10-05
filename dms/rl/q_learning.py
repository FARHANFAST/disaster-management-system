"""Tabular Q-learning (Watkins) for GridEvacEnv.

    Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s',a') - Q(s,a) ]

Exploration: epsilon-greedy with a linearly decaying epsilon, plus
*optimistic initialisation* — every true Q value is negative (each move
costs time), so a table initialised at ~0 makes untried actions look better
than tried ones and the agent sweeps the floor systematically instead of
relying on random dithering alone. A tiny random offset breaks argmax ties
so the agent isn't biased toward action 0 ("N") early on. Invalid moves
(into a wall) are fixed at -inf, so neither the greedy choice nor a random
exploratory one ever picks them.

gamma defaults to 1.0: the task is episodic and rewards are -seconds, so
max_a Q(s,a) converges to -(shortest evacuation time from s).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .env import N_ACTIONS, GridEvacEnv


@dataclass(slots=True)
class QConfig:
    episodes: int = 100_000
    # 1.0 is exact for a deterministic env (each backup replaces the estimate
    # with the true one-step target). Drop to ~0.1-0.5 once transitions become
    # stochastic (crowds, hazards), or Q will chase noise.
    alpha: float = 1.0
    gamma: float = 1.0
    eps_start: float = 0.3
    eps_end: float = 0.01
    eps_decay_frac: float = 0.8  # fraction of episodes over which epsilon decays linearly
    max_steps: int = 2_000
    seed: int = 0


@dataclass(slots=True)
class TrainLog:
    episode_return: list[float] = field(default_factory=list)
    episode_steps: list[int] = field(default_factory=list)
    reached_exit: list[bool] = field(default_factory=list)


def epsilon_at(cfg: QConfig, episode: int) -> float:
    span = max(1, int(cfg.episodes * cfg.eps_decay_frac))
    frac = min(1.0, episode / span)
    return cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start)


def train(env: GridEvacEnv, cfg: QConfig = QConfig(), *, Q: np.ndarray | None = None,
          progress_every: int = 0) -> tuple[np.ndarray, TrainLog]:
    rng = np.random.default_rng(cfg.seed)
    if Q is None:
        Q = -rng.uniform(0.0, 1e-6, size=(env.n_states, N_ACTIONS))
    Q[~env.valid] = -np.inf
    Q[env.terminal] = 0.0  # never updated, but keeps V(exit) = 0 for evaluation

    # Plain-Python views of the transition tables: per-step numpy scalar
    # indexing is the bottleneck, and lists are several times faster here.
    nxt = env.next_state.tolist()
    rew = env.reward.tolist()
    term = env.terminal.tolist()
    valid_actions = [np.flatnonzero(row).tolist() for row in env.valid]
    log = TrainLog()

    for ep in range(cfg.episodes):
        eps = epsilon_at(cfg, ep)
        s = env.reset(rng)
        ret, steps, done = 0.0, 0, False
        randoms = rng.random(cfg.max_steps)
        rand_picks = rng.integers(0, 1 << 30, cfg.max_steps)
        while steps < cfg.max_steps:
            q_s = Q[s]
            if randoms[steps] < eps:
                choices = valid_actions[s]
                a = choices[int(rand_picks[steps]) % len(choices)]
            else:
                a = int(q_s.argmax())
            s2, r = nxt[s][a], rew[s][a]
            done = term[s2]
            target = r if done else r + cfg.gamma * Q[s2].max()
            q_s[a] += cfg.alpha * (target - q_s[a])
            ret += r
            steps += 1
            s = s2
            if done:
                break
        log.episode_return.append(ret)
        log.episode_steps.append(steps)
        log.reached_exit.append(done)
        if progress_every and (ep + 1) % progress_every == 0:
            window = slice(-progress_every, None)
            print(f"  ep {ep + 1:6d}  eps={eps:.3f}  "
                  f"success={np.mean(log.reached_exit[window]):6.1%}  "
                  f"mean steps={np.mean(log.episode_steps[window]):7.1f}  "
                  f"mean return={np.mean(log.episode_return[window]):8.2f}s")
    return Q, log


@dataclass(slots=True)
class Rollout:
    states: list[int]
    time_s: float
    reached_exit: bool
    reason: str = ""


def greedy_rollout(env: GridEvacEnv, Q: np.ndarray, start: int, *, max_steps: int = 2_000) -> Rollout:
    """Follow argmax_a Q from `start` with no exploration. Stops at an exit,
    on revisiting a state (a deterministic greedy policy that revisits a
    state is stuck in a loop forever), or after max_steps."""
    s, t = start, 0.0
    states = [s]
    seen = {s}
    for _ in range(max_steps):
        if env.terminal[s]:
            return Rollout(states, t, True)
        a = int(Q[s].argmax())
        t -= env.reward[s, a]
        s = int(env.next_state[s, a])
        if s in seen:
            return Rollout(states, t, False, "loop")
        seen.add(s)
        states.append(s)
    return Rollout(states, t, bool(env.terminal[s]), "" if env.terminal[s] else "max_steps")
