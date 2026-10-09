"""DQN route policy over the navigation graph (GraphEvacEnv).

The tabular learners (q_learning.py, graph_q.py) share one Q-table: state =
node index, so every agent votes with the same policy. Here a single small MLP
computes Q over the same <=4 neighbour actions, but the input is the graph node
*plus* the agent's own attribute features (dms/rl/agent_profile.py) — one
shared set of weights whose output is conditioned on who is asking, so
heterogeneous agents can diverge at the same junction without an N-way split of
the experience.

Replay buffer + target network stabilise the update; training uses the exact
GraphEvacEnv transitions (reward = -walk seconds, done at an exit node), so a
trained DQN is directly comparable to the tabular run on the same references
(cell optimum / graph+walker optimum). Inference stays argmax over neighbours,
so greedy rollouts and evaluation consume a network exactly as they do a table.

This first version trains on the current physics, where attributes only scale
walk times; routes therefore converge to the same optimum as the table — the
architecture is the change, ready for the crowd-density and smoke inputs that
make attributes alter which route is best.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from .agent_profile import profile_features, sample_population
from .evaluate import optimal_times
from .graph_env import GraphEvacEnv
from .graph_q import GraphEvalSummary, GraphRollout


@dataclass(slots=True)
class DQNConfig:
    episodes: int = 4_000
    hidden: int = 64
    lr: float = 1e-3
    gamma: float = 1.0
    eps_start: float = 0.5
    eps_end: float = 0.03
    eps_decay_frac: float = 0.8
    explore_familiar: float = 0.5   # fraction of random steps that head toward an exit
    buffer_size: int = 100_000
    batch_size: int = 256
    train_start: int = 1_000      # transitions before gradient steps begin
    train_every: int = 8           # gradient step frequency (per N transitions)
    target_sync: int = 1_000      # hard-copy online -> target every N steps
    max_hops: int = 200
    seed: int = 0
    device: str = "auto"


@dataclass(slots=True)
class DQNTrainLog:
    episode_time: list[float] = field(default_factory=list)
    episode_hops: list[int] = field(default_factory=list)
    reached_exit: list[bool] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)


class QNetwork(nn.Module):
    """Q(node, agent-attributes) over node's neighbour slots.

    state = [one-hot of the graph node | normalised attribute features].
    Slots the node doesn't have (it owns fewer than max_deg neighbours) are
    masked to -inf by the caller — at selection time and in the target too, so
    a target never uses an unrealisable action.
    """

    def __init__(self, n_nodes: int, n_attr: int, n_actions: int, hidden: int = 64):
        super().__init__()
        self.n_nodes = n_nodes
        self.net = nn.Sequential(
            nn.Linear(n_nodes + n_attr, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, node: torch.Tensor, attr: torch.Tensor) -> torch.Tensor:
        if node.dim() == 0:
            node = node.unsqueeze(0)
        if attr.dim() == 1:
            attr = attr.unsqueeze(0)
        node_onehot = nn.functional.one_hot(node, num_classes=self.n_nodes).to(attr.dtype)
        return self.net(torch.cat([node_onehot, attr], dim=1))


class ReplayBuffer:
    def __init__(self, capacity: int, rng: np.random.Generator):
        self.capacity = capacity
        self.rng = rng
        self.size = 0
        self.pos = 0
        self._node = np.empty(capacity, dtype=np.int64)
        self._next = np.empty(capacity, dtype=np.int64)
        self._action = np.empty(capacity, dtype=np.int64)
        self._reward = np.empty(capacity, dtype=np.float32)
        self._done = np.empty(capacity, dtype=bool)
        self._attr = [None] * capacity

    def push(self, node: int, attr: np.ndarray, action: int, reward: float, next_node: int, done: bool) -> None:
        self._node[self.pos] = node
        self._next[self.pos] = next_node
        self._action[self.pos] = action
        self._reward[self.pos] = reward
        self._done[self.pos] = done
        self._attr[self.pos] = attr
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        idx = self.rng.integers(0, self.size, size=n)
        return (self._node[idx], np.stack([self._attr[i] for i in idx]), self._action[idx],
                self._reward[idx], self._next[idx], self._done[idx])


class DQN:
    """Online + target networks and the replay buffer; exposes training, greedy
    action selection, and policy evaluation comparable to the tabular leader."""

    def __init__(self, env: GraphEvacEnv, cfg: DQNConfig = DQNConfig()):
        self.env = env
        self.cfg = cfg
        self.device = torch.device("cuda" if (cfg.device == "auto" and torch.cuda.is_available())
                                   else torch.device(cfg.device))
        torch.manual_seed(cfg.seed)
        self.rng = np.random.default_rng(cfg.seed)
        self.n_attr = int(profile_features(sample_population(1, np.random.default_rng(0))[0]).size)
        self.online = QNetwork(env.n_nodes, self.n_attr, env.max_deg, cfg.hidden).to(self.device)
        self.target = QNetwork(env.n_nodes, self.n_attr, env.max_deg, cfg.hidden).to(self.device)
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.buffer = ReplayBuffer(cfg.buffer_size, self.rng)
        self.valid_slots = [np.flatnonzero(env.valid[u]).tolist() for u in range(env.n_nodes)]
        self.valid_t = torch.tensor(env.valid, device=self.device)
        self.familiar_slots = self._familiarity(env)
        self.steps = 0

    def _familiarity(self, env: GraphEvacEnv) -> np.ndarray:
        """Familiar guidance: for each node, the neighbour whose cells are
        closest to an exit in cell-optimum time. Used to seed exploration with
        route knowledge (docs v0.2: staff know the plant, visitors don't) so
        early episodes reach exits and Q-learning gets a terminal signal —
        the neural analogue of the tabular learner's optimistic init."""
        t = optimal_times(env.cell_env)
        slots = np.zeros(env.n_nodes, dtype=np.int64)
        for u in range(env.n_nodes):
            k = np.flatnonzero(env.valid[u])
            gains = [t[env.cells_of_node[int(env.neighbours[u, int(kk)])]].min() for kk in k]
            slots[u] = int(k[int(np.argmin(gains))])
        return slots

    # ------------------------------------------------------------- utilities

    def _feats(self) -> np.ndarray:
        return profile_features(sample_population(1, self.rng)[0])

    def _q_valid(self, net: nn.Module, nodes: np.ndarray, attr: np.ndarray) -> torch.Tensor:
        node_t = torch.tensor(nodes, device=self.device)
        attr_t = torch.tensor(np.asarray(attr, dtype=np.float32), device=self.device)
        out = net(node_t, attr_t)
        return out.where(torch.tensor(self.env.valid[nodes], device=self.device),
                         torch.full_like(out, -1e18))

    @torch.no_grad()
    def q_values(self, node: int, attr: np.ndarray) -> np.ndarray:
        """Q over neighbour slots for one (node, agent) decision, on CPU."""
        return self._q_valid(self.online, np.asarray([node]), np.asarray(attr)[None]).cpu().numpy()[0]

    @torch.no_grad()
    def q_values_all(self, attr: np.ndarray, value: bool = True) -> np.ndarray:
        """Q per (node, action) for a fixed agent across every node, on CPU.
        `value=True` returns V = -max over valid actions (best time to exit)."""
        nodes = np.arange(self.env.n_nodes)
        q = self._q_valid(self.online, nodes, np.tile(np.asarray(attr, dtype=np.float32), (self.env.n_nodes, 1)))
        q = q.cpu().numpy()
        return -q.max(axis=1) if value else q

    @torch.no_grad()
    def best_action(self, node: int, attr: np.ndarray) -> int:
        return int(np.argmax(self.q_values(node, attr)))

    # ---------------------------------------------------------------- policy

    @torch.no_grad()
    def greedy_rollout(self, start: int, attr: np.ndarray, *, with_cells: bool = False,
                       max_hops: int | None = None) -> GraphRollout:
        max_hops = max_hops or self.cfg.max_hops
        s, t = start, 0.0
        nodes = [self.env.node(s)]
        cells = [s]
        seen = {nodes[0]}
        for _ in range(max_hops):
            u = self.env.node(s)
            if self.env.terminal[u]:
                return GraphRollout(nodes, cells, t, True)
            a = self.best_action(u, attr)
            v = int(self.env.neighbours[u, a])
            if with_cells:
                cells.extend(self.env.walker.path(u, v, s)[1:])
            s, r, _done = self.env.step(s, a)
            t -= r
            if v in seen:
                return GraphRollout(nodes + [v], cells, t, False, "loop")
            seen.add(v)
            nodes.append(v)
        return GraphRollout(nodes, cells, t, bool(self.env.terminal[self.env.node(s)]), "max_hops")

    def evaluate(self, t_cell_opt: np.ndarray, attr: np.ndarray) -> GraphEvalSummary:
        starts = [int(s) for s in self.env.start_states if np.isfinite(t_cell_opt[s]) and t_cell_opt[s] > 0]
        excess = []
        for s in starts:
            ro = self.greedy_rollout(s, attr)
            if ro.reached_exit:
                excess.append(ro.time_s / t_cell_opt[s] - 1.0)
        ex = np.asarray(excess)
        return GraphEvalSummary(
            n_starts=len(starts),
            success_rate=len(ex) / len(starts),
            within_5pct=float((ex <= 0.05).sum()) / len(starts),
            mean_excess=float(ex.mean()) if ex.size else float("nan"),
            p90_excess=float(np.percentile(ex, 90)) if ex.size else float("nan"),
            max_excess=float(ex.max()) if ex.size else float("nan"),
        )

    # --------------------------------------------------------------- training

    def _optimize(self) -> float | None:
        if self.buffer.size < self.cfg.train_start:
            return None
        node, attr, action, reward, next_node, done = self.buffer.sample(self.cfg.batch_size)
        q_s = self._q_valid(self.online, node, attr)
        picked = q_s.gather(1, torch.tensor(action, device=self.device)[:, None]).squeeze(1)
        with torch.no_grad():
            q_next_sel = self._q_valid(self.online, next_node, attr)      # double-Q: select with online
            q_next_tgt = self._q_valid(self.target, next_node, attr)      #           evaluate with target
            a_best = q_next_sel.argmax(1)
            target = q_next_tgt.gather(1, a_best[:, None]).squeeze(1)
            targets = (torch.tensor(reward, device=self.device)
                       + self.cfg.gamma * target * (~torch.tensor(done, device=self.device)))
        loss = nn.functional.smooth_l1_loss(picked, targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu())

    def train(self, *, progress_every: int = 0) -> DQNTrainLog:
        cfg = self.cfg
        log = DQNTrainLog()
        span = max(1, int(cfg.episodes * cfg.eps_decay_frac))
        for ep in range(cfg.episodes):
            eps = cfg.eps_start + min(1.0, ep / span) * (cfg.eps_end - cfg.eps_start)
            feats_np = self._feats()
            feats_t = torch.tensor(feats_np, device=self.device)
            s = int(self.rng.choice(self.env.start_states))
            t, hops = 0.0, 0
            node = torch.ones((1,), dtype=torch.long, device=self.device)
            while hops < cfg.max_hops:
                u = self.env.node(s)
                if self.env.terminal[u]:
                    break
                if self.rng.random() < eps:
                    if self.rng.random() < cfg.explore_familiar:
                        a = int(self.familiar_slots[u])
                    else:
                        a = self.valid_slots[u][int(self.rng.integers(len(self.valid_slots[u])))]
                else:
                    node[0] = u
                    out = self.online(node, feats_t)
                    out = out.where(self.valid_t[u], torch.full_like(out, -1e18))
                    a = int(out.argmax().item())
                s2, r, done = self.env.step(s, a)
                self.buffer.push(u, feats_np, a, r, self.env.node(s2), done)
                self.steps += 1
                if self.steps % cfg.target_sync == 0:
                    self.target.load_state_dict(self.online.state_dict())
                if self.steps % cfg.train_every == 0:
                    loss = self._optimize()
                    if loss is not None:
                        log.losses.append(loss)
                t -= r
                hops += 1
                s = s2
                if done:
                    break
            log.episode_time.append(t)
            log.episode_hops.append(hops)
            log.reached_exit.append(done)
            if progress_every and (ep + 1) % progress_every == 0:
                w = slice(-progress_every, None)
                losses = log.losses[-progress_every:] if log.losses else [0.0]
                print(f"  ep {ep + 1:6d}  eps={eps:.3f}  success={np.mean(log.reached_exit[w]):6.1%}  "
                      f"mean hops={np.mean(log.episode_hops[w]):5.1f}  mean time={np.mean(log.episode_time[w]):6.1f}s  "
                      f"loss={np.mean(losses):.4f}")
        return log


def calibration_features(seed: int = 0) -> np.ndarray:
    """Deterministic reference agent for policy evaluation."""
    return profile_features(sample_population(1, np.random.default_rng(seed))[0])