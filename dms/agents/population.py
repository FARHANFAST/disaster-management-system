"""Population sampling, split as the supervisor asked: first by age, then
by sex, then by seniority (role) and VIP status. Every behavioural attribute
is derived from that path (design doc v0.3 §3):

    speed       = U(range[age, sex]) x mobility(age)                 (capped 2.0 m/s)
    reaction    = U(range[age]) x role x sex x VIP x (1 - 0.3 exp) x LogNormal(0, s)
    familiarity = U(range[role])
    experience  = min(1, Poisson(drills[role]) / 5)
    vision      = by age (15 m for responders)
    persona     ~ P(persona | role), Elderly shifted toward Cautious; responders fixed
    W1..W3 base = softmax(log W_persona + attribute modifiers)   (see weights.py)
    known exits = main entrance + each other exit with p = familiarity + 0.15 exp
    susceptibility = health factor x (1.3 if Elderly)

Sampling is seeded, so a seed reproduces the same population exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import AGE_GROUPS, ROLES, PopulationConfig

MOBILITY = ("full", "reduced", "assisted")
_HEALTH0 = ("Healthy", "Irritated", "Distressed")


@dataclass(slots=True)
class Profile:
    agent_id: int
    age: str
    sex: str
    role: str
    vip: bool
    responder: bool
    persona: str
    mobility: str
    speed: float  # free walking speed, m/s (already x mobility)
    reaction_s: float
    familiarity: float
    drills: int
    experience: float  # 0..1
    vision_m: float
    panic_threshold: float
    health0: str
    susceptibility: float
    known_exits: frozenset[str]  # exit ref ids this person knows before the event
    escort: int | None = None  # VIP -> agent_id of its escorting responder

    @property
    def group(self) -> str:
        return f"{self.age}/{self.sex}/{'VIP' if self.vip else self.role}"


def _choice(rng: np.random.Generator, options, p) -> str:
    return str(options[int(rng.choice(len(options), p=np.asarray(p) / np.sum(p)))])


def sample_population(cfg: PopulationConfig, exits: list[str], rng: np.random.Generator) -> list[Profile]:
    """`exits`: ref ids of every usable exit of the building."""
    n = cfg.n_agents
    # level 1 -> 3 of the split
    ages = [_choice(rng, AGE_GROUPS, cfg.age_p) for _ in range(n)]
    sexes = ["M" if rng.random() < cfg.male_p else "F" for _ in range(n)]
    roles = [_choice(rng, ROLES, cfg.role_p) for _ in range(n)]
    visitors = [i for i, r in enumerate(roles) if r == "Visitor"]
    n_vip = min(len(visitors), int(round(cfg.vip_p * n)))
    vips = set(int(i) for i in rng.choice(visitors, size=n_vip, replace=False)) if n_vip else set()
    staff = [i for i, r in enumerate(roles) if r in ("Senior staff", "Mid staff") and ages[i] != "Elderly"]
    n_resp = min(len(staff), max(1 if vips else 0, int(round(cfg.responder_p * n))))
    responders = set(int(i) for i in rng.choice(staff, size=n_resp, replace=False)) if n_resp else set()

    out: list[Profile] = []
    for i in range(n):
        age, sex, role = ages[i], sexes[i], roles[i]
        vip, resp = i in vips, i in responders
        lo, hi = cfg.speed_range[(age, sex)]
        mobility = _choice(rng, MOBILITY, cfg.mobility_p[age])
        speed = min(float(rng.uniform(lo, hi)) * cfg.mobility_factor[MOBILITY.index(mobility)], 2.0)
        drills = int(rng.poisson(cfg.drills_mean[role] + (2.0 if resp else 0.0)))
        experience = min(1.0, drills / 5.0)
        reaction = (float(rng.uniform(*cfg.reaction_range[age])) * cfg.role_reaction[role] * cfg.sex_reaction[sex]
                    * (cfg.vip_reaction if vip else 1.0) * (1.0 - cfg.experience_reaction * experience)
                    * float(rng.lognormal(0.0, cfg.reaction_sigma)))
        familiarity = float(rng.uniform(*cfg.familiarity[role]))
        if resp:
            persona = "Responder"
        else:
            p = np.asarray(cfg.persona_p[role], dtype=float)
            if age == "Elderly":
                p = p + np.array([-cfg.elderly_cautious_shift / 2, -cfg.elderly_cautious_shift / 2,
                                  cfg.elderly_cautious_shift])
            persona = _choice(rng, ("Rationalist", "Follower", "Cautious"), np.clip(p, 0.01, None))
        health0 = _choice(rng, _HEALTH0, cfg.health_p[age])
        sus = cfg.health_susceptibility[health0] * (cfg.elderly_susceptibility if age == "Elderly" else 1.0)
        if resp:
            known = frozenset(exits)
        else:
            p_know = 0.0 if role == "Visitor" else min(1.0, familiarity + cfg.exit_knowledge_experience * experience)
            known = frozenset([cfg.main_exit] + [e for e in exits if e != cfg.main_exit and rng.random() < p_know])
        out.append(Profile(
            agent_id=i, age=age, sex=sex, role=role, vip=vip, responder=resp, persona=persona,
            mobility=mobility, speed=speed, reaction_s=reaction, familiarity=familiarity, drills=drills,
            experience=experience, vision_m=cfg.responder_vision_m if resp else cfg.vision_m[age],
            panic_threshold=float(rng.uniform(*cfg.panic_threshold.get(persona, (0.6, 0.8)))),
            health0=health0, susceptibility=sus, known_exits=known))
    resp_ids = sorted(responders)
    for k, i in enumerate(sorted(vips)):
        out[i].escort = resp_ids[k % len(resp_ids)] if resp_ids else None
    return out


def counts(pop: list[Profile], key: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in pop:
        v = getattr(p, key) if key != "vip" else ("VIP" if p.vip else "-")
        out[str(v)] = out.get(str(v), 0) + 1
    return out
