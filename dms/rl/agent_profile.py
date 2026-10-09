"""Agent demographics and attribute sampling, shared by the live page and the
DQN trainer.

Every number here is a tunable sampling parameter (docs: Agent Data & Decision
Model v0.2, Data Dictionary Table 6/7, Li 2022 Table 3). Demographics (gender /
age / role+seniority) drive the derived physiological attributes. Currently only
the movement attributes — base speed, mobility factor, reaction time and the
demographics they derive from — change the simulation; the rest are carried on
every agent (and exposed to the DQN as state features) ready for the crowd /
smoke layers that make them matter.
"""

from __future__ import annotations

import numpy as np

from .env import V0

_GENDER_OPT = ("M", "F"), (0.60, 0.40)
_AGE_OPT = ("Young", "Adult", "Elderly"), (0.20, 0.55, 0.25)
_ROLE_OPT = ("Senior staff", "Mid staff", "Junior staff", "Visitor", "Contractor"), (0.25, 0.30, 0.20, 0.15, 0.10)

_SPEED_RANGE = {  # (age, gender) -> (lo, hi) walking speed, m/s
    ("Young", "M"): (1.30, 1.60), ("Young", "F"): (1.20, 1.45),
    ("Adult", "M"): (1.15, 1.35), ("Adult", "F"): (1.00, 1.25),
    ("Elderly", "M"): (0.80, 1.05), ("Elderly", "F"): (0.75, 1.00),
}
_MOBILITY_P = {"Young": (0.97, 0.03, 0.00), "Adult": (0.92, 0.07, 0.01), "Elderly": (0.70, 0.25, 0.05)}
_MOBILITY_FACTOR = {"full": 1.0, "reduced": 0.7, "assisted": 0.5}
_REACTION_RANGE_S = {"Young": (5.0, 15.0), "Adult": (8.0, 20.0), "Elderly": (15.0, 40.0)}
_ROLE_REACTION = {"Senior staff": 0.7, "Mid staff": 1.0, "Junior staff": 1.2, "Visitor": 1.3, "Contractor": 1.1}
MAX_SPEED = 2.0  # m/s cap on effective walking speed
MAX_REACTION_S = max(v[1] for v in _REACTION_RANGE_S.values())  # 40 s, for feature normalisation

_PERSONA_OPT = (("Rationalist", (0.50, 0.15, 0.35)),
                ("Follower", (0.20, 0.55, 0.25)),
                ("Cautious", (0.45, 0.10, 0.45)),
                ("Responder", (0.55, 0.20, 0.25))), (0.48, 0.26, 0.21, 0.05)
_PANIC_THRESHOLD = {"Rationalist": (0.65, 0.85), "Follower": (0.50, 0.70),
                    "Cautious": (0.55, 0.75), "Responder": (0.75, 0.90)}
_FAMILIARITY = {"Senior staff": (0.82, 0.90), "Mid staff": (0.75, 0.85),
                "Junior staff": (0.65, 0.80), "Visitor": (0.15, 0.40),
                "Contractor": (0.45, 0.60)}
_HEALTH_OPT = {"Young": (0.985, 0.010, 0.005), "Adult": (0.970, 0.020, 0.010),
               "Elderly": (0.920, 0.050, 0.030)}, ("Healthy", "Irritated", "Distressed")


def sample_population(n: int, rng: np.random.Generator) -> list[dict]:
    """Sample `n` agents' full attribute set. Demographic and cognitive
    distributions are designed here; mobility, walking speed and reaction time
    drive the simulation, everything else is carried along for the future
    Agent/Decision module (docs v0.2, Data Dictionary Table 6)."""
    genders, gp = _GENDER_OPT
    ages, ap = _AGE_OPT
    roles, rp = _ROLE_OPT
    personas, pp = _PERSONA_OPT
    health_p, health_states = _HEALTH_OPT
    pop = []
    for i, g in enumerate(rng.choice(genders, n, p=gp)):
        age = str(rng.choice(ages, p=ap))
        role = str(rng.choice(roles, p=rp))
        persona, (w1, w2, w3) = personas[int(rng.choice(len(personas), p=pp))]
        health = str(rng.choice(health_states, p=health_p[age]))
        lo, hi = _SPEED_RANGE[(age, str(g))]
        base = float(rng.uniform(lo, hi))
        mobility = str(rng.choice(tuple(_MOBILITY_FACTOR), p=_MOBILITY_P[age]))
        reaction = float(rng.uniform(*_REACTION_RANGE_S[age]) * _ROLE_REACTION[role])
        speed = min(base * _MOBILITY_FACTOR[mobility], MAX_SPEED)
        pop.append(dict(
            agentId=f"AG-{i + 1:03d}",
            gender=str(g), age=age, role=role,
            # cognitive profile (inert)
            persona=persona, personaW1=w1, personaW2=w2, personaW3=w3,
            panicThreshold=round(float(rng.uniform(*_PANIC_THRESHOLD[persona])), 2),
            familiarity=round(float(rng.uniform(*_FAMILIARITY[role])), 2),
            sensingRadius=15.0 if persona == "Responder" else 5.0,
            isResponder=persona == "Responder",
            # physiological state, initial (inert)
            healthStatus=health, stressLevel=round(float(rng.uniform(0.0, 0.15)), 2),
            fatigueLevel=round(float(rng.uniform(0.0, 0.05)), 2), panicLevel=0.0,
            toxicDose=round(float(rng.uniform(0.0, 0.5)), 2) if health != "Healthy" else 0.0,
            peakPpm=0.0, currentState="NORMAL", assignedExit="",
            # movement (drives the simulation)
            baseSpeed=base, mobility=mobility, mobilityFactor=_MOBILITY_FACTOR[mobility],
            reaction_s=reaction, speed=speed, factor=V0 / speed,
        ))
    return pop


# --------------------------------------------------------------------------
# Numeric encoding for the DQN state. Fixed category order so the mapping is
# deterministic and reproducible across runs.

_GENDER_ORDER = ("M", "F")
_AGE_ORDER = ("Young", "Adult", "Elderly")
_ROLE_ORDER = ("Senior staff", "Mid staff", "Junior staff", "Visitor", "Contractor")
_HEALTH_ORDER = ("Healthy", "Irritated", "Distressed")


def _onehot(value: str, order: tuple[str, ...]) -> list[float]:
    out = [0.0] * len(order)
    out[order.index(value)] = 1.0
    return out


def profile_features(p: dict) -> np.ndarray:
    """Fixed-order, normalised feature vector describing agent `p` for the DQN.

    Movement attributes (baseSpeed, mobilityFactor, reaction) already change
    the world; the cognitive/physiological fields are carried as inputs so the
    network can condition on them once crowd-density and smoke inputs arrive.
    The last index is the raw categorical-ordered vector — do not reorder.
    """
    return np.asarray([
        *[p[k] for k in ("personaW1", "personaW2", "personaW3")],
        p["panicThreshold"], p["familiarity"],
        p["reaction_s"] / MAX_REACTION_S,
        p["baseSpeed"] / MAX_SPEED,
        p["mobilityFactor"],
        p["stressLevel"], p["fatigueLevel"],
        min(p["toxicDose"] / 0.5, 1.0),
        *_onehot(p["gender"], _GENDER_ORDER),
        *_onehot(p["age"], _AGE_ORDER),
        *_onehot(p["role"], _ROLE_ORDER),
        *_onehot(p["healthStatus"], _HEALTH_ORDER),
        float(p["isResponder"]), p["sensingRadius"] / 15.0,
    ], dtype=np.float32)