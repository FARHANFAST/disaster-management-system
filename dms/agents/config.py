"""Every tunable number of the agent behaviour model (design doc v0.3,
docs/Agent-Behaviour-Design-v0.3.md). Nothing in dms/agents hard-codes a
value that appears here. Values marked [cal] are starting values to be
calibrated against drill data; [lit] are taken from the cited source.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ------------------------------------------------------------------ population

AGE_GROUPS = ("Young", "Adult", "Elderly")
SEXES = ("M", "F")
ROLES = ("Senior staff", "Mid staff", "Junior staff", "Contractor", "Visitor")
PERSONAS = ("Rationalist", "Follower", "Cautious", "Responder")
HEALTH = ("Healthy", "Irritated", "Distressed", "Disabled", "Incapacitated")

# persona weights (W1 knowledge, W2 social, W3 hazard); design spec v0.1 §3.4
PERSONA_W = {
    "Rationalist": (0.50, 0.20, 0.30),
    "Follower": (0.20, 0.60, 0.20),
    "Cautious": (0.25, 0.15, 0.60),
    "Responder": (0.45, 0.35, 0.20),
}
# where the knowledge weight goes when an agent's own route fails: (to W2, to W3)
FAIL_SPLIT = {"Rationalist": (0.3, 0.7), "Follower": (0.7, 0.3), "Cautious": (0.2, 0.8), "Responder": (0.5, 0.5)}


@dataclass(slots=True)
class PopulationConfig:
    n_agents: int = 550
    age_p: tuple[float, ...] = (0.20, 0.55, 0.25)
    male_p: float = 0.60
    role_p: tuple[float, ...] = (0.20, 0.30, 0.20, 0.12, 0.18)  # Senior, Mid, Junior, Contractor, Visitor
    vip_p: float = 0.03  # share of the population that is a VIP guest (drawn from Visitors) [cal]
    responder_p: float = 0.05  # share that are trained responders (drawn from Senior / Mid staff) [cal]
    # walking speed (m/s) by (age, sex): Li et al. 2022 Table 3 ranges
    speed_range: dict = field(default_factory=lambda: {
        ("Young", "M"): (1.30, 1.60), ("Young", "F"): (1.20, 1.45),
        ("Adult", "M"): (1.15, 1.35), ("Adult", "F"): (1.00, 1.25),
        ("Elderly", "M"): (0.80, 1.05), ("Elderly", "F"): (0.75, 1.00)})
    mobility_p: dict = field(default_factory=lambda: {
        "Young": (0.97, 0.03, 0.00), "Adult": (0.92, 0.07, 0.01), "Elderly": (0.70, 0.25, 0.05)})
    mobility_factor: tuple[float, ...] = (1.0, 0.7, 0.5)  # full, reduced, assisted
    # reaction time: base range by age (s) x role factor x experience factor x lognormal noise
    reaction_range: dict = field(default_factory=lambda: {
        "Young": (5.0, 15.0), "Adult": (8.0, 20.0), "Elderly": (15.0, 40.0)})
    sex_reaction: dict = field(default_factory=lambda: {"M": 1.0, "F": 1.0})  # no sourced difference: neutral
    role_reaction: dict = field(default_factory=lambda: {
        "Senior staff": 0.7, "Mid staff": 1.0, "Junior staff": 1.2, "Contractor": 1.1, "Visitor": 1.3})
    vip_reaction: float = 1.3  # waits for the escort [cal]
    experience_reaction: float = 0.3  # full drill experience cuts reaction time by 30% [cal]
    reaction_sigma: float = 0.25  # lognormal spread
    # familiarity with the building by role (uniform range)
    familiarity: dict = field(default_factory=lambda: {
        "Senior staff": (0.82, 0.90), "Mid staff": (0.75, 0.85), "Junior staff": (0.65, 0.80),
        "Contractor": (0.45, 0.60), "Visitor": (0.15, 0.40)})
    # evacuation drills attended (Poisson mean) by role; experience = min(1, drills / 5)
    drills_mean: dict = field(default_factory=lambda: {
        "Senior staff": 4.0, "Mid staff": 2.5, "Junior staff": 1.0, "Contractor": 1.0, "Visitor": 0.0})
    # vision / sensing radius (m)
    vision_m: dict = field(default_factory=lambda: {"Young": 5.0, "Adult": 5.0, "Elderly": 4.0})
    responder_vision_m: float = 15.0
    # persona mix by role, then adjusted by age (Elderly -> more Cautious)
    persona_p: dict = field(default_factory=lambda: {
        "Senior staff": (0.60, 0.15, 0.25), "Mid staff": (0.50, 0.25, 0.25),
        "Junior staff": (0.40, 0.35, 0.25), "Contractor": (0.45, 0.30, 0.25),
        "Visitor": (0.25, 0.50, 0.25)})  # Rationalist, Follower, Cautious (Responders assigned separately)
    elderly_cautious_shift: float = 0.10
    panic_threshold: dict = field(default_factory=lambda: {
        "Rationalist": (0.70, 0.85), "Follower": (0.50, 0.65), "Cautious": (0.55, 0.70), "Responder": (0.80, 0.90)})
    health_p: dict = field(default_factory=lambda: {
        "Young": (0.985, 0.010, 0.005), "Adult": (0.970, 0.020, 0.010), "Elderly": (0.920, 0.050, 0.030)})
    # gas susceptibility: health status x age
    health_susceptibility: dict = field(default_factory=lambda: {"Healthy": 1.0, "Irritated": 1.5, "Distressed": 2.0})
    elderly_susceptibility: float = 1.3
    # chance a staff member knows a non-main exit = familiarity + this x experience
    exit_knowledge_experience: float = 0.15
    main_exit: str = "X-S"  # the entrance everyone came in through


# ------------------------------------------------------------------ behaviour


@dataclass(slots=True)
class BehaviourConfig:
    dt: float = 0.5  # cognition / movement tick (s)
    adapt_weights: bool = True  # False = ablation: weights fixed at their base values
    signage: bool = True  # exit signs point low-familiarity agents to the nearest exit
    # base-weight modifiers from attributes (added to log persona weights)
    mod_fam_knowledge: float = 0.8  # z1 += a (familiarity - 0.5)
    mod_exp_knowledge: float = 0.5  # z1 += b * experience
    mod_fam_social: float = 0.8  # z2 += c (0.5 - familiarity)
    mod_vip_social: float = 0.7  # z2 += d * [VIP]
    mod_sus_hazard: float = 0.6  # z3 += e (susceptibility - 1)
    # event impulses on the logit offsets dz, which relax back with tau_recover
    fail_impulse: float = 1.0  # own route blocked or gassed: scaled by (1 - experience)
    smell_impulse: float = 0.5  # first smell of gas, or a rise >= smell_rise_ppm: scaled by susceptibility
    smell_rise_ppm: float = 10.0
    crowd_impulse: float = 0.3  # most visible people head elsewhere: scaled by (1 - familiarity)
    crowd_min_seen: int = 3
    crowd_majority: float = 0.6
    tau_recover_s: float = 30.0  # dz relaxation time when calm; stress slows recovery
    panic_shift: float = 0.2  # PANICKED moves this much weight from W1 to W2
    # decision: softmax over S(v) / T, T = t0 + t_stress * stress
    temp0: float = 0.05
    temp_stress: float = 0.15
    backtrack_penalty: float = 0.15
    hazard_ref_ppm: float = 100.0  # H(v) = 1 - min(1, C / ref)
    avoid_base_ppm: float = 5.0  # an exit counts as "bad" at C >= avoid_base / W3
    responder_avoid_ppm: float = 10.0  # responders work to the routing threshold, not to their W3
    gas_memory_s: float = 60.0  # a 'that exit is gassed' belief fades unless refreshed; blocked is kept
    # state machine
    alert_neighbours: int = 2  # NORMAL -> ALERT on this many alert neighbours (1 for Followers)
    smell_ppm: float = 0.5
    spike_ppm: float = 50.0  # gas rise within spike_window_s that triggers PANICKED
    spike_window_s: float = 5.0
    panic_neighbours: int = 3
    persona_shift_hold_s: float = 3.0  # 3 panicked neighbours for 3 s -> Follower
    persona_revert_s: float = 10.0
    panic_recover_margin: float = 0.1
    panic_recover_s: float = 5.0
    panic_speed: float = 1.2  # desired-speed boost while PANICKED (S2a impatience, capped)
    max_speed: float = 2.0
    no_progress_s: float = 15.0  # MOVING -> TRAPPED
    trapped_wait_s: float = 10.0  # TRAPPED -> DECIDING
    # stress dynamics (per second)
    stress_band_rate: tuple[float, ...] = (0.0, 0.005, 0.02, 0.08, 0.2)  # by gas band B0..B4
    stress_per_panicked: float = 0.004  # per panicked neighbour, counted up to panic_neighbours
    stress_fail: float = 0.15  # first failed route; the k-th failure adds stress_fail / k
    stress_decay: float = 0.03


@dataclass(slots=True)
class PhysiologyConfig:
    band_edges_ppm: tuple[float, ...] = (0.5, 10.0, 100.0, 500.0)
    band_speed: tuple[float, ...] = (1.0, 1.0, 0.8, 0.5, 0.0)  # supervisor brief: -20% in Distress
    haber_n: float = 1.0  # D += C^n dt / 60; brief uses n = 1
    collapse_rate: tuple[float, ...] = (0.0, 0.0, 0.001, 0.03, float("inf"))  # per s, by band B0..B4
    dose_ref: float = 100.0  # hazard rate x (1 + D / dose_ref)
    dose_limit: float = 500.0  # ppm.min: incapacitated (Haber limit)
    dose_status: tuple[float, ...] = (1.0, 20.0, 100.0)  # Irritated, Distressed, Disabled


@dataclass(slots=True)
class AgentModelConfig:
    population: PopulationConfig = field(default_factory=PopulationConfig)
    behaviour: BehaviourConfig = field(default_factory=BehaviourConfig)
    physiology: PhysiologyConfig = field(default_factory=PhysiologyConfig)
    duration_s: float = 300.0
    seed: int = 7
