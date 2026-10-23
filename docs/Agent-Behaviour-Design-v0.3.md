# Agent Data & Decision Model: Design v0.3

**Project:** AI-based Simulation Model for Disaster Management System (EVAC-SIM) · **Module owner:** Mr. Hamdan · **Status:** draft for supervisor review, with a working, tested prototype (`dms/agents/`) · **Date:** 23 October 2026

v0.3 answers the supervisor's request: split the agents by **age, then sex, then seniority and VIP status**; derive each group's behaviour (reaction time, knowledge, prior experience, vision, speed) as equations; and **refresh** each agent's weights, social force and knowledge as the situation develops, across a population of 500–600 agents. It builds on v0.1 (design specification) and v0.2 (concise design), and changes them where noted.

---

## 1. What we are building, and what we are not

The project documents are clear that this is a **pre-disaster preparedness simulator**, not a guidance system. The AI-DMS deck describes "Pre-Disaster Preparedness Simulation" built to "highlight insufficiencies and bottlenecks, warn emergency risk managers, develop evacuation strategies". The EVAC-SIM deck describes it as replacing "the single scripted drill with a repeatable, safe, computational experiment" that produces the RSET distribution against ASET.

So the agents must behave **like people, not like optimal evacuees**:
- They know part of the building, not all of it.
- They see only a few metres.
- They react late.
- They follow others.
- They change their minds.
- Sometimes they make the wrong choice.

Guidance appears only as things a planner wants to test: the alarm, signage, responders, the PA system. A learned optimal policy (the gas-aware DQN) is kept as a **benchmark** ("if everyone were perfectly guided") and as the knowledge a trained responder has. It is not used for ordinary agents.

## 2. Three levels of behaviour

Following Hoogendoorn & Bovy (2004), pedestrian behaviour has three levels. This module owns the first two.

| Level | Question | Owner |
|---|---|---|
| Strategic | Which exit or assembly point? | Agent module: known exits, beliefs about which are usable |
| Tactical | Which way through the building, next graph node? | Agent module: decision rule S(v) on the navigation graph |
| Operational | How do I walk without hitting people? | Social force model (SYSTEM_SPEC §2.1). In this prototype a grid walker stands in for it; there are no collisions yet |

Decisions are made on the **navigation graph** (176 nodes: rooms, doors, corridors, stairs, exits). The grid is used only to walk between two nodes. The agent passes three things to the social-force layer: the direction to its next node, its desired speed, and its herding weight (§10).

## 3. Population: the split

Every agent is sampled along the path **age → sex → seniority/role → VIP**. Every behavioural attribute is a function of that path plus seeded randomness, so one seed reproduces the same 550 people exactly (`dms/agents/population.py`).

| Level | Categories | Shares (default) |
|---|---|---|
| 1. Age | Young / Adult / Elderly | 20 / 55 / 25 % |
| 2. Sex | Male / Female | 60 / 40 % |
| 3. Seniority / role | Senior staff / Mid staff / Junior staff / Contractor / Visitor | 20 / 30 / 20 / 12 / 18 % |
| 4. VIP | VIP guest (drawn from Visitors), escorted by a responder | 3 % of all agents |
| (special) | Responder: trained staff (Senior/Mid, not Elderly) | 5 % of all agents |

"Local vs visitor" from the supervisor's note maps onto role: staff and contractors are locals with familiarity and drill experience; visitors and VIPs are not.

## 4. Attribute equations

`U(a,b)` is uniform, `LN(0,s)` is lognormal noise. Ranges come from Li et al. 2022 Table 3 and the v0.1 data dictionary unless marked **[cal]** (a starting value to calibrate against drill data).

**Walking speed:** `v0 = U(range[age, sex]) × mobility_factor(age)`, capped at 2.0 m/s.

| (age, sex) | M | F |
|---|---|---|
| Young | 1.30–1.60 | 1.20–1.45 |
| Adult | 1.15–1.35 | 1.00–1.25 |
| Elderly | 0.80–1.05 | 0.75–1.00 |

Mobility (full ×1.0 / reduced ×0.7 / assisted ×0.5) is drawn with age-dependent probabilities: Elderly 70 / 25 / 5 %.

**Drill experience:** `drills ~ Poisson(mean[role])`, with means Senior 4, Mid 2.5, Junior 1, Contractor 1, Visitor 0, and +2 for responders. `experience = min(1, drills / 5)`.

**Reaction time** (alarm → starts deciding):
```
t_react = U(range[age]) × role_factor × sex_factor × (1.3 if VIP) × (1 − 0.3·experience) × LN(0, 0.25)
```
Age ranges: Young 5–15 s, Adult 8–20 s, Elderly 15–40 s. Role factors: Senior 0.7, Mid 1.0, Junior 1.2, Contractor 1.1, Visitor 1.3. The sex factor is 1.0 for both, since we found no source for a difference; it is a parameter if one is supplied. VIPs wait for their escort [cal].

**Familiarity:** `U(range[role])`. Senior 0.82–0.90, Mid 0.75–0.85, Junior 0.65–0.80, Contractor 0.45–0.60, Visitor 0.15–0.40.

**Vision / sensing radius:** 5 m (Young, Adult), 4 m (Elderly) [cal], 15 m for responders.

**Gas susceptibility:** `health_factor(Healthy 1.0, Irritated 1.5, Distressed 2.0) × (1.3 if Elderly)` [cal].

**Persona:** drawn from `P(persona | role)`. Elderly agents shift 10 percentage points toward Cautious. Responders are always "Responder".

| Role | Rationalist | Follower | Cautious |
|---|---|---|---|
| Senior staff | 60 % | 15 % | 25 % |
| Mid staff | 50 % | 25 % | 25 % |
| Junior staff | 40 % | 35 % | 25 % |
| Contractor | 45 % | 30 % | 25 % |
| Visitor | 25 % | 50 % | 25 % |

**Panic threshold:** by persona (v0.1 §3.4): Rationalist 0.70–0.85, Follower 0.50–0.65, Cautious 0.55–0.70, Responder 0.80–0.90.

## 5. What an agent knows: a personal map

**Nobody plans over the whole building** (SYSTEM_SPEC §4.1, `OEK`). This is the main reason v0.2 agents all took the same path; it is fixed in v0.3.

- **Known exits at the start.** Everyone knows the main entrance X-S (they came in through it). Staff and contractors know each other exit with probability `familiarity + 0.15·experience`. Visitors and VIPs know only X-S. Responders know all exits.
- **Learning.** An exit becomes known when it comes within the vision radius, or when more than 2 visible people are heading for it.
- **Unusable exits** are discovered the same way: a locked door only when it is in sight, a gassed exit when the agent sees the gas there, or from a responder in range. "Locked" is remembered for good. "Gassed" **fades after 60 s** unless refreshed, and is cleared at once if the agent sees that exit in clean air, because gas moves.
- **When every known exit seems gassed**, agents still head for the least-bad unlocked one. They do not freeze.

## 6. Perception

Each tick (0.5 s) an agent reads:
- The air where it stands.
- The gas level at nodes within its vision radius.
- Exits within its vision radius (learnt, locked or gassed).
- The people inside its sensing bubble: their state, next node and target exit.
- The global alarm.

It cannot see anyone's persona or dose. Responders also have a **radio**: the 60 s forecast at every exit. Agents read a frozen copy of everyone's state taken at the start of the tick, so the order agents are updated in cannot change the result.

## 7. Weights that depend on who you are, and on what happens

W1 (own knowledge), W2 (other people) and W3 (hazard) are a **behaviour rule, not an optimiser**. They model how a person's trust shifts, whether or not the shift is wise. This is the supervisor's intent: "agent A is a Rationalist, finds his path closed, so his weights update and he trusts his own knowledge a little less."

```
W = softmax(z_base + dz)                                   (W1 + W2 + W3 = 1 always)

z_base,1 = ln W1_persona + 0.8 (familiarity − 0.5) + 0.5 · experience
z_base,2 = ln W2_persona + 0.8 (0.5 − familiarity) + 0.7 · [VIP]
z_base,3 = ln W3_persona + 0.6 (susceptibility − 1)

dz decays toward 0 with time constant 30 s × (1 + stress)       (people calm down; stress slows it)
```

Persona weights (v0.1 §3.4): Rationalist 0.50 / 0.20 / 0.30 · Follower 0.20 / 0.60 / 0.20 · Cautious 0.25 / 0.15 / 0.60 · Responder 0.45 / 0.35 / 0.20.

**Events** (impulses added to `dz`, all [cal]):

| Event | Change to dz | Scaled by |
|---|---|---|
| The exit I was heading for is locked or gassed | `[−1, s2, s3] × 1.0` | `(1 − 0.5·experience)`. The persona sets the split (s2, s3): Rationalist 0.3 / 0.7 (turns to the hazard), Follower 0.7 / 0.3 (turns to the crowd), Cautious 0.2 / 0.8, Responder 0.5 / 0.5. Stress also rises by `0.15 / k` for the k-th failure. |
| I smell gas, or it rises by 10 ppm or more | `+0.5` on W3 | susceptibility |
| Most of the people I can see go somewhere my knowledge would not | `+0.3` on W2 | `(1 − familiarity)` |
| I am PANICKED | move 0.2 of weight from W1 to W2 | (v0.1 §3.3) |
| 3 or more panicked neighbours for 3 s | Rationalist acts as a Follower; reverts after 10 calm seconds | Responders immune |

`--no-adapt` freezes the weights at `z_base`. This is the ablation used in §12.

## 8. Decision rule (tactical level)

At each graph node `u` the agent scores every neighbouring node `v` it believes passable:
```
S(v) = W1 K(v) + W2 F(v) + W3 H(v) − 0.15 · [v is where I just came from]

K(v) = fam · prog(D_known, v) + (1 − fam) · prog(D_sign, v)
       prog(D, v) = clip(0.5 + 0.5 (D(u) − D(v)) / t_uv, 0, 1)
       D_known = travel time to the nearest known, usable exit; D_sign = to the nearest signposted
       exit (signs do not know about gas). Responders: fam = 1.
F(v) = share of the moving people I can see whose next node is v. A VIP who can see its escort
       uses the escort's next node instead.
H(v) = 1 − min(1, C_seen(v) / 100 ppm). Gas I cannot see counts as 0. Responders use the forecast.

v ~ softmax(S / T),   T = 0.05 + 0.15 · stress
```
An exit counts as "bad" to an agent at `C ≥ 5 ppm / W3`. A cautious agent (W3 ≈ 0.6) refuses an exit at about 8 ppm; a low-W3 agent (W3 ≈ 0.2) walks through up to about 25 ppm. Responders use the routing threshold, 10 ppm. This is how the same gas splits a crowd by attributes.

## 9. State machine

| ID | From → To | Trigger |
|---|---|---|
| T1 | NORMAL → ALERT | global alarm; smells gas (≥ 0.5 ppm); ≥ 2 alert neighbours (1 for Followers); an active responder in range |
| T2 | ALERT → DECIDING | own reaction time has passed |
| T3 | DECIDING → MOVING_TO_ASSEMBLY | a next node was chosen |
| T5 | MOVING → PANICKED | stress ≥ own panic threshold, or a gas rise ≥ 50 ppm within 5 s |
| T9 | PANICKED → MOVING | stress below threshold − 0.1 for 5 s |
| T6 | MOVING / PANICKED → TRAPPED | no passable neighbour, or no movement for 15 s |
| T9 | TRAPPED → DECIDING | after 10 s |
| T7 | MOVING → AT_ASSEMBLY | finished the egress walk from the exit to the assembly point |
| T13 | any active → INCAPACITATED | collapse (§10) |

Without collisions (§2), "no movement for 15 s" almost never fires. TRAPPED-in-a-jam arrives with the crowd model.

## 10. Gas, health and the social-force coupling

**Body** (supervisor brief §3; `dms/agents/physiology.py`):
- **Dose:** `D += C^n · dt / 60` (ppm·min, Haber's rule, n = 1 as in the brief; a parameter).
- **Band** of the current air, scaled by susceptibility. Below 0.5 ppm: none. 0.5–10 Irritation; 10–100 Distress (speed × 0.8); 100–500 Danger (× 0.5); 500 and above Lethal (incapacitated).
- **Collapse:** hazard rate `λ(band) · (1 + D/100)` per second, with λ = 0.001 (Distress) and 0.03 (Danger). Probability `1 − exp(−h·dt)`. Immediate at 500 ppm or more, or at D ≥ 500 ppm·min.
- **Health status** is the worst of the band and the dose level (1 / 20 / 100 ppm·min → Irritated / Distressed / Disabled), latched for the rest of the run. Speed is limited by the latched status as well as by the current band.

**Coupling to the social force model** (SYSTEM_SPEC §2.1, revised from the literature in this release):
- **Desired speed:** `v_des = [1 − p] v0 + p · v_max`, with `p = 1 − v̄/v0` (impatience; Helbing, Farkas & Vicsek 2000) and `v_max = min(2.0, 1.5 v0)`. In this prototype a PANICKED agent's speed is multiplied by 1.2, capped at 2.0 m/s.
- **Herding:** `e = N[(1 − p_h) e_own + p_h ⟨e_neighbours⟩]` with **p_h = W2** (Helbing et al. 2000, Eq. 4). The agent's social weight is literally the strength of its herding force. This is the "adjust their social forces" link the supervisor asked for, and it will apply once the SFM is built.

## 11. Interface and telemetry

`Agent.step(snapshot, dt) -> Action(agent_id, state, cell, target_node, speed, arrived, incapacitated)`. The `Snapshot` holds the time, the alarm flag, the gas history, the locked exits and a spatial hash of everyone (grid cell = largest sensing radius; no O(N²) loops).

Telemetry (`out/agents_telemetry.csv`) has one row per event. Each row records time, agent, event, state (from and to, with the rule ID), persona, node, ppm, dose, stress, health and W1/W2/W3. Events are state changes, `route_failed`, `exit_unusable` (with the reason), `exit_clear`, `learnt_exit` (seen or followed others), `smelt_gas`, `persona_shift` and `left_building`. The per-agent summary (`out/agents_summary.csv`) adds time to safety (assembly time − alarm time), exit used, dose, peak, health, reroutes, panics and persona shifts.

## 12. Prototype results

These come from `python -m dms.agents.sim --agents 550 [--block X-E] [--compare]`, run on B1 under the H₂S tank-leak scenario (dms/gas; alarm at 18 s; the main exit X-S reaches about 94 ppm, X-W and X-NW about 8 ppm, X-E and X-N clear after the wind shift). The run covers 300 s and takes about 70 s on a laptop.

**Default leak, adaptive weights (550 agents):** 98.2 % reach an assembly point, 2 incapacitated, 8 still walking at 300 s. Mean time to safety 115.6 s, P95 197 s. Only 2 people use X-S: everyone else sees the gas, is told, or never knew X-S and follows signs or other people.

| Group | n | Safe | Time to safety (mean / P95) | Dose (ppm·min) | Reroutes per person |
|---|---|---|---|---|---|
| Young | 106 | 100 % | 92 / 143 s | 1.85 | 1.35 |
| Adult | 299 | 99.3 % | 109 / 157 s | 1.67 | 1.14 |
| Elderly | 145 | 94.5 % (2 incapacitated) | 149 / 241 s | 1.93 | 0.77 |
| Male / Female | 345 / 205 | 98.6 / 97.6 % | 114 / 119 s | 1.73 / 1.86 | 1.10 / 1.07 |
| Senior staff | 94 | 98.9 % | 120 / 201 s | 2.89 | 1.48 |
| Visitor | 80 | 95.0 % | 123 / 212 s | 1.23 | 1.23 |
| VIP (escorted) | 16 | 100 % | 106 / 150 s | 1.32 | 0.38 |
| Responder | 28 | 100 % | 87 / 110 s | 0.10 | 0.75 |
| Cautious | 156 | 94.9 % | 128 / 236 s | 1.17 | 1.20 |
| Rationalist | 223 | 99.6 % | 110 / 164 s | 2.30 | 1.10 |

The groups now behave differently for the reasons the design intends:
- **Elderly** are slower and react later, so they spend longer inside and some are caught.
- **Cautious** agents take the least gas but the longest time; a few keep turning away from exits at about 8 ppm that others walk through.
- **Rationalists** are fastest among evacuees but take more gas, because they trust their route.
- **Responders** see the forecast and are fastest and almost unexposed.
- **VIPs** follow their escort and do well.

**Exit X-E locked, discovered only on arrival:** 97.1 % safe, 3 incapacitated, 13 still walking. Mean time to safety rises from 115.6 s to 131.2 s. Reroutes per person rise from 1.09 to 1.63, and Elderly success drops to 89.7 %. Everyone ends up at X-N (376), X-NW (136) or X-W.

**Ablation, weights fixed (same 550 people, default leak):** 99.3 % safe, mean dose 1.90 (adaptive 1.78), 1.32 reroutes per person (adaptive 1.09). **Adaptation trades a little speed and success for less exposure**: agents who have been surprised lean on the hazard and on other people. That is the realistic "you may learn the wrong lesson" effect, not an optimisation.

## 13. Calibration lessons from building the prototype

Three behaviours had to be corrected before the results made sense. They belong in the design because each is a real-world question:
1. **Permanent "that exit is gassed" beliefs** made everyone give up on every exit within a minute. Beliefs about gas must expire, because gas moves. They now fade after 60 s.
2. **Responders repeating each other's warnings** kept stale warnings alive indefinitely (an echo chamber). Responders now trust only their own radio, and a repeated rumour does not refresh a belief.
3. **A panic cascade** (57 % of agents panicked, starting in the lobby with no gas) came from stacking stress from repeated route failures and from panicked neighbours. Panic should be rare: Helbing et al. 2000 cite Keating's *The Myth of Panic*, and SYSTEM_SPEC disables panic by default. Repeated failures now add less and less stress (`0.15 / k`), panic contagion is capped at 3 neighbours, and panicked agents choose less randomly. Result: 1 panic in 550.

These need drill data to calibrate. Until then they are labelled starting values.

## 14. What is not done yet, and the next pieces

- **No collisions, queues or crowd density.** Movement is a stand-in. Next: door and stair capacity queues and density, then the social force model with S2a / S7h above.
- **Shelter-in-place** (v0.2 T4) is not in the prototype; the dms/gas simulator has it, ready to port.
- **The egress walk** uses the worst air on the exit's path (conservative).
- **Groups / families** (Moussaïd et al. 2010: up to 70 % of people walk in groups) are not modelled yet.
- **Prior experience** affects reaction time, knowledge and the size of the weight shift, but not yet route memory decay.
- **Calibration:** every [cal] value and the event impulse sizes need drill data or a literature fit.

## 15. Questions for the supervisor

1. **The split.** Are these the right levels and shares (age 20/55/25, male 60 %, the role mix, VIP 3 %, responders 5 %), and should VIP status also come from senior staff (for example a visiting director)?
2. **Sex differences.** Speed differs by sex (Li 2022). We have no source for reaction-time or decision differences, so they are neutral. Do you have one?
3. **Event sizes.** Is "a failed route moves about 0.3 of weight from W1" the right order of magnitude? These are the main knobs for how realistic adaptation looks.
4. **Panic.** Keep it rare, as now, or make it more common for some scenarios?
5. **Thresholds.** Alarm 30 ppm; Distress speed factor 0.8; Danger 0.5 (SYSTEM_SPEC says 0.4). Which is canonical?

## References

Helbing, Farkas & Vicsek 2000, *Nature* 407:487 · Hoogendoorn & Bovy 2004, *Transp. Res. B* 38:169 · Johansson, Helbing & Shukla 2007, *Adv. Complex Syst.* 10:271 · Lakoba, Kaup & Finkelstein 2005, *Simulation* 81:339 · Li, Hou, Cheng & Gai 2022, *PSEP* 163:543 · Moussaïd et al. 2010, *PLoS ONE* 5:e10047 · Weidmann 1993, ETH IVT · Kuligowski, NIST IR 7914 (Protective Action Decision Model) · project documents: Agent Data and Decision Model (brief, v0.1, v0.2), DMS Data Dictionary, EVAC-SIM and AI-DMS decks, SYSTEM_SPEC.md.

**All parameters:** `dms/agents/config.py`. **Tests:** `tests/test_agents.py`.
