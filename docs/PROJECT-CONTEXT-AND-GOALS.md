# Disaster Management System (DMS) — Project Context & Goals

**Automated evacuation simulation for a toxic-gas release (H2S) in an industrial/nuclear plant**
*Compiled from every specification document in the repository and the one-level-up project folder, the governing `SYSTEM_SPEC.md`, and the verified state of the codebase. Intended as a self-contained brief for an external professional / engineering second opinion.*

> How to use this document: it is deliberately self-contained. Section 2 gives a one-page summary; sections 3–5 carry the complete context; sections 6–8 describe exactly what is built and what is being handed to you for review; section 9 is the RL algorithm decision (DQN vs PPO); section 12 lists the explicit questions we want a second opinion on.

---

## 1. Project identity

| Field | Value |
|---|---|
| Name | Disaster Management System (DMS); internal codename Cl-EVAC-SIM |
| Vision | AI-based, agent-based (multi-agent) evacuation simulation for a nuclear/industrial plant |
| Partners | PAEC / NUST AI Solution Lab initiative (Dr. Mazhar Sajjad) |
| Hazard now modelled | Toxic hydrogen sulfide (H2S) release, indoor (building) + outdoor dispersion up to plant scale |
| Current focus building | B1 — a two-storey lab/office building (GF + FF, 4.0 m floors), 100-agent simulation |
| Long-term scale | Whole plant: 5 two-storey buildings, ~3,000 workers (evacuating at phases up to 10,000 agents) |
| Codebase | Python 3.10+ package `dms`, git remote `github.com/FARHANFAST/disaster-management-system` |
| Active branch | `agents-with-attributes` (DQN route policy, dashboard). New branches `agent-behaviour-model` (agent decision/FSM module, in progress) cut from it |

The source specifications, in priority order used by the team:

1. **`SYSTEM_SPEC.md`** (repository root) — the binding technical specification: frames, clocks, governing equations (SFM, density–speed, Eikonal, gas dispersion, toxicology), FSM, parameters. *(Fully reproduced conceptually in §4–5 below.)*
2. **`PROMPT_01_two_storey_building.md`** — environment-foundation milestone brief (L0 grids + L2 graph + L3 site).
3. **One-level-up folder `Ai Solutions Lab\`** — nine design/decision documents:
   - *Agent Data and Decision Model - Design Specification v0.1.docx*
   - *Agent Data and Decision Model.docx* (full)
   - *Agent_Data_and_Decision_Model_Design_Concise.docx*
   - *DMS - Data Dictionary - 15-09-2026.docx*
   - *Module-Dispersion-Weather Model.docx* and *Module-Environment & Spatial Model.docx*
   - *AI_DMS_MSajjad.pptx* and *Cl-EVAC-SIM_MAS_Evacuation_Presentation.pptx*
   - *CV-CS474-Assignment 1.pdf* (personal CV — not authoritative for the model)
4. **Research papers the model is grounded in** (as cited in the specs): Helbing & Molnár 1995 (Social Force), Makmul 2023 (S1–S9, Eikonal nav), Liu 2021 (CA + exit logit), Chang, Hsu, Su 2024 (two-phase toxic-gas evacuation, *Comput. Oper. Res.* 167:106645), Li, Hou, Cheng, Gai 2022 (chemical-plant evacuation, Pathfinder/SFM engine, *Process Saf. Environ. Prot.* 163:543–557), Huo et al. 2024 (CA + toxic gas + dynamic exit-risk perception, *Int. J. Disaster Risk Reduct.* 109:104589).
5. **Step-summary documents in `CODE\docs\`**: `Agent-Attributes-Step-Summary.docx`, `DQN-Change-Proposal.docx`, `DQN-Step-Summary.docx`, `Social-Force-Proposal.docx` — project history and decisions, produced milestone-by-milestone.

---

## 2. The project in one page

We are building an **agent-based, physics-grounded evacuation simulator**. An H2S release inside a two-storey building (and eventually across an industrial site) is detected; an alarm triggers; **heterogeneous people** — varying by age, gender, role, mobility, walking speed, personality, stress, health — decide where to go and walk there under crowd and gas constraints. The system must produce, for any hazard scenario, trustworthy statistics: evacuation time, exposure, risk heatmaps, and whether people reach safety or are incapacitated.

The design story in five increments, all but the last two already proven in code:

1. **World** (done). One declarative YAML builds the building geometry and validates it; rasterised to grids; a navigation graph; shortest-time (Eikonal and Dijkstra) fields to the exits. B1: two floors, three staircases, five usable exits + one locked service door, outdoor assembly points on all four sides.
2. **People** (done). 100 heterogeneous agents on a self-contained live web page. Only movement attributes (speed, mobility, reaction time) currently affect behaviour.
3. **Decide** (done). A **Deep Q-Network route policy** (PyTorch) replaces shared tabular Q-learning; each agent conditions the shared network on its own attribute vector, so routes can differ per person. Graded against provable optimal times.
4. **Behave** (in progress). The agent behaviour module: an explicit finite-state machine (NORMAL → ALERT → DECIDING → MOVING → AT_ASSEMBLY, plus PANICKED / TRAPPED / INCAPACITATED), a perception–decision–action loop, gas-dose physiology (Haber + probit bands), herding/panic, and per-agent reproducible randomness.
5. **Interact** (proposed next). The **social-force crowd model** — continuous SFM steering, door/stairs capacity discharge, congestion-aware routing costs, emergent clogging/arching — replacing today's simple stand-in movement.

Since Milestone 3 the running hypothesis has been exactly what the Data Dictionary recommended: **a hybrid — a discrete decision layer (state machine + utility scoring + RL routing) above a continuous Social Force locomotion layer (SFM)**, with gas dispersion feeding both. `SYSTEM_SPEC.md` already fixes the equations and parameters for all of it.

---

## 3. Domain context and why this simulation exists

From `AI_DMS_MSajjad.pptx` (project vision) and the Module documents:

- The long-horizon operational context is a **nuclear power plant (NPP)** emergency response. The briefing defines the **Precautionary Action Zone (PAZ)** ≈ 5 km around the plant and the **Urgent Protective Action Planning Zone (UPZ)** ≈ 5–30 km; the end-goal is moving people to safe shelters per PAZ/UPZ recommendations, with an estimate of **210,000 people within a 10 km radius**.
- The simulator first solves the tractable industrial case: an **H2S toxic release** in multi-storey buildings, then scales to site-wide (outdoor Gaussian/puff dispersion, wind) and eventually plant town planning.
- **Why H2S and why this matters.** H2S is heavier than air (density ratio ≈ 1.18 — a *dense* gas, which invalidates the classical Gaussian formulas and demands a heavy-gas path), smells at low ppm, and becomes dangerous well below its detectable-smell threshold. The band thresholds used (0.5 / 10 / 100 / 500 ppm) and the exposure–incapacitation physiology (Haber dose, probit) are the toxicology core. A release can incapacitate a person before taste/smell warning is useful — which is precisely why the *decision model* (route choice away from concentration, alarm propagation) must be tested carefully.
- The system's job: **"If a release happens here, how long until everyone currently inside is out / safe / incapacitated, and which routes, signs, exits and assembly points perform best?"**

---

## 4. Governing architecture and equations (`SYSTEM_SPEC.md`)

The spec is binding; the code is implemented to it. The essentials a reviewer needs:

### 4.1 Frames and clocks
- **L0 cells** (grids): smoke/Eikonal at `h_s = 0.2 m`, occupancy/CA at `h_c = 0.5 m`. Arrays are `[row=y, col=x]`, row index grows with `+y`.
- **L1 local** (metres, continuous): SFM kinematics inside a building.
- **L2 macro** (graph): `G=(N,E)`, one node per room/stair/door/corridor junction — routing decisions.
- **L3 global** (plant SW corner origin): outdoor dispersion + macro egress.
- **Clocks:** locomotion/dose `Δt_L = 0.02 s`; cognition/exit choice `Δt_C = 0.5 s = 25·Δt_L`; outdoor macro `Δt_O = 0.5 s`; dispersion `Δt_D = 5–10 s`; potential-field (FMM) recompute is **event-driven** (§4.4).

### 4.2 Scale, phasing, complexity rule
- Phases: P0 prototype = **50** agents; P1 intermediate 200–2,000 (literature benchmarks); P2 target stress = **10,000**. Agent count is a config value only — **the same code runs at all N**.
- Hard rule: **no O(N²) pair loops at any scale**. Neighbour queries go through a uniform-grid spatial hash (cutoff `r_c = 2.0 m`, CSR, rebuilt every Δt_L, O(N)) and a `scipy` `cKDTree` for perception (≤ 5 m) rebuilt every Δt_C.

### 4.3 Social force locomotion (continuous, SFM) — S1–S9
The canonical equations (Helbing–Molnár form, per the spec):
- **S1** `m·dv/dt = f_will + Σ(f_soc + f_ph) + Σ f_wall + f_group + ξ`
- **S2** will force: `f_will = m (σ·v⁰·e − v) / τ` (τ = 0.5 s)
- **S3** social repulsion: `f_soc = A·exp((r−d)/B)·n·[λ + (1−λ)(1+cosφ)/2]`, `A = 2000 N`, `B = 0.08 m`, anisotropy `λ = 0.61`
- **S4** physical contact: `f_ph = k·g(r−d)·n + κ·g(r−d)·Δv_t·t`, `k = 1.2e5 kg/s²`, `κ = 2.4e5 kg/(m·s)`
- **S5** walls/obstacles, **S6** group cohesion (`A_g = 800 N`, cohesion radius `d_g = 1.5 m`), **S7–S9** leader-follower + herding
- Integrator: **Ralston RK2**; deadlock resolution: if net force ~0 for 3 ticks, add `ξ = f_will/3`.
- Parameters: mass `m = 80 kg`, body radius `r ≈ 0.2 m` (SFPE shoulder data in the Data Dictionary ranges 31.9–46.9 cm), desired speed `v⁰` plant 1.19 m/s (USGEM bounds 0.1–1.4 m/s).

### 4.4 Potential-field navigation & reactive recompute (N1–N2)
- Heading `e = −∇T/|∇T|` from an **Eikonal travel-time field** `T` (FMM), exits seeded to `T=0`. Speed field `F = U_max` on free cells, `F_b = 0.001 m/s` on obstacles.
- **Dynamic density is deliberately NOT in F**. Crowding enters via SFM repulsion (continuous layer) and the congestion weight `w_ρ` in the graph cost (discrete layer). *(This separation is important for the reviewer — see §12.)*
- FMM/`T` recompute only on *dirty events* (door/link state change; an INCAPACITATED agent blocking a door/stair cell; a cell first exceeding 100 ppm; smoke exceeding `C_hi`), debounced to ≤ 1 recompute per building per Δt_C.

### 4.5 Speed–density coupling (N3–N4, N4s)
- Density `ρ(x) = #{j: |x−x_j| < R_v} / (π R_v²)`, `R_v ≈ 5 m` percepti/impact radius, `ρ_max = 10 ped/m²`.
- `U(ρ) = U_max·(1 − ρ/ρ_max)`, and effective speed `v = σ·v⁰·(1 − ρ/ρ_max)·φ` ∈ [0.1, 1.4] m/s.
- Stair kinetics: `v_stair(ρ_s) = max(0.1, 0.70·v_flat·(1 − 0.266·ρ_s))` — reaches the 0.1 m/s floor at ≈ 3.2 ped/m².

### 4.6 Link / A* edge cost — congestion + gas (N5)
`c_l = (L/(χ·σ·v_ref))·(1 + w_ρ·ρ̄/ρ_max + w_C·C̄/C_ref + …)·ω_p`, with defaults `w_ρ = 1.5`, `w_C = 3.0`, `C_ref = 30 ppm` (alarm level), `v_ref = 1.19 m/s`. `ρ̄`, `C̄` are per-link averages — **today `ρ̄ = C̄ = 0` in the code, the integration hook is already there** (`dms/graph/cost.py`, `Link.rho`).

### 4.7 Exit choice (E1) and perception
Multinomial logit over known exits: `P_k ∝ exp(−θ·T_k)`, where `T` blends travel time, congestion (impatience `α`), and gas-distance. Perception radii `R_vis = 5 m`, `R_aud = 2.5 m`; people route-choice periodically (Δ_tC), re-scoring when a neighbour exits is seen or patience exhausted.

### 4.8 Toxicology & health (X1–X3)
- **Haber dose** `D += C^2·Δt/60` (ppm²·min, exponent 2) and **probit** `Pr = a + b·ln(L)`, `a = −15.67`, `b = 1.024`, incapacitated if `Pr ≥ 5`. **Honest caveat in the spec:** with these constants the probit almost never fires alone; incapacitation is governed by the band logic below.
- **Bands (instantaneous C, ppm):** B0 < 0.5 · B1 0.5–10 · B2 10–100 · B3 100–500 · B4 ≥ 500. Speed factors `φ = {1.00, 1.00, 0.80, 0.40, 0.00}`. **B4 = immediate collapse.** A non-collapsing person's status only ever gets worse (latched worst-state).
- ⚠️ *Flag for review:* the **agent-behaviour module spec** (in progress, §7) uses speed factors **{1.0, 0.8, 0.5, 0.0}** for Distress/Danger/Lethal and dose thresholds ≥ 1 / ≥ 20 / ≥ 100 / ≥ 500 ppm·min — slightly different from SYSTEM_SPEC's {1.00, 0.80, 0.40, 0.00}. This is a known reconciliation item (§12).

### 4.9 Agent finite-state machine (spec §4.1)
- States: **NORMAL → ALERT → DECIDING → MOVING_TO_ASSEMBLY** with sub-states {EXIT_ZONE, TO_SAFETY} and navigation {EIKONAL, FOLLOW_GUIDER, FOLLOW_WALL, FOLLOW_STREAM, HERD}; orthogonal **PANICKED** (rare, non-contagious by default, `panic_enabled = False`); absorbing **INCAPACITATED** (freezes, becomes obstacle + FMM dirty event) and **AT_ASSEMBLY** (absorbing; only at an outdoor assembly point/shelter).
- Key triggers: NORMAL→ALERT on ≥ MINOR health, hazard/peer/dead-agent within R_vis, trust-count of ALERT neighbours > 5, zone self-trigger, or **global alarm**; ALERT→DECIDING after pre-movement time `t_pre ~ LogNormal(3.8, 0.6) s` (default); DECIDING→MOVING on (E1) selection; state-locking: past ALERT nothing returns to NORMAL/ALERT.

### 4.10 Integration contracts
- `AgentState`, `ConcentrationGrid`, `Node`, `Link` (with live `rho`/`C`), `EnvironmentSnapshot`, `SpatialHash`, `SimConfig` dataclasses are all pre-defined in the spec — the modules are implemented against these.
- `dms/tools/build_env.py` exports everything a downstream module needs to `out/layers.npz` + `out/graph.json` (202 nodes / 244 links for B1) so agent physics and dispersion **never re-rasterise** from the YAML.

---

## 5. What exists and is proven in code today

### 5.1 Repository layout (core)
```
dms/
  geometry/   loader, building/site model, validator   [M1]
  grid/       rasterisation (H_COARSE=0.5, H_FINE=0.2), FloorGrid, overlays  [M2]
  graph/      nav-graph builder, A*, redundancy report, Link cost (N5 hook)  [M3]
  fields/     Eikonal FMM travel-time fields, outdoor fields  [M4]
  hazards/    gas-permeability topology derived from geometry
  rl/         env.py (cell MDP, 8-moves + STAIRS, no-corner rule), evaluate (Dijkstra optimum),
              q_learning.py (tabular Q), graph_q.py (graph Q + walker), graph_env.py (GridWalker),
              dqn.py (Deep Q-Network), train_graph.py / train_dqn.py, agent_profile.py,
              live_page.py (100-agent live page), dash_dqn.py (training dashboard)  [M5–M6]
  scenario/   scenario presets (blocked exits, closed stairs, clutter, parked cars)
  tools/      build_env.py (artifacts), viz/ (plots, demo simulation, walker test)
tests/        180 pytest tests (baseline) — env, graphs, validation, redundancy, RL, DQN
configs/buildings/two_storey.yaml   # single source of truth (no coordinates in code)
out/                                # generated artifacts (layers.npz, graph.json, live page, policy)
```

### 5.2 B1 building fact sheet
- Two floors, `floor_to_floor = 4.0 m`; **three staircases** — West (enclosed), East (enclosed), Central (open, main). The FF has a deliberately designed dead-end corridor (a trap test).
- **GF exits:** X-S, X-N, X-W, X-E, plus stair-discharge X-NW; **X-SE is a service door locked by default** (tests prove an overlay can unlock it). **FF has no direct exit** — everyone on FF must pass through a stair.
- Outdoor **assembly points on all four sides** of the building → the `Site` L3 network (exit→assembly-point distance matrix, e.g. `X-S|AP-N: 117.7 m`).
- Size: **16,488 walkable cells** at h_c = 0.5 m across both floors; navigation graph **202 nodes / 244 links**.
- **Watertightness & redundancy are tested**: sealed-envelope leak test, blocked corridor ⇒ next-best path (the "#1 quality" from PROMPT_01).

### 5.3 Milestones delivered (each ends with a step-summary .docx)
| # | Milestone | Verified by |
|---|---|---|
| M1–M4 | Geometry, grids, nav graph + A*, Eikonal fields | test per module; 173 → 180 tests |
| M5 | 100 heterogeneous agents (full attribute set) on the live page | `python -m dms.rl.live_page --agents 100 --seed 7` → `out/rl_live.html` |
| M6 | **DQN route policy** with agent-attribute state (commit `4abe91c`) | `tests/test_dqn.py` (+6); dashboard; grading vs cell optimum |
| M7 | Agent behaviour module (FSM + decision + physiology + telemetry) | **in progress** on branch `agent-behaviour-model` (uncommitted) |
| M8 | Social-force crowd interaction model | **proposed** — `docs/Social-Force-Proposal.docx` |

### 5.4 The 100-agent population (agent attribute model)
`dms/rl/agent_profile.py` is the **shared source of truth** for people:
- Demographics (gender M/F 60/40, age Young/Adult/Elderly 20/55/25, role Senior/Mid/Junior/Visitor/Contractor 25/30/20/15/10) drive: base speed (Li 2022 Table 3, e.g. elderly F 0.75–1.00, young M 1.30–1.60 m/s), mobility (full≈1.0 / reduced≈0.7 / assisted≈0.5, age-weighted), reaction time (Young 5–15 s, Adult 8–20 s, Elderly 15–40 s, × role seniority 0.7–1.3), capped effective speed ≤ 2.0 m/s.
- **Cognitive profile:** persona (Rationalist 48%, Follower 26%, Cautious 21%, Responder 5%) with weights (W1 knowledge, W2 social, W3 hazard), panic threshold by persona, familiarity by role (Senior 0.82–0.90 → Visitor 0.15–0.40), sensing radius 5 m (15 m for Responders), reaction time.
- **Physiological (initial):** health status Healthy/Irritated/Distressed (age-weighted; some start impaired), stress, fatigue, small initial toxic dose, peak ppm.
- Exposed to the DQN as a **normalised 26-feature vector** (persona weights, thresholds, movement, health one-hots, responder flag, sensing radius).
- **⚠️ Spec-vs-code reconciliation (already resolved, documented):** the behaviour-module *brief* quoted persona weights/panic ranges that differ from `agent_profile.py` (e.g. Cautious (0.45,0.10,0.45) vs brief's (0.25,0.15,0.60); panic ranges similar offsets). **Rule applied: the code (agent_profile.py) is the source of truth.** Full difference list is in the module's final report.

### 5.5 The live page & dashboard (how a reviewer can see it)
- **Live page** (`out/rl_live.html`, 100 agents): trains graph Q-learning in the browser on the same navigation graph as Python; per-agent trait table + tooltips; Test mode measures real route times × each agent's speed factor + reaction time.
- **Dashboard** (`dms/rl/dash_dqn.py`, currently `:8768`): starts DQN training runs on the GPU, streams the learning curve, final grade and per-room routes, embeds the live page. `/state` reports `cuda` (RTX 4050, 6 GB, driver 592.82, torch 2.5.1+cu121).

### 5.6 Proven RL quality (measured, from DQN-Step-Summary)
Both learners graded from all 16,488 walkable start cells against the cell optimum (best physically possible):
| Scenario | Policy | Success | Within 5% | Mean excess | p90 | Max |
|---|---|---|---|---|---|---|
| Baseline | Tabular | 99.77% | 60.6% | +4.92% | +11.1% | +58% |
| Baseline | **DQN** | **100.00%** | 62.2% | +4.70% | +10.6% | +73% |
| Blocked X-E+X-S | Tabular | 90.61% | 72.0% | +3.10% | +6.6% | +73% |
| Blocked X-E+X-S | **DQN** | **99.46%** | 73.9% | +3.40% | +7.6% | +36% |
| Closed stair ST-C | Tabular | 98.61% | 67.5% | +3.98% | +7.8% | +73% |
| Closed stair ST-C | **DQN** | **100.00%** | 69.6% | +3.80% | +7.5% | +73% |

Training cost: tabular ~0.3–0.4 s; DQN ~35 s @ 2,000 eps, ~68–156 s @ 4,000 eps. **Measured honestly: at this problem size the rollout loop is Python-bound; CPU and CUDA take about the same wall time.** CUDA is default so the GPU is exercised now and pays off once crowd/smoke features enlarge the state.

---

## 6. The actual goals (short → long)

**Near-term (building-level, being developed now):**
1. **Agent behaviour module** (M7, in progress): the full perception–decision–action FSM — states, transitions with reasons, the weighted cell-scoring decision rule `S(c) = W1·K + W2·F + W3·H`, noised argmax, gas-exposure physiology with per-agent seeded randomness, herding + persona shift, telemetry (CSV) and a summary report — **pure and deterministic given inputs + seed**, consumable without the rest of the simulator.
2. **Social-force crowd interaction** (M8, proposed): continuous SFM locomotion on the real grid, door/stair capacity `Q_max`, density feeding the `w_ρ` congestion term in link costs, emergence (arching, clogging, lane formation, faster-is-slower), validation against the fundamental diagram (Weidmann / SFPE), everything per `SYSTEM_SPEC` §2.1 and `Social-Force-Proposal.docx`.

**Medium-term (per module documents):**
3. Indoor H2S smoke transport inside buildings (convective + diffusive, `κ_d`, upwind-advection operator-split solver; a teammate owns this **dispersion** module).
4. Outdoor Gaussian/puff + heavy-gas dispersion with wind and stability classes; dense-gas (Britter-McQuaid / box / FEM-3 / learned CA surrogate) for H2S.
5. Whole-plant scale-up: 5 buildings, ~3,000 workers, 10,000-agent stress runs with spatial hashing (no O(N²) anywhere), risk zones `Z^(L)`, 60-min-plume risk maps.
6. Validation against the literature benchmark table (SFM test-room flows 1.73–2.01 p/s at 50–300 agents; Song 2000-agent simulations; asymmetric trim/heel inclines).

**Long-term (vision, AI_DMS deck):**
7. Site/urban evacuation studies against PAZ/UPZ recommendations (210k people within 10 km), shelter assignment, signage/exit-allocation optimisation, and a decision-support layer ("which exits/signs/assembly points minimise casualties for this release?").

---

## 7. Current work in flight (exactly what is uncommitted / being reviewed)

1. **`agent-behaviour-model` branch (uncommitted):** the `dms/agents/` package — config dataclass (no magic numbers), state machine with reason-string transitions, physiology (bands, Haber, collapse, stress), decision rule with pluggable progress provider (`GridDistanceProgress` wrapping the existing Dijkstra field; hook reserved for a DQN-based provider — the DQN is **not** retrained or changed), social sensing via grid-hash (no O(N²)), telemetry, synthetic gas/EnvData demo, and a deterministic pytest suite. Movement in the demo is a clearly-labelled kinematic stand-in for the social-force orchestrator.
2. **`docs/Social-Force-Proposal.docx`** — the approved design/build proposal for the SFM module (uncommitted): scope, grounding in all source docs, what it changes, `dms/crowd/` package architecture, phases A–F with file-level plan, constants registry, risks.
3. **Dashboard files** (`dash_dqn.py`, `dash_template.html`, `test_dqn.py` additions) uncommitted on `agents-with-attributes`.

Rule for all new work: **zero behavioural change to `dms/rl/{q_learning,graph_q,dqn,train_dqn,live_page}.py` and the DQN dashboard** (imports allowed); edit existing files only if strictly needed and listed.

---

## 8. Validation strategy (how we prove things)

- **Property tests are the backbone** (180 today): no-corner-cutting moves, rewards = −seconds, Dijkstra optimum = hand-computed times, watertight envelope, overlay unlocks locked exits, closed stair never makes FF faster, Q-learning converges to optimum, DQN feature vector and seeding reproducibility, attribute conditioning actually changes Q.
- **Cross-check vs literature** (§4.5 benchmark table): SFM room-exit flows, 1-exit/2-exit evacuation counts, speed–density (Weidmann diagram), stair slowness, Song/Senanayake plant-scale simulations.
- **Determinism rule**: same YAML + config ⇒ bit-identical arrays/graph (hashed); all randomness through seeded `numpy.random.Generator`; per-agent generators keyed `(global_seed, agent_id)` so results are **independent of agent update order**.
- **Honest claims policy**: nothing in this project is labelled "tested" unless actually run; CPU-vs-GPU times are measured and reported as-is; source-paper formulas marked corrected/unverified are flagged in the spec.

---

## 9. RL decision under review: current DQN vs PPO

This is the explicit ask for a second opinion. We currently run a **PyTorch Deep Q-Network** as the route-policy layer, and the question is whether **Proximal Policy Optimization** (often the default answer to "should we switch") would be better — especially since people sometimes associate PPO with *continuous* action spaces.

### 9.1 What the current DQN is (in this codebase)
- **Form:** one shared MLP — input = 176 one-hot node features + 26 attribute features → 64 hidden units → ≤ 4 outputs, one `Q(s,a)` per neighbour slot of the current navigation-graph node. All 100 agents share the weights; the attribute vector makes them condition differently.
- **Learning:** off-policy **value-based** (learns expected return per action, acts by `argmax Q`). Replay buffer, target network, double-Q updates, epsilon decay plus familiarity-guided exploration. Reward = −walking seconds, `γ = 1.0`.
- **Interfaces it must satisfy:** `greedy_rollout`, `evaluate_graph_policy`, the live page and dashboard all consume **argmax over neighbours**. DQN drops into those unchanged.
- **Action space:** discrete and tiny (pick 1 of ≤ 4 neighbours).

### 9.2 What PPO is
- **Form:** on-policy **policy-gradient** actor-critic. The actor emits a distribution `π(a|s)` (a categorical over the ≤ 4 neighbours here; a Gaussian mean+std for continuous control); the critic estimates value to form an advantage `Â`. Updates use the **clipped surrogate objective** `L = E[ min(r(θ)·Â, clip(r(θ), 1−ε, 1+ε)·Â) ]`.
- **Key contrast:** on-policy — data is collected under the *current* policy and discarded after an update; exploration via stochastic sampling + entropy bonus; famous for continuous control (robots, RLHF), but handles discrete equally well.
- Requires a policy-→argmax shim to feed the existing greedy consumers.

### 9.3 Direct comparison
| Axis | DQN (current) | PPO |
|---|---|---|
| Learns | `Q(s,a)` value | policy `π(a\|s)` (+ value critic) |
| Data usage | **off-policy** (replay reuse) | **on-policy** (fresh rollouts, discard) |
| Action space | naturally discrete | discrete or continuous |
| Exploration | ε-greedy | stochastic sampling + entropy bonus |
| Stability | OK with target net + double-Q; can overestimate | generally more robust across hyperparameters |
| Sample efficiency | **better** — reuses old transitions | worse — one epoch per rollout |
| Deterministic exploit | trivially `argmax` | `argmax` of logits (policy stored as distribution) |
| Compute / update | cheap (mini-batches from memory) | multiple epochs over fresh rollouts |
| Well-known sweet spot | small discrete action spaces, off-policy reuse | high-dim / continuous control |

### 9.4 Our assessment (why DQN is the right choice *for this problem*)
1. **Discrete ≤4-action routing** is DQN's natural domain; PPO adds an actor, critic and advantage machinery for no benefit at this action count.
2. **The whole downstream stack consumes `argmax`.** DQN slots in with zero API change; PPO forces a distribution→argmax conversion at every evaluation call site and in the browser page.
3. **Sample efficiency matters more than robustness here.** Training time is dominated by the Python rollout loop (measured: GPU ≈ CPU). Replay reuse extracts more learning per rollout; on-policy PPO burns each rollout after one update.
4. **Tiny state space (202 nodes × ≤4 actions).** The convergence-fragility that motivates PPO (overestimation, high variance) is minor here — the DQN converges near-instantly and its measured policy is at or beyond the tabular baseline's quality (100% success in the nominal and closed-stair scenarios).

### 9.5 When PPO would win (and why it is on our radar)
- **Continuous motion control.** The upcoming **social-force layer** gives agents forces / 2-D steering vectors (S1). If the policy layer ever must *compute continuous steering* (choosing force direction/magnitude) instead of picking a neighbour slot, DQN's argmax doesn't generalise; a Gaussian-actor PPO is the standard tool. That is the natural end-state of §4.3 — *if* we choose to learn low-level motion rather than keep SFM hand-engineered (recommendation: keep SFM deterministic/physics-driven and let RL stay at the discrete routing layer — see §10/§12).
- If we later want **stochastic, explicitly risk-shaped route policies** (e.g., modelled behavioural variability in evacuation choices), a learned policy distribution is easier to shape than a value table.

**Bottom line we are defending:** *DQN now for discrete graph routing (correct tool, matches every consumer, sample-efficient); keep PPO in the toolbox for the day — if ever — the action space becomes continuous steering. A switch today buys risk without benefit.*

---

## 10. Real vs stand-in (honesty inventory)

| Layer | Status in code today | Replacement / owner |
|---|---|---|
| Building geometry, grids, graph, fields | **Real & verified** | — |
| RL route policy (DQN) | **Real**, graded vs optimum | — |
| Agent behaviour (FSM, decision, physiology, telemetry) | **Real** in `dms/agents/` (in progress, tested) | — |
| Cost-layer congestion/gas coupling (`Link.rho`, `Link.C`) | Stubbed at 0; hook ready | SFM milestone + dispersion teammate |
| Pedestrian motion in the agent demo | **Kinematic stand-in** (advance toward target cell at desired speed) | SFM orchestrator (M8) |
| Indoor gas field in the agent demo | **Synthetic** (analytical expanding cloud + 5-min forecast) | Dispersion teammate (M3 medium-term) |
| Crowd density in the agent demo | **Synthetic** (occupancy in a 2 m radius → ped/m²) | Occupancy from the SFM layer |
| Alarm | Global alarm flag once any sensor ≥ 30 ppm (synthetic) | Dispersion/ODOR monitoring |
| Door/stairs capacity discharge | Grid-level move times only (STAIRS cost); no queueing | SFM `Q_max` + door capacity (M8) |
| Outdoor egress | L3 field exists for the site; not driven by agents yet | Macro movement (M3 medium-term) |
| Heavy-gas dispersion | Equations specified; not implemented | Dispersion teammate |
| Probit vs band dominance | Bands govern; probit flagged as dominated in spec | Verify against TNO Green Book / CCPS |

---

## 11. How to run & inspect (Windows, venv)

```powershell
# Baseline tests (180 passing expected)
.\.venv\Scripts\python.exe -m pytest

# Live 100-agent page → out/rl_live.html
.\.venv\Scripts\python.exe -m dms.rl.live_page --agents 100 --seed 7

# DQN training dashboard → http://127.0.0.1:8768/
.\.venv\Scripts\python.exe -m dms.rl.dash_dqn --port 8768

# CLI trainers (tabular & neural) with grading
.\.venv\Scripts\python.exe -m dms.rl.train_graph
.\.venv\Scripts\python.exe -m dms.rl.train_dqn --episodes 4000 --device auto

# Environment artifacts (layers.npz + graph.json)
python -m dms.tools.build_env --config configs/buildings/two_storey.yaml --out-dir out --visualize
```

---

## 12. Open questions we want a professional second opinion on

1. **DQN vs PPO here** (§9). Our position: stay on DQN for discrete routing; adopt PPO only if the action space becomes continuous steering via the SFM layer. Do you agree, or is there a reason to adopt PPO now (e.g., future-proofing, or robustness if the state grows with crowd/smoke features)?
2. **Discrete-route-RL / continuous-SFM split.** Is the right architecture "RL only on the discrete graph; physics-driven SFM for continuous motion" — or should the low-level motion also be learned (RL spends a lot of wall-clock for little gain at 10,000 agents)? What would you do at the 10k scale?
3. **Value-based vs policy-gradient trade-offs at plant scale.** When the state grows to include per-link `ρ̄` and `C̄` (N5 costs) and the population is 2,000–10,000, is off-policy replay still the right call, or do the non-stationary dynamics (crowds, evolving plume) tip the balance toward on-policy methods?
4. **Toxicology reconciliation** (§4.8): the agent-behaviour module uses speed factors {1.0, 0.8, 0.5, 0.0} and dose thresholds ≥1/≥20/≥100/≥500 ppm·min, while SYSTEM_SPEC uses {1.00, 1.00, 0.80, 0.40, 0.00} and a probit that the spec itself flags as effectively dominated by the band logic. Which set should be canonical, and should fatal-dose use Haber `C²·t` (exponent 2) or the CCPS `C^1.43·t` alternative?
5. **Density not in the potential field** (§4.4): the spec keeps dynamic density out of the Eikonal `F` by design (crowding = SFM repulsion + `w_ρ` in graph cost). Is that the right separation, or should `w_ρ` be folded back into a velocity/scalar field for smoother routing, as some of the source papers do?
6. **Validation sufficiency.** Beyond the SFM benchmark table (§4.5): against which real datasets (e.g., building-evacuation drill, Weidmann, SFPE handbook) should the crowd layer be validated before the 10k-scale runs are trustworthy?
7. **Governance of "true" stochasticity.** The goal is fully reproducible runs (per-agent seeded RNG, update-order independence). For evacuation statistics, even the spec's own protocol asks for 100-run means and 500-run P95s. Would a reviewer require a formal t-test/CI protocol, or are the current determinism + config-seed rules adequate for engineering decisions?

---
*Prepared from the repository and its source documents. Every number above is traceable to `SYSTEM_SPEC.md`, the parent-folder specifications, or a measured test/dashboard result; nothing is claimed tested that was not run.*