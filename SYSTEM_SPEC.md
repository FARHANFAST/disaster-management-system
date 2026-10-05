# SYSTEM_SPEC.md

**Source tags:** [M] Makmul 2023 · [L] Liu 2021 · [T] Templeton 2024 · [F] Fang 2022 · [Y] Yoon 2026 · [C] Chen 2021 · [S] Song 2025 · [D] Dou 2022 · [R] Senanayake 2024 · [X] derived/implementation assumption · **[U]** project default set by the project owner. It is not in the three sources and is not yet verified against a primary standard; keep it in config, not in code. **[EXT]** still absent: value is `None` and must be sourced externally before use.

---

## 1. UNIFIED SYSTEM ARCHITECTURE & COORDINATES

### 1.1 Frames
| Frame | Coords | Unit | Resolution | Role |
|---|---|---|---|---|
| L0 cell | `(y,x)` = (row, col), int, row-major | cell | $h_s=0.2$ m smoke/Eikonal [M]; $h_c=0.5$ m occupancy/CA [L] | fields, occupancy |
| L1 local | $\mathbf p=(x,y)$ | m | continuous | SFM kinematics inside building $b$ |
| L2 macro | graph $G=(N,E)$ | node/link | per room/stair/door | routing [Y] |
| L3 global | $(X,Y)$, X=East, Y=North, origin = plant SW corner | m | plant ≈1.312 km² [S] | dispersion, outdoor egress |

**Transforms** [X]. Canonical rule: row index increases with $+y$. [S] rasters use a southward row index, so flip $dN=-(row-row_s)h$ on import.
$$\mathbf p=\big((col+\tfrac12)h,\ (row+\tfrac12)h\big),\qquad (row,col)=(\lfloor y/h\rfloor,\lfloor x/h\rfloor)$$
$$\begin{bmatrix}X\\Y\end{bmatrix}=R(\psi_b)\mathbf p+\mathbf O_b,\quad R(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi\\ \sin\psi&\cos\psi\end{bmatrix}$$

L0→L2 mapping: `node_of[b][row,col] -> Node.id`. A multi-level node carries `floor`. Cells in the $h_s$ grid nest into the $h_c$ grid with ratio 2.5, so resample bilinearly [X].

### 1.2 Clocks
| Layer | Step | Value |
|---|---|---|
| Locomotion (SFM), dose accumulation | $\Delta t_L$ | 0.02 s [M]. [F] used $10^{-6}$ s; $10^{-3}$ s with semi-implicit Euler is stable since $\sqrt{k/m}\approx39$ rad/s [X]. Verify by Δt-halving. |
| Cognition, exit choice, CA | $\Delta t_C$ | 0.5 s $=25\,\Delta t_L$ [L] |
| Indoor smoke PDE | $\Delta t_L$ | CFL $\lvert w\rvert\Delta t/h_s=0.05$, $\kappa_d\Delta t/h_s^2=0.025$ |
| Outdoor dispersion | $\Delta t_D$ | 5–10 s (puff). Steady plume only on weather change [S]. |
| Outdoor macro movement | $\Delta t_O$ | 0.5 s $=\Delta t_C$ [X] |
| Potential fields (FMM) | event-driven | §1.6 |
| CA constraint | – | $v_{\max}\Delta t\le h$ |

### 1.3 Execution scale & phasing
| Phase | $N$ | Purpose | Acceptance |
|---|---|---|---|
| P0 Prototype & validation | **50** (default) | unit and physics tests | [F] N=50: 25.0 s, 1.93 p/s; IMO test 6 corner, no overlap |
| P1 Intermediate | 200–2,000 | literature benchmarks | §4.5: [M] 100/200, [F] ≤300, [S] 1500/2000/2500 |
| P2 Target stress | **10,000** | scalability, full plant | no code path changes from P0; profile per-tick cost |

`N` is a config value only. Every loop is O(N·k̄) through the spatial index in §1.4, with no O(N²) pair loops at any N. The same code runs in P0–P2.

### 1.4 Spatial indexing [X]
| Query | Structure | Cell or cutoff | Rebuild |
|---|---|---|---|
| SFM pair forces (S3–S4) | uniform-grid spatial hash, sorted by cell (CSR `cell_start`, `cell_count`) | $r_c=2.0$ m (repulsion at $d=r_{ij}+0.8$ is <0.1 N) | every $\Delta t_L$, O(N) |
| Wall forces (S5) | static segment hash, per building | $r_c$ | at init |
| Perception ($R_v$, $R_{vis}$, $R_{aud}$, $r_p$, VCA, density ρ) | `scipy.spatial.cKDTree`, `query_ball_point` | ≤5 m | every $\Delta t_C$ |
| Outdoor link density | per-link and per-cell counters | link / 5 m cell | every $\Delta t_O$ |

Pair forces run vectorised over the neighbour list (numpy, or numba for P2). Densities come from the same index.

### 1.5 Kinematic partitioning
| Domain | Model | Step | Scope |
|---|---|---|---|
| INDOOR_SFM | continuous SFM (S1–S5), neighbour lists | $\Delta t_L$ | building interiors, stairs, doors, and bottleneck zones within $R_{bn}=10$ m [X] of any outdoor gate or shelter door |
| OUTDOOR_MACRO | vector field on L3: $\mathbf x\leftarrow\mathbf x+\Delta t_O\,v_l\,\mathbf e_l$, with $v_l=\sigma\,v^0_i\,\phi\,(1-\bar\rho_l/\rho_{\max})$ | $\Delta t_O$ | plant open ground between buildings |

- $\mathbf e_l$ is the unit tangent of the current A* link. Inflow to a link is capped at $Q_{\max,l}$ and blocked agents queue.
- Hand-off happens at door nodes: INDOOR→OUTDOOR when an agent crosses an exterior door, and OUTDOOR→INDOOR when it enters a building, a shelter, or a zone of radius $R_{bn}$.
- On hand-off, keep $\mathbf v$ and snap position to the non-overlapping free point nearest the door.

### 1.6 Potential-field (FMM) scheduling
- **Init:** precompute a static $T^{eik}_{b,k}$ once per building $b$ and exit set $k$, with $F=U_{\max}$ on free cells and $F_b$ on obstacles.
- **Dynamic density is not in $F$.** Crowding is handled by SFM repulsion and by $w_\rho$ in (N5).
- **Recompute** only $T^{eik}_{b,\cdot}$ for the affected building when a dirty event fires:
  - a door or node changes state (closure, shutter, UNAVAILABLE);
  - an obstruction appears (a new INCAPACITATED agent inside a door or stair cell);
  - a cell's gas concentration first exceeds 100 ppm, which sets $F\leftarrow F_b$ on that cell;
  - a smoke cell first exceeds $C\ge C_{hi}$.
- **Debounce:** at most one recompute per building per $\Delta t_C$. A* costs are recomputed in the same event batch.

### 1.7 Tick loop
```
init: build index, static T_eik, graph G; sample t_pre; assign domain
for t in range(0, t_end, Δt_L):
  if t % Δt_D == 0: C_out = dispersion(t)             # §2.3
  C_in = smoke_step(C_in)                              # (M14)
  hash.rebuild(indoor agents)
  if t % Δt_C == 0:
      kdtree.rebuild(); perceive(); fsm_step()         # §4
      reselect_exits()                                 # (E1)
      dirty = detect_events()                          # §1.6
      recompute T_eik[b] for b in dirty; reroute()
      outdoor_macro_step(Δt_O)                         # §1.5
  sfm_step(indoor agents, Δt_L)                        # e, σ, φ, forces, integrator
  handoff(); dose(); health(); remove at-assembly
  stop if no living, non-assembled agents
```
Run protocol: 100 runs, mean [F]; 500 runs, P95 if non-convergent [C].

---

## 2. GOVERNING MATHEMATICAL EQUATIONS

### 2.1 Social & group dynamics

**Locomotion** (canonical dimensional form [F]; anisotropy [M]; $\mathbf{f}_b+\mathbf{f}_{adj}\approx0$ with incline via σ [X]):
$$\dot{\mathbf x}_i=\mathbf v_i,\quad m_i\dot{\mathbf v}_i=\mathbf f^{will}_i+\sum_{j\ne i}(\mathbf f^{soc}_{ij}+\mathbf f^{ph}_{ij})+\sum_W\mathbf f_{iW}+\mathbf f^{grp}_i+\boldsymbol\xi_i \tag{S1}$$
$$\mathbf f^{will}_i=m_i\frac{\sigma_i v^{des}_i\mathbf e_i-\mathbf v_i}{\tau} \tag{S2}$$
$$\mathbf f^{soc}_{ij}=A e^{(r_{ij}-d_{ij})/B}\,\mathbf n_{ij}\Big[\lambda+(1-\lambda)\tfrac{1+\cos\varphi_{ij}}2\Big],\quad \cos\varphi_{ij}=-\mathbf n_{ij}\cdot\hat{\mathbf v}_i \tag{S3}$$
$$\mathbf f^{ph}_{ij}=k\,g(r_{ij}-d_{ij})\mathbf n_{ij}+\kappa\,g(r_{ij}-d_{ij})\,\Delta v^t_{ji}\mathbf t_{ij} \tag{S4}$$
$$\mathbf f_{iW}=\big[Ae^{(r_i-d_{iW})/B}+k\,g(r_i-d_{iW})\big]\mathbf n_{iW}-\kappa\,g(r_i-d_{iW})(\mathbf v_i\!\cdot\!\mathbf t_{iW})\mathbf t_{iW} \tag{S5}$$
Here $g(z)=\max(z,0)$ (a ramp; the sources call it "Heaviside"), $\mathbf n_{ij}=(\mathbf x_i-\mathbf x_j)/d_{ij}$, $\mathbf t_{ij}=(-n^{(2)},n^{(1)})$, and $\Delta v^t_{ji}=(\mathbf v_j-\mathbf v_i)\cdot\mathbf t_{ij}$. In (S5), [F] prints $+\kappa$; the $-$ sign opposes slip [X].

Incline force [F]: $\lvert\mathbf f_b\rvert=m g\sin\theta$, directed down-slope. $\mathbf f_{adj}$ has no closed form; uphill stall occurs at θ≈15°.

**Deadlock** [F]: if $\lvert\sum\mathbf F\rvert=0$ for 3 consecutive ticks, apply $\boldsymbol\xi=\tfrac13\mathbf f^{will}$. Remove it once equilibrium breaks. Withhold it if the agent has no free space.

**Integrators:** Ralston RK2 [M]: $\mathbf k_1=f(t,\mathbf u)$, $\mathbf k_2=f(t+\tfrac23\Delta t,\mathbf u+\tfrac23\Delta t\mathbf k_1)$, $\mathbf u'=\mathbf u+\Delta t(\tfrac14\mathbf k_1+\tfrac34\mathbf k_2)$. Semi-implicit Euler is an alternative [F].

**Group cohesion** [T,R] (mechanism qualitative in the sources; form [X]; $A_g=800$ N and $d_g=1.5$ m [U]). The force acts between designated peer pairs $P_i$. The distance term saturates so the result is in N and bounded by $A_g$:
$$\mathbf f^{grp}_i=\frac{A_g}{\lvert P_i\rvert}\sum_{j\in P_i}\min\!\Big(1,\frac{g(d_{ij}-d_g)}{d_g}\Big)\frac{\mathbf x_j-\mathbf x_i}{d_{ij}},\qquad v^{0}_{G}=\min_{j\in G}v^0_j \tag{S6}$$
At the cap, $A_g$ exceeds the maximum will force ($80\cdot2.48/0.5\approx397$ N), so a separated agent turns back to its peer. This is the intended "meet-first" behaviour.
Missing members trigger SEEKING, with "meet-first" attraction before joint egress [R].

**Leader-follower and herding** [M] (see nav sub-FSM, §4.3):
$$\mathbf e_i=\begin{cases}\text{random }\mathbf d\in D & p=\alpha_f\\ (\mathbf x_{tgt}-\mathbf x_i)/\lvert\cdot\rvert & p=(1-\alpha_f)\beta_f\\ \hat{\mathbf v}_{tgt} & p=(1-\alpha_f)(1-\beta_f)\end{cases} \tag{S7}$$
$$\mathbf e^{herd}_i=\mathbf d_{m^*},\quad m^*=\arg\max_m\#\{j:\lvert\mathbf x_i-\mathbf x_j\rvert<R_v,\ md_j(t-\Delta t)=m\}\ \text{(ties random)} \tag{S8}$$
$$md_i=\arg\min_{\mathbf d_m\in D}\arccos\frac{\mathbf d_m\cdot\mathbf e_i}{\lvert\mathbf d_m\rvert},\quad D=\{(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1),(-1,0),(-1,1)\} \tag{S9}$$
If the snapped direction enters a wall, resample uniformly from the directions that point away from it.

**Exit choice (multinomial logit)** [L], over the known exits $K'$:
$$P_{ik}=\frac{e^{-\theta_{ik}T_{ik}}}{\sum_{l\in K'}e^{-\theta_{il}T_{il}}},\ \theta_{ik}=1/\delta_i^{s_k};\qquad T_{ik}=TF_{ik}+\alpha_i TW_{ik}\mathbb 1[Q_{ik}>Pa_i]+\beta_i(1-DG_k)\mathbb 1[OGK_i] \tag{E1}$$
$$TF_{ik}=\frac{d_{ik}}{\sum_K d_{il}},\ TW_{ik}=\frac{Q_{ik}}{\sum_K Q_{il}},\ DG_k=\frac{d_{kg}}{\sum_K d_{lg}},\ Q_{ik}=\frac{N_{ik}}{f_kw_k},\ \alpha_i=\frac{Pa^*}{Pa_i} \tag{E2}$$
[L] prints $+\beta_iDG_k$, which penalises exits far from the gas; $(1-DG_k)$ is the corrected sign. $N_{ik}$ counts agents inside the visual boundary and the SFF equipotential through $i$. Normalisation sums run over all $K$ exits.

**CA movement** (alternative locomotion) [L]: $P_{(y,x)}\propto e^{-k_SS^k_{(y,x)}}e^{-k_CC_{(y,x)}}(1-occ_{(y,x)})$ over the Moore neighbourhood. If every weight is 0, stay and add Δt to CWT.

### 2.2 Physical egress & topography

**Eikonal navigation** [M]:
$$\mathbf e_i=-\frac{\nabla T}{\lvert\nabla T\rvert},\quad\lvert\nabla T\rvert=\frac1F,\ T|_{exits}=0 \tag{N1}$$
$$F(\mathbf x)=\begin{cases}F_b & \mathbf x\in\Omega_b\ (\text{obstacle}\vee C\ge C_{hi})\\ U(\rho)&\text{else}\end{cases} \tag{N2}$$
[M] Eq. 6 uses $F_b=0.001$ and Algorithm 1 uses 0.01; this is a config value. $\Omega_b$ also includes cells with $C_{gas}>100$ ppm [U]. Solve (N1) with FMM (e.g. `skfmm.travel_time`) on the event schedule in §1.6, not every step as [M] does. Static fields use $F=U_{\max}$ outside $\Omega_b$.

**Speed** [M,F,R]:
$$U(\rho)=U_{\max}\Big(1-\frac{\rho}{\rho_{\max}}\Big),\quad \rho(\mathbf x)=\frac{\#\{j:\lvert\mathbf x-\mathbf x_j\rvert<R_v\}}{\pi R_v^2},\quad R_v=\frac{cV}{K_m\,\varepsilon M} \tag{N3}$$
$$v(\alpha,\rho)=\sigma(\theta,\theta_\perp,\lambda)\cdot v^0_i\Big(1-\frac\rho{\rho_{\max}}\Big)\cdot\phi(L_i),\qquad v\in[0.1,1.4]\ \text{m/s (SGEM)} \tag{N4}$$
The multiplicative composition in (N4) is [X]. $\phi$ is defined in §2.4.

$R_v$ scheduling [M]: one source, linear from 3.37 m to 2.0 m over the run; two sources, 1.68 m. $R_v$ is a global scalar, not coupled to local $C$. Coupling it is a TODO.

**Inclined-surface reduction** [F]: 4-Gaussian fits, θ in degrees, with trim θ∈[−30,30] (positive = uphill) and heel θ⊥∈[0,30]:
$$\sigma(\theta)=\sum_{i=1}^4e_i\exp\!\big[-((\theta-f_i)/g_i)^2\big],\qquad \sigma=\tfrac{2\lambda}{\pi}\sigma(\theta_\perp)+\tfrac{\pi-2\lambda}{\pi}\sigma(\theta),\ \lambda\in[0,\tfrac\pi2]$$
If λ>π/2, set λ←π−λ and θ←−θ [X].

| i | trim e | f | g | heel e | f | g |
|---|---|---|---|---|---|---|
|1|0.21|20.23|4.87|96.43|7.45|8.70|
|2|1.05|−5.01|16.14|−95.52|7.46|8.64|
|3|0.56|15.18|10.62|0.28|24.08|5.33|
|4|0.40|−20.56|6.16|1.55e5|−37.08|9.80|

Evaluate the heel fit in float64, since terms 1–2 nearly cancel. The alternative is the [F] Table 2 1° lookup: `TRIM[-30..30]`, `HEEL[0..30]`, range 0.088–1.075. These fits apply to ship-deck trim and heel only, not to stairs.

**Stair kinetics** [U], SFPE-type linear form:
$$v_{stair}(\rho_s)=\max\!\big(0.1,\ 0.70\,v_{flat}\,(1-0.266\,\rho_s)\big),\qquad v_{flat}=v^0_i\,\phi_i \tag{N4s}$$
- On stair links, (N4s) replaces the $(1-\rho/\rho_{\max})$ factor in (N4). Here $v_{flat}$ is the free (not density-reduced) speed, so density is not counted twice.
- $\rho_s$ is agents per m² of the stair link's plan area.
- (N4s) reaches the 0.1 m/s floor at $\rho_s\approx3.2$ ped/m² (for $v_{flat}=1.19$).

**Doorway and link discharge** [L,F,U]:
$$Q_{\max,l}=f_s\,w_{eff,l}\,\chi_l,\qquad \chi_l=\begin{cases}0.70&\text{stair}\\1&\text{level}\end{cases},\qquad f_s=1.92\ (\text{lit})\,/\,1.75\ (\text{dark})\ \text{ped/(m·s)}$$
Benchmark: a 1 m door measured 1.73–2.01 p/s [F]. In OUTDOOR_MACRO, $Q_{\max,l}$ caps link inflow per $\Delta t_O$.

**A\* edge weight** ([Y] lists the factors only; functional form [X]; weights [U]):
$$c_l(t)=\begin{cases}\infty & state_l=\text{UNAVAILABLE}\ \vee\ shutter_l\\ \dfrac{L_l}{\chi_l\sigma_lv_{ref}}\Big(1+w_\rho\dfrac{\bar\rho_l}{\rho_{\max}}+w_C\dfrac{\bar C_l}{C_{ref}}+w_{vis}\big(1-\tfrac{R_v}{R_{v,0}}\big)+w_T\dfrac{T_l-T_0}{T_{ten}-T_0}+w_H\big(1-\tfrac{d_{l,haz}}{R_H}\big)_{+}\Big)\,\omega_{p}^{\mathbb 1[\text{PARTIAL}]}\end{cases} \tag{N5}$$

Defaults:
- $w_\rho=1.5$, $w_C=3.0$ with $C_{ref}=30$ ppm (alarm level), $w_{vis}=0.5$ with $R_{v,0}=3.37$ m.
- $w_T=0$ (inactive), so $T_{ten}$ and $T_0$ are unused.
- $w_H=2.0$ with $R_H=161$ m (the [S] annual low-risk distance) [X].
- $\omega_p=2.0$. The matching capacity multiplier for a PARTIAL link is $1/\omega_p$ [X].
- $v_{ref}=1.19$ m/s.
Heuristic: $h(n)=\lVert\mathbf X_n-\mathbf X_{goal}\rVert/v_{\max}$ (admissible). Goals are outdoor assembly nodes only [Y].

**Occupant load** [Y]: $N_{15}=0.30N_{pk,1h}$; $N_{plat}=N_{15}+2C_{train}$; $N_{train}=N_{15}+C_{full}$; $N_{conc}=\tfrac{N_{hr}}2\cdot\tfrac5{15}$. [Y] applies this inconsistently (it used 945 onboard, 1699 total), so choose a convention explicitly.

**Criteria** [Y]: $T_{plat}\le240$ s; $T_{safe}\le360$ s; $T_{plat\to exit}\le270$ s.

### 2.3 Hazard dispersion

**Wind rotation** [X]: travel azimuth $\vartheta=(\phi_{from}+180°)\bmod360°$ (compass). With $dE=X-X_s$ and $dN=Y-Y_s$:
$$\begin{bmatrix}x\\y\end{bmatrix}=\begin{bmatrix}\sin\vartheta&\cos\vartheta\\ \cos\vartheta&-\sin\vartheta\end{bmatrix}\begin{bmatrix}dE\\dN\end{bmatrix},\qquad C=0\ \text{for}\ x\le0$$

**Continuous plume** [D, corrected]:
$$C=\frac{Q}{2\pi u\sigma_y\sigma_z}e^{-\frac{y^2}{2\sigma_y^2}}\Big[e^{-\frac{(z-H)^2}{2\sigma_z^2}}+e^{-\frac{(z+H)^2}{2\sigma_z^2}}\Big] \tag{D1}$$
Validity: steady state, $\rho_g\approx\rho_a$, non-reactive, flat terrain, $u\ge1$ m/s, $x\le10$ km. [L]'s $Q/(\pi u\sigma_y\sigma_z)$ form is the ground-source special case.

**Instantaneous puff** [D, corrected]:
$$C=\frac{M}{(2\pi)^{3/2}\sigma_x\sigma_y\sigma_z}e^{-\frac{(x-ut)^2}{2\sigma_x^2}}e^{-\frac{y^2}{2\sigma_y^2}}\Big[e^{-\frac{(z-H)^2}{2\sigma_z^2}}+e^{-\frac{(z+H)^2}{2\sigma_z^2}}\Big] \tag{D2}$$
Range ≤50 km. For a finite release, sum puffs emitted every $\Delta t_r$ with $M_k=Q\Delta t_r$ and $\sigma_x\approx\sigma_y$ [X]. The sum reproduces arrival time.

**Sutton** (light or neutral gas only) [D, corrected]:
$$C=\frac{M}{\pi C_yC_zu}e^{-\frac{y^2}{C_y^2x^{2-n_s}}}\Big[e^{-\frac{(z-H)^2}{C_z^2x^{2-n_s}}}+e^{-\frac{(z+H)^2}{C_z^2x^{2-n_s}}}\Big]$$

**Stability class** [D]:
| u (m/s) | Strong sun | Moderate sun | Slight sun | Night, >4/8 cloud | Night, <3/8 cloud |
|---|---|---|---|---|---|
|<2|A|A–B|B|F|F|
|2–3|A–B|B|C|E|F|
|3–4|B|B–C|C|D|E|
|4–6|C|C–D|D|D|D|
|>6|C|D|D|D|D|

**Briggs σ(x)**, with x in m and $x\leftarrow\max(x,1)$. Rural $\sigma_y=a_y x(1+10^{-4}x)^{-1/2}$; urban $\sigma_y=a_y x(1+4\cdot10^{-4}x)^{-1/2}$.
| Rural | $a_y$ | $\sigma_z$ | Urban (default for plant) | $a_y$ | $\sigma_z$ |
|---|---|---|---|---|---|
|A|0.22|$0.20x$|A–B|0.32|$0.24x(1+10^{-3}x)^{1/2}$|
|B|0.16|$0.12x$|C|0.22|$0.20x$|
|C|0.11|$0.08x(1+2\cdot10^{-4}x)^{-1/2}$|D|0.16|$0.14x(1+3\cdot10^{-4}x)^{-1/2}$|
|D|0.08|$0.06x(1+1.5\cdot10^{-3}x)^{-1/2}$|E–F|0.11|$0.08x(1+1.5\cdot10^{-3}x)^{-1/2}$|
|E|0.06|$0.03x(1+3\cdot10^{-4}x)^{-1}$| | | |
|F|0.04|$0.016x(1+3\cdot10^{-4}x)^{-1}$| | | |

Rows marked corrected against Briggs (rural E and F; urban A–B and C) should be verified against a primary source.

**Heavy gas.** Classify by $\rho_g/\rho_a$: <0.9 light, 0.9–1.1 neutral, >1.1 heavy. H₂S has ratio ≈1.18 (heavy), which puts (D1) and (D2) outside validity; implement the dense-gas path or flag the result.
- Britter–McQuaid: $C_M/C_0=f_c\big(x(u/V_{c0})^{1/2},\,g_0'V_{c0}^{1/2}u^{-5/2}\big)$ and $f_i\big(xV_{i0}^{-1/3},\,g_0'V_{i0}^{1/3}u^{-2}\big)$, with $g_0'=g(\rho_0-\rho_a)/\rho_a$. $f_c$ and $f_i$ are nomograms [EXT].
- Box model: $\dot R=C_E(g'L)^{1/2}$, $\dot V=\pi R^2u_e+2\pi RLw_e$, $V=\pi R^2L$, $L_0=R_0/2$. The centroid advects at $u$. $C_E$, $u_e$, $w_e$ are [EXT].
- Shallow layer: $\partial_th+\nabla\!\cdot(h\mathbf u)=0$ and $\partial_t(h\rho\mathbf u)+\nabla\!\cdot(h\rho\mathbf u\otimes\mathbf u)+\nabla[\tfrac12g(\rho-\rho_a)h^2]=0$.
- FEM-3 (CFD):
  - Conservation: $\nabla\!\cdot(\rho_c\mathbf u)=0$; $\partial_t(\rho_c\mathbf u)+\rho_c\mathbf u\!\cdot\!\nabla\mathbf u=-\nabla p+\nabla\!\cdot(\rho_cK^m\nabla\mathbf u)+(\rho_c-\rho_h)\mathbf g$.
  - Energy: $\partial_t\theta+\mathbf u\!\cdot\!\nabla\theta=\nabla\!\cdot(K^\theta\nabla\theta)+\frac{c_{pg}-c_{pa}}{c_{pc}}(K^\omega\nabla\omega)\!\cdot\!\nabla\theta+S$.
  - Species: $\partial_t\omega+\mathbf u\!\cdot\!\nabla\omega=\nabla\!\cdot(K^\omega\nabla\omega)$.
  - State: $\rho_c=PM/RT$, $M=[\omega/M_N+(1-\omega)/M_A]^{-1}$.
  - Diffusivities: $K_v=k_{vK}[(u_{*c}z)^2+(w_{*c}h)^2]^{1/2}/\Phi(Ri)$, $K_h=\beta^*u_{*c}z/\Phi(Ri)$, $u_{*c}=u_*\lvert u_c/u\rvert$.
  - Precompute wind fields, since they take 99% of CFD runtime.
- CA surrogate: $C^{k+1}_{ij}=\mathcal F(C^k_{ij},C^k_{i\pm1,j},C^k_{i,j\pm1},U^k_{ij})$, with $\mathcal F$ Gaussian-derived, SVM, or ANN. It runs 1.5–120× faster than CFD but drifts over long horizons, so assimilate data periodically.

**Indoor smoke** [M]: $\partial_tC+\mathbf w\!\cdot\!\nabla C=\kappa_d\nabla^2C+Q_c\delta_{ij,s}$. Use x/y operator splitting with explicit upwind advection and central diffusion [X], and Dirichlet $C=0$ on the boundary. The source term per cell is $Q_c\Delta t/h_s^2$.

**Units:** $\text{ppm}=C[\text{mg/m}^3]\cdot24.45/MW$ (25 °C, 1 atm). For H₂S, MW = 34.08.

**Risk zone** [S]: $Z^{(L)}=\bigcup_{s}\{\mathbf r:C_s(\mathbf r,t\le60\text{ min})\ge L\}$, computed as a cell-wise max over the four seasons.

### 2.4 Toxicological burden
$$L_i=\int C_i^{\,n}\,dt\approx\sum_t (C_i^t)^n\,\frac{\Delta t_L}{60},\qquad n=2.0\ [U],\quad C\ [\text{ppm}],\ t\ [\text{min}],\ L\ [\text{ppm}^2\text{·min}] \tag{X1}$$
[L] Eq. 3 weights $C$ by the step index; that is corrected here to Δt. The Cl₂ legacy case ($n=1$, g·min/m³) is kept only for reproducing [L].
$$Pr=a+b\ln L_i,\quad a=-15.67,\ b=1.024\ [U];\qquad P_{inc}=\Phi(Pr-5),\quad \text{INCAPACITATED if }P_{inc}\ge p_{inc}=0.50\ (Pr\ge5) \tag{X2}$$
$$v^0_{i,eff}=v^0_i\,\phi\big(B(C_i)\big),\qquad \phi=\{\phi_0,\dots,\phi_4\}=\{1.00,1.00,0.80,0.40,0.00\}\ [U] \tag{X3}$$
Bands by instantaneous local $C_i$ (ppm): B0 <0.5, B1 0.5–10, B2 10–100, B3 100–500, B4 ≥500. B4 means immediate collapse: φ=0 and INCAPACITATED.

⚠️ **Consistency check on the [U] probit.** At $Pr=5$, $L_{50}=e^{20.67/1.024}\approx5.9\times10^8$ ppm²·min. That gives $t_{50}\approx39$ h at 500 ppm and ≈9.8 h at 1000 ppm. So with these constants (X2) essentially never fires, and incapacitation is governed entirely by B4 in (X3).

Before production use, verify $a$, $b$ and $n$ against a primary source (TNO Green Book, CCPS). A frequently cited alternative, which must also be verified, is Crowl & Louvar / CCPS H₂S lethality: $Pr=-31.42+3.008\ln(C^{1.43}t)$ (ppm, min). It gives $Pr\approx5.5$ at 500 ppm for 30 min. Keep the probit set as a config key (`probit_set`).

⚠️ **Band 4 label.** B4 is labelled "IDLH" in the project brief, but the NIOSH IDLH for H₂S is 100 ppm, which is the B2/B3 edge. B4 (≥500 ppm) is a knockdown band, not the IDLH.

---

## 3. CANONICAL CONSTANTS & PARAMETER REGISTRY
| Parameter | Symbol | Unit | Default | Functional role |
|---|---|---|---|---|
| Agent count | $N$ | ped | 50 (P0) · 200–2,000 (P1) · 10,000 (P2) | phasing §1.3 [U] |
| SFM neighbour cutoff | $r_c$ | m | 2.0 | spatial hash [X] |
| Bottleneck SFM radius | $R_{bn}$ | m | 10 | domain hand-off [X] |
| Outdoor macro step | $\Delta t_O$ | s | 0.5 | vector-field update [X] |
| FMM gas trigger | $C_{fmm}$ | ppm | 100 | dirty event, $\Omega_b$ [U] |
| Locomotion step | $\Delta t_L$ | s | 0.02 | SFM, dose [M] |
| Cognition step | $\Delta t_C$ | s | 0.5 | FSM, logit, CA [L] |
| Dispersion step | $\Delta t_D$ | s | 5–10 | puff update [S] |
| Grid cell (smoke / CA) | $h_s,h_c$ | m | 0.2 / 0.5 | L0 [M,L] |
| Agent mass | $m$ | kg | 80, Normal (SD [EXT]) | S1 [F] |
| Body radius | $r_i$ | m | 0.2 [F]; 0.25 ($r_{ij}$=0.5) [M] | contact |
| Shoulder breadth (M/F, as published) | – | mm | 297.43±18.89 / 352.95±16.14; max 494/414; sim uses max | [Y]. Columns likely swapped. |
| Relaxation time | $\tau$ | s | 0.5 | S2 |
| Desired speed (emergency) | $v^0$ | m/s | 2.48 [F]; 1.19 plant [S]; $U_{\max}$=1.65 [M] | S2 |
| Jam density | $\rho_{\max}$ | ped/m² | 10 | N3 |
| Social strength | $A$ | N | 2000 (mass-normalised 0.1 [M] is inconsistent) | S3 |
| Repulsion range | $B$ | m | 0.08 [F]; 0.21 [M] | S3 |
| Anisotropy | $\lambda$ | – | 0.61 | S3 [M] |
| Body stiffness | $k$ | kg/s² | 1.2e5 | S4 |
| Sliding friction | $\kappa$ | kg/(m·s) | 2.4e5 | S4 |
| Deadlock window / ratio | – | ticks / – | 3 / 1/3 | ξ [F] |
| Group attraction strength | $A_g$ | N | 800.0 | S6, peer pairs [U] |
| Cohesion radius | $d_g$ | m | 1.5 | S6 [U] |
| Random-dir prob. / seek-target prob. | $\alpha_f,\beta_f$ | – | 0.2 / 0.3 | S7 [M] |
| Guider fraction | – | % | 3 (tested 0, 1, 3, 5) | leaders [M] |
| Blocked front speed | $F_b$ | m/s | 0.001 (alt. 0.01) | N2 |
| High-smoke threshold | $C_{hi}$ | g/m² | 0.05 | N2 |
| Visibility | $R_v$ | m | 3.37→2.0 (1 source); 1.68 (2 sources) | N3 |
| Sign constant | $c$ | – | 8 emitting / 3 reflecting | N3 |
| Extinction coefficient | $K_m$ | m²/g | 7.6 flaming / 4.42 pyrolysis | N3 |
| Smoke yield; mass burned; volume | $\varepsilon,M,V$ | –, g, m³ | 0.15; 1000; 1280 | N3 |
| Smoke diffusion; wind components | $\kappa_d$; $w_{1,2}$ | m²/s; m/s | 0.05; U[−0.5, 0.5] per step | smoke |
| Smoke source rate | $Q_c$ | g/s | 10 at t=0, then 0.1 | smoke |
| Stair speed multiplier | – | – | 0.70 | N4s [U] |
| Stair density slope | – | m²/ped | 0.266 | N4s [U] |
| Stair capacity factor | $\chi_{stair}$ | – | 0.70 (−30% vs level) | $Q_{\max}$ [U] |
| Incline factor | σ | – | 0.088–1.075 | N4 [F] |
| Speed bounds | – | m/s | 0.1–1.4 | N4 [R] |
| Specific door flow | $f_s$ | ped/(m·s) | 1.92 lit / 1.75 dark | Q_max [L] |
| Congestion weight | $w_\rho$ | – | 1.5 | N5 [U] |
| Gas weight / reference | $w_C$ / $C_{ref}$ | – / ppm | 3.0 / 30.0 | N5 [U] |
| Visibility weight | $w_{vis}$ | – | 0.5 | N5 [U] |
| Temperature weight | $w_T$ | – | 0.0 (inactive) | N5 [U] |
| Hazard-proximity weight / range | $w_H$ / $R_H$ | – / m | 2.0 / 161 | N5 [U]/[X] |
| PARTIAL penalty | $\omega_p$ | – | 2.0 (capacity ×1/ω_p) | N5 [U] |
| Floor-field sensitivity | $k_S$; $k_C$ | –; m³/mg | 5 (dark), 10 (lit); None | CA |
| Exit preference | $\delta^{s_k}$ | – | upwind 0.8 / downwind 0.2 (θ = 1.25 / 5) | E1 |
| Patience (mean) | $Pa^*$, $Pa_i$ | s | 10; $Pa_i=Pa^*$ (distribution [EXT]) | E1 |
| Impatience / danger weights | $\alpha_i,\beta_i$ | – | 1 / 1 | E1 persona |
| Perception radii | $R_{vis},R_{aud}$ | m | 5 / 2.5 (tested 1–10) | OGK/OEK |
| VCA radius | – | m | 5 | reselect |
| Trust count | $N_{trust}$ | ped | 5 (tested 2–20) | info uptake |
| Peer trigger | $r_p,p_p$ | m, – | 2.0, 0.5 once per pair | [S] |
| Pre-evac, plant | $t_{pre}$ | s | $\mathcal N^+(900,300)$ | [S] |
| Pre-evac, station | $t_{pre}$ | s | 120 fixed (W1) | [Y] |
| Pre-evac, general (default) | $\ln t_{pre}\sim\mathcal N(\mu,\sigma)$ | ln s | μ=3.8, σ=0.6 (median 44.7 s, mean 53.5 s) | normalcy bias [U] |
| Panic threshold / neighbour ratio | $\theta_{pan}$ | – | None; disabled by default [T] | FSM |
| Stability class | – | – | [S] u=1.25: A–B day / F night; set per scenario | σ(x) |
| Release (validation case) | $Q,H,u$ | g/s, m, m/s | 16 667 (1 t/min), 1, 1.25 SE | [S] |
| H₂S band edges | B0–B4 | ppm | 0.5 / 10 / 100 / 500 | X3, health [U] |
| Speed factor by band | $\phi_{0..4}$ | – | 1.00 / 1.00 / 0.80 / 0.40 / 0.00 | X3 [U] |
| Haber exponent | $n$ | – | 2.0 | X1 [U] |
| Probit constants | $a,b$ | – | −15.67, 1.024 (see ⚠ in §2.4) | X2 [U] |
| Incapacitation probability | $p_{inc}$ | – | 0.50 (Pr = 5) | X2 [U] |
| Cl₂ TL bands (legacy [L] repro) | – | g·min/m³ | 0.48 / 3.2 / 6.4 | health [L] |
| von Kármán; β* | $k_{vK}$; β* | – | 0.41; 6.5 | FEM-3 |

**Walking speed** [Y] (m/s, M/F): child 1.0/1.0; adolescent 1.3/1.3; adult 1.2/1.1; elderly 0.7/0.97 (garbled cell, source text says ≤0.7); disabled 0.5/0.5. Mix: adult M/F 40/40, elderly M/F 10/10 %.

**Pre-movement windows** (NFA, min, W1/W2/W3): office <1/3/>4; shop <2/3/>6; residential <2/4/>5; hotel <2/4/>6; care <3/5/>8.

---

## 4. DISCRETE STATE SCHEMAS & TRANSITION LOGIC

### 4.1 Agent FSM
Source mapping:
- NORMAL = IDLE / PRE_MOVEMENT, unaware.
- ALERT = aware but still inside the delay.
- DECIDING = one $\Delta t_C$ spent on logit and route.
- MOVING_TO_ASSEMBLY has substates {EXIT_ZONE, TO_SAFETY} × nav {EIKONAL, FOLLOW_GUIDER, FOLLOW_WALL, FOLLOW_STREAM}.
- INCAPACITATED = DEAD or frozen.
- AT_ASSEMBLY is absorbing and is reached only at an outdoor assembly point or shelter [Y,S].

| From → To | Trigger |
|---|---|
| NORMAL → ALERT | $OGK_i{:}0{\to}1$ if health ≥ MINOR ($C_i\ge10$ ppm) ∨ source within $R_{vis}$ ∨ dead agent within $R_{vis}$ ∨ $\#\{j:OGK_j{=}1,d_{ij}\le R_{aud}\}>N_{trust}$ ∨ zone self-trigger ∨ peer trigger ∨ global alarm |
| NORMAL → ALERT (zone) | Zone 3: p=1.00, $t\sim U(5,20)$ s; Zone 2: 0.75, $U(10,35)$; Zone 1: 0.50 (Table: 0.55), $U(20,60)$ [S] |
| ALERT → DECIDING | $t\ge t_{start}=\min(t_{alarm},t_{self},t_{peer})$, where $t_{alarm}=t_0+t_{pre}$. Default $t_{pre}\sim\text{LogNormal}(3.8,0.6)$ s; validation profiles use $\mathcal N^+(900,300)$ [S] or 120 s fixed [Y]. |
| DECIDING → MOVING | $k^*\sim$(E1); A* route computed; in $Z$ ⇒ EXIT_ZONE (target = nearest one-way door), else TO_SAFETY |
| MOVING → DECIDING | if `allow_reselect` ([L] true, [S] false): first entry into $VCA_k$ ∨ $CWT_i>Pa_i$ (then CWT←0) ∨ OGK 0→1 ∨ any $OEK_{ik}$ 0→1 |
| EXIT_ZONE → TO_SAFETY | one-way door out of $Z$ crossed; $Z$ becomes non-enterable |
| ALERT, DECIDING, MOVING → PANICKED | `panic_enabled` ∧ $\theta_{pan}$ exceeded [EXT]; panic is rare, with no proximity contagion [T]. Nav = $\mathbf e^{herd}$ (S8) |
| PANICKED → MOVING | [EXT] |
| any living → INCAPACITATED | $C_i\ge500$ ppm (B4, φ=0) ∨ $P_{inc}\ge0.50$ (X2). The agent freezes, becomes an obstacle and a visual cue, and fires an FMM dirty event if it is in a door or stair cell. |
| MOVING → AT_ASSEMBLY | reaches an outdoor assembly point or shelter; removed from the grid |

State locking [S]: once past ALERT, no trigger re-enters NORMAL or ALERT.

$OEK_{ik'}{:}0{\to}1$ when exit $k'$ is within $R_{vis}$, or when $\#\{j\in R_{vis}:tgt_j=k',\ hd_j\ne hd_i\}>N_{trust}$. Each agent starts knowing its entry exit.

**Orthogonal health state** (H₂S, [X] mapping of the [U] bands; worst state reached is latched):
- UNHARMED: B0–B1.
- MINOR: B2. Also sets OGK=1.
- SERIOUS: B3. Sets RESCUE_REQUIRED [R].
- DEAD / INCAPACITATED: B4, or $P_{inc}\ge0.5$.

Legacy Cl₂ thresholds from [L] (0.48 / 3.2 / 6.4 g·min/m³) apply only in the `cl2_liu` profile.

**Social constraints** [T]:
- Weight information uptake by in-group membership, tie strength, and source trust.
- Helping behaviour should be more common than competing.
- Group identity reduces pushing.
- Compliance depends on who the source is, whether a reason is given, and whether guidance fits the agent's own knowledge.
- Information-seeking extends $t_{pre}$ (unquantified).

### 4.2 Kinematic sub-FSM [F]
`COMPUTE_HEADING → λ → σ → CHECK_IMPASSE`
- If $\lvert\Sigma\mathbf F\rvert=0$ for 3 ticks: DEADLOCK, apply ξ.
- If there is no free space: WAIT.
- Otherwise: CHECK_TARGET.

### 4.3 Nav sub-FSM [M] (non-guider, evaluated every tick)
Priority order: guider within $R_v$ (pick one at random) → FOLLOW_GUIDER; exit within $R_v$ → EIKONAL; wall within $R_v$ → FOLLOW_WALL (left or right, p=0.5); otherwise FOLLOW_STREAM (S8). Guiders are always EIKONAL. Mix directions via (S7).

**Infrastructure masks** [Y], fixed per scenario:
- E1–2, S1–8: AVAILABLE | PARTIAL | UNAVAILABLE.
- F1–3: AVAILABLE | UNAVAILABLE.
- FS1–3: OPERATIONAL | NOT_OPERATIONAL.
- PARTIAL: cost ×$\omega_p=2.0$ and capacity ×$1/\omega_p$ [U]/[X].

### 4.4 Data structures
```python
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np

class St(Enum): NORMAL=auto(); ALERT=auto(); DECIDING=auto(); MOVING_TO_ASSEMBLY=auto(); PANICKED=auto(); INCAPACITATED=auto(); AT_ASSEMBLY=auto()
class Sub(Enum): EXIT_ZONE=auto(); TO_SAFETY=auto()
class Nav(Enum): EIKONAL=auto(); FOLLOW_GUIDER=auto(); FOLLOW_WALL=auto(); FOLLOW_STREAM=auto(); HERD=auto()
class Health(Enum): UNHARMED=auto(); MINOR=auto(); SERIOUS=auto(); DEAD=auto()
class Domain(Enum): INDOOR_SFM=auto(); OUTDOOR_MACRO=auto()

PHI = (1.00, 1.00, 0.80, 0.40, 0.00); BAND_EDGES = (0.5, 10.0, 100.0, 500.0)   # ppm
def band(c_ppm: float) -> int: return int(np.searchsorted(BAND_EDGES, c_ppm, side="right"))

@dataclass(slots=True)
class AgentState:
    id: int; building: int | None; node: str
    x: np.ndarray; v: np.ndarray            # L1 (x,y) m if INDOOR_SFM; L3 (X,Y) if OUTDOOR_MACRO
    domain: Domain = Domain.INDOOR_SFM; link: tuple[str, str] | None = None
    m: float = 80.0; r: float = 0.2; v0: float = 1.19; tau: float = 0.5
    group: int | None = None; peers: tuple[int, ...] = ()   # S6 designated pairs
    is_guider: bool = False; persona: str = "adult_M"
    state: St = St.NORMAL; sub: Sub | None = None; nav: Nav = Nav.EIKONAL
    health: Health = Health.UNHARMED
    t_start: float = float("inf"); OGK: int = 0
    known_exits: set[int] = field(default_factory=set); visited_vca: set[int] = field(default_factory=set)
    target_exit: int | None = None; route: list[str] = field(default_factory=list)
    Pa: float = 10.0; CWT: float = 0.0; alpha_i: float = 1.0; beta_i: float = 1.0
    dose: float = 0.0; phi: float = 1.0; zero_force_ticks: int = 0   # dose in ppm²·min
    peer_tested: set[int] = field(default_factory=set); heading_bin: int = 0

@dataclass(slots=True)
class ConcentrationGrid:
    data: np.ndarray                        # [row=y, col=x], row ↑ = +y
    h: float; origin_XY: tuple[float, float]; psi: float = 0.0
    unit: str = "ppm"                       # "ppm" | "g/m2" (indoor smoke) | "mg/m3"
    z: float = 1.5; t: float = 0.0; species: str = "H2S"; mw: float = 34.08
    def cell(self, p: np.ndarray) -> tuple[int, int]:
        return int(p[1] // self.h), int(p[0] // self.h)

@dataclass(slots=True)
class Node:
    id: str; floor: str; kind: str          # platform|vertical|transition|hazard_ctrl|terminal
    area_m2: float; width_m: float; capacity: int; XY: tuple[float, float]

@dataclass(slots=True)
class Link:
    u: str; v: str; length_m: float; width_m: float; slope_deg: float; oneway: bool
    state: str = "AVAILABLE"; shutter_closed: bool = False
    rho: float = 0.0; C: float = 0.0; temp: float = 20.0; d_haz: float = float("inf")

@dataclass(slots=True)
class EnvironmentSnapshot:
    t: float; wind_u: float; wind_from_deg: float; stability: str; season: str
    source_XY: tuple[float, float]; Q_gps: float; H_m: float
    C_out: ConcentrationGrid; C_in: dict[int, ConcentrationGrid]
    T_eik: dict[tuple[int, int], np.ndarray]   # (building, exit_set) -> static field (§1.6)
    fmm_dirty: set[int]                        # buildings awaiting recompute this Δt_C
    rho: dict[int, np.ndarray]
    nodes: dict[str, Node]; links: dict[tuple[str, str], Link]
    zone_Z: np.ndarray                      # bool mask, L3
    Rv: float; alarm_t: float | None; assembly: set[str]

@dataclass(slots=True)
class SpatialHash:                           # rebuilt every Δt_L; O(N)
    cell: float = 2.0                        # = r_c
    keys: np.ndarray | None = None           # per-agent flat cell id
    order: np.ndarray | None = None          # agent ids sorted by cell
    cell_start: dict[int, int] = field(default_factory=dict)
    cell_count: dict[int, int] = field(default_factory=dict)
    def neighbors(self, i: int) -> np.ndarray: ...   # agents in the 3x3 cells around i

@dataclass(slots=True)
class SimConfig:
    N: int = 50; phase: str = "P0"           # P0=50 | P1=200-2000 | P2=10000
    dt_L: float = 0.02; dt_C: float = 0.5; dt_O: float = 0.5; dt_D: float = 5.0
    probit_set: str = "project_default"      # a=-15.67, b=1.024, n=2.0
    tpre_profile: str = "lognormal"          # lognormal(3.8,0.6) | song_normal | yoon_120
    allow_reselect: bool = True; panic_enabled: bool = False
```

### 4.5 Validation targets
| Case | Target |
|---|---|
| [M] 100 agents / 1 exit, 50 s | evacuated 38.7 (no guider) → 44.4 (3% guiders) |
| [M] 100 agents / 2 exits | 77.7 → 91.1 |
| [M] 200 agents / 2 exits | 119.2 → 191.5 (3%), 193.7 (5%) |
| [F] 15×15 m room, 1 m exit, N=50 | 25.0 s (range 19.4–33.9), 1.93 p/s |
| [F] N = 100 / 150 / 200 / 250 / 300 | 2.01 / 1.87 / 1.80 / 1.79 / 1.73 p/s |
| [F] trim 25° / 30° (N=100) | 0.90 / 0.25 p/s; 374 s at 30° |
| [F] incline speed, θ=20° | 2.07 m/s (λ=0°), 1.39 m/s (λ=90°) |
| [S] Sim 1 / Sim 2 (N=2000) | 1359.03 s (E1–3: 1141/636/223) / 1647.3 s (1014/429/557) |
| [S] Sim 6 (exits added on E, S, W) | 1103.8 s |
| [S] Sim 7 (two shelters) | 869.5 s |
| [S] Sim 8 vs Sim 2 | slower before ≈520 s, faster after |
