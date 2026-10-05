# PROMPT 01: Two-Storey Building Environment (L0 grids + L2 graph + L3 site)

> Give this prompt to the coding agent **together with `SYSTEM_SPEC.md`**. The spec is binding: its frames, clocks, symbols, dataclass names and parameter values win over anything here unless this prompt says otherwise.

---

## 0. Your role and the goal of this step

You are implementing the **environment foundation** of an AI-based, agent-based disaster evacuation simulator for a nuclear/industrial plant. The project is a PAEC / NUST AI Solution Lab effort (Dr. Mazhar Sajjad). Later phases add:
- agents (social force model, logit exit choice, FSM);
- indoor smoke and H₂S transport;
- outdoor Gaussian/puff dispersion with wind;
- a whole-plant scale-up to 10,000 agents.

**This step builds only the world those modules will live in**: one realistic two-storey building on an outdoor site. It has:
- multiple rooms of different sizes;
- a looped corridor network, so there is always an alternative path;
- three staircases;
- five usable exits plus one locked service door;
- outdoor assembly points on all four sides;
- hazard source locations;
- windows, and furniture/clutter obstacles;
- dynamic door, link and exit states.

The environment must be **built once from a declarative file** and must expose everything the later modules need:
- occupancy grids at two resolutions;
- a routing graph for A*;
- Eikonal travel-time fields;
- wall segments for SFM;
- a gas-permeability topology for the smoke/H₂S solvers.

The single most important quality is this: **if any one corridor segment, door, stair or exit is blocked, routing and potential fields immediately find the next-best path, and tests prove it.**

---

## 1. Scope

**In scope**
1. A declarative building + site description (YAML, metres), with a loader and a validator.
2. A geometry model: floors, rooms, corridors, doors, stairs, exits, voids, windows, obstacles, hazard sources.
3. Rasterisation to L0 grids at $h_c=0.5$ m (occupancy/CA) and $h_s=0.2$ m (smoke/Eikonal), per floor, from the same vector geometry.
4. Construction of the L2 graph $G=(N,E)$ using the spec's `Node` / `Link` dataclasses, plus A* with the (N5) edge cost.
5. Static Eikonal fields $T^{eik}_{b,k}$ per floor and exit set (FMM via `scikit-fmm`), including correct multi-floor coupling through stairs (§6.3).
6. A dynamic state and event layer: door, link and exit states; door leaf open/closed; obstacle insertion. It emits the "dirty building" events of spec §1.6.
7. An L3 outdoor site with the building placed at $\mathbf O_b$, walkable ground, parking obstacles and assembly points.
8. Visualisation, a validation report, a redundancy report, and a pytest suite.

**Out of scope for this step** (leave clean interfaces only; do not implement):
- the SFM integrator and agents' behaviour;
- the smoke PDE solver and the dispersion models;
- dose and health;
- the FSM.

You may write a trivial "ghost walker" that follows `-∇T` or an A* route, purely for visual checks.

---

## 2. Design principles (non-negotiable)

1. **Single source of truth.** All geometry lives in `configs/buildings/two_storey.yaml` in building-local metres (L1 frame). Code contains **no coordinates**. Grids, graph, wall segments and gas topology are all *derived* and must be regenerable at any `h`.
2. **Spec alignment.**
   - Use `SYSTEM_SPEC.md` names: `Node`, `Link`, `ConcentrationGrid`, `EnvironmentSnapshot`, `SimConfig`, and the enums.
   - You may **add** fields to the spec dataclasses; do not rename or remove them.
   - Arrays are `[row=y, col=x]`, and row index increases with $+y$ (north).
   - The transforms are those of spec §1.1, extended with a grid origin offset (grids have a margin; §5.1).
3. **Static vs dynamic separation.**
   - Static geometry is compiled once.
   - Dynamic state (door state, leaf open/closed, link state, added obstacles, parked cars on/off) is an **overlay** applied to the compiled layers.
   - Changing a door state never recompiles the building. It updates the overlay masks, marks the building dirty (spec §1.6, debounced to one recompute per $\Delta t_C$), and updates the A* costs in the same batch.
4. **Multi-building ready.** A `Site` holds a list of `Building`s. Each has an id `b`, an origin $\mathbf O_b$ and a rotation $\psi_b$. This step uses one building, but nothing may assume only one exists.
5. **Scalable by construction.**
   - No O(N²) loops anywhere.
   - All per-cell work is vectorised numpy.
   - Graph size is O(rooms + corridor segments).
6. **Deterministic.** The same YAML and config produce bit-identical arrays and graph (tested by hashing). All randomness goes through a seeded `numpy.random.Generator`.
7. **Stack:**
   - Python ≥ 3.11: `numpy`, `scipy`, `shapely` (geometry and rasterisation), `networkx` (graph container; A* is ours, so costs stay under our control), `scikit-fmm`, `pyyaml`, `matplotlib`, `pytest`.
   - Type hints everywhere; `dataclass(slots=True)`; `ruff` clean.

---

## 3. The building (authoritative layout)

### 3.1 Global facts
| Item | Value |
|---|---|
| Footprint (both floors) | 60 m (x, east) × 40 m (y, north); local origin = SW outer corner |
| Floors | `GF` at z = 0.0 m and `FF` at z = 4.0 m (floor-to-floor 4.0 m, clear ceiling 3.5 m) |
| Wall thickness | 0.2 m for all walls (a config value) |
| Corridor width | 3.0 m (ring and spine) |
| Coordinates | All values in the tables below are wall centre-lines / room boundaries in metres. A door is given as the wall it sits on plus its span. |

**Band structure (identical on both floors, not to scale):**
```
y=40 +------------------------ NORTH BAND (rooms) -------------------------+
     |          |                 north stub                               |
y=30 |  ST-W  +=========== RING corridor (north leg) =================+    |
     | (0-8)  ||    WEST CORE        ||S||      EAST CORE             ||   |
y=21 | W stub ||   (x 11-28.5)       ||P||     (x 31.5-49)            || E stub
y=18 |        ||                     ||I||  ST-C                      ||   |
     |        ||                     ||N||                            ||   |
y=13 |        +=========== RING corridor (south leg) =================+    |
y=10 |                                                                     |
     |                  SOUTH BAND (rooms)   LOBBY (x 25-35)               |
y=0  +---------------------------------------------------------------------+
    x=0     x=8/11                x=28.5-31.5                    x=49/52   x=60
```
- **Ring corridor:** the outer rectangle x 8–52, y 10–30, minus the core x 11–49, y 13–27. So there are four 3 m legs: south (y 10–13), north (y 27–30), west (x 8–11) and east (x 49–52).
- **Spine:** x 28.5–31.5, y 13–27. It is open (no door) to the ring at both ends.
- The ring and spine form **two independent loops**. This is the core redundancy mechanism.

### 3.2 Ground floor (GF)
Door notation: `wall@coordinate: span`. For example, `y=10: x 9.0–10.5` is a 1.5 m door in the wall y=10 between x=9.0 and x=10.5. "opening" means there is no door leaf (always passable, gas-open).

| ID | Name | Rect (x0–x1, y0–y1) | Area m² | Doors / openings | Notes |
|---|---|---|---|---|---|
| G-01 | Workshop | 0–12, 0–10 | 120 | y=10: x 9.0–10.5 | furniture: 2 workbenches |
| G-03 | Office | 12–18.5, 0–10 | 65 | y=10: x 15.0–16.0 | single door |
| G-04 | Office | 18.5–25, 0–10 | 65 | y=10: x 21.0–22.0 | single door |
| LOBBY | Main lobby | 25–35, 0–10 | 100 | opening y=10: x 28–32; **exit X-S** y=0: x 28.8–31.2 (2.4 m double) | FF above is a VOID (atrium) |
| G-05 | Cafeteria | 35–47, 0–10 | 120 | y=10: x 37.0–38.5 and x 44.0–45.5; x=35: y 3.0–4.5 (to lobby) | **through-room**: bypasses the lobby opening; tables as obstacles |
| G-06 | Chemical / gas store | 47–60, 0–10 | 130 | y=10: x 49.5–51.0; **X-SE** service door x=60: y 3.0–4.5 | X-SE default `UNAVAILABLE` (locked); hazard H-1 |
| G-07 | Office | 0–8, 10–18.5 | 68 | x=8: y 14.0–15.0 | |
| W-stub | West exit corridor | 0–8, 18.5–21.5 | 24 | opening x=8 (full width); **exit X-W** x=0: y 19.4–20.6 (1.2 m) | |
| G-08 | Toilets | 0–8, 21.5–24 | 20 | x=8: y 22.2–23.2 | **designed dead-end** |
| ST-W | West stair (enclosed) | 0–8, 24–30 | 48 | fire door x=8: y 26.0–27.5; **exit X-NW** x=0: y 26.0–27.5 (direct discharge) | FF→outside without entering the GF corridor |
| ST-E | East stair (enclosed) | 52–60, 10–15 | 40 | fire door x=52: y 12.0–13.5 | discharges into the GF ring |
| G-09 | Office | 52–60, 15–18.5 | 28 | x=52: y 16.5–17.5 | |
| E-stub | East exit corridor | 52–60, 18.5–21.5 | 24 | opening x=52 (full width); **exit X-E** x=60: y 19.4–20.6 (1.2 m) | |
| G-10 | Server room | 52–60, 21.5–30 | 68 | x=52: y 25.0–26.0 | |
| G-11 | Main control room (C2-MCR) | 0–14, 30–40 | 140 | y=30: x 9.0–10.5 and x 12.0–13.5 | consoles as obstacles; named in the project slides |
| G-12 | Meeting room | 14–22, 30–40 | 80 | y=30: x 17.0–18.0 | |
| G-13 | Office | 22–28.5, 30–40 | 65 | x=28.5: y 33.0–34.0 (opens onto the N-stub, not the ring) | |
| N-stub | North exit corridor | 28.5–31.5, 30–40 | 30 | opening y=30 (full width); **exit X-N** y=40: x 29.1–30.9 (1.8 m) | |
| G-14 | Records / archive | 31.5–40, 30–40 | 85 | y=30: x 35.0–36.0 | |
| G-15 | Laboratory | 40–60, 30–40 | 200 | y=30: x 43.0–44.5 and x 49.0–50.5 | hazard H-2; benches |
| G-16 | Open-plan office | 11–20, 13–27 | 126 | y=13: x 14.0–15.5; x=11: y 20.0–21.5; y=27: x 16.0–17.5 | **3-door through-room**; desk rows |
| G-17 | Office | 20–28.5, 13–20 | 60 | y=13: x 23.0–24.0 | |
| G-18 | Office | 20–28.5, 20–27 | 60 | x=28.5: y 23.0–24.0 (onto spine) | |
| ST-C | Central stair (open, main) | 31.5–37.5, 13–19 | 36 | opening y=13: x 33.0–35.0; opening x=31.5: y 15.0–17.0 | no fire doors; gas can travel through it |
| G-20 | Store | 31.5–37.5, 19–27 | 48 | x=31.5: y 22.0–23.0 | |
| G-19 | Training room | 37.5–49, 13–27 | 161 | y=13: x 40.0–41.5; x=49: y 20.0–21.5; y=27: x 45.0–46.5 | **3-door through-room** |

**GF exits** (terminal nodes):
- X-S: 2.4 m, main.
- X-N: 1.8 m.
- X-W: 1.2 m.
- X-E: 1.2 m.
- X-NW: 1.5 m, stair discharge.
- X-SE: 1.5 m, locked by default.

There are exits on all four façades, which is what the upwind/downwind preference $\delta^{s_k}$ in (E1) needs.

### 3.3 First floor (FF)
The ring, spine and stairs are in the same positions. There are no exterior exits on FF, so its only egress is the three stairs.

| ID | Name | Rect | Area m² | Doors / openings | Notes |
|---|---|---|---|---|---|
| F-01 | Office | 0–12, 0–10 | 120 | y=10: x 9.0–10.5 | |
| F-02 | Open office | 12–25, 0–10 | 130 | y=10: x 14.0–15.5 and x 22.0–23.5 | |
| VOID | Atrium over lobby | 25–35, 0–10 | – | balustrade along y=10: x 25–35 | **not walkable, gas-open**; couples GF↔FF vertically |
| F-03 | Office pool | 35–47, 0–10 | 120 | y=10: x 37.0–38.5 and x 44.0–45.5 | |
| F-04 | Office | 47–60, 0–10 | 130 | y=10: x 49.5–51.0 | |
| F-05 | Office | 0–8, 10–24 | 112 | x=8: y 14.0–15.0 and y 21.0–22.0 | |
| ST-W | West stair | 0–8, 24–30 | – | fire door x=8: y 26.0–27.5 | |
| ST-E | East stair | 52–60, 10–15 | – | fire door x=52: y 12.0–13.5 | |
| F-06 | Office | 52–60, 15–30 | 120 | x=52: y 17.0–18.0 and y 27.0–28.0 | |
| F-07 | Conference hall | 0–28.5, 30–40 | 285 | y=30: x 9.0–10.5 and x 20.0–21.5; x=28.5: y 35.0–36.5 (to FF N-stub) | large, 3 doors; seating rows |
| FN-stub | Lounge corridor | 28.5–31.5, 30–40 | 30 | opening y=30 | **designed dead-end corridor** (no exit on FF): trap test |
| F-08 | Office | 31.5–45, 30–40 | 135 | y=30: x 36.0–37.0 | |
| F-09 | Library | 45–60, 30–40 | 150 | y=30: x 47.0–48.5 | |
| F-10 | Archive | 11–28.5, 13–27 | 245 | y=13: x 19.0–20.0 | **large single-door dead-end room** |
| ST-C | Central stair | 31.5–37.5, 13–19 | – | same openings as GF | |
| F-13 | WC | 31.5–37.5, 19–27 | 48 | x=31.5: y 22.0–23.0 | dead-end |
| F-12 | Office | 37.5–49, 13–27 | 161 | y=13: x 40.0–41.5; x=49: y 20.0–21.5 | through-room |
| H-3 | Pipe run (hazard point) | FF north ring leg at (30.0, 28.5), z = 4+3.0 | – | – | leak here blocks the most-used FF corridor |

### 3.4 Stairs (vertical links)
- **Model:** each stair is a switchback with two flights and a mid landing. The riser is 0.17 m, the tread 0.28 m, and the rise is floor-to-floor. The flight width is ST-W 1.5 m, ST-E 1.5 m and ST-C 2.0 m (all config values).
- **Derived values:** risers per flight, horizontal run, slope θ, and walking length $L_{stair}=\sum_{flights}\sqrt{run^2+rise^2}+L_{landing}$. These must be computed from the parameters, not typed in.
- **Default representation (P0):** the stair is a 1-D **vertical `Link`** between a `vertical` node on GF and one on FF.
  - kind = stair;
  - `slope_deg` = θ;
  - $\chi=0.70$;
  - speed by (N4s);
  - capacity $Q_{\max}=f_s w \chi$.
  - Its landing footprints are ordinary walkable cells on each floor grid, flagged `stair` so the gas coupling and later SFM can find them.
- **Future representation:** keep a `StairModel` interface so an explicit inclined 2-D SFM surface can replace the queue link without touching callers.
- **Enclosure:** ST-W and ST-E are enclosed. Their fire doors are self-closing: passable for agents, and gas-closed by default (leaf closed, permeability 0.05). ST-C is open: permeability 1.

### 3.5 Obstacles and clutter (toggleable)
- **Furniture (static, walkable = no):**
  - desk rows in G-16 (at least 3 rows, 1.2 m aisles);
  - tables in G-05;
  - consoles in G-11;
  - benches in G-15;
  - seating rows in F-07.
  - Furniture must never cut the connectivity of its room (tested).
- **Corridor clutter slots (`clutter` layer, default off):** 4–6 named rectangles, for example boxes in the south ring leg near G-04 or a cart in the spine. Each slot narrows its corridor to ≤1.5 m or fully blocks it.
- **Parked vehicles** (outdoor, `parking` layer, default off): a row of 2 × 4.5 m car rectangles in front of X-S and X-E.

The clutter and parking layers exist so that the four scenario rows of project slide 23 are exact presets (§3.8).

### 3.6 Windows and envelope
- Every exterior wall of a normal room gets windows (≈1.5 m wide, ≈3 m spacing, sill 0.9 m), generated by a rule, not listed by hand.
- Windows are **not walkable**. They have a `window_open: bool` (default closed) and a gas permeability: 0.02 when closed, 0.5 when open.
- They exist so that the future indoor–outdoor H₂S exchange can couple each indoor boundary cell to `C_out`.

### 3.7 Hazard sources (catalogue, no physics yet)
Each source has: id, building/floor or outdoor, XY (local or L3), z, species (default H₂S, MW 34.08), and default release parameters taken from spec §3 (`Q, H, u`) as placeholders.

| ID | Location | Purpose |
|---|---|---|
| H-1 | G-06 chemical store (54, 5), z 1.0 | indoor leak on the south façade near X-SE |
| H-2 | G-15 lab (55, 36), z 1.0 | indoor leak in the NE, threatening X-N / X-E approaches |
| H-3 | FF north ring (30, 28.5), z 7.0 | corridor leak that forces FF re-routing |
| H-4 | outdoor tank at L3 (140, 40) | outdoor dispersion toward the building with wind |

### 3.8 Scenario presets (YAML, applied as overlays)
| Preset | Effect |
|---|---|
| `baseline` | everything AVAILABLE; X-SE locked; clutter and parking off |
| `slide23_A` | no clutter; exits open; no parked cars |
| `slide23_B` | no clutter; exits open; cars parked |
| `slide23_C` | clutter on; emergency exits closed (X-NW, X-E UNAVAILABLE); no cars |
| `slide23_D` | clutter on; exits closed; cars parked |
| `block_main_exit` | X-S UNAVAILABLE |
| `block_ringS_mid` | south ring segment in front of LOBBY UNAVAILABLE |
| `leak_H1`, `leak_H2`, `leak_H3` | mark the source active; the links/cells within a config radius become PARTIAL (a stand-in until gas exists) |
| `stair_C_lost` | ST-C UNAVAILABLE |
| `two_stairs_lost` | ST-C and ST-E UNAVAILABLE (FF must still egress via ST-W) |

A scenario is data (a list of timed events). The same events can be fired at runtime (§6.5).

### 3.9 Outdoor site (L3)
- **Site:** 160 m (X) × 140 m (Y), origin at the site SW corner. The building has $\mathbf O_b=(50,50)$ and $\psi_b=0$, so it spans X 50–110, Y 50–90.
- **Ground:** open ground is walkable. Obstacles: the parking layer, one small outbuilding (for example 10×8 m at (20,110)), and the tank H-4.
- **Assembly points** (terminal goals for A*, spec [Y]), all 15 m diameter:
  - AP-N (80, 125);
  - AP-S (80, 15);
  - AP-W (15, 70);
  - AP-E (145, 70).
- **Outdoor network:** outdoor links from every exit to every AP, with lengths from an outdoor FMM (not Euclidean, so obstacles count).
- **Bottleneck zones:** mark the $R_{bn}=10$ m bottleneck zones around each exit (spec §1.5) as a layer.

---

## 4. File and module layout
```
dms/
  config/            # SimConfig + YAML schema (pydantic or dataclass validation)
  geometry/
    primitives.py    # Rect, WallSegment, Door, Opening, Window, Stair, Obstacle, HazardSource
    building.py      # Building, Floor, Space(room|corridor|stub|stair|void)
    site.py          # Site, AssemblyPoint, outdoor obstacles
    loader.py        # YAML -> Site; auto-generates walls from space boundaries, then cuts doors
    validate.py      # geometric validation (§7.1)
  grid/
    transforms.py    # cell<->local<->L3 (spec §1.1 + margin origin)
    raster.py        # vector -> layers at any h (deterministic rules, §5.1)
    layers.py        # FloorGrid, GridStack; static layers + dynamic overlay
  graph/
    builder.py       # Node/Link construction (§6.1)
    cost.py          # (N5) edge cost, reads live link state
    astar.py         # A* with admissible h(n)=||X_n - X_goal||/v_max
    redundancy.py    # single/double-failure analysis
  fields/
    eikonal.py       # T_eik per floor per exit set, multi-floor coupling (§6.3)
  hazards/
    topology.py      # gas permeability, vertical couplings, window couplings (no solver)
  scenario/
    events.py        # Event types, EventBus, dirty-building debounce (spec §1.6)
    presets.py
  viz/plot.py
  tools/build_env.py # CLI
configs/buildings/two_storey.yaml
configs/scenarios/*.yaml
tests/
```

---

## 5. Derived L0 grids

### 5.1 Rasterisation rules (must be deterministic and tested)
- **Grid extent:** each floor grid covers the footprint plus a **2 m margin** of `outside` cells on every side (needed for the Dirichlet $C=0$ boundary and for exit discharge). `origin_local = (-2, -2)`. The cell → local transform is $p=((col+\tfrac12)h+x_0,\ (row+\tfrac12)h+y_0)$.
- **Space fill:** a cell belongs to a space if its **centre** lies inside the space rectangle. Use half-open intervals `[x0, x1)` so shared boundaries never double-assign.
- **Walls:**
  - Walls are generated as segments on every space boundary that is not an opening.
  - A cell is wall if its centre is within $h/2$ of the segment. On an exact tie, the cell on the −x / −y side wins.
  - Result: walls are exactly one cell thick at every $h$, and a boundary on a cell edge never eats two cells.
- **Doors:** carved after walls. Cells whose centre projects onto the door span become `door` cells with `door_id`.
  - Every door must be ≥2 cells wide at $h_c$ and ≥5 cells wide at $h_s$ (validated).
  - Store the **geometric** width on the door; $Q_{\max}$ always uses the geometric width, never the cell count.
- **Other features:**
  - Furniture and clutter cells are marked `obstacle` (clutter in its own toggle layer).
  - VOID cells are `void`: not walkable, but gas-open.

### 5.2 Layers per `FloorGrid` (numpy, `[row, col]`)
| Layer | dtype | Content |
|---|---|---|
| `walkable_static` | bool | free floor, excluding walls, furniture, void and outside |
| `wall` | bool | |
| `space_id` | int32 | room/corridor/stair index, −1 elsewhere |
| `door_id` | int32 | −1 if not a door cell |
| `exit_id` | int32 | exterior door cells |
| `stair_id` | int32 | landing footprint cells |
| `obstacle_static` / `clutter` | bool | |
| `void` / `outside` / `window_id` | bool / bool / int32 | |
| `node_of` | int32 | **every** walkable cell → exactly one graph node index (spec §1.1 `node_of[b][row,col]`) |
| `gas_perm` | float32 | 0 for walls; 1 for open space; door, window and void values from the overlay |
| `bottleneck` | bool | within $R_{bn}$ of an exit |

The **dynamic overlay** holds door state, leaf state, clutter on/off and added obstacles. It produces `walkable_now`, `gas_perm_now` and `speed_F_now` (for N2) by vectorised masking. Nothing is re-rasterised.

Provide `GridStack.at(h)` for both $h_c$ and $h_s$ from the same geometry, plus `resample(layer, h_from, h_to)`: nearest for integer/bool layers, bilinear for float layers (spec: ratio 2.5, so bilinear).

### 5.3 SFM export
`building.wall_segments(floor, overlay)` returns an `(M, 4)` float array of wall segments in local metres. A door that is UNAVAILABLE or closed-for-agents becomes a segment. A static segment hash (spec §1.4, cell $r_c=2.0$ m) is built here and cached per floor.

---

## 6. Derived L2 graph, costs and fields

### 6.1 Nodes and links
Use the spec `Node` / `Link`. Add fields such as `building`, `door_id`, `chi`, `cells` (flat indices on each grid) and `kind_detail`. Node ids are readable, for example `GF:R:G-16`, `GF:C:ringS-03`, `GF:D:G-16-d2`, `ST-W:GF`, `X:X-S`, `AP:AP-N`.

**Node kinds** (keep the spec vocabulary and add `kind_detail`):
| Kind | kind_detail |
|---|---|
| platform | room / corridor / stub |
| transition | door / opening |
| vertical | stair landing |
| hazard_ctrl | fire door / service door |
| terminal | exit / assembly point |

**Corridor segmentation:**
- Split corridors at every junction and at every door centre projected onto the axis.
- Merge pieces shorter than 1.5 m; split pieces longer than 6 m.
- This gives fine-grained blocking and link-level density $\bar\rho_l$ and concentration $\bar C_l$ (needed by N5), averaged over `link.cells`.

**Rooms:** one node per room, at its centroid (all rooms are convex rectangles, so Euclidean door distances are exact). Add a config flag `subdivide_area_m2` (default off) so large rooms (F-07, F-10) can later be split into zones.

**Links:**
- room ↔ door ↔ corridor-segment;
- corridor ↔ corridor;
- stair landing ↔ stair landing (vertical);
- exit ↔ outdoor ↔ AP.

Each link carries `length_m`, `width_m` (geometric bottleneck width along it), `slope_deg`, `oneway`, `state`, $\chi$, and its capacity $Q_{\max,l}=f_s w_{eff}\chi$. Node capacity is `floor(area × ρ_cap)` with `ρ_cap` in config.

### 6.2 A* (N5)
- Implement (N5) exactly, reading live link state:
  - $c_l=\infty$ for UNAVAILABLE or shutter;
  - the PARTIAL multiplier $\omega_p$ and capacity $1/\omega_p$;
  - the $w_\rho, w_C, w_{vis}, w_T, w_H$ terms. With no agents or gas yet, $\rho$ and $C$ read zero from the (future) snapshot, but the code path must exist and be unit-tested with synthetic values.
- The heuristic is admissible. Goals are **assembly points only** (spec [Y]). An exit is an intermediate node.
- API: `route(src_node, goals=None, snapshot) -> Route(nodes, links, cost, eta_s)`.

### 6.3 Eikonal fields (N1–N2), multi-floor
- **Masking:** walls and void are masked out (`numpy.ma`) and are never traversable. Obstacles, clutter and future high-gas/smoke cells get $F=F_b$. Free cells get $F=U_{\max}$.
- **GF:** targets are the exit cells of the chosen exit set, giving $T_{GF}$ in seconds.
- **FF:** targets are the stair landing cells. skfmm cannot seed different initial times, so compute one field per stair and combine:
  $$T_{FF}(\mathbf x)=\min_s\big[T_{FF,s}(\mathbf x)+t_{stair,s}+T_{GF}(\text{discharge}_s)\big]$$
  - $t_{stair,s}=L_{stair}/v_{stair}$;
  - $T_{GF}(\text{discharge}_s)=0$ for ST-W, because it discharges directly through X-NW.
- **Exit sets** are named in YAML: `all`, `all_but_S`, `west`, `east`, one per exit, and so on.
- Fields are cached by `(building, floor, exit_set, overlay_hash)`. The dirty event recomputes only the affected building (spec §1.6).
- **Output:** a direction field $\mathbf e=-\nabla T/|\nabla T|$ via central differences on the masked field, with one-sided differences next to walls.

### 6.4 Gas topology export (for the future smoke/H₂S solver)
`hazards/topology.py` returns, per building:
- per-floor `gas_perm_now`;
- a list of **vertical couplings**: (GF cell, FF cell, coefficient) for every VOID cell column and every stair landing footprint (open ST-C = 1.0, enclosed ST-W/ST-E scaled by fire-door permeability);
- a list of **window couplings**: (indoor boundary cell, L3 outdoor cell, permeability).

No solver; just a `ConcentrationGrid` factory that allocates correctly shaped, correctly originated grids for `C_in` (per floor, $h_s$) and `C_out` (site, configurable $h$).

### 6.5 Events
- Event types:
  - `SetDoorState(id, AVAILABLE|PARTIAL|UNAVAILABLE)`;
  - `SetLeaf(id, open|closed)`;
  - `SetLinkState`;
  - `SetExitState`;
  - `ToggleLayer(clutter|parking, slot, on)`;
  - `AddObstacle(floor, rect)`;
  - `ActivateSource(id)`.
- Every event updates the overlay, marks the building dirty, and triggers one debounced recompute of $T^{eik}$ and the A* costs per $\Delta t_C$.
- Presets (§3.8) are lists of these events.

---

## 7. Validation, tests and reports (acceptance criteria)

### 7.1 Geometric validation (`validate.py`, run on every load)
- No two spaces overlap on a floor; the spaces tile the footprint except for designated walls.
- Every door lies on exactly one shared boundary between two spaces, or between a space and outside.
- Every exit lies on the exterior envelope.
- Stair footprints coincide on GF and FF.
- **Envelope watertight:** with all exterior doors and windows closed, a flood fill from `outside` never reaches an interior cell (at both $h$).
- Door widths meet the cell minimums of §5.1. Furniture never disconnects a room.

### 7.2 Connectivity and redundancy (`redundancy.py`, and the report `out/redundancy.md`)
- In `baseline`, every walkable cell and every room node reaches an assembly point.
- **Single-failure test:** for every link, door, corridor segment, stair and exit taken UNAVAILABLE one at a time, every room still reaches an AP. Exception: rooms whose *only* door is the failed element. Those are listed in a YAML `designed_dead_ends` list (G-08, F-10, F-13, FN-stub, single-door offices), and the test asserts that exactly the declared set is affected.
- **Double-failure tests:** `two_stairs_lost` (FF still egresses via ST-W), `block_main_exit` + `block_ringS_mid`, and every pair of exits.
- The report gives, for each failure: the affected rooms, the new route of 3 reference rooms (G-16, G-11 on GF; F-07 on FF), and the cost increase (%).
- **Edge-disjoint paths:** from every FF ring segment there are at least 2 edge-disjoint paths to an AP (`networkx.edge_disjoint_paths`).

### 7.3 Behavioural tests of routing and fields
- `block_main_exit`: LOBBY's route leaves via the ring and the cafeteria to another exit, and its cost rises.
- Blocking the lobby opening: ring occupants still reach X-S through the G-05 → LOBBY door.
- `leak_H3` (PARTIAL north ring on FF): F-07's A* route cost rises by the expected factor, and the path switches to its alternative door when that is cheaper.
- $T$ is finite on every walkable cell for `all`. It is larger in dead-ends than at their door. It increases monotonically along any computed route. $|\nabla T|\approx1/F$ within tolerance on free cells.
- Changing one door state changes $T$ only in the affected building, and does **not** re-rasterise (assert on a call counter).
- The FF field via the stair min-combination matches a brute-force check on a coarse grid (a multi-floor graph Dijkstra over cells) within 5%.

### 7.4 Engineering tests
- Transforms round-trip (cell ↔ local ↔ L3) and follow the row-up convention.
- Determinism: SHA-256 of all layers and the node-link JSON is identical across runs.
- Performance on a laptop:
  - full build (both $h$, graph, `all` fields) < 5 s;
  - one dirty recompute of one building < 0.5 s at $h_c$.
- A spawn sampler: `sample_agents(N, seed, density_cap)` places non-overlapping discs (r from config) in rooms proportional to area, and refuses with a clear message if $N$ exceeds the building's capacity. This building is sized for P0/P1. P2's 10,000 agents come from the multi-building plant later.

---

## 8. Deliverables, in this order
1. The YAML plus the loader and validator. **Plot both floors** with labels (space ids, door ids, exits, stairs, hazards) at $h_c$. Stop and compare against §3; fix any discrepancy before continuing.
2. Rasteriser plus `GridStack` at both resolutions, with layer plots and the watertight test.
3. Graph builder, the node-link JSON export and a graph overlay plot.
4. A* with (N5), the events/overlay layer and the scenario presets.
5. Eikonal fields with multi-floor coupling, heatmap plots of $T$ per scenario, and quiver plots of $\mathbf e$.
6. Redundancy analysis plus `out/redundancy.md`.
7. Gas topology export, the site/outdoor network and the AP routing.
8. CLI:
   - `python -m dms.tools.build_env configs/buildings/two_storey.yaml --scenario baseline --out out/` writes the PNGs, `layers.npz`, `graph.json`, `validation.md` and `redundancy.md`;
   - `... route --from GF:R:G-16 --scenario block_main_exit` prints the route and ETA.
9. A `README.md` section covering how to add a room, a door or a scenario **without touching code**.

## 9. Don'ts
- No coordinates, room ids or exit ids in Python code; everything comes from YAML.
- No re-rasterisation on state changes, no O(N²) loops, and no global mutable state.
- Do not implement SFM, the gas physics or the FSM now. Leave typed interfaces (`StairModel`, `ConcentrationGrid` factory, a `snapshot`-reading cost function) so those modules plug in without edits here.
- Do not silently "fix" the layout. If a coordinate in §3 is inconsistent, report it in `validation.md` and propose a correction.
