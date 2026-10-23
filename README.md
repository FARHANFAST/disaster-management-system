# Disaster Management System

## Environment build artifacts (`dms/tools/build_env.py`)

```
python -m dms.tools.build_env --config configs/buildings/two_storey.yaml --out-dir out --visualize
```

Compiles one building's geometry, both floors' raster layers (`H_COARSE`/`H_FINE`),
the L2 navigation graph, and the L3 outdoor exit↔assembly-point network into two
artifacts, so downstream modules (agent physics, hazard dispersion) never have to
re-run the geometry/raster/graph/outdoor pipeline themselves.

### `out/layers.npz`

A flat `numpy.savez_compressed` archive. Keys are `/`-joined paths; load with
`numpy.load("out/layers.npz")` and index like a dict:

- `{building_id}/{floor_id}/h{h}/{layer}` — one array per `FloorGrid` layer
  (`wall`, `void`, `door_id`, `gas_perm_static`, `node_of`, …) at each of
  `h0.5`/`h0.2`, plus `{...}/origin_local` for the cell→metre transform.
- `outdoor/walkable`, `outdoor/building_mask`, `outdoor/origin_local`, `outdoor/h`
  — the L3 site grid.
- `outdoor/nearest_ap_field_data` / `_mask` — the combined (all-APs-seeded) FMM
  distance field an evacuating agent follows outdoors; `-grad` of this, not of
  any single per-AP field, is the field to steer an exiting agent by.
- `outdoor/ap_field/{ap_id}/data` / `_mask` — one field per individual assembly
  point, used for the exit↔AP distance matrix, not for live navigation.

A consumer only ever needs the `node_of` and `gas_perm_static`/`wall`/`void`
layers plus the outdoor fields — it should never re-rasterize from the YAML.

### `out/graph.json`

Node-link JSON, one entry per building in `buildings`:

```json
{
  "site": "...",
  "buildings": [
    {
      "building_id": "B1",
      "nodes": [{"id": "...", "floor": "...", "kind": "...", "x": 0.0, "y": 0.0, ...}],
      "links": [{"u": "...", "v": "...", "length_m": 0.0, "chi": 1.0, "Q_max": 0.0, ...}],
      "outdoor_exit_ap_distance_m": {"X-S|AP-N": 117.7, ...}
    }
  ]
}
```

`nodes[].id` matches `layers.npz`'s `node_of` indices 1:1 in the order the file
lists them, so a consumer can rebuild the index→id lookup with
`nodes[i]["id"]` for `node_of == i`. `links[].state`/`rho`/`C` are the live
fields (N5) expects a cost/routing module to mutate per tick — this export is
only the static snapshot at build time.

**Verified by:** `python -m dms.tools.build_env --config configs/buildings/two_storey.yaml --out-dir out --visualize`
followed by reloading both files and checking key/node counts (70 npz arrays,
202 nodes / 244 links for the two-storey building) — see `tests/` for the
per-module unit tests this CLI composes.

## Gas dispersion + gas-aware evacuation (`dms/gas/`)

```
python -m dms.gas.simulate                      # -> out/gas_sim.html (replay) + a summary table
python -m dms.gas.simulate --scenario configs/scenarios/h2s_tank_leak.yaml --no-blind
```

Implements "Gas Dispersion Prototype — Model Design" for H₂S, coupled to the
graph the graph-RL agent decides on (`GraphEvacEnv`):

- **Outdoor** (`dispersion.py`): Gaussian puffs with ground reflection,
  Pasquill–Gifford / Briggs σ (urban by default), advected by a time-varying
  wind (`meteo.py`), with wet deposition in rain. The steady plume is kept as the fast
  mode and verification check. Under constant wind the puff sum agrees with it
  to within 8% from 20 m out (`tests/test_gas.py`).
- **Indoor** (`indoor.py`): one well-mixed box per room / corridor segment,
  `dC/dt = ACH/3600 · (C_out − C_in)`. HVAC on/off, sealed rooms and open exit
  doors change ACH, and agents and the alarm response write these through the map state.
- **Graph coupling** (`zones.py`): every `GraphEvacEnv` node gets a
  concentration. Zones are boxes, doors read their worst neighbour, and an exit
  reads the worst outdoor air on the walk from that door to its assembly point.
- **Alarm** (`alarm.py`): EMERGENCY latches when a detector reads ≥ 30 ppm for 5 s.
  **Forecast** (`model.py`): 5-minute persistence projection of every node and
  the field, plus time-to-hazard, refreshed every 30 s and on wind shifts.
- **Routing**: A* is the default planner (`dms/rl/graph_astar.py`). Its edges
  are the same grid-walker legs `GraphEvacEnv.step` uses, so with no gas it returns
  exactly `graph_walker_optimum`. `dms/gas/routing.py` adds the N5 gas cost and
  forbids entering any node at or above `avoid_ppm` (10 ppm) at the agent's ETA,
  according to the forecast. If no gas-free route exists it falls back to IDLH-limited, then least-exposure.
- **Scenario** (`configs/scenarios/*.yaml`): one file per run with `map`, `sim`,
  `gas` and `agents` sections. Runs are deterministic for a seed.

`simulate.py` runs the scenario twice: once with gas-aware A* and once with a
gas-blind fastest-route baseline, with the same people and pre-movement times. It writes a
self-contained replay page. Open it at a given moment with `out/gas_sim.html#t=60&run=both&forecast=1`.
