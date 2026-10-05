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
