"""CLI builder (PROMPT_01 §8, item 8): compiles one YAML config end-to-end
into the artifacts downstream modules (agent physics, hazard dispersion)
consume — raster layers for every floor at both resolutions, the L2
navigation graph, and the L3 outdoor exit<->assembly-point network — without
any of them having to re-run the geometry/raster/graph/outdoor pipeline
themselves.

    python -m dms.tools.build_env --config configs/buildings/two_storey.yaml --out-dir out --visualize

Writes:
    <out-dir>/layers.npz   -- every FloorGrid layer (both buildings' floors,
                              both H_COARSE/H_FINE resolutions) plus the
                              outdoor site grid and its FMM distance fields.
    <out-dir>/graph.json   -- the L2 nav graph as node-link JSON, plus the
                              outdoor exit<->AP distance matrix.
    <out-dir>/*.png        -- only with --visualize: per-floor layer plots
                              and a graph-overview plot, one set per building.

See the "Environment build artifacts" section in README.md for how a
downstream module is expected to load and use these two files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ..fields.outdoor import build_outdoor_network
from ..geometry.loader import load_site
from ..graph.builder import build_navigation_graph
from ..grid.layers import H_COARSE
from ..grid.raster import build_grid_stack

_FLOOR_LAYER_NAMES = (
    "walkable_static", "wall", "space_id", "door_id", "exit_id", "stair_id",
    "obstacle_static", "clutter_mask", "void", "outside", "window_id", "gas_perm_static",
)


def _floor_layers_to_npz(building_id: str, floor_id: str, h: float, grid, node_of: np.ndarray | None) -> dict[str, np.ndarray]:
    prefix = f"{building_id}/{floor_id}/h{h}/"
    out = {prefix + name: getattr(grid, name) for name in _FLOOR_LAYER_NAMES}
    out[prefix + "origin_local"] = np.array(grid.transform.origin_local, dtype=np.float64)
    if node_of is not None:
        out[prefix + "node_of"] = node_of
    return out


def _outdoor_to_npz(net) -> dict[str, np.ndarray]:
    grid = net.site_grid
    out = {
        "outdoor/walkable": grid.walkable,
        "outdoor/building_mask": grid.building_mask,
        "outdoor/origin_local": np.array(grid.transform.origin_local, dtype=np.float64),
        "outdoor/h": np.array([grid.transform.h], dtype=np.float64),
        "outdoor/nearest_ap_field_data": net.nearest_ap_field.data,
        "outdoor/nearest_ap_field_mask": net.nearest_ap_field.mask,
    }
    for ap_id, field in net.ap_fields.items():
        out[f"outdoor/ap_field/{ap_id}/data"] = field.data
        out[f"outdoor/ap_field/{ap_id}/mask"] = field.mask
    return out


def _node_to_dict(node) -> dict:
    d = {
        "id": node.id, "floor": node.floor, "kind": node.kind, "kind_detail": node.kind_detail,
        "area_m2": node.area_m2, "width_m": node.width_m, "capacity": node.capacity,
        "x": node.XY[0], "y": node.XY[1],
        "building": node.building, "space_id": node.space_id, "ref_id": node.ref_id,
    }
    return d


def _link_to_dict(link) -> dict:
    return {
        "u": link.u, "v": link.v, "length_m": link.length_m, "width_m": link.width_m,
        "slope_deg": link.slope_deg, "oneway": link.oneway, "state": link.state,
        "shutter_closed": link.shutter_closed, "rho": link.rho, "C": link.C,
        "temp": link.temp, "d_haz": link.d_haz, "chi": link.chi, "Q_max": link.Q_max,
    }


def build_env(config: Path, out_dir: Path, *, visualize: bool = False) -> None:
    site = load_site(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    npz_arrays: dict[str, np.ndarray] = {}
    graph_payload = {"site": site.name, "buildings": []}

    for building_id, building in site.buildings.items():
        net = build_outdoor_network(site, building)
        npz_arrays.update(_outdoor_to_npz(net))

        nav = build_navigation_graph(site, building_id, outdoor_distances=net.exit_ap_distance_m)

        for floor_id, floor in building.floors.items():
            stack = build_grid_stack(building, floor)
            for h, grid in stack.grids.items():
                node_of = nav.node_of.get((floor_id, h))
                npz_arrays.update(_floor_layers_to_npz(building_id, floor_id, h, grid, node_of))

        graph_payload["buildings"].append({
            "building_id": building_id,
            "nodes": [_node_to_dict(nav.node(nid)) for nid in nav.node_ids],
            "links": [_link_to_dict(nav.link(u, v)) for u, v in nav.g.edges()],
            "outdoor_exit_ap_distance_m": {f"{eid}|{apid}": d for (eid, apid), d in net.exit_ap_distance_m.items()},
        })

        if visualize:
            _write_visuals(building_id, building, net, nav, out_dir)

    npz_path = out_dir / "layers.npz"
    np.savez_compressed(npz_path, **npz_arrays)

    graph_path = out_dir / "graph.json"
    graph_path.write_text(json.dumps(graph_payload, indent=2))

    print(f"wrote {npz_path} ({len(npz_arrays)} arrays)")
    print(f"wrote {graph_path} ({sum(len(b['nodes']) for b in graph_payload['buildings'])} nodes, "
          f"{sum(len(b['links']) for b in graph_payload['buildings'])} links)")


def _write_visuals(building_id: str, building, net, nav, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ..viz.plot import save_floor_plot

    for floor_id, floor in building.floors.items():
        grid = build_grid_stack(building, floor, h_values=(H_COARSE,)).at(H_COARSE)
        out_path = out_dir / f"{building_id}_{floor_id}_layers.png"
        save_floor_plot(building, floor, grid, out_path, title=f"{building_id} {floor_id} (h={H_COARSE} m)")
        print(f"wrote {out_path}")

    fig, ax = plt.subplots(figsize=(10, 9))
    kind_colors = {
        "platform": "#64748b", "transition": "#c45c26", "vertical": "#7c3aed",
        "hazard_ctrl": "#dc2626", "terminal": "#16a34a",
    }
    for u, v in nav.g.edges():
        (x1, y1), (x2, y2) = nav.node(u).XY, nav.node(v).XY
        ax.plot([x1, x2], [y1, y2], color="#94a3b8", lw=0.6, zorder=1)
    for nid in nav.node_ids:
        node = nav.node(nid)
        ax.scatter([node.XY[0]], [node.XY[1]], s=18, color=kind_colors.get(node.kind, "#000"), zorder=2)
    ax.set_title(f"{building_id}: L2 navigation graph ({len(nav.node_ids)} nodes)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (site m)")
    ax.set_ylabel("Y (site m)")
    ax.set_aspect("equal")
    out_path = out_dir / f"{building_id}_graph_overview.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m dms.tools.build_env", description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True, help="path to a building YAML config")
    parser.add_argument("--out-dir", type=Path, default=Path("out"), help="output directory (default: out)")
    parser.add_argument("--visualize", action="store_true", help="also write per-floor and graph-overview PNGs")
    args = parser.parse_args()
    build_env(args.config, args.out_dir, visualize=args.visualize)


if __name__ == "__main__":
    main()
