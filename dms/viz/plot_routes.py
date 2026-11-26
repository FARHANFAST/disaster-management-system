"""Milestone 3 visual verification: render A* evacuation routes (baseline
and scenario detours) and the full L2 graph network over the rasterised
GF/FF layouts. Run as:

    python -m dms.viz.plot_routes

Writes out/route_baseline.png, out/route_blocked_scenarios.png and
out/l2_graph_network.png.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from ..geometry.building import Building
from ..geometry.loader import load_site
from ..geometry.site import Site
from ..graph.astar import Route
from ..graph.astar import route as astar_route
from ..graph.builder import NavGraph, Node, build_navigation_graph
from ..grid.layers import H_COARSE
from ..grid.raster import rasterize_floor
from ..scenario.presets import apply_scenario
from .plot import plot_floor_layers

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"


def _local_xy(building: Building, node: Node) -> tuple[float, float]:
    return building.from_site_xy(*node.XY)


def _pad_limits(ax, pad: float = 2.5) -> None:
    """Push both ends of each axis outward by `pad`, preserving whatever
    direction (possibly inverted) the axis already runs — so an exit-to-AP
    label drawn just past the building edge has room and isn't clipped by
    the view imshow() pinned to the image extent."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - math.copysign(pad, x0 - x1), x1 + math.copysign(pad, x1 - x0))
    ax.set_ylim(y0 - math.copysign(pad, y0 - y1), y1 + math.copysign(pad, y1 - y0))


def _outward_normal(xy: tuple[float, float], footprint, mag: float = 1.6) -> tuple[float, float]:
    """A short vector pointing out of the building, from whichever footprint
    edge `xy` is closest to — used to draw an exit's onward arrow toward its
    assembly point without leaving the building's own axes."""
    x, y = xy
    candidates = {
        abs(y - footprint.y0): (0.0, -mag),
        abs(y - footprint.y1): (0.0, mag),
        abs(x - footprint.x0): (-mag, 0.0),
        abs(x - footprint.x1): (mag, 0.0),
    }
    return candidates[min(candidates)]


def _arrow(ax, p1: tuple[float, float], p2: tuple[float, float], *, color: str, lw: float = 2.6, zorder: int = 8) -> None:
    ax.annotate(
        "", xy=p2, xytext=p1, zorder=zorder,
        arrowprops={"arrowstyle": "-|>", "color": color, "lw": lw, "shrinkA": 0, "shrinkB": 0, "mutation_scale": 14},
    )


def draw_route(
    ax_by_floor: dict[str, object], footprint_by_floor: dict[str, object],
    building: Building, nav: NavGraph, r: Route | None, *, color: str, lw: float = 2.6, annotate: bool = True,
) -> None:
    """Draw whichever portion of route `r` belongs to each floor present in
    `ax_by_floor`, with markers where it crosses a stair (vertical link) or
    leaves the building via an exit toward an assembly point. `annotate`
    controls the text labels, so a faint baseline-comparison underlay can
    skip them and just show the line."""
    if r is None or not r.links:
        return
    zorder = 7 if lw < 2.0 else 8
    for u, v in r.links:
        nu, nv = nav.node(u), nav.node(v)

        if nu.floor == nv.floor and nu.floor in ax_by_floor:
            ax = ax_by_floor[nu.floor]
            _arrow(ax, _local_xy(building, nu), _local_xy(building, nv), color=color, lw=lw, zorder=zorder)
            continue

        if nu.kind == "vertical" and nv.kind == "vertical":
            # Stair landings share the same local (x,y) on both floors (spec
            # §7.1 stair-alignment invariant), so one position serves both panels.
            xy = _local_xy(building, nu)
            x, y = xy
            if nu.floor in ax_by_floor:
                ax = ax_by_floor[nu.floor]
                _arrow(ax, (x, y), (x, y - 1.8), color=color, lw=lw, zorder=zorder)
                if annotate:
                    ax.text(x + 0.6, y - 1.0, f"↓ {nu.ref_id}\nto {nv.floor}", color=color,
                            fontsize=7.5, fontweight="bold", va="center", zorder=9)
            if nv.floor in ax_by_floor:
                ax = ax_by_floor[nv.floor]
                _arrow(ax, (x, y + 1.8), (x, y), color=color, lw=lw, zorder=zorder)
                if annotate:
                    ax.text(x + 0.6, y + 1.0, f"↑ {nv.ref_id}\nfrom {nu.floor}", color=color,
                            fontsize=7.5, fontweight="bold", va="center", zorder=9)
            continue

        if nv.kind == "terminal" and nv.kind_detail == "assembly_point" and nu.floor in ax_by_floor:
            ax = ax_by_floor[nu.floor]
            xy = _local_xy(building, nu)
            dx, dy = _outward_normal(xy, footprint_by_floor[nu.floor])
            x, y = xy
            _arrow(ax, (x, y), (x + dx, y + dy), color=color, lw=lw, zorder=zorder)
            if annotate:
                dist = math.hypot(nv.XY[0] - nu.XY[0], nv.XY[1] - nu.XY[1])
                ax.text(x + dx * 1.2, y + dy * 1.2, f"{nv.ref_id}\n{dist:.0f} m", color=color,
                        fontsize=8, fontweight="bold", ha="center", va="center", zorder=9)


def _route_legend(fig, entries: list[tuple[str, str]]) -> None:
    handles = [Line2D([0], [0], color=color, lw=2.8, label=label) for label, color in entries]
    ncol = min(len(entries), 2)
    fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=10,
               frameon=True, bbox_to_anchor=(0.5, -0.06))


def _two_panel(building: Building, gf, ff, gf_grid, ff_grid, *, suptitle: str):
    # figsize matched to the building's own (margin-inclusive) aspect ratio
    # (~1.45:1 landscape per floor) — a taller box just leaves blank bands
    # above/below once set_aspect("equal") centres the data inside it.
    fig, (ax_gf, ax_ff) = plt.subplots(1, 2, figsize=(20, 6.6))
    plot_floor_layers(building, gf, gf_grid, ax=ax_gf, title=f"{building.id} GF")
    plot_floor_layers(building, ff, ff_grid, ax=ax_ff, title=f"{building.id} FF")
    for ax in (ax_gf, ax_ff):
        _pad_limits(ax)
    fig.subplots_adjust(top=0.85, bottom=0.20, wspace=0.18)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    return fig, {"GF": ax_gf, "FF": ax_ff}


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------- network diagram

_KIND_COLOR = {
    "platform": "#1d4ed8", "transition": "#c4a35a", "vertical": "#0e7490", "hazard_ctrl": "#b91c1c",
}


def _draw_network_on_floor(ax, building: Building, nav: NavGraph, floor_id: str) -> None:
    for u, v in nav.g.edges():
        nu, nv = nav.node(u), nav.node(v)
        if nu.floor == floor_id and nv.floor == floor_id:
            p1, p2 = _local_xy(building, nu), _local_xy(building, nv)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="#1d4ed8", lw=0.7, alpha=0.5, zorder=6)
    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.floor != floor_id:
            continue
        x, y = _local_xy(building, node)
        ax.scatter([x], [y], s=16, color=_KIND_COLOR.get(node.kind, "#334155"), zorder=7,
                   edgecolors="white", linewidths=0.4)


def _draw_site_network(ax, site: Site, building: Building, nav: NavGraph) -> None:
    gf_id = building.floor_ids()[0]
    fp = building.get_floor(gf_id).footprint
    corners_local = [(fp.x0, fp.y0), (fp.x1, fp.y0), (fp.x1, fp.y1), (fp.x0, fp.y1), (fp.x0, fp.y0)]
    xs, ys = zip(*(building.to_site_xy(x, y) for x, y in corners_local))
    ax.plot(xs, ys, color="#334155", lw=2, zorder=3)
    ax.fill(xs, ys, color="#cbd5e1", alpha=0.4, zorder=1)

    for nid in nav.node_ids:
        node = nav.node(nid)
        if node.kind == "terminal" and node.kind_detail == "exit":
            x, y = node.XY
            ax.scatter([x], [y], marker="s", s=55, c="#c45c26", zorder=5, edgecolors="#333", linewidths=0.5)
            for ap in site.assembly_points.values():
                ax.plot([x, ap.x], [y, ap.y], color="#94a3b8", lw=0.5, alpha=0.5, zorder=2)

    for ap_id, ap in site.assembly_points.items():
        ax.scatter([ap.x], [ap.y], marker="P", s=180, c="#22c55e", edgecolors="#14532d", linewidths=0.8, zorder=6)
        ax.text(ap.x, ap.y - 5, ap_id, ha="center", fontsize=9, fontweight="bold", color="#14532d", zorder=6)

    ax.set_title("Site network: exits ↔ assembly points", fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#c45c26", markersize=9, label="Exit"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor="#22c55e", markeredgecolor="#14532d",
               markersize=11, label="Assembly point"),
        Line2D([0], [0], color="#94a3b8", lw=1, label="Exit↔AP link"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)


def build_network_figure(site: Site, building: Building, nav: NavGraph, gf, ff, gf_grid, ff_grid):
    fig, axes = plt.subplots(1, 3, figsize=(26, 7.0), gridspec_kw={"width_ratios": [1.0, 1.0, 0.8]})
    plot_floor_layers(building, gf, gf_grid, ax=axes[0], title=f"{building.id} GF — L2 graph")
    plot_floor_layers(building, ff, ff_grid, ax=axes[1], title=f"{building.id} FF — L2 graph")
    _draw_network_on_floor(axes[0], building, nav, "GF")
    _draw_network_on_floor(axes[1], building, nav, "FF")
    _draw_site_network(axes[2], site, building, nav)
    for ax in (axes[0], axes[1]):
        _pad_limits(ax)

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor="white", markersize=8, label=k)
        for k, c in _KIND_COLOR.items()
    ] + [Line2D([0], [0], color="#1d4ed8", lw=1, alpha=0.6, label="link")]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.subplots_adjust(top=0.85, bottom=0.14, wspace=0.22)

    n_nodes = nav.g.number_of_nodes()
    n_edges = nav.g.number_of_edges()
    fig.suptitle(
        f"{building.id} — L2 navigation graph ({n_nodes} nodes, {n_edges} links)",
        fontsize=14, fontweight="bold", y=0.98,
    )
    return fig


def main() -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    ff = building.get_floor("FF")
    gf_grid = rasterize_floor(building, gf, H_COARSE)
    ff_grid = rasterize_floor(building, ff, H_COARSE)
    footprint_by_floor = {"GF": gf.footprint, "FF": ff.footprint}

    # ---- 1-2: baseline routes ----
    # LOBBY and ST-C:FF are the rooms that actually use X-S and ST-C on their
    # shortest path (unlike G-16/F-07, which route via X-W/ST-W regardless),
    # so blocking those elements below produces a real, visible detour.
    nav = build_navigation_graph(site, "B1", compute_node_of=False)
    route_lobby = astar_route(nav, "GF:R:LOBBY")
    route_stc = astar_route(nav, "ST-C:FF")
    print(f"baseline LOBBY -> {route_lobby.nodes[-1]}: cost={route_lobby.cost:.1f}s, {len(route_lobby.nodes)} nodes")
    print(f"baseline ST-C:FF -> {route_stc.nodes[-1]}: cost={route_stc.cost:.1f}s, {len(route_stc.nodes)} nodes")

    fig1, axes1 = _two_panel(building, gf, ff, gf_grid, ff_grid, suptitle="Milestone 3: baseline A* evacuation routes")
    draw_route(axes1, footprint_by_floor, building, nav, route_lobby, color="#ea580c")
    draw_route(axes1, footprint_by_floor, building, nav, route_stc, color="#7c3aed")
    _route_legend(fig1, [
        (f"LOBBY → {route_lobby.nodes[-1].split(':')[-1]} (baseline, {route_lobby.cost:.0f}s)", "#ea580c"),
        (f"ST-C:FF → {route_stc.nodes[-1].split(':')[-1]} (baseline, {route_stc.cost:.0f}s)", "#7c3aed"),
    ])
    p1 = _save(fig1, OUT_DIR / "route_baseline.png")
    print(f"wrote {p1}")

    # ---- 3-4: scenario detours ----
    nav_block = build_navigation_graph(site, "B1", compute_node_of=False)
    apply_scenario(nav_block, site, "block_main_exit")
    detour_lobby = astar_route(nav_block, "GF:R:LOBBY")
    lobby_changed = detour_lobby.nodes != route_lobby.nodes
    print(f"block_main_exit LOBBY -> {detour_lobby.nodes[-1]}: cost={detour_lobby.cost:.1f}s "
          f"(baseline was {route_lobby.cost:.1f}s) {'[REROUTED]' if lobby_changed else '[UNCHANGED]'}")

    nav_stair = build_navigation_graph(site, "B1", compute_node_of=False)
    apply_scenario(nav_stair, site, "stair_C_lost")
    detour_stc = astar_route(nav_stair, "ST-C:FF")
    stc_changed = detour_stc.nodes != route_stc.nodes
    print(f"stair_C_lost ST-C:FF -> {detour_stc.nodes[-1]}: cost={detour_stc.cost:.1f}s "
          f"(baseline was {route_stc.cost:.1f}s) {'[REROUTED]' if stc_changed else '[UNCHANGED]'}")

    fig2, axes2 = _two_panel(building, gf, ff, gf_grid, ff_grid, suptitle="Milestone 3: scenario detours")
    # Faint baseline underlay so the detour is visibly "a different line", not just a new one.
    draw_route(axes2, footprint_by_floor, building, nav, route_lobby, color="#fdba74", lw=1.3, annotate=False)
    draw_route(axes2, footprint_by_floor, building, nav, route_stc, color="#c4b5fd", lw=1.3, annotate=False)
    draw_route(axes2, footprint_by_floor, building, nav_block, detour_lobby, color="#dc2626")
    draw_route(axes2, footprint_by_floor, building, nav_stair, detour_stc, color="#0891b2")

    lobby_tag = "REROUTED" if lobby_changed else "unchanged"
    stc_tag = "REROUTED" if stc_changed else "unchanged"
    _route_legend(fig2, [
        (f"LOBBY baseline ({route_lobby.cost:.0f}s)", "#fdba74"),
        (f"LOBBY + block_main_exit: {lobby_tag} ({detour_lobby.cost:.0f}s)", "#dc2626"),
        (f"ST-C:FF baseline ({route_stc.cost:.0f}s)", "#c4b5fd"),
        (f"ST-C:FF + stair_C_lost: {stc_tag} ({detour_stc.cost:.0f}s)", "#0891b2"),
    ])
    p2 = _save(fig2, OUT_DIR / "route_blocked_scenarios.png")
    print(f"wrote {p2}")

    # ---- 5: full L2 graph network ----
    nav_full = build_navigation_graph(site, "B1", compute_node_of=False)
    fig3 = build_network_figure(site, building, nav_full, gf, ff, gf_grid, ff_grid)
    p3 = _save(fig3, OUT_DIR / "l2_graph_network.png")
    print(f"wrote {p3}")


if __name__ == "__main__":
    main()
