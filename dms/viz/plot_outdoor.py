"""Milestone 6 visual check: the outdoor site grid, obstacles, assembly
points, and an example AP's FMM distance field (confirms routes correctly
detour around the building rather than cutting through it). Run as:

    python -m dms.viz.plot_outdoor

Writes out/outdoor_network.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from ..fields.outdoor import build_outdoor_network, exit_discharge_site_xy
from ..geometry.loader import load_site

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"


def main() -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    net = build_outdoor_network(site, building)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ---- left: site layout ----
    ax = axes[0]
    grid = net.site_grid
    h = grid.transform.h
    x0, y0 = grid.transform.origin_local
    rows, cols = grid.transform.shape
    extent = (x0, x0 + cols * h, y0 + rows * h, y0)

    base = np.zeros((rows, cols, 4))
    base[grid.walkable] = to_rgba("#c5d6a1")
    base[grid.building_mask] = to_rgba("#a8c5d4")
    base[~grid.walkable & ~grid.building_mask] = to_rgba("#8a8f98")
    ax.imshow(base, extent=extent, origin="upper", interpolation="nearest")

    for ap_id, ap in site.assembly_points.items():
        ax.scatter([ap.x], [ap.y], marker="P", s=160, c="#22c55e", edgecolors="#14532d", linewidths=0.8, zorder=5)
        ax.text(ap.x, ap.y - 4, ap_id, ha="center", fontsize=8, fontweight="bold", color="#14532d")

    for door in gf.exit_doors:
        ex, ey = exit_discharge_site_xy(building, "GF", door)
        ax.scatter([ex], [ey], marker="s", s=60, c="#c45c26", edgecolors="#333", zorder=5)
        for ap_id, ap in site.assembly_points.items():
            ax.plot([ex, ap.x], [ey, ap.y], color="#475569", lw=0.4, alpha=0.4, zorder=2)
        ax.annotate(door.id, (ex, ey), textcoords="offset points", xytext=(4, 4), fontsize=7, fontweight="bold")

    ax.set_title(f"{building.id}: outdoor site grid (h={h} m)", fontsize=12, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#a8c5d4", markersize=12, label="Building (not walkable)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#8a8f98", markersize=12, label="Obstacle"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#c5d6a1", markersize=12, label="Open ground"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="#c45c26", markeredgecolor="#333", markersize=9, label="Exit"),
        Line2D([0], [0], marker="P", color="none", markerfacecolor="#22c55e", markeredgecolor="#14532d", markersize=11, label="Assembly point"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    # ---- right: AP-S's FMM distance field (chosen because it's across the
    # building from several exits, making the detour visible) ----
    ax2 = axes[1]
    field = net.ap_fields["AP-S"]
    T_display = np.ma.MaskedArray(field.data, mask=field.mask | grid.building_mask)
    im = ax2.imshow(T_display, cmap="viridis", extent=extent, origin="upper", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax2, fraction=0.04, pad=0.02)
    cbar.set_label("distance to AP-S (m)", fontsize=9)

    blocked_rgba = np.zeros((rows, cols, 4))
    blocked_rgba[grid.building_mask] = to_rgba("#1f1f1f")
    blocked_rgba[~grid.walkable & ~grid.building_mask] = to_rgba("#52525b")  # other obstacles (e.g. the outbuilding)
    ax2.imshow(blocked_rgba, extent=extent, origin="upper", interpolation="nearest")

    ap = site.assembly_points["AP-S"]
    ax2.scatter([ap.x], [ap.y], marker="P", s=160, c="white", edgecolors="#14532d", linewidths=1.2, zorder=5)
    for door in gf.exit_doors:
        ex, ey = exit_discharge_site_xy(building, "GF", door)
        d = net.exit_ap_distance_m[(door.id, "AP-S")]
        ax2.scatter([ex], [ey], marker="s", s=50, c="#c45c26", edgecolors="white", zorder=5)
        ax2.annotate(f"{door.id}\n{d:.0f} m", (ex, ey), textcoords="offset points", xytext=(5, 5),
                     fontsize=7, fontweight="bold", color="white")

    ax2.set_title("FMM distance field seeded at AP-S (building blocks straight lines)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)")
    ax2.set_aspect("equal")

    fig.suptitle(f"{building.id} — outdoor L3 network (spec §3.9)", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.88, wspace=0.2)

    out = OUT_DIR / "outdoor_network.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
