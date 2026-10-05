"""Milestone 4 visual verification: Eikonal travel-time heatmaps with
direction-field quivers, baseline and under a blocking scenario. Run as:

    python -m dms.viz.plot_fields

Writes out/eikonal_gf.png and out/eikonal_ff.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba

from ..fields.eikonal import (
    EikonalField,
    FieldCache,
    direction_field,
    stairs_blocked_by_scenario,
)
from ..geometry.building import Building, Floor
from ..geometry.loader import load_site
from ..grid.layers import H_COARSE, DynamicOverlay
from ..grid.raster import rasterize_floor

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"


def plot_eikonal_field(building: Building, floor: Floor, grid, field: EikonalField, *, ax=None,
                        title: str | None = None, quiver_stride: int = 5):
    fig, ax = (plt.subplots(figsize=(10, 7)) if ax is None else (ax.figure, ax))

    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    rows, cols = grid.shape
    extent = (x0, x0 + cols * h, y0 + rows * h, y0)

    # Furniture cells crawl at F_b=0.001 m/s, so their T can be 10-100x the
    # real walkable range (spec N2 — "difficult but not impossible", not
    # meant to set the colour scale). Clip the heatmap to the walkable
    # cells' own range and show obstacles as a flat colour instead, so the
    # actual spatial T gradient across the floor stays visible.
    walkable_domain = field.domain & ~grid.obstacle_static
    vmax = float(np.percentile(field.T.data[walkable_domain], 98)) if walkable_domain.any() else 1.0
    T_display = np.ma.MaskedArray(field.T.data, mask=~walkable_domain)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="#d4d4d8")
    im = ax.imshow(T_display, cmap=cmap, vmin=0.0, vmax=vmax, extent=extent, origin="upper",
                    interpolation="nearest", zorder=1)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, extend="max")
    cbar.set_label("T (s)  [clipped at P98 of walkable cells; furniture is slower, shown separately]", fontsize=8)

    obstacle_rgba = np.zeros((rows, cols, 4))
    obstacle_rgba[grid.obstacle_static] = to_rgba("#8a8f98")
    ax.imshow(obstacle_rgba, extent=extent, origin="upper", interpolation="nearest", zorder=2)

    wall_rgba = np.zeros((rows, cols, 4))
    wall_rgba[grid.wall] = to_rgba("#1f1f1f")
    ax.imshow(wall_rgba, extent=extent, origin="upper", interpolation="nearest", zorder=2)

    exit_rgba = np.zeros((rows, cols, 4))
    exit_rgba[grid.exit_id != -1] = to_rgba("#ffffff")
    ax.imshow(exit_rgba, extent=extent, origin="upper", interpolation="nearest", zorder=3, alpha=0.9)

    ex, ey = direction_field(field.T, field.domain, h, degraded=field.speed < 0.01)
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    X, Y = np.meshgrid(xc, yc)
    sl = (slice(None, None, quiver_stride), slice(None, None, quiver_stride))
    qmask = field.domain[sl] & ~np.isnan(ex[sl])
    ax.quiver(X[sl][qmask], Y[sl][qmask], ex[sl][qmask], ey[sl][qmask],
              color="white", scale=22, width=0.0028, zorder=5, alpha=0.85)

    for stair_id, sidx in zip(grid.stair_ids, range(len(grid.stair_ids))):
        cells = np.nonzero(grid.stair_id == sidx)
        if cells[0].size:
            cy = yc[cells[0]].mean()
            cx = xc[cells[1]].mean()
            ax.text(cx, cy, stair_id, color="white", fontsize=8, fontweight="bold",
                    ha="center", va="center", zorder=6)

    ax.set_title(title or f"{building.id} {floor.id} — T^eik ({field.exit_set})", fontsize=11, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    return fig, ax


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    ff = building.get_floor("FF")
    gf_grid = rasterize_floor(building, gf, H_COARSE)
    ff_grid = rasterize_floor(building, ff, H_COARSE)
    cache = FieldCache()

    # ---- GF: baseline vs block_main_exit ----
    fig, (ax_base, ax_blocked) = plt.subplots(1, 2, figsize=(20, 7.5))
    base = cache.get_gf(building, gf, gf_grid)
    plot_eikonal_field(building, gf, gf_grid, base, ax=ax_base, title="GF baseline (all exits)")
    print(f"GF baseline: T range [{base.T.min():.1f}, {base.T.max():.1f}] s over {base.domain.sum()} cells")

    overlay = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    blocked = cache.get_gf(building, gf, gf_grid, overlay=overlay)
    plot_eikonal_field(building, gf, gf_grid, blocked, ax=ax_blocked, title="GF: block_main_exit (X-S closed)")
    print(f"GF block_main_exit: T range [{blocked.T.min():.1f}, {blocked.T.max():.1f}] s")

    fig.suptitle(f"{building.id} — Eikonal field T^eik, heatmap + direction quiver", fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.86, wspace=0.28)
    p1 = _save(fig, OUT_DIR / "eikonal_gf.png")
    print(f"wrote {p1}")

    # ---- FF: baseline vs stair_C_lost ----
    fig, (ax_base, ax_blocked) = plt.subplots(1, 2, figsize=(20, 7.5))
    ff_base = cache.get_ff(building, ff, ff_grid, gf_grid)
    plot_eikonal_field(building, ff, ff_grid, ff_base, ax=ax_base, title="FF baseline (all 3 stairs)")
    print(f"FF baseline: T range [{ff_base.T.min():.1f}, {ff_base.T.max():.1f}] s over {ff_base.domain.sum()} cells")

    blocked_stairs = stairs_blocked_by_scenario(site, building, "stair_C_lost")
    ff_blocked = cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=blocked_stairs)
    plot_eikonal_field(building, ff, ff_grid, ff_blocked, ax=ax_blocked, title="FF: stair_C_lost (ST-C unavailable)")
    print(f"FF stair_C_lost: T range [{ff_blocked.T.min():.1f}, {ff_blocked.T.max():.1f}] s")

    fig.suptitle(f"{building.id} — FF Eikonal field (multi-floor stair coupling)", fontsize=14, fontweight="bold", y=1.0)
    fig.subplots_adjust(top=0.86, wspace=0.28)
    p2 = _save(fig, OUT_DIR / "eikonal_ff.png")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
