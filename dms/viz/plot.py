"""Plot rasterised floor layers (walkable, walls, spaces, doors, stairs,
hazards) for a single floor at a given resolution — PROMPT_01 §8 deliverable.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..geometry.building import Building, Floor
from ..grid.layers import FloorGrid

_KIND_CATEGORY = {
    "outside": 0, "wall": 1, "room": 2, "corridor": 3, "stub": 4, "stair": 5, "void": 6, "obstacle": 7,
}
_CATEGORY_COLORS = [
    "#dfe7ef",  # 0 outside (2 m margin)
    "#2b2b2b",  # 1 wall
    "#e8e4d9",  # 2 room
    "#cfe3f0",  # 3 corridor
    "#bcd7ea",  # 4 stub
    "#3d7ea6",  # 5 stair
    "#c9b7e0",  # 6 void (not walkable)
    "#8a8f98",  # 7 furniture (static obstacle)
]
_CMAP = ListedColormap(_CATEGORY_COLORS)

_DOOR_COLOR = "#c4a35a"
_EXIT_COLOR = "#c45c26"
_HAZARD_COLOR = "#d62828"


def _category_grid(floor: Floor, grid: FloorGrid) -> np.ndarray:
    kind_of_space = np.array([_KIND_CATEGORY[sp.kind] for sp in floor.spaces.values()], dtype=np.int8)
    cat = np.full(grid.shape, _KIND_CATEGORY["outside"], dtype=np.int8)
    inside = grid.space_id != -1
    cat[inside] = kind_of_space[grid.space_id[inside]]
    cat[grid.wall] = _KIND_CATEGORY["wall"]
    cat[grid.obstacle_static] = _KIND_CATEGORY["obstacle"]
    return cat


def plot_floor_layers(building: Building, floor: Floor, grid: FloorGrid, *, ax=None, title: str | None = None):
    fig, ax = (plt.subplots(figsize=(12, 8.5)) if ax is None else (ax.figure, ax))

    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    rows, cols = grid.shape
    extent = (x0, x0 + cols * h, y0 + rows * h, y0)  # left,right,bottom,top for origin="upper"

    cat = _category_grid(floor, grid)
    ax.imshow(cat, cmap=_CMAP, vmin=0, vmax=len(_CATEGORY_COLORS) - 1, extent=extent,
              interpolation="nearest", origin="upper")

    door_rgba = np.zeros((rows, cols, 4))
    is_exit = grid.exit_id != -1
    is_interior_door = (grid.door_id != -1) & ~is_exit
    door_rgba[is_interior_door] = to_rgba(_DOOR_COLOR)
    door_rgba[is_exit] = to_rgba(_EXIT_COLOR)
    ax.imshow(door_rgba, extent=extent, interpolation="nearest", origin="upper")

    primary_floor_id = building.floor_ids()[0]
    for stair in building.stairs.values():
        space_id = stair.gf_space if floor.id == primary_floor_id else stair.ff_space
        sp = floor.spaces.get(space_id)
        if sp is not None:
            cx, cy = sp.rect.center
            ax.text(cx, cy, stair.id, ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white", zorder=6)

    if floor.hazards:
        hx = [hz.x for hz in floor.hazards.values()]
        hy = [hz.y for hz in floor.hazards.values()]
        ax.scatter(hx, hy, marker="*", s=220, c=_HAZARD_COLOR, edgecolors="black", linewidths=0.8, zorder=7)
        for hz in floor.hazards.values():
            ax.annotate(hz.id, (hz.x, hz.y), textcoords="offset points", xytext=(6, 6),
                        fontsize=8, fontweight="bold", color=_HAZARD_COLOR, zorder=7)

    ax.set_title(title or f"{building.id} {floor.id} — rasterised layout (h={h} m)", fontsize=12, fontweight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color="black")

    legend_handles = [
        Patch(facecolor=_CATEGORY_COLORS[2], edgecolor="#333", label="Room"),
        Patch(facecolor=_CATEGORY_COLORS[3], edgecolor="#333", label="Corridor / stub"),
        Patch(facecolor=_CATEGORY_COLORS[5], edgecolor="#333", label="Stair"),
        Patch(facecolor=_CATEGORY_COLORS[6], edgecolor="#333", label="Void (not walkable)"),
        Patch(facecolor=_CATEGORY_COLORS[7], edgecolor="#333", label="Furniture"),
        Patch(facecolor=_CATEGORY_COLORS[1], edgecolor="#333", label="Wall"),
        Patch(facecolor=_CATEGORY_COLORS[0], edgecolor="#333", label="Outside (margin)"),
        Patch(facecolor=_DOOR_COLOR, edgecolor="#333", label="Door"),
        Patch(facecolor=_EXIT_COLOR, edgecolor="#333", label="Exit"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=_HAZARD_COLOR,
               markeredgecolor="black", markersize=14, label="Hazard source"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)

    return fig, ax


def save_floor_plot(building: Building, floor: Floor, grid: FloorGrid, out_path: str | Path, **kwargs) -> Path:
    fig, _ax = plot_floor_layers(building, floor, grid, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
