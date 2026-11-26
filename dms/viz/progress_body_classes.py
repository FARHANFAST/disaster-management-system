"""Quick progress diagnostics for body
classes/clearance-aware routing -- NOT the full presentation layout.
Generic over any building: every size/colour/clip-length comes from
configs/viz.yaml (dms/viz/viz_config.py), every body class from
configs/agent_defaults.yaml.

Run as:

    python -m dms.viz.progress_body_classes [building_yaml ...]

Writes out/progress/phase1_body_classes_<building_id>.png for each.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..fields.eikonal import clearance_domain
from ..geometry.body_class_report import door_passability_table
from ..geometry.building import Building
from ..geometry.loader import load_site
from ..grid.layers import H_FINE, DynamicOverlay
from ..grid.raster import rasterize_floor
from ..scenario.agent_defaults import AgentDefaults, load_agent_defaults
from .plot import north_arrow
from .viz_config import load_viz_config

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "progress"
_VIZ = load_viz_config()
_BUCKET_KEYS = ("standard_only", "not_mobility_aid", "not_wheelchair", "all_classes")  # ascending radius reach


def _bucket_grid(
    grid, overlay: DynamicOverlay, classes_by_radius: list[tuple[str, float]], door_margin_m: float,
) -> np.ndarray:
    """-1 = not walkable even for the smallest-radius class; otherwise the
    index into `classes_by_radius` (ascending radius) of the LARGEST-radius
    class that can still occupy each cell.

    Uses the same margin-inflated radius (`clearance_radius_m +
    door_margin_m/2`) as `door_passability_table`'s own formula, so the area
    shading and the door-line colouring (which reads the table) agree on one
    "clear for this class" convention instead of two silently different ones."""
    bucket = np.full(grid.shape, -1, dtype=np.int64)
    for idx, (_cid, radius) in enumerate(classes_by_radius):
        domain = clearance_domain(grid, overlay, radius + door_margin_m / 2.0)
        bucket[domain] = idx
    return bucket


def _narrowest_passing_class(door_id: str, floor_id: str, table, classes_by_radius: list[tuple[str, float]]) -> int:
    """Index (ascending radius) of the largest-radius class that can still
    pass this door -- -1 if not even the smallest-radius class can."""
    passable = {row.body_class for row in table if row.door_id == door_id and row.floor_id == floor_id and row.passable}
    best = -1
    for idx, (cid, _radius) in enumerate(classes_by_radius):
        if cid in passable:
            best = idx
    return best


def _bucket_color(idx: int, n_classes: int) -> str:
    if idx < 0:
        return _VIZ.category_colors.get("wall", "#2b2b2b")
    # map idx (0..n_classes-1, ascending radius) onto the 4 fixed display
    # buckets positionally -- "can only the narrowest class(es) pass" ...
    # "even the widest class can pass".
    key_idx = round(idx * (len(_BUCKET_KEYS) - 1) / max(1, n_classes - 1))
    return _VIZ.class_shade_colors[_BUCKET_KEYS[key_idx]]


def _passing_classes(door_id: str, floor_id: str, table, classes_by_radius: list[tuple[str, float]]) -> list[str]:
    passable = {row.body_class for row in table if row.door_id == door_id and row.floor_id == floor_id and row.passable}
    return [cid for cid, _r in classes_by_radius if cid in passable]


def _narrow_gaps(floor, *, max_gap_m: float = 1.2) -> list[tuple[float, float, float]]:
    """(x, y, gap_m) for every pair of active obstacles in the same space
    whose facing edges leave a clear gap below `max_gap_m` -- a generic
    detector (works for any building), not specific to any one fixture's
    furniture layout. Axis-aligned pairs only (shared x- or y-overlap),
    which covers the common "two pieces of furniture pinch a path" case."""
    out: list[tuple[float, float, float]] = []
    by_space: dict[str, list] = {}
    for obs in floor.obstacles.values():
        if not obs.active_by_default:
            continue
        by_space.setdefault(obs.space_id, []).append(obs)
    for obs_list in by_space.values():
        for i in range(len(obs_list)):
            for j in range(i + 1, len(obs_list)):
                a, b = obs_list[i].rect, obs_list[j].rect
                # Facing along x (side by side in y): their x-ranges overlap.
                x_overlap = min(a.x1, b.x1) - max(a.x0, b.x0)
                if x_overlap > 0:
                    gap = max(a.y0, b.y0) - min(a.y1, b.y1) if a.y0 >= b.y1 or b.y0 >= a.y1 else None
                    if gap is not None and 0 < gap < max_gap_m:
                        mx = (max(a.x0, b.x0) + min(a.x1, b.x1)) / 2.0
                        my = (min(a.y1, b.y1) + max(a.y0, b.y0)) / 2.0
                        out.append((mx, my, gap))
                # Facing along y (side by side in x): their y-ranges overlap.
                y_overlap = min(a.y1, b.y1) - max(a.y0, b.y0)
                if y_overlap > 0:
                    gap = max(a.x0, b.x0) - min(a.x1, b.x1) if a.x0 >= b.x1 or b.x0 >= a.x1 else None
                    if gap is not None and 0 < gap < max_gap_m:
                        mx = (min(a.x1, b.x1) + max(a.x0, b.x0)) / 2.0
                        my = (max(a.y0, b.y0) + min(a.y1, b.y1)) / 2.0
                        out.append((mx, my, gap))
    return out


def _draw_floor(ax, building: Building, floor_id: str, grid, overlay: DynamicOverlay,
                 classes_by_radius: list[tuple[str, float]], door_table, door_margin_m: float,
                 label_doors: bool = False) -> None:
    floor = building.get_floor(floor_id)
    bucket = _bucket_grid(grid, overlay, classes_by_radius, door_margin_m)
    n_classes = len(classes_by_radius)
    rgba = np.zeros(bucket.shape + (4,))
    for idx in range(-1, n_classes):
        from matplotlib.colors import to_rgba
        rgba[bucket == idx] = to_rgba(_bucket_color(idx, n_classes))

    tr = grid.transform
    x0, y0 = tr.origin_local
    rows, cols = grid.shape
    extent = (x0, x0 + cols * tr.h, y0, y0 + rows * tr.h)
    ax.imshow(rgba, extent=extent, origin="lower", interpolation="nearest")
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])

    for did, door in list(floor.doors.items()) + list(floor.openings.items()):
        idx = _narrowest_passing_class(did, floor_id, door_table, classes_by_radius)
        color = _bucket_color(idx, n_classes)
        mid = (door.lo + door.hi) / 2.0
        width_m = door.hi - door.lo
        if door.axis == "y":
            x, y = mid, door.coord
            dx, dy = width_m, 0
        else:
            x, y = door.coord, mid
            dx, dy = 0, width_m
        ax.plot([x - dx / 2, x + dx / 2], [y - dy / 2, y + dy / 2], color=color, linewidth=4, solid_capstyle="butt",
                 zorder=6, path_effects=None)
        if label_doors:
            classes = _passing_classes(did, floor_id, door_table, classes_by_radius)
            label = f"{width_m:.2f}m\n{'+'.join(classes) if classes else 'none'}"
            ax.annotate(label, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=5.5,
                         color="#1f2933", zorder=9, ha="left", va="bottom",
                         bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75})

    if label_doors:
        for gx, gy, gap in _narrow_gaps(floor):
            ax.annotate(f"gap {gap:.2f}m", (gx, gy), textcoords="offset points", xytext=(0, 0), fontsize=5.5,
                         color="#7c2d12", zorder=9, ha="center", va="center", fontweight="bold",
                         bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#7c2d12", "alpha": 0.85})

    ax.set_title(f"{building.id} {floor_id}", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    north_arrow(ax)


def _scale_bar(ax, length_m: float = 10.0) -> None:
    x0, x1 = ax.get_xlim()
    y0, _y1 = ax.get_ylim()
    bx = x0 + 0.05 * (x1 - x0)
    by = y0 + 0.05 * (ax.get_ylim()[1] - y0)
    ax.plot([bx, bx + length_m], [by, by], color="black", linewidth=3, zorder=10)
    ax.text(bx + length_m / 2, by, f"{length_m:.0f} m", ha="center", va="bottom", fontsize=9, fontweight="bold")


def render_body_class_diagnostic(
    building: Building, *, agent_defaults: AgentDefaults | None = None, h: float = H_FINE,
    out_name: str | None = None, label_doors: bool | None = None,
) -> Path:
    """`label_doors` (width + passing classes on every door, furniture-gap
    widths) defaults to on for a small building (<=10 doors+openings on
    its busiest floor -- a hand-built fixture) and off otherwise (a real
    building's door count makes per-door labels illegible clutter, not a
    readability aid) -- pass it explicitly to override either way."""
    agent_defaults = agent_defaults if agent_defaults is not None else load_agent_defaults()
    classes_by_radius = sorted(
        ((cid, bc.clearance_radius_m) for cid, bc in agent_defaults.body_classes.items()), key=lambda kv: kv[1],
    )
    door_table = door_passability_table(building, agent_defaults=agent_defaults)
    overlay = DynamicOverlay()
    floor_ids = building.floor_ids()

    if label_doors is None:
        busiest = max(
            (len(building.get_floor(fid).doors) + len(building.get_floor(fid).openings) for fid in floor_ids),
            default=0,
        )
        label_doors = busiest <= 10

    fig_w = _VIZ.progress_fig_width_px / _VIZ.progress_dpi
    fig_h = _VIZ.progress_fig_height_px / _VIZ.progress_dpi
    fig, axes = plt.subplots(1, len(floor_ids), figsize=(fig_w, fig_h), dpi=_VIZ.progress_dpi)
    if len(floor_ids) == 1:
        axes = [axes]

    for ax, fid in zip(axes, floor_ids):
        grid = rasterize_floor(building, building.get_floor(fid), h)
        _draw_floor(ax, building, fid, grid, overlay, classes_by_radius, door_table, agent_defaults.door_margin_m,
                     label_doors=label_doors)
        _scale_bar(ax)

    legend_handles = [
        Patch(facecolor=_VIZ.class_shade_colors[k], edgecolor="#333",
              label={"all_classes": "Clear for all classes", "not_wheelchair": "Clear up to mobility_aid (not wheelchair)",
                     "not_mobility_aid": "Clear up to large (not mobility_aid)", "standard_only": "Clear for standard only"}[k])
        for k in _BUCKET_KEYS
    ]
    legend_handles.append(Patch(facecolor=_VIZ.category_colors.get("wall", "#2b2b2b"), edgecolor="#333", label="Not walkable (any class)"))
    legend_handles.append(Line2D([0], [0], color=_VIZ.class_shade_colors["standard_only"], lw=4, label="Door/exit coloured by narrowest passing class"))
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    table_lines = ["Body class       shoulder(m)  clearance_r(m)  can_use_stairs"]
    for cid, bc in agent_defaults.body_classes.items():
        table_lines.append(f"{cid:<16} {bc.shoulder_width_m:>10.2f}  {bc.clearance_radius_m:>13.2f}  {bc.can_use_stairs!s:>13}")
    fig.text(0.01, 0.985, "\n".join(table_lines), fontsize=_VIZ.progress_table_fontsize_px * 0.72,
              family="monospace", va="top", ha="left",
              bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#333"})

    fig.suptitle(f"{building.id}: walkable space and door/exit passability by body class", fontsize=13, fontweight="bold")
    fig.subplots_adjust(top=0.88, bottom=0.14)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / (out_name or f"phase1_body_classes_{building.id}.png")
    fig.savefig(out_path, dpi=_VIZ.progress_dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    # The primary building gets the un-suffixed name; every other fixture
    # gets phase1_body_classes_<building-id>.png, to show this is generic.
    jobs = [("configs/buildings/two_storey_large.yaml", "phase1_body_classes.png")]
    if not sys.argv[1:]:
        jobs.append(("configs/buildings/three_storey_large.yaml", None))
    else:
        jobs = [(p, None) for p in sys.argv[1:]]
    for p, forced_name in jobs:
        site = load_site(Path(p))
        for building in site.buildings.values():
            name = forced_name if len(site.buildings) == 1 else None
            out = render_body_class_diagnostic(building, out_name=name)
            print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
