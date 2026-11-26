"""Plot rasterised floor layers (walkable, walls, spaces, doors, stairs,
hazards) for a single floor at a given resolution — spec §8 deliverable,
colors from configs/viz.yaml (spec §1.1).
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, to_rgba
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from ..geometry.building import Building, Floor
from ..geometry.site import Site
from ..grid.layers import DynamicOverlay, FloorGrid
from .viz_config import VizConfig, load_viz_config

_KIND_CATEGORY = {
    "outside": 0, "wall": 1, "room": 2, "corridor": 3, "stub": 4, "stair": 5, "void": 6, "obstacle": 7,
}
_CATEGORY_ORDER = ("outside", "wall", "room", "corridor", "stub", "stair", "void", "obstacle")

_VIZ: VizConfig = load_viz_config()
_CATEGORY_COLORS = [_VIZ.category_colors[name] for name in _CATEGORY_ORDER]
_CMAP = ListedColormap(_CATEGORY_COLORS)

_DOOR_COLOR = _VIZ.door_color
_HAZARD_COLOR = _VIZ.hazard_color
_EXIT_STATE_ORDER = ("AVAILABLE", "PARTIAL", "UNAVAILABLE", "BLOCKED_UNKNOWN")

# A fixed z-order tier for text that must always
# stay legible -- stair labels, exit labels, hazard labels -- so neither a
# sign marker nor an agent dot can ever draw on top of one. Everything
# else (signs, exit bars, agents) stays below this.
_LABEL_Z = 20

# Every floor the generated stair-sign marker shows a floor NUMBER below
# a down arrow (e.g. "↓1"): building.floor_ids()[0] is always the
# building's own ground/egress floor by construction (by floor ordering),
# so "the stairs lead down toward floor 1" is a structural fact, not a
# building-specific one -- this is never a per-building literal.
_STAIR_SIGN_FLOOR_NUMBER = 1


def north_arrow(ax, *, x: float = 0.05, y: float = 0.92, size: float = 0.05) -> None:
    """A small north-up indicator in axes fraction coordinates, so it sits
    in the same on-screen corner regardless of the panel's data extent."""
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - size), xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#1f2933", zorder=10,
        arrowprops={"arrowstyle": "-|>", "color": "#1f2933", "lw": 1.6, "shrinkA": 0, "shrinkB": 2},
    )


def scale_bar(ax, length_m: float, *, x: float = 0.05, y: float = 0.05) -> None:
    """A map scale bar -- `length_m` of real-world
    distance drawn as a short horizontal line near axes fraction `(x, y)`,
    labeled with its own length. Independent of the axes' own tick marks
    (a map panel like this hides those), so this is the only distance
    reference on such a panel."""
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    bar_x0 = x0 + x * (x1 - x0)
    bar_y = y0 + y * (y1 - y0)
    ax.plot([bar_x0, bar_x0 + length_m], [bar_y, bar_y], color="#1f2933", lw=2.4, solid_capstyle="butt", zorder=10)
    for cap_x in (bar_x0, bar_x0 + length_m):
        ax.plot([cap_x, cap_x], [bar_y - (y1 - y0) * 0.006, bar_y + (y1 - y0) * 0.006],
                 color="#1f2933", lw=2.4, zorder=10)
    ax.annotate(f"{length_m:g} m", (bar_x0 + length_m / 2.0, bar_y), textcoords="offset points", xytext=(0, 5),
                 ha="center", fontsize=8, fontweight="bold", color="#1f2933", zorder=10)


def wind_arrow(ax, *, direction_deg: float | None = None, x: float = 0.94, y: float = 0.78) -> None:
    """Placeholder for a future wind-direction indicator (spec: hidden until
    wind exists). Pass `direction_deg` (meteorological, 0=from north) once a
    wind model exists; until then this draws nothing."""
    if direction_deg is None:
        return
    import math

    rad = math.radians(direction_deg)
    dx, dy = -math.sin(rad) * 0.04, -math.cos(rad) * 0.04
    ax.annotate(
        "wind", xy=(x, y), xytext=(x - dx, y - dy), xycoords="axes fraction", textcoords="axes fraction",
        ha="center", va="center", fontsize=8, color="#0e7490", zorder=10,
        arrowprops={"arrowstyle": "-|>", "color": "#0e7490", "lw": 1.4, "shrinkA": 0, "shrinkB": 2},
    )


def _category_grid(floor: Floor, grid: FloorGrid) -> np.ndarray:
    kind_of_space = np.array([_KIND_CATEGORY[sp.kind] for sp in floor.spaces.values()], dtype=np.int8)
    cat = np.full(grid.shape, _KIND_CATEGORY["outside"], dtype=np.int8)
    inside = grid.space_id != -1
    cat[inside] = kind_of_space[grid.space_id[inside]]
    cat[grid.wall] = _KIND_CATEGORY["wall"]
    cat[grid.obstacle_static] = _KIND_CATEGORY["obstacle"]
    return cat


def _exit_state(floor: Floor, door_id: str, overlay: DynamicOverlay | None) -> str:
    door = floor.doors[door_id]
    if overlay is not None and door_id in overlay.door_states:
        return overlay.door_states[door_id]
    return door.default_state


def _draw_exits(ax, floor: Floor, grid: FloorGrid, extent: tuple[float, float, float, float],
                 overlay: DynamicOverlay | None) -> None:
    """One color fill per exit's live state (AVAILABLE/PARTIAL/UNAVAILABLE/
    BLOCKED_UNKNOWN), plus a X marker for the two blocked states and a hatch
    outline on BLOCKED_UNKNOWN — the viz-only way to tell "known blocked"
    (solid red X) from "blocked but not yet discovered" (hatched red X)."""
    rows, cols = grid.shape
    door_rgba = np.zeros((rows, cols, 4))
    is_exit = grid.exit_id != -1
    is_interior_door = (grid.door_id != -1) & ~is_exit
    door_rgba[is_interior_door] = to_rgba(_DOOR_COLOR)

    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    for idx, door_id in enumerate(grid.door_ids):
        mask = grid.exit_id == idx
        if not mask.any():
            continue
        state = _exit_state(floor, door_id, overlay)
        color = _VIZ.exit_colors.get(state, _VIZ.exit_colors["UNAVAILABLE"])
        door_rgba[mask] = to_rgba(color)

        rr, cc = np.nonzero(mask)
        cx = (cc.mean() + 0.5) * h + x0
        cy = (rr.mean() + 0.5) * h + y0
        marker = _VIZ.exit_markers.get(state)
        if marker:
            ax.scatter([cx], [cy], marker=marker, s=90, c="white", linewidths=2.2, zorder=7)
        hatch = _VIZ.exit_hatch.get(state)
        if hatch:
            r_lo, r_hi, c_lo, c_hi = rr.min(), rr.max() + 1, cc.min(), cc.max() + 1
            rx, ry = c_lo * h + x0, r_lo * h + y0
            ax.add_patch(Rectangle((rx, ry), (c_hi - c_lo) * h, (r_hi - r_lo) * h,
                                     facecolor="none", edgecolor="black", hatch=hatch, lw=0, zorder=6))

    ax.imshow(door_rgba, extent=extent, interpolation="nearest", origin="lower")


def px_to_pt(px: float, dpi: float) -> float:
    """Points for a given on-screen pixel size at the given dpi -- so a
    size declared in configs/viz.yaml as "N px at 1080p" renders at
    exactly N physical pixels when the figure is drawn at that dpi (and at
    2x pixels, same relative size, at the pitch dpi -- see presentation.py)."""
    return px * 72.0 / dpi


def _door_wall_and_outward(door, floor: Floor) -> tuple[float, float, float, float, float, float]:
    """The wall-line segment a door/exit sits on, (x0, y0, x1, y1), and the
    outward unit normal (nx, ny) -- away from its own interior space
    (`door.space_a`), the same "which side is inside" rule
    dms/geometry/signage_gen.py's `_door_heading_deg` uses for exit
    headings, independently re-derived here since plot.py doesn't import
    geometry/signage_gen (no viz -> field-solving dependency)."""
    inside = floor.spaces.get(door.space_a) if door.space_a else None
    if door.axis == "y":
        x0, y0, x1, y1 = door.lo, door.coord, door.hi, door.coord
        inward_is_below = inside is None or inside.rect.y1 <= door.coord + 1e-6
        nx, ny = 0.0, (1.0 if inward_is_below else -1.0)
    else:  # axis == "x": a vertical wall, coord is its X position, door spans Y from lo to hi
        x0, y0, x1, y1 = door.coord, door.lo, door.coord, door.hi
        inward_is_left = inside is None or inside.rect.x1 <= door.coord + 1e-6
        nx, ny = (1.0 if inward_is_left else -1.0), 0.0
    return x0, y0, x1, y1, nx, ny


def _draw_exit_bars(
    ax, floor: Floor, grid: FloorGrid, overlay: DynamicOverlay | None, *,
    exit_labels: dict[str, str] | None, discovered_exit_ids: frozenset[str],
) -> None:
    """Each exit is drawn on its own wall as a true-width
    thick bar with an outward arrow and a label (e.g. "X-N -> AP-N"),
    instead of `_draw_exits`'s cell-fill style -- state -> color/hatch:
    available green; locked (declared UNAVAILABLE) grey
    + hatch; blocked and known (BLOCKED_UNKNOWN, in `discovered_exit_ids`)
    solid red; blocked and not yet discovered hatched red."""
    cfg = _VIZ.presentation
    thickness = grid.transform.h * 3.0
    # Kept inside the rasterised "outside" margin (2 m, spec §1.1) so
    # the label never clips at the panel's own edge -- found doing so at
    # the old 2.0 m arrow + 0.4 m label gap, which landed right at/past
    # the margin boundary.
    arrow_len = max(1.2, grid.transform.h * 6.0)
    fontsize = px_to_pt(cfg.exit_label_fontsize_px, cfg.preview_dpi)
    # Two exits close together on the same wall (e.g. a
    # shared corner) can otherwise stack their labels on top of each
    # other -- greedily nudge a colliding label sideways, ALONG the wall
    # (not farther out, which would risk clipping the outside margin),
    # until it clears every label already placed.
    label_min_gap_m = 2.2
    placed_label_xy: list[tuple[float, float]] = []

    for door in floor.exit_doors:
        state = _exit_state(floor, door.id, overlay)
        x0, y0, x1, y1, nx, ny = _door_wall_and_outward(door, floor)
        hatch = None
        if state == "AVAILABLE":
            key = "AVAILABLE"
        elif state == "PARTIAL":
            key = "PARTIAL"
        elif state == "BLOCKED_UNKNOWN":
            if door.id in discovered_exit_ids:
                key = "BLOCKED_KNOWN"
            else:
                key, hatch = "BLOCKED_UNDISCOVERED", "////"
        else:  # UNAVAILABLE ("locked")
            key, hatch = "LOCKED", "////"
        color = cfg.exit_bar_colors.get(key, "#888888")

        if door.axis == "y":  # horizontal wall: the bar spans x, with a thin y-thickness
            rx, ry, rw, rh = x0, y0 - thickness / 2.0, x1 - x0, thickness
        else:  # vertical wall: the bar spans y, with a thin x-thickness
            rx, ry, rw, rh = x0 - thickness / 2.0, y0, thickness, y1 - y0
        ax.add_patch(Rectangle((rx, ry), rw, rh, facecolor=color, edgecolor="#222",
                                 linewidth=0.6, hatch=hatch, zorder=9))

        mx, my = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        ax.annotate(
            "", xy=(mx + nx * arrow_len, my + ny * arrow_len), xytext=(mx, my),
            arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.2, "shrinkA": 0, "shrinkB": 0}, zorder=10,
        )
        label = door.id if exit_labels is None else exit_labels.get(door.id, door.id)
        tx, ty = -ny, nx  # tangent along the wall
        lx0, ly0 = mx + nx * (arrow_len + 0.3), my + ny * (arrow_len + 0.3)
        shift, tries = 0.0, 0
        while tries < 6 and any(
            math.hypot((lx0 + tx * shift) - px, (ly0 + ty * shift) - py) < label_min_gap_m
            for px, py in placed_label_xy
        ):
            tries += 1
            shift = (tries // 2 + 1) * 1.4 * (1.0 if tries % 2 else -1.0)
        lx, ly = lx0 + tx * shift, ly0 + ty * shift
        placed_label_xy.append((lx, ly))
        # The panel's own axes sits close to the LEFT figure edge (main
        # area starts at only 3% of figure width) -- a west-wall label's
        # bbox was spilling past the canvas edge and getting truncated.
        # The main area has real headroom above/below,
        # so only the horizontal anchor needs to flip; flipping va too
        # pushed a north-wall label down far enough to cover that exit's
        # own "EXIT" sign just inside the door (tried, reverted).
        ha = "right" if nx > 0.3 else "left" if nx < -0.3 else "center"
        ax.annotate(label, (lx, ly), ha=ha, va="center",
                     fontsize=fontsize, fontweight="bold", color="#222",
                     bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "#222", "lw": 0.6, "alpha": 0.9},
                     zorder=_LABEL_Z)


def plot_floor_layers(
    building: Building, floor: Floor, grid: FloorGrid, *, ax=None, title: str | None = None,
    overlay: DynamicOverlay | None = None,
    exit_style: str = "fill",  # "fill" (plain cell-color style) | "bar" (thick wall bar + arrow + label)
    exit_labels: dict[str, str] | None = None,  # door_id -> label text, e.g. "X-N -> AP-N" ("bar" style only)
    discovered_exit_ids: frozenset[str] = frozenset(),  # BLOCKED_UNKNOWN doors known-blocked at this frame ("bar" style only)
    show_signs: bool = True,
    show_room_labels: bool = False,  # small per-room `use` labels
    hazard_scenario=None,  # active/inactive hazard-source fading
    show_axes: bool = True,  # False hides ticks/"x (m)"/"y (m)" (keeps scale bar + north arrow)
):
    fig, ax = (plt.subplots(figsize=(12, 8.5)) if ax is None else (ax.figure, ax))

    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    rows, cols = grid.shape
    # left,right,bottom,top, ascending — row 0 (smallest y, south) at the
    # bottom of the screen, paired with origin="lower" below (spec §1.1:
    # row increases with +y/north; north must render at the top).
    extent = (x0, x0 + cols * h, y0, y0 + rows * h)

    cat = _category_grid(floor, grid)
    ax.imshow(cat, cmap=_CMAP, vmin=0, vmax=len(_CATEGORY_COLORS) - 1, extent=extent,
              interpolation="nearest", origin="lower")

    if exit_style == "bar":
        _draw_exit_bars(ax, floor, grid, overlay, exit_labels=exit_labels, discovered_exit_ids=discovered_exit_ids)
    else:
        _draw_exits(ax, floor, grid, extent, overlay)

    if show_room_labels:
        fontsize = px_to_pt(_VIZ.presentation.room_label_fontsize_px, _VIZ.presentation.preview_dpi)
        for sp in floor.spaces.values():
            if sp.kind != "room" or not sp.use:
                continue
            if sp.rect.width < 1.5 or sp.rect.height < 1.0:
                continue  # too small to hold a readable label
            cx, cy = sp.rect.center
            ax.text(cx, cy, sp.use, ha="center", va="center", fontsize=fontsize, color="#333", alpha=0.85, zorder=3)

    # Hazard-room tint: a light red fill for rooms that physically contain a
    # hazard source (floor.hazards[*].space_id) -- deliberately independent
    # of Space.restricted (operator-only access). A restricted room with no
    # hazard (e.g. G-11's control room) is never tinted; a hazard room is
    # tinted whether or not it happens to also be restricted.
    hazard_space_ids = {hz.space_id for hz in floor.hazards.values() if hz.space_id}
    for space_id in hazard_space_ids:
        sp = floor.spaces.get(space_id)
        if sp is None:
            continue
        r = sp.rect
        ax.add_patch(Rectangle((r.x0, r.y0), r.width, r.height,
                                 facecolor=_VIZ.hazard_room_tint, edgecolor="none", alpha=0.6, zorder=2))

    for stair in building.stairs.values():
        space_id = stair.landings.get(floor.id)
        sp = floor.spaces.get(space_id) if space_id else None
        if sp is not None:
            cx, cy = sp.rect.center
            ax.text(cx, cy, stair.id, ha="center", va="center", fontsize=9, fontweight="bold",
                     color="white", zorder=_LABEL_Z)

    if floor.hazards:
        # Same active/inactive fading as draw_site_panel
        # -- a source named in hazard_scenario.releases draws at full
        # strength, every other declared source fades. No hazard_scenario
        # at all (every caller that doesn't pass one) draws every source active, as
        # before.
        active_ids = {r.source_id for r in hazard_scenario.releases} if hazard_scenario is not None else set(floor.hazards)
        near_right_edge = extent[1] - 0.15 * (extent[1] - extent[0])
        for hz in floor.hazards.values():
            is_active = hz.id in active_ids
            alpha = 1.0 if is_active else _VIZ.presentation.hazard_inactive_alpha
            ax.scatter([hz.x], [hz.y], marker="*", s=220, c=_HAZARD_COLOR, edgecolors="black",
                       linewidths=0.8, alpha=alpha, zorder=7)
            flip = hz.x > near_right_edge
            ax.annotate(f"{hz.id}{'' if is_active else ' (inactive)'}", (hz.x, hz.y), textcoords="offset points",
                        xytext=(-6 if flip else 6, 6), ha="right" if flip else "left",
                        fontsize=8, fontweight="bold", color=_HAZARD_COLOR, alpha=alpha, zorder=_LABEL_Z)

    # Refuge areas (marked wait zones for can_use_stairs=false
    # classes) and wayfinding signs (small arrows pointing toward the exit they
    # name) -- both no-ops for a building that declares neither.
    for refuge in floor.refuges.values():
        r = refuge.rect
        ax.add_patch(Rectangle((r.x0, r.y0), r.width, r.height, facecolor=_VIZ.refuge_fill,
                                 edgecolor=_VIZ.refuge_edge, linewidth=1.5, alpha=0.85, zorder=5))
        cx, cy = r.center
        ax.text(cx, cy, "R", ha="center", va="center", fontsize=9, fontweight="bold",
                 color=_VIZ.refuge_edge, zorder=6)

    if show_signs:
        # A sign's own display reads by type --
        # a small "EXIT" box at an exit-door sign, a down-arrow + floor
        # number at a stair sign, a plain directional arrow (toward the
        # sign's own heading_deg, the gradient direction already computed
        # when it was generated -- not a recomputed line to the target
        # door's raw coordinates, which can mislead through a bend)
        # everywhere else. All three stay below _LABEL_Z, never covering
        # a stair/exit/hazard label.
        for sign in floor.signs.values():
            if sign.sign_type == "exit_door":
                ax.annotate(
                    "EXIT", (sign.x, sign.y), ha="center", va="center", fontsize=6.5, fontweight="bold",
                    color="white", zorder=9,
                    bbox={"boxstyle": "round,pad=0.18", "fc": _VIZ.sign_arrow, "ec": "none"},
                )
                continue
            if sign.sign_type == "stair_level":
                ax.annotate(
                    f"↓{_STAIR_SIGN_FLOOR_NUMBER}", (sign.x, sign.y), ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white", zorder=9,
                    bbox={"boxstyle": "round,pad=0.15", "fc": _VIZ.sign_stair_box, "ec": "none"},
                )
                continue
            rad = math.radians(sign.heading_deg)
            dx, dy = math.cos(rad), math.sin(rad)
            arrow_len = 1.2
            ax.annotate(
                "", xy=(sign.x + dx * arrow_len, sign.y + dy * arrow_len), xytext=(sign.x, sign.y),
                arrowprops={"arrowstyle": "-|>", "color": _VIZ.sign_arrow, "lw": 2.2, "shrinkA": 0, "shrinkB": 0},
                zorder=9,
            )
            ax.scatter([sign.x], [sign.y], marker="s", s=22, c=_VIZ.sign_arrow, zorder=9)

    ax.set_title(title or f"{building.id} {floor.id} — rasterised layout (h={h} m)", fontsize=12, fontweight="bold")
    ax.set_aspect("equal")
    if show_axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True, alpha=0.15, color="black")
    else:
        # No ticks/axis labels on the
        # presentation layout's floor panels (the scale bar + north arrow
        # are the only distance reference) -- this also removes the
        # upper-floor panel's own y tick labels bleeding into the
        # ground-floor panel next to it.
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    north_arrow(ax)
    wind_arrow(ax)

    legend_handles = [
        Patch(facecolor=_CATEGORY_COLORS[2], edgecolor="#333", label="Room"),
        Patch(facecolor=_CATEGORY_COLORS[3], edgecolor="#333", label="Corridor / stub"),
        Patch(facecolor=_CATEGORY_COLORS[5], edgecolor="#333", label="Stair"),
        Patch(facecolor=_CATEGORY_COLORS[6], edgecolor="#333", label="Void (not walkable)"),
        Patch(facecolor=_CATEGORY_COLORS[7], edgecolor="#333", label="Furniture"),
        Patch(facecolor=_CATEGORY_COLORS[1], edgecolor="#333", label="Wall"),
        Patch(facecolor=_CATEGORY_COLORS[0], edgecolor="#333", label="Outside (margin)"),
        Patch(facecolor=_DOOR_COLOR, edgecolor="#333", label="Door"),
    ]
    if exit_style == "bar":
        ebc = _VIZ.presentation.exit_bar_colors
        legend_handles += [
            Patch(facecolor=ebc["AVAILABLE"], edgecolor="#333", label="Exit (available)"),
            Patch(facecolor=ebc["LOCKED"], edgecolor="#333", hatch="////", label="Exit (locked)"),
            Patch(facecolor=ebc["BLOCKED_KNOWN"], edgecolor="#333", label="Exit (blocked, known)"),
            Patch(facecolor=ebc["BLOCKED_UNDISCOVERED"], edgecolor="#333", hatch="////", label="Exit (blocked, undiscovered)"),
        ]
    else:
        legend_handles += [
            Patch(facecolor=_VIZ.exit_colors["AVAILABLE"], edgecolor="#333", label="Exit (available)"),
            Patch(facecolor=_VIZ.exit_colors["PARTIAL"], edgecolor="#333", label="Exit (partial)"),
            Patch(facecolor=_VIZ.exit_colors["UNAVAILABLE"], edgecolor="#333", label="Exit (unavailable, known)"),
            Patch(facecolor=_VIZ.exit_colors["BLOCKED_UNKNOWN"], edgecolor="#333", hatch="////", label="Exit (blocked, undiscovered)"),
        ]
    legend_handles += [
        Line2D([0], [0], marker="*", color="none", markerfacecolor=_HAZARD_COLOR,
               markeredgecolor="black", markersize=14, label="Hazard source"),
        Patch(facecolor=_VIZ.hazard_room_tint, edgecolor="none", alpha=0.6, label="Room with hazard source"),
    ]
    if floor.refuges:
        legend_handles.append(Patch(facecolor=_VIZ.refuge_fill, edgecolor=_VIZ.refuge_edge, label="Refuge (wait area)"))
    if show_signs and floor.signs:
        sign_types_here = {s.sign_type for s in floor.signs.values()}
        if "exit_door" in sign_types_here:
            legend_handles.append(Patch(facecolor=_VIZ.sign_arrow, edgecolor="none", label='Sign: "EXIT"'))
        if "stair_level" in sign_types_here:
            legend_handles.append(Patch(facecolor=_VIZ.sign_stair_box, edgecolor="none", label="Sign: stair (down)"))
        if "directional" in sign_types_here:
            legend_handles.append(Line2D([0], [0], color=_VIZ.sign_arrow, lw=2.2, marker="s", markersize=5,
                                          label="Sign: directional arrow"))
    ax.legend(handles=legend_handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=True)

    return fig, ax


def draw_site_panel(
    ax, site: Site, building: Building, gf: Floor, site_grid, *,
    overlay: DynamicOverlay | None = None,
    outdoor_agents: list[tuple[float, float]] | None = None,
    title: str | None = None,
    agent_defaults=None,
    hazard_scenario=None,
    show_axes: bool = True,
    show_scale_bar: bool = False,
) -> None:
    """Shared outdoor/site panel (spec §1.2): building footprint with
    state-colored exits, assembly points, every site-level hazard (e.g.
    H-4), and optional outdoor agent markers. `site_grid` already reflects
    whichever obstacle layers the caller rasterised it with (pass
    `rasterize_site(site, parking_on=True)`/`build_outdoor_network(...,
    parking_on=True)` to show the parking layer) — this function only
    draws what it's given, it never re-rasterises.
    """
    from ..fields.outdoor import exit_discharge_site_xy
    from ..scenario.agent_defaults import load_agent_defaults

    if agent_defaults is None:
        agent_defaults = load_agent_defaults()
    ap_m2_per_person = agent_defaults.site.ap_m2_per_person

    h = site_grid.transform.h
    x0, y0 = site_grid.transform.origin_local
    rows, cols = site_grid.transform.shape
    extent = (x0, x0 + cols * h, y0, y0 + rows * h)

    sc = _VIZ.site_colors
    base = np.zeros((rows, cols, 4))
    base[site_grid.walkable] = to_rgba(sc.get("open_ground", "#c5d6a1"))
    base[site_grid.building_mask] = to_rgba(sc.get("building", "#a8c5d4"))
    base[~site_grid.walkable & ~site_grid.building_mask] = to_rgba(sc.get("obstacle", "#8a8f98"))
    ax.imshow(base, extent=extent, origin="lower", interpolation="nearest")

    # Walkways, fence (+ gates), structures, weather
    # station -- all a no-op for a site/building that declares none.
    for wk in site.walkways.values():
        xs = [p[0] for p in wk.points]
        ys = [p[1] for p in wk.points]
        width = max(2.0, wk.width_m * 1.5)
        # "make walkways clearly visible" -- the old tan
        # (#d9cba3) was nearly the same luminance as open_ground's sage
        # green. A dark outline drawn first, then the (now higher-
        # contrast) walkway colour on top and slightly narrower, reads
        # against open ground AND the building footprint alike.
        ax.plot(xs, ys, color=sc.get("walkway_outline", "#2b2b2b"), linewidth=width + 1.4,
                 solid_capstyle="round", zorder=2, alpha=0.9)
        ax.plot(xs, ys, color=sc.get("walkway", "#7a5c2e"), linewidth=width,
                 solid_capstyle="round", zorder=2, alpha=0.95)

    for fence in site.fences.values():
        xs = [p[0] for p in fence.points]
        ys = [p[1] for p in fence.points]
        ax.plot(xs, ys, color=sc.get("fence", "#4b2e2e"), linewidth=1.6, linestyle=(0, (6, 3)), zorder=3)
        label_x = sum(xs[:-1]) / max(1, len(xs) - 1)  # last point repeats the first on a closed polygon
        label_y = min(ys) - 3.0
        # Renamed from "(perimeter fence)" -- the
        # assembly points sit OUTSIDE this fence by design (AP-sizing
        # clearance rules), so "perimeter" is
        # misleading; it only encloses the building area itself.
        ax.annotate(f"{fence.id} (building-area fence)", (label_x, label_y), ha="center", fontsize=7.5,
                     fontweight="bold", color=sc.get("fence", "#4b2e2e"), zorder=_LABEL_Z,
                     bbox={"boxstyle": "round,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8})
    for gate in site.gates.values():
        open_ = gate.default_state != "UNAVAILABLE"
        ax.scatter([gate.x], [gate.y], marker="|", s=260, linewidths=3,
                   c=sc.get("gate_open", "#2e7d32") if open_ else "#b91c1c", zorder=4)
        ax.annotate(gate.id, (gate.x, gate.y), ha="center", va="top", textcoords="offset points", xytext=(0, -9),
                     fontsize=6.5, fontweight="bold", color="#333", zorder=_LABEL_Z,
                     bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})

    for obs in site.obstacles.values():
        if obs.layer != "structure" or not obs.active_by_default:
            continue
        r = obs.rect
        ax.add_patch(plt.Rectangle((r.x0, r.y0), r.width, r.height, facecolor=sc.get("structure", "#5c6672"),
                                     edgecolor="#333", linewidth=0.8, zorder=3))
        label = obs.type or obs.id
        ax.annotate(label, (r.x0 + r.width / 2, r.y0 + r.height / 2), ha="center", va="center",
                     fontsize=6, color="white", fontweight="bold", zorder=4)

    for ws in site.weather_stations.values():
        ax.scatter([ws.x], [ws.y], marker="^", s=140, c="#0e7490", edgecolors="black", linewidths=0.8, zorder=6)
        ax.annotate(ws.id, (ws.x, ws.y), textcoords="offset points", xytext=(6, 4), fontsize=7,
                     fontweight="bold", color="#0e7490", zorder=6)
        if ws.wind_from_deg is not None:
            rad = math.radians(ws.wind_from_deg)
            dx, dy = -math.sin(rad) * 8.0, -math.cos(rad) * 8.0
            ax.annotate(
                "", xy=(ws.x, ws.y), xytext=(ws.x - dx, ws.y - dy), xycoords="data", textcoords="data",
                zorder=6, arrowprops={"arrowstyle": "-|>", "color": "#0e7490", "lw": 1.4, "shrinkA": 0, "shrinkB": 6},
            )

    for ap_id, ap in site.assembly_points.items():
        ax.add_patch(plt.Circle((ap.x, ap.y), ap.diameter_m / 2, facecolor=_VIZ.assembly_point_color,
                                  alpha=0.25, edgecolor="#8a6d00", lw=1.0, zorder=3))
        ax.scatter([ap.x], [ap.y], marker="P", s=120, c=_VIZ.assembly_point_color,
                   edgecolors="#8a6d00", linewidths=0.8, zorder=4)
        capacity = int(math.pi * (ap.diameter_m / 2) ** 2 / max(ap_m2_per_person, 1e-9))
        ax.text(ap.x, ap.y - ap.diameter_m / 2 - 3, f"{ap_id} (cap {capacity})",
                 ha="center", va="top", fontsize=8, fontweight="bold", color="#8a6d00", zorder=_LABEL_Z,
                 bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})

    for door in gf.exit_doors:
        ex, ey = exit_discharge_site_xy(building, gf.id, door)
        state = _exit_state(gf, door.id, overlay)
        color = _VIZ.exit_colors.get(state, _VIZ.exit_colors["UNAVAILABLE"])
        ax.scatter([ex], [ey], marker="s", s=60, c=color, edgecolors="#333", zorder=5)
        ax.annotate(door.id, (ex, ey), ha="left", va="bottom", textcoords="offset points", xytext=(5, 5),
                     fontsize=7, fontweight="bold", zorder=_LABEL_Z,
                     bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.8})

    if site.hazards:
        # An active source (named in hazard_scenario,
        # if one is given) draws at full strength; every other declared
        # source fades (hazard_inactive_alpha) -- still visible (it's
        # real site data) but visually secondary. With no hazard_scenario
        # at all, every source draws active (the default behaviour).
        active_ids = {r.source_id for r in hazard_scenario.releases} if hazard_scenario is not None else set(site.hazards)
        for hz in site.hazards.values():
            is_active = hz.id in active_ids
            alpha = 1.0 if is_active else _VIZ.presentation.hazard_inactive_alpha
            ax.scatter([hz.x], [hz.y], marker="*", s=220, c=_VIZ.hazard_color, edgecolors="black",
                       linewidths=0.8, alpha=alpha, zorder=7 if is_active else 6)
            ax.annotate(f"{hz.id}{'' if is_active else ' (inactive)'}", (hz.x, hz.y), textcoords="offset points",
                        xytext=(6, 6), fontsize=8, fontweight="bold", color=_VIZ.hazard_color, alpha=alpha,
                        zorder=_LABEL_Z)

    if outdoor_agents:
        ax_, ay_ = zip(*outdoor_agents)
        ax.scatter(ax_, ay_, marker="o", s=50, c=sc.get("agent", "#2563eb"), edgecolors="white",
                   linewidths=0.8, zorder=8)

    ax.set_title(title or f"{building.id}: outdoor site", fontsize=11, fontweight="bold")
    ax.set_aspect("equal")
    if show_axes:
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    north_arrow(ax)
    wind_set = hazard_scenario is not None and hazard_scenario.releases
    wind_arrow(ax, direction_deg=hazard_scenario.wind_from_deg if wind_set else None)
    if show_scale_bar:
        scale_bar(ax, _VIZ.presentation.scale_bar_m)


def save_floor_plot(building: Building, floor: Floor, grid: FloorGrid, out_path: str | Path, **kwargs) -> Path:
    fig, _ax = plot_floor_layers(building, floor, grid, **kwargs)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
