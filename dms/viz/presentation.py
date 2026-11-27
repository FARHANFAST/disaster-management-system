"""The presentation layout -- title bar, ground-floor
and top-floor plans side by side at the same scale, and a bottom band
(outdoor site map / live status panel / legend). Renders a single static
frame of a `simulate()` run at a given sim time, at a chosen resolution
(1920x1080 "preview" or 3840x2160 "pitch" -- configs/viz.yaml
presentation.resolutions). Video/frames.json/player updates are a later
step (spec §3-4, after the static-frame approval this module
exists to produce); this module only ever writes one PNG per call.

Generic over any building/scenario: every id/exit/AP/hazard drawn here
comes from whatever the loaded Site/Building/EvacuationWorld declare.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch

from ..geometry.building import Building
from ..geometry.site import Site
from ..graph.astar import route
from ..graph.builder import build_navigation_graph
from ..grid.layers import DynamicOverlay
from ..scenario.evacuation import (
    DISC,
    IN_REFUGE,
    MOVE,
    PRE,
    QUEUE,
    Agent,
    EvacuationWorld,
    SimResult,
)
from ..scenario.hazard_scenario import HazardScenario, resolve_hazard_scenario
from ..scenario.presets import (
    active_groups_from_events,
    blocked_stairs_from_events,
    door_states_from_events,
)
from .plot import draw_site_panel, plot_floor_layers, px_to_pt, scale_bar
from .status_panel import build_status_snapshot, status_lines
from .viz_config import load_viz_config

_VIZ = load_viz_config()

_STATE_LABELS = {
    PRE: "Not yet moving (pre-movement)",
    MOVE: "Evacuating",
    QUEUE: "Queueing at stair",
    DISC: "Found exit blocked (pausing)",
    IN_REFUGE: "In refuge (safe, awaiting assisted evacuation)",
}

# Layout fractions (figure-relative) -- structural geometry, same convention
# as evacuation_video.py's own _GF_AX/_FF_AX/_SITE_AX constants, not a
# per-run tunable.
_TITLE_Y0 = 0.93
_MAIN_Y0, _MAIN_H = 0.42, 0.49
_MAIN_X0, _MAIN_X1 = 0.03, 0.99
_MAIN_GAP = 0.02
_BOTTOM_Y0, _BOTTOM_H = 0.03, 0.34
_BOTTOM_GAP = 0.015
# Rebalanced from (site 0.36, status 0.30, legend 0.235) to
# site 45% / status 25% / legend 30% of the available band width (the site
# map was judged too small -- overlapping labels, hard-to-read walkways).
_BOTTOM_AVAIL = (_MAIN_X1 - _MAIN_X0) - 2 * _BOTTOM_GAP
_SITE_W = _BOTTOM_AVAIL * 0.45
_STATUS_W = _BOTTOM_AVAIL * 0.25
_LEGEND_W = _BOTTOM_AVAIL * 0.30
_SITE_AX_RECT = (_MAIN_X0, _BOTTOM_Y0, _SITE_W, _BOTTOM_H)
_STATUS_AX_RECT = (_MAIN_X0 + _SITE_W + _BOTTOM_GAP, _BOTTOM_Y0, _STATUS_W, _BOTTOM_H)
_LEGEND_AX_RECT = (_MAIN_X0 + _SITE_W + _STATUS_W + 2 * _BOTTOM_GAP, _BOTTOM_Y0, _LEGEND_W, _BOTTOM_H)

_AGENT_Z = 12  # below _LABEL_Z (dms/viz/plot.py) -- an agent dot never covers a stair/exit/hazard label
_FIGURE_WIDTH_IN = 16.0  # the fixed 16x9in figure (render_presentation_frame) -- used to convert a
# figure-fraction axes width into physical inches, for the true-body-size agent marker's pixel floor below


def _min_agent_radius_m(axes_width_fraction: float, extent_width_m: float, min_px: float, dpi: float) -> float:
    """The data-unit (metre) radius that renders as exactly `min_px`
    physical pixels on an axes of this fractional width, at this dpi --
    the floor under a true-body-size agent marker so it never disappears
    on a zoomed-out panel (e.g. the site map). Always evaluated at
    preview_dpi, by the same "stated in px at 1080p" convention px_to_pt
    uses: a size that clears the floor at 1080p renders strictly MORE
    physical pixels (not smaller) at the 2x pitch dpi, since the figure's
    own physical size in inches never changes."""
    axes_width_in = axes_width_fraction * _FIGURE_WIDTH_IN
    if axes_width_in <= 0 or extent_width_m <= 0:
        return 0.0
    data_units_per_px = extent_width_m / (axes_width_in * dpi)
    return min_px * data_units_per_px


def _exit_ap_labels(site: Site, building: Building, bottom_floor_id: str) -> dict[str, str]:
    """door_id -> "X-N -> AP-N": the assembly point each exit's cheapest
    route reaches, via the real outdoor walkway/gate graph -- the same ids
    the site map already uses for both exits and
    APs."""
    nav = build_navigation_graph(site, building.id)
    labels: dict[str, str] = {}
    for door in building.get_floor(bottom_floor_id).exit_doors:
        r = route(nav, f"X:{door.id}")
        if r is None or not r.nodes:
            labels[door.id] = door.id
            continue
        ap_node = r.nodes[-1]
        ap_id = ap_node.split(":", 1)[1] if ap_node.startswith("AP:") else ap_node
        labels[door.id] = f"{door.id} -> {ap_id}"
    return labels


def _side_by_side_rects(extent_bottom: tuple[float, float, float, float], extent_top: tuple[float, float, float, float]):
    """Two axes rects in [_MAIN_X0, _MAIN_X1] x [_MAIN_Y0, _MAIN_Y0+_MAIN_H],
    each panel's width proportional to its own floor's physical width, so
    a metre on one panel and a metre on the other end up the same number
    of on-screen pixels (`aspect='equal'` then keeps x/y matched within
    each panel)."""
    w_bottom = extent_bottom[1] - extent_bottom[0]
    w_top = extent_top[1] - extent_top[0]
    avail = (_MAIN_X1 - _MAIN_X0) - _MAIN_GAP
    w0 = avail * w_bottom / (w_bottom + w_top)
    w1 = avail * w_top / (w_bottom + w_top)
    rect_bottom = (_MAIN_X0, _MAIN_Y0, w0, _MAIN_H)
    rect_top = (_MAIN_X0 + w0 + _MAIN_GAP, _MAIN_Y0, w1, _MAIN_H)
    return rect_bottom, rect_top


def _floor_extent(grid) -> tuple[float, float, float, float]:
    x0, y0 = grid.transform.origin_local
    h = grid.transform.h
    rows, cols = grid.shape
    return (x0, x0 + cols * h, y0, y0 + rows * h)


def render_presentation_frame(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, t: float,
    scenario_title: str, subtitle: str, profile: str, seed: int, hazard_seed: int | None = None,
    resolution: str = "preview",
):
    """Builds the figure (does not save it -- see `save_presentation_png`
    for that, which also controls the output resolution via dpi)."""
    site, building = world.site, world.building
    preset = site.scenarios[world.scenario] if world.scenario != "baseline" else {"events": []}
    overlay = DynamicOverlay(
        door_states=door_states_from_events(preset), active_groups=active_groups_from_events(preset),
        blocked_stairs=blocked_stairs_from_events(preset, building),
    )
    # This scenario's own hazard data -- if it declares none
    # (the overwhelming majority of pure-egress scenarios, including
    # baseline), every declared hazard source renders as explicitly
    # inactive/faded, never "active by default".
    hazard_scenario = resolve_hazard_scenario(site, preset, agent_defaults=world.agent_defaults, seed=hazard_seed)
    if hazard_scenario is None:
        hazard_scenario = HazardScenario(releases=(), wind_from_deg=0.0, wind_speed_mps=0.0, stability="")

    bottom_id, top_id = world.floor_order[0], world.floor_order[-1]
    bottom_grid, top_grid = world.grids[bottom_id], world.grids[top_id]
    extent_bottom, extent_top = _floor_extent(bottom_grid), _floor_extent(top_grid)
    rect_bottom, rect_top = _side_by_side_rects(extent_bottom, extent_top)

    fig = plt.figure(figsize=(16.0, 9.0))
    fig.patch.set_facecolor("white")

    ax_bottom = fig.add_axes(rect_bottom)
    ax_top = fig.add_axes(rect_top)
    ax_site = fig.add_axes(_SITE_AX_RECT)
    ax_status = fig.add_axes(_STATUS_AX_RECT)
    ax_legend = fig.add_axes(_LEGEND_AX_RECT)

    # Room-use labels are small text inside often-small
    # rooms -- shown only in the 4K "pitch" render by default
    # (room_labels_pitch_only), hidden at 1080p "preview" to avoid
    # unreadable clutter rather than forced to a size that clears
    # min_text_px everywhere.
    show_room_labels = resolution == "pitch" or not _VIZ.presentation.room_labels_pitch_only

    exit_labels = _exit_ap_labels(site, building, bottom_id)
    plot_floor_layers(building, building.get_floor(bottom_id), bottom_grid, ax=ax_bottom,
                       title=f"{bottom_id} -- {scenario_title}", overlay=overlay,
                       exit_style="bar", exit_labels=exit_labels, show_room_labels=show_room_labels,
                       hazard_scenario=hazard_scenario, show_axes=False)
    plot_floor_layers(building, building.get_floor(top_id), top_grid, ax=ax_top,
                       title=f"{top_id} -- {scenario_title}", overlay=overlay,
                       exit_style="bar", exit_labels=exit_labels, show_room_labels=show_room_labels,
                       hazard_scenario=hazard_scenario, show_axes=False)
    draw_site_panel(ax_site, site, building, building.get_floor(bottom_id), world.site_grid,
                     overlay=overlay, title="Outdoor site", hazard_scenario=hazard_scenario,
                     agent_defaults=world.agent_defaults, show_axes=False, show_scale_bar=True)
    for ax in (ax_bottom, ax_top, ax_site):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    scale_bar(ax_bottom, _VIZ.presentation.scale_bar_m)
    scale_bar(ax_top, _VIZ.presentation.scale_bar_m)

    # Title bar.
    fontsize_title = px_to_pt(_VIZ.presentation.title_fontsize_px, _VIZ.presentation.preview_dpi)
    fontsize_sub = px_to_pt(_VIZ.presentation.subtitle_fontsize_px, _VIZ.presentation.preview_dpi)
    fontsize_clock = px_to_pt(_VIZ.presentation.clock_fontsize_px, _VIZ.presentation.preview_dpi)
    fig.text(0.03, _TITLE_Y0 + 0.045, f"{building.id} evacuation  |  {scenario_title}",
              fontsize=fontsize_title, fontweight="bold", va="center")
    fig.text(0.03, _TITLE_Y0 + 0.012, subtitle, fontsize=fontsize_sub, color="#444", va="center")
    fig.text(0.99, _TITLE_Y0 + 0.045, f"profile = {profile}   N = {len(agents)}   seed = {seed}",
              fontsize=fontsize_sub, ha="right", va="center", color="#444")
    snap = build_status_snapshot(world, agents, res, t)
    mm, ss = divmod(int(snap.t), 60)
    fig.text(0.99, _TITLE_Y0 + 0.012, f"t = {mm:02d}:{ss:02d}  ({snap.t:5.1f} s)",
              fontsize=fontsize_clock, fontweight="bold", ha="right", va="center", family="monospace")

    # Agents, at the nearest recorded frame to t, on the bottom/top/site
    # panels -- true body size, drawn as a Circle patch with radius =
    # shoulder_width_m / 2 (the agent's actual body, not the routing
    # clearance_radius_m it's eroded by) in DATA units (metres), which is
    # correct-to-scale regardless of each panel's own zoom -- a scatter
    # marker's `s` is in display points^2 and can't represent a metre size
    # consistently across panels with different px/m scales. Clamped to
    # each panel's own min_agent_px floor so an agent never shrinks below
    # visibility on a zoomed-out panel (the site map, mainly).
    from .evacuation_video import _agent_color

    site_idx = len(world.floor_order)
    panel_of = {0: ax_bottom, site_idx - 1: ax_top, site_idx: ax_site}
    panel_rect = {0: rect_bottom, site_idx - 1: rect_top, site_idx: _SITE_AX_RECT}
    _site_cols = world.site_grid.transform.shape[1]
    panel_extent_w = {
        0: extent_bottom[1] - extent_bottom[0], site_idx - 1: extent_top[1] - extent_top[0],
        site_idx: _site_cols * world.site_grid.transform.h,
    }
    frame_times = np.array([fr.t for fr in res.frames])
    fr = res.frames[int(np.argmin(np.abs(frame_times - t)))]
    radii = np.array(
        [world.agent_defaults.body_classes[a.body_class].shoulder_width_m / 2.0 for a in agents],
        dtype=np.float64,
    )
    for fidx, ax in panel_of.items():
        m = (fr.floor == fidx) & np.isin(fr.state, (PRE, MOVE, QUEUE, DISC, IN_REFUGE))
        P = fr.pos[m]
        if not len(P):
            continue
        cols = [_agent_color(s, k) for s, k in zip(fr.state[m], fr.know[m])]
        min_r = _min_agent_radius_m(
            panel_rect[fidx][2], panel_extent_w[fidx], _VIZ.presentation.min_agent_px, _VIZ.presentation.preview_dpi,
        )
        rs = np.maximum(radii[m], min_r)
        for (px, py), r, c in zip(P, rs, cols):
            ax.add_patch(Circle((px, py), r, facecolor=c, edgecolor=_VIZ.agent_outline,
                                  linewidth=0.5, zorder=_AGENT_Z))

    # Status panel (bottom, middle).
    ax_status.axis("off")
    ax_status.set_title("Status", fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px + 2, _VIZ.presentation.preview_dpi),
                         fontweight="bold", loc="left")
    ax_status.text(0.0, 0.95, "\n".join(status_lines(snap)), transform=ax_status.transAxes,
                    fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px - 1, _VIZ.presentation.preview_dpi),
                    va="top", family="monospace", linespacing=1.4)

    # Legend (bottom, right).
    ax_legend.axis("off")
    ebc = _VIZ.presentation.exit_bar_colors
    handles = [
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["PRE"], mec=_VIZ.agent_pre_edge_color,
               ms=7, label=_STATE_LABELS[PRE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["MOVE"], mec="none", ms=7, label=_STATE_LABELS[MOVE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["QUEUE"], mec="none", ms=7, label=_STATE_LABELS[QUEUE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["IN_REFUGE"], mec="none", ms=7, label=_STATE_LABELS[IN_REFUGE]),
        Patch(facecolor=ebc["AVAILABLE"], edgecolor="#333", label="Exit: available"),
        Patch(facecolor=ebc["LOCKED"], edgecolor="#333", hatch="////", label="Exit: locked"),
        Patch(facecolor=ebc["BLOCKED_KNOWN"], edgecolor="#333", label="Exit: blocked, known"),
        Patch(facecolor=ebc["BLOCKED_UNDISCOVERED"], edgecolor="#333", hatch="////", label="Exit: blocked, undiscovered"),
        Patch(facecolor=_VIZ.sign_arrow, edgecolor="none", label='Sign: "EXIT"'),
        Patch(facecolor=_VIZ.sign_stair_box, edgecolor="none", label="Sign: stair (down)"),
        Line2D([0], [0], color=_VIZ.sign_arrow, lw=2.2, marker="s", markersize=5, label="Sign: directional arrow"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=_VIZ.hazard_color, markeredgecolor="black",
               markersize=12, label="Hazard source (active)"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor=_VIZ.hazard_color, markeredgecolor="black",
               markersize=12, alpha=_VIZ.presentation.hazard_inactive_alpha, label="Hazard source (inactive)"),
    ]
    ax_legend.legend(handles=handles, loc="upper left", fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px - 1, _VIZ.presentation.preview_dpi),
                      frameon=False, title="Legend", title_fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px, _VIZ.presentation.preview_dpi))

    return fig


def save_presentation_png(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, t: float, out_path: str | Path,
    resolution: str = "preview", scenario_title: str, subtitle: str, profile: str, seed: int,
    hazard_seed: int | None = None,
) -> Path:
    fig = render_presentation_frame(
        world, agents, res, t=t, scenario_title=scenario_title, subtitle=subtitle, profile=profile, seed=seed,
        hazard_seed=hazard_seed, resolution=resolution,
    )
    dpi = _VIZ.presentation.preview_dpi if resolution == "preview" else _VIZ.presentation.pitch_dpi
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path
