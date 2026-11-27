"""The presentation layout -- title bar, ground-floor
and top-floor plans side by side at the same scale, and a bottom band
(outdoor site map / live status panel / legend), at a chosen resolution
(1920x1080 "preview" or 3840x2160 "pitch" -- configs/viz.yaml
presentation.resolutions). `save_presentation_png` renders a single static
frame of a `simulate()` run at a given sim time; `write_presentation_video_and_frames`
renders the same layout as a full MP4 (title card, pause-and-banner on first
discovery, end hold) plus a schema-v3 `frames.json` export. Both share the
same panel-building code (`_build_presentation_panels`) so the video can
never silently drift from the already-approved static layout.

Generic over any building/scenario: every id/exit/AP/hazard drawn here
comes from whatever the loaded Site/Building/EvacuationWorld declare.
"""

from __future__ import annotations

from dataclasses import dataclass
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


def _legend_handles():
    """The legend's handle list -- identical for the static frame and the
    video, factored out once so the two renderers can't silently drift
    apart."""
    ebc = _VIZ.presentation.exit_bar_colors
    return [
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


@dataclass(slots=True)
class _PresentationPanels:
    """Everything built once per run -- the figure, its five axes, and the
    per-panel geometry needed to place true-body-size agents on any of
    them. Shared by the static-frame renderer and the video renderer so
    the two can never silently diverge in layout."""

    fig: object
    ax_bottom: object
    ax_top: object
    ax_site: object
    ax_status: object
    ax_legend: object
    panel_of: dict
    min_r: dict  # fidx -> this panel's min_agent_px floor, in data-unit (metre) radius
    radii: np.ndarray  # per-agent shoulder_width_m / 2, constant across frames
    fontsize_title: float
    fontsize_sub: float
    fontsize_clock: float


def _build_presentation_panels(
    world: EvacuationWorld, agents: list[Agent], *, scenario_title: str, profile: str, seed: int,
    hazard_seed: int | None = None, resolution: str = "preview",
) -> _PresentationPanels:
    """Builds the figure, the five axes (bottom floor, top floor, site,
    status, legend) and draws every static layer (floor plans, site map,
    scale bars, title-bar labels, legend) -- everything that does not
    depend on simulation time. Returns the panel handles a caller then
    populates with a clock, a live status text, and per-frame agents."""
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

    # Title bar (the static parts only -- clock is left to the caller).
    fontsize_title = px_to_pt(_VIZ.presentation.title_fontsize_px, _VIZ.presentation.preview_dpi)
    fontsize_sub = px_to_pt(_VIZ.presentation.subtitle_fontsize_px, _VIZ.presentation.preview_dpi)
    fontsize_clock = px_to_pt(_VIZ.presentation.clock_fontsize_px, _VIZ.presentation.preview_dpi)
    fig.text(0.03, _TITLE_Y0 + 0.045, f"{building.id} evacuation  |  {scenario_title}",
              fontsize=fontsize_title, fontweight="bold", va="center")
    fig.text(0.99, _TITLE_Y0 + 0.045, f"profile = {profile}   N = {len(agents)}   seed = {seed}",
              fontsize=fontsize_sub, ha="right", va="center", color="#444")

    # Legend (bottom, right) -- static, same handles for every caller.
    ax_legend.axis("off")
    ax_legend.legend(handles=_legend_handles(), loc="upper left",
                      fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px - 1, _VIZ.presentation.preview_dpi),
                      frameon=False, title="Legend", title_fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px, _VIZ.presentation.preview_dpi))

    # Per-panel agent geometry: true body size (shoulder_width_m / 2),
    # clamped to each panel's own min_agent_px floor -- same convention a
    # scatter marker can't express across panels at different px/m scales.
    site_idx = len(world.floor_order)
    panel_of = {0: ax_bottom, site_idx - 1: ax_top, site_idx: ax_site}
    panel_rect = {0: rect_bottom, site_idx - 1: rect_top, site_idx: _SITE_AX_RECT}
    _site_cols = world.site_grid.transform.shape[1]
    panel_extent_w = {
        0: extent_bottom[1] - extent_bottom[0], site_idx - 1: extent_top[1] - extent_top[0],
        site_idx: _site_cols * world.site_grid.transform.h,
    }
    radii = np.array(
        [world.agent_defaults.body_classes[a.body_class].shoulder_width_m / 2.0 for a in agents],
        dtype=np.float64,
    )
    min_r = {
        fidx: _min_agent_radius_m(panel_rect[fidx][2], panel_extent_w[fidx], _VIZ.presentation.min_agent_px, _VIZ.presentation.preview_dpi)
        for fidx in panel_of
    }

    return _PresentationPanels(
        fig=fig, ax_bottom=ax_bottom, ax_top=ax_top, ax_site=ax_site, ax_status=ax_status, ax_legend=ax_legend,
        panel_of=panel_of, min_r=min_r, radii=radii,
        fontsize_title=fontsize_title, fontsize_sub=fontsize_sub, fontsize_clock=fontsize_clock,
    )


def _draw_agents_at_frame(panels: _PresentationPanels, fr, *, remove_previous: dict | None = None) -> dict:
    """Draws (or redraws) true-body-size Circle patches for recorded frame
    `fr` on every agent-bearing panel. If `remove_previous` (a fidx ->
    list-of-patches map from an earlier call) is given, those patches are
    removed first -- this is how the video renderer animates without
    leaking patches frame over frame. Returns the new fidx -> patches map."""
    from .evacuation_video import _agent_color

    if remove_previous is not None:
        for fidx, patches in remove_previous.items():
            for p in patches:
                p.remove()

    new_patches: dict = {fidx: [] for fidx in panels.panel_of}
    for fidx, ax in panels.panel_of.items():
        m = (fr.floor == fidx) & np.isin(fr.state, (PRE, MOVE, QUEUE, DISC, IN_REFUGE))
        P = fr.pos[m]
        if not len(P):
            continue
        cols = [_agent_color(s, k) for s, k in zip(fr.state[m], fr.know[m])]
        rs = np.maximum(panels.radii[m], panels.min_r[fidx])
        for (px, py), r, c in zip(P, rs, cols):
            patch = ax.add_patch(Circle((px, py), r, facecolor=c, edgecolor=_VIZ.agent_outline,
                                          linewidth=0.5, zorder=_AGENT_Z))
            new_patches[fidx].append(patch)
    return new_patches


def render_presentation_frame(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, t: float,
    scenario_title: str, subtitle: str, profile: str, seed: int, hazard_seed: int | None = None,
    resolution: str = "preview",
):
    """Builds the figure (does not save it -- see `save_presentation_png`
    for that, which also controls the output resolution via dpi)."""
    panels = _build_presentation_panels(
        world, agents, scenario_title=scenario_title, profile=profile, seed=seed,
        hazard_seed=hazard_seed, resolution=resolution,
    )
    fig = panels.fig

    fig.text(0.03, _TITLE_Y0 + 0.012, subtitle, fontsize=panels.fontsize_sub, color="#444", va="center")
    snap = build_status_snapshot(world, agents, res, t)
    mm, ss = divmod(int(snap.t), 60)
    fig.text(0.99, _TITLE_Y0 + 0.012, f"t = {mm:02d}:{ss:02d}  ({snap.t:5.1f} s)",
              fontsize=panels.fontsize_clock, fontweight="bold", ha="right", va="center", family="monospace")

    # Agents, at the nearest recorded frame to t.
    frame_times = np.array([fr.t for fr in res.frames])
    fr = res.frames[int(np.argmin(np.abs(frame_times - t)))]
    _draw_agents_at_frame(panels, fr)

    # Status panel (bottom, middle).
    panels.ax_status.axis("off")
    panels.ax_status.set_title("Status", fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px + 2, _VIZ.presentation.preview_dpi),
                         fontweight="bold", loc="left")
    panels.ax_status.text(0.0, 0.95, "\n".join(status_lines(snap)), transform=panels.ax_status.transAxes,
                    fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px - 1, _VIZ.presentation.preview_dpi),
                    va="top", family="monospace", linespacing=1.4)

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


def _panel_pixel_extents(fig, panels: dict[str, object]) -> tuple[dict, tuple[int, int]]:
    """Same shape evacuation_video.py's `_axes_pixel_extents` already
    produces (px0/px1/py0/py1 in screen pixels, x0/x1/y0/y1 in data
    units), generalized to however many named panels are passed in --
    here, all five presentation panels, not just the three legacy ones."""
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    out = {}
    for name, ax in panels.items():
        bb = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        out[name] = {"px0": bb.x0, "px1": bb.x1, "py0": H - bb.y1, "py1": H - bb.y0, "x0": x0, "x1": x1, "y0": y0, "y1": y1}
    return out, (W, H)


FRAMES_SCHEMA_VERSION_V3 = 3  # v1/v2 (dms/viz/evacuation_video.py) describe the
# legacy 3-panel (GF/FF/site) layout at a fixed 1760x990 canvas. v3 adds a
# "layout" block (one pixel+data extent per presentation panel: the bottom
# floor, the top floor, the site map, the status panel, the legend) and the
# real canvas "size" for whichever resolution ("preview"/"pitch") this file
# was rendered at, so a player can place/scale agents without a hardcoded
# 1760x990 and can draw the two new chrome panels (status, legend) if it
# wants to, instead of only ever knowing about three map panels. Every v1/v2
# field (x/y/c, exits, body_classes, agent_body_class, summary, ...) is kept
# as-is for a player that only ever reads those.


def write_presentation_video_and_frames(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, scenario_title: str, subtitle: str,
    profile: str, seed: int, out_dir: str | Path, resolution: str = "preview",
    ffmpeg_exe: str | None = None, hazard_seed: int | None = None,
) -> tuple[Path, Path, Path]:
    """Renders one `simulate()` run to an MP4 in the presentation layout
    (title card, pause-and-banner on first discovery, end hold -- same
    semantics as the legacy `evacuation_video.write_video_and_frames`,
    reusing its timing constants and agent-colour function directly rather
    than redefining them) plus a background PNG and a schema-v3
    `frames.json`. `resolution` is "preview" (1920x1080) or "pitch"
    (3840x2160) -- see configs/viz.yaml `presentation.resolutions`."""
    from matplotlib.animation import FFMpegWriter

    from .evacuation_video import DISCOVERY_PAUSE_S, END_HOLD_S, FPS, TITLE_CARD_S

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(agents)
    scen = world.scenario
    dpi = _VIZ.presentation.preview_dpi if resolution == "preview" else _VIZ.presentation.pitch_dpi

    panels = _build_presentation_panels(
        world, agents, scenario_title=scenario_title, profile=profile, seed=seed,
        hazard_seed=hazard_seed, resolution=resolution,
    )
    fig = panels.fig
    fig.set_dpi(dpi)  # the figure's own stored dpi defaults to rcParams'
    # 100, not this resolution's actual render dpi -- must be set explicitly
    # before _panel_pixel_extents() below measures the canvas, or the
    # exported pixel extents/canvas size silently describe a 1600x900 (or
    # 3200x1800) image instead of the real 1920x1080 (or 3840x2160) one.
    fig.text(0.03, _TITLE_Y0 + 0.012, subtitle, fontsize=panels.fontsize_sub, color="#444", va="center")
    clock_text = fig.text(0.99, _TITLE_Y0 + 0.012, "", fontsize=panels.fontsize_clock, fontweight="bold",
                           ha="right", va="center", family="monospace")
    banner_text = fig.text(0.51, _MAIN_Y0 + _MAIN_H + 0.012, "", fontsize=panels.fontsize_title * 0.7,
                            fontweight="bold", ha="center", va="center", color="white",
                            bbox={"boxstyle": "round,pad=0.5", "fc": "#C0392B", "ec": "none"}, zorder=30)
    banner_text.set_visible(False)

    panels.ax_status.axis("off")
    panels.ax_status.set_title(
        "Status", fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px + 2, _VIZ.presentation.preview_dpi),
        fontweight="bold", loc="left",
    )
    status_text = panels.ax_status.text(
        0.0, 0.95, "", transform=panels.ax_status.transAxes,
        fontsize=px_to_pt(_VIZ.presentation.subtitle_fontsize_px - 1, _VIZ.presentation.preview_dpi),
        va="top", family="monospace", linespacing=1.4,
    )

    panel_px, canvas_size = _panel_pixel_extents(
        fig, {"bottom": panels.ax_bottom, "top": panels.ax_top, "site": panels.ax_site,
              "status": panels.ax_status, "legend": panels.ax_legend},
    )

    bg_path = out_dir / f"{scen}_bg_{resolution}.png"
    fig.savefig(bg_path, dpi=dpi)

    if ffmpeg_exe is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    writer = FFMpegWriter(fps=FPS, bitrate=6000 if resolution == "pitch" else 3500, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    mp4_path = out_dir / f"evac_{scen}_{profile}_{n}agents_{resolution}.mp4"
    frames = res.frames
    disc_t = res.events[0]["t"] if res.events else None

    agent_patches: dict = {fidx: [] for fidx in panels.panel_of}

    def _update(fr) -> None:
        nonlocal agent_patches
        agent_patches = _draw_agents_at_frame(panels, fr, remove_previous=agent_patches)
        snap = build_status_snapshot(world, agents, res, fr.t)
        mm, ss = divmod(int(snap.t), 60)
        clock_text.set_text(f"t = {mm:02d}:{ss:02d}  ({snap.t:5.1f} s)")
        status_text.set_text("\n".join(status_lines(snap)))

    with writer.saving(fig, mp4_path, dpi=dpi):
        _update(frames[0])
        banner_text.set_text(f"{scenario_title}: {n} people, alarm at t = 0 s")
        banner_text.set_visible(True)
        for _ in range(int(TITLE_CARD_S * FPS)):
            writer.grab_frame()
        banner_text.set_visible(False)

        paused = False
        for fr in frames:
            _update(fr)
            if disc_t is not None and not paused and fr.t >= disc_t:
                paused = True
                exit_id = res.events[0]["exit"]
                banner_text.set_text(f"Exit {exit_id} found BLOCKED on arrival  ->  agents pause, then re-route")
                banner_text.set_visible(True)
                for _ in range(int(DISCOVERY_PAUSE_S * FPS)):
                    writer.grab_frame()
                banner_text.set_visible(False)
            writer.grab_frame()

        banner_text.set_text(f"All {n} people at assembly points  -  t = {frames[-1].t:.0f} s")
        banner_text.set_visible(True)
        for _ in range(int(END_HOLD_S * FPS)):
            writer.grab_frame()
    plt.close(fig)

    frames_path = out_dir / f"{scen}_frames_{resolution}.json"
    _export_presentation_frames_json(
        world, agents, res, panel_px=panel_px, canvas_size=canvas_size, resolution=resolution,
        scenario_title=scenario_title, subtitle=subtitle, profile=profile, seed=seed, out_path=frames_path,
    )
    return mp4_path, bg_path, frames_path


def _export_presentation_frames_json(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, panel_px: dict, canvas_size: tuple[int, int],
    resolution: str, scenario_title: str, subtitle: str, profile: str, seed: int, out_path: Path,
) -> Path:
    """Schema v3 -- every v1/v2 field (dms/viz/evacuation_video.py) kept
    as-is, plus a `layout` block (one pixel+data extent per presentation
    panel) and the real `size` for this resolution, so a player can place
    true-body-size agents on the new 5-panel canvas without a hardcoded
    1760x990."""
    import base64
    import json

    from .evacuation_video import VIDEO_DT_S

    frames = res.frames
    n = len(agents)
    F = len(frames)
    xs = np.zeros((F, n), dtype=np.int16)
    ys = np.zeros((F, n), dtype=np.int16)
    code = np.zeros((F, n), dtype=np.uint8)
    for i, fr in enumerate(frames):
        xs[i] = np.round(fr.pos[:, 0] * 10).astype(np.int16)
        ys[i] = np.round(fr.pos[:, 1] * 10).astype(np.int16)
        code[i] = (fr.floor.astype(np.uint8) << 4) | fr.state.astype(np.uint8) | (fr.know.astype(np.uint8) << 3)

    body_class_ids = sorted({a.body_class for a in agents}) or ["standard"]
    body_class_table = [
        {"id": bcid, "shoulder_width_m": world.agent_defaults.body_classes[bcid].shoulder_width_m}
        for bcid in body_class_ids
    ]
    bc_index = {bcid: i for i, bcid in enumerate(body_class_ids)}
    agent_body_class = [bc_index[a.body_class] for a in agents]

    ex_names = [e.id for e in world.exits]
    pack = {
        "schema_version": FRAMES_SCHEMA_VERSION_V3,
        "scenario": world.scenario, "title": scenario_title, "subtitle": subtitle, "profile": profile,
        "seed": seed, "n": n, "dt": VIDEO_DT_S, "resolution": resolution,
        "times": [fr.t for fr in frames],
        "canvas": {"width_px": canvas_size[0], "height_px": canvas_size[1]},
        "layout": panel_px,  # panel name -> {px0,px1,py0,py1,x0,x1,y0,y1} -- NOT the same
        # panel-naming convention as v1/v2's "axes" (those keyed by floor id; this layout has
        # 5 roles -- bottom/top/site/status/legend -- not 1:1 with v1/v2's 3 floor-id keys), so
        # a v1/v2-only reader should not expect "layout" to look like its old "axes" block.
        "size": list(canvas_size),  # same [width, height] shape v1/v2 readers already expect
        "exits": ex_names, "exit_used": res.exit_used.tolist(),
        "t_exit": [None if np.isnan(v) else round(float(v), 1) for v in res.t_exit],
        "ap_used": res.ap_used.tolist(), "aps": world.ap_ids,
        "events": res.events,
        "body_classes": body_class_table, "agent_body_class": agent_body_class,
        "x": base64.b64encode(xs.tobytes()).decode(),
        "y": base64.b64encode(ys.tobytes()).decode(),
        "c": base64.b64encode(code.tobytes()).decode(),
    }
    from ..scenario.evacuation import summarize

    pack["summary"] = summarize(world, agents, res)
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pack, f)
    return out_path
