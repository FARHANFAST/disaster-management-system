"""A quick 90s baseline clip, agents coloured
by body class (not state) and drawn at true shoulder width, with first-
floor can_use_stairs=False agents ringed as "awaiting refuge" -- NOT the
full presentation layout. Reuses evacuation_video.py's existing panel
layout fractions and plot.py's background drawing, at the progress-
diagnostics resolution/clip length from configs/viz.yaml.

Run as:

    python -m dms.viz.progress_baseline_clip

Writes out/progress/phase1_baseline_clip.mp4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D

from ..geometry.loader import load_site
from ..scenario.evacuation import (
    DISC,
    DONE,
    IN_REFUGE,
    MOVE,
    PRE,
    QUEUE,
    Agent,
    EvacuationWorld,
    SimResult,
    build_world,
    sample_agents,
    simulate,
)
from .evacuation_video import _FF_AX, _GF_AX, _SITE_AX
from .plot import draw_site_panel, plot_floor_layers
from .viz_config import load_viz_config

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "progress"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey_large.yaml"
SCENARIO = "baseline"
PROFILE = "day"
SEED = 7
_VIZ = load_viz_config()


def _points_per_metre(ax) -> float:
    bbox = ax.get_window_extent()
    fig_dpi = ax.figure.dpi
    width_in = bbox.width / fig_dpi
    x0, x1 = ax.get_xlim()
    return (width_in * 72.0) / max(1e-6, (x1 - x0))


def render_progress_clip(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, ffmpeg_exe: str | None = None,
    out_name: str = "phase1_baseline_clip.mp4", title: str | None = None,
) -> Path:
    body_class = np.array([a.body_class for a in agents])
    can_use_stairs = np.array([a.can_use_stairs for a in agents], dtype=bool)
    shoulder_m = np.array([world.agent_defaults.body_classes[a.body_class].shoulder_width_m for a in agents])

    fig = plt.figure(figsize=(_VIZ.progress_fig_width_px / _VIZ.progress_dpi, _VIZ.progress_fig_height_px / _VIZ.progress_dpi),
                      dpi=_VIZ.progress_dpi)
    fig.patch.set_facecolor("white")
    axg = fig.add_axes(_GF_AX)
    axf = fig.add_axes(_FF_AX)
    axs = fig.add_axes(_SITE_AX)

    bottom, top = world.floor_order[0], world.floor_order[-1]
    plot_floor_layers(world.building, world.building.get_floor(bottom), world.grids[bottom], ax=axg,
                       title=f"{bottom} -- {SCENARIO} (by body class)")
    plot_floor_layers(world.building, world.building.get_floor(top), world.grids[top], ax=axf,
                       title=f"{top} -- {SCENARIO} (by body class)")
    draw_site_panel(axs, world.site, world.building, world.building.get_floor(bottom), world.site_grid,
                     title="Outdoor site")
    for ax in (axg, axf, axs):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.text(0.045, 0.975, title or f"B1-L evacuation progress diagnostic -- {SCENARIO}, first {_VIZ.progress_clip_seconds:.0f}s",
              fontsize=14, fontweight="bold", va="center")
    clock = fig.text(0.615, 0.90, "", fontsize=16, fontweight="bold", va="center", family="monospace")

    handles = [
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_class_colors[cid], mec="none", ms=8, label=cid)
        for cid in world.agent_defaults.body_classes
    ]
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=_VIZ.awaiting_refuge_ring_color, mew=2, ms=10,
                            label="awaiting refuge (can't use stairs, off the ground floor)"))
    handles.append(Line2D([], [], marker="o", ls="", mfc="none", mec=_VIZ.refuge_edge, mew=2.4, ms=10,
                            label="in refuge (safe, awaiting assisted evacuation)"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 0.0))

    panel_names = [bottom, top, "SITE"]
    axes_by_name = {bottom: axg, top: axf, "SITE": axs}
    pts_per_m = {name: _points_per_metre(axes_by_name[name]) for name in panel_names}
    scatter = {name: axes_by_name[name].scatter([], [], zorder=12, linewidths=1.2) for name in panel_names}
    ring = {name: axes_by_name[name].scatter([], [], zorder=13, facecolors="none",
                                              edgecolors=_VIZ.awaiting_refuge_ring_color, linewidths=1.8)
            for name in panel_names}
    safe_ring = {name: axes_by_name[name].scatter([], [], zorder=13, facecolors="none",
                                                    edgecolors=_VIZ.refuge_edge, linewidths=2.2)
                 for name in panel_names}

    clip_frames = [fr for fr in res.frames if fr.t <= _VIZ.progress_clip_seconds]

    def draw(fr) -> None:
        clock.set_text(f"t = {fr.t:5.1f} s")
        for fidx, name in enumerate(panel_names):
            visible = (fr.floor == fidx) & np.isin(fr.state, (PRE, MOVE, QUEUE, DISC, IN_REFUGE))
            P = fr.pos[visible]
            colors = [_VIZ.agent_class_colors.get(c, "#888888") for c in body_class[visible]]
            sizes = (shoulder_m[visible] * pts_per_m[name]) ** 2
            sc = scatter[name]
            sc.set_offsets(P if len(P) else np.empty((0, 2)))
            sc.set_facecolors(colors if len(colors) else "none")
            sc.set_sizes(sizes if len(sizes) else np.array([]))

            awaiting = visible & (~can_use_stairs) & (fr.floor != 0) & ~np.isin(fr.state, (DONE, IN_REFUGE))
            Pr = fr.pos[awaiting]
            rsizes = (shoulder_m[awaiting] * pts_per_m[name] * 1.6) ** 2
            rg = ring[name]
            rg.set_offsets(Pr if len(Pr) else np.empty((0, 2)))
            rg.set_sizes(rsizes if len(rsizes) else np.array([]))

            safe = visible & (fr.state == IN_REFUGE)
            Ps = fr.pos[safe]
            ssizes = (shoulder_m[safe] * pts_per_m[name] * 1.6) ** 2
            sg = safe_ring[name]
            sg.set_offsets(Ps if len(Ps) else np.empty((0, 2)))
            sg.set_sizes(ssizes if len(ssizes) else np.array([]))

    if ffmpeg_exe is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    writer = FFMpegWriter(fps=_VIZ.progress_clip_fps, bitrate=3500, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    out_path = OUT_DIR / out_name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, out_path, dpi=_VIZ.progress_dpi):
        for fr in clip_frames:
            draw(fr)
            writer.grab_frame()
    plt.close(fig)
    return out_path


def main() -> None:
    import imageio_ffmpeg

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    world = build_world(site, building, SCENARIO)
    agents = sample_agents(world, PROFILE, seed=SEED)
    res = simulate(world, agents, t_end=_VIZ.progress_clip_seconds, seed=1)
    out = render_progress_clip(world, agents, res, ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe())
    print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
