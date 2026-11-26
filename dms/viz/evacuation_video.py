"""spec §4 demo pipeline: renders one `simulate()` run to an MP4 (with a
title card, a pause-and-banner on the scenario's first BLOCKED_UNKNOWN
discovery, and an end hold) plus a `frames.json` export matched field-for-
field to the independent reference checker's own web-player format (so the
existing player can replay a run produced by this module directly) and a
background PNG with per-panel pixel-extent metadata.

Generic over any building/scenario -- every id drawn here (exits, stairs,
APs) comes from whatever the loaded Building/Site/EvacuationWorld declare.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from ..geometry.building import Building
from ..geometry.site import Site
from ..grid.layers import DynamicOverlay
from ..scenario.evacuation import (
    DISC,
    DONE,
    IN_REFUGE,
    MOVE,
    PRE,
    QUEUE,
    STAIR,
    Agent,
    EvacuationWorld,
    SimResult,
    summarize,
)
from ..scenario.presets import (
    active_groups_from_events,
    blocked_stairs_from_events,
    door_states_from_events,
)
from .plot import draw_site_panel, plot_floor_layers
from .viz_config import load_viz_config

FIG_W_IN, FIG_H_IN = 16.0, 9.0
DPI = 110  # 1760 x 990 px, fixed web-player canvas
FPS = 6
VIDEO_DT_S = 0.5  # matches evacuation.REC_EVERY * evacuation.DT
TITLE_CARD_S = 3.0
DISCOVERY_PAUSE_S = 3.0
END_HOLD_S = 4.0

_VIZ = load_viz_config()
_STATE_LABELS = {
    PRE: "Not yet moving (pre-movement)",
    MOVE: "Evacuating",
    QUEUE: "Queueing at stair",
    DISC: "Found exit blocked (pausing)",
    IN_REFUGE: "In refuge (safe, awaiting assisted evacuation)",
}

_GF_AX = (0.045, 0.55, 0.50, 0.34)
_FF_AX = (0.045, 0.10, 0.50, 0.34)
_SITE_AX = (0.615, 0.26, 0.355, 0.60)


def _agent_color(state: int, know: bool) -> str:
    if state == MOVE and know:
        return _VIZ.agent_state_colors["rerouted"]
    return _VIZ.agent_state_colors.get(
        {PRE: "PRE", MOVE: "MOVE", QUEUE: "QUEUE", DISC: "DISC", IN_REFUGE: "IN_REFUGE"}.get(state, "MOVE"),
        "#888888",
    )


def setup_figure(
    world: EvacuationWorld, building: Building, site: Site, *, title: str, subtitle: str,
    profile: str, n: int, seed: int,
):
    preset = world.site.scenarios[world.scenario] if world.scenario != "baseline" else {"events": []}
    scenario_overlay = DynamicOverlay(
        door_states=door_states_from_events(preset), active_groups=active_groups_from_events(preset),
        blocked_stairs=blocked_stairs_from_events(preset, building),
    )

    fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=DPI)
    fig.patch.set_facecolor("white")
    axg = fig.add_axes(_GF_AX)
    axf = fig.add_axes(_FF_AX)
    axs = fig.add_axes(_SITE_AX)

    floor_order = world.floor_order
    bottom, top = floor_order[0], floor_order[-1]
    plot_floor_layers(building, building.get_floor(bottom), world.grids[bottom], ax=axg,
                       title=f"{bottom} -- {title}", overlay=scenario_overlay)
    plot_floor_layers(building, building.get_floor(top), world.grids[top], ax=axf,
                       title=f"{top} -- {title}", overlay=scenario_overlay)
    draw_site_panel(axs, site, building, building.get_floor(bottom), world.site_grid,
                     overlay=scenario_overlay, title="Outdoor site")
    for ax in (axg, axf, axs):
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    fig.text(0.045, 0.975, f"B1-L evacuation  |  {title}", fontsize=15, fontweight="bold", va="center")
    fig.text(0.045, 0.945, subtitle, fontsize=9.5, color="#444", va="center")
    fig.text(0.98, 0.975, f"profile = {profile}   N = {n}   seed = {seed}   e = -grad T (FMM)", fontsize=9,
             ha="right", va="center", color="#444")

    handles = [
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["PRE"], mec=_VIZ.agent_pre_edge_color,
               ms=7, label=_STATE_LABELS[PRE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["MOVE"], mec="none", ms=7, label=_STATE_LABELS[MOVE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["QUEUE"], mec="none", ms=7, label=_STATE_LABELS[QUEUE]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["DISC"], mec="none", ms=7, label=_STATE_LABELS[DISC]),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["rerouted"], mec="none", ms=7,
               label="Re-routed after blocked exit"),
        Line2D([], [], marker="o", ls="", mfc=_VIZ.agent_state_colors["IN_REFUGE"], mec="none", ms=7,
               label=_STATE_LABELS[IN_REFUGE]),
        Patch(facecolor=_VIZ.exit_colors["AVAILABLE"], edgecolor="#333", label="Exit (available)"),
        Patch(facecolor=_VIZ.exit_colors["UNAVAILABLE"], edgecolor="#333", label="Exit (unavailable, known)"),
        Patch(facecolor=_VIZ.exit_colors["BLOCKED_UNKNOWN"], edgecolor="#333", hatch="////", label="Exit (blocked, undiscovered)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, fontsize=7.5,
               frameon=False, handletextpad=0.4, columnspacing=1.2)
    return fig, axg, axf, axs


def _axes_pixel_extents(fig, panels: dict[str, object]) -> dict[str, dict]:
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    out = {}
    for name, ax in panels.items():
        bb = ax.get_window_extent()
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        out[name] = {"px0": bb.x0, "px1": bb.x1, "py0": H - bb.y1, "py1": H - bb.y0, "x0": x0, "x1": x1, "y0": y0, "y1": y1}
    return out, (W, H)


class RunRenderer:
    """Holds the one-time figure + animated artists for a single run; `draw`
    updates them in place for frame `fr` -- called once per video frame and
    once per `frames.json` sample, so both outputs are built from exactly
    the same draw path."""

    def __init__(self, world: EvacuationWorld, agents: list[Agent], res: SimResult, *, title: str, subtitle: str, profile: str, seed: int):
        self.world = world
        self.n = len(agents)
        self.res = res
        n_floors = len(world.floor_order)
        self.site_idx = n_floors
        self.fig, self.axg, self.axf, self.axs = setup_figure(
            world, world.building, world.site, title=title, subtitle=subtitle, profile=profile, n=self.n, seed=seed,
        )
        bottom_id, top_id = world.floor_order[0], world.floor_order[-1]
        self.panel_names = [bottom_id, top_id, "SITE"]  # bottom floor, top floor, outdoor ("SITE" is a
        # structural layer name used throughout dms/ -- e.g. Node.floor == "SITE" in dms/graph/builder.py
        # -- not a building-specific id; the two floor labels come from world.floor_order itself.
        self.ax_px, self.size = _axes_pixel_extents(
            self.fig, {bottom_id: self.axg, top_id: self.axf, "SITE": self.axs},
        )
        self.scatter = {
            bottom_id: self.axg.scatter([], [], s=9, zorder=12, linewidths=0.4),
            top_id: self.axf.scatter([], [], s=9, zorder=12, linewidths=0.4),
            "SITE": self.axs.scatter([], [], s=7, zorder=12, linewidths=0.4),
        }
        self.clock = self.fig.text(0.615, 0.90, "", fontsize=16, fontweight="bold", va="center", family="monospace")
        self.hud = self.fig.text(0.615, 0.22, "", fontsize=8.2, va="top", family="monospace", linespacing=1.35)
        self.banner = self.fig.text(0.5, 0.90, "", fontsize=12, fontweight="bold", ha="center", va="center",
                                     color="white", bbox={"boxstyle": "round,pad=0.5", "fc": "#C0392B", "ec": "none"}, zorder=30)
        self.banner.set_visible(False)
        self.ap_txt = [
            self.axs.text(world.ap_xy[k, 0], world.ap_xy[k, 1], "", ha="center", va="center", fontsize=8,
                          fontweight="bold", zorder=13)
            for k in range(len(world.ap_ids))
        ]
        self.ex_names = [e.id for e in world.exits]
        self.stair_names = [s.id for s in world.stairs]

    def draw(self, fr) -> None:
        st, fl, pos, kn = fr.state, fr.floor, fr.pos, fr.know
        t = fr.t
        for fidx, key in enumerate(self.panel_names):
            m = (fl == fidx) & np.isin(st, (PRE, MOVE, QUEUE, DISC, IN_REFUGE))
            P = pos[m]
            cols = np.array([_agent_color(s, k) for s, k in zip(st[m], kn[m])]) if len(P) else np.empty((0,))
            ec = np.where(st[m] == PRE, _VIZ.agent_pre_edge_color, "none")
            sc = self.scatter[key]
            sc.set_offsets(P if len(P) else np.empty((0, 2)))
            sc.set_facecolors(cols if len(cols) else "none")
            sc.set_edgecolors(ec if len(ec) else "none")
            sc.set_sizes(np.full(len(P), 9 if fidx < self.site_idx else 7))

        done = st == DONE
        exited = ~np.isnan(self.res.t_exit) & (self.res.t_exit <= t)
        mm, ss = divmod(int(t), 60)
        self.clock.set_text(f"t = {mm:02d}:{ss:02d}  ({t:5.1f} s)")
        counts = [(self.ex_names[k], int(((self.res.exit_used == k) & exited).sum())) for k in range(len(self.ex_names))]
        half = max(1, len(counts) // 2 + len(counts) % 2)
        lines = [
            f"Not yet moving    {int((st == PRE).sum()):4d}",
            f"Evacuating        {int(((st == MOVE) & (fl < self.site_idx)).sum()):4d}",
            f"Queue at stairs   {int((st == QUEUE).sum()):4d}",
            f"On stairs         {int((st == STAIR).sum()):4d}",
            f"Outside, walking  {int(((st == MOVE) & (fl == self.site_idx)).sum()):4d}",
            f"In refuge (safe)  {int((st == IN_REFUGE).sum()):4d}",
            f"At assembly pt    {int(done.sum()):4d} / {self.n}",
            "",
            "Left building by exit:",
            "  " + "  ".join(f"{a}:{b}" for a, b in counts[:half]),
            "  " + "  ".join(f"{a}:{b}" for a, b in counts[half:]),
        ]
        self.hud.set_text("\n".join(lines))
        for k, tx in enumerate(self.ap_txt):
            c = int(((fr.state == DONE) & (self.res.ap_used == k)).sum())
            tx.set_text(str(c) if c else "")


def write_video_and_frames(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, *, scenario_title: str, subtitle: str,
    profile: str, seed: int, out_dir: Path, ffmpeg_exe: str | None = None,
) -> tuple[Path, Path, Path]:
    """Writes `{scenario}_{profile}_{n}agents.mp4`, `{scenario}_bg.png`, and
    `{scenario}_frames.json` into `out_dir`. Returns their paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n = len(agents)
    scen = world.scenario
    renderer = RunRenderer(world, agents, res, title=scenario_title, subtitle=subtitle, profile=profile, seed=seed)
    fig = renderer.fig

    bg_path = out_dir / f"{scen}_bg.png"
    fig.savefig(bg_path, dpi=DPI)

    if ffmpeg_exe is not None:
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_exe
    writer = FFMpegWriter(fps=FPS, bitrate=3500, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-preset", "medium"])
    mp4_path = out_dir / f"evac_{scen}_{profile}_{n}agents.mp4"
    frames = res.frames
    disc_t = res.events[0]["t"] if res.events else None

    with writer.saving(fig, mp4_path, dpi=DPI):
        renderer.draw(frames[0])
        renderer.banner.set_text(f"{scenario_title}: {n} people, alarm at t = 0 s")
        renderer.banner.set_visible(True)
        for _ in range(int(TITLE_CARD_S * FPS)):
            writer.grab_frame()
        renderer.banner.set_visible(False)

        paused = False
        for fr in frames:
            renderer.draw(fr)
            if disc_t is not None and not paused and fr.t >= disc_t:
                paused = True
                exit_id = res.events[0]["exit"]
                renderer.banner.set_text(f"Exit {exit_id} found BLOCKED on arrival  ->  agents pause, then re-route")
                renderer.banner.set_visible(True)
                for _ in range(int(DISCOVERY_PAUSE_S * FPS)):
                    writer.grab_frame()
                renderer.banner.set_visible(False)
            writer.grab_frame()

        renderer.banner.set_text(f"All {n} people at assembly points  -  t = {frames[-1].t:.0f} s")
        renderer.banner.set_visible(True)
        for _ in range(int(END_HOLD_S * FPS)):
            writer.grab_frame()
    plt.close(fig)

    frames_path = out_dir / f"{scen}_frames.json"
    _export_frames_json(world, agents, res, renderer, scenario_title=scenario_title, subtitle=subtitle,
                         profile=profile, seed=seed, out_path=frames_path)
    return mp4_path, bg_path, frames_path


FRAMES_SCHEMA_VERSION = 2  # v1 had no explicit
# version and no body-class data; v2 adds "schema_version", "body_classes" and
# "agent_body_class" while keeping every v1 field in place, so a v1-only player
# still works (it just won't draw true body sizes). "size" (already present
# since v1) is what a player should read its canvas dimensions from -- never a
# hardcoded 1760x1990.


def _export_frames_json(
    world: EvacuationWorld, agents: list[Agent], res: SimResult, renderer: RunRenderer, *,
    scenario_title: str, subtitle: str, profile: str, seed: int, out_path: Path,
) -> Path:
    """Packs exactly the schema the independent reference checker's own
    `video.py` writes (read in full before implementing this): base64 int16
    x/y in decimetres, a packed `uint8 code = floor<<4 | know<<3 | state`
    per agent per frame, plus frame times, per-panel pixel/data axis
    extents, exits, exit_used, t_exit, ap_used, aps, events and a
    human-readable summary -- so the existing web player can replay a run
    this module produced directly. Schema version 2 adds `schema_version`,
    a `body_classes` table and a per-agent `agent_body_class` index so the
    player can draw true body sizes."""
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

    pack = {
        "schema_version": FRAMES_SCHEMA_VERSION,
        "scenario": world.scenario, "title": scenario_title, "subtitle": subtitle, "profile": profile,
        "seed": seed, "n": n, "dt": VIDEO_DT_S,
        "times": [fr.t for fr in frames], "axes": renderer.ax_px, "size": list(renderer.size),
        "exits": renderer.ex_names, "exit_used": res.exit_used.tolist(),
        "t_exit": [None if np.isnan(v) else round(float(v), 1) for v in res.t_exit],
        "ap_used": res.ap_used.tolist(), "aps": world.ap_ids,
        "events": res.events,
        "body_classes": body_class_table, "agent_body_class": agent_body_class,
        "x": base64.b64encode(xs.tobytes()).decode(),
        "y": base64.b64encode(ys.tobytes()).decode(),
        "c": base64.b64encode(code.tobytes()).decode(),
        "summary": summarize(world, agents, res),
    }
    out_path = Path(out_path)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(pack, f)
    return out_path
