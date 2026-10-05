"""Supervisor demo: 20 agents, 2 scenarios, walking the Eikonal direction field
e = -grad(T)/|grad(T)| end-to-end -- room -> corridor -> (stair -> GF) ->
exterior exit -> outdoor -> nearest assembly point. Rendered as an MP4 (pausable,
scrubbable in any player) and a slow GIF. Built on the verified M1-M6 artifacts;
it adds no solver, just a stepper and a renderer.

Run as:

    python -m dms.viz.demo_simulation

Writes out/evacuation_demo_20agents.mp4 and out/evacuation_demo_20agents.gif and
prints the per-agent completion table. The MP4 needs an ffmpeg binary, supplied by
the imageio-ffmpeg package.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

from ..fields.eikonal import FieldCache, direction_field
from ..fields.outdoor import (
    DISCHARGE_MARGIN_M,
    OutdoorNetwork,
    build_outdoor_network,
    exit_discharge_site_xy,
)
from ..geometry.building import Building
from ..geometry.loader import load_site
from ..geometry.site import Site
from ..grid.layers import H_COARSE, DynamicOverlay, FloorGrid
from ..grid.raster import rasterize_floor
from .plot import plot_floor_layers

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"
MP4_PATH = OUT_DIR / "evacuation_demo_20agents.mp4"
GIF_PATH = OUT_DIR / "evacuation_demo_20agents.gif"

DT = 0.05  # s, physics step (same as the M4 ghost walker)
V0 = 1.2  # m/s, walking speed
MAX_TIME_PHASE = 180.0  # s, per-phase cap (room->exit, exit->AP)
ARRIVE_EPS = 1e-6
VIDEO_DT = 0.1  # s of simulated time per MP4 frame
VIDEO_FPS = 10  # MP4 plays back at 1x simulated time
GIF_DT = 0.5  # s of simulated time per GIF frame
GIF_FPS = 10
TRAIL_SECONDS = 20.0  # fading trail length behind each agent
DPI = 150
FIG_SIZE = (16, 10)  # inches -> 2400 x 1500 px at 150 dpi
GF_XLIM = (-2.0, 62.0)
GF_YLIM = (42.0, -2.0)  # inverted y, matching the architectural plan orientation
SITE_MARGIN_M = 8.0

SCENARIOS = [("baseline", "Baseline"), ("block_main_exit", "block_main_exit (X-S blocked)")]

# (floor, room_id, agent_count) -- 20 agents total, GF first, then FF.
AGENT_PLAN = [
    ("GF", "G-05", 3), ("GF", "LOBBY", 2), ("GF", "G-01", 1), ("GF", "G-08", 1), ("GF", "G-11", 1),
    ("GF", "G-15", 1), ("GF", "G-16", 1),
    ("FF", "F-01", 3), ("FF", "F-10", 3), ("FF", "FN-stub", 1), ("FF", "F-04", 1), ("FF", "F-08", 1),
    ("FF", "F-13", 1),
]

AGENT_COLORS = [matplotlib.colormaps["tab20"](i) for i in range(20)]


# --------------------------------------------------------------------- walking


@dataclass(slots=True)
class PathPoint:
    t: float
    floor: str  # "GF" | "FF" | "SITE"
    x: float
    y: float


@dataclass(slots=True)
class AgentRun:
    agent_id: int
    floor: str
    room: str
    scenario: str
    path: list[PathPoint] = field(default_factory=list)
    arrived: bool = False
    exit_id: str | None = None
    ap_id: str | None = None
    total_time: float = 0.0
    reason: str = ""


@dataclass(slots=True)
class Track:
    times: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    floors: np.ndarray
    total_time: float
    arrived: bool

    @classmethod
    def from_run(cls, run: AgentRun) -> Track:
        return cls(
            times=np.array([p.t for p in run.path]),
            xs=np.array([p.x for p in run.path]),
            ys=np.array([p.y for p in run.path]),
            floors=np.array([p.floor for p in run.path]),
            total_time=run.total_time,
            arrived=run.arrived,
        )

    def index_at(self, t: float) -> int:
        return int(np.clip(np.searchsorted(self.times, t, side="right") - 1, 0, len(self.times) - 1))


def _scenario_overlays(name: str) -> tuple[DynamicOverlay, DynamicOverlay]:
    if name == "baseline":
        return DynamicOverlay(), DynamicOverlay()
    if name == "block_main_exit":
        ov = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
        return ov, ov
    raise ValueError(f"unknown scenario {name!r}")


def _dir_arrays(grid: FloorGrid, eik_field):
    degraded = eik_field.speed < 0.01
    return direction_field(eik_field.T, eik_field.domain, grid.transform.h, degraded=degraded)


def _try_step(transform, domain: np.ndarray, x: float, y: float, dx: float, dy: float, step_len: float):
    """Full 2D step if in-domain, else slide along whichever single axis is --
    the same wall response as the M4 ghost walker, for indoor and outdoor grids."""
    for nx, ny in ((x + dx * step_len, y + dy * step_len), (x + dx * step_len, y), (x, y + dy * step_len)):
        row, col = transform.local_to_cell(nx, ny)
        if transform.in_bounds(row, col) and domain[row, col]:
            return nx, ny, True
    return x, y, False


def _walk_indoor(
    grid: FloorGrid, eik_field, ex_arr: np.ndarray, ey_arr: np.ndarray, x: float, y: float, t0: float, *,
    stop_on_exit: bool, path: list[PathPoint],
) -> tuple[float, str | None, bool, str]:
    t = t0
    while t - t0 < MAX_TIME_PHASE:
        row, col = grid.transform.local_to_cell(x, y)
        if not grid.transform.in_bounds(row, col) or not eik_field.domain[row, col]:
            return t, None, True, "left the walkable domain"
        if stop_on_exit:
            if grid.exit_id[row, col] != -1:
                return t, grid.door_ids[grid.exit_id[row, col]], False, ""
        elif grid.stair_id[row, col] != -1:
            return t, grid.stair_ids[grid.stair_id[row, col]], False, ""
        dx, dy = ex_arr[row, col], ey_arr[row, col]
        if math.isnan(dx) or math.isnan(dy) or math.hypot(dx, dy) < ARRIVE_EPS:
            return t, None, True, "zero/undefined gradient (local extremum)"
        nx, ny, ok = _try_step(grid.transform, eik_field.domain, x, y, dx, dy, DT * V0)
        if not ok:
            return t, None, True, "blocked on all axes (wall collision)"
        x, y = nx, ny
        t += DT
        path.append(PathPoint(t=t, floor=grid.floor_id, x=x, y=y))
    return t, None, True, f"exceeded max_time ({MAX_TIME_PHASE:.0f}s)"


def _walk_outdoor(
    net: OutdoorNetwork, site: Site, ex_arr: np.ndarray, ey_arr: np.ndarray,
    x: float, y: float, t0: float, path: list[PathPoint],
) -> tuple[float, str | None, bool, str]:
    """Follows -grad of the combined nearest-AP field until the agent enters an
    assembly point's own circle (APs are gathering areas, not points)."""
    transform = net.site_grid.transform
    domain = net.site_grid.walkable
    t = t0
    while t - t0 < MAX_TIME_PHASE:
        for ap_id, ap in site.assembly_points.items():
            if math.hypot(x - ap.x, y - ap.y) <= ap.diameter_m / 2:
                return t, ap_id, False, ""
        row, col = transform.local_to_cell(x, y)
        if not transform.in_bounds(row, col) or not domain[row, col]:
            return t, None, True, "left the outdoor walkable domain"
        dx, dy = ex_arr[row, col], ey_arr[row, col]
        if math.isnan(dx) or math.isnan(dy) or math.hypot(dx, dy) < ARRIVE_EPS:
            return t, None, True, "zero/undefined gradient outdoors"
        nx, ny, ok = _try_step(transform, domain, x, y, dx, dy, DT * V0)
        if not ok:
            return t, None, True, "blocked on all axes outdoors"
        x, y = nx, ny
        t += DT
        path.append(PathPoint(t=t, floor="SITE", x=x, y=y))
    return t, None, True, f"exceeded max_time ({MAX_TIME_PHASE:.0f}s) outdoors"


def run_agent(
    agent_id: int, floor: str, room: str, start_xy: tuple[float, float], scenario: str, *,
    building: Building, site: Site, gf, ff, gf_grid: FloorGrid, ff_grid: FloorGrid, cache: FieldCache,
    net: OutdoorNetwork, outdoor_dir: tuple[np.ndarray, np.ndarray],
) -> AgentRun:
    gf_overlay, ff_overlay = _scenario_overlays(scenario)
    gf_field = cache.get_gf(building, gf, gf_grid, overlay=gf_overlay)
    run = AgentRun(agent_id=agent_id, floor=floor, room=room, scenario=scenario)
    x, y = start_xy
    run.path.append(PathPoint(t=0.0, floor=floor, x=x, y=y))
    t = 0.0

    if floor == "FF":
        ff_field = cache.get_ff(building, ff, ff_grid, gf_grid, overlay=ff_overlay)
        ex_ff, ey_ff = _dir_arrays(ff_grid, ff_field)
        t, _stair_id, stuck, reason = _walk_indoor(
            ff_grid, ff_field, ex_ff, ey_ff, x, y, 0.0, stop_on_exit=False, path=run.path,
        )
        if stuck:
            run.total_time, run.reason = t, reason
            return run
        # Stair landings share one local (x, y) on both floors: continue on GF from here.
        x, y = run.path[-1].x, run.path[-1].y
        run.path.append(PathPoint(t=t, floor="GF", x=x, y=y))

    ex_gf, ey_gf = _dir_arrays(gf_grid, gf_field)
    t, exit_id, stuck, reason = _walk_indoor(gf_grid, gf_field, ex_gf, ey_gf, x, y, t, stop_on_exit=True, path=run.path)
    if stuck:
        run.total_time, run.reason = t, reason
        return run
    run.exit_id = exit_id

    ex_x, ex_y = exit_discharge_site_xy(building, "GF", gf.doors[exit_id])
    t += DISCHARGE_MARGIN_M / V0
    run.path.append(PathPoint(t=t, floor="SITE", x=ex_x, y=ex_y))

    ex_out, ey_out = outdoor_dir
    t, ap_id, stuck, reason = _walk_outdoor(net, site, ex_out, ey_out, ex_x, ex_y, t, run.path)
    run.total_time, run.ap_id, run.reason = t, ap_id, reason
    run.arrived = not stuck
    return run


# --------------------------------------------------------------------- agent placement


def _start_points(grid: FloorGrid, rect, k: int) -> list[tuple[float, float]]:
    """k well-spread walkable cell centres inside `rect`: the first is the
    candidate nearest the room centre, each next one is the candidate farthest
    from those already chosen, so co-located agents fan out across the room."""
    h = grid.transform.h
    x0, y0 = grid.transform.origin_local
    rows, cols = grid.transform.shape
    xc = (np.arange(cols) + 0.5) * h + x0
    yc = (np.arange(rows) + 0.5) * h + y0
    X, Y = np.meshgrid(xc, yc)
    margin = min(1.0, min(rect.width, rect.height) / 4)
    mask = (
        grid.walkable_static
        & (X >= rect.x0 + margin) & (X < rect.x1 - margin)
        & (Y >= rect.y0 + margin) & (Y < rect.y1 - margin)
    )
    cand = np.column_stack([X[mask], Y[mask]])
    if cand.shape[0] < k:
        raise ValueError(f"room too small for {k} agents ({cand.shape[0]} walkable candidates)")
    cx, cy = rect.center
    chosen = [int(np.argmin(np.hypot(cand[:, 0] - cx, cand[:, 1] - cy)))]
    while len(chosen) < k:
        d = np.min(np.linalg.norm(cand[:, None, :] - cand[chosen][None, :, :], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return [(float(cand[i, 0]), float(cand[i, 1])) for i in chosen]


def build_agent_specs(building: Building, gf, ff, gf_grid: FloorGrid, ff_grid: FloorGrid) -> list[tuple[int, str, str, tuple[float, float]]]:
    specs = []
    for floor, room, count in AGENT_PLAN:
        floor_obj, grid = (gf, gf_grid) if floor == "GF" else (ff, ff_grid)
        for xy in _start_points(grid, floor_obj.spaces[room].rect, count):
            specs.append((len(specs) + 1, floor, room, xy))
    return specs


# --------------------------------------------------------------------- rendering


def _draw_outdoor_background(ax, net: OutdoorNetwork, site: Site) -> None:
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
        ax.add_patch(plt.Circle((ap.x, ap.y), ap.diameter_m / 2, facecolor="#22c55e", alpha=0.18,
                                edgecolor="#14532d", lw=1.0, zorder=3))
        ax.scatter([ap.x], [ap.y], marker="P", s=120, c="#22c55e", edgecolors="#14532d", linewidths=0.8, zorder=4)
        ax.text(ap.x, ap.y - 4, ap_id, ha="center", fontsize=9, fontweight="bold", color="#14532d", zorder=4)


def _outdoor_view(building: Building, site: Site, gf) -> tuple[tuple[float, float], tuple[float, float]]:
    """Crop box covering the building, every exit discharge point and every
    assembly point circle, plus a margin: the outer site whitespace is dropped."""
    pts = []
    fp = gf.footprint
    for lx, ly in [(fp.x0, fp.y0), (fp.x1, fp.y0), (fp.x1, fp.y1), (fp.x0, fp.y1)]:
        pts.append(building.to_site_xy(lx, ly))
    for door in gf.exit_doors:
        pts.append(exit_discharge_site_xy(building, "GF", door))
    radius = max(ap.diameter_m / 2 for ap in site.assembly_points.values())
    for ap in site.assembly_points.values():
        pts += [(ap.x - radius, ap.y - radius), (ap.x + radius, ap.y + radius)]
    xs, ys = zip(*pts)
    xlim = (min(xs) - SITE_MARGIN_M, max(xs) + SITE_MARGIN_M)
    ylim = (max(ys) + SITE_MARGIN_M, min(ys) - SITE_MARGIN_M)  # inverted, matching the floor plans
    return xlim, ylim


def _trail_segments(track: Track, t: float, idx: int, floor_id: str) -> np.ndarray:
    lo = int(np.searchsorted(track.times, t - TRAIL_SECONDS, side="left"))
    sl = slice(lo, idx + 1)
    keep = track.floors[sl] == floor_id
    xs, ys = track.xs[sl][keep], track.ys[sl][keep]
    if xs.size < 2:
        return np.empty((0, 2, 2))
    starts = np.column_stack([xs[:-1], ys[:-1]])
    ends = np.column_stack([xs[1:], ys[1:]])
    return np.stack([starts, ends], axis=1)


def build_animation(
    tracks: dict[tuple[str, int], Track], specs, building: Building, site: Site, gf, ff,
    gf_grid: FloorGrid, ff_grid: FloorGrid, net: OutdoorNetwork, frame_dt: float,
):
    floor_cols = ["GF", "FF", "SITE"]
    label_offset = {"GF": 0.45, "FF": 0.45, "SITE": 1.6}

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE)
    site_xlim, site_ylim = _outdoor_view(building, site, gf)
    timer_texts = []
    for row, (scenario, label) in enumerate(SCENARIOS):
        for col, floor_id in enumerate(floor_cols):
            ax = axes[row][col]
            if floor_id == "GF":
                plot_floor_layers(building, gf, gf_grid, ax=ax, title=f"GF -- {label}")
                ax.set_xlim(*GF_XLIM)
                ax.set_ylim(*GF_YLIM)
            elif floor_id == "FF":
                plot_floor_layers(building, ff, ff_grid, ax=ax, title=f"FF -- {label}")
                ax.set_xlim(*GF_XLIM)
                ax.set_ylim(*GF_YLIM)
            else:
                _draw_outdoor_background(ax, net, site)
                ax.set_title(f"Outdoor site -- {label}", fontsize=11, fontweight="bold")
                ax.set_xlim(*site_xlim)
                ax.set_ylim(*site_ylim)
            legend = ax.get_legend()
            if legend is not None:
                legend.remove()
            ax.set_xlabel("x (m)")
            ax.set_ylabel("y (m)")
        timer_texts.append(axes[row][0].text(
            0.015, 0.985, "", transform=axes[row][0].transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", color="#111827",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#333"},
        ))

    artists: dict[tuple[int, int, int], tuple] = {}
    for row in range(2):
        for col in range(3):
            ax = axes[row][col]
            for agent_id, _, _, _ in specs:
                color = AGENT_COLORS[agent_id - 1]
                trail = LineCollection([], linewidths=2.0, zorder=7)
                ax.add_collection(trail)
                (point,) = ax.plot([], [], "o", color=color, markersize=9, markeredgecolor="white",
                                   markeredgewidth=0.8, zorder=8)
                label = ax.text(0, 0, "", fontsize=8, fontweight="bold", color="#111827", zorder=9,
                                ha="left", va="bottom", visible=False,
                                bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "alpha": 0.75, "edgecolor": "none"})
                artists[(row, col, agent_id)] = (trail, point, label)

    t_end = max(tr.total_time for tr in tracks.values())
    frame_times = np.arange(0.0, t_end + frame_dt, frame_dt)

    handles = [Line2D([0], [0], marker="o", color=AGENT_COLORS[aid - 1], lw=0, markersize=8,
                       label=f"{aid}: {room} ({floor})") for aid, floor, room, _ in specs]
    handles.append(Line2D([0], [0], marker="*", color="#111827", lw=0, markersize=14, label="arrived"))
    fig.legend(handles=handles, loc="lower center", ncol=11, fontsize=8, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("B1 evacuation demo -- 20 agents, v0 = 1.2 m/s, e = -grad(T)/|grad(T)|",
                 fontsize=14, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.035, right=0.99, top=0.93, bottom=0.14, hspace=0.16, wspace=0.07)

    def draw(t: float):
        changed = []
        for row, (scenario, _label) in enumerate(SCENARIOS):
            timer_texts[row].set_text(f"{scenario}\nt = {t:.1f} s")
            changed.append(timer_texts[row])
            for agent_id, floor, _room, _xy in specs:
                tr = tracks[(scenario, agent_id)]
                idx = tr.index_at(t)
                arrived_now = tr.arrived and t >= tr.total_time
                for col, floor_id in enumerate(floor_cols):
                    trail, point, label = artists[(row, col, agent_id)]
                    if tr.floors[idx] == floor_id:
                        trail.set_segments(_trail_segments(tr, t, idx, floor_id))
                        point.set_data([tr.xs[idx]], [tr.ys[idx]])
                        point.set_marker("*" if arrived_now else "o")
                        point.set_markersize(16 if arrived_now else 10)
                        off = label_offset[floor_id]
                        label.set_position((tr.xs[idx] + off, tr.ys[idx] + off))
                        label.set_text(str(agent_id))
                        label.set_visible(True)
                    else:
                        trail.set_segments([])
                        point.set_data([], [])
                        label.set_visible(False)
                    changed += [trail, point, label]
        return changed

    def make(times: np.ndarray) -> animation.FuncAnimation:
        return animation.FuncAnimation(
            fig, lambda i: draw(float(times[i])), frames=len(times), interval=100, blit=False, cache_frame_data=False,
        )

    return fig, make, frame_times


def _frames_stride(frame_times: np.ndarray, dt: float) -> np.ndarray:
    stride = max(1, round(dt / (frame_times[1] - frame_times[0])))
    return frame_times[::stride]


# --------------------------------------------------------------------- main


def _exit_cell(run: AgentRun) -> str:
    if not run.arrived:
        return f"STUCK ({run.reason})"
    return f"{run.exit_id} -> {run.ap_id} ({run.total_time:.1f}s)"


def _print_summary(specs, runs: dict[tuple[str, int], AgentRun]) -> int:
    headers = ["Agent ID", "Start Room/Floor", "Baseline Exit (Time)", "Blocked Exit (Time)", "Status"]
    rows, stuck = [], 0
    for agent_id, floor, room, _xy in specs:
        base = runs[("baseline", agent_id)]
        block = runs[("block_main_exit", agent_id)]
        if not base.arrived or not block.arrived:
            status = "STUCK"
            stuck += 1
        elif base.exit_id != block.exit_id:
            delta = block.total_time - base.total_time
            status = f"REROUTED ({delta:+.1f}s)"
        else:
            status = "SAME EXIT"
        rows.append([str(agent_id), f"{room}/{floor}", _exit_cell(base), _exit_cell(block), status])
    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(headers)]

    def fmt(cells):
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print("\n" + "=" * 118)
    print("EVACUATION DEMO -- 20-AGENT SUMMARY (baseline vs block_main_exit)")
    print("=" * 118)
    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt(r))
    print("=" * 118)
    print(f"Agents reaching safety: {len(specs) - stuck}/{len(specs)} per scenario "
          f"x {len(SCENARIOS)} scenarios; stuck/deadlocked runs: {stuck}")
    return stuck


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def main() -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    ff = building.get_floor("FF")
    gf_grid = rasterize_floor(building, gf, H_COARSE)
    ff_grid = rasterize_floor(building, ff, H_COARSE)

    net = build_outdoor_network(site, building)
    outdoor_dir = direction_field(net.nearest_ap_field, net.site_grid.walkable, net.site_grid.transform.h)
    cache = FieldCache()

    specs = build_agent_specs(building, gf, ff, gf_grid, ff_grid)
    runs: dict[tuple[str, int], AgentRun] = {}
    tracks: dict[tuple[str, int], Track] = {}
    for scenario, _label in SCENARIOS:
        for agent_id, floor, room, xy in specs:
            run = run_agent(agent_id, floor, room, xy, scenario, building=building, site=site, gf=gf, ff=ff,
                            gf_grid=gf_grid, ff_grid=ff_grid, cache=cache, net=net, outdoor_dir=outdoor_dir)
            runs[(scenario, agent_id)] = run
            tracks[(scenario, agent_id)] = Track.from_run(run)

    stuck = _print_summary(specs, runs)
    if stuck:
        raise SystemExit(f"{stuck} agent run(s) did not reach safety")

    fig, make, frame_times = build_animation(tracks, specs, building, site, gf, ff, gf_grid, ff_grid, net, VIDEO_DT)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams["animation.ffmpeg_path"] = _ffmpeg_exe()
    writer = animation.FFMpegWriter(
        fps=VIDEO_FPS, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", "20"],
    )
    make(frame_times).save(MP4_PATH, writer=writer, dpi=DPI)
    print(f"wrote {MP4_PATH}")

    gif_frames = _frames_stride(frame_times, GIF_DT)
    anim_gif = make(gif_frames)
    anim_gif.save(GIF_PATH, writer="pillow", fps=GIF_FPS, dpi=DPI)
    print(f"wrote {GIF_PATH}")
    plt.close(fig)


if __name__ == "__main__":
    main()
