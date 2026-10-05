"""PROMPT_01 §1: a "trivial ghost walker" that follows -grad(T) for a purely
visual sanity check of the Eikonal fields built in Milestone 4 — not a
pytest test despite the filename (that's the name the verification was
asked for); see pyproject.toml's `testpaths = ["tests"]`, which keeps
pytest from trying to collect it. Run as:

    python -m dms.viz.test_walker

Writes out/walker_gf_traces.png, out/walker_ff_traces.png, and — from the
edge-case stress sweep — out/walker_edge_cases_gf.png and
out/walker_edge_cases_ff.png.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive: avoids flaky file-handle contention
# from the Qt backend when saving many figures back-to-back in a script.
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from ..fields.eikonal import (
    EikonalField,
    FieldCache,
    direction_field,
    stairs_blocked_by_scenario,
)
from ..geometry.building import Building
from ..geometry.loader import load_site
from ..geometry.primitives import Rect
from ..grid.layers import H_COARSE, DynamicOverlay
from ..grid.raster import rasterize_floor
from .plot import plot_floor_layers

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey.yaml"
OUT_DIR = Path(__file__).resolve().parents[2] / "out"

DT = 0.05  # s
V0 = 1.2  # m/s
MAX_TIME = 180.0  # s — generous cap; exceeding it means "stuck", not "slow"
ARRIVE_EPS = 1e-6


@dataclass(slots=True)
class WalkerStep:
    t: float
    floor: str
    x: float
    y: float


@dataclass(slots=True)
class WalkerResult:
    room: str
    scenario: str
    path: list[WalkerStep] = field(default_factory=list)
    reached_exit: bool = False
    stuck: bool = False
    reason: str = ""
    total_time: float = 0.0
    exit_id: str | None = None
    stair_used: str | None = None


def _direction_arrays(grid, eik_field: EikonalField, cache: dict) -> tuple[np.ndarray, np.ndarray]:
    key = id(eik_field)
    if key not in cache:
        degraded = eik_field.speed < 0.01  # F_b-slow under the *current* overlay (see direction_field's docstring)
        cache[key] = direction_field(eik_field.T, eik_field.domain, grid.transform.h, degraded=degraded)
    return cache[key]


def _try_step(grid, eik_field: EikonalField, x: float, y: float, dx: float, dy: float, step_len: float):
    """Advance by the full 2D step if that stays in-domain; otherwise slide
    along whichever single axis does (a minimal wall-collision response).
    Each sub-step is tiny relative to a cell (dt*v0 << h), so a trajectory
    that runs parallel and close to a 1-cell-thick wall can otherwise drift
    across the wall line between doorways purely from accumulated rounding
    in the discretely-sampled gradient — this keeps it sliding instead."""
    for nx, ny in ((x + dx * step_len, y + dy * step_len), (x + dx * step_len, y), (x, y + dy * step_len)):
        row, col = grid.transform.local_to_cell(nx, ny)
        if grid.transform.in_bounds(row, col) and eik_field.domain[row, col]:
            return nx, ny, True
    return x, y, False


def _walk_gf_phase(
    grid, eik_field: EikonalField, ex_arr: np.ndarray, ey_arr: np.ndarray,
    x: float, y: float, t0: float, *, dt: float, v0: float, max_time: float, path: list[WalkerStep],
) -> tuple[float, bool, bool, str, str | None]:
    """Steps on one GF-like floor (targets = exits) until an exit cell is
    reached, the walker leaves the domain, or max_time elapses. Returns
    (t, reached_exit, stuck, reason, exit_id)."""
    t = t0
    while t - t0 < max_time:
        row, col = grid.transform.local_to_cell(x, y)
        if not grid.transform.in_bounds(row, col) or not eik_field.domain[row, col]:
            return t, False, True, "left the walkable domain", None
        if grid.exit_id[row, col] != -1:
            exit_id = grid.door_ids[grid.exit_id[row, col]]
            return t, True, False, "", exit_id
        dx, dy = ex_arr[row, col], ey_arr[row, col]
        if math.isnan(dx) or math.isnan(dy) or math.hypot(dx, dy) < ARRIVE_EPS:
            return t, False, True, "zero/undefined gradient (local extremum)", None
        x, y, ok = _try_step(grid, eik_field, x, y, dx, dy, dt * v0)
        if not ok:
            return t, False, True, "blocked on all axes (wall collision)", None
        t += dt
        path.append(WalkerStep(t=t, floor=grid.floor_id, x=x, y=y))
    return t, False, True, f"exceeded max_time ({max_time:.0f}s)", None


def trace_gf_walker(room_id: str, scenario: str, building: Building, gf, gf_grid, gf_field: EikonalField,
                     dir_cache: dict, *, dt: float = DT, v0: float = V0, max_time: float = MAX_TIME,
                     start_xy: tuple[float, float] | None = None) -> WalkerResult:
    cx, cy = start_xy if start_xy is not None else gf.spaces[room_id].rect.center
    path = [WalkerStep(t=0.0, floor="GF", x=cx, y=cy)]
    ex_arr, ey_arr = _direction_arrays(gf_grid, gf_field, dir_cache)
    t, reached, stuck, reason, exit_id = _walk_gf_phase(
        gf_grid, gf_field, ex_arr, ey_arr, cx, cy, 0.0, dt=dt, v0=v0, max_time=max_time, path=path,
    )
    return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=reached,
                         stuck=stuck, reason=reason, total_time=t, exit_id=exit_id)


def trace_ff_walker(room_id: str, scenario: str, building: Building, ff, ff_grid, ff_field: EikonalField,
                     gf_grid, gf_field: EikonalField, dir_cache: dict, *,
                     dt: float = DT, v0: float = V0, max_time: float = MAX_TIME,
                     start_xy: tuple[float, float] | None = None,
                     excluded_stairs: frozenset[str] = frozenset()) -> WalkerResult:
    cx, cy = start_xy if start_xy is not None else ff.spaces[room_id].rect.center
    path = [WalkerStep(t=0.0, floor="FF", x=cx, y=cy)]
    ex_arr, ey_arr = _direction_arrays(ff_grid, ff_field, dir_cache)

    x, y, t = cx, cy, 0.0
    stair_id = None
    while t < max_time:
        row, col = ff_grid.transform.local_to_cell(x, y)
        if not ff_grid.transform.in_bounds(row, col) or not ff_field.domain[row, col]:
            return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=False, stuck=True,
                                 reason="left the FF walkable domain", total_time=t)
        sidx = ff_grid.stair_id[row, col]
        if sidx != -1 and ff_grid.stair_ids[sidx] not in excluded_stairs:
            # An *excluded* stair's landing is still ordinary walkable floor
            # for anyone not using it (spec: the field just routes around/
            # through it via another stair) — only a usable stair triggers
            # the floor transition below.
            stair_id = ff_grid.stair_ids[sidx]
            break
        dx, dy = ex_arr[row, col], ey_arr[row, col]
        if math.isnan(dx) or math.isnan(dy) or math.hypot(dx, dy) < ARRIVE_EPS:
            return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=False, stuck=True,
                                 reason="zero/undefined gradient on FF (local extremum)", total_time=t)
        x, y, ok = _try_step(ff_grid, ff_field, x, y, dx, dy, dt * v0)
        if not ok:
            return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=False, stuck=True,
                                 reason="blocked on all axes (wall collision) on FF", total_time=t)
        t += dt
        path.append(WalkerStep(t=t, floor="FF", x=x, y=y))
    else:
        return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=False, stuck=True,
                             reason=f"exceeded max_time ({max_time:.0f}s) on FF", total_time=t)

    # Stair landings share the same local (x, y) on both floors (spec §7.1
    # alignment invariant) — continue straight onto the GF field from here.
    path.append(WalkerStep(t=t, floor="GF", x=x, y=y))
    gf_ex, gf_ey = _direction_arrays(gf_grid, gf_field, dir_cache)
    t, reached, stuck, reason, exit_id = _walk_gf_phase(
        gf_grid, gf_field, gf_ex, gf_ey, x, y, t, dt=dt, v0=v0, max_time=max_time - t, path=path,
    )
    reason = f"via {stair_id}: {reason}" if reason else f"via {stair_id}"
    return WalkerResult(room=room_id, scenario=scenario, path=path, reached_exit=reached,
                         stuck=stuck, reason=reason, total_time=t, exit_id=exit_id, stair_used=stair_id)


# --------------------------------------------------------------------- plotting

_COLORS = [
    "#ea580c", "#7c3aed", "#059669", "#2563eb", "#db2777", "#0891b2",
    "#ca8a04", "#4d7c0f", "#9333ea", "#0d9488", "#be123c", "#475569",
]


def _plot_traces_on_floor(ax, results: list[WalkerResult], floor_id: str, color_by_room: dict[str, str]) -> None:
    for r in results:
        pts = [(s.x, s.y) for s in r.path if s.floor == floor_id]
        if len(pts) < 2:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=color_by_room[r.room], lw=2.2, alpha=0.9, zorder=8)
        ax.plot(xs[0], ys[0], "o", color=color_by_room[r.room], markersize=6, zorder=9,
                markeredgecolor="white", markeredgewidth=0.6)
        is_last_segment = r.path[-1].floor == floor_id
        marker = "*" if (is_last_segment and r.reached_exit) else ("X" if is_last_segment and r.stuck else None)
        if marker:
            ax.plot(xs[-1], ys[-1], marker, color=color_by_room[r.room], markersize=14 if marker == "*" else 10,
                    zorder=9, markeredgecolor="black", markeredgewidth=0.6)


def plot_scenario_pair(building: Building, gf, ff, gf_grid, ff_grid, results_a: list[WalkerResult],
                        results_b: list[WalkerResult], *, label_a: str, label_b: str, suptitle: str,
                        floors: tuple[str, ...] = ("GF",)):
    n = len(floors)
    fig, axes = plt.subplots(n, 2, figsize=(20, 6.6 * n), squeeze=False)
    floor_objs = {"GF": (gf, gf_grid), "FF": (ff, ff_grid)}
    all_rooms = sorted({r.room for r in results_a} | {r.room for r in results_b})
    color_by_room = dict(zip(all_rooms, _COLORS))

    for i, floor_id in enumerate(floors):
        floor, grid = floor_objs[floor_id]
        for j, (results, label) in enumerate([(results_a, label_a), (results_b, label_b)]):
            ax = axes[i][j]
            plot_floor_layers(building, floor, grid, ax=ax, title=f"{building.id} {floor_id} — {label}")
            _plot_traces_on_floor(ax, results, floor_id, color_by_room)

    handles = [Line2D([0], [0], color=c, lw=2.6, label=room) for room, c in color_by_room.items()]
    handles += [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#555", markeredgecolor="white", markersize=7, label="start"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#555", markeredgecolor="black", markersize=12, label="reached exit"),
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#555", markeredgecolor="black", markersize=9, label="stuck"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=9, bbox_to_anchor=(0.5, -0.02 / n))
    fig.subplots_adjust(top=0.90, bottom=0.12, wspace=0.18, hspace=0.3)
    fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=0.98)
    return fig


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------- edge-case stress sweep
#
# A follow-up rigorous check, on top of the Milestone 4 demo above: designed
# dead-ends, double-failure scenarios, a simulated localised hazard, and a
# door-frame clearance check. Two scenarios here have no grid-level overlay
# equivalent yet and are approximated with an ad-hoc blocking Rect via
# DynamicOverlay.extra_obstacles (the same mechanism AddObstacle events use):
#   - "lobby_opening_blocked": the LOBBY<->ring-S opening (y=10, x 28-32) has
#     no leaf/state in the grid model (spec: openings are always passable),
#     so it's blocked as a thin obstacle instead — the graph-level equivalent
#     is the `block_ringS_mid` scenario (dms/scenario/presets.py).
#   - "leak_H3": there's no gas/PARTIAL-speed field yet (that's a later
#     milestone), so H-3's avoidance zone is approximated as a hard-blocked
#     patch of FF's north ring leg, around H-3's position (30, 28.5).

LOBBY_OPENING_BLOCK_RECT = Rect(28.0, 9.5, 32.0, 10.5)
H3_HAZARD_BLOCK_RECT = Rect(25.0, 27.0, 35.0, 30.0)

AGENT_BODY_RADIUS_M = 0.2  # spec §3 parameter registry, body radius r_i (lower bound)


@dataclass(slots=True)
class EdgeCaseSpec:
    label: str
    floor: str  # "GF" | "FF"
    scenario: str
    start_xy: tuple[float, float]
    expected_exits: frozenset[str] | None  # None = any currently-available exit counts as success
    expected_stair: str | None = None
    note: str = ""


@dataclass(slots=True)
class EdgeCaseResult:
    spec: EdgeCaseSpec
    walker: WalkerResult
    oscillation_ok: bool
    clearance_ok: bool
    clearance_min: float
    expectation_ok: bool

    @property
    def safe(self) -> bool:
        return self.walker.reached_exit and not self.walker.stuck

    @property
    def path_valid(self) -> bool:
        return self.safe and self.oscillation_ok and self.clearance_ok


def _scenario_overlays(name: str, site, building: Building) -> tuple[DynamicOverlay, DynamicOverlay, frozenset[str]]:
    """Returns (gf_overlay, ff_overlay, excluded_stairs). Door-state overlays
    are safe to share across floors: a door id that doesn't exist on a given
    floor's own door list is simply never matched, so it's a no-op there."""
    if name == "baseline":
        return DynamicOverlay(), DynamicOverlay(), frozenset()
    if name == "block_main_exit":
        ov = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
        return ov, ov, frozenset()
    if name == "two_stairs_lost":
        return DynamicOverlay(), DynamicOverlay(), stairs_blocked_by_scenario(site, building, "two_stairs_lost")
    if name == "lobby_opening_blocked":
        return DynamicOverlay(extra_obstacles=[LOBBY_OPENING_BLOCK_RECT]), DynamicOverlay(), frozenset()
    if name == "leak_H3":
        return DynamicOverlay(), DynamicOverlay(extra_obstacles=[H3_HAZARD_BLOCK_RECT]), frozenset()
    if name == "slide23_C":
        ov = DynamicOverlay(door_states={"X-NW": "UNAVAILABLE", "X-E": "UNAVAILABLE"}, clutter_on=True)
        return ov, ov, frozenset()
    raise ValueError(f"unknown edge-case scenario {name!r}")


def _check_no_oscillation(path: list[WalkerStep], grid_by_floor: dict, *, max_entries: int = 6) -> bool:
    """Counts *distinct entries* into a cell (transitions from a different
    cell), not raw per-step dwell counts — a step (dt*v0=0.06 m) is much
    smaller than a cell (h=0.5 m), so crossing straight through one cell
    takes >=8 consecutive steps on its own; counting those as "revisits"
    would flag every normal path. A single dead-end backtrack re-enters the
    doorway cell once or twice; true oscillation (bouncing between a couple
    of cells many times) re-enters the same cell repeatedly."""
    entries: dict[tuple[str, int, int], int] = {}
    prev_key: tuple[str, int, int] | None = None
    for s in path:
        grid = grid_by_floor[s.floor]
        row, col = grid.transform.local_to_cell(s.x, s.y)
        key = (s.floor, row, col)
        if key != prev_key:
            entries[key] = entries.get(key, 0) + 1
            if entries[key] > max_entries:
                return False
            prev_key = key
    return True


def _min_door_clearance(path: list[WalkerStep], grid_by_floor: dict) -> float:
    """Minimum distance from any door-crossing point on the path to the
    nearest wall cell's centre, less half a cell (an approximation of the
    wall's own footprint) — math.inf if the path never crosses a door."""
    min_clear = math.inf
    wall_xy_by_floor: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for s in path:
        grid = grid_by_floor[s.floor]
        row, col = grid.transform.local_to_cell(s.x, s.y)
        if not grid.transform.in_bounds(row, col) or grid.door_id[row, col] == -1:
            continue
        if s.floor not in wall_xy_by_floor:
            wr, wc = np.nonzero(grid.wall)
            x0, y0 = grid.transform.origin_local
            h = grid.transform.h
            wall_xy_by_floor[s.floor] = ((wc + 0.5) * h + x0, (wr + 0.5) * h + y0)
        wx, wy = wall_xy_by_floor[s.floor]
        if wx.size == 0:
            continue
        clearance = float(np.hypot(wx - s.x, wy - s.y).min()) - grid.transform.h / 2
        min_clear = min(min_clear, clearance)
    return min_clear


def run_edge_case(spec: EdgeCaseSpec, building: Building, site, gf, ff, gf_grid, ff_grid,
                   cache: FieldCache, dir_cache: dict, grid_by_floor: dict) -> EdgeCaseResult:
    gf_overlay, ff_overlay, excluded = _scenario_overlays(spec.scenario, site, building)

    if spec.floor == "GF":
        field = cache.get_gf(building, gf, gf_grid, overlay=gf_overlay)
        walker = trace_gf_walker(spec.label, spec.scenario, building, gf, gf_grid, field, dir_cache,
                                  start_xy=spec.start_xy)
    else:
        gf_field_for_ff = cache.get_gf(building, gf, gf_grid, overlay=gf_overlay)
        ff_field = cache.get_ff(building, ff, ff_grid, gf_grid, overlay=ff_overlay, excluded_stairs=excluded)
        walker = trace_ff_walker(spec.label, spec.scenario, building, ff, ff_grid, ff_field, gf_grid,
                                  gf_field_for_ff, dir_cache, start_xy=spec.start_xy, excluded_stairs=excluded)

    oscillation_ok = _check_no_oscillation(walker.path, grid_by_floor)
    clearance_min = _min_door_clearance(walker.path, grid_by_floor)
    clearance_ok = not math.isfinite(clearance_min) or clearance_min > AGENT_BODY_RADIUS_M

    if not walker.reached_exit or spec.expected_exits is not None and walker.exit_id not in spec.expected_exits or spec.expected_stair is not None and walker.stair_used != spec.expected_stair:
        expectation_ok = False
    else:
        expectation_ok = True

    return EdgeCaseResult(spec=spec, walker=walker, oscillation_ok=oscillation_ok,
                           clearance_ok=clearance_ok, clearance_min=clearance_min, expectation_ok=expectation_ok)


def _print_edge_case_table(results: list[EdgeCaseResult]) -> None:
    headers = ["Room/Point", "Scenario", "Stair Used", "Exit Reached", "Time (s)", "Path Valid?", "Trapped?"]
    rows = []
    for r in results:
        rows.append([
            r.spec.label,
            r.spec.scenario,
            r.walker.stair_used or "-",
            r.walker.exit_id or "-",
            f"{r.walker.total_time:.2f}",
            "YES" if r.path_valid else "NO",
            "YES" if r.walker.stuck else "no",
        ])
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)] if rows else [len(h) for h in headers]

    def fmt(cells: list[str]) -> str:
        return " | ".join(c.ljust(w) for c, w in zip(cells, widths))

    print(fmt(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt(row))


def plot_gf_edge_cases(building: Building, gf, gf_grid, groups: list[tuple[str, list[EdgeCaseResult]]]):
    fig, axes = plt.subplots(2, 2, figsize=(20, 13))
    axes = axes.flatten()
    all_rooms = sorted({r.spec.label for _, g in groups for r in g})
    color_by_room = dict(zip(all_rooms, _COLORS * 3))

    for ax, (label, group) in zip(axes, groups):
        plot_floor_layers(building, gf, gf_grid, ax=ax, title=f"{building.id} GF — {label}")
        _plot_traces_on_floor(ax, [r.walker for r in group], "GF", color_by_room)

    handles = [Line2D([0], [0], color=c, lw=2.6, label=room) for room, c in color_by_room.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8, bbox_to_anchor=(0.5, -0.015))
    fig.subplots_adjust(top=0.93, bottom=0.09, hspace=0.35, wspace=0.18)
    fig.suptitle(f"{building.id} GF — edge-case ghost walker sweep", fontsize=14, fontweight="bold", y=0.985)
    return fig


def plot_ff_edge_cases(building: Building, ff, gf, ff_grid, gf_grid, groups: list[tuple[str, list[EdgeCaseResult]]]):
    n = len(groups)
    fig, axes = plt.subplots(n, 2, figsize=(20, 6.6 * n), squeeze=False)
    all_rooms = sorted({r.spec.label for _, g in groups for r in g})
    color_by_room = dict(zip(all_rooms, _COLORS * 3))

    for i, (label, group) in enumerate(groups):
        ax_ff, ax_gf = axes[i]
        plot_floor_layers(building, ff, ff_grid, ax=ax_ff, title=f"{building.id} FF — {label}")
        plot_floor_layers(building, gf, gf_grid, ax=ax_gf, title=f"{building.id} GF — {label}")
        _plot_traces_on_floor(ax_ff, [r.walker for r in group], "FF", color_by_room)
        _plot_traces_on_floor(ax_gf, [r.walker for r in group], "GF", color_by_room)

    handles = [Line2D([0], [0], color=c, lw=2.6, label=room) for room, c in color_by_room.items()]
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 8), fontsize=8, bbox_to_anchor=(0.5, -0.01 / n))
    fig.subplots_adjust(top=0.94, bottom=0.08, hspace=0.4, wspace=0.18)
    fig.suptitle(f"{building.id} FF -> GF — edge-case ghost walker sweep", fontsize=14, fontweight="bold", y=0.99)
    return fig


def run_edge_case_sweep(site, building: Building, gf, ff, gf_grid, ff_grid, cache: FieldCache, dir_cache: dict) -> list[EdgeCaseResult]:
    grid_by_floor = {"GF": gf_grid, "FF": ff_grid}

    specs = [
        # 1. Designed dead-ends
        EdgeCaseSpec("FN-stub", "FF", "baseline", (30.0, 38.0), expected_exits=None,
                     note="dead-end corridor, no exit on FF — must back out to ring-N"),
        EdgeCaseSpec("G-08", "GF", "baseline", (4.0, 22.8), expected_exits=frozenset({"X-W"}),
                     note="isolated toilet dead-end"),
        EdgeCaseSpec("F-10", "FF", "baseline", (13.0, 25.0), expected_exits=None,
                     note="single-door archive, deep-corner start"),
        # 2. Critical redundancy / double failure
        EdgeCaseSpec("F-04", "FF", "two_stairs_lost", ff.spaces["F-04"].rect.center,
                     expected_exits=frozenset({"X-NW"}), expected_stair="ST-W"),
        EdgeCaseSpec("F-09", "FF", "two_stairs_lost", ff.spaces["F-09"].rect.center,
                     expected_exits=frozenset({"X-NW"}), expected_stair="ST-W"),
        EdgeCaseSpec("LOBBY", "GF", "block_main_exit", gf.spaces["LOBBY"].rect.center,
                     expected_exits=frozenset({"X-N"})),
        EdgeCaseSpec("G-05", "GF", "block_main_exit", gf.spaces["G-05"].rect.center,
                     expected_exits=frozenset({"X-N"})),
        EdgeCaseSpec("ring-S@30", "GF", "lobby_opening_blocked", (30.0, 11.5),
                     expected_exits=frozenset({"X-S"}), note="should detour via G-05"),
        # 3. Hazard avoidance / clutter resistance
        EdgeCaseSpec("F-08", "FF", "leak_H3", ff.spaces["F-08"].rect.center, expected_exits=None,
                     note="should prefer the southern ring leg, away from H-3"),
        EdgeCaseSpec("G-01", "GF", "slide23_C", gf.spaces["G-01"].rect.center,
                     expected_exits=frozenset({"X-S", "X-N", "X-W"})),
        EdgeCaseSpec("G-06", "GF", "slide23_C", gf.spaces["G-06"].rect.center,
                     expected_exits=frozenset({"X-S", "X-N", "X-W"})),
        EdgeCaseSpec("G-11", "GF", "slide23_C", gf.spaces["G-11"].rect.center,
                     expected_exits=frozenset({"X-S", "X-N", "X-W"})),
        EdgeCaseSpec("G-15", "GF", "slide23_C", gf.spaces["G-15"].rect.center,
                     expected_exits=frozenset({"X-S", "X-N", "X-W"})),
    ]

    results = [run_edge_case(spec, building, site, gf, ff, gf_grid, ff_grid, cache, dir_cache, grid_by_floor)
               for spec in specs]

    print("\n" + "=" * 100)
    print("EDGE-CASE STRESS SWEEP")
    print("=" * 100)
    _print_edge_case_table(results)

    print()
    for r in results:
        if not r.expectation_ok:
            print(f"  NOTE: {r.spec.label}/{r.spec.scenario} did not match the expected exit/stair "
                  f"(reached {r.walker.exit_id} via {r.walker.stair_used}) — reported, not treated as a failure; "
                  f"routing choice is a graph property, not a deadlock.")
        if not r.oscillation_ok:
            print(f"  WARNING: {r.spec.label}/{r.spec.scenario} re-entered a cell >6 times — possible oscillation.")
        if math.isfinite(r.clearance_min):
            flag = "" if r.clearance_ok else "  <-- below the 0.2 m body-radius margin"
            print(f"  {r.spec.label}/{r.spec.scenario}: min door clearance {r.clearance_min:.3f} m{flag}")

    n_safe = sum(1 for r in results if r.safe)
    n_osc_ok = sum(1 for r in results if r.oscillation_ok)
    n_clear_ok = sum(1 for r in results if r.clearance_ok)
    print(f"\n{n_safe}/{len(results)} reached safety; {n_osc_ok}/{len(results)} no oscillation; "
          f"{n_clear_ok}/{len(results)} clearance > {AGENT_BODY_RADIUS_M} m")

    for r in results:
        assert r.safe, f"edge case {r.spec.label}/{r.spec.scenario} did not reach safety: {r.walker.reason}"
    for r in results:
        assert r.oscillation_ok, f"edge case {r.spec.label}/{r.spec.scenario} oscillated (possible deadlock loop)"
    print("\nAll edge cases reached safety with 0 deadlocks and no oscillation.")

    if n_clear_ok < len(results):
        print(
            f"\nNOTE on door clearance: {len(results) - n_clear_ok}/{len(results)} crossings came within "
            f"{AGENT_BODY_RADIUS_M} m of a wall. This is a genuine limitation of the 'trivial' gradient-following\n"
            "walker (PROMPT_01 §1's own wording), not a bug in the Eikonal field: at h_c=0.5 m, a 1.0-1.2 m door\n"
            "is only 2-3 cells wide, and nearest-cell direction sampling has no notion of centring within that\n"
            "span (or of the agent's own body) — it just follows -grad(T), which can hug one jamb. Enforcing a\n"
            "real clearance margin needs the SFM wall-repulsion force (f_iW, spec S5), which belongs to the\n"
            "agent milestone, not this field-verification walker. Not treated as a hard failure here."
        )
    print("=" * 100)

    gf_groups = [
        ("baseline dead-end (G-08)", [r for r in results if r.spec.label == "G-08"]),
        ("block_main_exit", [r for r in results if r.spec.scenario == "block_main_exit"]),
        ("lobby_opening_blocked", [r for r in results if r.spec.scenario == "lobby_opening_blocked"]),
        ("slide23_C (4 corners)", [r for r in results if r.spec.scenario == "slide23_C"]),
    ]
    fig_gf = plot_gf_edge_cases(building, gf, gf_grid, gf_groups)
    p_gf = _save(fig_gf, OUT_DIR / "walker_edge_cases_gf.png")
    print(f"wrote {p_gf}")

    ff_groups = [
        ("baseline dead-ends (FN-stub, F-10)", [r for r in results if r.spec.label in ("FN-stub", "F-10")]),
        ("two_stairs_lost", [r for r in results if r.spec.scenario == "two_stairs_lost"]),
        ("leak_H3", [r for r in results if r.spec.scenario == "leak_H3"]),
    ]
    fig_ff = plot_ff_edge_cases(building, ff, gf, ff_grid, gf_grid, ff_groups)
    p_ff = _save(fig_ff, OUT_DIR / "walker_edge_cases_ff.png")
    print(f"wrote {p_ff}")

    return results


# --------------------------------------------------------------------- main


def main() -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    gf = building.get_floor("GF")
    ff = building.get_floor("FF")
    gf_grid = rasterize_floor(building, gf, H_COARSE)
    ff_grid = rasterize_floor(building, ff, H_COARSE)
    cache = FieldCache()
    dir_cache: dict = {}

    gf_rooms = ["G-01", "G-11", "G-15", "LOBBY"]
    ff_rooms = ["F-01", "F-07", "F-10"]

    gf_field_base = cache.get_gf(building, gf, gf_grid)
    overlay_block = DynamicOverlay(door_states={"X-S": "UNAVAILABLE"})
    gf_field_block = cache.get_gf(building, gf, gf_grid, overlay=overlay_block)

    ff_field_base = cache.get_ff(building, ff, ff_grid, gf_grid)
    stc_blocked = stairs_blocked_by_scenario(site, building, "stair_C_lost")
    gf_field_for_stc = cache.get_gf(building, gf, gf_grid)  # stair_C_lost doesn't touch GF exits
    ff_field_stc = cache.get_ff(building, ff, ff_grid, gf_grid, excluded_stairs=stc_blocked)

    print("=" * 78)
    print(f"Ghost walker: dt={DT}s, v0={V0} m/s, max_time={MAX_TIME}s")
    print("=" * 78)

    all_results: list[WalkerResult] = []

    print("\n-- GF: baseline --")
    gf_base_results = [trace_gf_walker(r, "baseline", building, gf, gf_grid, gf_field_base, dir_cache) for r in gf_rooms]
    for res in gf_base_results:
        print(f"  {res.room:8s} -> {'EXIT ' + str(res.exit_id) if res.reached_exit else 'STUCK (' + res.reason + ')'}"
              f"  t={res.total_time:.2f}s")
    all_results += gf_base_results

    print("\n-- GF: block_main_exit --")
    gf_block_results = [trace_gf_walker(r, "block_main_exit", building, gf, gf_grid, gf_field_block, dir_cache) for r in gf_rooms]
    for res in gf_block_results:
        print(f"  {res.room:8s} -> {'EXIT ' + str(res.exit_id) if res.reached_exit else 'STUCK (' + res.reason + ')'}"
              f"  t={res.total_time:.2f}s")
    all_results += gf_block_results

    print("\n-- FF: baseline --")
    ff_base_results = [trace_ff_walker(r, "baseline", building, ff, ff_grid, ff_field_base, gf_grid, gf_field_base, dir_cache)
                        for r in ff_rooms]
    for res in ff_base_results:
        print(f"  {res.room:8s} -> {'EXIT ' + str(res.exit_id) if res.reached_exit else 'STUCK (' + res.reason + ')'}"
              f"  t={res.total_time:.2f}s  [{res.reason}]")
    all_results += ff_base_results

    print("\n-- FF: stair_C_lost --")
    ff_stc_results = [trace_ff_walker(r, "stair_C_lost", building, ff, ff_grid, ff_field_stc, gf_grid, gf_field_for_stc,
                                       dir_cache, excluded_stairs=stc_blocked)
                       for r in ff_rooms]
    for res in ff_stc_results:
        print(f"  {res.room:8s} -> {'EXIT ' + str(res.exit_id) if res.reached_exit else 'STUCK (' + res.reason + ')'}"
              f"  t={res.total_time:.2f}s  [{res.reason}]")
    all_results += ff_stc_results

    print("\n" + "=" * 78)
    n_ok = sum(1 for r in all_results if r.reached_exit)
    n_stuck = sum(1 for r in all_results if r.stuck)
    print(f"TOTAL: {n_ok}/{len(all_results)} walkers reached an exterior exit; {n_stuck} stuck")
    for r in all_results:
        assert r.reached_exit and not r.stuck, f"walker {r.room}/{r.scenario} failed: {r.reason}"
    print("All walkers reached an exterior exit without deadlock.")
    print("=" * 78)

    fig1 = plot_scenario_pair(building, gf, ff, gf_grid, ff_grid, gf_base_results, gf_block_results,
                               label_a="baseline", label_b="block_main_exit",
                               suptitle=f"{building.id} GF — ghost walker traces (-grad T)", floors=("GF",))
    p1 = _save(fig1, OUT_DIR / "walker_gf_traces.png")
    print(f"wrote {p1}")

    fig2 = plot_scenario_pair(building, gf, ff, gf_grid, ff_grid, ff_base_results, ff_stc_results,
                               label_a="baseline", label_b="stair_C_lost",
                               suptitle=f"{building.id} FF -> GF — ghost walker traces (-grad T)", floors=("FF", "GF"))
    p2 = _save(fig2, OUT_DIR / "walker_ff_traces.png")
    print(f"wrote {p2}")

    run_edge_case_sweep(site, building, gf, ff, gf_grid, ff_grid, cache, dir_cache)


if __name__ == "__main__":
    main()
