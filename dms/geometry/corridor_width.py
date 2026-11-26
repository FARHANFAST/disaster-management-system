"""Geometric corridor clear-width (spec §4) -- shared by the graph
builder (Link.width_m/Q_max) and the validator (clear-width report + the
"no default-on obstacle below 1.0 m" rule), so the two can never disagree.

Computed from each active obstacle's already-rotation-resolved AABB
projected onto the corridor's short axis, not from a declared scalar: a
mid-corridor obstacle leaves two lanes (sum them), an obstacle against one
wall leaves one.
"""

from __future__ import annotations

from .primitives import Obstacle


def lanes_for_segment(
    short_lo: float, short_hi: float, long_lo: float, long_hi: float, axis: str,
    obstacles: list[Obstacle],
) -> list[float]:
    """Every candidate lane's width across one corridor segment, including
    ones too narrow to count as usable (the caller sums only lanes
    `>= min_lane_width_m` -- that
    threshold is config data, not a module constant here, see
    dms/scenario/agent_defaults.py -- for the effective width; the full
    breakdown is kept for the clear-width report's diagnostics).

    `narrows_to_width_m`, when an obstacle declares it, overrides that one
    obstacle's own geometric contribution with a carved range of that width
    anchored at whichever wall its centre is nearer -- a deliberate
    approximation for obstacles whose AABB doesn't capture the true pinch
    point (e.g. a staggered slalom of several pieces).
    """
    carved: list[tuple[float, float]] = []
    for obs in obstacles:
        r = obs.rect
        if axis == "x":
            o_long_lo, o_long_hi, o_short_lo, o_short_hi = r.x0, r.x1, r.y0, r.y1
        else:
            o_long_lo, o_long_hi, o_short_lo, o_short_hi = r.y0, r.y1, r.x0, r.x1
        if o_long_hi <= long_lo or o_long_lo >= long_hi:
            continue  # doesn't overlap this segment's span at all

        if obs.narrows_to_width_m is not None:
            full = short_hi - short_lo
            carve_width = max(0.0, full - obs.narrows_to_width_m)
            centre = (o_short_lo + o_short_hi) / 2.0
            if centre - short_lo <= short_hi - centre:
                lo, hi = short_lo, short_lo + carve_width  # nearer the low wall
            else:
                lo, hi = short_hi - carve_width, short_hi  # nearer the high wall
        else:
            lo, hi = max(o_short_lo, short_lo), min(o_short_hi, short_hi)
        if hi > lo:
            carved.append((lo, hi))

    carved.sort()
    merged: list[tuple[float, float]] = []
    for lo, hi in carved:
        if merged and lo <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))

    lanes: list[float] = []
    cursor = short_lo
    for lo, hi in merged:
        if lo > cursor:
            lanes.append(lo - cursor)
        cursor = max(cursor, hi)
    if cursor < short_hi:
        lanes.append(short_hi - cursor)
    return lanes


def effective_width(lanes: list[float], min_lane_width_m: float) -> float:
    return sum(w for w in lanes if w >= min_lane_width_m)
