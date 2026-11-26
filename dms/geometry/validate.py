"""Geometric validation of a loaded Site (spec §7.1), scoped to Milestone 1:

- no two spaces overlap on a floor;
- every door/opening sits on exactly one shared boundary between two spaces,
  or between a space and outside (independently recomputed from the Space
  rects, not trusted from the loader's own bookkeeping);
- stair footprints coincide between GF and FF;
- envelope watertight: spaces exactly tile each floor's footprint (no gaps,
  no double coverage), and declared exits/windows on the exterior envelope
  don't overlap each other. This is the vector-geometry stand-in for the
  grid-based flood-fill test; the real flood fill arrives with the rasteriser.

Nothing here mutates the model or "fixes" bad data — it only reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, pairwise

from shapely.ops import unary_union

from .building import Building, Floor
from .site import Site

AREA_TOL = 1e-3  # m^2
LEN_TOL = 1e-6  # m


@dataclass(slots=True)
class ValidationIssue:
    severity: str  # "error" | "warning"
    code: str
    message: str
    building: str | None = None
    floor: str | None = None


@dataclass(slots=True)
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, *, building: str | None = None, floor: str | None = None) -> None:
        self.issues.append(ValidationIssue(severity, code, message, building, floor))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def to_markdown(self) -> str:
        lines = ["# Validation report", ""]
        lines.append(
            f"**{'OK' if self.ok else 'FAILED'}** — {len(self.errors)} error(s), {len(self.warnings)} warning(s)."
        )
        lines.append("")
        for i in self.issues:
            loc = " ".join(p for p in (i.building, i.floor) if p)
            lines.append(f"- **{i.severity.upper()}** [{i.code}] {loc}: {i.message}")
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- 1. overlap

def check_no_overlaps(building: Building, floor: Floor, report: ValidationReport) -> None:
    for a, b in combinations(floor.spaces.values(), 2):
        area = a.rect.overlap_area(b.rect)
        if area > AREA_TOL:
            report.add(
                "error", "SPACE_OVERLAP",
                f"{a.id} and {b.id} overlap by {area:.3f} m^2",
                building=building.id, floor=floor.id,
            )


# --------------------------------------------------------------------------- 2. doors

def _neighbors(floor: Floor, axis: str, coord: float, lo: float, hi: float, *, tol: float = LEN_TOL) -> list[str]:
    hits = []
    for sid, sp in floor.spaces.items():
        seg = sp.rect.edge_segment(axis, coord, tol=tol)
        if seg is None:
            continue
        slo, shi = seg
        if slo - tol <= lo and hi <= shi + tol:
            hits.append(sid)
    return hits


def check_doors_on_shared_boundary(building: Building, floor: Floor, report: ValidationReport) -> None:
    fp = floor.footprint
    for d in floor.doors.values():
        hits = _neighbors(floor, d.axis, d.coord, d.lo, d.hi)
        expected_interior = {d.space_a}
        if d.space_b != "outside":
            expected_interior.add(d.space_b)

        if d.space_b == "outside":
            on_envelope = (
                d.axis == "x" and (abs(d.coord - fp.x0) <= LEN_TOL or abs(d.coord - fp.x1) <= LEN_TOL)
            ) or (
                d.axis == "y" and (abs(d.coord - fp.y0) <= LEN_TOL or abs(d.coord - fp.y1) <= LEN_TOL)
            )
            if not on_envelope:
                report.add(
                    "error", "EXIT_NOT_ON_ENVELOPE",
                    f"door {d.id} claims an exterior exit but {d.axis}={d.coord} is not on the footprint boundary",
                    building=building.id, floor=floor.id,
                )
            if set(hits) != {d.space_a}:
                report.add(
                    "error", "DOOR_BOUNDARY_MISMATCH",
                    f"door {d.id}: recomputed interior neighbours {hits} != expected {{{d.space_a}}}",
                    building=building.id, floor=floor.id,
                )
        else:
            if set(hits) != expected_interior:
                report.add(
                    "error", "DOOR_BOUNDARY_MISMATCH",
                    f"door {d.id}: recomputed neighbours {hits} != expected {sorted(expected_interior)}",
                    building=building.id, floor=floor.id,
                )

    for o in floor.openings.values():
        hits = _neighbors(floor, o.axis, o.coord, o.lo, o.hi)
        expected = {o.space_a, o.space_b}
        if set(hits) != expected:
            report.add(
                "error", "OPENING_BOUNDARY_MISMATCH",
                f"opening {o.id}: recomputed neighbours {hits} != expected {sorted(expected)}",
                building=building.id, floor=floor.id,
            )


def check_exits_on_envelope(building: Building, floor: Floor, report: ValidationReport) -> None:
    fp = floor.footprint
    for d in floor.exit_doors:
        on_envelope = (
            d.axis == "x" and (abs(d.coord - fp.x0) <= LEN_TOL or abs(d.coord - fp.x1) <= LEN_TOL)
        ) or (
            d.axis == "y" and (abs(d.coord - fp.y0) <= LEN_TOL or abs(d.coord - fp.y1) <= LEN_TOL)
        )
        if not on_envelope:
            report.add(
                "error", "EXIT_NOT_ON_ENVELOPE",
                f"exit {d.id} at {d.axis}={d.coord} is not on floor {floor.id}'s exterior envelope",
                building=building.id, floor=floor.id,
            )


# --------------------------------------------------------------------------- 3. stairs

def check_stair_alignment(building: Building, report: ValidationReport) -> None:
    """Every floor a stair serves must share the same landing footprint (one
    vertical shaft) -- generalized from the original GF==FF-only check
    (spec §8) to compare every floor pairwise against the first."""
    for stair in building.stairs.values():
        floor_ids = list(stair.footprints)
        if not floor_ids:
            continue
        ref_id, ref = floor_ids[0], stair.footprints[floor_ids[0]]
        for floor_id in floor_ids[1:]:
            fp = stair.footprints[floor_id]
            if not (
                abs(fp.x0 - ref.x0) <= LEN_TOL and abs(fp.y0 - ref.y0) <= LEN_TOL
                and abs(fp.x1 - ref.x1) <= LEN_TOL and abs(fp.y1 - ref.y1) <= LEN_TOL
            ):
                report.add(
                    "error", "STAIR_FOOTPRINT_MISALIGNED",
                    f"stair {stair.id}: {floor_id} footprint {fp} != {ref_id} footprint {ref}",
                    building=building.id,
                )


# --------------------------------------------------------------------------- 4. watertight

def check_tiling(building: Building, floor: Floor, report: ValidationReport) -> None:
    polys = [sp.rect.to_polygon() for sp in floor.spaces.values()]
    union = unary_union(polys)
    footprint_poly = floor.footprint.to_polygon()
    gap_or_overshoot = footprint_poly.symmetric_difference(union).area
    if gap_or_overshoot > AREA_TOL:
        report.add(
            "error", "FLOOR_NOT_TILED",
            f"spaces leave {gap_or_overshoot:.3f} m^2 of gap/overshoot vs the {floor.footprint.area:.1f} m^2 footprint",
            building=building.id, floor=floor.id,
        )


def check_watertight_envelope(building: Building, floor: Floor, report: ValidationReport) -> None:
    """Spaces must tile the footprint, and every exit/window on a given wall
    line must not overlap another — otherwise the envelope has an ambiguous
    or double-counted breach."""
    check_tiling(building, floor, report)

    breaks: dict[tuple[str, float], list[tuple[float, float, str]]] = {}
    for d in floor.exit_doors:
        breaks.setdefault((d.axis, d.coord), []).append((d.lo, d.hi, d.id))
    for w in floor.windows.values():
        breaks.setdefault((w.axis, w.coord), []).append((w.lo, w.hi, w.id))

    for (axis, coord), spans in breaks.items():
        spans.sort()
        for (lo1, hi1, id1), (lo2, hi2, id2) in pairwise(spans):
            if lo2 < hi1 - LEN_TOL:
                report.add(
                    "error", "ENVELOPE_BREACH_OVERLAP",
                    f"{id1} and {id2} overlap on {axis}={coord} ([{lo1},{hi1}] vs [{lo2},{hi2}])",
                    building=building.id, floor=floor.id,
                )


# --------------------------------------------------------------------------- driver

def validate_site(site: Site) -> ValidationReport:
    report = ValidationReport()
    for building in site.buildings.values():
        for floor in building.floors.values():
            check_no_overlaps(building, floor, report)
            check_doors_on_shared_boundary(building, floor, report)
            check_exits_on_envelope(building, floor, report)
            check_watertight_envelope(building, floor, report)
        check_stair_alignment(building, report)
        check_corridor_clear_width(building, report)
    return report


# --------------------------------------------------------------------------- 4. corridor clear width


def _default_active_obstacles(floor: Floor, space_id: str) -> list:
    return [
        obs for obs in floor.obstacles.values()
        if obs.space_id == space_id and (obs.layer == "furniture" or obs.active_by_default)
    ]


def _iter_corridor_segments(building: Building):
    """Yields (floor, space, axis, short_lo, short_hi, seg_lo, seg_hi) for
    every corridor/stub segment, reusing the graph builder's own
    segmentation (dms/graph/builder.py) rather than a second copy of it --
    deferred import since `graph` depends on `geometry`, not the reverse."""
    from ..graph.builder import _corridor_cut_points, _long_axis, _segment_lengths

    for floor in building.floors.values():
        for space in floor.spaces.values():
            if space.kind not in ("corridor", "stub"):
                continue
            axis = _long_axis(space)
            cuts = _corridor_cut_points(floor, space, axis)
            if axis == "x":
                short_lo, short_hi = space.rect.y0, space.rect.y1
            else:
                short_lo, short_hi = space.rect.x0, space.rect.x1
            for lo, hi in _segment_lengths(cuts):
                yield floor, space, axis, short_lo, short_hi, lo, hi


def check_corridor_clear_width(
    building: Building, report: ValidationReport, *, min_lane_width_m: float | None = None,
) -> None:
    """spec §4.3: no *default-on* obstacle may reduce a corridor
    segment's clear width below the minimum lane width (config).
    Uses the same lanes_for_segment geometry
    dms/graph/builder.py uses for Link.width_m, so this check and the live
    routing capacity can never disagree."""
    from ..scenario.agent_defaults import load_agent_defaults
    from .corridor_width import effective_width, lanes_for_segment

    if min_lane_width_m is None:
        min_lane_width_m = load_agent_defaults().min_lane_width_m

    for floor, space, axis, short_lo, short_hi, lo, hi in _iter_corridor_segments(building):
        obstacles = _default_active_obstacles(floor, space.id)
        clear = effective_width(lanes_for_segment(short_lo, short_hi, lo, hi, axis, obstacles), min_lane_width_m)
        if clear < min_lane_width_m:
            report.add(
                "error", "CORRIDOR_CLEAR_WIDTH_BELOW_MIN",
                f"{space.id} segment [{lo:.1f},{hi:.1f}]: clear width {clear:.2f} m "
                f"< {min_lane_width_m} m from default-on obstacles",
                building=building.id, floor=floor.id,
            )


def write_clear_width_report(building: Building, out_path, *, min_lane_width_m: float | None = None):
    """Markdown report of every corridor segment's clear-width lane
    breakdown (spec §4.3), including segments that
    pass, for visibility into how close each one runs to the configured
    minimum."""
    from pathlib import Path

    from ..scenario.agent_defaults import load_agent_defaults
    from .corridor_width import effective_width, lanes_for_segment

    if min_lane_width_m is None:
        min_lane_width_m = load_agent_defaults().min_lane_width_m

    lines = ["# Corridor clear-width report", "", f"Building: **{building.id}**", "",
             "| Floor | Corridor | Segment | Raw width (m) | Effective width (m) | Lanes (m) |",
             "|---|---|---|---|---|---|"]
    for floor, space, axis, short_lo, short_hi, lo, hi in _iter_corridor_segments(building):
        obstacles = _default_active_obstacles(floor, space.id)
        lanes = lanes_for_segment(short_lo, short_hi, lo, hi, axis, obstacles)
        clear = effective_width(lanes, min_lane_width_m)
        raw = short_hi - short_lo
        flag = "" if clear >= min_lane_width_m else "  <-- BELOW MIN"
        lane_str = ", ".join(f"{w:.2f}" for w in lanes) or "-"
        lines.append(f"| {floor.id} | {space.id} | [{lo:.1f},{hi:.1f}] | {raw:.2f} | {clear:.2f}{flag} | {lane_str} |")

    text = "\n".join(lines) + "\n"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return out_path
