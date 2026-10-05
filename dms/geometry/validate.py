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
    for stair in building.stairs.values():
        gf, ff = stair.footprint_gf, stair.footprint_ff
        if not (
            abs(gf.x0 - ff.x0) <= LEN_TOL
            and abs(gf.y0 - ff.y0) <= LEN_TOL
            and abs(gf.x1 - ff.x1) <= LEN_TOL
            and abs(gf.y1 - ff.y1) <= LEN_TOL
        ):
            report.add(
                "error", "STAIR_FOOTPRINT_MISALIGNED",
                f"stair {stair.id}: GF footprint {gf} != FF footprint {ff}",
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
    return report
