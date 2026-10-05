"""Vector geometry primitives shared by the building/site model.

All coordinates are in metres, in the frame of whatever owns the object
(L1 building-local, or L3 site-global — spec SYSTEM_SPEC.md §1.1). Nothing
in this module knows about grids or cell resolutions; rasterisation is a
later milestone.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.geometry import Polygon, box

AXES = ("x", "y")
STATES = ("AVAILABLE", "PARTIAL", "UNAVAILABLE")


@dataclass(frozen=True, slots=True)
class Rect:
    """Axis-aligned rectangle, x1 > x0 and y1 > y0."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        if not (self.x1 > self.x0 and self.y1 > self.y0):
            raise ValueError(f"degenerate rect: {self}")

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0

    def contains_point(self, x: float, y: float, *, eps: float = 1e-9) -> bool:
        """Half-open containment [x0,x1) x [y0,y1): shared boundaries never double-assign."""
        return (self.x0 - eps <= x < self.x1 - eps) and (self.y0 - eps <= y < self.y1 - eps)

    def overlap_area(self, other: Rect) -> float:
        ox0, oy0 = max(self.x0, other.x0), max(self.y0, other.y0)
        ox1, oy1 = min(self.x1, other.x1), min(self.y1, other.y1)
        if ox1 <= ox0 or oy1 <= oy0:
            return 0.0
        return (ox1 - ox0) * (oy1 - oy0)

    def overlaps(self, other: Rect, *, tol: float = 1e-6) -> bool:
        """True if the two rects share positive interior area (touching edges don't count)."""
        return self.overlap_area(other) > tol

    def shared_edge(
        self, other: Rect, *, tol: float = 1e-6
    ) -> tuple[str, float, float, float] | None:
        """
        If `self` and `other` are edge-adjacent (touch along a line, no interior
        overlap), return (axis, coord, lo, hi) describing the shared segment in
        the *fixed-coordinate-on-wall* convention used by Door/Opening ("x=C"
        or "y=C" spanning [lo, hi]). Returns None if they don't touch cleanly.
        """
        if self.overlaps(other, tol=tol):
            return None

        # Vertical shared edge: self.x1 == other.x0 (or vice versa), y-ranges overlap.
        for a, b in ((self, other), (other, self)):
            if abs(a.x1 - b.x0) <= tol:
                lo, hi = max(a.y0, b.y0), min(a.y1, b.y1)
                if hi - lo > tol:
                    return "x", a.x1, lo, hi
        # Horizontal shared edge: self.y1 == other.y0 (or vice versa), x-ranges overlap.
        for a, b in ((self, other), (other, self)):
            if abs(a.y1 - b.y0) <= tol:
                lo, hi = max(a.x0, b.x0), min(a.x1, b.x1)
                if hi - lo > tol:
                    return "y", a.y1, lo, hi
        return None

    def edge_segment(self, axis: str, coord: float, *, tol: float = 1e-6) -> tuple[float, float] | None:
        """If this rect has a boundary edge at axis=coord, return its (lo, hi) span."""
        if axis == "x" and (abs(self.x0 - coord) <= tol or abs(self.x1 - coord) <= tol):
            return self.y0, self.y1
        if axis == "y" and (abs(self.y0 - coord) <= tol or abs(self.y1 - coord) <= tol):
            return self.x0, self.x1
        return None

    def to_polygon(self) -> Polygon:
        return box(self.x0, self.y0, self.x1, self.y1)


def _axis_span(wall: str) -> tuple[str, float]:
    """Parse a 'x=C' / 'y=C' wall spec into (axis, coord)."""
    axis, _, coord = wall.partition("=")
    axis = axis.strip()
    if axis not in AXES:
        raise ValueError(f"bad wall spec {wall!r}: axis must be 'x' or 'y'")
    return axis, float(coord)


@dataclass(slots=True)
class BoundarySpan:
    """Shared fields for any wall-line crossing (door / opening / exit)."""

    id: str
    floor: str
    axis: str
    coord: float
    lo: float
    hi: float

    @property
    def width_m(self) -> float:
        return self.hi - self.lo

    @classmethod
    def from_wall(cls, id: str, floor: str, wall: str, lo: float, hi: float, **extra):
        axis, coord = _axis_span(wall)
        return cls(id=id, floor=floor, axis=axis, coord=coord, lo=lo, hi=hi, **extra)  # type: ignore[call-arg]


@dataclass(slots=True)
class Door:
    """A leafed crossing: interior door, fire door, service door, or exit."""

    id: str
    floor: str
    axis: str
    coord: float
    lo: float
    hi: float
    kind: str = "interior"  # interior | exit | fire_door | service_door
    default_state: str = "AVAILABLE"  # AVAILABLE | PARTIAL | UNAVAILABLE
    leaf: str = "open"  # "open" | "closed" — gas posture, independent of agent passability
    gas_perm_closed: float = 0.05
    gas_perm_open: float = 1.0
    space_a: str | None = None
    space_b: str | None = None  # "outside" for exits

    def __post_init__(self) -> None:
        if self.default_state not in STATES:
            raise ValueError(f"door {self.id}: bad state {self.default_state!r}")

    @property
    def width_m(self) -> float:
        return self.hi - self.lo

    @property
    def is_exterior(self) -> bool:
        return self.space_b == "outside" or self.kind == "exit"


@dataclass(slots=True)
class Opening:
    """A doorless, always-passable, gas-open crossing (ring/spine joins, stub mouths)."""

    id: str
    floor: str
    axis: str
    coord: float
    lo: float
    hi: float
    space_a: str | None = None
    space_b: str | None = None

    @property
    def width_m(self) -> float:
        return self.hi - self.lo


@dataclass(slots=True)
class Window:
    id: str
    floor: str
    space_id: str
    axis: str
    coord: float
    lo: float
    hi: float
    sill_height_m: float = 0.9
    is_open: bool = False
    perm_closed: float = 0.02
    perm_open: float = 0.5

    @property
    def width_m(self) -> float:
        return self.hi - self.lo


@dataclass(slots=True)
class Obstacle:
    id: str
    floor: str
    rect: Rect
    space_id: str | None = None
    layer: str = "furniture"  # furniture | clutter | parking
    active_by_default: bool = True
    narrows_to_width_m: float | None = None  # for clutter slots that partially block a corridor


@dataclass(slots=True)
class HazardSource:
    id: str
    species: str = "H2S"
    mw: float = 34.08
    building: str | None = None
    floor: str | None = None
    space_id: str | None = None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    Q_gps: float | None = None
    H_m: float | None = None
    u_mps: float | None = None
    purpose: str = ""


@dataclass(slots=True)
class Stair:
    """
    Vertical link between a GF landing and an FF landing (spec §3.4, P0 model).
    Riser/run/slope/length are *derived*, never typed in.
    """

    id: str
    gf_space: str
    ff_space: str
    width_m: float
    riser_m: float
    tread_m: float
    floor_to_floor_m: float
    enclosed: bool
    footprint_gf: Rect
    footprint_ff: Rect
    flights: int = 2
    chi: float = 0.70
    fire_door_id: str | None = None
    discharge_exit_id: str | None = None
    permeability_open: float = 1.0
    permeability_enclosed_closed: float = 0.05

    @property
    def n_risers(self) -> int:
        return round(self.floor_to_floor_m / self.riser_m)

    @property
    def risers_per_flight(self) -> tuple[int, ...]:
        n = self.n_risers
        base = n // self.flights
        rem = n - base * self.flights
        return tuple(base + (1 if i < rem else 0) for i in range(self.flights))

    @property
    def rise_per_flight(self) -> tuple[float, ...]:
        return tuple(r * self.riser_m for r in self.risers_per_flight)

    @property
    def run_per_flight(self) -> tuple[float, ...]:
        # Treads = risers - 1: the top riser lands on the floor/landing, no tread beyond it.
        return tuple(max(r - 1, 0) * self.tread_m for r in self.risers_per_flight)

    @property
    def slope_deg(self) -> float:
        rise = self.rise_per_flight[0]
        run = self.run_per_flight[0]
        return math.degrees(math.atan2(rise, run)) if run > 0 else 90.0

    @property
    def landing_length_m(self) -> float:
        return self.width_m

    @property
    def walking_length_m(self) -> float:
        flights_len = sum(
            math.hypot(run, rise) for run, rise in zip(self.run_per_flight, self.rise_per_flight)
        )
        n_landings = self.flights - 1
        return flights_len + n_landings * self.landing_length_m
