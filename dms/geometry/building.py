"""Building / Floor / Space containers — the compiled vector model of one building."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .primitives import (
    Door,
    HazardSource,
    Obstacle,
    Opening,
    Rect,
    Refuge,
    Riser,
    Sign,
    Stair,
    Window,
)

SPACE_KINDS = ("room", "corridor", "stub", "stair", "void")


@dataclass(slots=True)
class Space:
    id: str
    floor: str
    name: str
    kind: str  # room | corridor | stub | stair | void
    rect: Rect
    notes: str = ""
    use: str | None = None  # occupancy-profile key (spec §6), e.g. "office", "plant_room"
    restricted: bool = False  # operator-only room (spec §7)

    def __post_init__(self) -> None:
        if self.kind not in SPACE_KINDS:
            raise ValueError(f"space {self.id}: bad kind {self.kind!r}")

    @property
    def walkable(self) -> bool:
        return self.kind != "void"


@dataclass(slots=True)
class Floor:
    id: str  # "GF" | "FF"
    z: float
    footprint: Rect
    spaces: dict[str, Space] = field(default_factory=dict)
    doors: dict[str, Door] = field(default_factory=dict)
    openings: dict[str, Opening] = field(default_factory=dict)
    windows: dict[str, Window] = field(default_factory=dict)
    obstacles: dict[str, Obstacle] = field(default_factory=dict)
    hazards: dict[str, HazardSource] = field(default_factory=dict)
    refuges: dict[str, Refuge] = field(default_factory=dict)
    signs: dict[str, Sign] = field(default_factory=dict)

    def get_space(self, space_id: str) -> Space:
        return self.spaces[space_id]

    def space_at(self, x: float, y: float) -> Space | None:
        for sp in self.spaces.values():
            if sp.rect.contains_point(x, y):
                return sp
        return None

    @property
    def exit_doors(self) -> list[Door]:
        return [d for d in self.doors.values() if d.kind == "exit"]

    @property
    def fire_doors(self) -> list[Door]:
        return [d for d in self.doors.values() if d.kind == "fire_door"]


@dataclass(slots=True)
class Building:
    id: str
    origin: tuple[float, float]
    rotation_deg: float
    wall_thickness_m: float
    floors: dict[str, Floor] = field(default_factory=dict)
    stairs: dict[str, Stair] = field(default_factory=dict)
    designed_dead_ends: list[str] = field(default_factory=list)
    exit_sets: dict[str, list[str] | None] = field(default_factory=dict)
    obstacle_types: dict[str, tuple[float, float]] = field(default_factory=dict)  # type -> (width_m, height_m), §4.1
    blocking_groups: list[str] = field(default_factory=list)  # groups allowed to fully block a corridor, §4.3
    risers: list[Riser] = field(default_factory=list)  # explicit vertical gas couplings, §7
    occupancy_profiles: dict[str, dict] = field(default_factory=dict)  # use -> {day|lunch|meeting|night: {...}}, §6
    persona_mix: dict[str, dict] = field(default_factory=dict)  # persona id -> {"share": float, "v0": float|None}, spec §6.1/§2
    open_between_kinds: list[tuple[str, str]] = field(default_factory=list)  # kind pairs with no wall between them, spec §2

    def floor_ids(self) -> list[str]:
        return list(self.floors.keys())

    def get_floor(self, floor_id: str) -> Floor:
        return self.floors[floor_id]

    def wall_segments(self, floor: Floor, overlay=None):
        """(M, 4) float array of (x1,y1,x2,y2) wall segments for SFM wall
        repulsion (spec §5.3). Deferred import: the real implementation
        lives in dms/grid/raster.py (it needs DynamicOverlay), and `grid`
        already imports `geometry` at module level, so importing the other
        way at module scope here would be circular -- this local import is
        the standard way out, and it's cheap (runs once per call, not once
        per cell)."""
        from ..grid.raster import wall_segments as _wall_segments

        return _wall_segments(self, floor, overlay)

    def to_site_xy(self, local_x: float, local_y: float) -> tuple[float, float]:
        """Spec §1.1: [X,Y] = R(psi_b) p + O_b."""
        psi = math.radians(self.rotation_deg)
        c, s = math.cos(psi), math.sin(psi)
        ox, oy = self.origin
        X = c * local_x - s * local_y + ox
        Y = s * local_x + c * local_y + oy
        return X, Y

    def from_site_xy(self, X: float, Y: float) -> tuple[float, float]:
        """Inverse of to_site_xy: R(psi) is orthogonal, so R^-1 = R(-psi)."""
        psi = math.radians(self.rotation_deg)
        c, s = math.cos(psi), math.sin(psi)
        ox, oy = self.origin
        dx, dy = X - ox, Y - oy
        local_x = c * dx + s * dy
        local_y = -s * dx + c * dy
        return local_x, local_y
