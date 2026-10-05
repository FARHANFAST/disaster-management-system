"""Outdoor L3 site: the buildings, assembly points and outdoor obstacles that place them."""

from __future__ import annotations

from dataclasses import dataclass, field

from .building import Building
from .primitives import HazardSource, Rect


@dataclass(slots=True)
class AssemblyPoint:
    id: str
    x: float
    y: float
    diameter_m: float = 15.0


@dataclass(slots=True)
class SiteObstacle:
    id: str
    rect: Rect
    layer: str = "static"  # static | parking
    active_by_default: bool = True


@dataclass(slots=True)
class Site:
    name: str
    width_m: float
    height_m: float
    buildings: dict[str, Building] = field(default_factory=dict)
    assembly_points: dict[str, AssemblyPoint] = field(default_factory=dict)
    obstacles: dict[str, SiteObstacle] = field(default_factory=dict)
    hazards: dict[str, HazardSource] = field(default_factory=dict)
    scenarios: dict[str, dict] = field(default_factory=dict)

    @property
    def footprint(self) -> Rect:
        return Rect(0.0, 0.0, self.width_m, self.height_m)

    def building_ids(self) -> list[str]:
        return list(self.buildings.keys())

    def get_building(self, building_id: str) -> Building:
        return self.buildings[building_id]
