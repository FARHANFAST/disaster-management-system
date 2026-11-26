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
    diameter_m: float = 15.0  # sized up by the loader to fit the full design
    # population at 0.5 m2/person if the declared value is too small -- see apply_ap_sizing()


@dataclass(slots=True)
class SiteObstacle:
    id: str
    rect: Rect
    layer: str = "static"  # static | parking | structure
    active_by_default: bool = True
    type: str | None = None  # structure label (pipe_rack | tank | vehicle_area | building), display only


@dataclass(slots=True)
class Walkway:
    """A paved path, polyline in site (L3) metres --
    cheaper to traverse than open ground (dms/fields/outdoor.py), not a
    walkability restriction (open ground stays walkable, just slower/less
    preferred, matching real pedestrian behaviour)."""

    id: str
    points: tuple[tuple[float, float], ...]
    width_m: float = 3.0


@dataclass(slots=True)
class Fence:
    """A perimeter barrier, polyline in site (L3)
    metres -- blocks outdoor movement except through its own `Gate`s."""

    id: str
    points: tuple[tuple[float, float], ...]


@dataclass(slots=True)
class Gate:
    id: str
    fence_id: str
    x: float
    y: float
    width_m: float = 4.0
    default_state: str = "AVAILABLE"  # AVAILABLE | UNAVAILABLE


@dataclass(slots=True)
class WeatherStation:
    """A single site marker (display only -- no wind
    model exists yet) with an optional fixed wind-from
    direction, rendered via dms/viz/plot.py's existing wind_arrow()."""

    id: str
    x: float
    y: float
    wind_from_deg: float | None = None


@dataclass(slots=True)
class Site:
    name: str
    width_m: float
    height_m: float
    buildings: dict[str, Building] = field(default_factory=dict)
    assembly_points: dict[str, AssemblyPoint] = field(default_factory=dict)
    obstacles: dict[str, SiteObstacle] = field(default_factory=dict)
    hazards: dict[str, HazardSource] = field(default_factory=dict)
    walkways: dict[str, Walkway] = field(default_factory=dict)
    fences: dict[str, Fence] = field(default_factory=dict)
    gates: dict[str, Gate] = field(default_factory=dict)
    weather_stations: dict[str, WeatherStation] = field(default_factory=dict)
    scenarios: dict[str, dict] = field(default_factory=dict)

    @property
    def footprint(self) -> Rect:
        return Rect(0.0, 0.0, self.width_m, self.height_m)

    def building_ids(self) -> list[str]:
        return list(self.buildings.keys())

    def get_building(self, building_id: str) -> Building:
        return self.buildings[building_id]
