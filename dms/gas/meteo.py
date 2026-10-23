"""Meteorology for the gas model (Gas Dispersion design doc, "Inputs"):
wind speed and direction, temperature, pressure, precipitation, humidity and
Pasquill–Gifford stability class, either constant or scripted over time.

Wind direction is the meteorological convention: the compass bearing the
wind blows FROM (0 = from north, 90 = from east). The site frame is x east,
y north, so a wind from 135° (SE) carries gas toward the north-west.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

STABILITY_CLASSES = ("A", "B", "C", "D", "E", "F")
INSOLATION = ("strong", "moderate", "slight", "night_cloudy", "night_clear")

# SYSTEM_SPEC.md §2.3 stability table. Mixed cells ("A–B") take the more
# stable class: it gives the higher ground-level concentration, which is the
# conservative side for an alarm/evacuation model.
_STABILITY_TABLE = (  # (u upper bound m/s, classes by insolation column)
    (2.0, ("A", "B", "B", "F", "F")),
    (3.0, ("B", "B", "C", "E", "F")),
    (4.0, ("B", "C", "C", "D", "E")),
    (6.0, ("C", "D", "D", "D", "D")),
    (math.inf, ("C", "D", "D", "D", "D")),
)


def stability_class(wind_speed: float, insolation: str) -> str:
    col = INSOLATION.index(insolation)
    for u_max, classes in _STABILITY_TABLE:
        if wind_speed < u_max:
            return classes[col]
    raise AssertionError("unreachable")


@dataclass(frozen=True, slots=True)
class Weather:
    wind_speed: float = 2.5  # m/s
    wind_from_deg: float = 270.0  # compass, direction the wind blows FROM
    temperature_c: float = 25.0
    pressure_pa: float = 101_325.0
    precipitation_mmh: float = 0.0
    humidity_pct: float = 60.0
    stability: str = "D"

    def __post_init__(self) -> None:
        if self.stability not in STABILITY_CLASSES:
            raise ValueError(f"bad stability class {self.stability!r}")
        if self.wind_speed < 0:
            raise ValueError("wind speed must be >= 0")

    @property
    def wind_vector(self) -> tuple[float, float]:
        """(vx, vy) in m/s in the site frame — where the air (and gas) goes."""
        a = math.radians(self.wind_from_deg)
        return -self.wind_speed * math.sin(a), -self.wind_speed * math.cos(a)


def _lerp_angle(a: float, b: float, f: float) -> float:
    d = (b - a + 180.0) % 360.0 - 180.0  # shortest arc, so 350 -> 10 veers through north
    return (a + f * d) % 360.0


class WeatherSchedule:
    """Weather keyframes at scenario times. Continuous quantities (wind speed
    and direction, temperature, pressure, rain, humidity) are interpolated
    linearly between keyframes — a wind shift is a gradual veer, not a jump.
    Stability class is categorical and switches at its keyframe."""

    def __init__(self, keyframes: list[tuple[float, Weather]]):
        if not keyframes:
            raise ValueError("need at least one weather keyframe")
        self.keyframes = sorted(keyframes, key=lambda kv: kv[0])

    @classmethod
    def constant(cls, weather: Weather) -> WeatherSchedule:
        return cls([(0.0, weather)])

    @classmethod
    def from_config(cls, items: list[dict]) -> WeatherSchedule:
        """`[{t: 0, wind_speed: 2.5, wind_from_deg: 135, stability: D}, ...]`.
        Fields left out of a keyframe carry over from the previous one;
        `stability: auto` derives the class from wind speed and `insolation`."""
        frames: list[tuple[float, Weather]] = []
        prev = Weather()
        for item in sorted(items, key=lambda d: float(d.get("t", 0.0))):
            item = dict(item)
            t = float(item.pop("t", 0.0))
            insolation = item.pop("insolation", None)
            if item.get("stability") == "auto":
                speed = float(item.get("wind_speed", prev.wind_speed))
                if insolation is None:
                    raise ValueError("stability: auto needs an insolation category")
                item["stability"] = stability_class(speed, insolation)
            prev = replace(prev, **{k: (v if k == "stability" else float(v)) for k, v in item.items()})
            frames.append((t, prev))
        return cls(frames)

    def at(self, t: float) -> Weather:
        frames = self.keyframes
        if t <= frames[0][0]:
            return frames[0][1]
        for (t0, w0), (t1, w1) in zip(frames, frames[1:]):
            if t < t1:
                f = (t - t0) / (t1 - t0)
                lerp = lambda a, b: a + f * (b - a)  # noqa: E731
                return Weather(
                    wind_speed=lerp(w0.wind_speed, w1.wind_speed),
                    wind_from_deg=_lerp_angle(w0.wind_from_deg, w1.wind_from_deg, f),
                    temperature_c=lerp(w0.temperature_c, w1.temperature_c),
                    pressure_pa=lerp(w0.pressure_pa, w1.pressure_pa),
                    precipitation_mmh=lerp(w0.precipitation_mmh, w1.precipitation_mmh),
                    humidity_pct=lerp(w0.humidity_pct, w1.humidity_pct),
                    stability=w0.stability,
                )
        return frames[-1][1]
