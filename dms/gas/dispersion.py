"""Outdoor H2S dispersion (Gas Dispersion design doc, "Outdoor: Gaussian
puffs"; SYSTEM_SPEC.md §2.3).

A continuous release is cut into puffs emitted every `puff_interval_s`; an
instantaneous release is one puff. Each puff advects with the *current* wind
and grows with its own travel distance, so the plume bends when the wind
shifts. Concentration at a receptor is the sum over puffs (D2 with σx = σy,
ground reflection):

    C = Σ M_i / ((2π)^1.5 σy² σz) · exp(-r_i² / 2σy²)
          · [exp(-(z-H)² / 2σz²) + exp(-(z+H)² / 2σz²)]

σ follows Pasquill–Gifford classes with Briggs' parameterisation (urban by
default for a plant). When the stability class changes mid-flight a puff
keeps its current size and continues growing along the new class's curve
from the *virtual distance* at which that curve has that size, so σ never
shrinks. Wet deposition removes mass at Λ = a·R^b per second while it rains.

`steady_plume_mg_m3` is the steady-state plume (D1), kept as the fast mode
and as the verification check: under constant wind the puff sum converges to
it (tests/test_gas.py).

Units: mass in mg, concentration in mg/m³; `mg_m3_to_ppm` converts with the
ambient temperature and pressure.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

import numpy as np

from .meteo import Weather

RECEPTOR_Z = 1.5  # m, breathing zone
U_MIN = 1.0  # m/s — Gaussian models are invalid in calm air (spec D1 validity u >= 1)
R_GAS = 8.314462618  # J/(mol K)
MW_H2S = 34.08


def mg_m3_to_ppm(c_mg_m3, mw: float = MW_H2S, temperature_c: float = 25.0, pressure_pa: float = 101_325.0):
    """ppm(v/v) = C · R T / (P · MW) · 1e3; 24.45/MW at 25 °C, 1 atm."""
    return c_mg_m3 * (R_GAS * (temperature_c + 273.15) / (pressure_pa * mw)) * 1e3


# ------------------------------------------------------------------ Briggs σ(x)

def _rural_sy(a):
    return lambda x: a * x * (1 + 1e-4 * x) ** -0.5


def _urban_sy(a):
    return lambda x: a * x * (1 + 4e-4 * x) ** -0.5


_SIGMA = {
    "rural": {
        "A": (_rural_sy(0.22), lambda x: 0.20 * x),
        "B": (_rural_sy(0.16), lambda x: 0.12 * x),
        "C": (_rural_sy(0.11), lambda x: 0.08 * x * (1 + 2e-4 * x) ** -0.5),
        "D": (_rural_sy(0.08), lambda x: 0.06 * x * (1 + 1.5e-3 * x) ** -0.5),
        "E": (_rural_sy(0.06), lambda x: 0.03 * x / (1 + 3e-4 * x)),
        "F": (_rural_sy(0.04), lambda x: 0.016 * x / (1 + 3e-4 * x)),
    },
    "urban": {},
}
_URBAN_AB = (_urban_sy(0.32), lambda x: 0.24 * x * (1 + 1e-3 * x) ** 0.5)
_URBAN_EF = (_urban_sy(0.11), lambda x: 0.08 * x * (1 + 1.5e-3 * x) ** -0.5)
_SIGMA["urban"] = {
    "A": _URBAN_AB, "B": _URBAN_AB,
    "C": (_urban_sy(0.22), lambda x: 0.20 * x),
    "D": (_urban_sy(0.16), lambda x: 0.14 * x * (1 + 3e-4 * x) ** -0.5),
    "E": _URBAN_EF, "F": _URBAN_EF,
}
X_MIN = 1.0  # m — spec: x <- max(x, 1)


def sigma_y(x, stability: str, terrain: str = "urban"):
    return _SIGMA[terrain][stability][0](np.maximum(x, X_MIN))


def sigma_z(x, stability: str, terrain: str = "urban"):
    return _SIGMA[terrain][stability][1](np.maximum(x, X_MIN))


_X_TABLE = np.geomspace(X_MIN, 1e5, 4000)
_INVERSE: dict[tuple[str, str, int], np.ndarray] = {}


def virtual_distance(sigma, stability: str, terrain: str, axis: int):
    """Travel distance at which the class's σ curve reaches `sigma` (axis 0 =
    σy, 1 = σz). Every Briggs curve is increasing in x, so a table lookup
    inverts it."""
    key = (terrain, stability, axis)
    if key not in _INVERSE:
        _INVERSE[key] = _SIGMA[terrain][stability][axis](_X_TABLE)
    return np.interp(sigma, _INVERSE[key], _X_TABLE)


# ------------------------------------------------------------------ sources


@dataclass(slots=True)
class ReleaseSource:
    id: str
    x: float  # site metres
    y: float
    height_m: float = 1.0
    kind: str = "continuous"  # continuous (pipe leak) | instantaneous (tank rupture)
    rate_gps: float = 0.0  # continuous: g/s
    mass_kg: float = 0.0  # instantaneous: kg
    start_s: float = 0.0
    duration_s: float = math.inf  # continuous only
    sigma0_m: float = 1.0  # initial puff size (source extent)
    _next_emit: float = field(default=math.nan, repr=False)

    def __post_init__(self) -> None:
        if self.kind not in ("continuous", "instantaneous"):
            raise ValueError(f"source {self.id}: bad kind {self.kind!r}")
        self._next_emit = self.start_s

    def active(self, t: float) -> bool:
        return self.kind == "continuous" and self.start_s <= t < self.start_s + self.duration_s


def steady_plume_mg_m3(px, py, source: ReleaseSource, weather: Weather, *, z: float = RECEPTOR_Z,
                       terrain: str = "urban"):
    """Steady-state Gaussian plume (spec D1) for a continuous source, at site
    points (px, py). Zero upwind of the source. The source's initial size
    `sigma0_m` enters as a virtual point source upwind, exactly as it does
    for puffs, so the two modes describe the same release."""
    u = max(weather.wind_speed, U_MIN)
    a = math.radians(weather.wind_from_deg)
    ex, ey = -math.sin(a), -math.cos(a)  # downwind unit vector
    dx, dy = np.asarray(px, float) - source.x, np.asarray(py, float) - source.y
    xd = dx * ex + dy * ey  # downwind distance
    yc = -dx * ey + dy * ex  # crosswind distance
    xv_y = virtual_distance(source.sigma0_m, weather.stability, terrain, 0)
    xv_z = virtual_distance(source.sigma0_m, weather.stability, terrain, 1)
    sy = sigma_y(np.maximum(xd, 0) + xv_y, weather.stability, terrain)
    sz = sigma_z(np.maximum(xd, 0) + xv_z, weather.stability, terrain)
    H = source.height_m
    q = source.rate_gps * 1e3
    c = q / (2 * math.pi * u * sy * sz) * np.exp(-yc**2 / (2 * sy**2)) * (
        np.exp(-(z - H) ** 2 / (2 * sz**2)) + np.exp(-(z + H) ** 2 / (2 * sz**2)))
    return np.where(xd > 0, c, 0.0)


# ------------------------------------------------------------------ puffs


class PuffModel:
    """All puffs in flight. `domain` = (x0, y0, x1, y1) of the area anyone
    cares about; puffs that can no longer put more than `drop_mg_m3` into it
    are discarded."""

    def __init__(self, sources: list[ReleaseSource], *, domain: tuple[float, float, float, float],
                 terrain: str = "urban", puff_interval_s: float = 1.0, washout_a: float = 1e-4,
                 washout_b: float = 0.8, drop_mg_m3: float = 1e-3):
        self.sources = sources
        self.domain = domain
        self.terrain = terrain
        self.puff_interval_s = puff_interval_s
        self.washout_a = washout_a
        self.washout_b = washout_b
        self.drop_mg_m3 = drop_mg_m3
        self.x = np.empty(0)
        self.y = np.empty(0)
        self.H = np.empty(0)
        self.mass = np.empty(0)  # mg
        self.sy = np.empty(0)
        self.sz = np.empty(0)
        self.released_mg = 0.0
        self.deposited_mg = 0.0
        self.dropped_mg = 0.0

    @property
    def n_puffs(self) -> int:
        return self.x.size

    @property
    def airborne_mg(self) -> float:
        return float(self.mass.sum())

    def copy(self) -> PuffModel:
        return copy.deepcopy(self)

    def _emit(self, t0: float, t1: float, weather: Weather) -> None:
        """Puffs whose emission time falls in [t0, t1): each is created at the
        source and pre-advected/pre-grown by its age at t1, so a continuous
        release lays down evenly spaced puffs whatever the step length."""
        new: list[tuple[float, float, float, float, float, float]] = []
        u = max(weather.wind_speed, U_MIN)
        vx, vy = weather.wind_vector
        for src in self.sources:
            if src.kind == "instantaneous":
                if t0 <= src.start_s < t1 and src._next_emit == src.start_s:
                    new.append(self._new_puff(src, src.mass_kg * 1e6, t1 - src.start_s, u, vx, vy, weather))
                    src._next_emit = math.inf
                continue
            end = src.start_s + src.duration_s
            while src._next_emit < min(t1, end):
                te = src._next_emit
                dt_e = min(self.puff_interval_s, end - te)
                new.append(self._new_puff(src, src.rate_gps * 1e3 * dt_e, t1 - te, u, vx, vy, weather))
                src._next_emit = te + self.puff_interval_s
        if not new:
            return
        arr = np.array(new)
        self.x = np.concatenate([self.x, arr[:, 0]])
        self.y = np.concatenate([self.y, arr[:, 1]])
        self.H = np.concatenate([self.H, arr[:, 2]])
        self.mass = np.concatenate([self.mass, arr[:, 3]])
        self.sy = np.concatenate([self.sy, arr[:, 4]])
        self.sz = np.concatenate([self.sz, arr[:, 5]])
        self.released_mg += float(arr[:, 3].sum())

    def _new_puff(self, src: ReleaseSource, mass_mg: float, age: float, u: float, vx: float, vy: float,
                  weather: Weather) -> tuple[float, float, float, float, float, float]:
        cls, ter = weather.stability, self.terrain
        xv_y = virtual_distance(src.sigma0_m, cls, ter, 0)
        xv_z = virtual_distance(src.sigma0_m, cls, ter, 1)
        d = u * age  # growth distance, same rule as step()
        return (src.x + vx * age, src.y + vy * age, src.height_m, mass_mg,
                float(sigma_y(xv_y + d, cls, ter)), float(sigma_z(xv_z + d, cls, ter)))

    def step(self, t: float, dt: float, weather: Weather) -> None:
        """Advance puffs from t to t + dt under `weather` (held for the step),
        then emit the puffs released during the step."""
        if self.n_puffs:
            cls, ter = weather.stability, self.terrain
            vx, vy = weather.wind_vector
            # Advection uses the real wind; growth uses at least U_MIN so a
            # calm still dilutes (the Gaussian model's own validity floor).
            self.x = self.x + vx * dt
            self.y = self.y + vy * dt
            d = max(weather.wind_speed, U_MIN) * dt
            self.sy = sigma_y(virtual_distance(self.sy, cls, ter, 0) + d, cls, ter)
            self.sz = sigma_z(virtual_distance(self.sz, cls, ter, 1) + d, cls, ter)
            if weather.precipitation_mmh > 0:
                lam = self.washout_a * weather.precipitation_mmh ** self.washout_b
                kept = self.mass * math.exp(-lam * dt)
                self.deposited_mg += float((self.mass - kept).sum())
                self.mass = kept
            self._drop_irrelevant()
        self._emit(t, t + dt, weather)

    def _drop_irrelevant(self) -> None:
        x0, y0, x1, y1 = self.domain
        gap = np.hypot(np.maximum(0, np.maximum(x0 - self.x, self.x - x1)),
                       np.maximum(0, np.maximum(y0 - self.y, self.y - y1)))
        peak = 2 * self.mass / ((2 * math.pi) ** 1.5 * self.sy**2 * self.sz)
        keep = peak * np.exp(-gap**2 / (2 * self.sy**2)) >= self.drop_mg_m3
        if not keep.all():
            self.dropped_mg += float(self.mass[~keep].sum())
            for name in ("x", "y", "H", "mass", "sy", "sz"):
                setattr(self, name, getattr(self, name)[keep])

    def concentration_mg_m3(self, px, py, pz=RECEPTOR_Z, *, chunk: int = 4096) -> np.ndarray:
        """Sum over puffs at receptor points. `pz` may be a scalar or per point."""
        px = np.asarray(px, float).ravel()
        py = np.asarray(py, float).ravel()
        pz = np.broadcast_to(np.asarray(pz, float), px.shape)
        out = np.zeros(px.shape)
        if not self.n_puffs:
            return out
        norm = self.mass / ((2 * math.pi) ** 1.5 * self.sy**2 * self.sz)
        two_sy2, two_sz2 = 2 * self.sy**2, 2 * self.sz**2
        for i in range(0, px.size, chunk):
            sl = slice(i, i + chunk)
            r2 = (px[sl, None] - self.x) ** 2 + (py[sl, None] - self.y) ** 2
            z = pz[sl, None]
            vert = np.exp(-(z - self.H) ** 2 / two_sz2) + np.exp(-(z + self.H) ** 2 / two_sz2)
            out[sl] = (norm * np.exp(-r2 / two_sy2) * vert).sum(axis=1)
        return out
