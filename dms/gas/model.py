"""The gas dispersion model (Gas Dispersion design doc): outdoor Gaussian
puffs + one indoor box per zone, detectors, the alarm, and the 5-minute
forecast — behind one `step(dt, map_state)`.

Per simulated second (doc, "Integration" steps 1-3 and 6):
  1. meteorology updates from the scenario schedule;
  2. puffs are emitted and advected, the outdoor field is recomputed and every
     zone's box is updated from the outdoor air at its own location;
  3. detectors are read and the alarm state updated;
  6. the forecast refreshes every `forecast_every_s`, and early (at most every
     `forecast_min_gap_s`) when the wind turns or changes speed — a persistence
     forecast made before a wind shift says nothing useful after it.
Agents read the most recent values between gas steps (`node_ppm`,
`alarm.state`, `forecast`) and write back only through MapVentilationState.

Outputs (all ppm at the 1.5 m breathing height):
  field_ppm     outdoor field on the site grid (`field_h` cells)
  point_ppm()   outdoor point query
  zone_ppm      indoor concentration per zone (one per room / corridor segment)
  node_ppm      per navigation-graph node (zones, doors, exits — see zones.py;
                an exit's value is the worst air on its egress path)
  alarm         NORMAL / WARNING / EMERGENCY
  forecast      projected node concentrations and field maximum over the next
                `forecast_horizon_s`, plus time-to-hazard per node, assuming the
                current weather persists (a forecaster does not know the script)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.site import Site
from .alarm import AlarmConfig, AlarmSystem, Detector
from .dispersion import MW_H2S, RECEPTOR_Z, PuffModel, ReleaseSource, mg_m3_to_ppm
from .indoor import BoxModel, MapVentilationState, Ventilation, zone_ach
from .meteo import Weather, WeatherSchedule
from .zones import NodeZones


@dataclass(slots=True)
class GasConfig:
    mw: float = MW_H2S
    terrain: str = "urban"
    puff_interval_s: float = 1.0
    washout_a: float = 1e-4  # Λ = a · R^b  [1/s, R in mm/h]
    washout_b: float = 0.8
    field_h: float = 2.0  # outdoor grid cell, m (the doc's 5 m is for a 1 km site; this one is 160 m)
    ventilation: Ventilation = field(default_factory=Ventilation)
    alarm: AlarmConfig = field(default_factory=AlarmConfig)
    forecast_horizon_s: float = 300.0
    forecast_every_s: float = 30.0
    forecast_field_every_s: float = 10.0
    forecast_min_gap_s: float = 5.0  # early refresh on a weather change, but no more often than this
    forecast_refresh_turn_deg: float = 10.0
    forecast_refresh_speed: float = 0.5  # m/s
    hazard_ppm: float = 10.0  # time-to-hazard threshold


@dataclass(slots=True)
class Forecast:
    t0: float
    dt: float
    node_ppm: np.ndarray  # (T, n_nodes) at t0 + dt * (i + 1)
    field_max: np.ndarray  # (rows, cols) max over the horizon
    time_to_hazard: np.ndarray  # per node, seconds after t0 (inf: not within the horizon)

    @property
    def horizon_s(self) -> float:
        return self.dt * self.node_ppm.shape[0]

    def node_max(self, k: int, t_from: float, t_to: float) -> float:
        """Max forecast ppm on node k over absolute times [t_from, t_to],
        clamped to the horizon (beyond it, the last frame stands in)."""
        T = self.node_ppm.shape[0]
        i0 = int(np.clip(np.floor((t_from - self.t0) / self.dt) - 1, 0, T - 1))
        i1 = int(np.clip(np.ceil((t_to - self.t0) / self.dt), i0 + 1, T))
        return float(self.node_ppm[i0:i1, k].max())


class GasModel:
    def __init__(self, site: Site, zones: NodeZones, sources: list[ReleaseSource], weather: WeatherSchedule,
                 detectors: list[Detector], cfg: GasConfig | None = None):
        self.cfg = cfg = cfg or GasConfig()
        self.zones = zones
        self.schedule = weather
        fp = site.footprint
        self.puffs = PuffModel(sources, domain=(fp.x0, fp.y0, fp.x1, fp.y1), terrain=cfg.terrain,
                               puff_interval_s=cfg.puff_interval_s, washout_a=cfg.washout_a,
                               washout_b=cfg.washout_b)
        self.boxes = BoxModel(zones.n_zones)
        self.t = 0.0
        self.weather: Weather = weather.at(0.0)

        h = cfg.field_h
        self.field_shape = (round((fp.y1 - fp.y0) / h), round((fp.x1 - fp.x0) / h))
        self.field_origin = (fp.x0, fp.y0)
        rows, cols = self.field_shape
        gx, gy = np.meshgrid((np.arange(cols) + 0.5) * h + fp.x0, (np.arange(rows) + 0.5) * h + fp.y0)
        self._grid_x, self._grid_y = gx.ravel(), gy.ravel()

        self.detectors = detectors
        self.alarm = AlarmSystem(detectors, cfg.alarm)
        outdoor = [d for d in detectors if d.node is None]
        self._det_outdoor = np.array([i for i, d in enumerate(detectors) if d.node is None], dtype=np.int64)
        self._det_indoor = np.array([i for i, d in enumerate(detectors) if d.node is not None], dtype=np.int64)
        self._det_indoor_node = np.array([d.node for d in detectors if d.node is not None], dtype=np.int64)

        # every outdoor receptor the model needs each step, in one batch:
        # zone receptors | exit egress-path samples | outdoor detectors
        nz, ne = zones.n_zones, zones.exit_nodes.size
        egress = np.vstack(zones.egress_xy) if ne else np.empty((0, 2))
        self._egress_starts = np.cumsum([0] + [len(p) for p in zones.egress_xy[:-1]]).astype(np.int64)
        self._pts_x = np.concatenate([zones.zone_xy[:, 0], egress[:, 0], [d.x for d in outdoor]])
        self._pts_y = np.concatenate([zones.zone_xy[:, 1], egress[:, 1], [d.y for d in outdoor]])
        self._pts_z = np.concatenate([zones.zone_z, np.full(len(egress), RECEPTOR_Z), [d.z for d in outdoor]])
        self._split = (nz, nz + len(egress))

        self.zone_out_ppm = np.zeros(nz)
        self.exit_ppm = np.zeros(ne)  # worst air on each exit's egress path
        self.node_ppm = np.zeros(zones.node_kind.size)
        self.detector_ppm = np.zeros(len(detectors))
        self.field_ppm = np.zeros(self.field_shape)
        self.forecast: Forecast | None = None
        self._forecast_weather: Weather | None = None
        self._next_forecast = 0.0

    # ------------------------------------------------------------- queries

    def ppm(self, c_mg_m3):
        return mg_m3_to_ppm(c_mg_m3, self.cfg.mw, self.weather.temperature_c, self.weather.pressure_pa)

    def point_ppm(self, x, y, z=RECEPTOR_Z) -> np.ndarray:
        return self.ppm(self.puffs.concentration_mg_m3(x, y, z))

    @property
    def zone_ppm(self) -> np.ndarray:
        return self.boxes.c

    # ------------------------------------------------------------- stepping

    def _outdoor_receptors(self, puffs: PuffModel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = self.ppm(puffs.concentration_mg_m3(self._pts_x, self._pts_y, self._pts_z))
        a, b = self._split
        exit_ppm = np.maximum.reduceat(c[a:b], self._egress_starts) if b > a else c[a:b]
        return c[:a], exit_ppm, c[b:]

    def step(self, dt: float, map_state: MapVentilationState, *, compute_field: bool = True) -> None:
        self.weather = self.schedule.at(self.t)
        self.puffs.step(self.t, dt, self.weather)
        self.t += dt

        self.zone_out_ppm, self.exit_ppm, det_out = self._outdoor_receptors(self.puffs)
        ach = zone_ach(self.zones.n_zones, self.cfg.ventilation, map_state, self.zones.zones_by_exit)
        self.boxes.step(dt, self.zone_out_ppm, ach)
        self.node_ppm = self.zones.node_ppm(self.boxes.c, self.exit_ppm)
        if compute_field:
            self.field_ppm = self.point_ppm(self._grid_x, self._grid_y).reshape(self.field_shape)

        readings = np.zeros(len(self.detectors))
        readings[self._det_outdoor] = det_out
        readings[self._det_indoor] = self.node_ppm[self._det_indoor_node]
        self.detector_ppm = readings
        self.alarm.update(self.t, dt, readings)

        if self.t >= self._next_forecast - 1e-9 or self._weather_changed():
            self.forecast = self.run_forecast(map_state)
            self._forecast_weather = self.weather
            self._next_forecast = self.t + self.cfg.forecast_every_s

    def _weather_changed(self) -> bool:
        """A persistence forecast is stale once the wind it assumed has moved
        on: refresh early (at most every forecast_min_gap_s) on a shift."""
        w0, w, c = self._forecast_weather, self.weather, self.cfg
        if w0 is None or self.forecast is None or self.t - self.forecast.t0 < c.forecast_min_gap_s - 1e-9:
            return False
        turn = abs((w.wind_from_deg - w0.wind_from_deg + 180.0) % 360.0 - 180.0)
        return (turn >= c.forecast_refresh_turn_deg or abs(w.wind_speed - w0.wind_speed) >= c.forecast_refresh_speed
                or w.stability != w0.stability)

    def run_forecast(self, map_state: MapVentilationState, *, dt: float = 1.0) -> Forecast:
        """Project the current state forward under persistent weather and the
        current ventilation, on copies — the live model is untouched."""
        cfg = self.cfg
        puffs, boxes = self.puffs.copy(), self.boxes.copy()
        ach = zone_ach(self.zones.n_zones, cfg.ventilation, map_state, self.zones.zones_by_exit)
        n = int(round(cfg.forecast_horizon_s / dt))
        field_every = max(1, int(round(cfg.forecast_field_every_s / dt)))
        node_ppm = np.zeros((n, self.zones.node_kind.size))
        field_max = self.field_ppm.copy()
        t = self.t
        for i in range(n):
            puffs.step(t, dt, self.weather)
            t += dt
            zone_out, exit_ppm, _ = self._outdoor_receptors(puffs)
            boxes.step(dt, zone_out, ach)
            node_ppm[i] = self.zones.node_ppm(boxes.c, exit_ppm)
            if (i + 1) % field_every == 0:
                f = self.ppm(puffs.concentration_mg_m3(self._grid_x, self._grid_y)).reshape(self.field_shape)
                np.maximum(field_max, f, out=field_max)
        hit = np.vstack([self.node_ppm[None], node_ppm]) >= cfg.hazard_ppm
        tth = np.where(hit.any(axis=0), hit.argmax(axis=0) * dt, np.inf)
        return Forecast(t0=self.t, dt=dt, node_ppm=node_ppm, field_max=field_max, time_to_hazard=tth)
