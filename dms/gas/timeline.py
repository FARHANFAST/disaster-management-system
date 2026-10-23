"""A scenario's gas history, precomputed once so a learner can replay it
thousands of times (dms/rl/gas_dqn.py).

The gas model is run for the scenario's duration with the facility response
it scripts (HVAC off on EMERGENCY) and recorded every gas step:

  node_ppm[i]   concentration on every GraphEvacEnv node at t = i · dt
  soon_ppm[i]   worst of now and the forecast the facility held at t, over
                the next `soon_s` — what an agent reading the forecast expects
  forecasts     the Forecast objects themselves (for the gas-aware A* planner)

Agents do not write back to the map here (no exit doors held open, no rooms
sealed): one evacuee's door changes infiltration only marginally, and a fixed
history is what makes the replay cheap. dms/gas/simulate.py keeps the full
two-way coupling. Queries past the end hold the last frame.

`wind_offset_deg` rotates every wind keyframe of the scenario, giving
variants of the same leak for training a policy that reads the gas rather
than memorising one plume.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..rl.graph_env import GraphEvacEnv
from .alarm import EMERGENCY
from .indoor import MapVentilationState
from .meteo import WeatherSchedule
from .model import Forecast, GasModel
from .scenario import Scenario
from .zones import NodeZones


@dataclass(slots=True)
class GasTimeline:
    dt: float
    node_ppm: np.ndarray  # (T + 1, n_nodes), row 0 is t = 0
    soon_ppm: np.ndarray  # (T + 1, n_nodes)
    forecast_t0: np.ndarray  # issue time of each forecast
    forecasts: list[Forecast]
    alarm_t: float  # EMERGENCY time (the scenario's end if it never latches)
    zones: NodeZones
    wind_offset_deg: float = 0.0

    @property
    def duration_s(self) -> float:
        return self.dt * (self.node_ppm.shape[0] - 1)

    def _i(self, t: float) -> int:
        return int(np.clip(np.floor(t / self.dt + 1e-9), 0, self.node_ppm.shape[0] - 1))

    def now(self, t: float) -> np.ndarray:
        return self.node_ppm[self._i(t)]

    def soon(self, t: float) -> np.ndarray:
        return self.soon_ppm[self._i(t)]

    def forecast_at(self, t: float) -> Forecast | None:
        j = int(np.searchsorted(self.forecast_t0, t + 1e-9, side="right")) - 1
        return self.forecasts[j] if j >= 0 else None

    @classmethod
    def from_scenario(cls, sc: Scenario, env: GraphEvacEnv, *, wind_offset_deg: float = 0.0,
                      soon_s: float = 60.0, zones: NodeZones | None = None) -> GasTimeline:
        building = sc.site.get_building(sc.building_id)
        zones = zones or NodeZones.from_graph_env(env, sc.site, building, ceiling_m=sc.ceiling_m)
        weather = sc.weather
        if wind_offset_deg:
            frames = copy.deepcopy(sc.raw["gas"]["weather"])
            for f in frames:
                if "wind_from_deg" in f:
                    f["wind_from_deg"] = (float(f["wind_from_deg"]) + wind_offset_deg) % 360.0
            weather = WeatherSchedule.from_config(frames)
        gas = GasModel(sc.site, zones, sc.make_sources(), weather, sc.make_detectors(env), sc.gas_cfg)

        dt = sc.gas_dt_s
        n = int(round(sc.duration_s / dt))
        node_ppm = np.zeros((n + 1, env.n_nodes))
        soon = np.zeros((n + 1, env.n_nodes))
        forecasts: list[Forecast] = []
        hvac_on = True
        for i in range(1, n + 1):
            gas.step(dt, MapVentilationState(hvac_on=hvac_on), compute_field=False)
            if gas.alarm.state == EMERGENCY and sc.hvac_shutdown_on_alarm:
                hvac_on = False
            node_ppm[i] = gas.node_ppm
            fc = gas.forecast
            if fc is not None and (not forecasts or fc is not forecasts[-1]):
                forecasts.append(fc)
            soon[i] = node_ppm[i] if fc is None else np.maximum(node_ppm[i], _window_max(fc, gas.t, soon_s))
        alarm_t = gas.alarm.emergency_t if gas.alarm.emergency_t is not None else sc.duration_s
        return cls(dt=dt, node_ppm=node_ppm, soon_ppm=soon,
                   forecast_t0=np.array([f.t0 for f in forecasts]), forecasts=forecasts,
                   alarm_t=float(alarm_t), zones=zones, wind_offset_deg=wind_offset_deg)


def _window_max(fc: Forecast, t: float, span: float) -> np.ndarray:
    T = fc.node_ppm.shape[0]
    i0 = int(np.clip(np.floor((t - fc.t0) / fc.dt), 0, T - 1))
    i1 = int(np.clip(np.ceil((t + span - fc.t0) / fc.dt), i0 + 1, T))
    return fc.node_ppm[i0:i1].max(axis=0)
