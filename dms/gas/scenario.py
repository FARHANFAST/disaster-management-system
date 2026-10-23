"""Scenario file -> the objects each model needs (one YAML, one section per
model: `map`, `sim`, `gas`, `agents`; see configs/scenarios/*.yaml)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from ..geometry.loader import load_site
from ..geometry.site import Site
from ..grid.layers import DynamicOverlay
from ..rl.graph_env import GraphEvacEnv
from .alarm import AlarmConfig, Detector
from .dispersion import ReleaseSource
from .indoor import Ventilation
from .meteo import WeatherSchedule
from .model import GasConfig
from .routing import RoutingParams

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AgentSpec:
    floor: str
    room: str


@dataclass(slots=True)
class Scenario:
    name: str
    site: Site
    building_id: str
    blocked_exits: list[str]
    closed_stairs: list[str]
    duration_s: float
    gas_dt_s: float
    agent_hz: float
    seed: int
    gas_cfg: GasConfig
    ceiling_m: float
    release_specs: list[dict]
    weather: WeatherSchedule
    detector_specs: list[dict]
    hvac_shutdown_on_alarm: bool
    exit_door_hold_open_s: float
    routing: RoutingParams
    compare_gas_blind: bool
    self_alert_ppm: float
    premovement_s: tuple[float, float]
    shelter_replan_s: float
    spawns: list[AgentSpec] = field(default_factory=list)
    raw: dict = field(default_factory=dict)

    def make_env(self) -> GraphEvacEnv:
        overlay = DynamicOverlay(door_states={x: "UNAVAILABLE" for x in self.blocked_exits})
        return GraphEvacEnv(self.site, self.building_id, overlay=overlay, closed_stairs=self.closed_stairs)

    def make_sources(self) -> list[ReleaseSource]:
        """Fresh sources (they carry emission state, so each run needs its own)."""
        out = []
        for spec in self.release_specs:
            spec = dict(spec)
            hz = self.site.hazards[spec.pop("source")]
            out.append(ReleaseSource(x=hz.x, y=hz.y, **spec))
        return out

    def make_detectors(self, env: GraphEvacEnv) -> list[Detector]:
        building = self.site.get_building(self.building_id)
        out = []
        for spec in self.detector_specs:
            if "room" in spec:
                nid = f"{spec['floor']}:R:{spec['room']}"
                k = env.node_ids.index(nid)
                x, y = env.nav.node(nid).XY
                z = building.get_floor(spec["floor"]).z + 1.5
                out.append(Detector(spec["id"], x, y, z, node=k))
            else:
                out.append(Detector(spec["id"], float(spec["x"]), float(spec["y"]), float(spec.get("z", 1.5))))
        return out


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    data = yaml.safe_load(path.read_text())
    m, s, g, a = data["map"], data["sim"], data["gas"], data["agents"]
    site = load_site(REPO_ROOT / m["config"])

    vent = dict(g.get("ventilation", {}))
    ceiling_m = float(vent.pop("ceiling_m", 3.0))
    fc = g.get("forecast", {})
    gas_cfg = GasConfig(
        mw=float(g.get("mw", 34.08)), terrain=g.get("terrain", "urban"),
        puff_interval_s=float(g.get("puff_interval_s", 1.0)), field_h=float(g.get("field_h_m", 2.0)),
        ventilation=Ventilation(**{k: float(v) for k, v in vent.items()}),
        alarm=AlarmConfig(**{k: float(v) for k, v in g.get("alarm", {}).items()}),
        forecast_horizon_s=float(fc.get("horizon_s", 300.0)), forecast_every_s=float(fc.get("every_s", 30.0)),
        forecast_field_every_s=float(fc.get("field_every_s", 10.0)),
        hazard_ppm=float(a.get("avoid_ppm", 10.0)),
    )
    if a.get("planner", "astar") != "astar":
        raise ValueError(f"unknown planner {a['planner']!r} (available: astar)")
    resp = g.get("response", {})
    return Scenario(
        name=path.stem, site=site, building_id=m.get("building", "B1"),
        blocked_exits=list(m.get("blocked_exits", [])), closed_stairs=list(m.get("closed_stairs", [])),
        duration_s=float(s["duration_s"]), gas_dt_s=float(s.get("gas_dt_s", 1.0)),
        agent_hz=float(s.get("agent_hz", 10)), seed=int(s.get("seed", 0)),
        gas_cfg=gas_cfg, ceiling_m=ceiling_m, release_specs=list(g["releases"]),
        weather=WeatherSchedule.from_config(g["weather"]), detector_specs=list(g.get("detectors", [])),
        hvac_shutdown_on_alarm=bool(resp.get("hvac_shutdown_on_alarm", True)),
        exit_door_hold_open_s=float(resp.get("exit_door_hold_open_s", 8.0)),
        routing=RoutingParams(avoid_ppm=float(a.get("avoid_ppm", 10.0)), hard_ppm=float(a.get("hard_ppm", 100.0)),
                              lookahead=bool(a.get("lookahead", True)), linger_s=float(a.get("linger_s", 15.0))),
        compare_gas_blind=bool(a.get("compare_gas_blind", True)),
        self_alert_ppm=float(a.get("self_alert_ppm", 10.0)),
        premovement_s=tuple(float(x) for x in a.get("premovement_s", (5.0, 25.0))),
        shelter_replan_s=float(a.get("shelter_replan_s", 5.0)),
        spawns=[AgentSpec(sp["floor"], sp["room"]) for sp in a.get("spawns", [])],
        raw=data,
    )
