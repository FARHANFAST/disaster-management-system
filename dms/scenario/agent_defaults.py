"""configs/agent_defaults.yaml loader (no numeric
behaviour/physics constant lives in dms/ code outside a config loader).
Follows dms/viz/viz_config.py's own convention: plain yaml.safe_load,
manual dict access with explicit casts, a dedicated exception type -- no
schema-validation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class AgentDefaultsError(ValueError):
    pass


DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "agent_defaults.yaml"


@dataclass(slots=True, frozen=True)
class BodyClass:
    id: str
    shoulder_width_m: float
    depth_m: float
    mass_kg: float
    clearance_radius_m: float
    can_use_stairs: bool
    stair_speed_factor: float


@dataclass(slots=True, frozen=True)
class Persona:
    id: str
    age_band: str  # child | adolescent | adult | elderly | disabled
    gender: str | None  # "M" | "F" | None
    speed_mps: float


@dataclass(slots=True, frozen=True)
class Perception:
    R_discover_m: float
    R_vis_m: float
    R_aud_m: float
    t_discover_s: float


@dataclass(slots=True, frozen=True)
class PreMovement:
    t_pre_mu: float
    t_pre_sigma: float


@dataclass(slots=True, frozen=True)
class Kinematics:
    dt_s: float
    rec_every_steps: int
    n_headings: int
    look_indoor_m: float
    look_outdoor_m: float
    heading_jitter_sd: float
    tie_break_eps_s: float
    density_radius_m: float
    rho_max_per_m2: float
    density_slowdown_min_factor: float
    density_slowdown_max_factor: float
    separation_push: float
    still_push_damping: float
    exit_capture_m: float
    exit_jitter_m: float
    stair_capture_m: float
    ap_target_frac: float
    ap_capture_frac: float
    stair_arrival_best_frac: int
    refuge_capture_m: float


@dataclass(slots=True, frozen=True)
class Simulation:
    t_end_s: float
    exit_token_burst: float


@dataclass(slots=True, frozen=True)
class Population:
    group_prob: float
    cluster_spread_m: float
    anchor_spread_m: float
    anchor_bias_prob: float


@dataclass(slots=True, frozen=True)
class RefugeGenDefaults:
    refuge_max_dist_m: float
    width_m: float
    depth_m: float
    capacity: int
    min_clear_margin_m: float


@dataclass(slots=True, frozen=True)
class SignageGenDefaults:
    turn_cos_threshold: float
    trace_step_m: float
    trace_max_steps: int
    sign_max_spacing_m: float
    sign_min_spacing_m: float
    viewing_angle_deg: float
    mount_height_m: float
    illuminated: bool


@dataclass(slots=True, frozen=True)
class RedundancyDefaults:
    single_door_max_occupancy: int
    common_path_max_m: float


@dataclass(slots=True, frozen=True)
class SiteDefaults:
    ap_m2_per_person: float
    ap_min_dist_from_facade_m: float
    ap_min_dist_from_hazard_m: float
    fence_thickness_m: float
    walkway_speed_factor: float
    gate_capture_m: float


@dataclass(slots=True, frozen=True)
class HazardScenarioDefaults:
    """Fallback values for a scenario's `hazard:` block
    (dms/scenario/hazard_scenario.py) when a field is left unspecified for
    an active source -- data only, no dispersion physics consumes these
    yet."""

    Q_gps: float
    H_m: float
    wind_from_deg: float
    wind_speed_mps: float
    stability: str
    start_time_s: float
    duration_s: float


@dataclass(slots=True, frozen=True)
class AgentDefaults:
    body_classes: dict[str, BodyClass]
    personas: dict[str, Persona]
    door_margin_m: float
    min_lane_width_m: float
    partial_capacity_omega_p: float
    perception: Perception
    pre_movement: PreMovement
    kinematics: Kinematics
    simulation: Simulation
    population: Population
    refuges: RefugeGenDefaults
    signage: SignageGenDefaults
    redundancy: RedundancyDefaults
    site: SiteDefaults
    hazard_scenario: HazardScenarioDefaults


def effective_clearance_radius_m(body_class: BodyClass, agent_defaults: AgentDefaults) -> float:
    """The radius `door_passability_table`'s own design-time formula
    (`width_m >= 2*clearance_radius_m + door_margin_m`) implies: this is
    exactly `clearance_radius_m + door_margin_m/2`, so `width_m >=
    2*effective_clearance_radius_m` is the identical statement. Used by the
    table itself and by the grid cross-check that verifies it
    (tests/test_b1l_body_classes.py, tests/test_clearance_fixture.py) --
    a follow-up review asked to confirm the walker uses
    this same rule too; tried it and verified empirically that it doesn't
    and shouldn't: clearance_domain at this radius disconnects every one of
    B1-L's exits from its own floor's interior (skfmm's own "no zero
    contour" on literally every exit). door_margin_m is a design-time
    compliance buffer (is this door *provably, conservatively* wide
    enough), not a requirement a real body needs merely to physically fit
    through a gap -- so the walker's own movement (`sample_agents`,
    `class_reachability_report`) deliberately keeps using the bare class
    radius, and this function is for the table/verification side only."""
    return body_class.clearance_radius_m + agent_defaults.door_margin_m / 2.0


def load_agent_defaults(path: str | Path = DEFAULT_PATH) -> AgentDefaults:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        body_classes = {}
        for cid, c in data["body_classes"].items():
            body_classes[str(cid)] = BodyClass(
                id=str(cid),
                shoulder_width_m=float(c["shoulder_width_m"]),
                depth_m=float(c["depth_m"]),
                mass_kg=float(c["mass_kg"]),
                clearance_radius_m=float(c["clearance_radius_m"]),
                can_use_stairs=bool(c["can_use_stairs"]),
                stair_speed_factor=float(c["stair_speed_factor"]),
            )

        personas = {}
        for pid, v in data["personas"].items():
            personas[str(pid)] = Persona(
                id=str(pid), age_band=str(v["age_band"]),
                gender=(str(v["gender"]) if v.get("gender") is not None else None),
                speed_mps=float(v["speed_mps"]),
            )

        p = data["perception"]
        perception = Perception(
            R_discover_m=float(p["R_discover_m"]), R_vis_m=float(p["R_vis_m"]),
            R_aud_m=float(p["R_aud_m"]), t_discover_s=float(p["t_discover_s"]),
        )

        pm = data["pre_movement"]
        pre_movement = PreMovement(t_pre_mu=float(pm["t_pre_mu"]), t_pre_sigma=float(pm["t_pre_sigma"]))

        sim = data["simulation"]
        simulation = Simulation(t_end_s=float(sim["t_end_s"]), exit_token_burst=float(sim["exit_token_burst"]))

        pop = data["population"]
        population = Population(
            group_prob=float(pop["group_prob"]), cluster_spread_m=float(pop["cluster_spread_m"]),
            anchor_spread_m=float(pop["anchor_spread_m"]), anchor_bias_prob=float(pop["anchor_bias_prob"]),
        )

        k = data["kinematics"]
        kinematics = Kinematics(
            dt_s=float(k["dt_s"]), rec_every_steps=int(k["rec_every_steps"]),
            n_headings=int(k["n_headings"]), look_indoor_m=float(k["look_indoor_m"]),
            look_outdoor_m=float(k["look_outdoor_m"]), heading_jitter_sd=float(k["heading_jitter_sd"]),
            tie_break_eps_s=float(k["tie_break_eps_s"]), density_radius_m=float(k["density_radius_m"]),
            rho_max_per_m2=float(k["rho_max_per_m2"]),
            density_slowdown_min_factor=float(k["density_slowdown_min_factor"]),
            density_slowdown_max_factor=float(k["density_slowdown_max_factor"]),
            separation_push=float(k["separation_push"]), still_push_damping=float(k["still_push_damping"]),
            exit_capture_m=float(k["exit_capture_m"]), exit_jitter_m=float(k["exit_jitter_m"]),
            stair_capture_m=float(k["stair_capture_m"]),
            ap_target_frac=float(k["ap_target_frac"]), ap_capture_frac=float(k["ap_capture_frac"]),
            stair_arrival_best_frac=int(k["stair_arrival_best_frac"]),
            refuge_capture_m=float(k["refuge_capture_m"]),
        )

        rg = data["refuges"]
        refuges = RefugeGenDefaults(
            refuge_max_dist_m=float(rg["refuge_max_dist_m"]), width_m=float(rg["width_m"]),
            depth_m=float(rg["depth_m"]), capacity=int(rg["capacity"]),
            min_clear_margin_m=float(rg["min_clear_margin_m"]),
        )

        sg = data["signage"]
        signage = SignageGenDefaults(
            turn_cos_threshold=float(sg["turn_cos_threshold"]), trace_step_m=float(sg["trace_step_m"]),
            trace_max_steps=int(sg["trace_max_steps"]),
            sign_max_spacing_m=float(sg["sign_max_spacing_m"]), sign_min_spacing_m=float(sg["sign_min_spacing_m"]),
            viewing_angle_deg=float(sg["viewing_angle_deg"]),
            mount_height_m=float(sg["mount_height_m"]), illuminated=bool(sg["illuminated"]),
        )

        rd = data["redundancy"]
        redundancy = RedundancyDefaults(
            single_door_max_occupancy=int(rd["single_door_max_occupancy"]),
            common_path_max_m=float(rd["common_path_max_m"]),
        )

        st = data["site"]
        site = SiteDefaults(
            ap_m2_per_person=float(st["ap_m2_per_person"]),
            ap_min_dist_from_facade_m=float(st["ap_min_dist_from_facade_m"]),
            ap_min_dist_from_hazard_m=float(st["ap_min_dist_from_hazard_m"]),
            fence_thickness_m=float(st["fence_thickness_m"]),
            walkway_speed_factor=float(st["walkway_speed_factor"]),
            gate_capture_m=float(st["gate_capture_m"]),
        )

        hz = data["hazard_scenario"]
        hazard_scenario = HazardScenarioDefaults(
            Q_gps=float(hz["Q_gps"]), H_m=float(hz["H_m"]),
            wind_from_deg=float(hz["wind_from_deg"]), wind_speed_mps=float(hz["wind_speed_mps"]),
            stability=str(hz["stability"]), start_time_s=float(hz["start_time_s"]),
            duration_s=float(hz["duration_s"]),
        )

        return AgentDefaults(
            body_classes=body_classes, personas=personas, door_margin_m=float(data["door_margin_m"]),
            min_lane_width_m=float(data["min_lane_width_m"]),
            partial_capacity_omega_p=float(data["partial_capacity_omega_p"]),
            perception=perception, pre_movement=pre_movement, kinematics=kinematics, simulation=simulation,
            population=population, refuges=refuges, signage=signage, redundancy=redundancy, site=site,
            hazard_scenario=hazard_scenario,
        )
    except KeyError as e:
        raise AgentDefaultsError(f"{path}: missing required key {e}") from e
