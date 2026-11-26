"""A scenario-level hazard-release interface -- which of a
site's declared `HazardSource`s are active, their release profile (rate,
start time, duration), and wind -- entirely as data, resolved and
validated here. No dispersion physics consumes this yet; this module
only makes the data real and checkable, so later
work (and RL training, which needs many randomised scenario instances) has
something solid to build on.

A scenario's YAML `hazard:` block (alongside its existing `events:` list,
see dms/scenario/presets.py):

    scenarios:
      h4_release_day:
        events: []
        hazard:
          active_sources: [H-4]      # or "all" -- every site.hazards id
          wind_from_deg: 225.0
          wind_speed_mps: 1.25
          stability: SE
          releases:                   # optional per-source overrides
            H-4: {Q_gps: 20000.0, start_time_s: 60.0, duration_s: 300.0}
        split: train                  # or "test" -- optional, for RL train/test scenario sets

Any scalar field above (wind_from_deg, wind_speed_mps, or a per-source
Q_gps/H_m/start_time_s/duration_s) may instead be given as a `{min, max}`
range, sampled uniformly by `resolve_hazard_scenario`'s own seed -- same
seed, same sample, so a run is reproducible, and training can draw many
distinct scenario instances from one declared range by varying the seed.
A field left out of both the per-source override and the scenario level
falls back to the source's own declared `HazardSource.Q_gps`/`H_m` (site
data), and finally to `agent_defaults.hazard_scenario`'s config default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..geometry.site import Site

_VALID_SPLITS = frozenset({None, "train", "test"})


@dataclass(slots=True, frozen=True)
class HazardRelease:
    source_id: str
    Q_gps: float
    H_m: float
    start_time_s: float
    duration_s: float


@dataclass(slots=True, frozen=True)
class HazardScenario:
    releases: tuple[HazardRelease, ...]
    wind_from_deg: float
    wind_speed_mps: float
    stability: str
    split: str | None = None


def _resolve_scalar(raw, rng: np.random.Generator, default: float) -> float:
    """A bare number is used as-is; a `{min, max}` dict is sampled
    uniformly (seeded via `rng`); `None` falls back to `default`."""
    if raw is None:
        return float(default)
    if isinstance(raw, dict):
        lo, hi = float(raw["min"]), float(raw["max"])
        return float(rng.uniform(lo, hi))
    return float(raw)


def _resolve_active_source_ids(site: Site, active) -> tuple[str, ...]:
    if active == "all":
        return tuple(sorted(site.hazards))
    return tuple(active)


def resolve_hazard_scenario(site: Site, preset: dict, *, agent_defaults=None, seed: int | None = None) -> HazardScenario | None:
    """`None` if this scenario declares no `hazard:` block at all (the
    overwhelming majority of scenarios won't -- most are pure egress
    presets). Every numeric field is resolved to a concrete float here;
    callers never see an unresolved range."""
    hz = preset.get("hazard")
    if not hz:
        return None
    if agent_defaults is None:
        from .agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    cfg = agent_defaults.hazard_scenario
    rng = np.random.default_rng(seed)

    wind_from_deg = _resolve_scalar(hz.get("wind_from_deg"), rng, cfg.wind_from_deg) % 360.0
    wind_speed_mps = _resolve_scalar(hz.get("wind_speed_mps"), rng, cfg.wind_speed_mps)
    stability = str(hz.get("stability", cfg.stability))
    split = hz.get("split", preset.get("split"))

    source_ids = _resolve_active_source_ids(site, hz.get("active_sources", []))
    per_source = hz.get("releases", {})
    releases = []
    for sid in source_ids:
        src = site.hazards.get(sid)
        override = per_source.get(sid, {})
        q_default = (src.Q_gps if src is not None and src.Q_gps is not None else cfg.Q_gps)
        h_default = (src.H_m if src is not None and src.H_m is not None else cfg.H_m)
        releases.append(HazardRelease(
            source_id=sid,
            Q_gps=_resolve_scalar(override.get("Q_gps"), rng, q_default),
            H_m=_resolve_scalar(override.get("H_m"), rng, h_default),
            start_time_s=_resolve_scalar(override.get("start_time_s"), rng, cfg.start_time_s),
            duration_s=_resolve_scalar(override.get("duration_s"), rng, cfg.duration_s),
        ))

    return HazardScenario(
        releases=tuple(releases), wind_from_deg=wind_from_deg, wind_speed_mps=wind_speed_mps,
        stability=stability, split=split,
    )


def validate_hazard_scenario(site: Site, hs: HazardScenario) -> list[str]:
    """Every problem found, human-readable -- empty means the scenario's
    resolved hazard data is internally consistent and references real
    hazard sources. Does not (and can't yet) check anything about
    dispersion physics -- that doesn't exist yet."""
    problems: list[str] = []
    if not (0.0 <= hs.wind_from_deg < 360.0):
        problems.append(f"wind_from_deg {hs.wind_from_deg} is out of [0, 360)")
    if hs.wind_speed_mps < 0.0:
        problems.append(f"wind_speed_mps {hs.wind_speed_mps} is negative")
    if hs.split not in _VALID_SPLITS:
        problems.append(f"split {hs.split!r} is not one of {sorted(s for s in _VALID_SPLITS if s)}")
    if not hs.releases:
        problems.append("no active hazard sources -- hazard.active_sources resolved to an empty set")

    for r in hs.releases:
        prefix = f"release {r.source_id}"
        if r.source_id not in site.hazards:
            problems.append(f"{prefix}: not a hazard source declared on this site")
            continue
        if r.Q_gps < 0.0:
            problems.append(f"{prefix}: Q_gps {r.Q_gps} is negative")
        if r.H_m < 0.0:
            problems.append(f"{prefix}: H_m {r.H_m} is negative")
        if r.start_time_s < 0.0:
            problems.append(f"{prefix}: start_time_s {r.start_time_s} is negative")
        if r.duration_s <= 0.0:
            problems.append(f"{prefix}: duration_s {r.duration_s} must be positive")

    return problems
