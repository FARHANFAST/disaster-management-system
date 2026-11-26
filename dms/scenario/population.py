"""Occupancy sampler (spec §6) -- turns a building's
`occupancy_profiles`/`persona_mix` YAML data into a concrete list of placed
agents for one time-of-day preset.

No physics/kinematics here: the output is just `(floor, space, x, y,
persona)` placements, for the viz layer and future SFM seeding, consistent
with there being no agent-physics module yet.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..geometry.building import Building, Space
from ..geometry.primitives import Rect
from .agent_defaults import BodyClass, Persona, Population, load_agent_defaults

# The four shipped time-of-day labels (match every `occupancy_profiles` use
# entry's leaf keys in configs/buildings/*.yaml). Any other string works too
# -- e.g. a scenario-specific "custom" schedule -- as long as `profiles`
# (defaulting to `building.occupancy_profiles`) has a matching leaf for it.
PRESETS = ("day", "lunch", "meeting", "night", "custom")

WORKSTATION_TYPES = frozenset({"table", "reception_desk"})


class PopulationError(ValueError):
    pass


# Persona is defined in dms/scenario/agent_defaults.py (re-exported here for
# existing callers) -- its catalog of speeds now lives entirely in
# configs/agent_defaults.yaml's `personas:` section (one source of truth,
# no persona speed is ever a
# Python literal). A building's own persona_mix.v0 still overrides it per
# building (see effective_speed below); nothing in dms/ holds a second,
# independent copy.
#
# Body size (shoulder width/depth/mass/clearance radius/can_use_stairs) is
# NOT part of a persona either -- it comes
# from a building's own `persona_mix[...].body_class_mix`, resolved against
# the `body_classes` catalog in the same config file.

DEFAULT_BODY_CLASS_MIX: dict[str, float] = {"standard": 1.0}  # fallback when a persona_mix
# entry declares no body_class_mix of its own (older buildings) -- every
# occupant is the "standard" body class, matching prior (implicit, single-size) behaviour.


@dataclass(slots=True, frozen=True)
class PlacedAgent:
    id: int
    floor: str
    space_id: str
    x: float
    y: float
    persona: str  # key into the persona catalog
    body_class: str  # key into the body-class catalog


def _resolve_profile_key(space: Space) -> str | None:
    """`Space.use` if set; corridor/stub kinds fall back to the "corridor"
    profile; stair/void kinds (and any untagged room) are skipped."""
    if space.use:
        return space.use
    if space.kind in ("corridor", "stub"):
        return "corridor"
    return None


def _headcount(leaf: dict, area_m2: float, rng: random.Random) -> int:
    if "count" in leaf:
        return int(leaf["count"])
    if "count_range" in leaf:
        lo, hi = leaf["count_range"]
        return rng.randint(int(lo), int(hi))
    if "density_per_m2" in leaf:
        return round(leaf["density_per_m2"] * area_m2)
    raise PopulationError(f"occupancy leaf has no count/count_range/density_per_m2: {leaf!r}")


def _persona_lookup(personas: dict[str, Persona], persona_id: str) -> Persona | None:
    """Case-insensitive catalog lookup: a building's own `persona_mix` ids
    (spec §2 preserves them verbatim from its source YAML, e.g.
    `adult_M`/`elderly_F`) are not guaranteed to match the config catalog's
    fixed lowercase spelling exactly -- the two are the same identity
    (age-band + gender), just spelled differently by whoever authored that
    building's YAML, so a bare `==` lookup would reject a perfectly valid
    id over case alone."""
    if persona_id in personas:
        return personas[persona_id]
    lowered = persona_id.lower()
    for key, persona in personas.items():
        if key.lower() == lowered:
            return persona
    return None


def _pick_persona(rng: random.Random, persona_mix: dict[str, dict]) -> str:
    ids = list(persona_mix.keys())
    weights = [v["share"] for v in persona_mix.values()]
    return rng.choices(ids, weights=weights, k=1)[0]


def _pick_body_class(rng: random.Random, persona_mix: dict[str, dict], persona_id: str) -> str:
    """Each persona's own `body_class_mix` -- falls
    back to 100% "standard" when the building's persona_mix entry declares
    none, matching the implicit single-body-size behaviour every building
    had before body classes existed."""
    entry = persona_mix.get(persona_id, {})
    mix = entry.get("body_class_mix") or DEFAULT_BODY_CLASS_MIX
    ids = list(mix.keys())
    weights = list(mix.values())
    return rng.choices(ids, weights=weights, k=1)[0]


def effective_speed(
    persona_id: str, persona_mix: dict[str, dict], personas: dict[str, Persona] | None = None,
) -> float:
    """A building's own `persona_mix` may declare an explicit `v0` override
    (spec §2) for a persona id; falls back to
    `configs/agent_defaults.yaml`'s `personas` catalog's `speed_mps` only
    when the building doesn't."""
    personas = personas if personas is not None else load_agent_defaults().personas
    entry = persona_mix.get(persona_id)
    if entry is not None and entry.get("v0") is not None:
        return entry["v0"]
    persona = _persona_lookup(personas, persona_id)
    if persona is None:
        raise PopulationError(f"persona id {persona_id!r} has no v0 override and matches no catalog persona")
    return persona.speed_mps


def _workstation_anchors(space: Space, floor) -> list[tuple[float, float]]:
    return [
        obs.rect.center
        for obs in floor.obstacles.values()
        if obs.space_id == space.id and obs.type in WORKSTATION_TYPES
    ]


def _room_obstacle_rects(space: Space, floor) -> list[Rect]:
    return [
        obs.rect
        for obs in floor.obstacles.values()
        if obs.space_id == space.id and obs.active_by_default
    ]


def _sample_point_in_room(
    space: Space,
    obstacle_rects: list[Rect],
    placed: list[tuple[float, float, float]],
    anchors: list[tuple[float, float]],
    rng: random.Random,
    radius: float,
    *,
    group_prob: float,
    cluster_spread_m: float,
    anchor_spread_m: float,
    anchor_bias_prob: float,
    max_tries: int = 200,
) -> tuple[float, float]:
    """Rejection-sample a disc of `radius` that avoids obstacle rects and
    other agents' own discs, biased toward workstation anchors (desks/
    tables) and toward existing colleagues in the room (`group_prob`) to
    represent clusters rather than a uniform scatter.

    `placed` is `(x, y, radius)` per already-placed agent (placement
    spacing uses each agent's own radius, not a
    single shared constant) -- two agents must keep at least the SUM of
    their own radii apart, not `2 * radius`."""
    r = space.rect
    lo_x, hi_x = r.x0 + radius, r.x1 - radius
    lo_y, hi_y = r.y0 + radius, r.y1 - radius
    if lo_x >= hi_x or lo_y >= hi_y:
        return r.center  # room too small for a full-radius margin

    placed_xy = [(px, py) for px, py, _ in placed]
    for _ in range(max_tries):
        if placed_xy and rng.random() < group_prob:
            cx, cy = rng.choice(placed_xy)
            spread = cluster_spread_m
        elif anchors and rng.random() < anchor_bias_prob:
            cx, cy = rng.choice(anchors)
            spread = anchor_spread_m
        else:
            cx = cy = None

        if cx is None:
            x = rng.uniform(lo_x, hi_x)
            y = rng.uniform(lo_y, hi_y)
        else:
            x = min(max(cx + rng.uniform(-spread, spread), lo_x), hi_x)
            y = min(max(cy + rng.uniform(-spread, spread), lo_y), hi_y)

        cand = Rect(x - radius, y - radius, x + radius, y + radius)
        if any(cand.overlaps(o) for o in obstacle_rects):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < (radius + pr) ** 2 for px, py, pr in placed):
            continue
        return x, y

    return r.center  # exhausted tries -- safe fallback, still inside the room


def sample_population(
    building: Building,
    preset: str,
    *,
    seed: int,
    N: int | None = None,
    group_prob: float | None = None,
    personas: dict[str, Persona] | None = None,
    profiles: dict[str, dict] | None = None,
    body_classes: dict[str, BodyClass] | None = None,
    population_cfg: Population | None = None,
) -> list[PlacedAgent]:
    """Sample one deterministic population snapshot.

    `preset` selects the time-of-day leaf (`day`/`lunch`/`meeting`/`night`,
    or any other key `profiles` defines -- e.g. a scenario-specific
    "custom" schedule passed via `profiles=`). `N`, if given, proportionally
    rescales every room's raw headcount so the total equals `N`; otherwise
    each room uses its own profile-derived count. Restricted rooms
    (`Space.restricted`) sample only the `operator` persona, regardless of
    `persona_mix`. Same `seed` always reproduces the same placement.

    `body_classes` and `population_cfg` (clustering probabilities/spreads)
    default to `configs/agent_defaults.yaml` -- pass a
    different value to override either per run (e.g. during later RL
    training randomisation). `group_prob` is kept as its own override
    convenience on top of `population_cfg`, since it's the one placement
    knob existing callers already set directly.
    """
    _defaults = None
    if personas is None or body_classes is None or population_cfg is None:
        _defaults = load_agent_defaults()
    personas = personas if personas is not None else _defaults.personas
    profiles = profiles if profiles is not None else building.occupancy_profiles
    body_classes = body_classes if body_classes is not None else _defaults.body_classes
    population_cfg = population_cfg if population_cfg is not None else _defaults.population
    group_prob = group_prob if group_prob is not None else population_cfg.group_prob

    unknown = [pid for pid in building.persona_mix if _persona_lookup(personas, pid) is None]
    if unknown:
        raise PopulationError(f"persona_mix references unknown persona id(s): {sorted(unknown)}")

    rng = random.Random(seed)

    rooms: list[tuple[Space, object, int]] = []  # (space, floor, raw_count)
    for floor in building.floors.values():
        for space in floor.spaces.values():
            if space.kind in ("stair", "void"):
                continue
            profile_key = _resolve_profile_key(space)
            if profile_key is None:
                continue
            profile = profiles.get(profile_key)
            if profile is None:
                continue
            leaf = profile.get(preset)
            if leaf is None:
                continue
            count = _headcount(leaf, space.rect.area, rng)
            if count > 0:
                rooms.append((space, floor, count))

    total_raw = sum(c for _, _, c in rooms)
    scale = (N / total_raw) if (N is not None and total_raw > 0) else 1.0

    agents: list[PlacedAgent] = []
    next_id = 0
    for space, floor, raw_count in rooms:
        scaled = raw_count * scale
        count = int(scaled)
        frac = scaled - count
        if rng.random() < frac:
            count += 1
        if count <= 0:
            continue

        obstacle_rects = _room_obstacle_rects(space, floor)
        anchors = _workstation_anchors(space, floor)
        placed: list[tuple[float, float, float]] = []
        for _ in range(count):
            persona_id = "operator" if space.restricted else _pick_persona(rng, building.persona_mix)
            body_class_id = _pick_body_class(rng, building.persona_mix, persona_id)
            body_class = body_classes.get(body_class_id)
            if body_class is None:
                raise PopulationError(f"body_class_mix references unknown body class id {body_class_id!r}")
            x, y = _sample_point_in_room(
                space, obstacle_rects, placed, anchors, rng, body_class.clearance_radius_m,
                group_prob=group_prob, cluster_spread_m=population_cfg.cluster_spread_m,
                anchor_spread_m=population_cfg.anchor_spread_m, anchor_bias_prob=population_cfg.anchor_bias_prob,
            )
            placed.append((x, y, body_class.clearance_radius_m))
            agents.append(PlacedAgent(
                id=next_id, floor=floor.id, space_id=space.id, x=x, y=y,
                persona=persona_id, body_class=body_class_id,
            ))
            next_id += 1

    return agents
