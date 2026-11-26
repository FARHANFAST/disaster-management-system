"""Assembly-point sizing -- every AP on a site must be
large enough to hold the site's full design population (spec: wind may
leave any one AP as the only safe one, so each is sized independently for
the whole population, not a per-AP share), at `ap_m2_per_person` (configs/
agent_defaults.yaml `site:`). Generic over any site: sums every building's
own worst-case headcount (dms/geometry/code_check.py:
total_design_population), never a hardcoded number.
"""

from __future__ import annotations

import math

from .code_check import total_design_population
from .primitives import Rect
from .site import Site


def required_ap_diameter_m(site: Site, agent_defaults=None) -> float:
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    population = sum(total_design_population(b) for b in site.buildings.values())
    area = agent_defaults.site.ap_m2_per_person * population
    return 2.0 * math.sqrt(area / math.pi)


def apply_ap_sizing(site: Site, agent_defaults=None) -> float:
    """Expands every AP's `diameter_m` up to `required_ap_diameter_m`, if
    its own declared/default value is smaller -- never shrinks a
    deliberately larger YAML value. Returns the required diameter (for
    reporting)."""
    required = required_ap_diameter_m(site, agent_defaults)
    for ap in site.assembly_points.values():
        ap.diameter_m = max(ap.diameter_m, required)
    return required


def _point_to_rect_dist(x: float, y: float, rect) -> float:
    dx = max(rect.x0 - x, 0.0, x - rect.x1)
    dy = max(rect.y0 - y, 0.0, y - rect.y1)
    return math.hypot(dx, dy)


def validate_assembly_points(site: Site, agent_defaults=None) -> list[str]:
    """Every problem found: an AP whose own disc (its edge, not just its
    centre point) comes within `ap_min_dist_from_facade_m` of a building
    footprint, or `ap_min_dist_from_hazard_m` of a hazard source. Generic:
    iterates whatever buildings/hazards/APs the site actually declares."""
    if agent_defaults is None:
        from ..scenario.agent_defaults import load_agent_defaults
        agent_defaults = load_agent_defaults()
    cfg = agent_defaults.site
    problems: list[str] = []
    for ap in site.assembly_points.values():
        r = ap.diameter_m / 2.0
        for building in site.buildings.values():
            for floor in building.floors.values():
                fp = floor.footprint
                x0, y0 = building.to_site_xy(fp.x0, fp.y0)
                x1, y1 = building.to_site_xy(fp.x1, fp.y1)
                lo_x, hi_x = min(x0, x1), max(x0, x1)
                lo_y, hi_y = min(y0, y1), max(y0, y1)
                rr = Rect(lo_x, lo_y, hi_x, hi_y)
                dist = _point_to_rect_dist(ap.x, ap.y, rr) - r
                if dist < cfg.ap_min_dist_from_facade_m:
                    problems.append(
                        f"{ap.id} is only {max(dist, 0.0):.1f} m from {building.id}:{floor.id}'s facade "
                        f"(< {cfg.ap_min_dist_from_facade_m:.0f} m min)"
                    )
        for hz in site.hazards.values():
            dist = math.hypot(ap.x - hz.x, ap.y - hz.y) - r
            if dist < cfg.ap_min_dist_from_hazard_m:
                problems.append(
                    f"{ap.id} is only {max(dist, 0.0):.1f} m from hazard {hz.id} "
                    f"(< {cfg.ap_min_dist_from_hazard_m:.0f} m min)"
                )
    return problems
