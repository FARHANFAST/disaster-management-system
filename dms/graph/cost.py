"""(N5) edge travel-time cost — SYSTEM_SPEC.md §2.2. Reads link state live;
with no agents or gas yet, rho/C/d_haz simply default to zero/inf on a Link,
but the full formula (and its UNAVAILABLE/PARTIAL handling) is exercised
end-to-end so later milestones only need to start writing non-zero values.

Scoping note for this milestone: sigma_l (N4's incline/anisotropy factor)
fits ship-deck trim/heel, not stairs or flat corridors (spec §2.2 explicitly
excludes stairs from that fit), so it is fixed at 1.0 here — the stair speed
reduction is carried entirely by chi_l (PROMPT_01 §6.1: chi=0.70 on stair
links), consistent with N4s already folding the 0.70 factor into chi rather
than sigma.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .builder import Link

V_REF = 1.19  # m/s, spec §3 parameter registry
W_RHO = 1.5
RHO_MAX = 10.0  # ped/m^2, spec §3 (N3 jam density)
W_C = 3.0
C_REF = 30.0  # ppm, alarm level
W_VIS = 0.5
RV0 = 3.37  # m, baseline visibility
W_T = 0.0  # inactive by default (spec §3)
W_H = 2.0
R_H = 161.0  # m
OMEGA_P = 2.0  # PARTIAL multiplier; matching capacity multiplier is 1/omega_p


@dataclass(slots=True)
class CostParams:
    v_ref: float = V_REF
    w_rho: float = W_RHO
    rho_max: float = RHO_MAX
    w_C: float = W_C
    C_ref: float = C_REF
    w_vis: float = W_VIS
    Rv: float = RV0  # current visibility; defaults to the no-smoke baseline
    Rv0: float = RV0
    w_T: float = W_T
    T_ten: float | None = None
    T0: float | None = None
    w_H: float = W_H
    R_H: float = R_H
    omega_p: float = OMEGA_P


_DEFAULT_PARAMS = CostParams()


def link_cost(link: Link, *, params: CostParams | None = None) -> float:
    """Spec (N5). Infinite for UNAVAILABLE/shuttered links; PARTIAL multiplies
    by omega_p. chi_l=0 would also blow up the base term, which is correct:
    an impassable link should never be chosen even if its state is stale."""
    if link.state == "UNAVAILABLE" or link.shutter_closed or link.chi <= 0:
        return math.inf

    p = params or _DEFAULT_PARAMS
    sigma = 1.0  # see module docstring

    base = link.length_m / (link.chi * sigma * p.v_ref)

    bracket = 1.0
    bracket += p.w_rho * (link.rho / p.rho_max)
    bracket += p.w_C * (link.C / p.C_ref)
    bracket += p.w_vis * (1.0 - p.Rv / p.Rv0)
    if p.w_T and p.T_ten is not None and p.T0 is not None and p.T_ten != p.T0:
        bracket += p.w_T * (link.temp - p.T0) / (p.T_ten - p.T0)
    if link.d_haz != math.inf:
        bracket += p.w_H * max(0.0, 1.0 - link.d_haz / p.R_H)

    cost = base * bracket
    if link.state == "PARTIAL":
        cost *= p.omega_p
    return cost
