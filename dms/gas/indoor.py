"""Indoor infiltration (Gas Dispersion design doc, "Indoor: box model").

Every zone is one well-mixed volume exchanging air with the outdoor
concentration at its own location:

    dC_in/dt = λ (C_out - C_in),    λ = ACH / 3600 s⁻¹

Integrated exactly over a step (C_out held constant within the step), so it
is stable for any dt:  C_in <- C_out + (C_in - C_out) · exp(-λ dt).

ACH comes from the map's ventilation state: HVAC on raises it, HVAC off
lowers it, a sealed room (occupants sheltering with the door shut) lowers it
further, and an exterior door held open adds a temporary boost to the zones
it opens onto. Room-to-room transport is out of scope for this version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class Ventilation:
    ach_hvac_on: float = 2.0  # typical mechanically ventilated office
    ach_hvac_off: float = 0.3  # natural infiltration, doors closed
    ach_sealed: float = 0.1  # room sealed for shelter-in-place
    ach_door_open: float = 6.0  # added while an exterior door onto the zone is open


@dataclass(slots=True)
class MapVentilationState:
    """The part of the map the gas model reads every step. Agents (and the
    facility's alarm response) write to it; changes take effect on the next
    gas step."""

    hvac_on: bool = True
    open_exits: frozenset[int] = frozenset()  # graph node indices of exterior doors currently open
    sealed_zones: frozenset[int] = frozenset()


class BoxModel:
    def __init__(self, n_zones: int):
        self.c = np.zeros(n_zones)  # ppm

    def copy(self) -> BoxModel:
        b = BoxModel(0)
        b.c = self.c.copy()
        return b

    def step(self, dt: float, c_out: np.ndarray, ach: np.ndarray) -> None:
        decay = np.exp(-(ach / 3600.0) * dt)
        self.c = c_out + (self.c - c_out) * decay


def zone_ach(n_zones: int, vent: Ventilation, state: MapVentilationState,
             zones_by_exit: dict[int, list[int]]) -> np.ndarray:
    ach = np.full(n_zones, vent.ach_hvac_on if state.hvac_on else vent.ach_hvac_off)
    for z in state.sealed_zones:
        ach[z] = min(ach[z], vent.ach_sealed)
    for e in state.open_exits:
        for z in zones_by_exit.get(e, ()):
            ach[z] += vent.ach_door_open
    return ach
