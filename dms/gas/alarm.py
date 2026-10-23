"""Detectors and site alarm state (Gas Dispersion design doc, "Outputs").

NORMAL -> WARNING while any detector reads >= warning_ppm (non-latching).
EMERGENCY latches once any one detector has read >= alarm_ppm continuously
for hold_s; after that the site stays in EMERGENCY for the rest of the run.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NORMAL, WARNING, EMERGENCY = "NORMAL", "WARNING", "EMERGENCY"


@dataclass(slots=True)
class Detector:
    id: str
    x: float  # site metres
    y: float
    z: float = 1.5
    node: int | None = None  # graph node of an indoor detector (reads its zone); None = outdoor


@dataclass(slots=True)
class AlarmConfig:
    alarm_ppm: float = 30.0
    warning_ppm: float = 10.0
    hold_s: float = 5.0


class AlarmSystem:
    def __init__(self, detectors: list[Detector], cfg: AlarmConfig | None = None):
        self.detectors = detectors
        self.cfg = cfg or AlarmConfig()
        self.state = NORMAL
        self.emergency_t: float | None = None
        self.trigger_detector: str | None = None
        self._above_s = np.zeros(len(detectors))

    def update(self, t: float, dt: float, readings_ppm: np.ndarray) -> str:
        """`readings_ppm[i]` is detector i's reading over the step ending at t."""
        above = readings_ppm >= self.cfg.alarm_ppm
        self._above_s = np.where(above, self._above_s + dt, 0.0)
        if self.state != EMERGENCY:
            fired = np.flatnonzero(self._above_s >= self.cfg.hold_s - 1e-9)
            if fired.size:
                self.state = EMERGENCY
                self.emergency_t = t
                self.trigger_detector = self.detectors[int(fired[0])].id
            else:
                self.state = WARNING if (readings_ppm >= self.cfg.warning_ppm).any() else NORMAL
        return self.state
