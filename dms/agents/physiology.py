"""What the gas does to an agent (design doc v0.3 §6; supervisor brief §3).

    dose        D += C^n dt / 60            ppm.min, Haber's rule (n = 1 by default)
    band        B0 < 0.5 <= B1 < 10 <= B2 < 100 <= B3 < 500 <= B4      (ppm, current air)
    speed       x band factor (1, 1, 0.8, 0.5, 0) and x the Disabled factor once latched
    collapse    hazard h = lambda(band) (1 + D / 100) per s,  P = 1 - exp(-h dt)
                B4 (>= 500 ppm) or D >= 500 ppm.min: incapacitated immediately
    status      worst of (band of peak, dose level); never improves during a run
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import HEALTH, PhysiologyConfig


def band(c_ppm: float, cfg: PhysiologyConfig) -> int:
    return int(np.searchsorted(cfg.band_edges_ppm, c_ppm, side="right"))


@dataclass(slots=True)
class Body:
    cfg: PhysiologyConfig
    susceptibility: float = 1.0
    health: str = "Healthy"
    dose: float = 0.0
    peak: float = 0.0
    incapacitated: bool = False

    def _status_rank(self) -> int:
        return HEALTH.index(self.health)

    def _latch(self, status: str) -> None:
        if HEALTH.index(status) > self._status_rank():
            self.health = status

    def breathe(self, c_ppm: float, dt: float, rng: np.random.Generator) -> bool:
        """Advance the body by dt in air of concentration c (ppm). Returns
        True when this step incapacitated the agent."""
        if self.incapacitated:
            return False
        cfg = self.cfg
        c_eff = c_ppm * self.susceptibility  # a susceptible body reacts as if the air were worse
        self.dose += (c_ppm ** cfg.haber_n) * dt / 60.0
        self.peak = max(self.peak, c_ppm)
        b = band(c_eff, cfg)
        self._latch(("Healthy", "Irritated", "Distressed", "Disabled", "Incapacitated")[b])
        d = self.dose * self.susceptibility
        for limit, status in zip(cfg.dose_status, ("Irritated", "Distressed", "Disabled")):
            if d >= limit:
                self._latch(status)
        rate = cfg.collapse_rate[b]
        if b == 4 or d >= cfg.dose_limit:
            hit = True
        elif rate > 0:
            hit = rng.random() < 1.0 - math.exp(-rate * (1.0 + d / cfg.dose_ref) * dt)
        else:
            hit = False
        if hit:
            self.incapacitated = True
            self.health = "Incapacitated"
        return hit

    def speed_factor(self, c_ppm: float) -> float:
        if self.incapacitated:
            return 0.0
        f = self.cfg.band_speed[band(c_ppm * self.susceptibility, self.cfg)]
        if self.health == "Disabled":
            f = min(f, self.cfg.band_speed[3])
        elif self.health == "Distressed":
            f = min(f, self.cfg.band_speed[2])
        return f
