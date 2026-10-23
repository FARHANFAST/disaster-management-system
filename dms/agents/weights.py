"""Decision weights W1 (own knowledge), W2 (other people), W3 (hazard)
that change with who the agent is and with what happens to it (design doc
v0.3 §5). This is a behaviour rule, not an optimiser: a Rationalist whose
known route turns out closed trusts its knowledge less even when that was
the wrong lesson to learn.

    W = softmax(z_base + dz)                          (always sums to 1)
    z_base = log W_persona + attribute modifiers
    dz -> 0 with time constant tau_recover (1 + stress)    (calms back down)
    events add impulses to dz (route failed, smelt gas, crowd heads elsewhere)
    PANICKED: move panic_shift of weight from W1 to W2 (spec v0.1 §3.3)
"""

from __future__ import annotations

import math

import numpy as np

from .config import FAIL_SPLIT, PERSONA_W, BehaviourConfig
from .population import Profile


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max())
    return e / e.sum()


def base_logits(persona: str, p: Profile, cfg: BehaviourConfig) -> np.ndarray:
    z = np.log(np.asarray(PERSONA_W[persona], dtype=float))
    z[0] += cfg.mod_fam_knowledge * (p.familiarity - 0.5) + cfg.mod_exp_knowledge * p.experience
    z[1] += cfg.mod_fam_social * (0.5 - p.familiarity) + (cfg.mod_vip_social if p.vip else 0.0)
    z[2] += cfg.mod_sus_hazard * (p.susceptibility - 1.0)
    return z


class WeightState:
    def __init__(self, profile: Profile, cfg: BehaviourConfig):
        self.p = profile
        self.cfg = cfg
        self.persona = profile.persona  # effective persona (may shift to Follower)
        self.z_base = base_logits(self.persona, profile, cfg)
        self.dz = np.zeros(3)

    def set_persona(self, persona: str) -> None:
        self.persona = persona
        self.z_base = base_logits(persona, self.p, self.cfg)

    def weights(self, panicked: bool = False) -> np.ndarray:
        w = softmax(self.z_base + (self.dz if self.cfg.adapt_weights else 0.0))
        if panicked:
            shift = min(self.cfg.panic_shift, w[0])
            w = w + np.array([-shift, shift, 0.0])
        return w

    def relax(self, dt: float, stress: float) -> None:
        self.dz *= math.exp(-dt / (self.cfg.tau_recover_s * (1.0 + stress)))

    # ------------------------------------------------------------- events

    def route_failed(self) -> None:
        to_social, to_hazard = FAIL_SPLIT[self.persona]
        a = self.cfg.fail_impulse * (1.0 - 0.5 * self.p.experience)
        self.dz += a * np.array([-1.0, to_social, to_hazard])

    def smelt_gas(self) -> None:
        self.dz[2] += self.cfg.smell_impulse * self.p.susceptibility

    def crowd_elsewhere(self) -> None:
        self.dz[1] += self.cfg.crowd_impulse * (1.0 - self.p.familiarity)
