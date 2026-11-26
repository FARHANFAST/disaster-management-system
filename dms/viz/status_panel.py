"""The live status-panel text -- counters, per-exit and
per-assembly-point counts -- factored out of the renderer so a future
dashboard can be fed by exactly the same function the presentation layout
and frames.json exporter use. Pure data in, strings out; no matplotlib
import here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..scenario.evacuation import (
    DISC,
    DONE,
    IN_REFUGE,
    MOVE,
    PRE,
    QUEUE,
    STAIR,
    Agent,
    EvacuationWorld,
    Frame,
    SimResult,
)


def nearest_frame(res: SimResult, t: float) -> Frame:
    """The recorded frame closest to `t` -- frames are recorded on a fixed
    stride (`evacuation.REC_EVERY * evacuation.DT`), so `t` rarely lands
    exactly on one."""
    times = np.array([fr.t for fr in res.frames])
    idx = int(np.argmin(np.abs(times - t)))
    return res.frames[idx]


@dataclass(slots=True, frozen=True)
class StatusSnapshot:
    t: float
    counts_by_state: dict[str, int]  # human-readable state label -> count
    at_assembly_point: int
    n_agents: int
    per_exit: list[tuple[str, int]]  # exit id -> count that has left by this exit, by this frame
    per_ap: list[tuple[str, int]]  # AP id -> count that has arrived at this AP, by this frame


def build_status_snapshot(world: EvacuationWorld, agents: list[Agent], res: SimResult, t: float) -> StatusSnapshot:
    fr = nearest_frame(res, t)
    st, fl = fr.state, fr.floor
    site_idx = len(world.floor_order)
    n = len(agents)

    exited = ~np.isnan(res.t_exit) & (res.t_exit <= fr.t)
    per_exit = [(e.id, int(((res.exit_used == k) & exited).sum())) for k, e in enumerate(world.exits)]

    arrived = fr.state == DONE
    per_ap = [(ap_id, int((arrived & (res.ap_used == k)).sum())) for k, ap_id in enumerate(world.ap_ids)]

    counts_by_state = {
        "Not yet moving": int((st == PRE).sum()),
        "Evacuating": int(((st == MOVE) & (fl < site_idx)).sum()),
        "Queue at stairs": int((st == QUEUE).sum()),
        "On stairs": int((st == STAIR).sum()),
        "Outside, walking": int(((st == MOVE) & (fl == site_idx)).sum()),
        "Found exit blocked (pausing)": int((st == DISC).sum()),
        "In refuge (safe)": int((st == IN_REFUGE).sum()),
    }

    return StatusSnapshot(
        t=fr.t, counts_by_state=counts_by_state, at_assembly_point=int(arrived.sum()), n_agents=n,
        per_exit=per_exit, per_ap=per_ap,
    )


def status_lines(snap: StatusSnapshot) -> list[str]:
    """The HUD's own multi-line text -- same content `evacuation_video.py`'s
    `RunRenderer` draws, now available to any caller (a static presentation
    frame, a future dashboard) without re-deriving it."""
    lines = [f"{label:<28}{count:4d}" for label, count in snap.counts_by_state.items()]
    lines.append(f"{'At assembly point':<28}{snap.at_assembly_point:4d} / {snap.n_agents}")
    lines.append("")
    lines.append("Left building by exit:")
    half = max(1, len(snap.per_exit) // 2 + len(snap.per_exit) % 2)
    lines.append("  " + "  ".join(f"{a}:{b}" for a, b in snap.per_exit[:half]))
    lines.append("  " + "  ".join(f"{a}:{b}" for a, b in snap.per_exit[half:]))
    lines.append("")
    lines.append("At assembly point:")
    lines.append("  " + "  ".join(f"{a}:{b}" for a, b in snap.per_ap))
    return lines
