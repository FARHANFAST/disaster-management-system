"""Shared rendering constants used by the current video/static renderer
(`dms/viz/presentation.py`) and by the earlier progress diagnostic
(`dms/viz/progress_baseline_clip.py`): the agent-state-and-knowledge to
colour mapping, the title-card/discovery-pause/end-hold timing, and the
legacy 3-panel axes-rect fractions `progress_baseline_clip.py` still uses
for its own, deliberately different, layout.

The old 3-panel/fixed-1760x990/scatter-dot video pipeline that used to
live here (`setup_figure`, `RunRenderer`, `write_video_and_frames`,
`_export_frames_json`, schema v1/v2) has been removed now that
`dms/viz/render_b1l_demo.py` and every other production caller render
through `dms/viz/presentation.py`'s 5-panel, true-body-size layout
instead.
"""

from __future__ import annotations

from ..scenario.evacuation import DISC, IN_REFUGE, MOVE, PRE, QUEUE
from .viz_config import load_viz_config

FPS = 6
VIDEO_DT_S = 0.5  # matches evacuation.REC_EVERY * evacuation.DT
TITLE_CARD_S = 3.0
DISCOVERY_PAUSE_S = 3.0
END_HOLD_S = 4.0

_VIZ = load_viz_config()

_GF_AX = (0.045, 0.55, 0.50, 0.34)
_FF_AX = (0.045, 0.10, 0.50, 0.34)
_SITE_AX = (0.615, 0.26, 0.355, 0.60)


def _agent_color(state: int, know: bool) -> str:
    if state == MOVE and know:
        return _VIZ.agent_state_colors["rerouted"]
    return _VIZ.agent_state_colors.get(
        {PRE: "PRE", MOVE: "MOVE", QUEUE: "QUEUE", DISC: "DISC", IN_REFUGE: "IN_REFUGE"}.get(state, "MOVE"),
        "#888888",
    )
