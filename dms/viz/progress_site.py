"""A static outdoor-site PNG (walkways,
fence+gates, structures, weather station, AP areas with capacities) plus a
90s clip of agents walking along the walkway network, through gates, to
their assembly point. Reuses `dms/viz/plot.py: draw_site_panel` (now
walkway/fence/gate/structure/weather-station-aware) for
the static image, and `dms/viz/progress_baseline_clip.py`'s existing
render pipeline for the clip -- the walker itself already routes through
the outdoor graph built in `dms/graph/builder.py:
build_outdoor_network_graph` and the clearance-aware outdoor field from
`dms/fields/outdoor.py`, so nothing new is needed to make agents visibly
use the walkways; this module only renders what that pipeline already
produces.

Run as:

    python -m dms.viz.progress_site

Writes out/progress/phase3_site.png and out/progress/phase3_site_clip.mp4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..fields.outdoor import rasterize_site
from ..geometry.loader import load_site
from ..scenario.evacuation import build_world, sample_agents, simulate
from .plot import draw_site_panel
from .progress_baseline_clip import (
    CONFIG_PATH,
    PROFILE,
    SCENARIO,
    SEED,
    render_progress_clip,
)
from .viz_config import load_viz_config

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "progress"
_VIZ = load_viz_config()


def render_site_png(site, building, *, agent_defaults=None) -> Path:
    gf = building.get_floor(building.floor_ids()[0])
    grid = rasterize_site(site, parking_on=True, agent_defaults=agent_defaults)
    fig, ax = plt.subplots(figsize=(11, 11), dpi=_VIZ.progress_dpi)
    draw_site_panel(ax, site, building, gf, grid,
                     title=f"{building.id}: outdoor site -- walkways, fence, gates, structures, AP capacities",
                     agent_defaults=agent_defaults)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase3_site.png"
    fig.savefig(out_path, dpi=_VIZ.progress_dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    import imageio_ffmpeg

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    world = build_world(site, building, SCENARIO)

    png = render_site_png(site, building, agent_defaults=world.agent_defaults)
    print(f"wrote {png} ({png.stat().st_size} bytes)")

    agents = sample_agents(world, PROFILE, seed=SEED)
    res = simulate(world, agents, t_end=_VIZ.progress_clip_seconds, seed=1)
    clip = render_progress_clip(
        world, agents, res, ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe(),
        out_name="phase3_site_clip.mp4",
        title=f"B1-L -- {SCENARIO}, first {_VIZ.progress_clip_seconds:.0f}s (agents walking the site's walkways/gates to their AP)",
    )
    print(f"wrote {clip} ({clip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
