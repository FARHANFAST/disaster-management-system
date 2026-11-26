"""A static floor-plan PNG showing refuges
and exit signs on both floors, plus a 90s clip of the baseline scenario
showing wheelchair-class agents reaching their floor's refuge. Reuses
`dms/viz/plot.py`'s now refuge/sign-aware `plot_floor_layers` (so the
static PNG and the clip's own background are drawn by the exact same code)
and `dms/viz/progress_baseline_clip.py`'s existing render pipeline for the
clip.

Run as:

    python -m dms.viz.progress_refuges_signs

Writes out/progress/phase2_refuges_signs.png and
out/progress/phase2_refuges_clip.mp4.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ..geometry.loader import load_site
from ..scenario.evacuation import build_world, sample_agents, simulate
from .plot import plot_floor_layers
from .progress_baseline_clip import CONFIG_PATH, render_progress_clip
from .viz_config import load_viz_config

OUT_DIR = Path(__file__).resolve().parents[2] / "out" / "progress"
_VIZ = load_viz_config()


def render_refuges_signs_png(building, world) -> Path:
    fig, axes = plt.subplots(1, len(world.floor_order), figsize=(16, 8.5), dpi=_VIZ.progress_dpi)
    if len(world.floor_order) == 1:
        axes = [axes]
    for ax, fid in zip(axes, world.floor_order):
        plot_floor_layers(building, building.get_floor(fid), world.grids[fid], ax=ax,
                           title=f"{building.id} {fid} -- refuges + exit signs")
    fig.suptitle(f"{building.id}: refuges and wayfinding signs", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "phase2_refuges_signs.png"
    fig.savefig(out_path, dpi=_VIZ.progress_dpi)
    plt.close(fig)
    return out_path


def main() -> None:
    import imageio_ffmpeg

    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    world = build_world(site, building, "baseline")

    png = render_refuges_signs_png(building, world)
    print(f"wrote {png} ({png.stat().st_size} bytes)")

    agents = sample_agents(world, "day", seed=7)
    res = simulate(world, agents, t_end=_VIZ.progress_clip_seconds, seed=1)
    clip = render_progress_clip(
        world, agents, res, ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe(),
        out_name="phase2_refuges_clip.mp4",
        title=f"B1-L -- baseline, first {_VIZ.progress_clip_seconds:.0f}s (wheelchair agents reaching refuges)",
    )
    print(f"wrote {clip} ({clip.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
