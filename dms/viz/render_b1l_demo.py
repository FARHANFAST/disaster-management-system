"""spec §4 deliverable: renders the baseline and compound-failure B1-L
evacuation runs to MP4 (out/videos/) plus frames.json and background PNG
for the web player (out/frames/).

Run as:

    python -m dms.viz.render_b1l_demo [scenario ...]

Defaults to both "baseline" and "compound". Needs an ffmpeg binary, supplied
by the imageio_ffmpeg package.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ..geometry.loader import load_site
from ..scenario.evacuation import build_world, sample_agents, simulate, summarize
from .presentation import write_presentation_video_and_frames

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "buildings" / "two_storey_large.yaml"
OUT_ROOT = Path(__file__).resolve().parents[2] / "out"
VIDEOS_DIR = OUT_ROOT / "videos"
FRAMES_DIR = OUT_ROOT / "frames"

SCENARIO_TEXT = {
    "baseline": ("Baseline", "All 7 exits and 5 stairs open; X-SE service door locked (normal)."),
    "compound": (
        "Compound failure",
        ("X-S main exit blocked (known) | X-N blocked but NOT known | main stair ST-C closed "
         "| maintenance barriers active."),
    ),
}
PROFILE = "day"
SEED = 7


def _ffmpeg_exe() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def run(scenario: str) -> None:
    site = load_site(CONFIG_PATH)
    building = site.get_building("B1")
    world = build_world(site, building, scenario)
    agents = sample_agents(world, PROFILE, seed=SEED)
    res = simulate(world, agents, seed=1)
    title, subtitle = SCENARIO_TEXT.get(scenario, (scenario, ""))
    other_stuck = res.n_stuck - res.n_awaiting_refuge
    print(
        f"{scenario}: {len(agents)} agents, {len(res.frames)} frames, "
        f"stuck={res.n_stuck} (awaiting_refuge={res.n_awaiting_refuge}, other={other_stuck})",
    )
    if other_stuck:
        raise SystemExit(f"{scenario}: {other_stuck} agent(s) genuinely stuck -- refusing to render")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    mp4, bg, fj = write_presentation_video_and_frames(
        world, agents, res, scenario_title=title, subtitle=subtitle, profile=PROFILE, seed=1,
        out_dir=VIDEOS_DIR, resolution="preview", ffmpeg_exe=_ffmpeg_exe(),
    )
    bg = bg.replace(FRAMES_DIR / bg.name)
    fj = fj.replace(FRAMES_DIR / fj.name)
    print(f"  wrote {mp4}")
    print(f"  wrote {bg}")
    print(f"  wrote {fj}")
    for line in summarize(world, agents, res):
        print("   ", line)


def main() -> None:
    scenarios = sys.argv[1:] or ["baseline", "compound"]
    for scenario in scenarios:
        run(scenario)


if __name__ == "__main__":
    main()
