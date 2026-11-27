"""The presentation layout (dms/viz/presentation.py): the static-frame
renderer and the video + schema-v3 frames.json pipeline that shares its
panel-building code. Exercises both against a small B1-L population so
the suite stays fast -- rendering itself is covered, not re-verifying the
simulation physics (that's tests/test_b1l_evacuation.py)."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest

from dms.geometry.loader import load_site
from dms.scenario.evacuation import build_world, sample_agents, simulate
from dms.viz.presentation import (
    FRAMES_SCHEMA_VERSION_V3,
    save_presentation_png,
    write_presentation_video_and_frames,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey_large.yaml"

_LAYOUT_PANELS = {"bottom", "top", "site", "status", "legend"}
_V1_V2_FIELDS = (
    "scenario", "title", "subtitle", "profile", "seed", "n", "dt",
    "times", "size", "exits", "exit_used", "t_exit", "ap_used", "aps",
    "events", "x", "y", "c", "summary", "body_classes", "agent_body_class",
)


@pytest.fixture(scope="module")
def site():
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="module")
def building(site):
    return site.get_building("B1")


@pytest.fixture(scope="module")
def small_run(site, building):
    world = build_world(site, building, "baseline")
    agents = sample_agents(world, "day", seed=7, N=20)
    res = simulate(world, agents, t_end=300.0, seed=1)
    return world, agents, res


def test_save_presentation_png_writes_a_file(tmp_path, small_run):
    world, agents, res = small_run
    out_path = tmp_path / "frame.png"
    result = save_presentation_png(
        world, agents, res, t=0.0, out_path=out_path, resolution="preview",
        scenario_title="Baseline", subtitle="test", profile="day", seed=1,
    )
    assert result == out_path
    assert out_path.exists() and out_path.stat().st_size > 1000


def test_write_presentation_video_and_frames_smoke(tmp_path, small_run):
    import imageio_ffmpeg

    world, agents, res = small_run
    mp4, bg, fj = write_presentation_video_and_frames(
        world, agents, res, scenario_title="Baseline", subtitle="test", profile="day", seed=1,
        out_dir=tmp_path, resolution="preview", ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe(),
    )
    assert mp4.exists() and mp4.stat().st_size > 1000
    assert bg.exists() and bg.stat().st_size > 1000
    assert fj.exists()
    data = json.loads(fj.read_text(encoding="utf-8"))
    assert data["schema_version"] == FRAMES_SCHEMA_VERSION_V3 == 3


def test_schema_v3_frames_json_structure(tmp_path, small_run):
    """v1/v2 fields all survive; the new `layout`/`canvas` blocks describe
    exactly the 5 presentation panels, self-consistently."""
    import imageio_ffmpeg

    world, agents, res = small_run
    _mp4, _bg, fj = write_presentation_video_and_frames(
        world, agents, res, scenario_title="Baseline", subtitle="test", profile="day", seed=1,
        out_dir=tmp_path, resolution="preview", ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe(),
    )
    data = json.loads(fj.read_text(encoding="utf-8"))

    for key in _V1_V2_FIELDS:
        assert key in data, f"missing v1/v2 field: {key}"
    assert data["dt"] == 0.5
    assert data["n"] == len(agents)

    assert "canvas" in data and "layout" in data
    w, h = data["canvas"]["width_px"], data["canvas"]["height_px"]
    assert (w, h) == tuple(data["size"])  # "size" (v1/v2) and "canvas" (v3) describe the same render
    assert set(data["layout"].keys()) == _LAYOUT_PANELS
    for name, extent in data["layout"].items():
        assert extent["px0"] < extent["px1"], name
        assert extent["py0"] < extent["py1"], name
        assert 0 <= extent["px0"] and extent["px1"] <= w, name
        assert 0 <= extent["py0"] and extent["py1"] <= h, name
        assert extent["x0"] != extent["x1"], name
        assert extent["y0"] != extent["y1"], name

    n = data["n"]
    f = len(data["times"])
    xs = np.frombuffer(base64.b64decode(data["x"]), dtype=np.int16).reshape(f, n)
    ys = np.frombuffer(base64.b64decode(data["y"]), dtype=np.int16).reshape(f, n)
    codes = np.frombuffer(base64.b64decode(data["c"]), dtype=np.uint8).reshape(f, n)
    assert np.array_equal(xs[0], np.round(res.frames[0].pos[:, 0] * 10).astype(np.int16))
    assert np.array_equal(ys[0], np.round(res.frames[0].pos[:, 1] * 10).astype(np.int16))
    floor0 = (codes[0] >> 4).astype(np.uint8)
    state0 = (codes[0] & 0b0111).astype(np.uint8)
    know0 = ((codes[0] >> 3) & 1).astype(bool)
    assert np.array_equal(floor0, res.frames[0].floor.astype(np.uint8))
    assert np.array_equal(state0, res.frames[0].state.astype(np.uint8))
    assert np.array_equal(know0, res.frames[0].know)


def test_pitch_resolution_canvas_is_4k(tmp_path, small_run):
    import imageio_ffmpeg

    world, agents, res = small_run
    _mp4, _bg, fj = write_presentation_video_and_frames(
        world, agents, res, scenario_title="Baseline", subtitle="test", profile="day", seed=1,
        out_dir=tmp_path, resolution="pitch", ffmpeg_exe=imageio_ffmpeg.get_ffmpeg_exe(),
    )
    data = json.loads(fj.read_text(encoding="utf-8"))
    assert tuple(data["size"]) == (3840, 2160)
    assert data["canvas"] == {"width_px": 3840, "height_px": 2160}
