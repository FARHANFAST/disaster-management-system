"""configs/viz.yaml loader (spec §1.1) — colors, markers, hatching and
obstacle-type initials for every viz script, so none of it is hard-coded in
plot.py/plot_fields.py/etc. Follows dms/geometry/loader.py's own convention:
plain yaml.safe_load, manual dict access with explicit casts, a dedicated
exception type — no schema-validation dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class VizConfigError(ValueError):
    pass


DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "viz.yaml"


@dataclass(slots=True)
class PresentationConfig:
    resolutions: dict[str, tuple[int, int]]
    preview_dpi: int
    pitch_dpi: int
    min_text_px: float
    min_agent_px: float
    target_px_per_m: float
    scale_bar_m: float
    title_fontsize_px: float
    subtitle_fontsize_px: float
    clock_fontsize_px: float
    room_label_fontsize_px: float
    room_labels_pitch_only: bool
    exit_label_fontsize_px: float
    exit_bar_width_px: float
    hazard_inactive_alpha: float
    exit_bar_colors: dict[str, str]


@dataclass(slots=True)
class VizConfig:
    category_colors: dict[str, str]
    door_color: str
    exit_colors: dict[str, str]  # state -> hex
    exit_markers: dict[str, str]  # state -> marker char, only for non-AVAILABLE states
    exit_hatch: dict[str, str]  # state -> matplotlib hatch pattern
    assembly_point_color: str
    hazard_color: str
    hazard_room_tint: str  # light fill for rooms that contain a hazard source -- NOT tied to Space.restricted
    obstacle_initials: dict[str, str]  # obstacle type -> short label
    site_colors: dict[str, str]  # open_ground | building | obstacle | agent
    agent_state_colors: dict[str, str]  # PRE | MOVE | QUEUE | DISC | rerouted (spec §4)
    agent_pre_edge_color: str
    progress_fig_width_px: int
    progress_fig_height_px: int
    progress_dpi: int
    progress_table_fontsize_px: int
    progress_clip_seconds: float
    progress_clip_fps: int
    class_shade_colors: dict[str, str]  # all_classes | not_wheelchair | not_mobility_aid | standard_only
    agent_class_colors: dict[str, str]  # body class id -> colour (progress diagnostics)
    awaiting_refuge_ring_color: str
    refuge_fill: str  # fill color for refuge wait-zone areas
    refuge_edge: str
    sign_arrow: str
    sign_stair_box: str
    agent_outline: str
    presentation: PresentationConfig  # settings for the presentation layout renderer


def load_viz_config(path: str | Path = DEFAULT_PATH) -> VizConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    try:
        colors = data["colors"]
        category_colors = {str(k): str(v) for k, v in colors["category"].items()}
        door_color = str(colors["door"])
        exit_colors = {str(k): str(v) for k, v in colors["exit"].items()}
        assembly_point_color = str(colors["assembly_point"])
        hazard_color = str(colors["hazard"])
    except KeyError as e:
        raise VizConfigError(f"{path}: missing required key {e}") from e

    try:
        pd = data["progress_diagnostics"]
        progress_fig_width_px = int(pd["fig_width_px"])
        progress_fig_height_px = int(pd["fig_height_px"])
        progress_dpi = int(pd["dpi"])
        progress_table_fontsize_px = int(pd["table_fontsize_px"])
        progress_clip_seconds = float(pd["clip_seconds"])
        progress_clip_fps = int(pd["clip_fps"])
        class_shade_colors = {str(k): str(v) for k, v in pd["class_shade_colors"].items()}
        agent_class_colors = {str(k): str(v) for k, v in pd["agent_class_colors"].items()}
        awaiting_refuge_ring_color = str(pd["awaiting_refuge_ring_color"])
    except KeyError as e:
        raise VizConfigError(f"{path}: missing required progress_diagnostics key {e}") from e

    try:
        pr = data["presentation"]
        presentation = PresentationConfig(
            resolutions={str(k): (int(v[0]), int(v[1])) for k, v in pr["resolutions"].items()},
            preview_dpi=int(pr["preview_dpi"]), pitch_dpi=int(pr["pitch_dpi"]),
            min_text_px=float(pr["min_text_px"]), min_agent_px=float(pr["min_agent_px"]),
            target_px_per_m=float(pr["target_px_per_m"]), scale_bar_m=float(pr["scale_bar_m"]),
            title_fontsize_px=float(pr["title_fontsize_px"]), subtitle_fontsize_px=float(pr["subtitle_fontsize_px"]),
            clock_fontsize_px=float(pr["clock_fontsize_px"]), room_label_fontsize_px=float(pr["room_label_fontsize_px"]),
            room_labels_pitch_only=bool(pr["room_labels_pitch_only"]),
            exit_label_fontsize_px=float(pr["exit_label_fontsize_px"]), exit_bar_width_px=float(pr["exit_bar_width_px"]),
            hazard_inactive_alpha=float(pr["hazard_inactive_alpha"]),
            exit_bar_colors={str(k): str(v) for k, v in pr["exit_bar_colors"].items()},
        )
    except KeyError as e:
        raise VizConfigError(f"{path}: missing required presentation key {e}") from e

    return VizConfig(
        category_colors=category_colors,
        door_color=door_color,
        exit_colors=exit_colors,
        exit_markers={str(k): str(v) for k, v in data.get("exit_markers", {}).items()},
        exit_hatch={str(k): str(v) for k, v in data.get("hatch", {}).items()},
        assembly_point_color=assembly_point_color,
        hazard_color=hazard_color,
        hazard_room_tint=str(colors.get("hazard_room_tint", hazard_color)),
        obstacle_initials={str(k): str(v) for k, v in data.get("obstacle_initials", {}).items()},
        site_colors={str(k): str(v) for k, v in colors.get("site", {}).items()},
        agent_state_colors={str(k): str(v) for k, v in colors.get("agent_state", {}).items()},
        agent_pre_edge_color=str(colors.get("agent_pre_edge_color", "#555555")),
        progress_fig_width_px=progress_fig_width_px, progress_fig_height_px=progress_fig_height_px,
        progress_dpi=progress_dpi, progress_table_fontsize_px=progress_table_fontsize_px,
        progress_clip_seconds=progress_clip_seconds, progress_clip_fps=progress_clip_fps,
        class_shade_colors=class_shade_colors, agent_class_colors=agent_class_colors,
        awaiting_refuge_ring_color=awaiting_refuge_ring_color,
        refuge_fill=str(colors.get("refuge_fill", "#A7F3D0")),
        refuge_edge=str(colors.get("refuge_edge", "#047857")),
        sign_arrow=str(colors.get("sign_arrow", "#0E7490")),
        sign_stair_box=str(colors.get("sign_stair_box", "#334155")),
        agent_outline=str(colors.get("agent_outline", "#1f2933")),
        presentation=presentation,
    )
