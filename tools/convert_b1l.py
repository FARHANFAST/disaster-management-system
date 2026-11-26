"""One-off converter: two_storey_large.yaml's own schema -> dms/geometry's
loader schema (spec §1/§2).

Every structural difference handled here is pure field-renaming/reshaping
(array rect -> dict rect, wall+span -> wall+lo+hi, building-level stairs ->
per-stair landings, etc.) -- every id, coordinate, width and state value is
copied verbatim, never renamed or rounded. The five genuinely new concepts
(open_between_kinds, SetStairState, obstacle size override, count_range,
designed_dead_ends as FLOOR:SPACE pairs) are NOT converter tricks -- they
are real, generic dms/ features that this converter's
output relies on directly, unchanged.

Run once:

    python tools/convert_b1l.py two_storey_large.yaml configs/buildings/two_storey_large.yaml

The emitted YAML is committed as a static fixture (same convention as
three_storey_large.yaml) -- never regenerated at test time.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dms.geometry.primitives import Rect


def _rect_from_array(arr: list[float]) -> dict[str, float]:
    x0, y0, x1, y1 = arr
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}


def _split_doors(space: dict[str, Any]) -> tuple[list[dict], list[dict], list[dict]]:
    """A space's single `doors:` list, split by its own per-item `kind`
    (door | opening | fire_door) into (my doors:, my openings:, fire-door
    entries destined for the owning stair's building-level `fire_doors:`
    list) -- source `kind` is a different axis than my Door.kind, so this is
    a real reinterpretation, not a rename."""
    doors, openings, fire_doors = [], [], []
    for d in space.get("doors", []):
        axis_wall = d["wall"]
        lo, hi = d["span"]
        if d["kind"] == "opening":
            openings.append({"id": d["id"], "wall": axis_wall, "lo": lo, "hi": hi})
        elif d["kind"] == "fire_door" and space["kind"] == "stair":
            fire_doors.append({"id": d["id"], "wall": axis_wall, "lo": lo, "hi": hi})
        else:
            doors.append({"id": d["id"], "wall": axis_wall, "lo": lo, "hi": hi, "kind": "interior"})
    return doors, openings, fire_doors


def _convert_floor(
    fd: dict[str, Any], footprint_wh: list[float], stair_fire_doors: dict[tuple[str, str], list[dict]],
) -> dict[str, Any]:
    spaces_out = []
    for sp in fd["spaces"]:
        doors, openings, fire_doors = _split_doors(sp)
        out = {
            "id": sp["id"],
            "name": sp.get("name", sp["id"]),
            "kind": sp["kind"],
            "rect": _rect_from_array(sp["rect"]),
        }
        if sp.get("use"):
            out["use"] = sp["use"]
        if sp.get("restricted"):
            out["restricted"] = True
        if doors:
            out["doors"] = doors
        if openings:
            out["openings"] = openings
        spaces_out.append(out)
        if fire_doors:
            stair_fire_doors[(sp["id"], fd["id"])] = fire_doors
    return {
        "id": fd["id"],
        "z": fd["z_m"],
        "footprint": _rect_from_array([0.0, 0.0, *footprint_wh]),
        "spaces": spaces_out,
    }


def _convert_stairs(bld: dict[str, Any], stair_fire_doors: dict[tuple[str, str], list[dict]]) -> list[dict]:
    geo = bld["stair_geometry"]
    out = []
    for st in bld["stairs"]:
        entry: dict[str, Any] = {
            "id": st["id"],
            "landings": [{"floor": fid, "space": st["id"]} for fid in st["floors"]],
            "width_m": st["width_m"],
            "enclosed": st["enclosed"],
            "riser_m": geo["riser_m"],
            "tread_m": geo["tread_m"],
            "flights": geo["flights"],
            "chi": geo["chi"],
            "landing_depth_m": geo["landing_depth_m"],
        }
        if "discharge_exit" in st:
            entry["discharge_exit"] = st["discharge_exit"]
        fds = []
        for fid in st["floors"]:
            for fd_entry in stair_fire_doors.get((st["id"], fid), []):
                fds.append({"floor": fid, **fd_entry})
        if fds:
            entry["fire_doors"] = fds
        out.append(entry)
    return out


def _convert_hazards(hazards: list[dict]) -> tuple[dict[str, list[dict]], list[dict]]:
    """Split the top-level `hazards:` list into per-floor indoor hazards and
    site-level outdoor hazards (floor: null -> site.hazards)."""
    by_floor: dict[str, list[dict]] = {}
    site_hazards: list[dict] = []
    for h in hazards:
        if h.get("floor") is None:
            site_hazards.append({
                "id": h["id"], "species": h.get("species", "H2S"), "mw": h.get("mw", 34.08),
                "x": h["site_XY"][0], "y": h["site_XY"][1], "z": h.get("z", 0.0),
                "purpose": h.get("note", ""),
            })
        else:
            by_floor.setdefault(h["floor"], []).append({
                "id": h["id"], "species": h.get("species", "H2S"), "mw": h.get("mw", 34.08),
                "space": h["space"], "x": h["xy"][0], "y": h["xy"][1], "z": h.get("z", 0.0),
                "purpose": h.get("note", ""),
            })
    return by_floor, site_hazards


def _resolve_owning_space(floors_src: list[dict], floor_id: str, cx: float, cy: float) -> str | None:
    for fd in floors_src:
        if fd["id"] != floor_id:
            continue
        for sp in fd["spaces"]:
            r = Rect(*sp["rect"])
            if r.contains_point(cx, cy):
                return sp["id"]
    return None


def _convert_obstacles(
    obstacles: list[dict], floors_src: list[dict], default_off_groups: set[str],
) -> dict[str, list[dict]]:
    """Every B1-L obstacle instance -> my `clutter:` floor-level schema
    (a strict superset of the room-nested `obstacles:` schema, so it works
    regardless of the resolved active-by-default state). `group` not in
    `default_off_groups` (furniture, and B1-L's "corridor"-group fixtures,
    which no scenario ever toggles) maps to layer="furniture" (always on,
    never a toggle target, matching how nothing in scenarios.py ever flips
    a "corridor" group); a default-off group keeps layer="clutter" with its
    own group name as the toggle key."""
    by_floor: dict[str, list[dict]] = {}
    for o in obstacles:
        space_id = _resolve_owning_space(floors_src, o["floor"], o["center"][0], o["center"][1])
        group = o["group"]
        layer = "furniture" if group not in default_off_groups else "clutter"
        entry = {
            "id": f"{o['floor']}-obs-{len(by_floor.get(o['floor'], []))}",
            "space": space_id,
            "type": o["type"],
            "x": o["center"][0],
            "y": o["center"][1],
            "rotation_deg": o.get("rotation_deg", 0.0),
            "layer": layer,
            "group": group,
            "active_by_default": group not in default_off_groups,
        }
        if "size" in o:
            entry["size"] = o["size"]
        by_floor.setdefault(o["floor"], []).append(entry)
    return by_floor


def _invert_occupancy_profiles(src_profiles: dict[str, dict]) -> dict[str, dict]:
    """{period: {use: leaf}} -> {use: {period: leaf}}, renaming `per_m2` ->
    `density_per_m2` at the leaf level."""
    out: dict[str, dict] = {}
    for period, by_use in src_profiles.items():
        for use, leaf in by_use.items():
            renamed = {("density_per_m2" if k == "per_m2" else k): v for k, v in leaf.items()}
            out.setdefault(use, {})[period] = renamed
    return out


def _convert_site(site_src: dict[str, Any], building_name: str, site_hazards: list[dict]) -> dict[str, Any]:
    site_obstacles = []
    for o in site_src.get("obstacles", []):
        if "rect_XY" in o:
            x0, y0, x1, y1 = o["rect_XY"]
        else:
            cx, cy = o["circle_XY"]
            r = o["radius_m"]
            x0, y0, x1, y1 = cx - r, cy - r, cx + r, cy + r  # bounding-square approximation of a circle
        site_obstacles.append({"id": o["id"], "x0": x0, "y0": y0, "x1": x1, "y1": y1, "layer": "static"})
    for p in site_src.get("parking", []):
        x0, y0, x1, y1 = p["rect_XY"]
        site_obstacles.append({
            "id": p["id"], "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "layer": "parking", "active_by_default": False,
        })
    return {
        "name": f"{building_name} site",
        "width_m": site_src["size_m"][0],
        "height_m": site_src["size_m"][1],
        "assembly_points": [
            {"id": ap["id"], "x": ap["XY"][0], "y": ap["XY"][1], "diameter_m": ap["diameter_m"]}
            for ap in site_src["assembly_points"]
        ],
        "obstacles": site_obstacles,
        "hazards": site_hazards,
    }


def convert(src: dict[str, Any]) -> dict[str, Any]:
    bld = src["building"]
    footprint_wh = list(bld["footprint_m"])

    stair_fire_doors: dict[tuple[str, str], list[dict]] = {}
    floors_out = [_convert_floor(fd, footprint_wh, stair_fire_doors) for fd in bld["floors"]]
    stairs_out = _convert_stairs(bld, stair_fire_doors)

    hazards_by_floor, site_hazards = _convert_hazards(src.get("hazards", []))
    for floor_out in floors_out:
        floor_out["hazards"] = hazards_by_floor.get(floor_out["id"], [])

    obstacles_by_floor = _convert_obstacles(
        src.get("obstacles", []), bld["floors"], set(src.get("default_off_groups", [])),
    )
    for floor_out in floors_out:
        floor_out["clutter"] = obstacles_by_floor.get(floor_out["id"], [])

    # exits are a building-level list in the source; every one of B1-L's own
    # exits references a GF space, so (matching two_storey/three_storey's
    # own convention of ground-floor-only exits) they all land on GF here.
    exits_by_floor: dict[str, list[dict]] = {fd["id"]: [] for fd in bld["floors"]}
    gf_space_ids = {sp["id"] for sp in bld["floors"][0]["spaces"]}
    for e in bld["exits"]:
        target_floor = bld["floors"][0]["id"] if e["space"] in gf_space_ids else bld["floors"][-1]["id"]
        entry = {
            "id": e["id"], "space": e["space"], "wall": e["wall"], "lo": e["span"][0], "hi": e["span"][1],
        }
        state = e.get("default_state", "AVAILABLE")
        entry["default_state"] = "UNAVAILABLE" if state == "LOCKED" else state
        exits_by_floor[target_floor].append(entry)
    for floor_out in floors_out:
        floor_out["exits"] = exits_by_floor.get(floor_out["id"], [])

    obstacle_types = {name: {"width_m": wh[0], "height_m": wh[1]} for name, wh in src["obstacle_types"].items()}

    building_out: dict[str, Any] = {
        "id": bld["id"],
        "origin": {"x": src["site"]["buildings"][0]["origin_XY"][0], "y": src["site"]["buildings"][0]["origin_XY"][1]},
        "rotation_deg": src["site"]["buildings"][0].get("psi_deg", 0.0),
        "wall_thickness_m": bld["wall_thickness_m"],
        "floor_to_floor_m": bld["floor_to_floor_m"],
        "designed_dead_ends": list(bld["designed_dead_ends"]),
        "open_between_kinds": [list(p) for p in bld["open_between_kinds"]],
        "obstacle_types": obstacle_types,
        "blocking_groups": list(src.get("blocking_groups", [])),
        "window_rule": {
            "width_m": bld["windows"]["width_m"], "spacing_m": bld["windows"]["spacing_m"],
            "sill_height_m": bld["windows"]["sill_m"],
            "perm_closed": bld["windows"]["perm_closed"], "perm_open": bld["windows"]["perm_open"],
        },
        "stairs": stairs_out,
        "risers": [
            {"id": r["id"], "a": r["a"], "b": r["b"], "coefficient": r.get("coefficient", 1.0)}
            for r in bld.get("risers", [])
        ],
        # Source nesting is {period: {use: leaf}} (day/lunch/meeting/night at
        # the outer level); mine is {use: {period: leaf}} -- inverted here,
        # not just renamed.
        "occupancy_profiles": _invert_occupancy_profiles(src["occupancy_profiles"]),
        "persona_mix": {
            pid: {"share": v["share"], "v0": v["v0"]} for pid, v in src["persona_mix"].items()
        },
        "floors": floors_out,
    }

    out = {
        "site": _convert_site(src["site"], bld.get("name", bld["id"]), site_hazards),
        "buildings": [building_out],
        "scenarios": {
            name: {"events": [{k: v for k, v in ev.items() if k != "t"} for ev in events]}
            for name, events in src.get("scenarios", {}).items()
        },
    }
    # normalize every event's own key: source uses "event", mine uses "type"
    for preset in out["scenarios"].values():
        for ev in preset["events"]:
            if "event" in ev:
                ev["type"] = ev.pop("event")
    return out


def main() -> None:
    if len(sys.argv) != 3:
        print("usage: python tools/convert_b1l.py <source.yaml> <dest.yaml>", file=sys.stderr)
        raise SystemExit(2)
    src_path, dst_path = Path(sys.argv[1]), Path(sys.argv[2])
    with src_path.open(encoding="utf-8") as f:
        src = yaml.safe_load(f)
    out = convert(src)
    header = (
        "# Large two-storey building fixture, converted from the attached\n"
        "# two_storey_large.yaml (B1-L) by tools/convert_b1l.py. Every id/coordinate/\n"
        "# width/state value is copied verbatim from the source -- only the YAML's\n"
        "# structural shape changed (array rects, wall+span doors, building-level\n"
        "# stairs, etc. -> this loader's own schema). This file is the static,\n"
        "# authoritative fixture -- it is not regenerated at test time.\n\n"
    )
    text = header + yaml.dump(out, sort_keys=False, default_flow_style=None, width=120, allow_unicode=True)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    dst_path.write_text(text, encoding="utf-8")
    print(f"wrote {dst_path}")


if __name__ == "__main__":
    main()
