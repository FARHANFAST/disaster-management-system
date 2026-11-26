"""YAML -> Site. Auto-generates wall boundaries (as the complement of declared
doors/openings), resolves which spaces each door/opening touches by pure
geometry, and generates windows by rule (spec §3.6). No coordinates live in
this module — every number comes from the YAML file.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .building import Building, Floor, Space
from .primitives import (
    Door,
    HazardSource,
    Obstacle,
    Opening,
    Rect,
    Refuge,
    Riser,
    Sign,
    Stair,
    Window,
)
from .site import (
    AssemblyPoint,
    Fence,
    Gate,
    Site,
    SiteObstacle,
    Walkway,
    WeatherStation,
)

TOL = 1e-6


class LoaderError(ValueError):
    """Raised when the YAML is structurally inconsistent (not a soft geometric issue)."""


def load_site(path: str | Path) -> Site:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return _build_site(data)


# --------------------------------------------------------------------------- primitives

def _rect(d: dict[str, Any]) -> Rect:
    return Rect(float(d["x0"]), float(d["y0"]), float(d["x1"]), float(d["y1"]))


def _obstacle_rect(d: dict[str, Any], obstacle_types: dict[str, tuple[float, float]]) -> Rect:
    """Either an explicit x0/y0/x1/y1 rect (legacy / hand-placed), or a
    catalog `type` + centre + rotation_deg (spec §4.2) -- resolved
    HERE, once, into the axis-aligned bounding box of the rotated footprint,
    so no downstream consumer needs to know about rotation at all. An
    optional `size: [L, W]` (spec §2) overrides the catalog's own size
    for this one instance -- e.g. a maintenance barrier sized to the exact
    corridor it blocks -- while `type` is kept for display/grouping."""
    if "x0" in d:
        return _rect(d)
    otype = d["type"]
    if otype not in obstacle_types:
        raise LoaderError(f"obstacle {d.get('id', '?')}: unknown obstacle type {otype!r} (not in obstacle_types:)")
    if "size" in d:
        w, h = float(d["size"][0]), float(d["size"][1])
    else:
        w, h = obstacle_types[otype]
    cx, cy = float(d["x"]), float(d["y"])
    rot = math.radians(float(d.get("rotation_deg", 0.0)))
    hw, hh = w / 2.0, h / 2.0
    cos_r, sin_r = math.cos(rot), math.sin(rot)
    xs, ys = [], []
    for lx, ly in ((-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)):
        xs.append(cx + lx * cos_r - ly * sin_r)
        ys.append(cy + lx * sin_r + ly * cos_r)
    return Rect(min(xs), min(ys), max(xs), max(ys))


def _axis_coord(wall: str) -> tuple[str, float]:
    axis, _, coord = wall.partition("=")
    axis = axis.strip()
    if axis not in ("x", "y"):
        raise LoaderError(f"bad wall spec {wall!r}")
    return axis, float(coord)


def _find_neighbors(
    spaces: dict[str, Space], axis: str, coord: float, lo: float, hi: float, *, exclude: frozenset[str] = frozenset()
) -> list[str]:
    hits = []
    for sid, sp in spaces.items():
        if sid in exclude:
            continue
        seg = sp.rect.edge_segment(axis, coord, tol=TOL)
        if seg is None:
            continue
        slo, shi = seg
        if slo - TOL <= lo and hi <= shi + TOL:
            hits.append(sid)
    return hits


def _resolve_owned(spaces: dict[str, Space], owner: str, axis: str, coord: float, lo: float, hi: float, *, what: str) -> str:
    hits = _find_neighbors(spaces, axis, coord, lo, hi, exclude=frozenset({owner}))
    if len(hits) != 1:
        raise LoaderError(
            f"{what} on {owner} at {axis}={coord} [{lo},{hi}] resolved to {len(hits)} "
            f"neighbours (expected 1): {hits}"
        )
    return hits[0]


def _resolve_pair(spaces: dict[str, Space], axis: str, coord: float, lo: float, hi: float, *, what: str) -> tuple[str, str]:
    hits = _find_neighbors(spaces, axis, coord, lo, hi)
    if len(hits) != 2:
        raise LoaderError(
            f"{what} at {axis}={coord} [{lo},{hi}] resolved to {len(hits)} spaces "
            f"(expected 2): {hits}"
        )
    return hits[0], hits[1]


# --------------------------------------------------------------------------- windows

def _spans_on_wall(floor: Floor, axis: str, coord: float) -> list[tuple[float, float]]:
    spans = []
    for d in floor.doors.values():
        if d.axis == axis and abs(d.coord - coord) <= TOL:
            spans.append((d.lo, d.hi))
    for o in floor.openings.values():
        if o.axis == axis and abs(o.coord - coord) <= TOL:
            spans.append((o.lo, o.hi))
    return spans


def _subtract_spans(lo: float, hi: float, blocked: list[tuple[float, float]]) -> list[tuple[float, float]]:
    free = [(lo, hi)]
    for blo, bhi in sorted(blocked):
        next_free = []
        for flo, fhi in free:
            if bhi <= flo or blo >= fhi:
                next_free.append((flo, fhi))
                continue
            if blo > flo:
                next_free.append((flo, blo))
            if bhi < fhi:
                next_free.append((bhi, fhi))
        free = next_free
    return free


def _generate_windows(floor: Floor, rule: dict[str, Any]) -> None:
    width = float(rule.get("width_m", 1.5))
    spacing = float(rule.get("spacing_m", 3.0))
    sill = float(rule.get("sill_height_m", 0.9))
    perm_closed = float(rule.get("perm_closed", 0.02))
    perm_open = float(rule.get("perm_open", 0.5))
    fp = floor.footprint
    counter = 0

    for sp in list(floor.spaces.values()):
        if sp.kind != "room":
            continue
        r = sp.rect
        edges = [
            ("x", r.x0),
            ("x", r.x1),
            ("y", r.y0),
            ("y", r.y1),
        ]
        for axis, coord in edges:
            on_boundary = (
                axis == "x" and (abs(coord - fp.x0) <= TOL or abs(coord - fp.x1) <= TOL)
            ) or (axis == "y" and (abs(coord - fp.y0) <= TOL or abs(coord - fp.y1) <= TOL))
            if not on_boundary:
                continue
            lo, hi = r.edge_segment(axis, coord, tol=TOL)
            blocked = _spans_on_wall(floor, axis, coord)
            for flo, fhi in _subtract_spans(lo, hi, blocked):
                span_len = fhi - flo
                if span_len < width:
                    continue
                pitch = width + spacing
                n = max(1, int((span_len + spacing) // pitch))
                used = n * width + (n - 1) * spacing
                start = flo + (span_len - used) / 2.0
                for i in range(n):
                    wlo = start + i * pitch
                    whi = wlo + width
                    counter += 1
                    wid = f"{sp.id}-win-{counter}"
                    floor.windows[wid] = Window(
                        id=wid,
                        floor=floor.id,
                        space_id=sp.id,
                        axis=axis,
                        coord=coord,
                        lo=wlo,
                        hi=whi,
                        sill_height_m=sill,
                        perm_closed=perm_closed,
                        perm_open=perm_open,
                    )


# --------------------------------------------------------------------------- floor / building

def _build_floor(building_id: str, fd: dict[str, Any], obstacle_types: dict[str, tuple[float, float]]) -> Floor:
    floor = Floor(id=fd["id"], z=float(fd["z"]), footprint=_rect(fd["footprint"]))

    for sdata in fd.get("spaces", []):
        sp = Space(
            id=sdata["id"],
            floor=floor.id,
            name=sdata.get("name", sdata["id"]),
            kind=sdata["kind"],
            rect=_rect(sdata["rect"]),
            notes=sdata.get("notes", ""),
            use=sdata.get("use"),
            restricted=bool(sdata.get("restricted", False)),
        )
        if sp.id in floor.spaces:
            raise LoaderError(f"duplicate space id {sp.id!r} on floor {floor.id}")
        floor.spaces[sp.id] = sp

    # Interior doors and openings declared under their owning space.
    for sdata in fd.get("spaces", []):
        owner = sdata["id"]
        for ddata in sdata.get("doors", []):
            axis, coord = _axis_coord(ddata["wall"])
            lo, hi = float(ddata["lo"]), float(ddata["hi"])
            other = _resolve_owned(floor.spaces, owner, axis, coord, lo, hi, what=f"door {ddata['id']}")
            door = Door(
                id=ddata["id"],
                floor=floor.id,
                axis=axis,
                coord=coord,
                lo=lo,
                hi=hi,
                kind=ddata.get("kind", "interior"),
                default_state=ddata.get("default_state", "AVAILABLE"),
                space_a=owner,
                space_b=other,
            )
            if door.id in floor.doors:
                raise LoaderError(f"duplicate door id {door.id!r} on floor {floor.id}")
            floor.doors[door.id] = door

        for odata in sdata.get("openings", []):
            axis, coord = _axis_coord(odata["wall"])
            lo, hi = float(odata["lo"]), float(odata["hi"])
            other = _resolve_owned(floor.spaces, owner, axis, coord, lo, hi, what=f"opening {odata['id']}")
            opening = Opening(
                id=odata["id"], floor=floor.id, axis=axis, coord=coord, lo=lo, hi=hi,
                space_a=owner, space_b=other,
            )
            if opening.id in floor.openings:
                raise LoaderError(f"duplicate opening id {opening.id!r} on floor {floor.id}")
            floor.openings[opening.id] = opening

        for obdata in sdata.get("obstacles", []):
            obs = Obstacle(
                id=obdata["id"],
                floor=floor.id,
                rect=_obstacle_rect(obdata, obstacle_types),
                space_id=owner,
                layer=obdata.get("layer", "furniture"),
                active_by_default=obdata.get("active_by_default", True),
                narrows_to_width_m=obdata.get("narrows_to_width_m"),
                type=obdata.get("type"),
                rotation_deg=float(obdata.get("rotation_deg", 0.0)),
                group=obdata.get("group", "clutter"),
            )
            floor.obstacles[obs.id] = obs

    # Floor-level corridor<->corridor joins (owned by no single room).
    for cdata in fd.get("corridor_openings", []):
        axis, coord = _axis_coord(cdata["wall"])
        lo, hi = float(cdata["lo"]), float(cdata["hi"])
        a, b = _resolve_pair(floor.spaces, axis, coord, lo, hi, what=f"corridor_opening {cdata['id']}")
        opening = Opening(id=cdata["id"], floor=floor.id, axis=axis, coord=coord, lo=lo, hi=hi, space_a=a, space_b=b)
        if opening.id in floor.openings:
            raise LoaderError(f"duplicate opening id {opening.id!r} on floor {floor.id}")
        floor.openings[opening.id] = opening

    # Exits: explicit owning space, other side is "outside" by definition.
    for xdata in fd.get("exits", []):
        axis, coord = _axis_coord(xdata["wall"])
        lo, hi = float(xdata["lo"]), float(xdata["hi"])
        owner = xdata["space"]
        if owner not in floor.spaces:
            raise LoaderError(f"exit {xdata['id']} references unknown space {owner!r}")
        seg = floor.spaces[owner].rect.edge_segment(axis, coord, tol=TOL)
        if seg is None:
            raise LoaderError(f"exit {xdata['id']}: space {owner!r} has no edge at {axis}={coord}")
        on_boundary = (
            axis == "x" and (abs(coord - floor.footprint.x0) <= TOL or abs(coord - floor.footprint.x1) <= TOL)
        ) or (
            axis == "y" and (abs(coord - floor.footprint.y0) <= TOL or abs(coord - floor.footprint.y1) <= TOL)
        )
        if not on_boundary:
            raise LoaderError(f"exit {xdata['id']} is not on the floor's exterior envelope")
        door = Door(
            id=xdata["id"], floor=floor.id, axis=axis, coord=coord, lo=lo, hi=hi,
            kind="exit", default_state=xdata.get("default_state", "AVAILABLE"),
            space_a=owner, space_b="outside",
        )
        if door.id in floor.doors:
            raise LoaderError(f"duplicate door id {door.id!r} on floor {floor.id}")
        floor.doors[door.id] = door

    for hdata in fd.get("hazards", []):
        hz = HazardSource(
            id=hdata["id"],
            species=hdata.get("species", "H2S"),
            mw=float(hdata.get("mw", 34.08)),
            building=building_id,
            floor=floor.id,
            space_id=hdata.get("space"),
            x=float(hdata["x"]), y=float(hdata["y"]), z=float(hdata.get("z", 0.0)),
            Q_gps=hdata.get("Q_gps"), H_m=hdata.get("H_m"), u_mps=hdata.get("u_mps"),
            purpose=hdata.get("purpose", ""),
        )
        floor.hazards[hz.id] = hz

    for rdata in fd.get("refuges", []):
        refuge = Refuge(
            id=rdata["id"], floor=floor.id, rect=_rect(rdata["rect"]),
            capacity=int(rdata.get("capacity", 2)), serves_stair=rdata.get("serves_stair"),
        )
        if refuge.id in floor.refuges:
            raise LoaderError(f"duplicate refuge id {refuge.id!r} on floor {floor.id}")
        floor.refuges[refuge.id] = refuge

    for sgdata in fd.get("signs", []):
        heading_deg = float(sgdata.get("heading_deg", 0.0))
        sign = Sign(
            id=sgdata["id"], floor=floor.id, x=float(sgdata["x"]), y=float(sgdata["y"]),
            points_to=sgdata["points_to"], heading_deg=heading_deg,
            reason=sgdata.get("reason", "manual"),
            sign_type=sgdata.get("sign_type", "directional"),
            facing_deg=float(sgdata.get("facing_deg", (heading_deg + 180.0) % 360.0)),
            mount_height_m=float(sgdata.get("mount_height_m", 2.1)),
            illuminated=bool(sgdata.get("illuminated", True)),
        )
        if sign.id in floor.signs:
            raise LoaderError(f"duplicate sign id {sign.id!r} on floor {floor.id}")
        floor.signs[sign.id] = sign

    for cdata in fd.get("clutter", []):
        obs = Obstacle(
            id=cdata["id"], floor=floor.id, rect=_obstacle_rect(cdata, obstacle_types), space_id=cdata.get("space"),
            layer=cdata.get("layer", "clutter"), active_by_default=cdata.get("active_by_default", False),
            narrows_to_width_m=cdata.get("narrows_to_width_m"),
            type=cdata.get("type"), rotation_deg=float(cdata.get("rotation_deg", 0.0)),
            group=cdata.get("group", "clutter"),
        )
        floor.obstacles[obs.id] = obs

    return floor


def _build_stairs(building: Building, bdata: dict[str, Any]) -> None:
    """A stair's `landings:` list is floor-id -> space-id pairs (spec
    §8) -- any 2+ floors, not necessarily every floor in the building, so a
    stair may span only a subset (e.g. floors 1-2 of a taller building)."""
    f2f = float(bdata.get("floor_to_floor_m", 4.0))

    for sdata in bdata.get("stairs", []):
        landings: dict[str, str] = {}
        footprints: dict[str, Rect] = {}
        for ldata in sdata["landings"]:
            floor_id, space_id = ldata["floor"], ldata["space"]
            floor = building.get_floor(floor_id)
            if space_id not in floor.spaces:
                raise LoaderError(f"stair {sdata['id']}: unknown space {space_id!r} on floor {floor_id!r}")
            landings[floor_id] = space_id
            footprints[floor_id] = floor.spaces[space_id].rect
        if len(landings) < 2:
            raise LoaderError(f"stair {sdata['id']}: needs landings on at least 2 floors, got {list(landings)}")

        fire_door_ids: list[str] = []
        for fdata in sdata.get("fire_doors", []):
            floor_id = fdata["floor"]
            floor = building.get_floor(floor_id)
            owner = landings[floor_id]
            axis, coord = _axis_coord(fdata["wall"])
            lo, hi = float(fdata["lo"]), float(fdata["hi"])
            other = _resolve_owned(floor.spaces, owner, axis, coord, lo, hi, what=f"fire door of {sdata['id']}")
            fdid = fdata.get("id", f"{sdata['id']}-fd-{floor_id}")
            door = Door(
                id=fdid, floor=floor_id, axis=axis, coord=coord, lo=lo, hi=hi,
                kind="fire_door", default_state="AVAILABLE", leaf="closed",
                space_a=owner, space_b=other,
            )
            if door.id in floor.doors:
                raise LoaderError(f"duplicate door id {door.id!r} on floor {floor_id}")
            floor.doors[door.id] = door
            fire_door_ids.append(fdid)

        stair = Stair(
            id=sdata["id"],
            landings=landings,
            width_m=float(sdata["width_m"]),
            riser_m=float(sdata.get("riser_m", 0.17)),
            tread_m=float(sdata.get("tread_m", 0.28)),
            floor_to_floor_m=f2f,
            enclosed=bool(sdata.get("enclosed", True)),
            footprints=footprints,
            flights=int(sdata.get("flights", 2)),
            chi=float(sdata.get("chi", 0.70)),
            fire_door_id=fire_door_ids[0] if len(fire_door_ids) == 1 else (fire_door_ids or None),
            discharge_exit_id=sdata.get("discharge_exit"),
            landing_depth_m=(float(sdata["landing_depth_m"]) if "landing_depth_m" in sdata else None),
        )
        building.stairs[stair.id] = stair


def _build_persona_mix(raw: dict[str, Any]) -> dict[str, dict]:
    """Normalize every entry to `{"share": float, "v0": float | None,
    "body_class_mix": dict[str, float] | None, "rank": str | None,
    "familiarity": str | None}` (spec §2): a bare
    float (two_storey/three_storey's existing shape) is share-only, every
    other field `None`/absent (falls back to
    configs/agent_defaults.yaml's `personas` catalog speed and
    dms/scenario/population.py's DEFAULT_BODY_CLASS_MIX -- "100% standard"
    -- at the point each is resolved); a `{share, v0, body_class_mix, rank, familiarity}` dict (a
    building's own explicit data) is kept exactly as declared, never
    dropped or assumed to match a Python-side default. `rank`/
    `familiarity` are data only -- nothing in dms/ consumes them yet."""
    out = {}
    for pid, v in raw.items():
        if isinstance(v, dict):
            body_class_mix = v.get("body_class_mix")
            out[pid] = {
                "share": float(v["share"]), "v0": (float(v["v0"]) if "v0" in v else None),
                "body_class_mix": ({str(k): float(s) for k, s in body_class_mix.items()}
                                    if body_class_mix is not None else None),
                "rank": v.get("rank"), "familiarity": v.get("familiarity"),
            }
        else:
            out[pid] = {"share": float(v), "v0": None, "body_class_mix": None, "rank": None, "familiarity": None}
    return out


def _generate_open_between_kinds_openings(floor: Floor, open_between_kinds: list[tuple[str, str]]) -> None:
    """spec §2: boundaries between two circulation spaces of the
    declared `(kind, kind)` pairs have no wall, without needing every such
    junction listed as an explicit `opening:` (unlike the manual
    `corridor_openings:` this replaces for buildings that declare it). Runs
    after doors/openings/corridor_openings are parsed, so an already-covered
    sub-span (an explicit door/opening between the same two spaces) is
    subtracted first, same pattern as window generation's `_subtract_spans`."""
    if not open_between_kinds:
        return
    pairs = {frozenset(p) for p in open_between_kinds}
    spaces = list(floor.spaces.values())
    counter = 0
    for i in range(len(spaces)):
        for j in range(i + 1, len(spaces)):
            a, b = spaces[i], spaces[j]
            if frozenset((a.kind, b.kind)) not in pairs:
                continue
            edge = a.rect.shared_edge(b.rect, tol=TOL)
            if edge is None:
                continue
            axis, coord, lo, hi = edge
            covered: list[tuple[float, float]] = []
            for d in floor.doors.values():
                if d.axis == axis and abs(d.coord - coord) <= TOL and {d.space_a, d.space_b} == {a.id, b.id}:
                    covered.append((d.lo, d.hi))
            for o in floor.openings.values():
                if o.axis == axis and abs(o.coord - coord) <= TOL and {o.space_a, o.space_b} == {a.id, b.id}:
                    covered.append((o.lo, o.hi))
            for flo, fhi in _subtract_spans(lo, hi, covered):
                counter += 1
                oid = f"{floor.id}:open:{a.id}|{b.id}:{counter}"
                floor.openings[oid] = Opening(
                    id=oid, floor=floor.id, axis=axis, coord=coord, lo=flo, hi=fhi, space_a=a.id, space_b=b.id,
                )


def _build_building(bdata: dict[str, Any]) -> Building:
    origin = (float(bdata["origin"]["x"]), float(bdata["origin"]["y"]))
    obstacle_types = {
        name: (float(size["width_m"]), float(size["height_m"]))
        for name, size in bdata.get("obstacle_types", {}).items()
    }
    building = Building(
        id=bdata["id"],
        origin=origin,
        rotation_deg=float(bdata.get("rotation_deg", 0.0)),
        wall_thickness_m=float(bdata.get("wall_thickness_m", 0.2)),
        designed_dead_ends=list(bdata.get("designed_dead_ends", [])),
        exit_sets={k: (list(v) if v is not None else None) for k, v in bdata.get("exit_sets", {}).items()},
        obstacle_types=obstacle_types,
        blocking_groups=list(bdata.get("blocking_groups", [])),
        occupancy_profiles={k: dict(v) for k, v in bdata.get("occupancy_profiles", {}).items()},
        persona_mix=_build_persona_mix(bdata.get("persona_mix", {})),
        open_between_kinds=[tuple(p) for p in bdata.get("open_between_kinds", [])],
    )

    for fd in bdata.get("floors", []):
        floor = _build_floor(building.id, fd, obstacle_types)
        _generate_open_between_kinds_openings(floor, building.open_between_kinds)
        building.floors[floor.id] = floor

    window_rule = bdata.get("window_rule", {})
    for floor in building.floors.values():
        _generate_windows(floor, window_rule)

    _build_stairs(building, bdata)
    _build_risers(building, bdata)

    # Refuges are generated by rule for every
    # enclosed stair on every upper floor of ANY building -- this is cheap
    # (pure geometry checks, no grid/field solve) so it runs unconditionally
    # here, unlike sign generation (which needs a routing-field solve and
    # stays an explicit, separate step -- dms/geometry/signage_gen.py). A
    # YAML-declared refuge at the generated id overrides it; the loader has
    # already set floor.refuges from the YAML by this point.
    from .refuge_gen import apply_generated_refuges
    apply_generated_refuges(building)

    return building


def _build_risers(building: Building, bdata: dict[str, Any]) -> None:
    """spec §7: explicit vertical gas couplings between two named
    (floor, space) endpoints -- possibly non-adjacent floors, unlike the
    void-column/stair-landing couplings discovered automatically from the
    geometry (dms/hazards/topology.py)."""
    for rdata in bdata.get("risers", []):
        a, b = rdata["a"], rdata["b"]
        for floor_id, space_id in ((a["floor"], a["space"]), (b["floor"], b["space"])):
            floor = building.get_floor(floor_id)
            if space_id not in floor.spaces:
                raise LoaderError(f"riser {rdata['id']}: unknown space {space_id!r} on floor {floor_id!r}")
        riser = Riser(
            id=rdata["id"], floor_a=a["floor"], space_a=a["space"], floor_b=b["floor"], space_b=b["space"],
            coefficient=float(rdata.get("coefficient", 1.0)),
        )
        building.risers.append(riser)


def _build_site(data: dict[str, Any]) -> Site:
    sdata = data["site"]
    site = Site(name=sdata.get("name", "site"), width_m=float(sdata["width_m"]), height_m=float(sdata["height_m"]))

    for apdata in sdata.get("assembly_points", []):
        ap = AssemblyPoint(
            id=apdata["id"], x=float(apdata["x"]), y=float(apdata["y"]),
            diameter_m=float(apdata.get("diameter_m", 15.0)),
        )
        site.assembly_points[ap.id] = ap

    for odata in sdata.get("obstacles", []):
        obs = SiteObstacle(
            id=odata["id"], rect=_rect(odata), layer=odata.get("layer", "static"),
            active_by_default=odata.get("active_by_default", True), type=odata.get("type"),
        )
        site.obstacles[obs.id] = obs

    for sgdata in sdata.get("structures", []):
        obs = SiteObstacle(
            id=sgdata["id"], rect=_rect(sgdata), layer="structure",
            active_by_default=sgdata.get("active_by_default", True), type=sgdata.get("type"),
        )
        site.obstacles[obs.id] = obs

    for hdata in sdata.get("hazards", []):
        hz = HazardSource(
            id=hdata["id"], species=hdata.get("species", "H2S"), mw=float(hdata.get("mw", 34.08)),
            x=float(hdata["x"]), y=float(hdata["y"]), z=float(hdata.get("z", 0.0)),
            purpose=hdata.get("purpose", ""),
        )
        site.hazards[hz.id] = hz

    for wdata in sdata.get("walkways", []):
        wk = Walkway(
            id=wdata["id"], points=tuple((float(p[0]), float(p[1])) for p in wdata["points"]),
            width_m=float(wdata.get("width_m", 3.0)),
        )
        site.walkways[wk.id] = wk

    for fdata in sdata.get("fences", []):
        fc = Fence(id=fdata["id"], points=tuple((float(p[0]), float(p[1])) for p in fdata["points"]))
        site.fences[fc.id] = fc

    for gdata in sdata.get("gates", []):
        gt = Gate(
            id=gdata["id"], fence_id=gdata["fence_id"], x=float(gdata["x"]), y=float(gdata["y"]),
            width_m=float(gdata.get("width_m", 4.0)), default_state=gdata.get("default_state", "AVAILABLE"),
        )
        site.gates[gt.id] = gt

    for sgdata in sdata.get("weather_stations", []):
        ws = WeatherStation(
            id=sgdata["id"], x=float(sgdata["x"]), y=float(sgdata["y"]),
            wind_from_deg=(float(sgdata["wind_from_deg"]) if sgdata.get("wind_from_deg") is not None else None),
        )
        site.weather_stations[ws.id] = ws

    for bdata in data.get("buildings", []):
        building = _build_building(bdata)
        site.buildings[building.id] = building

    site.scenarios = dict(data.get("scenarios", {}))

    # Size every assembly point up to hold the full
    # design population of every building on the site (never shrinks a
    # YAML value that's already big enough) -- cheap (no field solve),
    # so it runs unconditionally here, same reasoning as refuge
    # generation in _build_building.
    from .ap_sizing import apply_ap_sizing
    apply_ap_sizing(site)

    return site
