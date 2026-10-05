"""YAML -> Site. Auto-generates wall boundaries (as the complement of declared
doors/openings), resolves which spaces each door/opening touches by pure
geometry, and generates windows by rule (spec §3.6). No coordinates live in
this module — every number comes from the YAML file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .building import Building, Floor, Space
from .primitives import Door, HazardSource, Obstacle, Opening, Rect, Stair, Window
from .site import AssemblyPoint, Site, SiteObstacle

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

def _build_floor(building_id: str, fd: dict[str, Any]) -> Floor:
    floor = Floor(id=fd["id"], z=float(fd["z"]), footprint=_rect(fd["footprint"]))

    for sdata in fd.get("spaces", []):
        sp = Space(
            id=sdata["id"],
            floor=floor.id,
            name=sdata.get("name", sdata["id"]),
            kind=sdata["kind"],
            rect=_rect(sdata["rect"]),
            notes=sdata.get("notes", ""),
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
                rect=_rect(obdata),
                space_id=owner,
                layer=obdata.get("layer", "furniture"),
                active_by_default=obdata.get("active_by_default", True),
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

    for cdata in fd.get("clutter", []):
        obs = Obstacle(
            id=cdata["id"], floor=floor.id, rect=_rect(cdata), space_id=cdata.get("space"),
            layer="clutter", active_by_default=cdata.get("active_by_default", False),
            narrows_to_width_m=cdata.get("narrows_to_width_m"),
        )
        floor.obstacles[obs.id] = obs

    return floor


def _build_stairs(building: Building, bdata: dict[str, Any]) -> None:
    f2f = float(bdata.get("floor_to_floor_m", 4.0))
    floor_ids = building.floor_ids()
    if len(floor_ids) != 2:
        raise LoaderError(f"building {building.id}: stairs require exactly 2 floors, got {floor_ids}")
    gf_id, ff_id = floor_ids

    for sdata in bdata.get("stairs", []):
        gf_space = sdata["gf_space"]
        ff_space = sdata["ff_space"]
        gf_floor = building.get_floor(gf_id)
        ff_floor = building.get_floor(ff_id)
        if gf_space not in gf_floor.spaces:
            raise LoaderError(f"stair {sdata['id']}: unknown GF space {gf_space!r}")
        if ff_space not in ff_floor.spaces:
            raise LoaderError(f"stair {sdata['id']}: unknown FF space {ff_space!r}")

        fire_door_ids: list[str] = []
        for fdata in sdata.get("fire_doors", []):
            floor_id = fdata["floor"]
            floor = building.get_floor(floor_id)
            owner = gf_space if floor_id == gf_id else ff_space
            axis, coord = _axis_coord(fdata["wall"])
            lo, hi = float(fdata["lo"]), float(fdata["hi"])
            other = _resolve_owned(floor.spaces, owner, axis, coord, lo, hi, what=f"fire door of {sdata['id']}")
            fdid = f"{sdata['id']}-fd-{floor_id}"
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
            gf_space=gf_space,
            ff_space=ff_space,
            width_m=float(sdata["width_m"]),
            riser_m=float(sdata.get("riser_m", 0.17)),
            tread_m=float(sdata.get("tread_m", 0.28)),
            floor_to_floor_m=f2f,
            enclosed=bool(sdata.get("enclosed", True)),
            footprint_gf=gf_floor.spaces[gf_space].rect,
            footprint_ff=ff_floor.spaces[ff_space].rect,
            flights=int(sdata.get("flights", 2)),
            chi=float(sdata.get("chi", 0.70)),
            fire_door_id=fire_door_ids[0] if len(fire_door_ids) == 1 else (fire_door_ids or None),
            discharge_exit_id=sdata.get("discharge_exit"),
        )
        building.stairs[stair.id] = stair


def _build_building(bdata: dict[str, Any]) -> Building:
    origin = (float(bdata["origin"]["x"]), float(bdata["origin"]["y"]))
    building = Building(
        id=bdata["id"],
        origin=origin,
        rotation_deg=float(bdata.get("rotation_deg", 0.0)),
        wall_thickness_m=float(bdata.get("wall_thickness_m", 0.2)),
        designed_dead_ends=list(bdata.get("designed_dead_ends", [])),
        exit_sets={k: (list(v) if v is not None else None) for k, v in bdata.get("exit_sets", {}).items()},
    )

    for fd in bdata.get("floors", []):
        floor = _build_floor(building.id, fd)
        building.floors[floor.id] = floor

    window_rule = bdata.get("window_rule", {})
    for floor in building.floors.values():
        _generate_windows(floor, window_rule)

    _build_stairs(building, bdata)
    return building


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
            active_by_default=odata.get("active_by_default", True),
        )
        site.obstacles[obs.id] = obs

    for hdata in sdata.get("hazards", []):
        hz = HazardSource(
            id=hdata["id"], species=hdata.get("species", "H2S"), mw=float(hdata.get("mw", 34.08)),
            x=float(hdata["x"]), y=float(hdata["y"]), z=float(hdata.get("z", 0.0)),
            purpose=hdata.get("purpose", ""),
        )
        site.hazards[hz.id] = hz

    for bdata in data.get("buildings", []):
        building = _build_building(bdata)
        site.buildings[building.id] = building

    site.scenarios = dict(data.get("scenarios", {}))
    return site
