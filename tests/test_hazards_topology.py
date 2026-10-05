"""Milestone 6: gas topology export (spec §6.4) — no solver, just the data
layout: per-floor gas_perm_now, vertical couplings (VOID + stairs),
window couplings, and the ConcentrationGrid factory."""

from __future__ import annotations

import numpy as np
import pytest

from dms.geometry.primitives import Rect
from dms.grid.layers import H_COARSE, H_FINE, DynamicOverlay
from dms.grid.raster import rasterize_floor
from dms.grid.transforms import GridTransform
from dms.hazards.topology import (
    ConcentrationGrid,
    c_in_factory,
    c_out_factory,
    export_gas_topology,
    make_concentration_grid,
    vertical_couplings,
    window_couplings,
)


@pytest.fixture(scope="module")
def gf_grid_fine(building, gf):
    return rasterize_floor(building, gf, H_FINE)


@pytest.fixture(scope="module")
def ff_grid_fine(building, ff):
    return rasterize_floor(building, ff, H_FINE)


@pytest.fixture(scope="module")
def site_transform(site):
    return GridTransform.for_footprint(site.footprint, 1.0, margin=0.0)


# --------------------------------------------------------------------- ConcentrationGrid


def test_concentration_grid_matches_spec_fields():
    cg = ConcentrationGrid(data=np.zeros((2, 2)), h=0.2, origin_XY=(-2.0, -2.0))
    assert cg.unit == "ppm"
    assert cg.species == "H2S"
    assert cg.mw == pytest.approx(34.08)
    assert cg.z == 1.5
    assert cg.t == 0.0
    assert cg.psi == 0.0


def test_concentration_grid_cell_method_matches_spec_formula():
    cg = ConcentrationGrid(data=np.zeros((5, 5)), h=0.5, origin_XY=(0.0, 0.0))
    row, col = cg.cell(np.array([1.2, 2.7]))
    assert (row, col) == (int(2.7 // 0.5), int(1.2 // 0.5))


def test_c_in_factory_shape_matches_floor_grid(gf_grid_fine):
    cg = c_in_factory(gf_grid_fine)
    assert cg.data.shape == gf_grid_fine.transform.shape
    assert cg.h == gf_grid_fine.transform.h
    assert cg.origin_XY == gf_grid_fine.transform.origin_local
    assert (cg.data == 0.0).all()


def test_c_out_factory_shape_matches_site_footprint(site):
    cg = c_out_factory(site, h=2.0)
    expected = GridTransform.for_footprint(site.footprint, 2.0, margin=0.0)
    assert cg.data.shape == expected.shape
    assert cg.h == 2.0


def test_make_concentration_grid_accepts_species_override():
    transform = GridTransform.for_footprint(Rect(0, 0, 10, 10), 1.0, margin=0.0)
    cg = make_concentration_grid(transform, species="CO", mw=28.0, unit="mg/m3")
    assert cg.species == "CO"
    assert cg.mw == 28.0
    assert cg.unit == "mg/m3"


# --------------------------------------------------------------------- vertical couplings


def test_vertical_couplings_cover_every_void_cell(building, gf_grid_fine, ff_grid_fine):
    """Every void cell couples *except* the void's own perimeter wall cells
    — a space's rect still claims its wall cells by centre-containment (the
    wall overlay is a separate raster pass), so `void` alone over-counts."""
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine)
    void_couplings = [c for c in couplings if c.kind == "void"]
    expected = ff_grid_fine.void & ~ff_grid_fine.wall & ~gf_grid_fine.wall
    assert len(void_couplings) == int(expected.sum())
    assert len(void_couplings) < int(ff_grid_fine.void.sum())  # confirms wall cells really were excluded
    assert all(c.coefficient == 1.0 for c in void_couplings)


def test_void_coupling_gf_and_ff_cells_are_the_same_index(building, gf_grid_fine, ff_grid_fine):
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine)
    void_couplings = [c for c in couplings if c.kind == "void"]
    assert all(c.gf_cell == c.ff_cell for c in void_couplings)


def test_stair_couplings_use_correct_default_permeability(building, gf_grid_fine, ff_grid_fine):
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine)
    by_stair = {}
    for c in couplings:
        if c.kind == "stair":
            by_stair.setdefault(c.ref_id, c.coefficient)
    assert by_stair["ST-C"] == pytest.approx(1.0)  # open stair
    assert by_stair["ST-W"] == pytest.approx(0.05)  # enclosed, fire door closed by default
    assert by_stair["ST-E"] == pytest.approx(0.05)


def test_stair_coupling_follows_leaf_override(building, gf_grid_fine, ff_grid_fine):
    overlay = DynamicOverlay(leaf_states={"ST-W-fd-FF": "open"})
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine, overlay=overlay)
    stw = next(c for c in couplings if c.ref_id == "ST-W")
    assert stw.coefficient == pytest.approx(1.0)


def test_stair_coupling_cell_count_matches_landing_footprint_minus_walls(building, gf_grid_fine, ff_grid_fine):
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine)
    for stair_id in building.stairs:
        n = sum(1 for c in couplings if c.ref_id == stair_id)
        sidx = ff_grid_fine.stair_ids.index(stair_id)
        landing = ff_grid_fine.stair_id == sidx
        expected = landing & ~ff_grid_fine.wall & ~gf_grid_fine.wall
        assert n == int(expected.sum())
        assert n < int(landing.sum()), f"{stair_id}'s landing should include some perimeter wall cells to exclude"


def test_mismatched_transforms_raise(building, gf_grid_fine):
    ff_coarse = rasterize_floor(building, building.get_floor("FF"), H_COARSE)
    with pytest.raises(ValueError):
        vertical_couplings(building, gf_grid_fine, ff_coarse)


def test_vertical_couplings_never_land_on_wall_or_outside(building, gf_grid_fine, ff_grid_fine):
    """Interface/edge-case check: no coupling should ever point at a wall or
    outside (margin) cell on either floor — those aren't places gas
    meaningfully flows vertically through."""
    couplings = vertical_couplings(building, gf_grid_fine, ff_grid_fine)
    assert couplings, "expected at least one coupling"
    for c in couplings:
        assert not gf_grid_fine.wall[c.gf_cell], f"{c.kind}/{c.ref_id} GF cell {c.gf_cell} is a wall cell"
        assert not gf_grid_fine.outside[c.gf_cell], f"{c.kind}/{c.ref_id} GF cell {c.gf_cell} is outside the footprint"
        assert not ff_grid_fine.wall[c.ff_cell], f"{c.kind}/{c.ref_id} FF cell {c.ff_cell} is a wall cell"
        assert not ff_grid_fine.outside[c.ff_cell], f"{c.kind}/{c.ref_id} FF cell {c.ff_cell} is outside the footprint"


# --------------------------------------------------------------------- window couplings


def test_window_couplings_cover_every_window(building, gf_grid_fine, site_transform):
    couplings = window_couplings(building, "GF", gf_grid_fine, site_transform)
    window_ids_seen = {c.window_id for c in couplings}
    assert window_ids_seen == set(gf_grid_fine.window_ids)


def test_window_coupling_default_permeability_is_closed(building, gf_grid_fine, site_transform):
    couplings = window_couplings(building, "GF", gf_grid_fine, site_transform)
    assert all(c.permeability == pytest.approx(0.02) for c in couplings)


def test_window_coupling_follows_open_override(building, gf_grid_fine, site_transform):
    wid = gf_grid_fine.window_ids[0]
    overlay = DynamicOverlay(window_open={wid: True})
    couplings = window_couplings(building, "GF", gf_grid_fine, site_transform, overlay=overlay)
    opened = [c for c in couplings if c.window_id == wid]
    assert opened and all(c.permeability == pytest.approx(0.5) for c in opened)


def test_window_outdoor_cell_is_within_site_bounds(building, gf_grid_fine, site, site_transform):
    from dms.fields.outdoor import rasterize_site

    site_grid = rasterize_site(site, h=1.0)
    couplings = window_couplings(building, "GF", gf_grid_fine, site_grid.transform)
    row, col = couplings[0].outdoor_cell
    assert site_grid.transform.in_bounds(row, col)


# --------------------------------------------------------------------- full export


def test_export_gas_topology_end_to_end(site, building, gf_grid_fine, ff_grid_fine):
    topo = export_gas_topology(site, building, {"GF": gf_grid_fine, "FF": ff_grid_fine})
    assert set(topo.gas_perm_now) == {"GF", "FF"}
    assert topo.gas_perm_now["GF"].shape == gf_grid_fine.transform.shape
    assert topo.vertical
    assert topo.windows
    assert set(topo.c_in) == {"GF", "FF"}
    assert topo.c_out is not None
    assert topo.building_id == building.id


def test_gas_perm_now_matches_floor_grid_method(site, building, gf_grid_fine, ff_grid_fine):
    topo = export_gas_topology(site, building, {"GF": gf_grid_fine, "FF": ff_grid_fine})
    expected = gf_grid_fine.gas_perm_now(DynamicOverlay())
    assert np.array_equal(topo.gas_perm_now["GF"], expected)
