from __future__ import annotations

from pathlib import Path

import pytest

from dms.geometry.loader import load_site
from dms.geometry.site import Site
from dms.graph.builder import NavGraph, build_navigation_graph
from dms.grid.layers import H_COARSE
from dms.grid.raster import rasterize_floor

CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "buildings" / "two_storey.yaml"


@pytest.fixture(scope="session")
def site() -> Site:
    return load_site(CONFIG_PATH)


@pytest.fixture(scope="session")
def building(site: Site):
    return site.get_building("B1")


@pytest.fixture(scope="session")
def gf(building):
    return building.get_floor("GF")


@pytest.fixture(scope="session")
def ff(building):
    return building.get_floor("FF")


@pytest.fixture(scope="session")
def nav(site: Site) -> NavGraph:
    """Session-scoped: building the graph is cheap, but link state is
    mutable (scenario presets write to it in place). Any test that applies
    a scenario must call reset_states(nav) itself first, so tests stay
    order-independent regardless of fixture scope."""
    return build_navigation_graph(site, "B1")


@pytest.fixture(scope="session")
def gf_grid(building, gf):
    return rasterize_floor(building, gf, H_COARSE)


@pytest.fixture(scope="session")
def ff_grid(building, ff):
    return rasterize_floor(building, ff, H_COARSE)
