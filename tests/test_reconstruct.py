"""Engine-bridge tests. The geometry/options builder needs vamtoolbox but no
GUI, OpenGL, or heavy compute -- so it is safe for CI when vamtoolbox is present
and skipped otherwise."""
import numpy as np
import pytest

from voxelcast.engine.reconstruct import ReconParams, OPTIMIZERS


def test_recon_params_defaults():
    p = ReconParams(stl_path="/tmp/x.stl")
    assert p.method in OPTIMIZERS
    assert 0 <= p.d_l <= p.d_h <= 1
    assert p.resolution > 0 and p.n_iter > 0 and p.num_angles >= 2


def test_build_proj_and_options():
    pytest.importorskip("vamtoolbox")
    from voxelcast.engine.reconstruct import build_proj_and_options

    p = ReconParams(stl_path="/tmp/x.stl", method="OSMO", n_iter=7, num_angles=120)
    proj_geo, options = build_proj_and_options(p)
    assert options.method == "OSMO"
    assert options.n_iter == 7
    assert proj_geo.CUDA is False
    assert len(proj_geo.angles) == 120
    # angles span [0, 360) without duplicating the endpoint
    assert np.isclose(proj_geo.angles[0], 0.0)
    assert proj_geo.angles[-1] < 360.0
