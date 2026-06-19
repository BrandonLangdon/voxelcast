"""High-level reconstruction helpers that bridge VoxelCast and VAMToolbox.

The STL -> recon pipeline has two stages with different threading rules:

* `voxelize_stl()` builds the target volume from an STL using VAMToolbox's
  OpenGL voxelizer. That creates a GL/pyglet window, which on macOS MUST run on
  the main (GUI) thread. It is fast (sub-second), so blocking briefly is fine.

* `build_proj_and_options()` + the actual `optimize()` run is long and pure
  CPU/numpy -- it belongs on a worker thread (see engine.worker.OptimizeWorker).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxelcast.model import Dataset

OPTIMIZERS = ("OSMO", "CAL", "BCLP")  # OSMO/BCLP stream progress; CAL does not


@dataclass
class ReconParams:
    stl_path: str
    resolution: int = 80
    method: str = "OSMO"
    n_iter: int = 20
    num_angles: int = 180
    d_h: float = 0.85
    d_l: float = 0.6
    filter: str = "hamming"


def voxelize_stl(params: ReconParams):
    """MAIN-THREAD ONLY (OpenGL). Returns (target_geo, target Dataset)."""
    import vamtoolbox as vam

    target_geo = vam.geometry.TargetGeometry(
        stlfilename=params.stl_path, resolution=params.resolution
    )
    ds = Dataset(
        array=np.asarray(target_geo.array),
        vol_type="target",
        name="target",
        source_path=params.stl_path,
        meta={"resolution": params.resolution},
    )
    return target_geo, ds


def build_proj_and_options(params: ReconParams):
    """Build the ProjectionGeometry + optimizer Options (no heavy work here)."""
    import vamtoolbox as vam

    angles = np.linspace(0, 360 - 360 / params.num_angles, params.num_angles)
    # CUDA=False forces the CPU path; on a machine with astra+CUDA the engine
    # would still fall back correctly, but being explicit avoids surprises.
    proj_geo = vam.geometry.ProjectionGeometry(
        angles, ray_type="parallel", CUDA=False
    )
    options = vam.optimize.Options(
        method=params.method,
        n_iter=params.n_iter,
        d_h=params.d_h,
        d_l=params.d_l,
        filter=params.filter,
        verbose=False,
    )
    return proj_geo, options
