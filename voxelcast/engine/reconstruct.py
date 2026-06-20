"""High-level reconstruction helpers that bridge VoxelCast and VAMToolbox.

The mesh -> recon pipeline has two stages with different threading rules:

* `voxelize_mesh()` builds the target volume from an STL or 3MF. STL uses
  VAMToolbox's OpenGL voxelizer (a GL/pyglet window, which on macOS MUST run on
  the main GUI thread); 3MF (incl. beam lattices) uses the analytic voxelizer.
  Both are fast (sub-second to a couple seconds), so blocking briefly is fine.
  3MF object names drive role assignment (print/insert/zero_dose).

* `build_proj_and_options()` + the actual `optimize()` run is long and pure
  CPU/numpy -- it belongs on a worker thread (see engine.worker.OptimizeWorker).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from voxelcast.model import Dataset

OPTIMIZERS = ("OSMO", "CAL", "BCLP")  # OSMO/BCLP stream progress; CAL does not


MESH_EXTS = (".stl", ".3mf")


@dataclass
class ReconParams:
    mesh_path: str
    resolution: int = 80
    method: str = "OSMO"
    n_iter: int = 20
    num_angles: int = 180
    d_h: float = 0.85
    d_l: float = 0.6
    filter: str = "hamming"


def voxelize_mesh(params: ReconParams):
    """Voxelize an STL or 3MF into a target. MAIN-THREAD ONLY (STL uses OpenGL).

    Returns (target_geo, list[Dataset]) where the datasets are the target plus
    any insert / zero_dose role volumes a 3MF defined (see the naming
    convention). The target is always first.
    """
    import vamtoolbox as vam

    # TargetGeometry auto-routes a .3mf passed via stlfilename to the 3MF path.
    target_geo = vam.geometry.TargetGeometry(
        stlfilename=params.mesh_path, resolution=params.resolution
    )
    datasets = [Dataset(
        array=np.asarray(target_geo.array), vol_type="target", name="target",
        source_path=params.mesh_path, meta={"resolution": params.resolution},
    )]
    for role in ("insert", "zero_dose"):
        arr = getattr(target_geo, role, None)
        if arr is not None and np.asarray(arr).any():
            datasets.append(Dataset(
                array=np.asarray(arr), vol_type="target", name=role,
                source_path=params.mesh_path,
            ))
    return target_geo, datasets


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
