"""The in-app data model.

`Dataset` is a deliberately thin, dependency-free container: a numpy array plus
a small amount of metadata.  Loaders convert VAMToolbox `Volume` objects (or
neutral files) into `Dataset`s so the rest of the app never imports or depends
on vamtoolbox internals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

# VAMToolbox vol_type values we understand.
VOL_TYPES = ("target", "recon", "sino")


@dataclass
class Dataset:
    """A loaded volume/sinogram/image ready for display.

    Parameters
    ----------
    array : np.ndarray
        2D or 3D data. For volumes the convention is (nY, nX, nZ); for
        sinograms (nR, nTheta[, nZ]) -- matching VAMToolbox.
    vol_type : str
        One of VOL_TYPES, or "unknown" for neutral files.
    name : str
        Display name (usually the file stem).
    source_path : str | None
        Where it came from, if loaded from disk.
    meta : dict
        Free-form extra metadata (angles, dims, etc.).
    """

    array: np.ndarray
    vol_type: str = "unknown"
    name: str = "dataset"
    source_path: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.array = np.asarray(self.array)

    @property
    def effective_ndim(self) -> int:
        """Dimensionality ignoring a trailing singleton z (VAM stores 2D as 3D)."""
        return int(np.squeeze(self.array).ndim)

    @property
    def is_volume(self) -> bool:
        """True if this should get a 3D rendering (real depth)."""
        return self.vol_type in ("target", "recon") and self.effective_ndim == 3

    @property
    def is_sinogram(self) -> bool:
        return self.vol_type == "sino"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.array.shape)

    def summary(self) -> str:
        return (
            f"{self.name} [{self.vol_type}] shape={self.shape} "
            f"dtype={self.array.dtype} range=({np.nanmin(self.array):.3g}, "
            f"{np.nanmax(self.array):.3g})"
        )
