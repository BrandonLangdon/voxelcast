"""File IO: turn files on disk into `Dataset`s, and export neutral formats.

Two classes of input:

* **Native VAMToolbox** (`.target` / `.sino` / `.recon`) — these are dill
  pickles of `vamtoolbox.geometry.Volume` objects, so *unpickling requires
  vamtoolbox to be importable*. We extract `.array` + `.vol_type` into a
  `Dataset` and immediately drop the original object, so nothing downstream
  depends on vamtoolbox.

* **Neutral** (`.npy`, `.tif`/`.tiff`) — portable, inspectable, no vamtoolbox
  needed. Useful for sharing results or viewing on a machine without the engine.

`save_neutral()` writes the portable path so VAMToolbox results can be exported
once and viewed anywhere.
"""
from __future__ import annotations

import os

import numpy as np

from voxelcast.model import Dataset

# Native VAM extension -> vol_type
NATIVE_EXTS = {".target": "target", ".sino": "sino", ".recon": "recon"}
NEUTRAL_EXTS = (".npy", ".tif", ".tiff")


class LoadError(Exception):
    """Raised when a file cannot be loaded into a Dataset."""


def load(path: str) -> Dataset:
    """Load any supported file into a `Dataset`. Raises `LoadError` on failure."""
    ext = os.path.splitext(path)[1].lower()
    name = os.path.splitext(os.path.basename(path))[0]

    if ext in NATIVE_EXTS:
        return _load_native(path, NATIVE_EXTS[ext], name)
    if ext == ".npy":
        return _load_npy(path, name)
    if ext in (".tif", ".tiff"):
        return _load_tiff(path, name)
    raise LoadError(
        f"Unsupported file type '{ext}'. Supported: "
        f"{', '.join(sorted(NATIVE_EXTS))}, {', '.join(NEUTRAL_EXTS)}"
    )


def _load_native(path: str, vol_type: str, name: str) -> Dataset:
    try:
        import vamtoolbox  # noqa: F401
        from vamtoolbox import geometry
    except Exception as e:  # pragma: no cover - environment dependent
        raise LoadError(
            f"'{os.path.basename(path)}' is a VAMToolbox file (a pickled Volume); "
            f"loading it needs vamtoolbox installed.\n  ({e})\n"
            "Tip: export to .npy/.tiff from VAMToolbox for a portable file."
        ) from e
    try:
        vol = geometry.loadVolume(path)
    except Exception as e:
        raise LoadError(f"Failed to unpickle '{path}': {e}") from e

    meta = {
        k: getattr(vol, k)
        for k in ("nX", "nY", "nZ", "nR", "nTheta", "n_dim")
        if hasattr(vol, k)
    }
    return Dataset(
        array=np.asarray(vol.array),
        vol_type=getattr(vol, "vol_type", vol_type) or vol_type,
        name=name,
        source_path=path,
        meta=meta,
    )


def _load_npy(path: str, name: str) -> Dataset:
    try:
        arr = np.load(path, allow_pickle=False)
    except Exception as e:
        raise LoadError(f"Failed to read '{path}': {e}") from e
    return Dataset(array=arr, vol_type=_guess_vol_type(arr, name), name=name,
                   source_path=path)


def _load_tiff(path: str, name: str) -> Dataset:
    try:
        import tifffile
    except Exception as e:
        raise LoadError(f"Reading TIFF needs the 'tifffile' package ({e}).") from e
    try:
        arr = tifffile.imread(path)
    except Exception as e:
        raise LoadError(f"Failed to read '{path}': {e}") from e
    # tifffile returns (z, y, x) for stacks; move to VAM's (y, x, z).
    if arr.ndim == 3:
        arr = np.moveaxis(arr, 0, 2)
    return Dataset(array=arr, vol_type=_guess_vol_type(arr, name), name=name,
                   source_path=path)


def _guess_vol_type(arr: np.ndarray, name: str) -> str:
    lname = name.lower()
    if "sino" in lname:
        return "sino"
    if "recon" in lname:
        return "recon"
    if "target" in lname:
        return "target"
    return "recon" if np.squeeze(arr).ndim == 3 else "unknown"


def save_neutral(dataset: Dataset, path: str) -> None:
    """Export a Dataset to a portable .npy or TIFF stack (no vamtoolbox needed)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        np.save(path, dataset.array)
    elif ext in (".tif", ".tiff"):
        import tifffile
        arr = dataset.array
        if arr.ndim == 3:  # back to (z, y, x) for standard stack viewers
            arr = np.moveaxis(arr, 2, 0)
        # photometric=minisblack: store as a grayscale z-stack, never RGB
        # (a 3-layer volume would otherwise be misread as a colour image).
        tifffile.imwrite(path, np.ascontiguousarray(arr), photometric="minisblack")
    else:
        raise LoadError(f"Cannot export to '{ext}'. Use {NEUTRAL_EXTS}.")
