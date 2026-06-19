"""IO + model tests that need no GUI backend (no PySide6/VTK import)."""
import numpy as np
import pytest

from voxelcast.io import load, save_neutral, LoadError
from voxelcast.model import Dataset


def test_dataset_classification():
    vol = Dataset(array=np.zeros((8, 8, 8)), vol_type="recon")
    assert vol.is_volume and not vol.is_sinogram and vol.effective_ndim == 3

    img = Dataset(array=np.zeros((8, 8, 1)), vol_type="target")
    assert not img.is_volume  # trailing singleton -> effectively 2D

    sino = Dataset(array=np.zeros((8, 60)), vol_type="sino")
    assert sino.is_sinogram and not sino.is_volume


def test_npy_roundtrip(tmp_path):
    arr = np.random.default_rng(0).random((6, 6, 4)).astype(np.float32)
    p = tmp_path / "recon.npy"
    save_neutral(Dataset(array=arr, vol_type="recon", name="recon"), str(p))
    ds = load(str(p))
    assert ds.shape == arr.shape
    assert ds.vol_type == "recon"          # guessed from filename
    np.testing.assert_allclose(ds.array, arr)


def test_tiff_roundtrip(tmp_path):
    pytest.importorskip("tifffile")
    arr = np.random.default_rng(1).random((5, 7, 3)).astype(np.float32)
    p = tmp_path / "sino_stack.tif"
    save_neutral(Dataset(array=arr, vol_type="sino", name="sino_stack"), str(p))
    ds = load(str(p))
    # (y,x,z) -> stack (z,y,x) -> back to (y,x,z): shape preserved
    assert ds.shape == arr.shape
    assert ds.vol_type == "sino"


def test_unsupported_extension(tmp_path):
    p = tmp_path / "thing.xyz"
    p.write_text("nope")
    with pytest.raises(LoadError):
        load(str(p))
