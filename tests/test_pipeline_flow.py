"""Tests for the VAMPipeline bridge and the guided stage flow.

The engine-backed parts (merge_stls, make_pipeline) need vamtoolbox + trimesh;
the StageFlow tests need Qt (offscreen). Both are skipped if unavailable."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("VOXELCAST_DISABLE_3D", "1")

import pytest


# --------------------------------------------------------------------------- #
# Engine bridge
# --------------------------------------------------------------------------- #
def test_make_pipeline_ignores_unknown_fields():
    pytest.importorskip("vamtoolbox")
    from voxelcast.engine import pipeline_bridge as pb
    cfg, pipe = pb.make_pipeline({
        "part_height_mm": 20.0, "method": "OSMO", "n_iterations": 7,
        "not_a_field": 123, "absorption": False,
    })
    assert cfg.part_height_mm == 20.0
    assert cfg.method == "OSMO" and cfg.n_iterations == 7
    assert cfg.absorption is False
    assert not hasattr(cfg, "not_a_field")


def test_merge_stls_single_passthrough(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from voxelcast.engine import pipeline_bridge as pb
    p = tmp_path / "one.stl"
    trimesh.creation.box(extents=(5, 5, 5)).export(str(p))
    out, is_temp = pb.merge_stls([str(p)])
    assert out == str(p) and is_temp is False


def test_merge_stls_multiple_combines(tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from voxelcast.engine import pipeline_bridge as pb
    a = tmp_path / "a.stl"
    b = tmp_path / "b.stl"
    m1 = trimesh.creation.box(extents=(5, 5, 5)); m1.apply_translation((-4, 0, 0))
    m2 = trimesh.creation.box(extents=(5, 5, 5)); m2.apply_translation((4, 0, 0))
    m1.export(str(a)); m2.export(str(b))
    out, is_temp = pb.merge_stls([str(a), str(b)])
    assert is_temp is True and os.path.exists(out)
    merged = trimesh.load(out, force="mesh")
    # combined bounding box spans both boxes (~13 mm wide)
    assert merged.bounds[1][0] - merged.bounds[0][0] > 12
    os.remove(out)


# --------------------------------------------------------------------------- #
# Stage flow widget
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PySide6")
    from PySide6 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_stage_flow_config_assembles(qapp):
    from voxelcast.widgets.stage_flow import StageFlow, STAGES
    sf = StageFlow()
    sf.prep.part_height.setValue(30.0)
    sf.prep.n_angles.setValue(240)
    sf.optimize.method.setCurrentText("BCLP")
    sf.optimize.n_iter.setValue(15)
    sf.optimize.diffusion.setChecked(True)
    cfg = sf.config_dict()
    assert cfg["part_height_mm"] == 30.0 and cfg["n_angles"] == 240
    assert cfg["method"] == "BCLP" and cfg["n_iterations"] == 15
    assert cfg["diffusion"] is True
    assert len(STAGES) == 4


def test_stage_rail_unlocks_progressively(qapp):
    from voxelcast.widgets.stage_flow import StageFlow
    from PySide6 import QtCore
    sf = StageFlow()
    # only Prep enabled initially
    assert sf.rail.item(0).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
    assert not (sf.rail.item(2).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled)
    # voxelizing unlocks Optimize; optimizing unlocks Preview
    sf.on_voxelized((10, 10, 8), 100)
    assert sf.rail.item(2).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled
    sf.on_optimized({"gel_mean": 0.8, "void_mean": 0.2, "contrast": 0.6}, 1.2)
    assert sf.rail.item(3).flags() & QtCore.Qt.ItemFlag.ItemIsEnabled


def test_diffusion_disabled_for_osmo(qapp):
    from voxelcast.widgets.stage_flow import StageFlow
    sf = StageFlow()
    sf.optimize.method.setCurrentText("OSMO")
    assert not sf.optimize.diffusion.isEnabled()
    sf.optimize.method.setCurrentText("BCLP")
    assert sf.optimize.diffusion.isEnabled()
