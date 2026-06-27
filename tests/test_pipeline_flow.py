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


def test_opencal_filename_convention():
    from voxelcast.engine import pipeline_bridge as pb
    # appends the rpm tag
    assert pb.opencal_filename("/x/part.mp4", 9).endswith("part_9rpm.mp4")
    # replaces an existing tag rather than stacking
    assert pb.opencal_filename("/x/part_12rpm.mp4", 9).endswith("part_9rpm.mp4")
    # forces .mp4 and strips the reserved 'recording' token
    out = pb.opencal_filename("/x/my_recording_clip.avi", 24)
    assert out.endswith(".mp4") and "recording" not in out and out.endswith("24rpm.mp4")
    # rpm -> deg/s mapping (9 RPM == VoxelCast's 54 deg/s default)
    assert pb.rpm_to_deg_s(9) == 54.0


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


def test_stage_bar_unlocks_progressively(qapp):
    from voxelcast.widgets.stage_flow import StageFlow
    sf = StageFlow()
    # only Prep enabled initially
    assert sf.bar.isTabEnabled(0)
    assert not sf.bar.isTabEnabled(2)
    # voxelizing unlocks Optimize; optimizing unlocks Preview
    sf.on_voxelized((10, 10, 8), 100)
    assert sf.bar.isTabEnabled(2)
    sf.on_optimized({"gel_mean": 0.8, "void_mean": 0.2, "contrast": 0.6}, 1.2)
    assert sf.bar.isTabEnabled(3)


def test_prep_per_model_transform(qapp, tmp_path):
    trimesh = pytest.importorskip("trimesh")
    from PySide6 import QtCore, QtWidgets
    from voxelcast.widgets.stage_flow import StageFlow, _PATH_ROLE, _XFORM_ROLE
    p = tmp_path / "m.stl"
    trimesh.creation.box(extents=(4, 4, 4)).export(str(p))
    sf = StageFlow()
    it = QtWidgets.QListWidgetItem("m.stl")
    it.setData(_PATH_ROLE, str(p))
    it.setData(_XFORM_ROLE, dict(tx=0.0, ty=0.0, tz=0.0, rx=0.0, ry=0.0, rz=0.0))
    sf.prep.list.addItem(it)
    sf.prep.list.setCurrentItem(it)
    sf.prep.tx.setValue(12.0)
    sf.prep.rz.setValue(90.0)
    models = sf.prep.models()
    assert len(models) == 1
    assert models[0]["tx"] == 12.0 and models[0]["rz"] == 90.0
    assert models[0]["path"] == str(p)


def test_voxelize_progress_bar(qapp):
    from voxelcast.widgets.stage_flow import StageFlow
    sf = StageFlow()
    vp = sf.voxelize
    assert vp.bar.isHidden()                # hidden until a run starts
    vp.begin_progress("Voxelizing…")
    assert not vp.bar.isHidden()
    assert vp.bar.minimum() == vp.bar.maximum() == 0   # indeterminate
    vp.set_progress(7, 40, "print")
    assert vp.bar.maximum() == 40 and vp.bar.value() == 7
    assert "7/40" in vp.status.text() and "print" in vp.status.text()
    vp.end_progress()
    assert vp.bar.isHidden()


def test_diffusion_disabled_for_osmo(qapp):
    from voxelcast.widgets.stage_flow import StageFlow
    sf = StageFlow()
    sf.optimize.method.setCurrentText("OSMO")
    assert not sf.optimize.diffusion.isEnabled()
    sf.optimize.method.setCurrentText("BCLP")
    assert sf.optimize.diffusion.isEnabled()
