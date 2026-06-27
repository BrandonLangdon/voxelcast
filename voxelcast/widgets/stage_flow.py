"""Guided four-stage workflow: Prep -> Voxelize -> Optimize -> Preview.

A horizontal *step bar* across the top drives a stacked set of control panels;
the surrounding viewer docks show live previews. The panels only collect
parameters and emit high-level requests -- MainWindow owns the VAMPipeline, the
worker thread, and the dataset/viewer plumbing, and calls back here to unlock the
next stage and stream progress.

Stages map onto VAMToolbox's VAMPipeline:
  Prep      -> choose STL/3MF model(s) + per-model transform + part/vial geometry
  Voxelize  -> pipeline.voxelize()  (GPU OpenGL, main thread)
  Optimize  -> pipeline.optimize()  (OSMO/BCLP; absorption/diffusion, z-slab,
               low-memory, hardware auto-tune)
  Preview   -> pipeline.rebin() + save_video()  (printer-ready MP4)
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

STAGES = ("Prep", "Voxelize", "Optimize", "Preview")

MODEL_EXTS = (".stl", ".3mf")
_PATH_ROLE = QtCore.Qt.ItemDataRole.UserRole
_XFORM_ROLE = QtCore.Qt.ItemDataRole.UserRole + 1


class StageFlow(QtWidgets.QWidget):
    """Container widget: [top step bar] over [stacked panels]."""

    voxelizeRequested = QtCore.Signal()
    optimizeRequested = QtCore.Signal()
    rebinRequested = QtCore.Signal()
    exportRequested = QtCore.Signal(dict)     # {path, rot_vel, num_loops}
    cancelRequested = QtCore.Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.prep = PrepPanel()
        self.voxelize = VoxelizePanel()
        self.optimize = OptimizePanel()
        self.preview = PreviewPanel()
        self._panels = [self.prep, self.voxelize, self.optimize, self.preview]

        # Each panel scrolls, so a small window never clips its controls.
        self.stack = QtWidgets.QStackedWidget()
        for p in self._panels:
            scroll = QtWidgets.QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
            scroll.setWidget(p)
            self.stack.addWidget(scroll)

        # Top horizontal step bar.
        self.bar = QtWidgets.QTabBar()
        self.bar.setDrawBase(True)
        self.bar.setExpanding(True)
        self.bar.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for i, name in enumerate(STAGES):
            self.bar.addTab(f"{i + 1}.  {name}")
        self.bar.currentChanged.connect(self.stack.setCurrentIndex)

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.bar)
        lay.addWidget(self.stack, 1)

        # Stages unlock as prerequisites are met.
        self._max_enabled = 0
        self._set_enabled_through(0)
        self.bar.setCurrentIndex(0)

        # Panel buttons -> flow requests / navigation
        self.prep.next_btn.clicked.connect(lambda: self.go_to(1))
        self.voxelize.run_btn.clicked.connect(self.voxelizeRequested)
        self.voxelize.next_btn.clicked.connect(lambda: self.go_to(2))
        self.optimize.run_btn.clicked.connect(self.optimizeRequested)
        self.optimize.cancel_btn.clicked.connect(self.cancelRequested)
        self.optimize.next_btn.clicked.connect(self._enter_preview)
        self.preview.export_btn.clicked.connect(self._emit_export)

    # -- step bar enabling ----------------------------------------------------
    def _set_enabled_through(self, idx: int) -> None:
        self._max_enabled = max(self._max_enabled, idx)
        for i in range(self.bar.count()):
            self.bar.setTabEnabled(i, i <= self._max_enabled)

    def go_to(self, idx: int) -> None:
        self._set_enabled_through(idx)
        self.bar.setCurrentIndex(idx)

    def _enter_preview(self) -> None:
        self.go_to(3)
        self.rebinRequested.emit()

    def _emit_export(self) -> None:
        path = self.preview.choose_path()
        if path:
            self.exportRequested.emit({
                "path": path,
                "rpm": self.preview.rpm.value(),
                "width": self.preview.proj_w.value(),
                "height": self.preview.proj_h.value(),
                "rotate": self.preview.rotate_deg(),
                "mirror": self.preview.mirror.isChecked(),
                "num_loops": self.preview.num_loops.value(),
            })

    # -- config ---------------------------------------------------------------
    def config_dict(self) -> dict:
        """Assemble PrintConfig fields from all panels."""
        d = {}
        d.update(self.prep.config_dict())
        d.update(self.optimize.config_dict())
        return d

    # -- state updates from MainWindow ---------------------------------------
    def on_voxelized(self, shape, filled) -> None:
        self.voxelize.set_result(shape, filled)
        self.voxelize.next_btn.setEnabled(True)
        self._set_enabled_through(2)

    def on_optimized(self, quality: dict, secs: float) -> None:
        self.optimize.set_result(quality, secs)
        self.optimize.set_running(False)
        self.optimize.next_btn.setEnabled(True)
        self._set_enabled_through(3)

    def on_rebinned(self, shape) -> None:
        self.preview.set_rebinned(shape)

    def on_exported(self, path: str) -> None:
        self.preview.set_exported(path)

    def set_hardware(self, text: str) -> None:
        self.voxelize.set_hardware(text)
        self.optimize.set_hardware(text)

    def progress(self, stage: str, frac: float, msg: str) -> None:
        if stage in ("voxelize", "hardware"):
            self.voxelize.set_status(msg)
        elif stage == "optimize":
            self.optimize.set_progress(frac, msg)
        elif stage in ("rebin", "video", "done"):
            self.preview.set_status(f"{stage}: {msg}")


# --------------------------------------------------------------------------- #
# Panels
# --------------------------------------------------------------------------- #
def _section(title: str) -> QtWidgets.QLabel:
    lbl = QtWidgets.QLabel(title)
    f = lbl.font()
    f.setBold(True)
    lbl.setFont(f)
    return lbl


def _spin(lo, hi, val, step=1.0, suffix="", decimals=None) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    if decimals is not None:
        s.setDecimals(decimals)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


class PrepPanel(QtWidgets.QWidget):
    """Choose STL/3MF model(s), set a per-model transform, and part/vial geometry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("1. Prep — models & geometry"))

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        self.list.currentItemChanged.connect(self._load_xform)
        lay.addWidget(self.list, 1)

        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add model(s)…")
        add.clicked.connect(self._add)
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        lay.addLayout(row)

        # Per-model transform editor (applies to the currently selected model).
        self.xform_box = QtWidgets.QGroupBox("Selected model — transform")
        xf = QtWidgets.QFormLayout(self.xform_box)
        self._loading = False
        self.tx = _spin(-1000, 1000, 0.0, 1.0, " mm")
        self.ty = _spin(-1000, 1000, 0.0, 1.0, " mm")
        self.tz = _spin(-1000, 1000, 0.0, 1.0, " mm")
        self.rx = _spin(-180, 180, 0.0, 5.0, "°")
        self.ry = _spin(-180, 180, 0.0, 5.0, "°")
        self.rz = _spin(-180, 180, 0.0, 5.0, "°")
        trow = QtWidgets.QHBoxLayout()
        for s in (self.tx, self.ty, self.tz):
            trow.addWidget(s)
        tw = QtWidgets.QWidget(); tw.setLayout(trow)
        xf.addRow("Translate X/Y/Z:", tw)
        rrow = QtWidgets.QHBoxLayout()
        for s in (self.rx, self.ry, self.rz):
            rrow.addWidget(s)
        rw = QtWidgets.QWidget(); rw.setLayout(rrow)
        xf.addRow("Rotate X/Y/Z:", rw)
        for s in (self.tx, self.ty, self.tz, self.rx, self.ry, self.rz):
            s.valueChanged.connect(self._save_xform)
        self.xform_box.setEnabled(False)
        lay.addWidget(self.xform_box)

        # Scene / print geometry (global).
        gbox = QtWidgets.QGroupBox("Part & vial")
        form = QtWidgets.QFormLayout(gbox)
        self.part_height = _spin(0.1, 1000.0, 25.4, 1.0, " mm")
        form.addRow("Part height:", self.part_height)
        self.voxel_pitch = _spin(1.0, 2000.0, 80.0, 1.0, " µm")
        self.voxel_pitch.setToolTip("Print voxel pitch (full resolution)")
        form.addRow("Voxel pitch:", self.voxel_pitch)
        self.res_scale = _spin(0.05, 1.0, 1.0, 0.05)
        self.res_scale.setToolTip("Optimize at this fraction of full resolution "
                                  "(< 1 is faster; upsampled before rebin)")
        form.addRow("Resolution scale:", self.res_scale)
        self.vial_radius = _spin(1.0, 500.0, 48.8, 1.0, " mm")
        form.addRow("Vial radius:", self.vial_radius)
        self.n_angles = QtWidgets.QSpinBox()
        self.n_angles.setRange(2, 3600)
        self.n_angles.setValue(360)
        form.addRow("Projection angles:", self.n_angles)
        lay.addWidget(gbox)

        nav = QtWidgets.QHBoxLayout()
        nav.addStretch(1)
        self.next_btn = QtWidgets.QPushButton("Next: Voxelize  ▶")
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

    # -- model list -----------------------------------------------------------
    def _add(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add models", "",
            "Models (*.stl *.3mf);;STL (*.stl);;3MF (*.3mf);;All files (*)")
        for p in paths:
            if os.path.splitext(p)[1].lower() not in MODEL_EXTS:
                continue
            if not self._has(p):
                it = QtWidgets.QListWidgetItem(os.path.basename(p))
                it.setData(_PATH_ROLE, p)
                it.setData(_XFORM_ROLE, dict(tx=0.0, ty=0.0, tz=0.0,
                                             rx=0.0, ry=0.0, rz=0.0))
                it.setToolTip(p)
                self.list.addItem(it)
        self.next_btn.setEnabled(self.list.count() > 0)

    def _has(self, path: str) -> bool:
        return any(self.list.item(i).data(_PATH_ROLE) == path
                   for i in range(self.list.count()))

    def _remove(self) -> None:
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))
        self.next_btn.setEnabled(self.list.count() > 0)

    # -- per-model transform editor ------------------------------------------
    def _load_xform(self, item, _prev=None) -> None:
        self.xform_box.setEnabled(item is not None)
        if item is None:
            return
        xf = item.data(_XFORM_ROLE) or {}
        self._loading = True
        try:
            self.tx.setValue(xf.get("tx", 0.0)); self.ty.setValue(xf.get("ty", 0.0))
            self.tz.setValue(xf.get("tz", 0.0)); self.rx.setValue(xf.get("rx", 0.0))
            self.ry.setValue(xf.get("ry", 0.0)); self.rz.setValue(xf.get("rz", 0.0))
        finally:
            self._loading = False

    def _save_xform(self, *_a) -> None:
        if self._loading:
            return
        item = self.list.currentItem()
        if item is None:
            return
        item.setData(_XFORM_ROLE, dict(
            tx=self.tx.value(), ty=self.ty.value(), tz=self.tz.value(),
            rx=self.rx.value(), ry=self.ry.value(), rz=self.rz.value()))

    # -- accessors ------------------------------------------------------------
    def models(self) -> list[dict]:
        """[{path, tx, ty, tz, rx, ry, rz}, ...] in list order."""
        out = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            xf = dict(it.data(_XFORM_ROLE) or {})
            xf["path"] = it.data(_PATH_ROLE)
            out.append(xf)
        return out

    def config_dict(self) -> dict:
        return {
            "part_height_mm": self.part_height.value(),
            "voxel_pitch_um": self.voxel_pitch.value(),
            "resolution_scale": self.res_scale.value(),
            "vial_radius_mm": self.vial_radius.value(),
            "n_angles": self.n_angles.value(),
        }


class VoxelizePanel(QtWidgets.QWidget):
    """Run GPU voxelization and report the resulting grid."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("2. Voxelize — GPU rasterization"))
        lay.addWidget(QtWidgets.QLabel(
            "Voxelize the model(s) on the GPU (OpenGL). The target appears in the "
            "3D view."))

        self.hardware = QtWidgets.QLabel("Hardware: (detecting…)")
        self.hardware.setWordWrap(True)
        lay.addWidget(self.hardware)

        self.run_btn = QtWidgets.QPushButton("Voxelize  ⚙")
        lay.addWidget(self.run_btn)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setVisible(False)
        lay.addWidget(self.bar)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        lay.addStretch(1)

        nav = QtWidgets.QHBoxLayout()
        nav.addStretch(1)
        self.next_btn = QtWidgets.QPushButton("Next: Optimize  ▶")
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

    def set_hardware(self, text: str) -> None:
        self.hardware.setText(text)

    def set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def begin_progress(self, label: str = "Voxelizing…") -> None:
        """Show an indeterminate bar before the first slice is reported."""
        self.bar.setVisible(True)
        self.bar.setRange(0, 0)
        self.status.setText(label)

    def set_progress(self, done: int, total: int, label: str = "") -> None:
        self.bar.setVisible(True)
        self.bar.setRange(0, max(1, int(total)))
        self.bar.setValue(int(done))
        pct = int(100 * done / total) if total else 0
        tag = f"{label}: " if label and label != "voxelizing" else ""
        self.status.setText(f"{tag}slicing {done}/{total}  ({pct}%)")

    def end_progress(self) -> None:
        self.bar.setVisible(False)

    def set_result(self, shape, filled) -> None:
        self.result.setText(f"Target grid: {tuple(shape)}  ·  {int(filled):,} filled voxels")


class OptimizePanel(QtWidgets.QWidget):
    """Optimizer choice + physics corrections + memory mode + live progress."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("3. Optimize — projections"))

        form = QtWidgets.QFormLayout()
        self.method = QtWidgets.QComboBox()
        self.method.addItems(["OSMO", "BCLP"])
        self.method.currentTextChanged.connect(self._method_changed)
        form.addRow("Method:", self.method)

        self.n_iter = QtWidgets.QSpinBox()
        self.n_iter.setRange(1, 2000)
        self.n_iter.setValue(20)
        form.addRow("Iterations:", self.n_iter)

        self.d_high = _spin(0.0, 1.0, 0.85, 0.05)
        form.addRow("Dose high (d_h):", self.d_high)
        self.d_low = _spin(0.0, 1.0, 0.65, 0.05)
        form.addRow("Dose low (d_l):", self.d_low)
        lay.addLayout(form)

        # Physics corrections
        lay.addWidget(_section("Corrections"))
        self.absorption = QtWidgets.QCheckBox("Absorption (Beer–Lambert attenuation)")
        self.absorption.setChecked(True)
        lay.addWidget(self.absorption)
        self.diffusion = QtWidgets.QCheckBox("Diffusion blur (BCLP only)")
        self.diffusion.setEnabled(False)
        self.diffusion.setToolTip("Requires the BCLP method")
        lay.addWidget(self.diffusion)

        # Memory mode
        lay.addWidget(_section("Memory"))
        mrow = QtWidgets.QFormLayout()
        self.slab = QtWidgets.QComboBox()
        self.slab.addItems(["auto", "off"])
        self.slab.setEditable(True)
        self.slab.setToolTip("z-slab depth for very large parts: auto / off / an integer")
        mrow.addRow("z-slab:", self.slab)
        self.low_mem = QtWidgets.QCheckBox("Low-memory BCLP buffers")
        self.low_mem.setEnabled(False)
        mrow.addRow("", self.low_mem)
        lay.addLayout(mrow)

        self.hardware = QtWidgets.QLabel("")
        self.hardware.setWordWrap(True)
        lay.addWidget(self.hardware)

        brow = QtWidgets.QHBoxLayout()
        self.run_btn = QtWidgets.QPushButton("Run optimization  ⚙")
        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        brow.addWidget(self.run_btn)
        brow.addWidget(self.cancel_btn)
        lay.addLayout(brow)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 100)
        lay.addWidget(self.bar)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.result = QtWidgets.QLabel("")
        self.result.setWordWrap(True)
        lay.addWidget(self.result)
        lay.addStretch(1)

        nav = QtWidgets.QHBoxLayout()
        nav.addStretch(1)
        self.next_btn = QtWidgets.QPushButton("Next: Preview  ▶")
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

    def _method_changed(self, m: str) -> None:
        is_bclp = (m == "BCLP")
        self.diffusion.setEnabled(is_bclp)
        self.low_mem.setEnabled(is_bclp)
        if not is_bclp:
            self.diffusion.setChecked(False)
            self.low_mem.setChecked(False)

    def config_dict(self) -> dict:
        return {
            "method": self.method.currentText(),
            "n_iterations": self.n_iter.value(),
            "d_high": self.d_high.value(),
            "d_low": self.d_low.value(),
            "absorption": self.absorption.isChecked(),
            "diffusion": self.diffusion.isChecked(),
            "slab": self.slab.currentText().strip() or "auto",
            "low_memory": self.low_mem.isChecked(),
        }

    def set_hardware(self, text: str) -> None:
        self.hardware.setText(text)

    def set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.bar.setRange(0, 0)  # indeterminate until first iter

    def set_progress(self, frac: float, msg: str) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(int(frac * 100))
        self.status.setText(msg)

    def set_result(self, quality: dict, secs: float) -> None:
        self.bar.setRange(0, 100)
        self.bar.setValue(100)
        if quality:
            self.result.setText(
                f"Done in {secs:.1f}s  ·  gel {quality.get('gel_mean', 0):.3f} / "
                f"void {quality.get('void_mean', 0):.3f}  ·  "
                f"contrast {quality.get('contrast', 0):.3f}")
        else:
            self.result.setText(f"Done in {secs:.1f}s")


class PreviewPanel(QtWidgets.QWidget):
    """Fan-beam rebin + printer-ready MP4 export."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("4. Preview & export"))
        lay.addWidget(QtWidgets.QLabel(
            "The optimized sinogram is rebinned to printer (fan-beam) geometry, "
            "then exported as a projection video."))

        self.status = QtWidgets.QLabel("Rebinning…")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        self.rebin_info = QtWidgets.QLabel("")
        self.rebin_info.setWordWrap(True)
        lay.addWidget(self.rebin_info)

        lay.addWidget(_section("Projection video (OpenCAL printer)"))
        form = QtWidgets.QFormLayout()
        # OpenCAL reads RPM from the filename; one video loop = one revolution.
        self.rpm = QtWidgets.QSpinBox()
        self.rpm.setRange(1, 60)
        self.rpm.setValue(9)
        self.rpm.setSuffix(" RPM")
        self.rpm.setToolTip("Vial rotation speed. Written into the filename "
                            "(<part>_<rpm>rpm.mp4) so the firmware sets the motor.")
        form.addRow("Vial rotation:", self.rpm)

        # Projector frame. Default 1920×1080 (the firmware's HDMI projector).
        self.proj_w = QtWidgets.QSpinBox()
        self.proj_w.setRange(320, 7680); self.proj_w.setSingleStep(4)
        self.proj_w.setValue(1920)
        self.proj_h = QtWidgets.QSpinBox()
        self.proj_h.setRange(240, 7680); self.proj_h.setSingleStep(4)
        self.proj_h.setValue(1080)
        res_row = QtWidgets.QHBoxLayout()
        res_row.addWidget(self.proj_w)
        res_row.addWidget(QtWidgets.QLabel("×"))
        res_row.addWidget(self.proj_h)
        res_w = QtWidgets.QWidget(); res_w.setLayout(res_row)
        form.addRow("Projector W×H:", res_w)

        self.rotate = QtWidgets.QComboBox()
        self.rotate.addItems(["0°", "90°", "180°", "270°"])
        self.rotate.setCurrentText("90°")
        self.rotate.setToolTip("Rotate frames CW. 90° stands an upright part along "
                               "a landscape projector's long axis.")
        form.addRow("Rotate frames:", self.rotate)

        self.mirror = QtWidgets.QCheckBox("Mirror (flip handedness / rotation dir)")
        self.mirror.setToolTip("The printer spins the vial CCW. Enable if a print "
                               "comes out mirrored.")
        form.addRow("", self.mirror)

        self.num_loops = QtWidgets.QSpinBox()
        self.num_loops.setRange(1, 100)
        self.num_loops.setValue(1)
        self.num_loops.setToolTip("Revolutions per file. Keep 1 — the firmware "
                                  "loops the video continuously.")
        form.addRow("Loops:", self.num_loops)
        lay.addLayout(form)

        self.export_btn = QtWidgets.QPushButton("Export projection video (.mp4)…")
        lay.addWidget(self.export_btn)
        self.exported = QtWidgets.QLabel("")
        self.exported.setWordWrap(True)
        lay.addWidget(self.exported)
        lay.addStretch(1)

    def rotate_deg(self) -> float:
        return float(self.rotate.currentText().rstrip("°"))

    def choose_path(self) -> str:
        suggested = f"print_{self.rpm.value()}rpm.mp4"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export projection video", suggested, "MP4 video (*.mp4)")
        return path

    def set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def set_rebinned(self, shape) -> None:
        self.status.setText("Rebin complete — ready to export.")
        self.rebin_info.setText(f"Printer sinogram: {tuple(shape)}")

    def set_exported(self, path: str) -> None:
        self.exported.setText(f"Saved: {path}")
