"""Guided four-stage workflow: Prep -> Voxelize -> Optimize -> Preview.

A vertical *stage rail* on the left drives a stacked set of control panels in the
center; the surrounding viewer docks show live previews. The panels only collect
parameters and emit high-level requests -- MainWindow owns the VAMPipeline, the
worker thread, and the dataset/viewer plumbing, and calls back here to unlock the
next stage and stream progress.

Stages map onto VAMToolbox's VAMPipeline:
  Prep      -> choose STL(s) + part/vial geometry  (builds PrintConfig)
  Voxelize  -> pipeline.voxelize()  (GPU OpenGL, main thread)
  Optimize  -> pipeline.optimize()  (OSMO/BCLP, worker; absorption/diffusion,
               z-slab, low-memory, hardware auto-tune)
  Preview   -> pipeline.rebin() + save_video()  (printer-ready MP4)
"""
from __future__ import annotations

import os

from PySide6 import QtCore, QtWidgets

STAGES = ("Prep", "Voxelize", "Optimize", "Preview")


class StageFlow(QtWidgets.QWidget):
    """Container widget: [stage rail | stacked panels]."""

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

        self.stack = QtWidgets.QStackedWidget()
        for p in self._panels:
            self.stack.addWidget(p)

        self.rail = QtWidgets.QListWidget()
        self.rail.setFixedWidth(132)
        self.rail.setFocusPolicy(QtCore.Qt.FocusPolicy.NoFocus)
        for i, name in enumerate(STAGES):
            item = QtWidgets.QListWidgetItem(f"{i + 1}.  {name}")
            item.setSizeHint(QtCore.QSize(0, 44))
            self.rail.addItem(item)
        self.rail.currentRowChanged.connect(self.stack.setCurrentIndex)

        lay = QtWidgets.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.rail)
        lay.addWidget(self.stack, 1)

        # Stages unlock as prerequisites are met.
        self._max_enabled = 0
        self._set_enabled_through(0)
        self.rail.setCurrentRow(0)

        # Panel buttons -> flow requests / navigation
        self.prep.next_btn.clicked.connect(lambda: self.go_to(1))
        self.voxelize.run_btn.clicked.connect(self.voxelizeRequested)
        self.voxelize.next_btn.clicked.connect(lambda: self.go_to(2))
        self.optimize.run_btn.clicked.connect(self.optimizeRequested)
        self.optimize.cancel_btn.clicked.connect(self.cancelRequested)
        self.optimize.next_btn.clicked.connect(self._enter_preview)
        self.preview.export_btn.clicked.connect(self._emit_export)

    # -- rail enabling --------------------------------------------------------
    def _set_enabled_through(self, idx: int) -> None:
        self._max_enabled = max(self._max_enabled, idx)
        for i in range(self.rail.count()):
            it = self.rail.item(i)
            flags = it.flags()
            if i <= self._max_enabled:
                it.setFlags(flags | QtCore.Qt.ItemFlag.ItemIsEnabled)
            else:
                it.setFlags(flags & ~QtCore.Qt.ItemFlag.ItemIsEnabled)

    def go_to(self, idx: int) -> None:
        self._set_enabled_through(idx)
        self.rail.setCurrentRow(idx)

    def _enter_preview(self) -> None:
        self.go_to(3)
        self.rebinRequested.emit()

    def _emit_export(self) -> None:
        path = self.preview.choose_path()
        if path:
            self.exportRequested.emit({
                "path": path,
                "rot_vel": self.preview.rot_vel.value(),
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


class PrepPanel(QtWidgets.QWidget):
    """Choose one or more STLs and set part/vial geometry."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("1. Prep — model & geometry"))
        lay.addWidget(QtWidgets.QLabel(
            "Load one or more STL files. Multiple parts are merged into one "
            "aligned voxel grid."))

        self.list = QtWidgets.QListWidget()
        self.list.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection)
        lay.addWidget(self.list, 1)

        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add STL(s)…")
        add.clicked.connect(self._add)
        rm = QtWidgets.QPushButton("Remove")
        rm.clicked.connect(self._remove)
        row.addWidget(add)
        row.addWidget(rm)
        row.addStretch(1)
        lay.addLayout(row)

        form = QtWidgets.QFormLayout()
        self.part_height = QtWidgets.QDoubleSpinBox()
        self.part_height.setRange(0.1, 1000.0)
        self.part_height.setValue(25.4)
        self.part_height.setSuffix(" mm")
        form.addRow("Part height:", self.part_height)

        self.voxel_pitch = QtWidgets.QDoubleSpinBox()
        self.voxel_pitch.setRange(1.0, 2000.0)
        self.voxel_pitch.setValue(80.0)
        self.voxel_pitch.setSuffix(" µm")
        self.voxel_pitch.setToolTip("Print voxel pitch (full resolution)")
        form.addRow("Voxel pitch:", self.voxel_pitch)

        self.res_scale = QtWidgets.QDoubleSpinBox()
        self.res_scale.setRange(0.05, 1.0)
        self.res_scale.setSingleStep(0.05)
        self.res_scale.setValue(1.0)
        self.res_scale.setToolTip("Optimize at this fraction of full resolution "
                                  "(< 1 is faster; upsampled before rebin)")
        form.addRow("Resolution scale:", self.res_scale)

        self.vial_radius = QtWidgets.QDoubleSpinBox()
        self.vial_radius.setRange(1.0, 500.0)
        self.vial_radius.setValue(48.8)
        self.vial_radius.setSuffix(" mm")
        form.addRow("Vial radius:", self.vial_radius)

        self.n_angles = QtWidgets.QSpinBox()
        self.n_angles.setRange(2, 3600)
        self.n_angles.setValue(360)
        form.addRow("Projection angles:", self.n_angles)

        self.rot_x = self._angle_spin()
        self.rot_y = self._angle_spin()
        self.rot_z = self._angle_spin()
        rrow = QtWidgets.QHBoxLayout()
        for s in (self.rot_x, self.rot_y, self.rot_z):
            rrow.addWidget(s)
        rotw = QtWidgets.QWidget()
        rotw.setLayout(rrow)
        form.addRow("Rotate X/Y/Z:", rotw)
        lay.addLayout(form)

        nav = QtWidgets.QHBoxLayout()
        nav.addStretch(1)
        self.next_btn = QtWidgets.QPushButton("Next: Voxelize  ▶")
        self.next_btn.setEnabled(False)
        nav.addWidget(self.next_btn)
        lay.addLayout(nav)

    @staticmethod
    def _angle_spin() -> QtWidgets.QDoubleSpinBox:
        s = QtWidgets.QDoubleSpinBox()
        s.setRange(-180.0, 180.0)
        s.setSuffix("°")
        return s

    def _add(self) -> None:
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Add STL files", "", "STL (*.stl);;All files (*)")
        for p in paths:
            if not self._has(p):
                it = QtWidgets.QListWidgetItem(os.path.basename(p))
                it.setData(QtCore.Qt.ItemDataRole.UserRole, p)
                it.setToolTip(p)
                self.list.addItem(it)
        self.next_btn.setEnabled(self.list.count() > 0)

    def _has(self, path: str) -> bool:
        return any(self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole) == path
                   for i in range(self.list.count()))

    def _remove(self) -> None:
        for it in self.list.selectedItems():
            self.list.takeItem(self.list.row(it))
        self.next_btn.setEnabled(self.list.count() > 0)

    def stl_paths(self) -> list[str]:
        return [self.list.item(i).data(QtCore.Qt.ItemDataRole.UserRole)
                for i in range(self.list.count())]

    def config_dict(self) -> dict:
        return {
            "part_height_mm": self.part_height.value(),
            "voxel_pitch_um": self.voxel_pitch.value(),
            "resolution_scale": self.res_scale.value(),
            "vial_radius_mm": self.vial_radius.value(),
            "n_angles": self.n_angles.value(),
        }

    def rot_angles(self) -> list[float]:
        return [self.rot_x.value(), self.rot_y.value(), self.rot_z.value()]


class VoxelizePanel(QtWidgets.QWidget):
    """Run GPU voxelization and report the resulting grid."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QtWidgets.QVBoxLayout(self)
        lay.addWidget(_section("2. Voxelize — GPU rasterization"))
        lay.addWidget(QtWidgets.QLabel(
            "Voxelize the merged mesh on the GPU (OpenGL). The target appears "
            "in the 3D view."))

        self.hardware = QtWidgets.QLabel("Hardware: (detecting…)")
        self.hardware.setWordWrap(True)
        lay.addWidget(self.hardware)

        self.run_btn = QtWidgets.QPushButton("Voxelize  ⚙")
        lay.addWidget(self.run_btn)

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

        self.d_high = QtWidgets.QDoubleSpinBox()
        self.d_high.setRange(0.0, 1.0)
        self.d_high.setSingleStep(0.05)
        self.d_high.setValue(0.85)
        form.addRow("Dose high (d_h):", self.d_high)

        self.d_low = QtWidgets.QDoubleSpinBox()
        self.d_low.setRange(0.0, 1.0)
        self.d_low.setSingleStep(0.05)
        self.d_low.setValue(0.65)
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

        lay.addWidget(_section("Projection video"))
        form = QtWidgets.QFormLayout()
        self.rot_vel = QtWidgets.QDoubleSpinBox()
        self.rot_vel.setRange(1.0, 3600.0)
        self.rot_vel.setValue(54.0)
        self.rot_vel.setSuffix(" °/s")
        form.addRow("Rotation speed:", self.rot_vel)
        self.num_loops = QtWidgets.QSpinBox()
        self.num_loops.setRange(1, 100)
        self.num_loops.setValue(1)
        form.addRow("Loops:", self.num_loops)
        lay.addLayout(form)

        self.export_btn = QtWidgets.QPushButton("Export projection video (.mp4)…")
        lay.addWidget(self.export_btn)
        self.exported = QtWidgets.QLabel("")
        self.exported.setWordWrap(True)
        lay.addWidget(self.exported)
        lay.addStretch(1)

    def choose_path(self) -> str:
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export projection video", "projections.mp4", "MP4 video (*.mp4)")
        return path

    def set_status(self, msg: str) -> None:
        self.status.setText(msg)

    def set_rebinned(self, shape) -> None:
        self.status.setText("Rebin complete — ready to export.")
        self.rebin_info.setText(f"Printer sinogram: {tuple(shape)}")

    def set_exported(self, path: str) -> None:
        self.exported.setText(f"Saved: {path}")
