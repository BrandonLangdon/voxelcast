"""VoxelCast main window.

Open VAMToolbox files (.target/.sino/.recon) or neutral arrays (.npy/.tiff), or
reconstruct directly from an STL (voxelize -> optimize) using the VAMToolbox
engine. Results are shown as a 3D volume render (pyvistaqt) plus a scrubbable
2D slice/sinogram view (pyqtgraph). Multiple datasets (target / sinogram /
recon) are held in a selector so you can switch between them.

Threading model:
* STL voxelization uses OpenGL and MUST run on the main thread (fast, blocks
  briefly with a status message).
* optimize() is long and runs on a worker thread (engine.OptimizeWorker),
  streaming progress back via Qt signals.
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from PySide6 import QtCore, QtWidgets

from voxelcast import __version__
from voxelcast.engine.worker import OptimizeWorker, engine_available
from voxelcast.engine import pipeline_bridge as pb
from voxelcast.io import load, save_neutral, NATIVE_EXTS, NEUTRAL_EXTS, LoadError
from voxelcast.model import Dataset


def _placeholder(message: str) -> QtWidgets.QWidget:
    w = QtWidgets.QLabel(message)
    w.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
    w.setWordWrap(True)
    return w


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"VoxelCast {__version__}")
        self.resize(1200, 780)

        self._current: Dataset | None = None
        self._datasets: dict[str, Dataset] = {}
        self._volume_view = None
        self._slice_view = None
        self._sinogram_view = None
        self._error_view = None
        self._compare_view = None
        self._thread: QtCore.QThread | None = None
        self._worker: OptimizeWorker | None = None

        # Guided-flow pipeline session (VAMPipeline) + its worker thread.
        self._pipe = None
        self._config = None
        self._merged_tmp: str | None = None
        self._export_path: str | None = None
        self._pp_thread: QtCore.QThread | None = None
        self._pp_worker: pb.PipelineWorker | None = None

        self._build_menu()
        self._build_toolbar()
        self._build_docks()
        self._build_stage_flow()  # central guided workflow
        self._build_view_menu()  # after docks exist (uses their toggle actions)
        self._build_statusbar()
        self._show_hardware()
        self.statusBar().showMessage(
            "Prep → Voxelize → Optimize → Preview, or open a file directly")

    # ----- UI construction -------------------------------------------------
    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_act = file_menu.addAction("&Open…")
        open_act.setShortcut("Ctrl+O")
        open_act.triggered.connect(self.open_file_dialog)

        self.export_act = file_menu.addAction("&Export current (neutral)…")
        self.export_act.setShortcut("Ctrl+E")
        self.export_act.triggered.connect(self.export_dialog)
        self.export_act.setEnabled(False)

        file_menu.addSeparator()
        demo_act = file_menu.addAction("Load &demo volume")
        demo_act.triggered.connect(self.load_demo)
        file_menu.addSeparator()
        quit_act = file_menu.addAction("&Quit")
        quit_act.setShortcut("Ctrl+Q")
        quit_act.triggered.connect(self.close)

        engine_menu = self.menuBar().addMenu("&Engine")
        self.recon_act = engine_menu.addAction("&Reconstruct from STL / 3MF…")
        self.recon_act.setShortcut("Ctrl+R")
        self.recon_act.triggered.connect(self.reconstruct_from_stl)

    def _build_toolbar(self) -> None:
        tb = self.addToolBar("Datasets")
        tb.setMovable(False)
        tb.addWidget(QtWidgets.QLabel("Dataset: "))
        self.dataset_combo = QtWidgets.QComboBox()
        self.dataset_combo.setMinimumWidth(240)
        self.dataset_combo.currentTextChanged.connect(self._on_combo_changed)
        tb.addWidget(self.dataset_combo)

    def _build_docks(self) -> None:
        # 3D volume dock. VTK's QtInteractor segfaults under the "offscreen" Qt
        # platform (headless/CI), so allow disabling it via env for testing.
        self.volume_dock = QtWidgets.QDockWidget("3D Volume", self)
        if os.environ.get("VOXELCAST_DISABLE_3D"):
            self.volume_dock.setWidget(
                _placeholder("3D view disabled (VOXELCAST_DISABLE_3D)")
            )
        else:
            try:
                from voxelcast.viewers.volume_view import VolumeView
                self._volume_view = VolumeView()
                self.volume_dock.setWidget(self._volume_view)
            except Exception as e:
                self.volume_dock.setWidget(
                    _placeholder(f"3D view unavailable:\n{e}\n\npip install pyvista pyvistaqt")
                )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.volume_dock)

        # 2D slice dock
        self.slice_dock = QtWidgets.QDockWidget("2D Slices", self)
        try:
            from voxelcast.viewers.slice_view import SliceView
            self._slice_view = SliceView()
            self.slice_dock.setWidget(self._slice_view)
            # Sync the 2D scrub position to the 3D slice-plane indicator.
            if self._volume_view is not None:
                self._slice_view.slice_changed.connect(self._volume_view.set_slice_marker)
        except Exception as e:
            self.slice_dock.setWidget(
                _placeholder(f"2D view unavailable:\n{e}\n\npip install pyqtgraph")
            )
        # 2D slices live in the RIGHT area, tabbed with the 3D view: both show the
        # same dataset, so you swap between them rather than view side by side.
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.slice_dock)
        self.tabifyDockWidget(self.volume_dock, self.slice_dock)

        # Dedicated sinogram / projection view (angle scrubber), tabbed on the right.
        self.sinogram_dock = QtWidgets.QDockWidget("Sinogram / Projections", self)
        try:
            from voxelcast.viewers.sinogram_view import SinogramView
            self._sinogram_view = SinogramView()
            self.sinogram_dock.setWidget(self._sinogram_view)
        except Exception as e:
            self.sinogram_dock.setWidget(
                _placeholder(f"Sinogram view unavailable:\n{e}\n\npip install pyqtgraph")
            )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.sinogram_dock)
        self.tabifyDockWidget(self.volume_dock, self.sinogram_dock)
        self.sinogram_dock.hide()

        # Side-by-side comparison (e.g. target vs recon). Hidden by default;
        # toggle from the View menu. Shares the left area, tabbed.
        self.compare_dock = QtWidgets.QDockWidget("Compare", self)
        try:
            from voxelcast.viewers.compare_view import CompareView
            self._compare_view = CompareView()
            self.compare_dock.setWidget(self._compare_view)
        except Exception as e:
            self.compare_dock.setWidget(
                _placeholder(f"Compare view unavailable:\n{e}\n\npip install pyqtgraph")
            )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.RightDockWidgetArea, self.compare_dock)
        self.tabifyDockWidget(self.volume_dock, self.compare_dock)
        self.compare_dock.hide()

        # Convergence / error plot (bottom, full width). Hidden by default --
        # it's mainly a debugging/reference tool; enable it from the View menu.
        self.error_dock = QtWidgets.QDockWidget("Convergence", self)
        try:
            from voxelcast.viewers.error_view import ErrorView
            self._error_view = ErrorView()
            self.error_dock.setWidget(self._error_view)
        except Exception as e:
            self.error_dock.setWidget(
                _placeholder(f"Convergence plot unavailable:\n{e}\n\npip install pyqtgraph")
            )
        self.addDockWidget(QtCore.Qt.DockWidgetArea.BottomDockWidgetArea, self.error_dock)
        self.error_dock.hide()

        # 3D is the default front tab of the right-hand viewer group.
        self.volume_dock.raise_()

    def _build_stage_flow(self) -> None:
        """The guided Prep → Voxelize → Optimize → Preview flow (central widget)."""
        from voxelcast.widgets.stage_flow import StageFlow
        self.stage = StageFlow()
        self.setCentralWidget(self.stage)
        self.stage.voxelizeRequested.connect(self._flow_voxelize)
        self.stage.optimizeRequested.connect(self._flow_optimize)
        self.stage.rebinRequested.connect(self._flow_rebin)
        self.stage.exportRequested.connect(self._flow_export)
        self.stage.cancelRequested.connect(self._flow_cancel)

    def _show_hardware(self) -> None:
        if not engine_available():
            self.stage.set_hardware("Engine: vamtoolbox not installed")
            return
        try:
            import vamtoolbox as vam
            info = vam.util.hardware.detect_system()
            backend = ("CUDA (astra GPU)" if info.get("cuda")
                       else "Apple Metal (GPU)" if info.get("metal")
                       else "CPU (sparse)")
            self.stage.set_hardware(
                f"Backend: {backend}  ·  CPU {info['cpu_logical']} cores  ·  "
                f"RAM {info['ram_avail_gb']} GB free")
        except Exception as e:
            self.stage.set_hardware(f"Hardware detection failed: {e}")

    def _build_view_menu(self) -> None:
        """A View menu with a show/hide toggle for every panel."""
        view_menu = self.menuBar().addMenu("&View")
        for dock in (self.slice_dock, self.sinogram_dock, self.volume_dock,
                     self.compare_dock, self.error_dock):
            view_menu.addAction(dock.toggleViewAction())

    def _build_statusbar(self) -> None:
        self.progress = QtWidgets.QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setVisible(False)
        self.statusBar().addPermanentWidget(self.progress)

    # ----- dataset registry / display -------------------------------------
    def _unique_name(self, name: str) -> str:
        if name not in self._datasets:
            return name
        i = 2
        while f"{name} ({i})" in self._datasets:
            i += 1
        return f"{name} ({i})"

    def add_dataset(self, ds: Dataset, select: bool = True) -> None:
        """Register a dataset in the selector; optionally show it."""
        ds.name = self._unique_name(ds.name)
        self._datasets[ds.name] = ds
        # Block signals so adding/selecting doesn't fire _on_combo_changed; we
        # call _display() once, explicitly, below (avoids a double render).
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.addItem(ds.name)
        if select:
            self.dataset_combo.setCurrentText(ds.name)
        self.dataset_combo.blockSignals(False)
        if self._compare_view is not None:
            self._compare_view.set_datasets(self._datasets)
        if select:
            self._display(ds)

    # Backwards-compatible alias used by tests / file open.
    def show_dataset(self, ds: Dataset) -> None:
        self.add_dataset(ds, select=True)

    def _on_combo_changed(self, name: str) -> None:
        ds = self._datasets.get(name)
        if ds is not None:
            self._display(ds)

    def _display(self, ds: Dataset) -> None:
        self._current = ds
        self.export_act.setEnabled(True)
        self.setWindowTitle(f"VoxelCast {__version__} — {ds.name}")
        self.statusBar().showMessage(ds.summary())

        is_sino = ds.is_sinogram
        # Sinograms -> dedicated angle-scrubber view; everything else -> slices.
        if is_sino and self._sinogram_view is not None:
            self._sinogram_view.set_dataset(ds)
        elif self._slice_view is not None:
            self._slice_view.set_dataset(ds)
        if self._volume_view is not None and not is_sino:
            self._volume_view.set_dataset(ds)

        self.volume_dock.setVisible(ds.is_volume)
        self.slice_dock.setVisible(not is_sino)
        self.sinogram_dock.setVisible(is_sino)
        # Raise the canonical tab for this dataset; tabs can still be swapped freely.
        if is_sino:
            self.sinogram_dock.raise_()
        elif ds.is_volume:
            self.volume_dock.raise_()
        else:
            self.slice_dock.raise_()

    # ----- file actions ----------------------------------------------------
    def open_file_dialog(self) -> None:
        exts = " ".join(f"*{e}" for e in (*NATIVE_EXTS, *NEUTRAL_EXTS))
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Open volume / sinogram", "", f"VoxelCast data ({exts});;All files (*)"
        )
        if path:
            self.open_path(path)

    def open_path(self, path: str) -> None:
        try:
            ds = load(path)
        except LoadError as e:
            QtWidgets.QMessageBox.critical(self, "Could not open file", str(e))
            return
        self.add_dataset(ds)

    def export_dialog(self) -> None:
        if self._current is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Export (neutral)", self._current.name, "NumPy (*.npy);;TIFF (*.tif)"
        )
        if not path:
            return
        try:
            save_neutral(self._current, path)
            self.statusBar().showMessage(f"Exported to {path}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def load_demo(self) -> None:
        """A synthetic volume so the GUI can be exercised without any files."""
        n = 64
        zz, yy, xx = np.mgrid[0:n, 0:n, 0:n].astype(float)
        r = np.sqrt((xx - n / 2) ** 2 + (yy - n / 2) ** 2 + (zz - n / 2) ** 2)
        blob = np.clip(1.0 - r / (n * 0.45), 0, 1) ** 2
        blob = np.moveaxis(blob, 0, 2)  # (nY, nX, nZ)
        self.add_dataset(Dataset(array=blob, vol_type="recon", name="demo blob"))

    # ----- reconstruction flow --------------------------------------------
    def reconstruct_from_stl(self) -> None:
        if self._thread is not None:
            QtWidgets.QMessageBox.information(
                self, "Busy", "A reconstruction is already running."
            )
            return
        if not engine_available():
            QtWidgets.QMessageBox.critical(
                self, "Engine unavailable",
                "vamtoolbox is not installed in this environment.\n"
                "Install it (pip install -e '.[engine]') to reconstruct from STL/3MF.",
            )
            return

        from voxelcast.widgets.reconstruct_dialog import ReconstructDialog
        dlg = ReconstructDialog(self)
        if dlg.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        params = dlg.params()

        # Stage 1: voxelize on the MAIN thread (STL uses OpenGL). Show target ASAP.
        self.statusBar().showMessage(f"Voxelizing {os.path.basename(params.mesh_path)}…")
        QtWidgets.QApplication.processEvents()
        try:
            from voxelcast.engine.reconstruct import voxelize_mesh, build_proj_and_options
            target_geo, datasets = voxelize_mesh(params)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Voxelization failed", f"{e}")
            self.statusBar().showMessage("Voxelization failed.")
            return
        # Show the target plus any 3MF role volumes (insert / zero_dose).
        for i, ds in enumerate(datasets):
            self.add_dataset(ds, select=(i == 0))
        if len(datasets) > 1:
            roles = ", ".join(d.name for d in datasets[1:])
            self.statusBar().showMessage(f"3MF roles imported: target + {roles}")

        # Stage 2: optimize on a WORKER thread.
        proj_geo, options = build_proj_and_options(params)
        self._start_optimize(target_geo, proj_geo, options, params)

    def _start_optimize(self, target_geo, proj_geo, options, params) -> None:
        self._thread = QtCore.QThread(self)
        self._worker = OptimizeWorker(target_geo, proj_geo, options)
        self._worker.moveToThread(self._thread)

        # Run worker when the thread starts.
        self._thread.started.connect(self._worker.run)
        # UI updates (run on the main thread via queued connections).
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_recon_done)
        self._worker.failed.connect(self._on_recon_failed)
        # Lifecycle: stop the thread's event loop, then let Qt delete both
        # objects safely once control returns to the event loop. Re-enabling the
        # UI / clearing refs happens on the thread's own `finished` signal, which
        # fires only after the event loop has actually exited -- so we never
        # destroy a QThread that is still running (no thread.wait() in a slot).
        self._worker.finished.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._worker.finished.connect(self._worker.deleteLater)
        self._worker.failed.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)

        self.recon_act.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)  # indeterminate until first iteration
        if self._error_view is not None:
            self._error_view.reset()
        self.statusBar().showMessage(
            f"Optimizing ({params.method}, {params.n_iter} iters)…"
        )
        self._thread.start()

    @QtCore.Slot(int, int, float)
    def _on_progress(self, i: int, n: int, loss: float) -> None:
        self.progress.setRange(0, n)
        self.progress.setValue(i)
        if self._error_view is not None:
            self._error_view.add_point(i, loss)
        self.statusBar().showMessage(f"Optimizing… iter {i}/{n}   loss={loss:.4g}")

    @QtCore.Slot(object, object)
    def _on_recon_done(self, recon_ds: Dataset, sino_ds: Dataset) -> None:
        self.add_dataset(sino_ds, select=False)
        self.add_dataset(recon_ds, select=True)
        if self._error_view is not None:
            self._error_view.set_complete()
        self.statusBar().showMessage("Reconstruction complete.")

    @QtCore.Slot(str)
    def _on_recon_failed(self, msg: str) -> None:
        self.statusBar().showMessage("Reconstruction failed.")
        QtWidgets.QMessageBox.critical(self, "Reconstruction failed", msg)

    @QtCore.Slot()
    def _on_thread_finished(self) -> None:
        self.progress.setVisible(False)
        self.recon_act.setEnabled(True)
        self._thread = None
        self._worker = None

    # ----- guided flow (VAMPipeline) --------------------------------------
    def _flow_busy(self) -> bool:
        if self._pp_thread is not None or (self._thread is not None):
            QtWidgets.QMessageBox.information(self, "Busy", "A job is already running.")
            return True
        if not engine_available():
            QtWidgets.QMessageBox.critical(
                self, "Engine unavailable",
                "vamtoolbox is not installed in this environment.")
            return True
        return False

    def _flow_voxelize(self) -> None:
        """Stage 2: build the target on the MAIN thread (OpenGL voxelizer).

        STL models are merged (each with its translate+rotate) into one aligned
        grid; a single 3MF is voxelized with its roles (insert / zero-dose).
        """
        if self._flow_busy():
            return
        models = self.stage.prep.models()
        if not models:
            QtWidgets.QMessageBox.warning(self, "No models", "Add at least one STL or 3MF model.")
            return
        import vamtoolbox as vam
        stl = [m for m in models if m["path"].lower().endswith(".stl")]
        tmf = [m for m in models if m["path"].lower().endswith(".3mf")]
        if tmf and (stl or len(tmf) > 1):
            QtWidgets.QMessageBox.warning(
                self, "Unsupported combination",
                "3MF carries its own bodies/roles, so it can't yet be merged with "
                "other models. Import a single 3MF on its own, or use STL parts.")
            return

        self._cleanup_tmp()
        fields = self.stage.config_dict()

        try:
            if tmf:
                # Single 3MF: voxelize with roles; rotation from its transform.
                m = tmf[0]
                fields["stl_path"] = m["path"]
                self._config, self._pipe = pb.make_pipeline(fields)
                self._apply_hardware_quiet()
                self.stage.voxelize.set_status(
                    f"Voxelizing 3MF at resolution {self._config.res_opt}…")
                QtWidgets.QApplication.processEvents()
                tg = vam.geometry.TargetGeometry(
                    threemffilename=m["path"], resolution=self._config.res_opt,
                    rot_angles=[m["rx"], m["ry"], m["rz"]], bodies="auto")
                self._pipe.target = tg
            else:
                # STL(s): merge with per-model transforms, then voxelize.
                merged, is_temp = pb.merge_meshes(stl)
                self._merged_tmp = merged if is_temp else None
                fields["stl_path"] = merged
                self._config, self._pipe = pb.make_pipeline(fields)
                self._apply_hardware_quiet()
                self.stage.voxelize.set_status(
                    f"Voxelizing {len(stl)} part(s) at resolution {self._config.res_opt}…")
                QtWidgets.QApplication.processEvents()
                tg = vam.geometry.TargetGeometry(
                    stlfilename=merged, resolution=self._config.res_opt)
                tg.insert = None
                self._pipe.target = tg
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Voxelization failed", f"{e}")
            self.stage.voxelize.set_status("Voxelization failed.")
            return

        arr = np.asarray(tg.array)
        self.add_dataset(Dataset(array=arr, vol_type="target", name="target"))
        # Surface any 3MF role volumes (insert / zero-dose) as datasets too.
        for role in ("insert", "zero_dose"):
            ra = getattr(tg, role, None)
            if ra is not None and np.asarray(ra).any():
                self.add_dataset(Dataset(array=np.asarray(ra), vol_type="target",
                                         name=role), select=False)
        self.stage.on_voxelized(arr.shape, int((arr > 0).sum()))
        self.statusBar().showMessage(f"Voxelized: {arr.shape}")

    def _apply_hardware_quiet(self) -> None:
        try:
            self._pipe.apply_hardware()      # fills use_cuda / rebin_jobs
        except Exception:
            pass

    def _flow_optimize(self) -> None:
        if self._pipe is None or self._pipe.target is None:
            QtWidgets.QMessageBox.warning(self, "Voxelize first", "Run the Voxelize stage first.")
            return
        if self._flow_busy():
            return
        # Apply optimize-panel settings onto the live config.
        for k, v in self.stage.optimize.config_dict().items():
            setattr(self._config, k, v)
        if self._error_view is not None:
            self._error_view.reset()
        self.stage.optimize.set_running(True)
        self.statusBar().showMessage(f"Optimizing ({self._config.method})…")
        self._run_stages(["optimize"])

    def _flow_rebin(self) -> None:
        if self._pipe is None or self._pipe.sinogram is None:
            return
        if self._flow_busy():
            return
        self.stage.preview.set_status("Rebinning to printer geometry…")
        self._run_stages(["rebin"])

    def _flow_export(self, opts: dict) -> None:
        if self._pipe is None or self._pipe.sinogram is None:
            return
        if self._flow_busy():
            return
        # Force the OpenCAL filename convention (<part>_<rpm>rpm.mp4).
        self._export_path = pb.opencal_filename(opts["path"], opts["rpm"])
        self.stage.preview.set_status("Rendering projection video…")
        self._run_stages(["video"], video_path=self._export_path,
                         video_kw={"rpm": opts["rpm"], "width": opts["width"],
                                   "height": opts["height"], "rotate": opts["rotate"],
                                   "mirror": opts["mirror"],
                                   "num_loops": opts["num_loops"]})

    def _flow_cancel(self) -> None:
        if self._pp_worker is not None:
            self._pp_worker.cancel()
            self.statusBar().showMessage("Cancelling…")

    def _run_stages(self, stages, video_path=None, video_kw=None) -> None:
        self._pp_thread = QtCore.QThread(self)
        self._pp_worker = pb.PipelineWorker(self._pipe, stages, video_path, video_kw)
        self._pp_worker.moveToThread(self._pp_thread)
        self._pp_thread.started.connect(self._pp_worker.run)
        self._pp_worker.progress.connect(self._on_pp_progress)
        self._pp_worker.stage_done.connect(self._on_pp_stage_done)
        self._pp_worker.failed.connect(self._on_pp_failed)
        self._pp_worker.finished.connect(self._pp_thread.quit)
        self._pp_worker.failed.connect(self._pp_thread.quit)
        self._pp_worker.finished.connect(self._pp_worker.deleteLater)
        self._pp_worker.failed.connect(self._pp_worker.deleteLater)
        self._pp_thread.finished.connect(self._pp_thread.deleteLater)
        self._pp_thread.finished.connect(self._on_pp_thread_finished)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._pp_thread.start()

    @QtCore.Slot(str, float, str)
    def _on_pp_progress(self, stage: str, frac: float, msg: str) -> None:
        self.stage.progress(stage, frac, msg)
        if frac > 0:
            self.progress.setRange(0, 100)
            self.progress.setValue(int(frac * 100))
        self.statusBar().showMessage(f"{stage}: {msg}")
        if stage == "optimize" and self._error_view is not None:
            m = re.search(r"iter (\d+)/(\d+).*dose error ([\d.eE+-]+)", msg)
            if m:
                self._error_view.add_point(int(m.group(1)), float(m.group(3)))

    @QtCore.Slot(str)
    def _on_pp_stage_done(self, stage: str) -> None:
        if stage == "optimize":
            self.add_dataset(pb.sino_dataset(self._pipe), select=False)
            self.add_dataset(pb.recon_dataset(self._pipe), select=True)
            if self._error_view is not None:
                self._error_view.set_complete()
            self.stage.on_optimized(self._pipe.compute_quality(),
                                    self._pipe.timing.get("optimize", 0.0))
        elif stage == "rebin":
            ds = pb.rebinned_dataset(self._pipe)
            self.add_dataset(ds, select=True)
            self.stage.on_rebinned(ds.array.shape)
        elif stage == "video":
            self.stage.on_exported(self._export_path or "")
            self.statusBar().showMessage(f"Exported {self._export_path}")

    @QtCore.Slot(str)
    def _on_pp_failed(self, msg: str) -> None:
        self.stage.optimize.set_running(False)
        if msg == "cancelled":
            self.statusBar().showMessage("Job cancelled.")
            return
        self.statusBar().showMessage("Job failed.")
        QtWidgets.QMessageBox.critical(self, "Pipeline error", msg)

    @QtCore.Slot()
    def _on_pp_thread_finished(self) -> None:
        self.progress.setVisible(False)
        self._pp_thread = None
        self._pp_worker = None

    def _cleanup_tmp(self) -> None:
        if self._merged_tmp and os.path.exists(self._merged_tmp):
            try:
                os.remove(self._merged_tmp)
            except OSError:
                pass
        self._merged_tmp = None

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt signature)
        for th in (self._thread, self._pp_thread):
            if th is not None and th.isRunning():
                if self._pp_worker is not None:
                    self._pp_worker.cancel()
                th.quit()
                th.wait()
        self._cleanup_tmp()
        # Child dock widgets don't get closeEvent when the main window closes, so
        # clean up the 3D view's pop-outs and plotter here -- otherwise leftover
        # VTK windows segfault at teardown on macOS.
        if self._volume_view is not None:
            self._volume_view.cleanup()
        super().closeEvent(event)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    app = QtWidgets.QApplication(argv)
    win = MainWindow()
    win.show()

    file_args = [a for a in argv[1:] if not a.startswith("-")]
    if file_args:
        win.open_path(file_args[0])
    elif "--demo" in argv:
        win.load_demo()

    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
