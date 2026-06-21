"""VoxelCast <-> VAMToolbox high-level pipeline bridge.

VoxelCast's guided flow (Prep -> Voxelize -> Optimize -> Preview) is driven by
VAMToolbox's `VAMPipeline`/`PrintConfig` -- the same engine path the bundled
"Tomo" GUI uses. This gives us, for free: GPU voxelization, OSMO/BCLP optimize
(on the Metal projector when CUDA is absent), absorption/diffusion correction,
hardware auto-tuning, z-slab memory mode, fan-beam rebin, and printer-ready MP4
export.

Threading rules (same as the rest of the app):
* `voxelize_target()` uses the OpenGL voxelizer and MUST run on the main thread.
* optimize / rebin / video are long and pure compute -> run on a `PipelineWorker`
  QThread, streaming `progress(stage, fraction, message)` back to the UI.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np
from PySide6 import QtCore

from voxelcast.model import Dataset


def engine_available() -> bool:
    try:
        import vamtoolbox  # noqa: F401
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------- #
# Multi-STL handling
# --------------------------------------------------------------------------- #
def merge_stls(paths: list[str]) -> tuple[str, bool]:
    """Merge one or more STLs (no per-model transform) into one aligned mesh.

    Returns (path, is_temp); a single STL is returned unchanged. Thin wrapper
    over merge_meshes for callers that only have paths.
    """
    return merge_meshes([{"path": p} for p in paths])


def _is_identity_xform(m: dict) -> bool:
    return all(abs(float(m.get(k, 0.0))) < 1e-9
               for k in ("tx", "ty", "tz", "rx", "ry", "rz"))


def merge_meshes(models: list[dict]) -> tuple[str, bool]:
    """Merge STL models, applying each model's translate+rotate, into one mesh.

    `models` is a list of {path, tx, ty, tz (mm), rx, ry, rz (deg)}. The meshes
    are transformed then concatenated so they voxelize into one aligned grid.
    Returns (path, is_temp); a single untransformed model is passed through.
    """
    if not models:
        raise ValueError("no models given")
    if len(models) == 1 and _is_identity_xform(models[0]):
        return models[0]["path"], False
    import math
    import trimesh
    meshes = []
    for m in models:
        mesh = trimesh.load(m["path"], force="mesh")
        R = trimesh.transformations.euler_matrix(
            math.radians(float(m.get("rx", 0.0))),
            math.radians(float(m.get("ry", 0.0))),
            math.radians(float(m.get("rz", 0.0))), "rxyz")
        mesh.apply_transform(R)
        mesh.apply_translation([float(m.get("tx", 0.0)),
                                float(m.get("ty", 0.0)),
                                float(m.get("tz", 0.0))])
        meshes.append(mesh)
    combined = trimesh.util.concatenate(meshes)
    fd, tmp = tempfile.mkstemp(suffix=".stl", prefix="voxelcast_merged_")
    os.close(fd)
    combined.export(tmp)
    return tmp, True


# --------------------------------------------------------------------------- #
# Config + pipeline construction
# --------------------------------------------------------------------------- #
def make_pipeline(config_fields: dict, on_progress=None):
    """Build a (PrintConfig, VAMPipeline) from a dict of PrintConfig fields.

    Unknown keys are ignored so the UI can pass a curated subset; everything
    else keeps PrintConfig's defaults.
    """
    import vamtoolbox as vam
    import dataclasses as dc

    valid = {f.name for f in dc.fields(vam.PrintConfig)}
    clean = {k: v for k, v in config_fields.items() if k in valid and v is not None}
    config = vam.PrintConfig(**clean)
    pipe = vam.VAMPipeline(config, on_progress=on_progress)
    return config, pipe


# --------------------------------------------------------------------------- #
# Dataset conversion
# --------------------------------------------------------------------------- #
def target_dataset(pipe, name="target", source_path=None) -> Dataset:
    return Dataset(array=np.asarray(pipe.target.array), vol_type="target",
                   name=name, source_path=source_path)


def recon_dataset(pipe, name="recon") -> Dataset:
    return Dataset(array=np.asarray(pipe.reconstruction.array), vol_type="recon",
                   name=name)


def sino_dataset(pipe, name="sinogram") -> Dataset:
    return Dataset(array=np.asarray(pipe.sinogram.array), vol_type="sino", name=name)


def rebinned_dataset(pipe, name="rebinned (printer)") -> Dataset:
    return Dataset(array=np.asarray(pipe.rebinned.array), vol_type="sino", name=name)


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class PipelineWorker(QtCore.QObject):
    """Runs a sequence of pipeline stages on a QThread.

    Stages are method names on the VAMPipeline: "optimize", "rebin", "video".
    (voxelize runs on the main thread before this worker is started.)

    Signals
    -------
    progress(str, float, str) : (stage, fraction 0..1, message) -- from the
        pipeline's on_progress hook.
    stage_done(str)           : a stage method returned (name).
    finished()                : all requested stages completed.
    failed(str)               : an error (or "cancelled").
    """

    progress = QtCore.Signal(str, float, str)
    stage_done = QtCore.Signal(str)
    finished = QtCore.Signal()
    failed = QtCore.Signal(str)

    def __init__(self, pipe, stages: list[str], video_path: str | None = None,
                 video_kw: dict | None = None) -> None:
        super().__init__()
        self._pipe = pipe
        self._stages = list(stages)
        self._video_path = video_path
        self._video_kw = video_kw or {}
        # Route the pipeline's progress callback through our Qt signal (queued to
        # the GUI thread). Raising PipelineCancelled from here aborts the run.
        pipe.on_progress = self._on_progress

    def _on_progress(self, stage, fraction, message):
        from vamtoolbox.pipeline import PipelineCancelled
        if getattr(self._pipe, "_cancelled", False):
            raise PipelineCancelled(stage)
        self.progress.emit(str(stage), float(fraction), str(message))

    def cancel(self) -> None:
        self._pipe.cancel()

    @QtCore.Slot()
    def run(self) -> None:
        from vamtoolbox.pipeline import PipelineCancelled
        try:
            for stage in self._stages:
                if stage == "optimize":
                    self._pipe.optimize()
                elif stage == "rebin":
                    self._pipe.rebin()
                elif stage == "video":
                    self._pipe.save_video(self._video_path, **self._video_kw)
                else:
                    raise ValueError(f"unknown stage: {stage}")
                self.stage_done.emit(stage)
            self.finished.emit()
        except PipelineCancelled:
            self.failed.emit("cancelled")
        except Exception as e:  # surface to the UI
            self.failed.emit(f"{type(e).__name__}: {e}")
