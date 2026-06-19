"""Run a VAMToolbox optimization off the UI thread.

`optimize()` is a long, blocking, CPU-heavy call; running it on the GUI thread
would freeze the window. `OptimizeWorker` is a QObject designed to be moved to a
QThread; it streams progress via Qt signals and emits the resulting recon +
sinogram as `Dataset`s when done.

This is the optional "in-process" coupling path. Loading saved files (io.load)
needs none of this.
"""
from __future__ import annotations

from PySide6 import QtCore

from voxelcast.model import Dataset


class EngineUnavailable(RuntimeError):
    """vamtoolbox is not importable in this environment."""


def engine_available() -> bool:
    try:
        import vamtoolbox  # noqa: F401
        return True
    except Exception:
        return False


class OptimizeWorker(QtCore.QObject):
    """Wraps vamtoolbox.optimize.optimize. Move to a QThread and call run().

    Signals
    -------
    progress(int, int, float) : (iteration, n_iter, loss)
    finished(object, object)  : (recon Dataset, sinogram Dataset)
    failed(str)               : error message
    """

    progress = QtCore.Signal(int, int, float)
    finished = QtCore.Signal(object, object)
    failed = QtCore.Signal(str)

    def __init__(self, target_geo, proj_geo, options) -> None:
        super().__init__()
        self._target_geo = target_geo
        self._proj_geo = proj_geo
        self._options = options

    @QtCore.Slot()
    def run(self) -> None:
        try:
            import vamtoolbox as vam
        except Exception as e:  # pragma: no cover - environment dependent
            self.failed.emit(f"vamtoolbox not available: {e}")
            return

        try:
            # The optimizers accept an iter_callback(i, n, loss) hook.
            def _cb(i, n, loss):
                self.progress.emit(int(i), int(n), float(loss))

            try:
                self._options.iter_callback = _cb
            except Exception:
                pass  # older Options without the hook: progress just won't stream

            sino, recon, _err = vam.optimize.optimize(
                self._target_geo, self._proj_geo, self._options
            )
            recon_ds = Dataset(array=recon.array, vol_type="recon", name="recon")
            sino_ds = Dataset(array=sino.array, vol_type="sino", name="sinogram")
            self.finished.emit(recon_ds, sino_ds)
        except Exception as e:  # surface any optimizer error to the UI
            self.failed.emit(f"{type(e).__name__}: {e}")
