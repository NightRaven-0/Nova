"""Run openWakeWord training under bleeding-edge scipy/numpy.

openWakeWord's pip-installed trainer pulls in the old `acoustics` package, which
imports names that newer scipy/numpy removed. We restore those names here (in our
OWN code — never editing site-packages) BEFORE openwakeword imports acoustics, then
run `openwakeword.train` as __main__.

Usage (from the Nova root):
    python scripts/wakeword/train_wakeword.py
"""
import runpy
import sys

# --- scipy compat: acoustics does `from scipy.special import sph_harm` and
#     `from scipy.interpolate import interp2d`, both removed in modern scipy.
try:
    import scipy.special as _sp
    if not hasattr(_sp, "sph_harm"):
        _sp.sph_harm = getattr(_sp, "sph_harm_y", lambda *a, **k: None)
except Exception:
    pass
try:
    import scipy.interpolate as _si
    if not hasattr(_si, "interp2d"):
        class _Interp2dStub:  # acoustics imports it but openWakeWord never calls it
            def __init__(self, *a, **k):
                raise NotImplementedError("interp2d removed in scipy>=1.14")
        _si.interp2d = _Interp2dStub
except Exception:
    pass

# --- numpy compat: old acoustics may use aliases removed in numpy>=2.0.
try:
    import numpy as _np
    for _n, _v in [("float", float), ("int", int), ("complex", complex),
                   ("bool", bool), ("object", object), ("str", str)]:
        if not hasattr(_np, _n):
            setattr(_np, _n, _v)
except Exception:
    pass

# --- torchaudio compat: torchaudio>=2.9 routes load() through TorchCodec, which
#     isn't installed (and needs FFmpeg). Fall back to soundfile (already installed).
try:
    import torch as _torch
    import torchaudio as _ta
    import soundfile as _sf

    def _ta_load(uri, frame_offset=0, num_frames=-1, normalize=True,
                 channels_first=True, format=None, buffer_size=4096, backend=None):
        f = _sf.SoundFile(str(uri))
        sr = f.samplerate
        if frame_offset:
            f.seek(int(frame_offset))
        n = int(num_frames) if (num_frames is not None and num_frames > 0) else -1
        data = f.read(frames=n, dtype="float32", always_2d=True)  # (frames, channels)
        f.close()
        arr = data.T if channels_first else data
        return _torch.from_numpy(arr.copy()), sr

    _ta.load = _ta_load

    class _AudioMeta:
        def __init__(self, num_frames, sample_rate, num_channels):
            self.num_frames = num_frames
            self.sample_rate = sample_rate
            self.num_channels = num_channels

    def _ta_info(uri, *a, **k):  # torchaudio.info was removed in 2.9+
        f = _sf.SoundFile(str(uri))
        meta = _AudioMeta(len(f), f.samplerate, f.channels)
        f.close()
        return meta

    _ta.info = _ta_info
except Exception:
    pass

# --- Windows fix: openWakeWord.data.trim_mmap np.loads a memmap then os.remove's it
#     without closing the handle -> WinError 32 on Windows. Replace with a version that
#     closes the memmaps first. (utils imports trim_mmap from data at call time, so
#     patching the data attribute is enough.)
try:
    import torchmetrics  # noqa: F401  import BEFORE speechbrain to avoid a torchmetrics<->k2_fsa clash
    import gc as _gc
    import os as _os3
    import numpy as _np3
    import openwakeword.data as _owd
    from numpy.lib.format import open_memmap as _open_memmap

    def _trim_mmap_winsafe(mmap_path):
        a = _np3.load(mmap_path, mmap_mode="r")
        i = -1
        while _np3.all(a[i, :, :] == 0):
            i -= 1
        n_new = a.shape[0] + i + 1
        tmp = mmap_path[:-4] + "_trim.npy"
        b = _open_memmap(tmp, mode="w+", dtype=_np3.float32,
                         shape=(n_new, a.shape[1], a.shape[2]))
        for j in range(0, n_new, 1024):
            end = min(j + 1024, n_new)
            b[j:end] = a[j:end]
        b.flush()
        for _arr in (b, a):
            try:
                _arr._mmap.close()
            except Exception:
                pass
        del a, b
        _gc.collect()
        _os3.remove(mmap_path)
        _os3.rename(tmp, mmap_path)

    _owd.trim_mmap = _trim_mmap_winsafe
except Exception:
    pass

sys.argv = [
    "openwakeword.train",
    "--training_config", "scripts/wakeword/hey_nova.yaml",
    "--generate_clips", "--augment_clips", "--train_model", "--overwrite",
]
runpy.run_module("openwakeword.train", run_name="__main__")
