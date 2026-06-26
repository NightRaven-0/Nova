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

sys.argv = [
    "openwakeword.train",
    "--training_config", "scripts/wakeword/hey_nova.yaml",
    "--generate_clips", "--augment_clips", "--train_model",
]
runpy.run_module("openwakeword.train", run_name="__main__")
