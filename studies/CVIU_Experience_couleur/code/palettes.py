"""Five 64-vectors; the combined descriptor uses abs and relative Lab with equal mass."""
import ctypes
from pathlib import Path
import numpy as np
from numpy.ctypeslib import ndpointer

def load(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    routine = library.palettes
    routine.argtypes = [ndpointer(np.uint8, flags="C_CONTIGUOUS"), ctypes.c_int,
                        ndpointer(np.float64, flags="C_CONTIGUOUS")]
    routine.restype = None
    def describe(rgb):
        rgb = np.ascontiguousarray(rgb)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("Expected an RGB uint8 image")
        z = np.empty((5,64),dtype=np.float64)
        routine(rgb,rgb.size//3,z)
        np.testing.assert_allclose((z*z).sum(axis=1),1.,atol=1e-12,rtol=0.)
        return z
    return describe
