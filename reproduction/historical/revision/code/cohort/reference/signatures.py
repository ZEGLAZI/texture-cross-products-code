"""Empreintes d’origine pour les photographies iNaturalist et Oxford."""
import hashlib
import io
import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn


def signatures(raw):
    with Image.open(io.BytesIO(raw)) as im:
        photo = im.convert("RGB")
    gray = photo.convert("L").resize((32, 32), Image.Resampling.LANCZOS)

    def ph(x):
        v = dctn(np.asarray(x, dtype=float), type=2, norm="ortho")[:8, :8].ravel()
        return np.packbits(v > np.median(v[1:])).tobytes().hex()

    return {
        "sha256_jpeg": hashlib.sha256(raw).hexdigest(),
        "sha256_rgb": hashlib.sha256(
            np.asarray(photo.size, dtype="<u4").tobytes() + photo.tobytes()
        ).hexdigest(),
        "phash": ph(gray),
        "phash_miroir": ph(ImageOps.mirror(gray)),
    }
