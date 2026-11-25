"""Square-root histograms on bicubically resized RGB images."""
import numpy as np
from PIL import Image

def describe(image):
    resized = image.convert("RGB").resize((224, 224), Image.Resampling.BICUBIC)
    rgb = np.asarray(resized, dtype=np.int32) // 64
    color_index = rgb[:, :, 0]*16 + rgb[:, :, 1]*4 + rgb[:, :, 2]
    gray_index = np.asarray(resized.convert("L"), dtype=np.int32) // 4
    total = resized.width * resized.height
    return [np.sqrt(np.bincount(index.ravel(), minlength=64) / total) for index in (color_index, gray_index)]
