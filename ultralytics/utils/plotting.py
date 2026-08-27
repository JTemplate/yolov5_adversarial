"""Minimal plotting helpers required during YOLOv5 model import/inference."""

from pathlib import Path


def colors(index, bgr=False):
    palette = ((255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29), (207, 210, 49), (72, 249, 10))
    value = palette[int(index) % len(palette)]
    return value[::-1] if bgr else value


class Annotator:
    """No-op-compatible annotator for code paths not used by patch training."""

    def __init__(self, im, *args, **kwargs):
        self.im = im

    def box_label(self, box, label="", color=(128, 128, 128), *args, **kwargs):
        return self.im

    def result(self):
        return self.im


def save_one_box(xyxy, im, file=Path("im.jpg"), *args, save=True, **kwargs):
    if save:
        try:
            import numpy as np
            from PIL import Image

            array = np.asarray(im)
            x1, y1, x2, y2 = [int(float(v)) for v in xyxy]
            Image.fromarray(array[y1:y2, x1:x2]).save(file)
        except Exception:
            pass
    return im
