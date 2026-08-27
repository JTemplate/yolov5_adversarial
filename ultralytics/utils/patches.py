"""Small subset of ultralytics.utils.patches used by this repository."""

import torch


def torch_load(*args, **kwargs):
    """Delegate to torch.load while preserving the YOLOv5 call signature."""
    return torch.load(*args, **kwargs)
