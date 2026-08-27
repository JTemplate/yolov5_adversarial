"""Confidence-aware hard-target objective for adversarial patch training.

The existing trainer reduces the single largest YOLO candidate score.  This
module keeps that behavior as a stable component, while adding a weighted
top-k pool.  High-confidence candidates receive larger *detached* weights, so
the optimizer spends more gradient on hard detections without creating a
gradient shortcut through the weights.  ``reference_scores`` can optionally
be supplied from a clean forward pass; when omitted, the current scores are a
deliberately cheaper online confidence proxy.
"""

from __future__ import annotations

import torch
from torch import nn


class ConfidenceAwareHardTargetLoss(nn.Module):
    """Pool per-candidate attack scores with confidence-aware hard weighting.

    Args:
        threshold: Confidence at which hard-target weighting turns on.
        temperature: Width of the sigmoid transition around ``threshold``.
        gamma: Focusing exponent; larger values concentrate on the hardest
            candidates.
        max_weight: Upper bound on the detached candidate weight.
        topk: Number of candidates in the hard pool. ``None`` uses all.
        hard_mix: Fraction of the weighted pool mixed with the original max.

    Inputs are scores in ``[0, 1]`` with shape ``[batch, candidates]``.  The
    optional reference tensor has the same shape and is detached internally.
    The returned tensor is one scalar per batch item.
    """

    def __init__(
        self,
        threshold: float = 0.80,
        temperature: float = 0.05,
        gamma: float = 2.0,
        max_weight: float = 4.0,
        topk: int | None = 64,
        hard_mix: float = 0.70,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if max_weight < 1:
            raise ValueError("max_weight must be at least 1")
        if topk is not None and topk < 1:
            raise ValueError("topk must be positive or None")
        if not 0 <= hard_mix <= 1:
            raise ValueError("hard_mix must be in [0, 1]")
        self.threshold = float(threshold)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.max_weight = float(max_weight)
        self.topk = topk
        self.hard_mix = float(hard_mix)

    def forward(self, scores: torch.Tensor, reference_scores: torch.Tensor | None = None) -> torch.Tensor:
        if scores.ndim != 2:
            raise ValueError(f"scores must have shape [batch, candidates], got {tuple(scores.shape)}")
        source = scores.detach() if reference_scores is None else reference_scores.detach()
        if source.shape != scores.shape:
            raise ValueError("reference_scores must have the same shape as scores")
        source = source.clamp(0.0, 1.0)
        scores = scores.clamp(0.0, 1.0)
        count = scores.shape[1] if self.topk is None else min(self.topk, scores.shape[1])
        # Select candidates by clean/reference confidence when available.  In
        # the online fallback this is equivalent to detached current top-k.
        _, indices = torch.topk(source, k=count, dim=1, largest=True, sorted=False)
        selected_scores = torch.gather(scores, 1, indices)
        selected_source = torch.gather(source, 1, indices)
        focus = torch.sigmoid((selected_source - self.threshold) / self.temperature).pow(self.gamma)
        weights = 1.0 + (self.max_weight - 1.0) * focus
        weighted_pool = (weights * selected_scores).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-6)
        original_max = scores.max(dim=1).values
        return (1.0 - self.hard_mix) * original_max + self.hard_mix * weighted_pool

