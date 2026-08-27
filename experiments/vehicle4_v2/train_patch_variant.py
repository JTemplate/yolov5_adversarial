#!/usr/bin/env python3
"""Train one B0/B1/B2 hard-target patch variant.

The legacy ``train_patch.py`` remains untouched.  This entry point swaps only
the probability extractor before importing the trainer, which keeps all
dataset, transforms, optimizer, regularizers, and checkpoint behavior identical
to the validated baseline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch import nn

from adv_patch_gen.utils.confidence_hard_loss import ConfidenceAwareHardTargetLoss


class VariantMaxProbExtractor(nn.Module):
    """YOLO score extractor for B1 (unweighted TopK) and B2 (weighted TopK)."""

    def __init__(self, config, variant: str):
        super().__init__()
        self.config = config
        if variant == "B1":
            self.pool = ConfidenceAwareHardTargetLoss(
                threshold=0.80,
                temperature=0.05,
                gamma=2.0,
                max_weight=1.0,
                topk=64,
                hard_mix=0.70,
            )
        elif variant == "B2":
            self.pool = ConfidenceAwareHardTargetLoss(
                threshold=0.80,
                temperature=0.05,
                gamma=2.0,
                max_weight=4.0,
                topk=64,
                hard_mix=0.70,
            )
        else:
            raise ValueError(f"Unsupported variant: {variant}")

    def candidate_scores(self, output: torch.Tensor) -> torch.Tensor:
        if output.size(-1) != 5 + self.config.n_classes:
            raise ValueError(f"unexpected YOLO output width {output.size(-1)}; expected {5 + self.config.n_classes}")
        class_confs = output[:, :, 5 : 5 + self.config.n_classes]
        objectness = output[:, :, 4]
        if self.config.objective_class_id is not None:
            class_confs = torch.softmax(class_confs, dim=2)
            class_confs = class_confs[:, :, self.config.objective_class_id]
        else:
            class_confs = torch.max(class_confs, dim=2).values
        return self.config.loss_target(objectness, class_confs)

    def forward(self, output: torch.Tensor) -> torch.Tensor:
        return self.pool(self.candidate_scores(output))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=Path, required=True)
    parser.add_argument("--variant", choices=["B0", "B1", "B2"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # Importing train_patch after the swap is intentional: PatchTrainer looks
    # up MaxProbExtractor when it is instantiated, while the old source file
    # and all B0 behavior remain unchanged.
    import train_patch
    from adv_patch_gen.utils.config_parser import load_config_object

    if args.variant != "B0":
        train_patch.MaxProbExtractor = lambda cfg: VariantMaxProbExtractor(cfg, args.variant)
    cfg = load_config_object(str(args.cfg))
    cfg.loss_variant = args.variant
    trainer = train_patch.PatchTrainer(cfg)
    trainer.train()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
