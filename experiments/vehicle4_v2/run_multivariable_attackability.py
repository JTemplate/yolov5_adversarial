#!/usr/bin/env python3
"""Runnable entry point for the Vehicle4 v2 multivariable analysis.

The implementation is kept in ``analyze_multivariable_attackability_v2.py``;
this wrapper supplies the adjusted-effect table using the CSV bucket column
names and makes the command safe to rerun from the repository root.
"""

from __future__ import annotations

import pandas as pd
from scipy.special import expit

from experiments.vehicle4_v2 import analyze_multivariable_attackability_v2 as analysis


def adjusted_effects(fit, outcome):
    factors = ["class", "clean_confidence", "bbox_area", "image_density", "model"]
    base = {
        "class": "car",
        "clean_confidence": "0.70–<0.80",
        "bbox_area": "250–<500 px2",
        "image_density": "31–40 detections/image",
        "model": "yolov5s",
    }
    rows = []
    for factor in factors:
        for level in analysis.FACTOR_LEVELS[factor]:
            row = {analysis.FACTOR_COLUMNS[key]: value for key, value in base.items()}
            row[analysis.FACTOR_COLUMNS[factor]] = level
            x, _, _ = analysis.design_matrix(pd.DataFrame([row]), factors)
            rows.append(
                {
                    "outcome": outcome,
                    "varied_factor": factor,
                    "factor_label": analysis.FACTOR_LABELS[factor],
                    "level": level,
                    "adjusted_attack_probability": float(expit(x @ fit["beta"])[0]),
                }
            )
    return rows


analysis.adjusted_effects = adjusted_effects

if __name__ == "__main__":
    analysis.main()
