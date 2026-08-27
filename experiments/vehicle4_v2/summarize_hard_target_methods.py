#!/usr/bin/env python3
"""Aggregate B0/B1/B2 YOLOv5s ASR and confidence buckets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments.vehicle4_v2.analyze_attackability import make_records, summarize

METHODS = ["B0", "B1", "B2"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, default=Path("experiment_data/vehicle4_v2/runs/method_comparison_v1"))
    parser.add_argument(
        "--analysis-root", type=Path, default=Path("experiment_data/vehicle4_v2/analysis/method_comparison_v1")
    )
    return parser.parse_args()


def main():
    args = parse_args()
    args.analysis_root.mkdir(parents=True, exist_ok=True)
    all_buckets, summary_rows = [], []
    for method in METHODS:
        evaluation_root = args.run_root / method / "evaluation"
        records = make_records("yolov5s", evaluation_root, confidence=0.4, iou_threshold=0.5)
        method_analysis = args.analysis_root / method
        method_analysis.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(method_analysis / "object_attackability.csv", index=False)
        buckets = summarize(records, "clean_confidence_bucket")
        for row in buckets:
            row["method"] = method
            all_buckets.append(row)
        learned_count = sum(row["learned_attacked_count"] for row in records)
        random_count = sum(row["random_attacked_count"] for row in records)
        trials = sum(row["condition_pairs"] for row in records)
        high = next(row for row in buckets if row["bucket"] == ">=0.90")
        summary_rows.append(
            {
                "method": method,
                "model": "yolov5s",
                "unique_clean_objects": len(records),
                "reference_trials": trials,
                "learned_asr": learned_count / trials,
                "random_asr": random_count / trials,
                "learned_minus_random_asr_gain": (learned_count - random_count) / trials,
                "high_confidence_bucket": ">=0.90",
                "high_confidence_unique_objects": high["unique_clean_objects"],
                "high_confidence_learned_asr": high["learned_asr"],
                "high_confidence_random_asr": high["random_asr"],
                "high_confidence_learned_minus_random_asr_gain": high["learned_minus_random_asr_gain"],
            }
        )
    summary_path = args.analysis_root / "method_summary.csv"
    buckets_path = args.analysis_root / "confidence_bucket_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(all_buckets).to_csv(buckets_path, index=False)
    manifest = {
        "methods": METHODS,
        "model": "yolov5s",
        "confidence_threshold": 0.4,
        "iou_threshold": 0.5,
        "conditions_per_method": 15,
        "primary_bucket": ">=0.90",
        "method_summary": str(summary_path),
        "confidence_bucket_summary": str(buckets_path),
        "source": "analyze_attackability.make_records; same greedy same-class IoU protocol",
    }
    (args.analysis_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
