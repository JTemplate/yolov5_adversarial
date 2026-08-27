#!/usr/bin/env python3
"""
Collect adversarial-patch evaluation results and create CSV + plots.

Expected evaluation structure:
    runs/test_adversarial/<experiment_name>/<timestamped_run>/
        clean_map_stats.txt
        patch_map_stats.txt
        noise_map_stats.txt

By default, the script scans and compares all experiments.
Use --prefix to select experiments by name prefix, or --include to select
an exact list of experiment names. Use --list to display available experiments.

For each selected top-level experiment directory, the script keeps the newest
result and writes:
    analysis/results_summary.csv
    analysis/asr_comparison.png
    analysis/ap_comparison.png
    analysis/asr_by_epoch.png   (if epoch numbers can be inferred)
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


AP_RE = re.compile(
    r"Average Precision\s+\(AP\).*?"
    r"IoU=0\.50:0\.95.*?"
    r"area=\s*all.*?"
    r"\]\s*=\s*([0-9]*\.?[0-9]+)"
)

ASR_RE = re.compile(
    r"Attack success rate.*?"
    r"area=\s*all.*?=\s*([0-9]*\.?[0-9]+)"
)

EPOCH_RE = re.compile(r"(?:^|_)e(\d+)(?:_|$)", re.IGNORECASE)


def parse_stats(path: Path):
    if not path.exists():
        return None, None

    text = path.read_text(encoding="utf-8", errors="ignore")

    ap_match = AP_RE.search(text)
    asr_matches = ASR_RE.findall(text)

    ap = float(ap_match.group(1)) if ap_match else None
    asr = float(asr_matches[-1]) if asr_matches else None

    return ap, asr


def fmt(v):
    return "" if v is None else f"{v:.6f}"


def newest_patch_stats_by_experiment(root: Path):
    grouped = {}

    for stats in root.rglob("patch_map_stats.txt"):
        try:
            rel = stats.relative_to(root)
        except ValueError:
            continue

        if not rel.parts:
            continue

        experiment = rel.parts[0]
        mtime = stats.stat().st_mtime

        if experiment not in grouped or mtime > grouped[experiment][0]:
            grouped[experiment] = (mtime, stats)

    return grouped


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and compare adversarial patch evaluation results."
    )
    parser.add_argument(
        "--root",
        default="runs/test_adversarial",
        help="Root directory containing evaluation experiments",
    )
    parser.add_argument(
        "--out",
        default="analysis",
        help="Directory for CSV and plots",
    )

    filters = parser.add_mutually_exclusive_group()
    filters.add_argument(
        "--prefix",
        default=None,
        help=(
            "Only compare experiments whose top-level directory name "
            "starts with this prefix, e.g. ablation_"
        ),
    )
    filters.add_argument(
        "--include",
        nargs="+",
        default=None,
        metavar="EXPERIMENT",
        help=(
            "Only compare the exact experiment names listed after --include. "
            "Example: --include train20_e1_fullval train20_e5_fullval"
        ),
    )

    parser.add_argument(
        "--list",
        action="store_true",
        help="List discovered experiment names and exit",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if not root.exists():
        raise SystemExit(f"Evaluation root does not exist: {root}")

    grouped_all = newest_patch_stats_by_experiment(root)

    if not grouped_all:
        raise SystemExit(f"No patch_map_stats.txt files found under: {root}")

    if args.list:
        print(f"Experiments found under {root}:")
        for name in sorted(grouped_all):
            print(f"  {name}")
        return

    if args.prefix:
        grouped = {
            name: grouped_all[name]
            for name in sorted(grouped_all)
            if name.startswith(args.prefix)
        }
    elif args.include:
        grouped = {}
        missing = []
        for name in args.include:
            if name in grouped_all:
                grouped[name] = grouped_all[name]
            else:
                missing.append(name)

        if missing:
            print("WARNING: These requested experiments were not found:")
            for name in missing:
                print(f"  - {name}")
            print()
    else:
        grouped = {
            name: grouped_all[name]
            for name in sorted(grouped_all)
        }

    if not grouped:
        print("No experiments matched your selection.")
        print()
        print("Available experiments:")
        for name in sorted(grouped_all):
            print(f"  {name}")
        raise SystemExit(1)

    rows = []

    for experiment, (_, patch_stats) in grouped.items():
        result_dir = patch_stats.parent

        patch_ap, patch_asr = parse_stats(result_dir / "patch_map_stats.txt")
        noise_ap, noise_asr = parse_stats(result_dir / "noise_map_stats.txt")
        clean_ap, _ = parse_stats(result_dir / "clean_map_stats.txt")

        epoch_match = EPOCH_RE.search(experiment)
        epoch = int(epoch_match.group(1)) if epoch_match else None

        if epoch_match:
            group_name = experiment[:epoch_match.start()].rstrip("_")
        else:
            group_name = experiment

        rows.append(
            {
                "experiment": experiment,
                "group": group_name,
                "epoch": epoch,
                "clean_ap50_95": clean_ap,
                "random_ap50_95": noise_ap,
                "random_asr": noise_asr,
                "adversarial_ap50_95": patch_ap,
                "adversarial_asr": patch_asr,
                "result_dir": str(result_dir),
            }
        )

    csv_path = out / "results_summary.csv"
    fieldnames = [
        "experiment",
        "group",
        "epoch",
        "clean_ap50_95",
        "random_ap50_95",
        "random_asr",
        "adversarial_ap50_95",
        "adversarial_asr",
        "result_dir",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            row_out = dict(row)
            for key in [
                "clean_ap50_95",
                "random_ap50_95",
                "random_asr",
                "adversarial_ap50_95",
                "adversarial_asr",
            ]:
                row_out[key] = fmt(row_out[key])
            row_out["epoch"] = "" if row_out["epoch"] is None else row_out["epoch"]
            writer.writerow(row_out)

    labels = [r["experiment"] for r in rows]
    x = np.arange(len(rows))
    width = 0.36

    adv_asr = [
        np.nan if r["adversarial_asr"] is None else r["adversarial_asr"]
        for r in rows
    ]
    random_asr = [
        np.nan if r["random_asr"] is None else r["random_asr"]
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.35), 6))
    ax.bar(x - width / 2, random_asr, width, label="Random patch ASR")
    ax.bar(x + width / 2, adv_asr, width, label="Adversarial patch ASR")
    ax.set_ylabel("ASR")
    ax.set_xlabel("Experiment")
    ax.set_title("Attack Success Rate by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "asr_comparison.png", dpi=180)
    plt.close(fig)

    clean_ap = [
        np.nan if r["clean_ap50_95"] is None else r["clean_ap50_95"]
        for r in rows
    ]
    random_ap = [
        np.nan if r["random_ap50_95"] is None else r["random_ap50_95"]
        for r in rows
    ]
    adv_ap = [
        np.nan if r["adversarial_ap50_95"] is None else r["adversarial_ap50_95"]
        for r in rows
    ]

    width = 0.26
    fig, ax = plt.subplots(figsize=(max(10, len(rows) * 1.35), 6))
    ax.bar(x - width, clean_ap, width, label="Clean AP50:95")
    ax.bar(x, random_ap, width, label="Random patch AP50:95")
    ax.bar(x + width, adv_ap, width, label="Adversarial patch AP50:95")
    ax.set_ylabel("AP50:95")
    ax.set_xlabel("Experiment")
    ax.set_title("AP50:95 by Experiment")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "ap_comparison.png", dpi=180)
    plt.close(fig)

    epoch_rows = [r for r in rows if r["epoch"] is not None]
    groups = {}

    for row in epoch_rows:
        groups.setdefault(row["group"], []).append(row)

    usable_groups = {
        name: sorted(items, key=lambda r: r["epoch"])
        for name, items in groups.items()
        if len(items) >= 2
    }

    if usable_groups:
        fig, ax = plt.subplots(figsize=(9, 6))

        for group_name, items in usable_groups.items():
            valid = [
                (r["epoch"], r["adversarial_asr"])
                for r in items
                if r["adversarial_asr"] is not None
            ]

            if len(valid) >= 2:
                e_vals = [v[0] for v in valid]
                a_vals = [v[1] for v in valid]
                ax.plot(e_vals, a_vals, marker="o", label=group_name)

        ax.set_xlabel("Epoch")
        ax.set_ylabel("Adversarial Patch ASR")
        ax.set_title("ASR vs Epoch")
        ax.set_ylim(0, 1)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "asr_by_epoch.png", dpi=180)
        plt.close(fig)

    print(f"Collected {len(rows)} experiment(s).")
    print(f"CSV: {csv_path}")
    print(f"ASR plot: {out / 'asr_comparison.png'}")
    print(f"AP plot: {out / 'ap_comparison.png'}")

    if usable_groups:
        print(f"Epoch plot: {out / 'asr_by_epoch.png'}")

    print()
    print("Summary:")
    for r in rows:
        print(
            f"- {r['experiment']}: "
            f"Adv ASR={fmt(r['adversarial_asr'])}, "
            f"Random ASR={fmt(r['random_asr'])}, "
            f"Adv AP50:95={fmt(r['adversarial_ap50_95'])}"
        )


if __name__ == "__main__":
    main()