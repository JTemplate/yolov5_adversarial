import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory containing transfer result folders",
    )

    parser.add_argument(
        "--include",
        nargs="+",
        required=True,
        help="Experiment folders to compare",
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output directory",
    )

    args = parser.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for exp in args.include:
        summary_file = root / exp / "summary.json"

        if not summary_file.exists():
            print(
                f"WARNING: summary.json not found: "
                f"{summary_file}"
            )
            continue

        with open(summary_file, "r") as f:
            data = json.load(f)

        rows.append(
            {
                "experiment": exp,
                "images": data["images"],
                "clean_references":
                    data["clean_reference_detections"],
                "random_asr":
                    data["random_asr"],
                "transfer_asr":
                    data["learned_transfer_asr"],
                "transfer_gain":
                    data["transfer_gain"],
            }
        )

    if not rows:
        raise RuntimeError(
            "No valid transfer experiments found."
        )

    df = pd.DataFrame(rows)

    # -----------------------------------------------------
    # Save CSV
    # -----------------------------------------------------

    csv_path = out_dir / "results_summary.csv"
    df.to_csv(csv_path, index=False)

    print()
    print("=" * 72)
    print(df.to_string(index=False))
    print("=" * 72)

    # -----------------------------------------------------
    # Mean ± Std
    # -----------------------------------------------------

    random = df["random_asr"].to_numpy() * 100
    transfer = df["transfer_asr"].to_numpy() * 100
    gain = df["transfer_gain"].to_numpy() * 100

    stats = {
        "random_asr_mean": random.mean(),
        "random_asr_std": random.std(ddof=1),
        "transfer_asr_mean": transfer.mean(),
        "transfer_asr_std": transfer.std(ddof=1),
        "transfer_gain_mean": gain.mean(),
        "transfer_gain_std": gain.std(ddof=1),
    }

    with open(
        out_dir / "mean_std.json",
        "w",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

    print()
    print("Mean ± Std")
    print("-" * 40)

    print(
        f"Random ASR   : "
        f"{random.mean():.2f} ± "
        f"{random.std(ddof=1):.2f}%"
    )

    print(
        f"Transfer ASR : "
        f"{transfer.mean():.2f} ± "
        f"{transfer.std(ddof=1):.2f}%"
    )

    print(
        f"Transfer Gain: "
        f"{gain.mean():.2f} ± "
        f"{gain.std(ddof=1):.2f} pp"
    )

    # -----------------------------------------------------
    # ASR comparison
    # -----------------------------------------------------

    labels = df["experiment"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        x - width / 2,
        random,
        width,
        label="Random Patch",
    )

    ax.bar(
        x + width / 2,
        transfer,
        width,
        label="YOLOv5 Adversarial Patch",
    )

    ax.set_ylabel("ASR (%)")
    ax.set_title(
        "YOLOv5 → Faster R-CNN Zero-shot Transfer"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        out_dir / "asr_comparison.png",
        dpi=200,
    )

    plt.close(fig)

    # -----------------------------------------------------
    # Transfer Gain
    # -----------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.bar(
        labels,
        gain,
    )

    ax.set_ylabel(
        "Transfer Gain (percentage points)"
    )

    ax.set_title(
        "Adversarial Patch Gain over Random Patch"
    )

    fig.tight_layout()

    fig.savefig(
        out_dir / "transfer_gain.png",
        dpi=200,
    )

    plt.close(fig)

    print()
    print("Saved to:", out_dir)


if __name__ == "__main__":
    main()
