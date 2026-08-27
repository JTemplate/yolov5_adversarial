#!/usr/bin/env python3
"""Conditional/multivariable attackability analysis for Vehicle4 v2.

This script consumes object_attackability.csv.  Each row is a clean reference
detection with 15 learned and 15 random attack trials.  It fits grouped
binomial-logit models, clusters uncertainty by image, and ranks factors by
leave-one-factor-out deviance and grouped cross-validated log loss.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import norm
from sklearn.model_selection import GroupKFold

DEFAULT_INPUT = Path("experiment_data/vehicle4_v2/analysis/attackability/object_attackability.csv")
DEFAULT_OUTPUT = Path("experiment_data/vehicle4_v2/analysis/attackability/multivariable")
MODEL_LEVELS = ["yolov5s", "yolov5m", "fasterrcnn"]
CLASS_LEVELS = ["car", "van", "truck", "bus"]
CONF_LEVELS = ["0.40–<0.50", "0.50–<0.60", "0.60–<0.70", "0.70–<0.80", "0.80–<0.90", ">=0.90"]
AREA_LEVELS = ["<100 px2", "100–<250 px2", "250–<500 px2", "500–<1,000 px2", "1,000–<2,500 px2", ">=2,500 px2"]
DENSITY_LEVELS = [
    "1–10 detections/image",
    "11–20 detections/image",
    "21–30 detections/image",
    "31–40 detections/image",
    ">40 detections/image",
]
FACTOR_LEVELS = {
    "class": CLASS_LEVELS,
    "clean_confidence": CONF_LEVELS,
    "bbox_area": AREA_LEVELS,
    "image_density": DENSITY_LEVELS,
    "model": MODEL_LEVELS,
}
FACTOR_COLUMNS = {
    "class": "class",
    "clean_confidence": "clean_confidence_bucket",
    "bbox_area": "bbox_area",
    "image_density": "image_density_bucket",
    "model": "model",
}
FACTOR_LABELS = {
    "class": "Class",
    "clean_confidence": "Clean confidence",
    "bbox_area": "BBox area",
    "image_density": "Image density",
    "model": "Detector model",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--folds", type=int, default=5)
    return p.parse_args()


def design_matrix(frame, factors, stacked=False):
    base_cols, base_terms, base_factors = [np.ones(len(frame))], ["Intercept"], ["intercept"]
    for factor in factors:
        source = FACTOR_COLUMNS[factor]
        for level in FACTOR_LEVELS[factor][1:]:
            base_cols.append((frame[source].astype(str).to_numpy() == level).astype(float))
            safe = level.replace(" ", "_").replace("–", "-").replace(">=", "ge")
            base_terms.append(f"{factor}={safe}")
            base_factors.append(factor)
    cols, terms, term_factors = list(base_cols), list(base_terms), list(base_factors)
    if stacked:
        kind = frame["kind"].to_numpy(float)
        cols.append(kind)
        terms.append("kind=learned")
        term_factors.append("kind")
        for i in range(1, len(base_terms)):
            cols.append(base_cols[i] * kind)
            terms.append(f"kind=learned:{base_terms[i]}")
            term_factors.append(f"interaction:{base_factors[i]}")
    return np.column_stack(cols), terms, term_factors


def fit_binomial(X, successes, trials):
    successes, trials = successes.astype(float), trials.astype(float)

    def objective(beta):
        eta = X @ beta
        value = np.sum(trials * np.logaddexp(0.0, eta) - successes * eta)
        gradient = X.T @ (trials * expit(eta) - successes)
        return float(value), gradient

    result = minimize(
        lambda b: objective(b),
        np.zeros(X.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success:
        print(f"warning: optimizer status={result.message}")
    beta = result.x
    p = np.clip(expit(X @ beta), 1e-9, 1 - 1e-9)
    w = np.maximum(trials * p * (1 - p), 1e-9)
    bread = np.linalg.pinv(X.T @ (w[:, None] * X))
    return beta, p, bread


def clustered_covariance(X, successes, trials, p, bread, clusters):
    residual = successes.astype(float) - trials.astype(float) * p
    scores = X * residual[:, None]
    clusters = np.asarray(clusters)
    meat = np.zeros((X.shape[1], X.shape[1]))
    for cluster in pd.Series(clusters).drop_duplicates():
        score = scores[clusters == cluster].sum(axis=0)
        meat += np.outer(score, score)
    n_clusters, n_obs, n_params = pd.Series(clusters).nunique(), len(X), X.shape[1]
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    return bread @ meat @ bread * correction


def deviance(successes, trials, p):
    successes, failures = successes.astype(float), trials.astype(float) - successes
    p = np.clip(p, 1e-12, 1 - 1e-12)
    out = 0.0
    mask = successes > 0
    out += np.sum(2 * successes[mask] * np.log(successes[mask] / (trials[mask] * p[mask])))
    mask = failures > 0
    out += np.sum(2 * failures[mask] * np.log(failures[mask] / (trials[mask] * (1 - p[mask]))))
    return float(out)


def log_loss(successes, trials, p):
    successes, failures = successes.astype(float), trials.astype(float) - successes
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(np.sum(-successes * np.log(p) - failures * np.log1p(-p)) / np.sum(trials))


def fit_summary(frame, factors, outcome, stacked=False):
    X, terms, term_factors = design_matrix(frame, factors, stacked)
    successes = frame["successes"].to_numpy(float) if stacked else frame[outcome].to_numpy(float)
    trials = frame["trials"].to_numpy(float) if stacked else np.full(len(frame), 15.0)
    beta, p, bread = fit_binomial(X, successes, trials)
    cov = clustered_covariance(X, successes, trials, p, bread, frame["image_id"].to_numpy())
    se = np.sqrt(np.maximum(np.diag(cov), 0))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    rows = []
    for i, (term, factor) in enumerate(zip(terms, term_factors)):
        rows.append(
            {
                "outcome": "paired" if stacked else outcome,
                "term": term,
                "factor": factor,
                "coefficient_logit": beta[i],
                "cluster_robust_se": se[i],
                "odds_ratio": math.exp(float(np.clip(beta[i], -50, 50))),
                "ci95_low": math.exp(float(np.clip(beta[i] - 1.96 * se[i], -50, 50))),
                "ci95_high": math.exp(float(np.clip(beta[i] + 1.96 * se[i], -50, 50))),
                "z": z[i],
                "p_value": 2 * norm.sf(abs(z[i])),
            }
        )
    return {
        "beta": beta,
        "p": p,
        "terms": terms,
        "term_factors": term_factors,
        "deviance": deviance(successes, trials, p),
        "rows": rows,
    }


def cv_loss(frame, factors, outcome, folds, stacked=False):
    splitter = GroupKFold(n_splits=folds)
    groups = frame["image_id"].to_numpy()
    losses = []
    for train, test in splitter.split(frame, groups=groups):
        tr, te = frame.iloc[train], frame.iloc[test]
        tx, _, _ = design_matrix(tr, factors, stacked)
        vx, _, _ = design_matrix(te, factors, stacked)
        if stacked:
            tk, vk = tr["successes"].to_numpy(float), te["successes"].to_numpy(float)
            tn, vn = tr["trials"].to_numpy(float), te["trials"].to_numpy(float)
        else:
            tk, vk = tr[outcome].to_numpy(float), te[outcome].to_numpy(float)
            tn, vn = np.full(len(tr), 15.0), np.full(len(te), 15.0)
        beta, _, _ = fit_binomial(tx, tk, tn)
        losses.append(log_loss(vk, vn, expit(vx @ beta)))
    return float(np.mean(losses)), float(np.std(losses, ddof=1) if len(losses) > 1 else 0)


def make_stacked(frame):
    learned = frame.copy()
    learned["kind"] = 1.0
    learned["successes"] = learned["learned_attacked_count"]
    learned["trials"] = learned["condition_pairs"]
    random = frame.copy()
    random["kind"] = 0.0
    random["successes"] = random["random_attacked_count"]
    random["trials"] = random["condition_pairs"]
    return pd.concat([random, learned], ignore_index=True)


def importance(frame, factors, outcome, folds, stacked=False):
    full = fit_summary(frame, factors, outcome, stacked)
    full_cv, full_sd = cv_loss(frame, factors, outcome, folds, stacked)
    rows = []
    for factor in factors:
        reduced = [f for f in factors if f != factor]
        fit = fit_summary(frame, reduced, outcome, stacked)
        cv, cv_sd = cv_loss(frame, reduced, outcome, folds, stacked)
        df = len(full["beta"]) - len(fit["beta"])
        delta = fit["deviance"] - full["deviance"]
        rows.append(
            {
                "outcome": "paired" if stacked else outcome,
                "factor": factor,
                "factor_label": FACTOR_LABELS[factor],
                "full_deviance": full["deviance"],
                "reduced_deviance": fit["deviance"],
                "partial_deviance": delta,
                "degrees_freedom": df,
                "partial_deviance_per_df": delta / max(df, 1),
                "full_cv_log_loss": full_cv,
                "full_cv_log_loss_sd": full_sd,
                "reduced_cv_log_loss": cv,
                "reduced_cv_log_loss_sd": cv_sd,
                "cv_log_loss_increase_without_factor": cv - full_cv,
            }
        )
    return full, rows


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
        for level in FACTOR_LEVELS[factor]:
            row = base.copy()
            row[factor] = level
            x, _, _ = design_matrix(pd.DataFrame([row]), factors)
            rows.append(
                {
                    "outcome": outcome,
                    "varied_factor": factor,
                    "factor_label": FACTOR_LABELS[factor],
                    "level": level,
                    "adjusted_attack_probability": float(expit(x @ fit["beta"])[0]),
                }
            )
    return rows


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = set(FACTOR_COLUMNS.values()) | {
        "image_id",
        "learned_attacked_count",
        "random_attacked_count",
        "condition_pairs",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    for factor, levels in FACTOR_LEVELS.items():
        column = FACTOR_COLUMNS[factor]
        unknown = ~df[column].isin(levels)
        if unknown.any():
            raise ValueError(f"unexpected values in {column}: {sorted(df.loc[unknown, column].unique())}")
        df[column] = df[column].astype(str)
    factors = ["class", "clean_confidence", "bbox_area", "image_density", "model"]
    coeffs, imps, effects = [], [], []
    for outcome in ["learned_attacked_count", "random_attacked_count"]:
        fit, rows = importance(df, factors, outcome, args.folds)
        coeffs += fit["rows"]
        imps += rows
        effects += adjusted_effects(fit, outcome)
    stacked = make_stacked(df)
    fit, rows = importance(stacked, factors, "paired", args.folds, True)
    coeffs += fit["rows"]
    imps += rows
    pd.DataFrame(coeffs).to_csv(args.output / "coefficients.csv", index=False)
    pd.DataFrame(imps).sort_values(["outcome", "partial_deviance"], ascending=[True, False]).to_csv(
        args.output / "factor_importance.csv", index=False
    )
    pd.DataFrame(effects).to_csv(args.output / "conditional_effects.csv", index=False)
    numeric = df[["clean_confidence", "bbox_area_px2", "image_density"]].copy()
    numeric["log_bbox_area"] = np.log1p(numeric.pop("bbox_area_px2"))
    numeric["log_image_density"] = np.log1p(numeric.pop("image_density"))
    numeric.corr().to_csv(args.output / "covariate_correlations.csv")
    manifest = {
        "input": str(args.input),
        "rows": len(df),
        "objects_per_model": {str(k): int(v) for k, v in df.groupby("model").size().items()},
        "model": "grouped-binomial logistic regression with 15 trials per clean object",
        "covariates": {k: FACTOR_LABELS[k] for k in factors},
        "uncertainty": "image_id-clustered sandwich SE; grouped 5-fold CV by image_id",
        "reference_levels": {k: v[0] for k, v in FACTOR_LEVELS.items()},
        "dominance_rule": "partial deviance per df and held-out log-loss increase after removing one factor",
        "caveats": [
            "associational, not causal",
            "detector clean predictions rather than ground truth",
            "size omitted because it is a deterministic coarsening of bbox area",
        ],
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Vehicle4 v2 multivariable attackability analysis",
        "",
        f"Input: `{args.input}` ({len(df):,} clean reference detections; 15 learned + 15 random trials per row).",
        "",
        "The grouped-binomial logit controls class, confidence bucket, bbox-area bucket, image-density bucket, and detector model. Uncertainty is clustered by image. The paired model stacks random and learned counts and includes kind × factor interactions, so its ranking is the most direct test of learned-patch-specific vulnerability. Estimates are conditional associations, not causal effects.",
        "",
    ]
    for outcome, label in [
        ("learned_attacked_count", "Learned attack"),
        ("random_attacked_count", "Random attack"),
        ("paired", "Learned-versus-random paired model"),
    ]:
        imp = pd.DataFrame([r for r in imps if r["outcome"] == outcome]).sort_values(
            "partial_deviance", ascending=False
        )
        lines += [
            f"## {label}: independent factor ranking",
            "",
            "| Rank | Factor | Partial deviance/df | CV log-loss increase without factor |",
            "|---:|---|---:|---:|",
        ]
        for rank, row in enumerate(imp.itertuples(index=False), 1):
            lines.append(
                f"| {rank} | {row.factor_label} | {row.partial_deviance_per_df:.2f} | {row.cv_log_loss_increase_without_factor:.5f} |"
            )
        lines.append("")
    lines += [
        "## Artifacts",
        "",
        "- `coefficients.csv`: image-clustered coefficients, odds ratios, and 95% intervals.",
        "- `factor_importance.csv`: leave-one-factor-out deviance and grouped CV results.",
        "- `conditional_effects.csv`: adjusted one-factor-at-a-time attack probabilities.",
        "- `covariate_correlations.csv`: descriptive correlations (not causal evidence).",
        "",
    ]
    (args.output / "multivariable_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "multivariable_report.md")


if __name__ == "__main__":
    main()
