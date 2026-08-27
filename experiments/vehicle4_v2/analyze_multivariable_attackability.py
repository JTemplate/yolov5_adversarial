#!/usr/bin/env python3
"""Multivariable and conditional analysis of Vehicle4 v2 attackability.

The input is the object-level table produced by ``analyze_attackability.py``.
Each row is a clean detector prediction and contains 15 learned and 15 random
paired trials. We fit grouped-binomial logistic models (success = attack) and
use image-clustered sandwich standard errors. Leave-one-factor-out deviance
and grouped cross-validated log loss quantify independent importance.
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
DENSITY_LEVELS = ["1–10 detections/image", "11–20 detections/image", "21–30 detections/image", "31–40 detections/image", ">40 detections/image"]

FACTOR_LEVELS = {
    "class": CLASS_LEVELS,
    "clean_confidence": CONF_LEVELS,
    "bbox_area": AREA_LEVELS,
    "image_density": DENSITY_LEVELS,
    "model": MODEL_LEVELS,
}
FACTOR_LABELS = {
    "class": "Class",
    "clean_confidence": "Clean confidence",
    "bbox_area": "BBox area",
    "image_density": "Image density",
    "model": "Detector model",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args()


def design_matrix(frame: pd.DataFrame, factors: list[str], stacked: bool = False):
    """Return X, term names, and term-to-factor mapping for fixed dummies."""
    base_cols = [np.ones(len(frame), dtype=float)]
    base_terms = ["Intercept"]
    base_factors = ["intercept"]
    for factor in factors:
        for level in FACTOR_LEVELS[factor][1:]:
            base_cols.append((frame[factor].astype(str).to_numpy() == level).astype(float))
            safe = level.replace(" ", "_").replace("–", "-").replace(">=", "ge")
            base_terms.append(f"{factor}={safe}")
            base_factors.append(factor)
    cols, terms, term_factors = list(base_cols), list(base_terms), list(base_factors)
    if stacked:
        kind = frame["kind"].to_numpy(dtype=float)
        cols.append(kind)
        terms.append("kind=learned")
        term_factors.append("kind")
        for idx in range(1, len(base_terms)):
            factor = base_factors[idx]
            cols.append(base_cols[idx] * kind)
            terms.append(f"kind=learned:{base_terms[idx]}")
            term_factors.append(f"interaction:{factor}")
    return np.column_stack(cols), terms, term_factors


def fit_binomial(X: np.ndarray, successes: np.ndarray, trials: np.ndarray):
    """Fit a grouped-binomial logit without requiring statsmodels."""
    successes = successes.astype(float)
    trials = trials.astype(float)

    def objective(beta):
        eta = X @ beta
        value = np.sum(trials * np.logaddexp(0.0, eta) - successes * eta)
        gradient = X.T @ (trials * expit(eta) - successes)
        return float(value), gradient

    result = minimize(
        lambda beta: objective(beta), np.zeros(X.shape[1]), jac=True,
        method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-11, "gtol": 1e-8},
    )
    if not result.success:
        print(f"warning: optimizer status={result.message}")
    beta = result.x
    p = np.clip(expit(X @ beta), 1e-9, 1 - 1e-9)
    w = np.maximum(trials * p * (1.0 - p), 1e-9)
    bread = np.linalg.pinv(X.T @ (w[:, None] * X))
    return beta, p, bread, result


def clustered_covariance(X, successes, trials, p, bread, clusters):
    residual = successes.astype(float) - trials.astype(float) * p
    scores = X * residual[:, None]
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    clusters = np.asarray(clusters)
    for cluster in pd.Series(clusters).drop_duplicates():
        score = scores[clusters == cluster].sum(axis=0)
        meat += np.outer(score, score)
    cov = bread @ meat @ bread
    n_clusters = pd.Series(clusters).nunique()
    n_obs, n_params = X.shape
    correction = (n_clusters / max(n_clusters - 1, 1)) * ((n_obs - 1) / max(n_obs - n_params, 1))
    return cov * correction


def deviance(successes, trials, p):
    successes = successes.astype(float)
    failures = trials.astype(float) - successes
    p = np.clip(p, 1e-12, 1 - 1e-12)
    out = 0.0
    mask = successes > 0
    out += np.sum(2.0 * successes[mask] * np.log(successes[mask] / (trials[mask] * p[mask])))
    mask = failures > 0
    out += np.sum(2.0 * failures[mask] * np.log(failures[mask] / (trials[mask] * (1.0 - p[mask]))))
    return float(out)


def log_loss(successes, trials, p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    successes = successes.astype(float)
    failures = trials.astype(float) - successes
    return float(np.sum(-successes * np.log(p) - failures * np.log1p(-p)) / np.sum(trials))


def fit_summary(frame, factors, outcome, stacked=False):
    X, terms, term_factors = design_matrix(frame, factors, stacked=stacked)
    successes = frame["successes"].to_numpy(float) if stacked else frame[outcome].to_numpy(float)
    trials = frame["trials"].to_numpy(float) if stacked else np.full(len(frame), 15.0)
    beta, p, bread, result = fit_binomial(X, successes, trials)
    cov = clustered_covariance(X, successes, trials, p, bread, frame["image_id"].to_numpy())
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvalue = 2.0 * norm.sf(np.abs(z))
    rows = []
    for idx, (term, factor) in enumerate(zip(terms, term_factors)):
        rows.append({
            "outcome": "paired" if stacked else outcome,
            "term": term,
            "factor": factor,
            "coefficient_logit": beta[idx],
            "cluster_robust_se": se[idx],
            "odds_ratio": math.exp(float(np.clip(beta[idx], -50, 50))),
            "ci95_low": math.exp(float(np.clip(beta[idx] - 1.96 * se[idx], -50, 50))),
            "ci95_high": math.exp(float(np.clip(beta[idx] + 1.96 * se[idx], -50, 50))),
            "z": z[idx], "p_value": pvalue[idx],
        })
    return {"X": X, "terms": terms, "term_factors": term_factors, "beta": beta, "p": p,
            "successes": successes, "trials": trials, "deviance": deviance(successes, trials, p),
            "rows": rows, "result": result}


def cv_loss(frame, factors, outcome, folds, stacked=False):
    splitter = GroupKFold(n_splits=folds)
    groups = frame["image_id"].to_numpy()
    losses = []
    for train, test in splitter.split(frame, groups=groups):
        train_frame, test_frame = frame.iloc[train], frame.iloc[test]
        train_X, _, _ = design_matrix(train_frame, factors, stacked=stacked)
        test_X, _, _ = design_matrix(test_frame, factors, stacked=stacked)
        if stacked:
            train_k, test_k = train_frame["successes"].to_numpy(float), test_frame["successes"].to_numpy(float)
            train_n, test_n = train_frame["trials"].to_numpy(float), test_frame["trials"].to_numpy(float)
        else:
            train_k, test_k = train_frame[outcome].to_numpy(float), test_frame[outcome].to_numpy(float)
            train_n, test_n = np.full(len(train_frame), 15.0), np.full(len(test_frame), 15.0)
        beta, _, _, _ = fit_binomial(train_X, train_k, train_n)
        test_p = np.clip(expit(test_X @ beta), 1e-9, 1 - 1e-9)
        losses.append(log_loss(test_k, test_n, test_p))
    return float(np.mean(losses)), float(np.std(losses, ddof=1) if len(losses) > 1 else 0.0)


def make_stacked(frame: pd.DataFrame) -> pd.DataFrame:
    learned = frame.copy()
    learned["kind"], learned["successes"], learned["trials"] = 1.0, learned["learned_attacked_count"], learned["condition_pairs"]
    random = frame.copy()
    random["kind"], random["successes"], random["trials"] = 0.0, random["random_attacked_count"], random["condition_pairs"]
    return pd.concat([random, learned], ignore_index=True)


def factor_importance(frame, outcome, folds):
    factors = ["class", "clean_confidence", "bbox_area", "image_density", "model"]
    full = fit_summary(frame, factors, outcome)
    full_cv, full_cv_sd = cv_loss(frame, factors, outcome, folds)
    rows = []
    for factor in factors:
        reduced = [f for f in factors if f != factor]
        fit = fit_summary(frame, reduced, outcome)
        cv, cv_sd = cv_loss(frame, reduced, outcome, folds)
        df = len(full["beta"]) - len(fit["beta"])
        rows.append({"outcome": outcome, "factor": factor, "factor_label": FACTOR_LABELS[factor],
                     "full_deviance": full["deviance"], "reduced_deviance": fit["deviance"],
                     "partial_deviance": fit["deviance"] - full["deviance"], "degrees_freedom": df,
                     "partial_deviance_per_df": (fit["deviance"] - full["deviance"]) / max(df, 1),
                     "full_cv_log_loss": full_cv, "full_cv_log_loss_sd": full_cv_sd,
                     "reduced_cv_log_loss": cv, "reduced_cv_log_loss_sd": cv_sd,
                     "cv_log_loss_increase_without_factor": cv - full_cv})
    return full, rows


def paired_importance(frame, folds):
    factors = ["class", "clean_confidence", "bbox_area", "image_density", "model"]
    stacked = make_stacked(frame)
    full = fit_summary(stacked, factors, outcome="paired", stacked=True)
    full_cv, full_cv_sd = cv_loss(stacked, factors, "paired", folds, stacked=True)
    rows = []
    for factor in factors:
        reduced = [f for f in factors if f != factor]
        fit = fit_summary(stacked, reduced, outcome="paired", stacked=True)
        cv, cv_sd = cv_loss(stacked, reduced, "paired", folds, stacked=True)
        df = len(full["beta"]) - len(fit["beta"])
        rows.append({"outcome": "paired", "factor": factor, "factor_label": FACTOR_LABELS[factor],
                     "full_deviance": full["deviance"], "reduced_deviance": fit["deviance"],
                     "partial_deviance": fit["deviance"] - full["deviance"], "degrees_freedom": df,
                     "partial_deviance_per_df": (fit["deviance"] - full["deviance"]) / max(df, 1),
                     "full_cv_log_loss": full_cv, "full_cv_log_loss_sd": full_cv_sd,
                     "reduced_cv_log_loss": cv, "reduced_cv_log_loss_sd": cv_sd,
                     "cv_log_loss_increase_without_factor": cv - full_cv})
    return stacked, full, rows


def conditional_effects(full_fit, outcome):
    factors = ["class", "clean_confidence", "bbox_area", "image_density", "model"]
    base = {"class": "car", "clean_confidence": "0.70–<0.80", "bbox_area": "250–<500 px2",
            "image_density": "31–40 detections/image", "model": "yolov5s"}
    rows = []
    for factor in factors:
        for level in FACTOR_LEVELS[factor]:
            row = base.copy(); row[factor] = level
            X, _, _ = design_matrix(pd.DataFrame([row]), factors)
            pred = float(expit(X @ full_fit["beta"])[0])
            rows.append({"outcome": outcome, "varied_factor": factor, "factor_label": FACTOR_LABELS[factor],
                         "level": level, "adjusted_attack_probability": pred})
    return rows


def main() -> None:
    args = parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input)
    required = set(FACTOR_LEVELS) | {"image_id", "learned_attacked_count", "random_attacked_count", "condition_pairs"}
    missing = required - set(df.columns)
    if missing: raise ValueError(f"missing columns: {sorted(missing)}")
    for column, levels in FACTOR_LEVELS.items():
        unknown = ~df[column].isin(levels)
        if unknown.any(): raise ValueError(f"unexpected values in {column}: {sorted(df.loc[unknown, column].unique())}")
        df[column] = df[column].astype(str)

    coeffs, importances, effects = [], [], []
    for outcome in ["learned_attacked_count", "random_attacked_count"]:
        full, rows = factor_importance(df, outcome, args.folds)
        coeffs.extend(full["rows"]); importances.extend(rows); effects.extend(conditional_effects(full, outcome))
    stacked, paired_fit, paired_rows = paired_importance(df, args.folds)
    coeffs.extend(paired_fit["rows"]); importances.extend(paired_rows)

    pd.DataFrame(coeffs).to_csv(args.output / "coefficients.csv", index=False)
    pd.DataFrame(importances).sort_values(["outcome", "partial_deviance"], ascending=[True, False]).to_csv(args.output / "factor_importance.csv", index=False)
    pd.DataFrame(effects).to_csv(args.output / "conditional_effects.csv", index=False)
    numeric = df[["clean_confidence", "bbox_area_px2", "image_density"]].copy()
    numeric["log_bbox_area"] = np.log1p(numeric.pop("bbox_area_px2")); numeric["log_image_density"] = np.log1p(numeric.pop("image_density"))
    numeric.corr().to_csv(args.output / "covariate_correlations.csv")

    manifest = {"input": str(args.input), "rows": int(len(df)),
                "objects_per_model": {str(k): int(v) for k, v in df.groupby("model", observed=False).size().items()},
                "model": "grouped-binomial logistic regression; 15 paired trials per clean object",
                "outcomes": {"learned_attacked_count": "learned patch attack", "random_attacked_count": "random patch attack", "paired": "stacked learned/random model with kind interactions"},
                "covariates": {"class": "categorical", "clean_confidence": "six fixed buckets", "bbox_area": "six fixed pixel-area buckets", "image_density": "five fixed detections-per-image buckets", "model": "categorical detector fixed effect"},
                "reference_levels": {factor: levels[0] for factor, levels in FACTOR_LEVELS.items()},
                "uncertainty": "image_id-clustered sandwich standard errors; grouped 5-fold CV by image_id",
                "dominance_rule": "rank factors by partial deviance and held-out log-loss increase when the factor is removed; paired model additionally tests factor x learned interactions",
                "caveats": ["objects are detector clean predictions, not ground-truth objects", "bbox area and density are partly scene/model dependent", "additive model is associational; it is not a causal intervention", "size is omitted from the main model because it is a deterministic coarsening of bbox area"]}
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Vehicle4 v2 multivariable attackability analysis", "", f"Input: `{args.input}` ({len(df):,} clean reference detections; 15 learned + 15 random trials per row).", "", "## Model and identification", "", "We fit additive grouped-binomial logistic models to attack counts, with detector model as a fixed effect and image-clustered robust uncertainty. Confidence, bbox area, and image density enter as pre-declared buckets so nonlinear trends are not forced to be linear. The paired model stacks random and learned outcomes and adds `kind × factor` interactions; those interactions test patch-specific differences beyond the random baseline.", "", "The estimates are conditional associations. They do not prove that changing confidence, area, or density would cause a change in ASR, because these variables are properties of the detector/image and are correlated.", ""]
    for outcome, label in [("learned_attacked_count", "Learned attack"), ("random_attacked_count", "Random attack"), ("paired", "Learned-versus-random paired model")]:
        imp = pd.DataFrame([r for r in importances if r["outcome"] == outcome]).sort_values("partial_deviance", ascending=False)
        lines += [f"## {label}: independent factor ranking", "", "| Rank | Factor | Partial deviance/df | CV log-loss increase without factor |", "|---:|---|---:|---:|"]
        for rank, row in enumerate(imp.itertuples(index=False), 1):
            lines.append(f"| {rank} | {row.factor_label} | {row.partial_deviance_per_df:.2f} | {row.cv_log_loss_increase_without_factor:.5f} |" )
        lines.append("")
    lines += ["## Interpretation", "", "The dominant factor is the one that remains most informative after the other dimensions and detector model are controlled. Use the paired ranking for the strongest claim about learned-patch specificity; a factor that ranks highly only for the learned model but not for the paired interaction is a general detector vulnerability rather than a learned-patch-specific failure mode.", "", "Full coefficients (cluster-robust SE and 95% OR intervals) are in `coefficients.csv`; leave-one-factor-out deviance/CV results are in `factor_importance.csv`; adjusted one-factor-at-a-time predictions are in `conditional_effects.csv`; covariate correlations are in `covariate_correlations.csv`.", ""]
    (args.output / "multivariable_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.output / "multivariable_report.md")


if __name__ == "__main__":
    main()
