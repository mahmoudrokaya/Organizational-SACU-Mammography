r"""
Stage2G_Statistical_Confidence_Intervals_and_Model_Comparison.py

Purpose
-------
Compute statistical confidence intervals and paired model-comparison evidence
for the Frontiers Atlam 2026 mammography SACU experiments.

This stage uses exact saved prediction probabilities only. It does NOT repeat:
- preprocessing
- feature extraction
- matrix construction
- model training

The script provides reviewer-facing evidence for:
1. Bootstrap 95% confidence intervals
2. Paired bootstrap differences between SACU and comparator models
3. Approximate two-sided bootstrap p-values
4. Metric stability across clinically relevant operating points

Primary model
-------------
SACU_LearnedShallowMetaFusion

Main comparators
----------------
SACU_FixedEqualWeightFusion
SACU_AdaptiveWeightFusion
LocalRegionalAgent
BilateralAgent
MultiViewAgent
TemporalSpatialAgent
MetadataAgent
AdaptiveControlAgent

If a baseline GradientBoosting prediction file exists, it is also included.

Input
-----
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D2B_Stage2E_Ready_Clean_Predictions.csv

Optional baseline inputs searched automatically:
- Stage2C_Baseline_Test_Predictions.csv
- Stage2E_Reconstructed_Model_Predictions.csv

Outputs
-------
Stage2G_Model_Metric_CI.csv
Stage2G_Paired_Model_Comparison.csv
Stage2G_Primary_Model_Operating_Point_CI.csv
Stage2G_Bootstrap_Settings.json
Stage2G_Statistical_Model_Comparison_Report.txt
Stage2G_ROCAUC_CI_Bar.png
Stage2G_BalancedAccuracy_CI_Bar.png
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

MAIN_INPUT = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"

OPTIONAL_BASELINE_FILES = [
    TABLE_DIR / "Stage2C_Baseline_Test_Predictions.csv",
    TABLE_DIR / "Stage2E_Reconstructed_Model_Predictions.csv",
]

OUT_MODEL_CI = TABLE_DIR / "Stage2G_Model_Metric_CI.csv"
OUT_PAIRWISE = TABLE_DIR / "Stage2G_Paired_Model_Comparison.csv"
OUT_PRIMARY_OP_CI = TABLE_DIR / "Stage2G_Primary_Model_Operating_Point_CI.csv"
OUT_SETTINGS = TABLE_DIR / "Stage2G_Bootstrap_Settings.json"
OUT_SUMMARY = TABLE_DIR / "Stage2G_Statistical_Summary.json"

OUT_FIG_ROC = FIGURE_DIR / "Stage2G_ROCAUC_CI_Bar.png"
OUT_FIG_BA = FIGURE_DIR / "Stage2G_BalancedAccuracy_CI_Bar.png"

OUT_REPORT = REPORT_DIR / "Stage2G_Statistical_Model_Comparison_Report.txt"

PRIMARY_MODEL = "SACU_LearnedShallowMetaFusion"

CORE_MODELS = [
    "SACU_LearnedShallowMetaFusion",
    "SACU_FixedEqualWeightFusion",
    "SACU_AdaptiveWeightFusion",
    "LocalRegionalAgent",
    "BilateralAgent",
    "MultiViewAgent",
    "TemporalSpatialAgent",
    "MetadataAgent",
    "AdaptiveControlAgent",
    "Baseline_GradientBoosting",
]

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05
RANDOM_SEED = 42

DEFAULT_THRESHOLD = 0.50
TARGET_HIGH_SENSITIVITY = 0.90
TARGET_HIGH_SPECIFICITY = 0.90


# =============================================================================
# Input handling
# =============================================================================

def normalize_name(x: str) -> str:
    return "".join(ch for ch in str(x).lower() if ch.isalnum())


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    normalized = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in normalized:
            return normalized[key]
    return None


def normalize_y(y) -> np.ndarray:
    s = pd.Series(y)

    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        unique = sorted(vals.dropna().unique().tolist())

        if set(unique).issubset({0, 1}):
            return vals.astype(int).values

        if set(unique).issubset({1, 2}):
            return (vals == 2).astype(int).values

        if len(unique) == 2:
            return (vals == max(unique)).astype(int).values

        return (vals > 0).astype(int).values

    text = s.astype(str).str.lower().str.strip()

    positives = {"1", "positive", "malignant", "cancer", "abnormal"}
    negatives = {"0", "negative", "benign", "normal"}

    out = []
    for v in text:
        if v in positives or "malig" in v or "cancer" in v:
            out.append(1)
        elif v in negatives or "benign" in v or "normal" in v:
            out.append(0)
        else:
            out.append(np.nan)

    out = pd.Series(out)

    if out.isna().any():
        raise ValueError("Could not normalize labels to binary format.")

    return out.astype(int).values


def load_main_predictions() -> pd.DataFrame:
    if not MAIN_INPUT.exists():
        raise FileNotFoundError(f"Missing main input file: {MAIN_INPUT}")

    df = pd.read_csv(MAIN_INPUT)

    required = {"model", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Main input is missing required columns: {missing}")

    out = df.copy()

    if "sample_index" not in out.columns:
        out["sample_index"] = out.groupby("model").cumcount()

    out["model"] = out["model"].astype(str)
    out["y_true"] = normalize_y(out["y_true"])
    out["y_score"] = pd.to_numeric(out["y_score"], errors="coerce").clip(0, 1)

    out = out.dropna(subset=["model", "sample_index", "y_true", "y_score"])
    out = out[out["y_true"].isin([0, 1])].copy()

    return out[["sample_index", "model", "y_true", "y_score"]]


def try_load_baseline_file(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    y_col = find_column(df, ["y_true", "true_label", "label", "target", "y_test", "actual"])
    score_col = find_column(
        df,
        [
            "y_score",
            "y_prob",
            "probability",
            "pred_proba",
            "positive_probability",
            "malignant_probability",
            "score",
        ],
    )
    model_col = find_column(df, ["model", "model_name", "method", "classifier"])
    index_col = find_column(df, ["sample_index", "index", "test_index", "row_id", "id"])

    if y_col is None or score_col is None:
        return None

    out = pd.DataFrame()

    if index_col is None:
        out["sample_index"] = np.arange(len(df))
    else:
        out["sample_index"] = df[index_col].values

    out["y_true"] = normalize_y(df[y_col])
    out["y_score"] = pd.to_numeric(df[score_col], errors="coerce").clip(0, 1)

    if model_col is not None:
        model_values = df[model_col].astype(str)
        gb_mask = model_values.str.lower().str.contains("gradient", regex=False)

        if gb_mask.any():
            out = out[gb_mask.values].copy()
            out["model"] = "Baseline_GradientBoosting"
        else:
            out["model"] = model_values.values
    else:
        out["model"] = "Baseline_GradientBoosting"

    out = out.dropna(subset=["sample_index", "y_true", "y_score", "model"])
    out = out[["sample_index", "model", "y_true", "y_score"]].copy()

    return out


def load_all_predictions() -> pd.DataFrame:
    main = load_main_predictions()

    frames = [main]

    existing_models = set(main["model"].unique())

    for path in OPTIONAL_BASELINE_FILES:
        base = try_load_baseline_file(path)
        if base is None or base.empty:
            continue

        if "Baseline_GradientBoosting" in existing_models:
            continue

        frames.append(base)

    all_df = pd.concat(frames, ignore_index=True)

    all_df = all_df[all_df["model"].isin(CORE_MODELS)].copy()

    return all_df


def make_wide(df: pd.DataFrame) -> pd.DataFrame:
    labels = (
        df[["sample_index", "y_true"]]
        .drop_duplicates(subset=["sample_index"])
        .sort_values("sample_index")
    )

    pivot = df.pivot_table(
        index="sample_index",
        columns="model",
        values="y_score",
        aggfunc="first",
    ).reset_index()

    wide = labels.merge(pivot, on="sample_index", how="inner")

    return wide


# =============================================================================
# Metric functions
# =============================================================================

def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_score))


def safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_score))


def threshold_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    if len(np.unique(y_true)) < 2:
        return {
            "accuracy": np.nan,
            "balanced_accuracy": np.nan,
            "sensitivity_recall": np.nan,
            "specificity": np.nan,
            "precision_ppv": np.nan,
            "npv": np.nan,
            "f1": np.nan,
            "mcc": np.nan,
        }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


def all_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    out = {
        "roc_auc": safe_auc(y_true, y_score),
        "average_precision": safe_ap(y_true, y_score),
    }
    out.update(threshold_metrics(y_true, y_score, threshold))
    return out


def compute_operating_thresholds(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr

    finite = np.isfinite(thresholds)

    youden = tpr + specificity - 1
    youden_threshold = thresholds[finite][np.argmax(youden[finite])]

    sens_mask = tpr >= TARGET_HIGH_SENSITIVITY
    if sens_mask.any():
        idxs = np.where(sens_mask)[0]
        best = idxs[np.argmax(specificity[idxs])]
        high_sens_threshold = thresholds[best]
    else:
        high_sens_threshold = thresholds[np.argmax(tpr)]

    spec_mask = specificity >= TARGET_HIGH_SPECIFICITY
    if spec_mask.any():
        idxs = np.where(spec_mask)[0]
        best = idxs[np.argmax(tpr[idxs])]
        high_spec_threshold = thresholds[best]
    else:
        high_spec_threshold = thresholds[np.argmax(specificity)]

    def clean(x):
        if not np.isfinite(x):
            return DEFAULT_THRESHOLD
        return float(np.clip(x, 0, 1))

    return {
        "Default_0.50": DEFAULT_THRESHOLD,
        "Youden": clean(youden_threshold),
        "HighSensitivity_0.90": clean(high_sens_threshold),
        "HighSpecificity_0.90": clean(high_spec_threshold),
    }


# =============================================================================
# Bootstrap
# =============================================================================

def bootstrap_indices(
    n: int,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> np.ndarray:
    return rng.integers(0, n, size=(n_bootstrap, n))


def percentile_ci(values: np.ndarray) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        return np.nan, np.nan

    lower = np.percentile(values, 100 * CI_ALPHA / 2)
    upper = np.percentile(values, 100 * (1 - CI_ALPHA / 2))

    return float(lower), float(upper)


def bootstrap_model_ci(wide: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    y_true_all = wide["y_true"].astype(int).values
    n = len(y_true_all)

    boot_idx = bootstrap_indices(n, N_BOOTSTRAP, rng)

    model_cols = [c for c in CORE_MODELS if c in wide.columns]

    rows = []

    for model in model_cols:
        y_score_all = wide[model].astype(float).values

        point = all_metrics(y_true_all, y_score_all, threshold=DEFAULT_THRESHOLD)

        boot_values = {metric: [] for metric in point.keys()}

        for idx in boot_idx:
            yt = y_true_all[idx]
            ys = y_score_all[idx]

            if len(np.unique(yt)) < 2:
                continue

            m = all_metrics(yt, ys, threshold=DEFAULT_THRESHOLD)

            for metric, value in m.items():
                boot_values[metric].append(value)

        for metric, point_value in point.items():
            arr = np.asarray(boot_values[metric], dtype=float)
            ci_low, ci_high = percentile_ci(arr)

            rows.append({
                "model": model,
                "metric": metric,
                "point_estimate": point_value,
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "n_bootstrap_valid": int(np.isfinite(arr).sum()),
                "threshold": DEFAULT_THRESHOLD,
            })

    return pd.DataFrame(rows)


def bootstrap_pairwise_comparison(wide: pd.DataFrame) -> pd.DataFrame:
    if PRIMARY_MODEL not in wide.columns:
        raise ValueError(f"Primary model not found in prediction table: {PRIMARY_MODEL}")

    rng = np.random.default_rng(RANDOM_SEED + 1)

    y_true_all = wide["y_true"].astype(int).values
    primary_score_all = wide[PRIMARY_MODEL].astype(float).values

    n = len(y_true_all)
    boot_idx = bootstrap_indices(n, N_BOOTSTRAP, rng)

    comparators = [
        c for c in CORE_MODELS
        if c in wide.columns and c != PRIMARY_MODEL
    ]

    metrics_to_compare = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "f1",
        "mcc",
    ]

    rows = []

    primary_point = all_metrics(y_true_all, primary_score_all, threshold=DEFAULT_THRESHOLD)

    for comp in comparators:
        comp_score_all = wide[comp].astype(float).values
        comp_point = all_metrics(y_true_all, comp_score_all, threshold=DEFAULT_THRESHOLD)

        diff_boot = {metric: [] for metric in metrics_to_compare}

        for idx in boot_idx:
            yt = y_true_all[idx]
            yp = primary_score_all[idx]
            yc = comp_score_all[idx]

            if len(np.unique(yt)) < 2:
                continue

            mp = all_metrics(yt, yp, threshold=DEFAULT_THRESHOLD)
            mc = all_metrics(yt, yc, threshold=DEFAULT_THRESHOLD)

            for metric in metrics_to_compare:
                diff_boot[metric].append(mp[metric] - mc[metric])

        for metric in metrics_to_compare:
            diffs = np.asarray(diff_boot[metric], dtype=float)
            diffs = diffs[np.isfinite(diffs)]

            ci_low, ci_high = percentile_ci(diffs)

            point_diff = primary_point[metric] - comp_point[metric]

            if diffs.size > 0:
                p_two_sided = 2 * min(
                    np.mean(diffs <= 0),
                    np.mean(diffs >= 0),
                )
                p_two_sided = float(min(1.0, p_two_sided))
            else:
                p_two_sided = np.nan

            rows.append({
                "primary_model": PRIMARY_MODEL,
                "comparison_model": comp,
                "metric": metric,
                "primary_point": primary_point[metric],
                "comparison_point": comp_point[metric],
                "point_difference_primary_minus_comparison": point_diff,
                "ci_lower_95_difference": ci_low,
                "ci_upper_95_difference": ci_high,
                "bootstrap_p_two_sided": p_two_sided,
                "n_bootstrap_valid": int(diffs.size),
                "threshold": DEFAULT_THRESHOLD,
                "interpretation": (
                    "primary_higher"
                    if point_diff > 0
                    else "comparison_higher_or_equal"
                ),
            })

    return pd.DataFrame(rows)


def bootstrap_primary_operating_points(wide: pd.DataFrame) -> pd.DataFrame:
    if PRIMARY_MODEL not in wide.columns:
        raise ValueError(f"Primary model not found: {PRIMARY_MODEL}")

    rng = np.random.default_rng(RANDOM_SEED + 2)

    y_true_all = wide["y_true"].astype(int).values
    y_score_all = wide[PRIMARY_MODEL].astype(float).values

    operating_thresholds = compute_operating_thresholds(y_true_all, y_score_all)

    n = len(y_true_all)
    boot_idx = bootstrap_indices(n, N_BOOTSTRAP, rng)

    rows = []

    for op_name, threshold in operating_thresholds.items():
        point = all_metrics(y_true_all, y_score_all, threshold=threshold)

        boot_values = {metric: [] for metric in point.keys()}

        for idx in boot_idx:
            yt = y_true_all[idx]
            ys = y_score_all[idx]

            if len(np.unique(yt)) < 2:
                continue

            m = all_metrics(yt, ys, threshold=threshold)

            for metric, value in m.items():
                boot_values[metric].append(value)

        for metric, point_value in point.items():
            arr = np.asarray(boot_values[metric], dtype=float)
            ci_low, ci_high = percentile_ci(arr)

            rows.append({
                "model": PRIMARY_MODEL,
                "operating_point": op_name,
                "threshold": threshold,
                "metric": metric,
                "point_estimate": point_value,
                "ci_lower_95": ci_low,
                "ci_upper_95": ci_high,
                "n_bootstrap_valid": int(np.isfinite(arr).sum()),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def plot_metric_ci(model_ci: pd.DataFrame, metric: str, output_path: Path):
    temp = model_ci[model_ci["metric"] == metric].copy()

    if temp.empty:
        return

    temp = temp.sort_values("point_estimate", ascending=True)

    y = np.arange(len(temp))
    x = temp["point_estimate"].values
    lower = temp["ci_lower_95"].values
    upper = temp["ci_upper_95"].values

    err_low = x - lower
    err_high = upper - x

    plt.figure(figsize=(10, 6))
    plt.barh(y, x)
    plt.errorbar(x, y, xerr=[err_low, err_high], fmt="none", capsize=3)
    plt.yticks(y, temp["model"].values)
    plt.xlabel(metric)
    plt.title(f"Stage2G {metric} with 95% Bootstrap CI")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


# =============================================================================
# Report
# =============================================================================

def write_report(
    wide: pd.DataFrame,
    model_ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    primary_op_ci: pd.DataFrame,
):
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2G STATISTICAL CONFIDENCE INTERVALS AND MODEL COMPARISON")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Main input: {MAIN_INPUT}")
    lines.append(f"Primary model: {PRIMARY_MODEL}")
    lines.append(f"Bootstrap repetitions: {N_BOOTSTRAP}")
    lines.append(f"Confidence interval: {int((1 - CI_ALPHA) * 100)}% percentile bootstrap")
    lines.append("")

    lines.append("AVAILABLE MODELS")
    lines.append("-" * 100)
    model_cols = [c for c in CORE_MODELS if c in wide.columns]
    for model in model_cols:
        lines.append(f"- {model}")
    lines.append("")

    lines.append("ROC-AUC CONFIDENCE INTERVALS")
    lines.append("-" * 100)
    lines.append(
        model_ci[model_ci["metric"] == "roc_auc"]
        .sort_values("point_estimate", ascending=False)
        .to_string(index=False)
    )
    lines.append("")

    lines.append("BALANCED ACCURACY CONFIDENCE INTERVALS")
    lines.append("-" * 100)
    lines.append(
        model_ci[model_ci["metric"] == "balanced_accuracy"]
        .sort_values("point_estimate", ascending=False)
        .to_string(index=False)
    )
    lines.append("")

    lines.append("PRIMARY MODEL PAIRWISE ROC-AUC COMPARISON")
    lines.append("-" * 100)
    lines.append(
        pairwise[pairwise["metric"] == "roc_auc"]
        .sort_values("point_difference_primary_minus_comparison", ascending=False)
        .to_string(index=False)
    )
    lines.append("")

    lines.append("PRIMARY MODEL OPERATING-POINT CI")
    lines.append("-" * 100)
    lines.append(
        primary_op_ci[
            primary_op_ci["metric"].isin(
                ["balanced_accuracy", "sensitivity_recall", "specificity", "f1", "mcc"]
            )
        ].to_string(index=False)
    )
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUT_MODEL_CI,
        OUT_PAIRWISE,
        OUT_PRIMARY_OP_CI,
        OUT_SETTINGS,
        OUT_SUMMARY,
        OUT_FIG_ROC,
        OUT_FIG_BA,
        OUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_settings_and_summary(
    wide: pd.DataFrame,
    model_ci: pd.DataFrame,
    pairwise: pd.DataFrame,
    primary_op_ci: pd.DataFrame,
):
    settings = {
        "generated": str(datetime.now()),
        "main_input": str(MAIN_INPUT),
        "optional_baseline_files": [str(p) for p in OPTIONAL_BASELINE_FILES],
        "primary_model": PRIMARY_MODEL,
        "core_models": CORE_MODELS,
        "n_bootstrap": N_BOOTSTRAP,
        "ci_alpha": CI_ALPHA,
        "random_seed": RANDOM_SEED,
        "default_threshold": DEFAULT_THRESHOLD,
        "target_high_sensitivity": TARGET_HIGH_SENSITIVITY,
        "target_high_specificity": TARGET_HIGH_SPECIFICITY,
    }

    with open(OUT_SETTINGS, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4)

    roc_rows = (
        model_ci[model_ci["metric"] == "roc_auc"]
        .sort_values("point_estimate", ascending=False)
        .to_dict(orient="records")
    )

    pairwise_roc = (
        pairwise[pairwise["metric"] == "roc_auc"]
        .sort_values("point_difference_primary_minus_comparison", ascending=False)
        .to_dict(orient="records")
    )

    summary = {
        "generated": str(datetime.now()),
        "n_samples": int(len(wide)),
        "positives": int((wide["y_true"] == 1).sum()),
        "negatives": int((wide["y_true"] == 0).sum()),
        "primary_model": PRIMARY_MODEL,
        "roc_auc_ci_ranking": roc_rows,
        "pairwise_roc_auc_primary_comparisons": pairwise_roc,
        "primary_operating_point_ci": primary_op_ci.to_dict(orient="records"),
    }

    with open(OUT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("STAGE 2G STATISTICAL CONFIDENCE INTERVALS AND MODEL COMPARISON")
    print("=" * 100)

    df = load_all_predictions()
    wide = make_wide(df)

    if PRIMARY_MODEL not in wide.columns:
        raise ValueError(f"Primary model not available: {PRIMARY_MODEL}")

    print(f"Samples: {len(wide):,}")
    print(f"Positives: {(wide['y_true'] == 1).sum():,}")
    print(f"Negatives: {(wide['y_true'] == 0).sum():,}")
    print(f"Bootstrap repetitions: {N_BOOTSTRAP:,}")
    print("-" * 100)

    model_ci = bootstrap_model_ci(wide)
    pairwise = bootstrap_pairwise_comparison(wide)
    primary_op_ci = bootstrap_primary_operating_points(wide)

    model_ci.to_csv(OUT_MODEL_CI, index=False, encoding="utf-8-sig")
    pairwise.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")
    primary_op_ci.to_csv(OUT_PRIMARY_OP_CI, index=False, encoding="utf-8-sig")

    plot_metric_ci(model_ci, "roc_auc", OUT_FIG_ROC)
    plot_metric_ci(model_ci, "balanced_accuracy", OUT_FIG_BA)

    save_settings_and_summary(wide, model_ci, pairwise, primary_op_ci)

    write_report(
        wide=wide,
        model_ci=model_ci,
        pairwise=pairwise,
        primary_op_ci=primary_op_ci,
    )

    print()
    print("STAGE 2G COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Model CI table:        {OUT_MODEL_CI}")
    print(f"Pairwise comparison:   {OUT_PAIRWISE}")
    print(f"Primary OP CI table:   {OUT_PRIMARY_OP_CI}")
    print(f"Settings:              {OUT_SETTINGS}")
    print(f"Summary:               {OUT_SUMMARY}")
    print(f"ROC-AUC CI figure:     {OUT_FIG_ROC}")
    print(f"Balanced Acc CI fig:   {OUT_FIG_BA}")
    print(f"Report:                {OUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
