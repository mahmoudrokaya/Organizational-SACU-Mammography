r"""
Stage2E_Clinical_Threshold_and_SACU_Operating_Point_Analysis.py

Exact-input version.

This version reads:
Stage2D2B_Stage2E_Ready_Clean_Predictions.csv

It performs threshold analysis for:
- SACU_LearnedShallowMetaFusion
- SACU_AdaptiveWeightFusion
- SACU_FixedEqualWeightFusion
- Individual SACU agents
"""

from pathlib import Path
from datetime import datetime
import json

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
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    matthews_corrcoef,
)

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"

OUT_PERFORMANCE = TABLE_DIR / "Stage2E_Exact_Clinical_Operating_Point_Performance.csv"
OUT_THRESHOLDS = TABLE_DIR / "Stage2E_Exact_Threshold_Comparison.csv"
OUT_CM = TABLE_DIR / "Stage2E_Exact_Clinical_Confusion_Matrices.csv"
OUT_ROCPR = TABLE_DIR / "Stage2E_Exact_ROC_PR_Summary.csv"
OUT_COMPARISON = TABLE_DIR / "Stage2E_Exact_SACU_Operating_Point_Comparison.csv"
OUT_JSON = TABLE_DIR / "Stage2E_Exact_Clinical_Threshold_Summary.json"

OUT_ROC_FIG = FIGURE_DIR / "Stage2E_Exact_ROC_Operating_Points.png"
OUT_PR_FIG = FIGURE_DIR / "Stage2E_Exact_PR_Operating_Points.png"
OUT_REPORT = REPORT_DIR / "Stage2E_Exact_Clinical_Threshold_Report.txt"

TARGET_HIGH_SENSITIVITY = 0.90
TARGET_HIGH_SPECIFICITY = 0.90

PRIMARY_MODEL = "SACU_LearnedShallowMetaFusion"


def load_predictions() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"model", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype(int)
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce").clip(0, 1)

    df = df.dropna(subset=["model", "y_true", "y_score"])
    df = df[df["y_true"].isin([0, 1])].copy()

    return df


def compute_thresholds(y_true, y_score):
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
            return 0.5
        return float(np.clip(x, 0, 1))

    return {
        "Default_0.50": 0.50,
        "Youden": clean(youden_threshold),
        "HighSensitivity_0.90": clean(high_sens_threshold),
        "HighSpecificity_0.90": clean(high_spec_threshold),
    }


def metric_at_threshold(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = tp / (tp + fp) if (tp + fp) else 0
    npv = tn / (tn + fn) if (tn + fn) else 0

    return {
        "threshold": float(threshold),
        "n": int(len(y_true)),
        "positives": int((y_true == 1).sum()),
        "negatives": int((y_true == 0).sum()),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_recall": sensitivity,
        "specificity": specificity,
        "precision_ppv": ppv,
        "npv": npv,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0,
    }


def analyze(df):
    performance = []
    thresholds_rows = []
    cm_rows = []
    rocp_rows = []

    for model, g in df.groupby("model"):
        y_true = g["y_true"].astype(int).values
        y_score = g["y_score"].astype(float).values

        roc_auc = roc_auc_score(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        rocp_rows.append({
            "model": model,
            "n": len(y_true),
            "positives": int((y_true == 1).sum()),
            "negatives": int((y_true == 0).sum()),
            "roc_auc": roc_auc,
            "average_precision": ap,
        })

        thresholds = compute_thresholds(y_true, y_score)

        for op, th in thresholds.items():
            m = metric_at_threshold(y_true, y_score, th)

            performance.append({
                "model": model,
                "operating_point": op,
                **m,
                "roc_auc": roc_auc,
                "average_precision": ap,
            })

            thresholds_rows.append({
                "model": model,
                "operating_point": op,
                "threshold": th,
                "roc_auc": roc_auc,
                "average_precision": ap,
            })

            cm_rows.append({
                "model": model,
                "operating_point": op,
                "threshold": th,
                "tn": m["tn"],
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            })

    return (
        pd.DataFrame(performance),
        pd.DataFrame(thresholds_rows),
        pd.DataFrame(cm_rows),
        pd.DataFrame(rocp_rows),
    )


def build_primary_comparison(performance_df):
    rows = []

    if PRIMARY_MODEL not in performance_df["model"].unique():
        primary = performance_df["model"].unique()[0]
    else:
        primary = PRIMARY_MODEL

    primary_df = performance_df[performance_df["model"] == primary]

    for model in sorted(performance_df["model"].unique()):
        if model == primary:
            continue

        other_df = performance_df[performance_df["model"] == model]

        merged = primary_df.merge(
            other_df,
            on="operating_point",
            suffixes=("_primary", "_other"),
        )

        for _, r in merged.iterrows():
            rows.append({
                "operating_point": r["operating_point"],
                "primary_model": primary,
                "comparison_model": model,
                "roc_auc_primary": r["roc_auc_primary"],
                "roc_auc_other": r["roc_auc_other"],
                "delta_roc_auc_primary_minus_other": r["roc_auc_primary"] - r["roc_auc_other"],
                "balanced_accuracy_primary": r["balanced_accuracy_primary"],
                "balanced_accuracy_other": r["balanced_accuracy_other"],
                "delta_balanced_accuracy_primary_minus_other": r["balanced_accuracy_primary"] - r["balanced_accuracy_other"],
                "sensitivity_primary": r["sensitivity_recall_primary"],
                "sensitivity_other": r["sensitivity_recall_other"],
                "delta_sensitivity_primary_minus_other": r["sensitivity_recall_primary"] - r["sensitivity_recall_other"],
                "specificity_primary": r["specificity_primary"],
                "specificity_other": r["specificity_other"],
                "delta_specificity_primary_minus_other": r["specificity_primary"] - r["specificity_other"],
                "f1_primary": r["f1_primary"],
                "f1_other": r["f1_other"],
                "delta_f1_primary_minus_other": r["f1_primary"] - r["f1_other"],
                "mcc_primary": r["mcc_primary"],
                "mcc_other": r["mcc_other"],
                "delta_mcc_primary_minus_other": r["mcc_primary"] - r["mcc_other"],
            })

    return pd.DataFrame(rows)


def plot_roc(df, threshold_df):
    plt.figure(figsize=(9, 7))

    for model, g in df.groupby("model"):
        y_true = g["y_true"].astype(int).values
        y_score = g["y_score"].astype(float).values

        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        linewidth = 2.4 if model == PRIMARY_MODEL else 1.2
        plt.plot(fpr, tpr, linewidth=linewidth, label=f"{model} AUC={auc:.3f}")

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / Sensitivity")
    plt.title("Stage2E Exact ROC Curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_ROC_FIG, dpi=300)
    plt.close()


def plot_pr(df):
    plt.figure(figsize=(9, 7))

    for model, g in df.groupby("model"):
        y_true = g["y_true"].astype(int).values
        y_score = g["y_score"].astype(float).values

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        linewidth = 2.4 if model == PRIMARY_MODEL else 1.2
        plt.plot(recall, precision, linewidth=linewidth, label=f"{model} AP={ap:.3f}")

    plt.xlabel("Recall / Sensitivity")
    plt.ylabel("Precision / PPV")
    plt.title("Stage2E Exact Precision-Recall Curves")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(OUT_PR_FIG, dpi=300)
    plt.close()


def write_report(df, performance_df, threshold_df, cm_df, rocp_df, comparison_df):
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2E EXACT CLINICAL THRESHOLD AND SACU OPERATING-POINT ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Input file: {INPUT_FILE}")
    lines.append("")

    lines.append("MODEL SUMMARY")
    lines.append("-" * 100)
    lines.append(df["model"].value_counts().to_string())
    lines.append("")

    lines.append("ROC / PR SUMMARY")
    lines.append("-" * 100)
    lines.append(rocp_df.sort_values("roc_auc", ascending=False).to_string(index=False))
    lines.append("")

    lines.append("PRIMARY MODEL CLINICAL OPERATING POINTS")
    lines.append("-" * 100)
    lines.append(
        performance_df[performance_df["model"] == PRIMARY_MODEL]
        .to_string(index=False)
    )
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUT_PERFORMANCE,
        OUT_THRESHOLDS,
        OUT_CM,
        OUT_ROCPR,
        OUT_COMPARISON,
        OUT_JSON,
        OUT_ROC_FIG,
        OUT_PR_FIG,
        OUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 100)
    print("STAGE 2E EXACT CLINICAL THRESHOLD AND SACU OPERATING-POINT ANALYSIS")
    print("=" * 100)

    df = load_predictions()

    performance_df, threshold_df, cm_df, rocp_df = analyze(df)
    comparison_df = build_primary_comparison(performance_df)

    performance_df.to_csv(OUT_PERFORMANCE, index=False, encoding="utf-8-sig")
    threshold_df.to_csv(OUT_THRESHOLDS, index=False, encoding="utf-8-sig")
    cm_df.to_csv(OUT_CM, index=False, encoding="utf-8-sig")
    rocp_df.to_csv(OUT_ROCPR, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")

    summary = {
        "generated": str(datetime.now()),
        "input_file": str(INPUT_FILE),
        "primary_model": PRIMARY_MODEL,
        "target_high_sensitivity": TARGET_HIGH_SENSITIVITY,
        "target_high_specificity": TARGET_HIGH_SPECIFICITY,
        "roc_pr_summary": rocp_df.to_dict(orient="records"),
        "primary_operating_points": performance_df[
            performance_df["model"] == PRIMARY_MODEL
        ].to_dict(orient="records"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    plot_roc(df, threshold_df)
    plot_pr(df)

    write_report(df, performance_df, threshold_df, cm_df, rocp_df, comparison_df)

    print()
    print("STAGE 2E EXACT COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Performance: {OUT_PERFORMANCE}")
    print(f"Thresholds:  {OUT_THRESHOLDS}")
    print(f"ROC/PR:      {OUT_ROCPR}")
    print(f"Comparison:  {OUT_COMPARISON}")
    print(f"Report:      {OUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
