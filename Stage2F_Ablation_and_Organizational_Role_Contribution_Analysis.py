r"""
Stage2F_Ablation_and_Organizational_Role_Contribution_Analysis.py

Purpose
-------
Perform post-hoc SACU organizational role ablation using exact Stage2D2B
cleaned probability outputs.

This stage does NOT repeat preprocessing, feature extraction, matrix creation,
or model training.

It evaluates:
1. Full learned SACU fusion
2. Full equal-weight fusion from all agents
3. Leave-one-agent-out equal-weight fusion
4. Individual agent performance
5. Contribution loss relative to full fusion

Input
-----
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D2B_Stage2E_Ready_Clean_Predictions.csv

Outputs
-------
Stage2F_Ablation_Performance.csv
Stage2F_Role_Contribution_Loss.csv
Stage2F_Individual_Agent_Performance.csv
Stage2F_Ablation_ROC_PR_Summary.csv
Stage2F_Ablation_Report.txt
Stage2F_Ablation_BalancedAccuracy_Bar.png
Stage2F_Role_Contribution_Loss_Bar.png
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
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
)

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"

OUT_ABLATION = TABLE_DIR / "Stage2F_Ablation_Performance.csv"
OUT_CONTRIBUTION = TABLE_DIR / "Stage2F_Role_Contribution_Loss.csv"
OUT_AGENT = TABLE_DIR / "Stage2F_Individual_Agent_Performance.csv"
OUT_ROCPR = TABLE_DIR / "Stage2F_Ablation_ROC_PR_Summary.csv"
OUT_JSON = TABLE_DIR / "Stage2F_Ablation_Summary.json"

OUT_FIG_BA = FIGURE_DIR / "Stage2F_Ablation_BalancedAccuracy_Bar.png"
OUT_FIG_LOSS = FIGURE_DIR / "Stage2F_Role_Contribution_Loss_Bar.png"

OUT_REPORT = REPORT_DIR / "Stage2F_Ablation_Report.txt"

PRIMARY_MODEL = "SACU_LearnedShallowMetaFusion"

AGENTS = [
    "LocalRegionalAgent",
    "MultiViewAgent",
    "BilateralAgent",
    "TemporalSpatialAgent",
    "MetadataAgent",
    "AdaptiveControlAgent",
]


def load_predictions() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)

    required = {"model", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Input file is missing required columns: {missing}")

    df = df.copy()
    df["model"] = df["model"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype(int)
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce").clip(0, 1)

    df = df.dropna(subset=["model", "y_true", "y_score"])
    df = df[df["y_true"].isin([0, 1])].copy()

    return df


def wide_agent_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if "sample_index" not in df.columns:
        df = df.copy()
        df["sample_index"] = df.groupby("model").cumcount()

    pivot = df.pivot_table(
        index="sample_index",
        columns="model",
        values="y_score",
        aggfunc="first",
    ).reset_index()

    labels = (
        df[["sample_index", "y_true"]]
        .drop_duplicates(subset=["sample_index"])
        .sort_values("sample_index")
    )

    out = labels.merge(pivot, on="sample_index", how="inner")

    missing_agents = [a for a in AGENTS if a not in out.columns]
    if missing_agents:
        raise ValueError(f"Missing agent columns: {missing_agents}")

    return out


def metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

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
        "mcc": matthews_corrcoef(y_true, y_pred) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": roc_auc_score(y_true, y_score),
        "average_precision": average_precision_score(y_true, y_score),
    }


def evaluate_ablation(wide: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_true = wide["y_true"].astype(int).values

    rows = []

    if PRIMARY_MODEL in wide.columns:
        rows.append({
            "condition": "Full_LearnedShallowMetaFusion",
            "removed_agent": "None",
            "fusion_type": "learned_meta_fusion",
            **metrics(y_true, wide[PRIMARY_MODEL].astype(float).values),
        })

    full_equal = wide[AGENTS].mean(axis=1).values

    rows.append({
        "condition": "Full_EqualWeight_AllAgents",
        "removed_agent": "None",
        "fusion_type": "equal_weight",
        **metrics(y_true, full_equal),
    })

    for removed_agent in AGENTS:
        remaining = [a for a in AGENTS if a != removed_agent]
        score = wide[remaining].mean(axis=1).values

        rows.append({
            "condition": f"Without_{removed_agent}",
            "removed_agent": removed_agent,
            "fusion_type": "leave_one_agent_out_equal_weight",
            **metrics(y_true, score),
        })

    ablation_df = pd.DataFrame(rows)

    agent_rows = []
    for agent in AGENTS:
        agent_rows.append({
            "agent": agent,
            **metrics(y_true, wide[agent].astype(float).values),
        })

    agent_df = pd.DataFrame(agent_rows)

    reference = ablation_df[ablation_df["condition"] == "Full_EqualWeight_AllAgents"].iloc[0]

    contribution_rows = []
    for _, row in ablation_df.iterrows():
        if row["removed_agent"] == "None":
            continue

        contribution_rows.append({
            "removed_agent": row["removed_agent"],
            "reference_condition": "Full_EqualWeight_AllAgents",
            "ablation_condition": row["condition"],
            "delta_roc_auc": reference["roc_auc"] - row["roc_auc"],
            "delta_average_precision": reference["average_precision"] - row["average_precision"],
            "delta_balanced_accuracy": reference["balanced_accuracy"] - row["balanced_accuracy"],
            "delta_f1": reference["f1"] - row["f1"],
            "delta_mcc": reference["mcc"] - row["mcc"],
            "interpretation": (
                "positive_loss_supports_contribution"
                if reference["roc_auc"] - row["roc_auc"] > 0
                else "no_positive_loss"
            ),
        })

    contribution_df = pd.DataFrame(contribution_rows)

    rocp_rows = []
    for _, row in ablation_df.iterrows():
        rocp_rows.append({
            "condition": row["condition"],
            "removed_agent": row["removed_agent"],
            "fusion_type": row["fusion_type"],
            "roc_auc": row["roc_auc"],
            "average_precision": row["average_precision"],
            "balanced_accuracy": row["balanced_accuracy"],
            "f1": row["f1"],
            "mcc": row["mcc"],
        })

    rocp_df = pd.DataFrame(rocp_rows)

    return ablation_df, contribution_df, agent_df, rocp_df


def plot_balanced_accuracy(ablation_df: pd.DataFrame):
    temp = ablation_df.sort_values("balanced_accuracy", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(temp["condition"], temp["balanced_accuracy"])
    plt.xlabel("Balanced Accuracy")
    plt.title("Stage2F SACU Ablation Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(OUT_FIG_BA, dpi=300)
    plt.close()


def plot_contribution_loss(contribution_df: pd.DataFrame):
    temp = contribution_df.sort_values("delta_roc_auc", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(temp["removed_agent"], temp["delta_roc_auc"])
    plt.xlabel("ROC-AUC Loss After Removing Agent")
    plt.title("Stage2F Organizational Role Contribution Loss")
    plt.tight_layout()
    plt.savefig(OUT_FIG_LOSS, dpi=300)
    plt.close()


def write_report(
    ablation_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
    agent_df: pd.DataFrame,
    rocp_df: pd.DataFrame,
):
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2F ABLATION AND ORGANIZATIONAL ROLE CONTRIBUTION ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Input file: {INPUT_FILE}")
    lines.append("")

    lines.append("ABLATION PERFORMANCE")
    lines.append("-" * 100)
    lines.append(ablation_df.sort_values("roc_auc", ascending=False).to_string(index=False))
    lines.append("")

    lines.append("ROLE CONTRIBUTION LOSS")
    lines.append("-" * 100)
    lines.append(contribution_df.sort_values("delta_roc_auc", ascending=False).to_string(index=False))
    lines.append("")

    lines.append("INDIVIDUAL AGENT PERFORMANCE")
    lines.append("-" * 100)
    lines.append(agent_df.sort_values("roc_auc", ascending=False).to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUT_ABLATION,
        OUT_CONTRIBUTION,
        OUT_AGENT,
        OUT_ROCPR,
        OUT_JSON,
        OUT_FIG_BA,
        OUT_FIG_LOSS,
        OUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    print("=" * 100)
    print("STAGE 2F ABLATION AND ORGANIZATIONAL ROLE CONTRIBUTION ANALYSIS")
    print("=" * 100)

    df = load_predictions()
    wide = wide_agent_matrix(df)

    ablation_df, contribution_df, agent_df, rocp_df = evaluate_ablation(wide)

    ablation_df.to_csv(OUT_ABLATION, index=False, encoding="utf-8-sig")
    contribution_df.to_csv(OUT_CONTRIBUTION, index=False, encoding="utf-8-sig")
    agent_df.to_csv(OUT_AGENT, index=False, encoding="utf-8-sig")
    rocp_df.to_csv(OUT_ROCPR, index=False, encoding="utf-8-sig")

    summary = {
        "generated": str(datetime.now()),
        "input_file": str(INPUT_FILE),
        "primary_model": PRIMARY_MODEL,
        "agents": AGENTS,
        "best_ablation_condition_by_roc_auc": (
            ablation_df.sort_values("roc_auc", ascending=False).iloc[0].to_dict()
        ),
        "largest_positive_role_loss": (
            contribution_df.sort_values("delta_roc_auc", ascending=False).iloc[0].to_dict()
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    plot_balanced_accuracy(ablation_df)
    plot_contribution_loss(contribution_df)

    write_report(ablation_df, contribution_df, agent_df, rocp_df)

    print()
    print("STAGE 2F COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Ablation performance: {OUT_ABLATION}")
    print(f"Contribution loss:    {OUT_CONTRIBUTION}")
    print(f"Agent performance:    {OUT_AGENT}")
    print(f"ROC/PR summary:       {OUT_ROCPR}")
    print(f"Report:               {OUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
