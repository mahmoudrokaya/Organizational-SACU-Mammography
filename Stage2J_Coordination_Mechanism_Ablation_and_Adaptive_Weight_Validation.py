"""
Stage2J_Coordination_Mechanism_Ablation_and_Adaptive_Weight_Validation.py

Purpose
-------
Validate the adaptive coordination mechanism of the SACU mammography framework.

This stage directly addresses reviewer concerns that:
1. The number of active agents appeared fixed across folds.
2. Evidence of case-level adaptive variability was missing.
3. The adaptive coordination mechanism itself was not ablated.
4. A fixed-weight vs dynamic-weight comparison was needed.

This script does NOT:
- repeat image preprocessing
- extract features
- rebuild Stage2 matrices
- retrain the full SACU framework

Instead, it uses exact saved agent probabilities and adaptive weights to compare:

A. Learned SACU meta-fusion
B. Adaptive weight fusion
C. Fixed equal-weight fusion
D. Static global-weight fusion
E. Top-k resource-limited adaptive fusion
F. Single dominant-pathway fusion

Inputs
------
Required:
- Stage2D2B_Stage2E_Ready_Clean_Predictions.csv
- Stage2D_Adaptive_Weights.csv

Optional:
- Stage2G_Model_Metric_CI.csv
- Stage2G_Paired_Model_Comparison.csv

Outputs
-------
Stage2J_Coordination_Ablation_Performance.csv
Stage2J_Coordination_Paired_Comparison.csv
Stage2J_Case_Level_Adaptive_Variability.csv
Stage2J_Resource_Limited_TopK_Performance.csv
Stage2J_Adaptive_Weight_Entropy_Summary.csv
Stage2J_Coordination_Response_Evidence_Table.csv
Stage2J_Coordination_Ablation_Report.txt
Stage2J_Coordination_Ablation_Bar.png
Stage2J_Adaptive_Weight_Entropy_Hist.png
Stage2J_Dominant_Pathway_Counts.png
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
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
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

PREDICTIONS_FILE = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"
ADAPTIVE_WEIGHTS_FILE = TABLE_DIR / "Stage2D_Adaptive_Weights.csv"

OUT_PERFORMANCE = TABLE_DIR / "Stage2J_Coordination_Ablation_Performance.csv"
OUT_PAIRWISE = TABLE_DIR / "Stage2J_Coordination_Paired_Comparison.csv"
OUT_VARIABILITY = TABLE_DIR / "Stage2J_Case_Level_Adaptive_Variability.csv"
OUT_TOPK = TABLE_DIR / "Stage2J_Resource_Limited_TopK_Performance.csv"
OUT_ENTROPY = TABLE_DIR / "Stage2J_Adaptive_Weight_Entropy_Summary.csv"
OUT_EVIDENCE = TABLE_DIR / "Stage2J_Coordination_Response_Evidence_Table.csv"
OUT_JSON = TABLE_DIR / "Stage2J_Coordination_Ablation_Summary.json"

OUT_FIG_PERFORMANCE = FIGURE_DIR / "Stage2J_Coordination_Ablation_Bar.png"
OUT_FIG_ENTROPY = FIGURE_DIR / "Stage2J_Adaptive_Weight_Entropy_Hist.png"
OUT_FIG_DOMINANT = FIGURE_DIR / "Stage2J_Dominant_Pathway_Counts.png"

OUT_RESPONSE_TEXT = REPORT_DIR / "Stage2J_Reviewer_Response_Text.txt"
OUT_REPORT = REPORT_DIR / "Stage2J_Coordination_Ablation_Report.txt"

PRIMARY_MODEL = "SACU_LearnedShallowMetaFusion"

AGENTS = [
    "LocalRegionalAgent",
    "MultiViewAgent",
    "BilateralAgent",
    "TemporalSpatialAgent",
    "MetadataAgent",
    "AdaptiveControlAgent",
]

PATHWAY_MAP = {
    "LocalRegionalAgent": "Local-Regional",
    "MultiViewAgent": "Multi-View",
    "BilateralAgent": "Bilateral",
    "TemporalSpatialAgent": "Temporal-Spatial",
    "MetadataAgent": "Metadata",
    "AdaptiveControlAgent": "Adaptive-Control",
}

RANDOM_SEED = 42
N_BOOTSTRAP = 2000
DEFAULT_THRESHOLD = 0.50


# =============================================================================
# Loading helpers
# =============================================================================

def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column_for_agent(columns: List[str], agent: str) -> Optional[str]:
    agent_norm = normalize_name(agent)
    short = agent_norm.replace("agent", "")

    for col in columns:
        if normalize_name(col) == agent_norm:
            return col

    for col in columns:
        col_norm = normalize_name(col)
        if agent_norm in col_norm or col_norm in agent_norm:
            return col

    for col in columns:
        col_norm = normalize_name(col)
        if short and short in col_norm:
            return col

    return None


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing required prediction file: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    required = {"model", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    df = df.copy()

    if "sample_index" not in df.columns:
        df["sample_index"] = df.groupby("model").cumcount()

    df["sample_index"] = pd.to_numeric(df["sample_index"], errors="coerce").astype(int)
    df["model"] = df["model"].astype(str)
    df["y_true"] = pd.to_numeric(df["y_true"], errors="coerce").astype(int)
    df["y_score"] = pd.to_numeric(df["y_score"], errors="coerce").clip(0, 1)

    df = df.dropna(subset=["sample_index", "model", "y_true", "y_score"])
    df = df[df["y_true"].isin([0, 1])].copy()

    return df


def make_wide_predictions(df: pd.DataFrame) -> pd.DataFrame:
    labels = (
        df[["sample_index", "y_true"]]
        .drop_duplicates("sample_index")
        .sort_values("sample_index")
    )

    pivot = df.pivot_table(
        index="sample_index",
        columns="model",
        values="y_score",
        aggfunc="first",
    ).reset_index()

    wide = labels.merge(pivot, on="sample_index", how="inner")

    missing_agents = [a for a in AGENTS if a not in wide.columns]
    if missing_agents:
        raise ValueError(f"Missing required agent prediction columns: {missing_agents}")

    if PRIMARY_MODEL not in wide.columns:
        print(f"Warning: primary model {PRIMARY_MODEL} not found. Learned fusion will be skipped.")

    return wide


def load_adaptive_weights(wide: pd.DataFrame) -> pd.DataFrame:
    if not ADAPTIVE_WEIGHTS_FILE.exists():
        raise FileNotFoundError(
            f"Missing adaptive-weight file: {ADAPTIVE_WEIGHTS_FILE}. "
            "Stage2J requires real adaptive weights."
        )

    raw = pd.read_csv(ADAPTIVE_WEIGHTS_FILE)

    if raw.empty:
        raise ValueError(f"Adaptive-weight file is empty: {ADAPTIVE_WEIGHTS_FILE}")

    if "sample_index" not in raw.columns:
        raw = raw.copy()
        raw["sample_index"] = np.arange(len(raw))

    out = pd.DataFrame()
    out["sample_index"] = pd.to_numeric(raw["sample_index"], errors="coerce").astype(int)

    used_columns = {}

    for agent in AGENTS:
        col = find_column_for_agent(list(raw.columns), agent)
        if col is None:
            raise ValueError(f"Could not find adaptive-weight column for agent: {agent}")

        out[agent] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(lower=0)
        used_columns[agent] = col

    # Normalize row-wise
    row_sum = out[AGENTS].sum(axis=1).replace(0, np.nan)
    for agent in AGENTS:
        out[agent] = out[agent] / row_sum

    out = out.fillna(1.0 / len(AGENTS))

    return out


# =============================================================================
# Fusion methods
# =============================================================================

def weighted_fusion(scores: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=float)
    scores = np.asarray(scores, dtype=float)

    weights = np.clip(weights, 0, None)
    weights = weights / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)

    return np.sum(scores * weights, axis=1)


def compute_fusion_conditions(wide: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    merged = wide.merge(weights, on="sample_index", suffixes=("", "_weight"), how="inner")

    scores = merged[AGENTS].astype(float).values
    W = merged[[f"{a}_weight" if f"{a}_weight" in merged.columns else a for a in AGENTS]]

    # Defensive handling after merge: since wide and weights share agent column names,
    # pandas adds suffixes to weight columns.
    weight_cols = []
    for agent in AGENTS:
        if f"{agent}_weight" in merged.columns:
            weight_cols.append(f"{agent}_weight")
        elif f"{agent}_y" in merged.columns:
            weight_cols.append(f"{agent}_y")
        elif agent in weights.columns:
            # Should not happen after merge, but keep fallback.
            weight_cols.append(agent)
        else:
            raise ValueError(f"Cannot identify merged adaptive-weight column for {agent}")

    W = merged[weight_cols].astype(float).values
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

    out = pd.DataFrame()
    out["sample_index"] = merged["sample_index"].values
    out["y_true"] = merged["y_true"].astype(int).values

    # A. Learned SACU meta-fusion, if available.
    if PRIMARY_MODEL in merged.columns:
        out["LearnedSACUMetaFusion"] = merged[PRIMARY_MODEL].astype(float).values

    # B. Adaptive case-level weights.
    out["AdaptiveWeightFusion"] = weighted_fusion(scores, W)

    # C. Fixed equal weights.
    equal_W = np.ones_like(W) / W.shape[1]
    out["FixedEqualWeightFusion"] = weighted_fusion(scores, equal_W)

    # D. Static global mean weights.
    mean_w = np.mean(W, axis=0)
    mean_W = np.tile(mean_w.reshape(1, -1), (len(out), 1))
    out["StaticGlobalWeightFusion"] = weighted_fusion(scores, mean_W)

    # E. Dominant single pathway.
    dominant_idx = np.argmax(W, axis=1)
    out["DominantPathwayOnly"] = scores[np.arange(scores.shape[0]), dominant_idx]

    # F. Top-k resource-limited adaptive fusions.
    for k in [1, 2, 3, 4]:
        topk_W = np.zeros_like(W)
        for i in range(W.shape[0]):
            idx = np.argsort(W[i])[-k:]
            topk_W[i, idx] = W[i, idx]
        topk_W = topk_W / np.maximum(topk_W.sum(axis=1, keepdims=True), 1e-12)
        out[f"Top{k}_AdaptiveFusion"] = weighted_fusion(scores, topk_W)

    # Case-level adaptive variability
    out["dominant_agent"] = [AGENTS[i] for i in dominant_idx]
    out["dominant_pathway"] = out["dominant_agent"].map(PATHWAY_MAP)
    out["dominant_weight"] = np.max(W, axis=1)

    sorted_w = np.sort(W, axis=1)
    out["top_minus_second_weight_margin"] = sorted_w[:, -1] - sorted_w[:, -2]

    entropy = -np.sum(W * np.log(W + 1e-12), axis=1) / np.log(W.shape[1])
    out["normalized_weight_entropy"] = entropy

    for j, agent in enumerate(AGENTS):
        out[f"{agent}_weight"] = W[:, j]
        out[f"{agent}_score"] = scores[:, j]

    return out


# =============================================================================
# Metrics and comparison
# =============================================================================

def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = DEFAULT_THRESHOLD) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0

    return {
        "n": int(len(y_true)),
        "positives": int(np.sum(y_true == 1)),
        "negatives": int(np.sum(y_true == 0)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "precision_ppv": float(ppv),
        "npv": float(npv),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def evaluate_conditions(fusion_df: pd.DataFrame) -> pd.DataFrame:
    y_true = fusion_df["y_true"].astype(int).values

    excluded = {
        "sample_index",
        "y_true",
        "dominant_agent",
        "dominant_pathway",
        "dominant_weight",
        "top_minus_second_weight_margin",
        "normalized_weight_entropy",
    }

    score_cols = [
        c for c in fusion_df.columns
        if c not in excluded
        and not c.endswith("_weight")
        and not c.endswith("_score")
    ]

    rows = []

    for col in score_cols:
        y_score = fusion_df[col].astype(float).values
        m = compute_metrics(y_true, y_score)

        rows.append({
            "condition": col,
            **m,
        })

    out = pd.DataFrame(rows)
    out = out.sort_values(["roc_auc", "balanced_accuracy"], ascending=False).reset_index(drop=True)

    return out


def bootstrap_pairwise(fusion_df: pd.DataFrame, reference_condition: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    if reference_condition not in fusion_df.columns:
        raise ValueError(f"Reference condition not found: {reference_condition}")

    y_true_all = fusion_df["y_true"].astype(int).values
    ref_score_all = fusion_df[reference_condition].astype(float).values

    comparators = [
        c for c in [
            "AdaptiveWeightFusion",
            "FixedEqualWeightFusion",
            "StaticGlobalWeightFusion",
            "DominantPathwayOnly",
            "Top1_AdaptiveFusion",
            "Top2_AdaptiveFusion",
            "Top3_AdaptiveFusion",
            "Top4_AdaptiveFusion",
        ]
        if c in fusion_df.columns and c != reference_condition
    ]

    metrics_to_compare = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "mcc",
        "sensitivity_recall",
        "specificity",
    ]

    n = len(fusion_df)
    idx_boot = rng.integers(0, n, size=(N_BOOTSTRAP, n))

    ref_point = compute_metrics(y_true_all, ref_score_all)

    rows = []

    for comp in comparators:
        comp_score_all = fusion_df[comp].astype(float).values
        comp_point = compute_metrics(y_true_all, comp_score_all)

        diff_boot = {metric: [] for metric in metrics_to_compare}

        for idx in idx_boot:
            yt = y_true_all[idx]
            if len(np.unique(yt)) < 2:
                continue

            ref_s = ref_score_all[idx]
            comp_s = comp_score_all[idx]

            mr = compute_metrics(yt, ref_s)
            mc = compute_metrics(yt, comp_s)

            for metric in metrics_to_compare:
                diff_boot[metric].append(mr[metric] - mc[metric])

        for metric in metrics_to_compare:
            diffs = np.asarray(diff_boot[metric], dtype=float)
            diffs = diffs[np.isfinite(diffs)]

            if len(diffs) == 0:
                ci_low = ci_high = p = np.nan
            else:
                ci_low = float(np.percentile(diffs, 2.5))
                ci_high = float(np.percentile(diffs, 97.5))
                p = 2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
                p = float(min(1.0, p))

            rows.append({
                "reference_condition": reference_condition,
                "comparison_condition": comp,
                "metric": metric,
                "reference_point": ref_point[metric],
                "comparison_point": comp_point[metric],
                "difference_reference_minus_comparison": ref_point[metric] - comp_point[metric],
                "ci_lower_95_difference": ci_low,
                "ci_upper_95_difference": ci_high,
                "bootstrap_p_two_sided": p,
                "n_bootstrap": int(len(diffs)),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Adaptive variability summaries
# =============================================================================

def summarize_case_level_variability(fusion_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    case_cols = [
        "sample_index",
        "y_true",
        "dominant_agent",
        "dominant_pathway",
        "dominant_weight",
        "top_minus_second_weight_margin",
        "normalized_weight_entropy",
    ] + [f"{a}_weight" for a in AGENTS]

    variability = fusion_df[case_cols].copy()

    entropy_values = variability["normalized_weight_entropy"].values

    entropy_summary = pd.DataFrame([
        {
            "n_cases": int(len(variability)),
            "unique_dominant_agents": int(variability["dominant_agent"].nunique()),
            "mean_normalized_entropy": float(np.mean(entropy_values)),
            "median_normalized_entropy": float(np.median(entropy_values)),
            "std_normalized_entropy": float(np.std(entropy_values)),
            "min_normalized_entropy": float(np.min(entropy_values)),
            "q25_normalized_entropy": float(np.percentile(entropy_values, 25)),
            "q75_normalized_entropy": float(np.percentile(entropy_values, 75)),
            "max_normalized_entropy": float(np.max(entropy_values)),
            "mean_dominant_weight": float(variability["dominant_weight"].mean()),
            "mean_top_minus_second_margin": float(variability["top_minus_second_weight_margin"].mean()),
        }
    ])

    counts = (
        variability["dominant_agent"]
        .value_counts()
        .rename_axis("dominant_agent")
        .reset_index(name="n_cases")
    )
    counts["case_fraction"] = counts["n_cases"] / len(variability)
    counts["dominant_pathway"] = counts["dominant_agent"].map(PATHWAY_MAP)

    return variability, entropy_summary, counts


def summarize_topk(performance_df: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "Top1_AdaptiveFusion",
        "Top2_AdaptiveFusion",
        "Top3_AdaptiveFusion",
        "Top4_AdaptiveFusion",
        "AdaptiveWeightFusion",
        "FixedEqualWeightFusion",
        "StaticGlobalWeightFusion",
    ]

    temp = performance_df[performance_df["condition"].isin(keep)].copy()

    temp["resource_setting"] = temp["condition"].map({
        "Top1_AdaptiveFusion": "Top-1 adaptive pathway",
        "Top2_AdaptiveFusion": "Top-2 adaptive pathways",
        "Top3_AdaptiveFusion": "Top-3 adaptive pathways",
        "Top4_AdaptiveFusion": "Top-4 adaptive pathways",
        "AdaptiveWeightFusion": "All pathways with case-level adaptive weights",
        "FixedEqualWeightFusion": "All pathways with fixed equal weights",
        "StaticGlobalWeightFusion": "All pathways with fixed global average weights",
    })

    return temp.sort_values("condition").reset_index(drop=True)


def build_evidence_table(
    performance_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    entropy_summary: pd.DataFrame,
    dominant_counts: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    def get_metric(condition: str, metric: str) -> float:
        row = performance_df[performance_df["condition"] == condition]
        if row.empty:
            return np.nan
        return float(row.iloc[0][metric])

    adaptive_auc = get_metric("AdaptiveWeightFusion", "roc_auc")
    equal_auc = get_metric("FixedEqualWeightFusion", "roc_auc")
    static_auc = get_metric("StaticGlobalWeightFusion", "roc_auc")
    learned_auc = get_metric("LearnedSACUMetaFusion", "roc_auc")

    adaptive_ba = get_metric("AdaptiveWeightFusion", "balanced_accuracy")
    equal_ba = get_metric("FixedEqualWeightFusion", "balanced_accuracy")
    static_ba = get_metric("StaticGlobalWeightFusion", "balanced_accuracy")
    learned_ba = get_metric("LearnedSACUMetaFusion", "balanced_accuracy")

    rows.append({
        "reviewer_question": "Does the adaptive coordination mechanism change across cases?",
        "quantitative_evidence": (
            f"Dominant pathway varied across {int(entropy_summary.iloc[0]['unique_dominant_agents'])} agents; "
            f"mean normalized weight entropy = {entropy_summary.iloc[0]['mean_normalized_entropy']:.4f}."
        ),
        "safe_interpretation": (
            "The active influence distribution is case-dependent, even when the nominal agent set remains constant."
        ),
    })

    if not dominant_counts.empty:
        count_text = "; ".join(
            f"{r['dominant_agent']}={int(r['n_cases'])} ({100*r['case_fraction']:.1f}%)"
            for _, r in dominant_counts.iterrows()
        )
        rows.append({
            "reviewer_question": "What is the evidence of case-level variability?",
            "quantitative_evidence": count_text,
            "safe_interpretation": (
                "Different pathways dominate different examinations, showing case-level adaptive redistribution."
            ),
        })

    rows.append({
        "reviewer_question": "Does dynamic weighting outperform fixed equal weighting?",
        "quantitative_evidence": (
            f"AdaptiveWeightFusion ROC-AUC={adaptive_auc:.4f}, balanced accuracy={adaptive_ba:.4f}; "
            f"FixedEqualWeightFusion ROC-AUC={equal_auc:.4f}, balanced accuracy={equal_ba:.4f}."
        ),
        "safe_interpretation": (
            "This directly tests dynamic case-level weights against a fixed equal-weight coordinator."
        ),
    })

    rows.append({
        "reviewer_question": "Does dynamic weighting outperform static global weighting?",
        "quantitative_evidence": (
            f"AdaptiveWeightFusion ROC-AUC={adaptive_auc:.4f}; "
            f"StaticGlobalWeightFusion ROC-AUC={static_auc:.4f}."
        ),
        "safe_interpretation": (
            "This tests whether using a case-specific weight vector differs from using one global weight vector."
        ),
    })

    rows.append({
        "reviewer_question": "How should the word adaptive be interpreted?",
        "quantitative_evidence": (
            "The nominal set of available agents may remain fixed, but their influence weights and dominant pathway vary by case."
        ),
        "safe_interpretation": (
            "Adaptivity refers to case-level influence redistribution and resource-limited pathway selection, not necessarily creation of new agents during inference."
        ),
    })

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def plot_performance(performance_df: pd.DataFrame) -> None:
    keep = [
        "LearnedSACUMetaFusion",
        "AdaptiveWeightFusion",
        "FixedEqualWeightFusion",
        "StaticGlobalWeightFusion",
        "DominantPathwayOnly",
        "Top1_AdaptiveFusion",
        "Top2_AdaptiveFusion",
        "Top3_AdaptiveFusion",
        "Top4_AdaptiveFusion",
    ]

    temp = performance_df[performance_df["condition"].isin(keep)].copy()
    temp = temp.sort_values("roc_auc", ascending=True)

    y = np.arange(len(temp))

    plt.figure(figsize=(12, 8))
    plt.barh(y, temp["roc_auc"].values)
    plt.yticks(y, temp["condition"].values, fontsize=12, fontweight="bold")
    plt.xlabel("ROC-AUC", fontsize=14, fontweight="bold")
    plt.title("Stage2J Coordination Mechanism Ablation", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG_PERFORMANCE, dpi=300)
    plt.close()


def plot_entropy(fusion_df: pd.DataFrame) -> None:
    plt.figure(figsize=(10, 8))
    plt.hist(fusion_df["normalized_weight_entropy"].values, bins=30)
    plt.xlabel("Normalized Adaptive Weight Entropy", fontsize=14, fontweight="bold")
    plt.ylabel("Number of Cases", fontsize=14, fontweight="bold")
    plt.title("Stage2J Case-Level Adaptive Weight Variability", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG_ENTROPY, dpi=300)
    plt.close()


def plot_dominant_counts(dominant_counts: pd.DataFrame) -> None:
    temp = dominant_counts.copy()
    temp = temp.sort_values("n_cases", ascending=True)

    plt.figure(figsize=(10, 8))
    plt.barh(temp["dominant_agent"], temp["n_cases"])
    plt.xlabel("Number of Cases", fontsize=14, fontweight="bold")
    plt.ylabel("Dominant Pathway", fontsize=14, fontweight="bold")
    plt.title("Stage2J Dominant Pathway Counts", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DOMINANT, dpi=300)
    plt.close()


# =============================================================================
# Response text and report
# =============================================================================

def build_response_text(evidence_df: pd.DataFrame) -> str:
    lines = []

    lines.append("Common Comment C4 — Adaptive Mechanism Validation")
    lines.append("")
    lines.append("Response:")
    lines.append("")
    lines.append(
        "We thank the reviewers for pointing out the need to clarify and validate the adaptive "
        "coordination mechanism. We agree that the term adaptive should be supported by "
        "explicit experimental evidence rather than architectural description alone."
    )
    lines.append("")
    lines.append(
        "To address this concern, we added a coordination-mechanism ablation experiment. "
        "The new experiment compares the proposed case-level adaptive weighting strategy "
        "against fixed equal-weight fusion, static global-weight fusion, dominant-pathway-only "
        "fusion, and resource-limited top-k adaptive fusion. This directly evaluates whether "
        "case-specific coordination provides measurable benefit relative to non-adaptive alternatives."
    )
    lines.append("")
    lines.append(
        "We also clarified that, in the current implementation, adaptivity does not mean that "
        "new agents are physically created during inference. Instead, the available cooperative "
        "agents remain structurally defined, while their relative influence and dominance are "
        "redistributed on a case-by-case basis according to the adaptive coordination weights. "
        "Thus, the number of listed active agent categories may remain stable across folds, but "
        "the pathway-level influence pattern varies across individual examinations."
    )
    lines.append("")
    lines.append("The added evidence is summarized as follows:")
    lines.append("")

    for _, r in evidence_df.iterrows():
        lines.append(f"- {r['reviewer_question']}: {r['quantitative_evidence']}")

    lines.append("")
    lines.append(
        "These additions clarify the operational meaning of adaptivity in the proposed SACU "
        "framework and provide a direct ablation of the coordination mechanism requested by "
        "the reviewers."
    )

    return "\n".join(lines)


def write_report(
    performance_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
    variability_df: pd.DataFrame,
    topk_df: pd.DataFrame,
    entropy_summary: pd.DataFrame,
    dominant_counts: pd.DataFrame,
    evidence_df: pd.DataFrame,
    response_text: str,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2J COORDINATION MECHANISM ABLATION AND ADAPTIVE WEIGHT VALIDATION")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Prediction input: {PREDICTIONS_FILE}")
    lines.append(f"Adaptive weight input: {ADAPTIVE_WEIGHTS_FILE}")
    lines.append("")

    lines.append("COORDINATION ABLATION PERFORMANCE")
    lines.append("-" * 100)
    lines.append(performance_df.to_string(index=False))
    lines.append("")

    lines.append("PAIRED BOOTSTRAP COMPARISON")
    lines.append("-" * 100)
    lines.append(pairwise_df.to_string(index=False))
    lines.append("")

    lines.append("ADAPTIVE WEIGHT ENTROPY SUMMARY")
    lines.append("-" * 100)
    lines.append(entropy_summary.to_string(index=False))
    lines.append("")

    lines.append("DOMINANT PATHWAY COUNTS")
    lines.append("-" * 100)
    lines.append(dominant_counts.to_string(index=False))
    lines.append("")

    lines.append("REVIEWER RESPONSE EVIDENCE")
    lines.append("-" * 100)
    lines.append(evidence_df.to_string(index=False))
    lines.append("")

    lines.append("REVIEWER RESPONSE TEXT")
    lines.append("-" * 100)
    lines.append(response_text)
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUT_PERFORMANCE,
        OUT_PAIRWISE,
        OUT_VARIABILITY,
        OUT_TOPK,
        OUT_ENTROPY,
        OUT_EVIDENCE,
        OUT_JSON,
        OUT_FIG_PERFORMANCE,
        OUT_FIG_ENTROPY,
        OUT_FIG_DOMINANT,
        OUT_RESPONSE_TEXT,
        OUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(OUT_RESPONSE_TEXT, "w", encoding="utf-8") as f:
        f.write(response_text)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("STAGE 2J COORDINATION MECHANISM ABLATION AND ADAPTIVE WEIGHT VALIDATION")
    print("=" * 100)

    pred_long = load_predictions()
    wide = make_wide_predictions(pred_long)
    weights = load_adaptive_weights(wide)

    fusion_df = compute_fusion_conditions(wide=wide, weights=weights)

    performance_df = evaluate_conditions(fusion_df)

    reference = "LearnedSACUMetaFusion" if "LearnedSACUMetaFusion" in fusion_df.columns else "AdaptiveWeightFusion"
    pairwise_df = bootstrap_pairwise(fusion_df, reference_condition=reference)

    variability_df, entropy_summary, dominant_counts = summarize_case_level_variability(fusion_df)
    topk_df = summarize_topk(performance_df)

    evidence_df = build_evidence_table(
        performance_df=performance_df,
        pairwise_df=pairwise_df,
        entropy_summary=entropy_summary,
        dominant_counts=dominant_counts,
    )

    response_text = build_response_text(evidence_df)

    performance_df.to_csv(OUT_PERFORMANCE, index=False, encoding="utf-8-sig")
    pairwise_df.to_csv(OUT_PAIRWISE, index=False, encoding="utf-8-sig")
    variability_df.to_csv(OUT_VARIABILITY, index=False, encoding="utf-8-sig")
    topk_df.to_csv(OUT_TOPK, index=False, encoding="utf-8-sig")
    entropy_summary.to_csv(OUT_ENTROPY, index=False, encoding="utf-8-sig")
    evidence_df.to_csv(OUT_EVIDENCE, index=False, encoding="utf-8-sig")

    plot_performance(performance_df)
    plot_entropy(fusion_df)
    plot_dominant_counts(dominant_counts)

    summary = {
        "generated": str(datetime.now()),
        "prediction_input": str(PREDICTIONS_FILE),
        "adaptive_weight_input": str(ADAPTIVE_WEIGHTS_FILE),
        "reference_condition": reference,
        "agents": AGENTS,
        "pathway_map": PATHWAY_MAP,
        "n_cases": int(len(fusion_df)),
        "positives": int((fusion_df["y_true"] == 1).sum()),
        "negatives": int((fusion_df["y_true"] == 0).sum()),
        "performance": performance_df.to_dict(orient="records"),
        "entropy_summary": entropy_summary.to_dict(orient="records"),
        "dominant_counts": dominant_counts.to_dict(orient="records"),
        "reviewer_evidence": evidence_df.to_dict(orient="records"),
        "safe_interpretation": (
            "Adaptivity is validated as case-level influence redistribution and resource-limited "
            "pathway selection. It should not be described as physical creation of new agents "
            "during inference unless implemented explicitly."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    write_report(
        performance_df=performance_df,
        pairwise_df=pairwise_df,
        variability_df=variability_df,
        topk_df=topk_df,
        entropy_summary=entropy_summary,
        dominant_counts=dominant_counts,
        evidence_df=evidence_df,
        response_text=response_text,
    )

    print()
    print("STAGE 2J COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Performance table:      {OUT_PERFORMANCE}")
    print(f"Pairwise comparison:    {OUT_PAIRWISE}")
    print(f"Case-level variability: {OUT_VARIABILITY}")
    print(f"Top-k performance:      {OUT_TOPK}")
    print(f"Entropy summary:        {OUT_ENTROPY}")
    print(f"Evidence table:         {OUT_EVIDENCE}")
    print(f"Response text:          {OUT_RESPONSE_TEXT}")
    print(f"Report:                 {OUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
