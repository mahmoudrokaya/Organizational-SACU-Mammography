r"""
Stage2I_Organizational_Interpretability_and_Pathway_Contribution_Analysis.py

Purpose
-------
Provide reviewer-facing quantitative interpretability evidence for the
organizational SACU mammography framework.

This stage avoids unsupported image-level saliency claims such as Grad-CAM,
Dice, or IoU because the current SACU implementation operates on structured
regional/pathway descriptors and shallow cooperative agents.

Instead, it quantifies organizational interpretability through:

1. Pathway-level contribution analysis
2. Adaptive influence-weight distribution
3. Case-level dominant pathway statistics
4. Agreement between dominant pathway and prediction correctness
5. Contribution ranking from ablation evidence
6. Interpretability summary text for reviewer response

Inputs
------
Required:
- Stage2D2B_Stage2E_Ready_Clean_Predictions.csv

Optional but recommended:
- Stage2D_Adaptive_Weights.csv
- Stage2F_Role_Contribution_Loss.csv
- Stage2F_Individual_Agent_Performance.csv
- Stage2F_Ablation_Performance.csv
- Stage2G_Paired_Model_Comparison.csv

Outputs
-------
Stage2I_Pathway_Contribution_Summary.csv
Stage2I_Case_Level_Dominant_Pathway.csv
Stage2I_Adaptive_Weight_Distribution.csv
Stage2I_Pathway_Correctness_Association.csv
Stage2I_Interpretability_Evidence_Table.csv
Stage2I_Reviewer_Response_Text.txt
Stage2I_Organizational_Interpretability_Report.txt
Stage2I_Pathway_Contribution_Bar.png
Stage2I_Dominant_Pathway_Distribution.png
Stage2I_Adaptive_Weight_Boxplot.png
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
    roc_auc_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
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

# Required exact predictions
PREDICTIONS_FILE = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"

# Optional evidence files
ADAPTIVE_WEIGHTS_FILE = TABLE_DIR / "Stage2D_Adaptive_Weights.csv"
ROLE_CONTRIBUTION_FILE = TABLE_DIR / "Stage2F_Role_Contribution_Loss.csv"
INDIVIDUAL_AGENT_FILE = TABLE_DIR / "Stage2F_Individual_Agent_Performance.csv"
ABLATION_FILE = TABLE_DIR / "Stage2F_Ablation_Performance.csv"
PAIRWISE_FILE = TABLE_DIR / "Stage2G_Paired_Model_Comparison.csv"

# Outputs
OUT_PATHWAY_SUMMARY = TABLE_DIR / "Stage2I_Pathway_Contribution_Summary.csv"
OUT_CASE_LEVEL = TABLE_DIR / "Stage2I_Case_Level_Dominant_Pathway.csv"
OUT_WEIGHT_DIST = TABLE_DIR / "Stage2I_Adaptive_Weight_Distribution.csv"
OUT_CORRECTNESS = TABLE_DIR / "Stage2I_Pathway_Correctness_Association.csv"
OUT_EVIDENCE = TABLE_DIR / "Stage2I_Interpretability_Evidence_Table.csv"
OUT_JSON = TABLE_DIR / "Stage2I_Interpretability_Summary.json"

OUT_FIG_CONTRIBUTION = FIGURE_DIR / "Stage2I_Pathway_Contribution_Bar.png"
OUT_FIG_DOMINANT = FIGURE_DIR / "Stage2I_Dominant_Pathway_Distribution.png"
OUT_FIG_WEIGHTS = FIGURE_DIR / "Stage2I_Adaptive_Weight_Boxplot.png"

OUT_RESPONSE_TEXT = REPORT_DIR / "Stage2I_Reviewer_Response_Text.txt"
OUT_REPORT = REPORT_DIR / "Stage2I_Organizational_Interpretability_Report.txt"

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
    "LocalRegionalAgent": "local_regional_tissue_pattern",
    "MultiViewAgent": "multi_view_correlation",
    "BilateralAgent": "bilateral_asymmetry",
    "TemporalSpatialAgent": "temporal_spatial_progression",
    "MetadataAgent": "clinical_metadata_context",
    "AdaptiveControlAgent": "adaptive_control_context",
}

CLINICAL_INTERPRETATION = {
    "LocalRegionalAgent": "localized tissue-density and regional pattern evidence",
    "MultiViewAgent": "cross-view CC/MLO consistency and complementarity",
    "BilateralAgent": "left-right breast asymmetry and structural difference evidence",
    "TemporalSpatialAgent": "temporal-spatial progression and longitudinal change evidence",
    "MetadataAgent": "available metadata-level contextual evidence",
    "AdaptiveControlAgent": "case-level adaptive-control and organizational context",
}


# =============================================================================
# Loading helpers
# =============================================================================

def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def load_predictions() -> pd.DataFrame:
    if not PREDICTIONS_FILE.exists():
        raise FileNotFoundError(f"Missing prediction file: {PREDICTIONS_FILE}")

    df = pd.read_csv(PREDICTIONS_FILE)

    required = {"model", "y_true", "y_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction file missing required columns: {missing}")

    df = df.copy()

    if "sample_index" not in df.columns:
        df["sample_index"] = df.groupby("model").cumcount()

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

    missing = [a for a in AGENTS if a not in wide.columns]
    if missing:
        raise ValueError(f"Missing agent prediction columns: {missing}")

    if PRIMARY_MODEL not in wide.columns:
        raise ValueError(f"Missing primary model prediction column: {PRIMARY_MODEL}")

    return wide


# =============================================================================
# Adaptive weights
# =============================================================================

def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def find_column_for_agent(columns: List[str], agent: str) -> Optional[str]:
    agent_norm = normalize_name(agent)
    short = agent_norm.replace("agent", "")

    for col in columns:
        col_norm = normalize_name(col)
        if agent_norm == col_norm:
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


def standardize_adaptive_weights(wide: pd.DataFrame) -> pd.DataFrame:
    """
    Reads Stage2D adaptive weights if available. If not available, approximates
    pathway influence using normalized absolute deviation of agent probabilities
    from the class prior. This fallback is explicitly marked as approximation.
    """
    if ADAPTIVE_WEIGHTS_FILE.exists():
        raw = pd.read_csv(ADAPTIVE_WEIGHTS_FILE)

        if "sample_index" not in raw.columns:
            raw = raw.copy()
            raw["sample_index"] = np.arange(len(raw))

        out = pd.DataFrame()
        out["sample_index"] = raw["sample_index"]

        used_cols = {}

        for agent in AGENTS:
            col = find_column_for_agent(list(raw.columns), agent)
            if col is not None:
                out[agent] = pd.to_numeric(raw[col], errors="coerce").fillna(0).clip(lower=0)
                used_cols[agent] = col

        if all(agent in out.columns for agent in AGENTS):
            weight_sum = out[AGENTS].sum(axis=1).replace(0, np.nan)
            for agent in AGENTS:
                out[agent] = out[agent] / weight_sum

            out = out.fillna(1.0 / len(AGENTS))
            out["weight_source"] = "Stage2D_Adaptive_Weights"
            return out

    # Fallback: approximate influence from agent score deviation.
    out = pd.DataFrame()
    out["sample_index"] = wide["sample_index"]

    prior = float(wide["y_true"].mean())

    raw_weights = []
    for agent in AGENTS:
        raw = np.abs(wide[agent].astype(float).values - prior)
        raw_weights.append(raw.reshape(-1, 1))

    W = np.hstack(raw_weights)
    W = W / np.maximum(W.sum(axis=1, keepdims=True), 1e-12)

    for i, agent in enumerate(AGENTS):
        out[agent] = W[:, i]

    out["weight_source"] = "approximate_probability_deviation_importance"

    return out


# =============================================================================
# Metrics
# =============================================================================

def basic_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity_recall": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)) if len(np.unique(y_pred)) > 1 else 0.0,
    }


# =============================================================================
# Interpretability analyses
# =============================================================================

def compute_pathway_summary(
    wide: pd.DataFrame,
    weights: pd.DataFrame,
    contribution_df: pd.DataFrame,
    agent_perf_df: pd.DataFrame,
    pairwise_df: pd.DataFrame,
) -> pd.DataFrame:

    merged = wide.merge(weights, on="sample_index", suffixes=("", "_weight"))

    y_true = merged["y_true"].astype(int).values

    rows = []

    for agent in AGENTS:
        score = merged[agent].astype(float).values
        w = merged[f"{agent}_weight"] if f"{agent}_weight" in merged.columns else merged[agent + "_y"]
        # The previous line is unlikely to be used, kept only for defensive compatibility.

    rows = []

    # Use weights frame directly.
    weight_lookup = weights.set_index("sample_index")

    for agent in AGENTS:
        score = wide[agent].astype(float).values
        metric = basic_metrics(y_true, score)

        agent_weight = weights[agent].astype(float).values

        ablation_loss_roc = np.nan
        ablation_loss_ap = np.nan
        ablation_loss_f1 = np.nan

        if not contribution_df.empty and "removed_agent" in contribution_df.columns:
            crow = contribution_df[contribution_df["removed_agent"] == agent]
            if not crow.empty:
                ablation_loss_roc = float(crow.iloc[0].get("delta_roc_auc", np.nan))
                ablation_loss_ap = float(crow.iloc[0].get("delta_average_precision", np.nan))
                ablation_loss_f1 = float(crow.iloc[0].get("delta_f1", np.nan))

        pairwise_delta = np.nan
        pairwise_p = np.nan

        if not pairwise_df.empty:
            prow = pairwise_df[
                (pairwise_df.get("comparison_model", "") == agent)
                & (pairwise_df.get("metric", "") == "roc_auc")
            ]
            if not prow.empty:
                pairwise_delta = float(prow.iloc[0].get("point_difference_primary_minus_comparison", np.nan))
                pairwise_p = float(prow.iloc[0].get("bootstrap_p_two_sided", np.nan))

        rows.append({
            "agent": agent,
            "clinical_pathway": PATHWAY_MAP[agent],
            "clinical_interpretation": CLINICAL_INTERPRETATION[agent],
            "mean_adaptive_weight": float(np.mean(agent_weight)),
            "median_adaptive_weight": float(np.median(agent_weight)),
            "std_adaptive_weight": float(np.std(agent_weight)),
            "dominant_case_count": int(np.sum(weights[AGENTS].idxmax(axis=1) == agent)),
            "dominant_case_fraction": float(np.mean(weights[AGENTS].idxmax(axis=1) == agent)),
            "agent_roc_auc": metric["roc_auc"],
            "agent_average_precision": metric["average_precision"],
            "agent_balanced_accuracy": metric["balanced_accuracy"],
            "agent_sensitivity": metric["sensitivity_recall"],
            "agent_specificity": metric["specificity"],
            "agent_f1": metric["f1"],
            "agent_mcc": metric["mcc"],
            "leave_one_out_delta_roc_auc": ablation_loss_roc,
            "leave_one_out_delta_average_precision": ablation_loss_ap,
            "leave_one_out_delta_f1": ablation_loss_f1,
            "primary_minus_agent_roc_auc_delta": pairwise_delta,
            "primary_vs_agent_bootstrap_p": pairwise_p,
            "interpretability_role": (
                "major_positive_contributor"
                if pd.notna(ablation_loss_roc) and ablation_loss_roc > 0.01
                else "supporting_or_contextual_contributor"
            ),
        })

    out = pd.DataFrame(rows)

    out = out.sort_values(
        ["mean_adaptive_weight", "agent_roc_auc"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return out


def compute_case_level_dominance(wide: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    merged = wide.merge(weights, on="sample_index", suffixes=("", "_weight"))

    agent_weight_cols = AGENTS

    dominant_agent = weights[AGENTS].idxmax(axis=1)
    dominant_weight = weights[AGENTS].max(axis=1)

    out = pd.DataFrame()
    out["sample_index"] = wide["sample_index"].values
    out["y_true"] = wide["y_true"].astype(int).values
    out["primary_score"] = wide[PRIMARY_MODEL].astype(float).values
    out["primary_pred_0p5"] = (out["primary_score"] >= 0.5).astype(int)
    out["primary_correct_0p5"] = (out["primary_pred_0p5"] == out["y_true"]).astype(int)
    out["dominant_agent"] = dominant_agent.values
    out["dominant_pathway"] = out["dominant_agent"].map(PATHWAY_MAP)
    out["dominant_weight"] = dominant_weight.values

    for agent in AGENTS:
        out[f"{agent}_score"] = wide[agent].astype(float).values
        out[f"{agent}_weight"] = weights[agent].astype(float).values

    out["dominance_strength"] = out["dominant_weight"]

    sorted_weights = np.sort(weights[AGENTS].values, axis=1)
    out["dominance_margin_top_minus_second"] = sorted_weights[:, -1] - sorted_weights[:, -2]

    return out


def compute_weight_distribution(weights: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for agent in AGENTS:
        values = weights[agent].astype(float).values

        rows.append({
            "agent": agent,
            "clinical_pathway": PATHWAY_MAP[agent],
            "mean_weight": float(np.mean(values)),
            "median_weight": float(np.median(values)),
            "std_weight": float(np.std(values)),
            "min_weight": float(np.min(values)),
            "q25_weight": float(np.percentile(values, 25)),
            "q75_weight": float(np.percentile(values, 75)),
            "max_weight": float(np.max(values)),
        })

    return pd.DataFrame(rows).sort_values("mean_weight", ascending=False)


def compute_correctness_association(case_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for agent in AGENTS:
        sub = case_df[case_df["dominant_agent"] == agent]

        if sub.empty:
            rows.append({
                "dominant_agent": agent,
                "clinical_pathway": PATHWAY_MAP[agent],
                "n_cases": 0,
                "case_fraction": 0.0,
                "accuracy_when_dominant": np.nan,
                "positive_fraction_when_dominant": np.nan,
                "mean_primary_score_when_dominant": np.nan,
                "mean_dominant_weight": np.nan,
            })
            continue

        rows.append({
            "dominant_agent": agent,
            "clinical_pathway": PATHWAY_MAP[agent],
            "n_cases": int(len(sub)),
            "case_fraction": float(len(sub) / len(case_df)),
            "accuracy_when_dominant": float(sub["primary_correct_0p5"].mean()),
            "positive_fraction_when_dominant": float(sub["y_true"].mean()),
            "mean_primary_score_when_dominant": float(sub["primary_score"].mean()),
            "mean_dominant_weight": float(sub["dominant_weight"].mean()),
        })

    return pd.DataFrame(rows).sort_values("n_cases", ascending=False)


def build_evidence_table(
    pathway_summary: pd.DataFrame,
    correctness_df: pd.DataFrame,
    contribution_df: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    top_weight = pathway_summary.sort_values("mean_adaptive_weight", ascending=False).iloc[0]
    top_auc = pathway_summary.sort_values("agent_roc_auc", ascending=False).iloc[0]

    rows.append({
        "interpretability_question": "Which clinical pathway most strongly influenced organizational inference?",
        "quantitative_evidence": (
            f"{top_weight['agent']} had the highest mean adaptive weight "
            f"({top_weight['mean_adaptive_weight']:.4f})."
        ),
        "clinical_interpretation": top_weight["clinical_interpretation"],
        "safe_claim": (
            "Adaptive influence weights provide pathway-level explanation of the model decision process."
        ),
    })

    rows.append({
        "interpretability_question": "Which individual pathway had the strongest standalone discrimination?",
        "quantitative_evidence": (
            f"{top_auc['agent']} had the highest individual-agent ROC-AUC "
            f"({top_auc['agent_roc_auc']:.4f})."
        ),
        "clinical_interpretation": top_auc["clinical_interpretation"],
        "safe_claim": (
            "Individual pathway performance helps identify which clinical reasoning stream carried the strongest independent signal."
        ),
    })

    if not contribution_df.empty and "delta_roc_auc" in contribution_df.columns:
        best_loss = contribution_df.sort_values("delta_roc_auc", ascending=False).iloc[0]

        rows.append({
            "interpretability_question": "Which pathway showed the clearest positive marginal contribution?",
            "quantitative_evidence": (
                f"Removing {best_loss['removed_agent']} produced the largest positive "
                f"ROC-AUC loss ({best_loss['delta_roc_auc']:.4f})."
            ),
            "clinical_interpretation": CLINICAL_INTERPRETATION.get(
                best_loss["removed_agent"],
                "clinically meaningful pathway contribution",
            ),
            "safe_claim": (
                "Leave-one-role-out evidence supports a measurable contribution for this pathway in the equal-weight fusion setting."
            ),
        })

    top_dom = correctness_df.sort_values("n_cases", ascending=False).iloc[0]

    rows.append({
        "interpretability_question": "Which pathway was most frequently dominant at the case level?",
        "quantitative_evidence": (
            f"{top_dom['dominant_agent']} was dominant in {int(top_dom['n_cases'])} cases "
            f"({top_dom['case_fraction']:.4f} of test cases)."
        ),
        "clinical_interpretation": CLINICAL_INTERPRETATION.get(
            top_dom["dominant_agent"],
            "dominant organizational pathway",
        ),
        "safe_claim": (
            "Case-level dominance provides an interpretable summary of which diagnostic pathway controlled inference most often."
        ),
    })

    rows.append({
        "interpretability_question": "Does this replace radiologist-annotated saliency validation?",
        "quantitative_evidence": (
            "No lesion-mask Dice/IoU analysis was performed because the SACU framework is descriptor/pathway-based."
        ),
        "clinical_interpretation": (
            "Interpretability is provided at the diagnostic-pathway level, not at pixel-level heatmap localization."
        ),
        "safe_claim": (
            "The manuscript should state this limitation clearly and avoid claiming expert-validated lesion localization."
        ),
    })

    return pd.DataFrame(rows)


# =============================================================================
# Figures
# =============================================================================

def plot_pathway_contribution(pathway_summary: pd.DataFrame) -> None:
    temp = pathway_summary.sort_values("mean_adaptive_weight", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(temp["agent"], temp["mean_adaptive_weight"])
    plt.xlabel("Mean Adaptive Influence Weight")
    plt.title("Stage2I Pathway-Level Organizational Influence")
    plt.tight_layout()
    plt.savefig(OUT_FIG_CONTRIBUTION, dpi=300)
    plt.close()


def plot_dominant_pathway(case_df: pd.DataFrame) -> None:
    counts = case_df["dominant_agent"].value_counts().reindex(AGENTS).fillna(0)

    plt.figure(figsize=(10, 6))
    plt.bar(counts.index, counts.values)
    plt.ylabel("Number of Test Cases")
    plt.title("Stage2I Dominant Pathway Distribution")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIG_DOMINANT, dpi=300)
    plt.close()


def plot_weight_boxplot(weights: pd.DataFrame) -> None:
    data = [weights[a].astype(float).values for a in AGENTS]

    plt.figure(figsize=(10, 6))
    plt.boxplot(data, labels=AGENTS, showfliers=False)
    plt.ylabel("Adaptive Influence Weight")
    plt.title("Stage2I Adaptive Weight Distribution by Pathway")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_FIG_WEIGHTS, dpi=300)
    plt.close()


# =============================================================================
# Text outputs
# =============================================================================

def build_reviewer_response_text(
    pathway_summary: pd.DataFrame,
    correctness_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    weights_source: str,
) -> str:

    top_weight = pathway_summary.sort_values("mean_adaptive_weight", ascending=False).iloc[0]
    top_auc = pathway_summary.sort_values("agent_roc_auc", ascending=False).iloc[0]

    lines = []

    lines.append("Common Comment C3 — Interpretability Analysis")
    lines.append("")
    lines.append("Response:")
    lines.append("")
    lines.append(
        "We thank the reviewers for emphasizing the importance of interpretability and clinical "
        "transparency. We agree that interpretability is essential for the adoption of AI systems "
        "in breast cancer screening. In response, we expanded the manuscript to clarify that the "
        "proposed SACU framework provides interpretability through organizational decomposition "
        "rather than through conventional pixel-level saliency maps alone."
    )
    lines.append("")
    lines.append(
        "Unlike monolithic CNN or transformer models, the proposed framework distributes diagnostic "
        "reasoning across clinically meaningful cooperative pathways, including local-regional tissue "
        "pattern analysis, multi-view correlation, bilateral asymmetry assessment, temporal-spatial "
        "progression modeling, metadata context, and adaptive-control context. This allows the "
        "relative contribution of each pathway to be quantified during inference."
    )
    lines.append("")
    lines.append(
        "To provide quantitative interpretability evidence, we added an organizational pathway "
        "contribution analysis. Specifically, we quantified adaptive influence weights, case-level "
        "dominant pathways, individual-agent discrimination, and leave-one-role-out contribution "
        "effects. The analysis showed that "
        f"{top_weight['agent']} had the highest mean adaptive influence weight "
        f"({top_weight['mean_adaptive_weight']:.4f}), while "
        f"{top_auc['agent']} achieved the strongest standalone pathway discrimination "
        f"(ROC-AUC={top_auc['agent_roc_auc']:.4f})."
    )
    lines.append("")
    lines.append(
        "We also added a case-level dominance analysis showing which diagnostic pathway controlled "
        "the organizational inference most strongly for each held-out case. This provides a transparent "
        "pathway-level explanation of model behavior and complements the ablation analysis."
    )
    lines.append("")
    lines.append(
        "We recognize that Grad-CAM, Dice, and IoU analyses are useful for image-level CNN or "
        "attention-based models when expert lesion masks are available. However, the proposed SACU "
        "implementation operates on localized regional descriptors and clinically structured pathway "
        "representations rather than on a single deep feature map. Therefore, pathway-level influence "
        "decomposition is the most appropriate interpretability mechanism for this architecture. We "
        "have clarified this point in the revised manuscript and explicitly note that expert-annotated "
        "pixel-level lesion-overlap validation remains an important direction for future work."
    )
    lines.append("")
    lines.append("Added manuscript evidence:")
    lines.append("- Stage2I_Pathway_Contribution_Summary.csv")
    lines.append("- Stage2I_Case_Level_Dominant_Pathway.csv")
    lines.append("- Stage2I_Adaptive_Weight_Distribution.csv")
    lines.append("- Stage2I_Pathway_Correctness_Association.csv")
    lines.append("- Stage2I_Interpretability_Evidence_Table.csv")
    lines.append("")
    lines.append(f"Adaptive weight source: {weights_source}")

    return "\n".join(lines)


def write_report(
    pathway_summary: pd.DataFrame,
    case_df: pd.DataFrame,
    weight_dist: pd.DataFrame,
    correctness_df: pd.DataFrame,
    evidence_df: pd.DataFrame,
    response_text: str,
    weights_source: str,
) -> None:

    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2I ORGANIZATIONAL INTERPRETABILITY AND PATHWAY CONTRIBUTION ANALYSIS")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Prediction input: {PREDICTIONS_FILE}")
    lines.append(f"Adaptive weight source: {weights_source}")
    lines.append("")

    lines.append("PATHWAY CONTRIBUTION SUMMARY")
    lines.append("-" * 100)
    lines.append(pathway_summary.to_string(index=False))
    lines.append("")

    lines.append("ADAPTIVE WEIGHT DISTRIBUTION")
    lines.append("-" * 100)
    lines.append(weight_dist.to_string(index=False))
    lines.append("")

    lines.append("CASE-LEVEL DOMINANT PATHWAY ASSOCIATION")
    lines.append("-" * 100)
    lines.append(correctness_df.to_string(index=False))
    lines.append("")

    lines.append("INTERPRETABILITY EVIDENCE TABLE")
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
        OUT_PATHWAY_SUMMARY,
        OUT_CASE_LEVEL,
        OUT_WEIGHT_DIST,
        OUT_CORRECTNESS,
        OUT_EVIDENCE,
        OUT_JSON,
        OUT_FIG_CONTRIBUTION,
        OUT_FIG_DOMINANT,
        OUT_FIG_WEIGHTS,
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
    print("STAGE 2I ORGANIZATIONAL INTERPRETABILITY AND PATHWAY CONTRIBUTION ANALYSIS")
    print("=" * 100)

    pred_long = load_predictions()
    wide = make_wide_predictions(pred_long)

    weights = standardize_adaptive_weights(wide)
    weights_source = str(weights["weight_source"].iloc[0]) if "weight_source" in weights.columns else "unknown"

    # Rename weight columns only when merging is not needed.
    clean_weights = weights[["sample_index"] + AGENTS].copy()

    contribution_df = read_csv_optional(ROLE_CONTRIBUTION_FILE)
    agent_perf_df = read_csv_optional(INDIVIDUAL_AGENT_FILE)
    ablation_df = read_csv_optional(ABLATION_FILE)
    pairwise_df = read_csv_optional(PAIRWISE_FILE)

    pathway_summary = compute_pathway_summary(
        wide=wide,
        weights=clean_weights,
        contribution_df=contribution_df,
        agent_perf_df=agent_perf_df,
        pairwise_df=pairwise_df,
    )

    case_df = compute_case_level_dominance(wide=wide, weights=clean_weights)
    weight_dist = compute_weight_distribution(clean_weights)
    correctness_df = compute_correctness_association(case_df)
    evidence_df = build_evidence_table(
        pathway_summary=pathway_summary,
        correctness_df=correctness_df,
        contribution_df=contribution_df,
    )

    response_text = build_reviewer_response_text(
        pathway_summary=pathway_summary,
        correctness_df=correctness_df,
        evidence_df=evidence_df,
        weights_source=weights_source,
    )

    pathway_summary.to_csv(OUT_PATHWAY_SUMMARY, index=False, encoding="utf-8-sig")
    case_df.to_csv(OUT_CASE_LEVEL, index=False, encoding="utf-8-sig")
    weight_dist.to_csv(OUT_WEIGHT_DIST, index=False, encoding="utf-8-sig")
    correctness_df.to_csv(OUT_CORRECTNESS, index=False, encoding="utf-8-sig")
    evidence_df.to_csv(OUT_EVIDENCE, index=False, encoding="utf-8-sig")

    summary = {
        "generated": str(datetime.now()),
        "prediction_input": str(PREDICTIONS_FILE),
        "adaptive_weight_source": weights_source,
        "primary_model": PRIMARY_MODEL,
        "agents": AGENTS,
        "pathway_map": PATHWAY_MAP,
        "top_mean_weight_agent": pathway_summary.sort_values("mean_adaptive_weight", ascending=False).iloc[0].to_dict(),
        "top_individual_auc_agent": pathway_summary.sort_values("agent_roc_auc", ascending=False).iloc[0].to_dict(),
        "dominant_pathway_distribution": correctness_df.to_dict(orient="records"),
        "evidence_table": evidence_df.to_dict(orient="records"),
        "important_limitation": (
            "This analysis provides organizational/pathway-level interpretability, "
            "not pixel-level lesion-mask saliency validation."
        ),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    plot_pathway_contribution(pathway_summary)
    plot_dominant_pathway(case_df)
    plot_weight_boxplot(clean_weights)

    write_report(
        pathway_summary=pathway_summary,
        case_df=case_df,
        weight_dist=weight_dist,
        correctness_df=correctness_df,
        evidence_df=evidence_df,
        response_text=response_text,
        weights_source=weights_source,
    )

    print()
    print("STAGE 2I COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Pathway summary:      {OUT_PATHWAY_SUMMARY}")
    print(f"Case-level dominance: {OUT_CASE_LEVEL}")
    print(f"Weight distribution:  {OUT_WEIGHT_DIST}")
    print(f"Correctness table:    {OUT_CORRECTNESS}")
    print(f"Evidence table:       {OUT_EVIDENCE}")
    print(f"Response text:        {OUT_RESPONSE_TEXT}")
    print(f"Report:               {OUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
