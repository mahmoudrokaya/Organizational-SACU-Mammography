r"""
Stage2H_Reviewer_Response_Evidence_Package.py

Purpose
-------
Create a reviewer-response evidence package for the Frontiers Atlam 2026
mammography SACU study.

This stage consolidates already completed evidence from:
- Stage2E exact clinical threshold and operating-point analysis
- Stage2F organizational ablation and role-contribution analysis
- Stage2G bootstrap confidence intervals and paired comparisons

This script does NOT:
- preprocess images
- extract features
- rebuild matrices
- train models
- recompute predictions

It creates concise manuscript-ready and reviewer-response-ready outputs.

Inputs
------
Stage2E_Exact_Clinical_Operating_Point_Performance.csv
Stage2E_Exact_ROC_PR_Summary.csv
Stage2F_Ablation_Performance.csv
Stage2F_Role_Contribution_Loss.csv
Stage2F_Individual_Agent_Performance.csv
Stage2G_Model_Metric_CI.csv
Stage2G_Paired_Model_Comparison.csv
Stage2G_Primary_Model_Operating_Point_CI.csv

Outputs
-------
Stage2H_Key_Reviewer_Evidence_Table.csv
Stage2H_Recommended_Manuscript_Tables.csv
Stage2H_Recommended_Manuscript_Figures.csv
Stage2H_Response_Claims_With_Evidence.csv
Stage2H_Reviewer_Response_Evidence_Package.txt
Stage2H_Manuscript_Insert_Text.txt
Stage2H_Evidence_Summary.json
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"
FIGURE_DIR = PROJECT_ROOT / "results" / "figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY_MODEL = "SACU_LearnedShallowMetaFusion"
BASELINE_MODEL = "Baseline_GradientBoosting"

# Stage2E
STAGE2E_PERFORMANCE = TABLE_DIR / "Stage2E_Exact_Clinical_Operating_Point_Performance.csv"
STAGE2E_ROCPR = TABLE_DIR / "Stage2E_Exact_ROC_PR_Summary.csv"
STAGE2E_THRESHOLDS = TABLE_DIR / "Stage2E_Exact_Threshold_Comparison.csv"

# Stage2F
STAGE2F_ABLATION = TABLE_DIR / "Stage2F_Ablation_Performance.csv"
STAGE2F_CONTRIBUTION = TABLE_DIR / "Stage2F_Role_Contribution_Loss.csv"
STAGE2F_AGENT = TABLE_DIR / "Stage2F_Individual_Agent_Performance.csv"

# Stage2G
STAGE2G_MODEL_CI = TABLE_DIR / "Stage2G_Model_Metric_CI.csv"
STAGE2G_PAIRWISE = TABLE_DIR / "Stage2G_Paired_Model_Comparison.csv"
STAGE2G_PRIMARY_OP_CI = TABLE_DIR / "Stage2G_Primary_Model_Operating_Point_CI.csv"

# Outputs
OUT_KEY_EVIDENCE = TABLE_DIR / "Stage2H_Key_Reviewer_Evidence_Table.csv"
OUT_MANUSCRIPT_TABLES = TABLE_DIR / "Stage2H_Recommended_Manuscript_Tables.csv"
OUT_MANUSCRIPT_FIGURES = TABLE_DIR / "Stage2H_Recommended_Manuscript_Figures.csv"
OUT_RESPONSE_CLAIMS = TABLE_DIR / "Stage2H_Response_Claims_With_Evidence.csv"
OUT_JSON = TABLE_DIR / "Stage2H_Evidence_Summary.json"

OUT_PACKAGE = REPORT_DIR / "Stage2H_Reviewer_Response_Evidence_Package.txt"
OUT_INSERT_TEXT = REPORT_DIR / "Stage2H_Manuscript_Insert_Text.txt"


# =============================================================================
# Helpers
# =============================================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file is missing: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Required file is empty: {path}")
    return df


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def fmt(value, digits: int = 4) -> str:
    try:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def fmt_ci(point, low, high, digits: int = 4) -> str:
    return f"{fmt(point, digits)} ({fmt(low, digits)}–{fmt(high, digits)})"


def extract_metric_ci(model_ci: pd.DataFrame, model: str, metric: str) -> Dict[str, float]:
    row = model_ci[
        (model_ci["model"] == model)
        & (model_ci["metric"] == metric)
    ]

    if row.empty:
        return {
            "point": np.nan,
            "low": np.nan,
            "high": np.nan,
        }

    r = row.iloc[0]

    return {
        "point": float(r["point_estimate"]),
        "low": float(r["ci_lower_95"]),
        "high": float(r["ci_upper_95"]),
    }


def extract_stage2e_op(perf: pd.DataFrame, model: str, op: str) -> Dict[str, float]:
    row = perf[
        (perf["model"] == model)
        & (perf["operating_point"] == op)
    ]

    if row.empty:
        return {}

    return row.iloc[0].to_dict()


def extract_pairwise(pairwise: pd.DataFrame, comparison_model: str, metric: str) -> Dict[str, float]:
    row = pairwise[
        (pairwise["primary_model"] == PRIMARY_MODEL)
        & (pairwise["comparison_model"] == comparison_model)
        & (pairwise["metric"] == metric)
    ]

    if row.empty:
        return {}

    return row.iloc[0].to_dict()


# =============================================================================
# Evidence table construction
# =============================================================================

def build_key_evidence_table(
    stage2e_perf: pd.DataFrame,
    stage2e_rocp: pd.DataFrame,
    stage2f_ablation: pd.DataFrame,
    stage2f_contribution: pd.DataFrame,
    stage2f_agent: pd.DataFrame,
    stage2g_ci: pd.DataFrame,
    stage2g_pairwise: pd.DataFrame,
    stage2g_op_ci: pd.DataFrame,
) -> pd.DataFrame:

    rows = []

    sacu_auc = extract_metric_ci(stage2g_ci, PRIMARY_MODEL, "roc_auc")
    sacu_ba = extract_metric_ci(stage2g_ci, PRIMARY_MODEL, "balanced_accuracy")
    baseline_auc = extract_metric_ci(stage2g_ci, BASELINE_MODEL, "roc_auc")
    baseline_ba = extract_metric_ci(stage2g_ci, BASELINE_MODEL, "balanced_accuracy")

    rows.append({
        "evidence_topic": "Primary SACU discrimination",
        "main_result": f"ROC-AUC = {fmt_ci(sacu_auc['point'], sacu_auc['low'], sacu_auc['high'])}",
        "supporting_result": f"Balanced accuracy = {fmt_ci(sacu_ba['point'], sacu_ba['low'], sacu_ba['high'])}",
        "source_stage": "Stage2G",
        "recommended_use": "Main Results / Statistical validation",
        "claim_strength": "Supported with bootstrap CI",
    })

    rows.append({
        "evidence_topic": "Baseline comparison",
        "main_result": f"Baseline ROC-AUC = {fmt_ci(baseline_auc['point'], baseline_auc['low'], baseline_auc['high'])}",
        "supporting_result": f"Baseline balanced accuracy = {fmt_ci(baseline_ba['point'], baseline_ba['low'], baseline_ba['high'])}",
        "source_stage": "Stage2G",
        "recommended_use": "Fair comparison paragraph",
        "claim_strength": "Do not claim significant ROC-AUC superiority over baseline",
    })

    baseline_pair = extract_pairwise(stage2g_pairwise, BASELINE_MODEL, "roc_auc")
    if baseline_pair:
        rows.append({
            "evidence_topic": "Paired SACU vs GradientBoosting ROC-AUC comparison",
            "main_result": (
                f"ΔROC-AUC = {fmt(baseline_pair['point_difference_primary_minus_comparison'])}, "
                f"95% CI {fmt(baseline_pair['ci_lower_95_difference'])}–"
                f"{fmt(baseline_pair['ci_upper_95_difference'])}"
            ),
            "supporting_result": f"bootstrap p = {fmt(baseline_pair['bootstrap_p_two_sided'], 3)}",
            "source_stage": "Stage2G",
            "recommended_use": "Reviewer response on statistical testing",
            "claim_strength": "Difference not statistically significant",
        })

    youden = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "Youden")
    default = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "Default_0.50")

    if youden and default:
        rows.append({
            "evidence_topic": "Clinical operating-point optimization",
            "main_result": (
                f"Youden threshold = {fmt(youden['threshold'])}; "
                f"balanced accuracy = {fmt(youden['balanced_accuracy'])}"
            ),
            "supporting_result": (
                f"Default balanced accuracy = {fmt(default['balanced_accuracy'])}; "
                f"specificity improved from {fmt(default['specificity'])} to {fmt(youden['specificity'])}"
            ),
            "source_stage": "Stage2E",
            "recommended_use": "Clinical threshold paragraph / Table",
            "claim_strength": "Supported as post-hoc operating-point analysis",
        })

    high_sens = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "HighSensitivity_0.90")
    if high_sens:
        rows.append({
            "evidence_topic": "High-sensitivity operating point",
            "main_result": f"Sensitivity = {fmt(high_sens['sensitivity_recall'])} at threshold {fmt(high_sens['threshold'])}",
            "supporting_result": (
                f"Specificity = {fmt(high_sens['specificity'])}; "
                f"NPV = {fmt(high_sens['npv'])}"
            ),
            "source_stage": "Stage2E",
            "recommended_use": "Supplementary clinical deployment analysis",
            "claim_strength": "Useful but specificity trade-off must be stated",
        })

    best_agents = (
        stage2f_agent
        .sort_values("roc_auc", ascending=False)
        .head(2)
    )

    if not best_agents.empty:
        agent_text = "; ".join(
            f"{r['agent']} ROC-AUC={fmt(r['roc_auc'])}"
            for _, r in best_agents.iterrows()
        )
        rows.append({
            "evidence_topic": "Individual organizational agents",
            "main_result": agent_text,
            "supporting_result": "Individual agents did not exceed learned SACU fusion in balanced accuracy.",
            "source_stage": "Stage2F",
            "recommended_use": "Ablation / organizational role contribution",
            "claim_strength": "Supports value of fusion over isolated agents",
        })

    if not stage2f_contribution.empty:
        best_loss = stage2f_contribution.sort_values("delta_roc_auc", ascending=False).iloc[0]
        rows.append({
            "evidence_topic": "Largest positive role contribution",
            "main_result": (
                f"Removing {best_loss['removed_agent']} caused ΔROC-AUC loss = "
                f"{fmt(best_loss['delta_roc_auc'])}"
            ),
            "supporting_result": (
                f"ΔAP = {fmt(best_loss['delta_average_precision'])}; "
                f"ΔF1 = {fmt(best_loss['delta_f1'])}"
            ),
            "source_stage": "Stage2F",
            "recommended_use": "Role-contribution table",
            "claim_strength": "Use cautiously as equal-weight leave-one-out evidence",
        })

    significant_agents = stage2g_pairwise[
        (stage2g_pairwise["metric"] == "roc_auc")
        & (stage2g_pairwise["bootstrap_p_two_sided"] < 0.05)
        & (stage2g_pairwise["point_difference_primary_minus_comparison"] > 0)
    ].copy()

    if not significant_agents.empty:
        names = ", ".join(significant_agents["comparison_model"].tolist())
        rows.append({
            "evidence_topic": "Statistically supported SACU superiority over weaker agents",
            "main_result": f"Primary SACU significantly exceeded: {names}",
            "supporting_result": "Bootstrap paired comparisons used 2,000 resamples.",
            "source_stage": "Stage2G",
            "recommended_use": "Statistical comparison paragraph",
            "claim_strength": "Supported for listed agents only",
        })

    return pd.DataFrame(rows)


def build_recommended_tables() -> pd.DataFrame:
    rows = [
        {
            "table_id": "Table 10",
            "proposed_title": "Clinical operating-point analysis of the learned SACU fusion",
            "source_files": (
                "Stage2E_Exact_Clinical_Operating_Point_Performance.csv; "
                "Stage2G_Primary_Model_Operating_Point_CI.csv"
            ),
            "main_content": (
                "Default, Youden, high-sensitivity, and high-specificity thresholds; "
                "balanced accuracy, sensitivity, specificity, PPV, NPV, F1, MCC"
            ),
            "reviewer_issue_addressed": (
                "Threshold selection, calibration-like operating behavior, clinical deployment interpretation"
            ),
        },
        {
            "table_id": "Table 11",
            "proposed_title": "Organizational role contribution and SACU ablation analysis",
            "source_files": (
                "Stage2F_Ablation_Performance.csv; "
                "Stage2F_Role_Contribution_Loss.csv; "
                "Stage2F_Individual_Agent_Performance.csv"
            ),
            "main_content": (
                "Full learned fusion, equal-weight fusion, leave-one-agent-out conditions, "
                "individual agents, and role contribution loss"
            ),
            "reviewer_issue_addressed": (
                "Ablation evidence, role contribution, model component necessity"
            ),
        },
        {
            "table_id": "Table 12",
            "proposed_title": "Bootstrap confidence intervals for baseline, SACU fusion, and organizational agents",
            "source_files": "Stage2G_Model_Metric_CI.csv",
            "main_content": (
                "ROC-AUC, PR-AUC, balanced accuracy, sensitivity, specificity, F1, MCC with 95% CIs"
            ),
            "reviewer_issue_addressed": (
                "Point estimates, uncertainty ranges, statistical reporting"
            ),
        },
        {
            "table_id": "Table 13",
            "proposed_title": "Paired bootstrap model-comparison analysis",
            "source_files": "Stage2G_Paired_Model_Comparison.csv",
            "main_content": (
                "Primary SACU vs GradientBoosting baseline and agents; paired metric differences, "
                "95% CIs, bootstrap p-values"
            ),
            "reviewer_issue_addressed": (
                "Statistical model comparison and fair interpretation of superiority claims"
            ),
        },
    ]

    return pd.DataFrame(rows)


def build_recommended_figures() -> pd.DataFrame:
    rows = [
        {
            "figure_id": "Figure 11",
            "proposed_title": "Clinical ROC and precision–recall operating-point curves",
            "source_files": (
                "Stage2E_Exact_ROC_Operating_Points.png; "
                "Stage2E_Exact_PR_Operating_Points.png"
            ),
            "main_message": "Learned SACU fusion has clinically interpretable threshold-dependent behavior.",
            "reviewer_issue_addressed": "ROC/PR curves and decision-threshold transparency",
        },
        {
            "figure_id": "Figure 12",
            "proposed_title": "Bootstrap confidence intervals for ROC-AUC and balanced accuracy",
            "source_files": (
                "Stage2G_ROCAUC_CI_Bar.png; "
                "Stage2G_BalancedAccuracy_CI_Bar.png"
            ),
            "main_message": "Uncertainty ranges show SACU is statistically comparable to the strongest baseline and stronger than weaker agents.",
            "reviewer_issue_addressed": "Confidence intervals and statistical uncertainty",
        },
        {
            "figure_id": "Figure 13",
            "proposed_title": "Organizational role contribution analysis",
            "source_files": (
                "Stage2F_Ablation_BalancedAccuracy_Bar.png; "
                "Stage2F_Role_Contribution_Loss_Bar.png"
            ),
            "main_message": "Bilateral and local-regional roles provide the clearest positive marginal contributions.",
            "reviewer_issue_addressed": "Ablation and organizational role contribution",
        },
    ]

    return pd.DataFrame(rows)


def build_response_claims() -> pd.DataFrame:
    rows = [
        {
            "reviewer_concern": "Results were reported mainly as point estimates.",
            "response_claim": (
                "We added bootstrap 95% confidence intervals for ROC-AUC, PR-AUC, "
                "balanced accuracy, sensitivity, specificity, F1, and MCC."
            ),
            "evidence_source": "Stage2G_Model_Metric_CI.csv",
            "safe_language": (
                "The revised analysis reports uncertainty ranges rather than relying only on point estimates."
            ),
        },
        {
            "reviewer_concern": "Model comparisons lacked statistical testing.",
            "response_claim": (
                "We added paired bootstrap comparisons between the learned SACU fusion, "
                "GradientBoosting baseline, fusion variants, and individual organizational agents."
            ),
            "evidence_source": "Stage2G_Paired_Model_Comparison.csv",
            "safe_language": (
                "The learned SACU fusion was statistically higher than weaker agents, "
                "while its ROC-AUC difference from GradientBoosting was not statistically significant."
            ),
        },
        {
            "reviewer_concern": "Clinical threshold selection was unclear.",
            "response_claim": (
                "We added a clinical operating-point analysis comparing default, Youden, "
                "high-sensitivity, and high-specificity thresholds."
            ),
            "evidence_source": "Stage2E_Exact_Clinical_Operating_Point_Performance.csv",
            "safe_language": (
                "Threshold optimization improved balanced operating behavior without changing the trained model."
            ),
        },
        {
            "reviewer_concern": "The contribution of organizational roles was insufficiently demonstrated.",
            "response_claim": (
                "We added individual-agent performance, leave-one-agent-out ablation, "
                "and role contribution loss analyses."
            ),
            "evidence_source": "Stage2F_Ablation_Performance.csv; Stage2F_Role_Contribution_Loss.csv",
            "safe_language": (
                "Bilateral and local-regional roles showed the clearest positive marginal signal; "
                "we avoid claiming that every role independently improves performance."
            ),
        },
        {
            "reviewer_concern": "Claims of superiority were too strong.",
            "response_claim": (
                "We revised the interpretation to distinguish discrimination, balanced operating behavior, "
                "and statistical significance."
            ),
            "evidence_source": "Stage2G_Paired_Model_Comparison.csv",
            "safe_language": (
                "The revised manuscript no longer claims universal state-of-the-art superiority."
            ),
        },
    ]

    return pd.DataFrame(rows)


# =============================================================================
# Text outputs
# =============================================================================

def build_manuscript_insert_text(
    stage2e_perf: pd.DataFrame,
    stage2g_ci: pd.DataFrame,
    stage2g_pairwise: pd.DataFrame,
    stage2f_agent: pd.DataFrame,
    stage2f_contribution: pd.DataFrame,
) -> str:

    sacu_auc = extract_metric_ci(stage2g_ci, PRIMARY_MODEL, "roc_auc")
    sacu_ba = extract_metric_ci(stage2g_ci, PRIMARY_MODEL, "balanced_accuracy")
    baseline_auc = extract_metric_ci(stage2g_ci, BASELINE_MODEL, "roc_auc")
    baseline_pair = extract_pairwise(stage2g_pairwise, BASELINE_MODEL, "roc_auc")

    default = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "Default_0.50")
    youden = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "Youden")
    high_sens = extract_stage2e_op(stage2e_perf, PRIMARY_MODEL, "HighSensitivity_0.90")

    best_agents = stage2f_agent.sort_values("roc_auc", ascending=False).head(2)
    agent_sentence = "the strongest individual agents were not available"
    if not best_agents.empty:
        agent_sentence = " and ".join(
            f"{r['agent']} (ROC-AUC={fmt(r['roc_auc'])})"
            for _, r in best_agents.iterrows()
        )

    contribution_sentence = "role contribution analysis was not available"
    if not stage2f_contribution.empty:
        best_loss = stage2f_contribution.sort_values("delta_roc_auc", ascending=False).iloc[0]
        contribution_sentence = (
            f"removing {best_loss['removed_agent']} produced the largest positive "
            f"equal-weight fusion loss (ΔROC-AUC={fmt(best_loss['delta_roc_auc'])})"
        )

    lines = []

    lines.append("Suggested Results Text")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        "To address the reviewers' request for statistical uncertainty and clinically "
        "interpretable decision behavior, we added a post-hoc operating-point and "
        "bootstrap validation analysis using the held-out test predictions. The learned "
        f"SACU meta-fusion achieved a ROC-AUC of {fmt_ci(sacu_auc['point'], sacu_auc['low'], sacu_auc['high'])} "
        f"and a balanced accuracy of {fmt_ci(sacu_ba['point'], sacu_ba['low'], sacu_ba['high'])} "
        "at the default threshold."
    )
    lines.append("")

    if baseline_pair:
        lines.append(
            f"The GradientBoosting baseline achieved ROC-AUC {fmt_ci(baseline_auc['point'], baseline_auc['low'], baseline_auc['high'])}. "
            f"The paired bootstrap difference between SACU and GradientBoosting was "
            f"ΔROC-AUC={fmt(baseline_pair['point_difference_primary_minus_comparison'])}, "
            f"95% CI {fmt(baseline_pair['ci_lower_95_difference'])}–"
            f"{fmt(baseline_pair['ci_upper_95_difference'])}, "
            f"p={fmt(baseline_pair['bootstrap_p_two_sided'], 3)}. "
            "Accordingly, we interpret SACU as statistically comparable to the strongest "
            "baseline in discrimination, rather than claiming significant ROC-AUC superiority."
        )
        lines.append("")

    if default and youden:
        lines.append(
            f"Clinical operating-point analysis showed that the default threshold of 0.50 produced "
            f"balanced accuracy {fmt(default['balanced_accuracy'])}, sensitivity {fmt(default['sensitivity_recall'])}, "
            f"and specificity {fmt(default['specificity'])}. The Youden operating point "
            f"(threshold={fmt(youden['threshold'])}) improved balanced accuracy to "
            f"{fmt(youden['balanced_accuracy'])} and specificity to {fmt(youden['specificity'])}, "
            "while preserving the same sensitivity in this test cohort."
        )
        lines.append("")

    if high_sens:
        lines.append(
            f"A high-sensitivity operating point reached sensitivity {fmt(high_sens['sensitivity_recall'])} "
            f"with NPV {fmt(high_sens['npv'])}, but at the expected cost of reduced specificity "
            f"({fmt(high_sens['specificity'])}). This operating point is therefore reported as "
            "a deployment trade-off rather than the primary balanced operating point."
        )
        lines.append("")

    lines.append(
        f"The organizational role analysis showed that {agent_sentence}. In the leave-one-role-out "
        f"analysis, {contribution_sentence}. These findings support the use of role-conditioned "
        "fusion while also showing that the marginal contribution differs across roles."
    )
    lines.append("")

    lines.append("Suggested Reviewer-Response Text")
    lines.append("=" * 100)
    lines.append("")
    lines.append(
        "We thank the reviewer for requesting stronger statistical and clinical validation. "
        "In the revised analysis, we added bootstrap 95% confidence intervals, paired bootstrap "
        "model comparisons, clinical operating-point analysis, and organizational role ablation. "
        "These additions clarify that the learned SACU fusion provides the strongest balanced "
        "operating behavior and improves over weaker organizational agents, while its ROC-AUC "
        "difference from the strongest GradientBoosting baseline is not statistically significant. "
        "We therefore revised the manuscript language to avoid overstated claims and to distinguish "
        "discrimination, threshold-dependent clinical behavior, and statistically supported comparisons."
    )

    return "\n".join(lines)


def write_text_package(
    key_evidence: pd.DataFrame,
    manuscript_tables: pd.DataFrame,
    manuscript_figures: pd.DataFrame,
    response_claims: pd.DataFrame,
    insert_text: str,
):
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2H REVIEWER RESPONSE EVIDENCE PACKAGE")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("KEY REVIEWER EVIDENCE")
    lines.append("-" * 100)
    lines.append(key_evidence.to_string(index=False))
    lines.append("")

    lines.append("RECOMMENDED MANUSCRIPT TABLES")
    lines.append("-" * 100)
    lines.append(manuscript_tables.to_string(index=False))
    lines.append("")

    lines.append("RECOMMENDED MANUSCRIPT FIGURES")
    lines.append("-" * 100)
    lines.append(manuscript_figures.to_string(index=False))
    lines.append("")

    lines.append("RESPONSE CLAIMS WITH SAFE LANGUAGE")
    lines.append("-" * 100)
    lines.append(response_claims.to_string(index=False))
    lines.append("")

    lines.append("MANUSCRIPT / RESPONSE INSERT TEXT")
    lines.append("-" * 100)
    lines.append(insert_text)
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUT_KEY_EVIDENCE,
        OUT_MANUSCRIPT_TABLES,
        OUT_MANUSCRIPT_FIGURES,
        OUT_RESPONSE_CLAIMS,
        OUT_JSON,
        OUT_PACKAGE,
        OUT_INSERT_TEXT,
    ]:
        lines.append(str(p))

    with open(OUT_PACKAGE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    with open(OUT_INSERT_TEXT, "w", encoding="utf-8") as f:
        f.write(insert_text)


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 100)
    print("STAGE 2H REVIEWER RESPONSE EVIDENCE PACKAGE")
    print("=" * 100)

    stage2e_perf = read_csv_required(STAGE2E_PERFORMANCE)
    stage2e_rocp = read_csv_required(STAGE2E_ROCPR)
    stage2e_thresholds = read_csv_optional(STAGE2E_THRESHOLDS)

    stage2f_ablation = read_csv_required(STAGE2F_ABLATION)
    stage2f_contribution = read_csv_required(STAGE2F_CONTRIBUTION)
    stage2f_agent = read_csv_required(STAGE2F_AGENT)

    stage2g_ci = read_csv_required(STAGE2G_MODEL_CI)
    stage2g_pairwise = read_csv_required(STAGE2G_PAIRWISE)
    stage2g_op_ci = read_csv_required(STAGE2G_PRIMARY_OP_CI)

    key_evidence = build_key_evidence_table(
        stage2e_perf=stage2e_perf,
        stage2e_rocp=stage2e_rocp,
        stage2f_ablation=stage2f_ablation,
        stage2f_contribution=stage2f_contribution,
        stage2f_agent=stage2f_agent,
        stage2g_ci=stage2g_ci,
        stage2g_pairwise=stage2g_pairwise,
        stage2g_op_ci=stage2g_op_ci,
    )

    manuscript_tables = build_recommended_tables()
    manuscript_figures = build_recommended_figures()
    response_claims = build_response_claims()

    insert_text = build_manuscript_insert_text(
        stage2e_perf=stage2e_perf,
        stage2g_ci=stage2g_ci,
        stage2g_pairwise=stage2g_pairwise,
        stage2f_agent=stage2f_agent,
        stage2f_contribution=stage2f_contribution,
    )

    key_evidence.to_csv(OUT_KEY_EVIDENCE, index=False, encoding="utf-8-sig")
    manuscript_tables.to_csv(OUT_MANUSCRIPT_TABLES, index=False, encoding="utf-8-sig")
    manuscript_figures.to_csv(OUT_MANUSCRIPT_FIGURES, index=False, encoding="utf-8-sig")
    response_claims.to_csv(OUT_RESPONSE_CLAIMS, index=False, encoding="utf-8-sig")

    summary = {
        "generated": str(datetime.now()),
        "primary_model": PRIMARY_MODEL,
        "baseline_model": BASELINE_MODEL,
        "input_files": {
            "stage2e_performance": str(STAGE2E_PERFORMANCE),
            "stage2e_rocp": str(STAGE2E_ROCPR),
            "stage2f_ablation": str(STAGE2F_ABLATION),
            "stage2f_contribution": str(STAGE2F_CONTRIBUTION),
            "stage2f_agent": str(STAGE2F_AGENT),
            "stage2g_model_ci": str(STAGE2G_MODEL_CI),
            "stage2g_pairwise": str(STAGE2G_PAIRWISE),
            "stage2g_primary_op_ci": str(STAGE2G_PRIMARY_OP_CI),
        },
        "key_evidence": key_evidence.to_dict(orient="records"),
        "recommended_tables": manuscript_tables.to_dict(orient="records"),
        "recommended_figures": manuscript_figures.to_dict(orient="records"),
        "response_claims": response_claims.to_dict(orient="records"),
    }

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    write_text_package(
        key_evidence=key_evidence,
        manuscript_tables=manuscript_tables,
        manuscript_figures=manuscript_figures,
        response_claims=response_claims,
        insert_text=insert_text,
    )

    print()
    print("STAGE 2H COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Key evidence table:       {OUT_KEY_EVIDENCE}")
    print(f"Recommended tables:       {OUT_MANUSCRIPT_TABLES}")
    print(f"Recommended figures:      {OUT_MANUSCRIPT_FIGURES}")
    print(f"Response claims:          {OUT_RESPONSE_CLAIMS}")
    print(f"Evidence summary JSON:    {OUT_JSON}")
    print(f"Reviewer package report:  {OUT_PACKAGE}")
    print(f"Manuscript insert text:   {OUT_INSERT_TEXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()
