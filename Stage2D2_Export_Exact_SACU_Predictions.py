
r"""
Stage2D2_Export_Exact_SACU_Predictions.py

Purpose
-------
Export exact SACU test-set probabilities from Stage2D outputs.

This script does NOT retrain SACU.

It uses:
1. Stage2D_Agent_Predictions.csv
2. Stage2D_Adaptive_Weights.csv

and reconstructs:
- Adaptive SACU fusion probability
- Fixed equal-weight fusion probability
- Optional baseline GradientBoosting predictions from Stage2C

Output
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D2_Exact_SACU_Test_Predictions.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage2D2_Exact_SACU_Prediction_Export_Report.txt
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREDICTIONS = TABLE_DIR / "Stage2D2_Exact_SACU_Test_Predictions.csv"
OUTPUT_REPORT = REPORT_DIR / "Stage2D2_Exact_SACU_Prediction_Export_Report.txt"


AGENT_FILE = TABLE_DIR / "Stage2D_Agent_Predictions.csv"
WEIGHT_FILE = TABLE_DIR / "Stage2D_Adaptive_Weights.csv"
BASELINE_FILE = TABLE_DIR / "Stage2C_Baseline_Test_Predictions.csv"


AGENT_NAME_MAP = {
    "LocalRegionalAgent": "LocalRegionalAgent",
    "MultiViewAgent": "MultiViewAgent",
    "BilateralAgent": "BilateralAgent",
    "TemporalSpatialAgent": "TemporalSpatialAgent",
    "MetadataAgent": "MetadataAgent",
    "AdaptiveControlAgent": "AdaptiveControlAgent",
}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def find_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm = {normalize_name(c): c for c in df.columns}
    for cand in candidates:
        key = normalize_name(cand)
        if key in norm:
            return norm[key]
    return None


def find_label_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["y_true", "true_label", "label", "target", "y_test", "actual"])


def find_index_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["sample_index", "index", "test_index", "case_index", "row_id", "id"])


def find_model_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(df, ["model", "model_or_agent", "agent", "agent_name", "method"])


def find_score_column(df: pd.DataFrame) -> Optional[str]:
    return find_column(
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


def read_required_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"File is empty: {path}")
    return df


def convert_long_agent_predictions(agent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles long format:
    sample_index | model_or_agent | y_true | y_score
    """
    model_col = find_model_column(agent_df)
    score_col = find_score_column(agent_df)
    label_col = find_label_column(agent_df)
    index_col = find_index_column(agent_df)

    if model_col is None or score_col is None:
        return pd.DataFrame()

    df = agent_df.copy()

    if index_col is None:
        df["sample_index"] = df.groupby(model_col).cumcount()
        index_col = "sample_index"

    pivot = df.pivot_table(
        index=index_col,
        columns=model_col,
        values=score_col,
        aggfunc="first",
    ).reset_index()

    pivot.columns = [str(c) for c in pivot.columns]

    if label_col is not None:
        labels = (
            df[[index_col, label_col]]
            .drop_duplicates(subset=[index_col])
            .rename(columns={label_col: "y_true"})
        )
        pivot = pivot.merge(labels, on=index_col, how="left")

    pivot = pivot.rename(columns={index_col: "sample_index"})

    return pivot


def convert_wide_agent_predictions(agent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles wide format:
    y_true | LocalRegionalAgent | MultiViewAgent | ...
    """
    df = agent_df.copy()

    index_col = find_index_column(df)
    label_col = find_label_column(df)

    if index_col is None:
        df["sample_index"] = np.arange(len(df))
    else:
        df = df.rename(columns={index_col: "sample_index"})

    if label_col is not None and label_col != "y_true":
        df = df.rename(columns={label_col: "y_true"})

    return df


def standardize_agent_predictions(agent_df: pd.DataFrame) -> pd.DataFrame:
    long_version = convert_long_agent_predictions(agent_df)

    if not long_version.empty and long_version.shape[1] > 2:
        return long_version

    return convert_wide_agent_predictions(agent_df)


def standardize_weights(weight_df: pd.DataFrame) -> pd.DataFrame:
    df = weight_df.copy()

    index_col = find_index_column(df)
    if index_col is None:
        df["sample_index"] = np.arange(len(df))
    elif index_col != "sample_index":
        df = df.rename(columns={index_col: "sample_index"})

    return df


def detect_agent_probability_columns(df: pd.DataFrame) -> List[str]:
    excluded = {
        "sample_index",
        "y_true",
        "true_label",
        "label",
        "target",
        "source",
        "split",
        "dataset",
        "patient_id",
        "exam_id",
    }

    cols = []

    for col in df.columns:
        if normalize_name(col) in {normalize_name(x) for x in excluded}:
            continue

        values = pd.to_numeric(df[col], errors="coerce")

        if values.notna().mean() < 0.95:
            continue

        if values.min() >= 0 and values.max() <= 1:
            cols.append(col)

    return cols


def match_weight_column(weight_df: pd.DataFrame, agent_col: str) -> Optional[str]:
    agent_norm = normalize_name(agent_col)

    candidates = []
    for col in weight_df.columns:
        col_norm = normalize_name(col)

        if col == "sample_index":
            continue

        if agent_norm in col_norm or col_norm in agent_norm:
            candidates.append(col)

    if candidates:
        return candidates[0]

    short = agent_norm.replace("agent", "")
    for col in weight_df.columns:
        if short and short in normalize_name(col):
            return col

    return None


def build_exact_sacu_predictions(agent_df: pd.DataFrame, weight_df: pd.DataFrame) -> pd.DataFrame:
    agents = standardize_agent_predictions(agent_df)
    weights = standardize_weights(weight_df)

    merged = agents.merge(weights, on="sample_index", how="inner", suffixes=("", "_weight"))

    if merged.empty:
        raise ValueError("Could not merge Stage2D agent predictions with adaptive weights.")

    agent_cols = detect_agent_probability_columns(agents)

    if not agent_cols:
        raise ValueError("No agent probability columns were detected in Stage2D_Agent_Predictions.csv.")

    weighted_sum = np.zeros(len(merged), dtype=float)
    weight_sum = np.zeros(len(merged), dtype=float)

    used_pairs = []

    for agent_col in agent_cols:
        wcol = match_weight_column(weights, agent_col)

        if wcol is None:
            continue

        p = pd.to_numeric(merged[agent_col], errors="coerce").fillna(0).values
        w = pd.to_numeric(merged[wcol], errors="coerce").fillna(0).values

        weighted_sum += p * w
        weight_sum += w

        used_pairs.append((agent_col, wcol))

    if not used_pairs:
        raise ValueError("No matching agent-probability and adaptive-weight columns were found.")

    adaptive_score = weighted_sum / np.maximum(weight_sum, 1e-12)

    equal_score = merged[agent_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).values

    label_col = "y_true" if "y_true" in merged.columns else find_label_column(merged)
    if label_col is None:
        raise ValueError("No true-label column found in Stage2D agent predictions.")

    out = pd.DataFrame()
    out["sample_index"] = merged["sample_index"]
    out["y_true"] = pd.to_numeric(merged[label_col], errors="coerce").astype(int)
    out["SACU_AdaptiveFusion_y_score"] = np.clip(adaptive_score, 0, 1)
    out["SACU_FixedEqualFusion_y_score"] = np.clip(equal_score, 0, 1)

    for agent_col in agent_cols:
        out[f"{agent_col}_y_score"] = pd.to_numeric(merged[agent_col], errors="coerce").clip(0, 1)

    return out, used_pairs


def add_baseline_predictions(out: pd.DataFrame) -> pd.DataFrame:
    if not BASELINE_FILE.exists():
        return out

    base = pd.read_csv(BASELINE_FILE)

    score_col = find_score_column(base)
    label_col = find_label_column(base)
    index_col = find_index_column(base)
    model_col = find_model_column(base)

    if score_col is None:
        return out

    if model_col is not None:
        model_values = base[model_col].astype(str)
        gb_mask = model_values.str.lower().str.contains("gradient", regex=False)
        if gb_mask.any():
            base = base[gb_mask].copy()

    if index_col is None:
        base["sample_index"] = np.arange(len(base))
    else:
        base = base.rename(columns={index_col: "sample_index"})

    base_small = base[["sample_index", score_col]].copy()
    base_small = base_small.rename(columns={score_col: "Baseline_GradientBoosting_y_score"})

    merged = out.merge(base_small, on="sample_index", how="left")

    return merged


def save_stage2e_ready_long_format(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    score_cols = [c for c in wide_df.columns if c.endswith("_y_score")]

    for col in score_cols:
        model = col.replace("_y_score", "")

        for row in wide_df.itertuples(index=False):
            rows.append({
                "sample_index": getattr(row, "sample_index"),
                "model": model,
                "y_true": getattr(row, "y_true"),
                "y_score": getattr(row, col),
                "source": "Stage2D2_exact_export",
            })

    long_df = pd.DataFrame(rows)

    long_path = TABLE_DIR / "Stage2D2_Stage2E_Ready_Long_Predictions.csv"
    long_df.to_csv(long_path, index=False, encoding="utf-8-sig")

    return long_df


def write_report(
    wide_df: pd.DataFrame,
    long_df: pd.DataFrame,
    used_pairs: List[tuple],
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2D2 EXACT SACU PREDICTION EXPORT REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("INPUT FILES")
    lines.append("-" * 100)
    lines.append(str(AGENT_FILE))
    lines.append(str(WEIGHT_FILE))
    lines.append(str(BASELINE_FILE))
    lines.append("")

    lines.append("MATCHED AGENT-WEIGHT PAIRS")
    lines.append("-" * 100)
    for agent_col, weight_col in used_pairs:
        lines.append(f"{agent_col}  <--->  {weight_col}")
    lines.append("")

    lines.append("OUTPUT SUMMARY")
    lines.append("-" * 100)
    lines.append(f"Wide prediction rows: {len(wide_df):,}")
    lines.append(f"Long prediction rows: {len(long_df):,}")
    lines.append("")
    lines.append("Models exported:")
    lines.append(long_df["model"].value_counts().to_string())
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_PREDICTIONS))
    lines.append(str(TABLE_DIR / "Stage2D2_Stage2E_Ready_Long_Predictions.csv"))
    lines.append(str(OUTPUT_REPORT))

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    print("=" * 100)
    print("STAGE 2D2 EXPORT EXACT SACU PREDICTIONS")
    print("=" * 100)

    agent_df = read_required_csv(AGENT_FILE)
    weight_df = read_required_csv(WEIGHT_FILE)

    wide_df, used_pairs = build_exact_sacu_predictions(agent_df, weight_df)
    wide_df = add_baseline_predictions(wide_df)

    wide_df.to_csv(OUTPUT_PREDICTIONS, index=False, encoding="utf-8-sig")

    long_df = save_stage2e_ready_long_format(wide_df)

    write_report(wide_df, long_df, used_pairs)

    print()
    print("STAGE 2D2 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Exact SACU predictions: {OUTPUT_PREDICTIONS}")
    print(f"Stage2E-ready long file: {TABLE_DIR / 'Stage2D2_Stage2E_Ready_Long_Predictions.csv'}")
    print(f"Report: {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()

