r"""
Stage2B_Build_SACU_Modeling_Matrices.py

Purpose
-------
Build leakage-safe supervised modeling matrices for the SACU framework.

This stage converts Stage2A organizational view features into clean train/test
matrices for the first supervised VinDr-Mammo experiment.

This script is designed to remain 100% consistent with the Methods:
1. Uses the VinDr complete four-view labeled cohort.
2. Preserves the official VinDr training/test split.
3. Uses only features produced by Stage2A:
   - local regional descriptors
   - multi-view CC/MLO descriptors
   - bilateral asymmetry descriptors
   - same-exam temporal-spatial descriptors
   - metadata descriptors
   - SACU complexity/resource cues
4. Does not create real longitudinal features.
5. Does not allow labels, BI-RADS-derived label proxies, IDs, paths, or split fields into X.
6. Fits imputation and scaling parameters on training data only.
7. Saves feature groups for later SACU branch-specific modeling.

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\features\\Stage2A_VinDr_Complete_Four_View_Features.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\features\\Stage2A_Feature_Dictionary.csv

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_Feature_Groups.json
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_Preprocessing_Params.json
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2B_Modeling_Matrix_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2B_Leakage_Audit.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage2B_Modeling_Matrix_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage2B_Build_SACU_Modeling_Matrices.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

FEATURE_DIR = PROJECT_ROOT / "features"
MATRIX_DIR = PROJECT_ROOT / "matrices"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

MATRIX_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FEATURES = FEATURE_DIR / "Stage2A_VinDr_Complete_Four_View_Features.csv"
INPUT_FEATURE_DICTIONARY = FEATURE_DIR / "Stage2A_Feature_Dictionary.csv"

OUTPUT_X_TRAIN = MATRIX_DIR / "Stage2B_X_train.csv"
OUTPUT_Y_TRAIN = MATRIX_DIR / "Stage2B_y_train.csv"
OUTPUT_X_TEST = MATRIX_DIR / "Stage2B_X_test.csv"
OUTPUT_Y_TEST = MATRIX_DIR / "Stage2B_y_test.csv"

OUTPUT_META_TRAIN = MATRIX_DIR / "Stage2B_train_metadata.csv"
OUTPUT_META_TEST = MATRIX_DIR / "Stage2B_test_metadata.csv"

OUTPUT_FEATURE_COLUMNS = MATRIX_DIR / "Stage2B_Feature_Columns.csv"
OUTPUT_FEATURE_GROUPS = MATRIX_DIR / "Stage2B_Feature_Groups.json"
OUTPUT_PREPROCESSING_PARAMS = MATRIX_DIR / "Stage2B_Preprocessing_Params.json"

OUTPUT_SUMMARY = RESULTS_TABLE_DIR / "Stage2B_Modeling_Matrix_Summary.csv"
OUTPUT_LEAKAGE_AUDIT = RESULTS_TABLE_DIR / "Stage2B_Leakage_Audit.csv"
OUTPUT_CLASS_BALANCE = RESULTS_TABLE_DIR / "Stage2B_Class_Balance_Audit.csv"
OUTPUT_FEATURE_GROUP_SUMMARY = RESULTS_TABLE_DIR / "Stage2B_Feature_Group_Summary.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage2B_Modeling_Matrix_Summary.json"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage2B_Modeling_Matrix_Report.txt"

RANDOM_SEED = 42

TARGET_COLUMN = "exam_label"
SPLIT_COLUMN = "normalized_split"

TRAIN_SPLIT_VALUE = "training"
TEST_SPLIT_VALUE = "test"


# =============================================================================
# Leakage Controls
# =============================================================================

EXPLICIT_NON_FEATURE_COLUMNS = {
    "record_id",
    "dataset",
    "patient_id",
    "study_id",
    "exam_id",
    "split",
    "normalized_split",
    "task_name",
    "task_family",
    "modeling_role",
    "available_views",
    "missing_views",
    "exam_label",
    "exam_label_text",
    "breast_density",
    "feature_extraction_status",
}

# These columns are label proxies and must not enter X.
# In the current setup, the target is derived from breast-level BI-RADS.
# Therefore, BI-RADS numeric fields and per-view label fields are excluded
# from supervised feature matrix to prevent label leakage.
LEAKAGE_NAME_PATTERNS = [
    r"(^|_)label($|_)",
    r"exam_label",
    r"label_text",
    r"birads",
    r"bi_rads",
    r"breast_birads",
    r"finding_birads",
    r"malignant",
    r"benign",
    r"target",
    r"diagnosis",
    r"outcome",
    r"split",
    r"patient",
    r"study",
    r"exam_id",
    r"record_id",
    r"image_id",
    r"path",
    r"processed_path",
    r"source_path",
    r"manifest",
    r"status",
]

# Features that are metadata but not direct label proxies may be retained.
# Density is a clinical covariate, but the raw string column is identity/non-numeric.
ALLOWED_METADATA_FEATURES = {
    "meta_breast_density_numeric",
    "meta_has_label",  # later excluded below because it is constant and target-related
}

FORCE_EXCLUDE_COLUMNS = {
    "meta_has_label",
}


# =============================================================================
# Helpers
# =============================================================================

def safe_str(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_split(value: str) -> str:
    v = safe_str(value).lower()

    if v in {"train", "training"}:
        return "training"
    if v in {"test", "testing"}:
        return "test"
    if v in {"validation", "valid", "val"}:
        return "validation"
    if v == "external":
        return "external"
    return v if v else "unknown"


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def is_numeric_series(s: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(s):
        return True
    converted = pd.to_numeric(s, errors="coerce")
    valid_ratio = converted.notna().mean()
    return valid_ratio > 0.95


def matches_any_pattern(name: str, patterns: List[str]) -> bool:
    lower = name.lower()
    return any(re.search(p, lower) for p in patterns)


def is_feature_column(col: str, df: pd.DataFrame) -> Tuple[bool, str]:
    if col in EXPLICIT_NON_FEATURE_COLUMNS:
        return False, "explicit_non_feature_column"

    if col in FORCE_EXCLUDE_COLUMNS:
        return False, "force_excluded_column"

    if matches_any_pattern(col, LEAKAGE_NAME_PATTERNS):
        if col in ALLOWED_METADATA_FEATURES and col not in FORCE_EXCLUDE_COLUMNS:
            return True, "allowed_metadata_feature"
        return False, "leakage_or_identity_name_pattern"

    if not is_numeric_series(df[col]):
        return False, "non_numeric_column"

    return True, "accepted_numeric_feature"


def get_feature_family(col: str) -> str:
    if col.startswith(("lcc_", "lmlo_", "rcc_", "rmlo_")):
        return "local_regional_descriptor"
    if col.startswith("mv_"):
        return "multi_view_reasoning"
    if col.startswith("bi_"):
        return "bilateral_asymmetry"
    if col.startswith("ts_"):
        return "same_exam_temporal_spatial"
    if col.startswith("meta_"):
        return "clinical_metadata"
    if col.startswith("sacu_"):
        return "adaptive_sacu_control"
    return "other_numeric"


def build_feature_groups(feature_cols: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {
        "local_regional_descriptor": [],
        "multi_view_reasoning": [],
        "bilateral_asymmetry": [],
        "same_exam_temporal_spatial": [],
        "clinical_metadata": [],
        "adaptive_sacu_control": [],
        "other_numeric": [],
    }

    for col in feature_cols:
        family = get_feature_family(col)
        groups.setdefault(family, []).append(col)

    return groups


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.copy()

    if SPLIT_COLUMN not in d.columns:
        if "split" not in d.columns:
            raise ValueError("No split or normalized_split column found.")
        d[SPLIT_COLUMN] = d["split"].map(normalize_split)
    else:
        d[SPLIT_COLUMN] = d[SPLIT_COLUMN].map(normalize_split)

    train_df = d[d[SPLIT_COLUMN] == TRAIN_SPLIT_VALUE].copy()
    test_df = d[d[SPLIT_COLUMN] == TEST_SPLIT_VALUE].copy()

    if train_df.empty:
        raise ValueError("Training split is empty.")
    if test_df.empty:
        raise ValueError("Test split is empty.")

    return train_df, test_df


def build_label_vector(df: pd.DataFrame) -> pd.Series:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column not found: {TARGET_COLUMN}")

    y = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")

    if y.isna().any():
        bad = int(y.isna().sum())
        raise ValueError(f"Target contains {bad} missing/non-numeric values.")

    return y.astype(int)


def impute_and_scale_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:

    X_train_raw = train_df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_test_raw = test_df[feature_cols].apply(pd.to_numeric, errors="coerce")

    params: Dict[str, Dict[str, float]] = {}

    X_train = pd.DataFrame(index=train_df.index)
    X_test = pd.DataFrame(index=test_df.index)

    for col in feature_cols:
        train_col = X_train_raw[col].copy()
        test_col = X_test_raw[col].copy()

        median = float(train_col.median()) if train_col.notna().any() else 0.0

        train_filled = train_col.fillna(median)
        test_filled = test_col.fillna(median)

        mean = float(train_filled.mean())
        std = float(train_filled.std(ddof=0))

        if std == 0 or np.isnan(std):
            std = 1.0

        X_train[col] = (train_filled - mean) / std
        X_test[col] = (test_filled - mean) / std

        params[col] = {
            "impute_median_train_only": median,
            "scale_mean_train_only": mean,
            "scale_std_train_only": std,
            "train_missing_count": int(train_col.isna().sum()),
            "test_missing_count": int(test_col.isna().sum()),
        }

    return X_train, X_test, params


def build_metadata(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "record_id",
        "dataset",
        "patient_id",
        "study_id",
        "exam_id",
        "split",
        "normalized_split",
        "exam_label",
        "exam_label_text",
        "breast_density",
        "available_views",
        "missing_views",
        "modeling_role",
    ]

    out = pd.DataFrame(index=df.index)

    for col in cols:
        out[col] = df[col] if col in df.columns else ""

    return out


# =============================================================================
# Audits
# =============================================================================

def build_leakage_audit(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    rows = []
    accepted = []

    for col in df.columns:
        is_feat, reason = is_feature_column(col, df)

        if is_feat:
            accepted.append(col)

        rows.append({
            "column": col,
            "accepted_as_feature": bool(is_feat),
            "reason": reason,
            "feature_family": get_feature_family(col) if is_feat else "",
            "dtype": str(df[col].dtype),
            "missing_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique(dropna=True)),
        })

    audit_df = pd.DataFrame(rows)

    return audit_df, accepted


def build_class_balance(train_y: pd.Series, test_y: pd.Series) -> pd.DataFrame:
    rows = []

    for split_name, y in [("training", train_y), ("test", test_y), ("all", pd.concat([train_y, test_y]))]:
        n = int(len(y))
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())

        rows.append({
            "split": split_name,
            "records": n,
            "positive": pos,
            "negative": neg,
            "positive_rate": pos / n if n > 0 else np.nan,
            "negative_rate": neg / n if n > 0 else np.nan,
            "class_weight_negative": n / (2 * neg) if neg > 0 else np.nan,
            "class_weight_positive": n / (2 * pos) if pos > 0 else np.nan,
        })

    return pd.DataFrame(rows)


def build_feature_group_summary(feature_groups: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []

    for family, cols in feature_groups.items():
        rows.append({
            "feature_family": family,
            "n_features": int(len(cols)),
            "sample_features": "|".join(cols[:15]),
        })

    return pd.DataFrame(rows)


def check_patient_overlap(train_meta: pd.DataFrame, test_meta: pd.DataFrame) -> Dict[str, object]:
    train_patients = set(train_meta["patient_id"].astype(str).tolist())
    test_patients = set(test_meta["patient_id"].astype(str).tolist())
    overlap = sorted(train_patients.intersection(test_patients))

    return {
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "overlap_patients": len(overlap),
        "overlap_patient_sample": overlap[:20],
    }


def build_summary(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_cols: List[str],
    feature_groups: Dict[str, List[str]],
    overlap_info: Dict[str, object],
) -> pd.DataFrame:

    rows = []

    rows.append({
        "item": "X_train_records",
        "value": int(X_train.shape[0]),
    })
    rows.append({
        "item": "X_test_records",
        "value": int(X_test.shape[0]),
    })
    rows.append({
        "item": "n_features",
        "value": int(len(feature_cols)),
    })
    rows.append({
        "item": "train_positive",
        "value": int((y_train == 1).sum()),
    })
    rows.append({
        "item": "train_negative",
        "value": int((y_train == 0).sum()),
    })
    rows.append({
        "item": "test_positive",
        "value": int((y_test == 1).sum()),
    })
    rows.append({
        "item": "test_negative",
        "value": int((y_test == 0).sum()),
    })
    rows.append({
        "item": "patient_overlap_train_test",
        "value": int(overlap_info["overlap_patients"]),
    })

    for family, cols in feature_groups.items():
        rows.append({
            "item": f"features_{family}",
            "value": int(len(cols)),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Save Outputs
# =============================================================================

def save_json_outputs(
    summary_df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    preprocessing_params: Dict[str, Dict[str, float]],
    overlap_info: Dict[str, object],
) -> None:
    with open(OUTPUT_FEATURE_GROUPS, "w", encoding="utf-8") as f:
        json.dump(feature_groups, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_PREPROCESSING_PARAMS, "w", encoding="utf-8") as f:
        json.dump({
            "generated": str(datetime.now()),
            "imputation_and_scaling": preprocessing_params,
            "train_test_patient_overlap": overlap_info,
            "notes": [
                "Imputation medians are fitted on training data only.",
                "Scaling means and standard deviations are fitted on training data only.",
                "Label proxies including BI-RADS-derived numeric fields are excluded from X.",
                "No real longitudinal features are included.",
            ],
        }, f, indent=4, ensure_ascii=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated": str(datetime.now()),
            "project_root": str(PROJECT_ROOT),
            "input_features": str(INPUT_FEATURES),
            "summary": summary_df.to_dict(orient="records"),
            "feature_groups": {k: len(v) for k, v in feature_groups.items()},
            "train_test_patient_overlap": overlap_info,
        }, f, indent=4, ensure_ascii=False)


def write_report(
    summary_df: pd.DataFrame,
    leakage_df: pd.DataFrame,
    class_balance_df: pd.DataFrame,
    feature_group_df: pd.DataFrame,
    overlap_info: Dict[str, object],
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2B SACU MODELING MATRIX REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input features: {INPUT_FEATURES}")
    lines.append("")

    lines.append("METHOD CONSISTENCY")
    lines.append("-" * 100)
    lines.append("Used VinDr complete four-view labeled feature cohort only.")
    lines.append("Preserved official VinDr training/test split.")
    lines.append("Excluded labels, BI-RADS-derived label proxies, IDs, paths, split fields, and text metadata from X.")
    lines.append("Fitted imputation and scaling parameters on training data only.")
    lines.append("No real longitudinal variables were introduced.")
    lines.append("Feature groups are preserved for SACU local, multi-view, bilateral, temporal-spatial, metadata, and adaptive-control branches.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("CLASS BALANCE")
    lines.append("-" * 100)
    lines.append(class_balance_df.to_string(index=False))
    lines.append("")

    lines.append("FEATURE GROUP SUMMARY")
    lines.append("-" * 100)
    lines.append(feature_group_df.to_string(index=False))
    lines.append("")

    lines.append("TRAIN/TEST PATIENT OVERLAP")
    lines.append("-" * 100)
    for k, v in overlap_info.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("LEAKAGE AUDIT SUMMARY")
    lines.append("-" * 100)
    dist = (
        leakage_df.groupby(["accepted_as_feature", "reason"])
        .size()
        .reset_index(name="columns")
        .sort_values(["accepted_as_feature", "columns"], ascending=[False, False])
    )
    lines.append(dist.to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_X_TRAIN,
        OUTPUT_Y_TRAIN,
        OUTPUT_X_TEST,
        OUTPUT_Y_TEST,
        OUTPUT_META_TRAIN,
        OUTPUT_META_TEST,
        OUTPUT_FEATURE_COLUMNS,
        OUTPUT_FEATURE_GROUPS,
        OUTPUT_PREPROCESSING_PARAMS,
        OUTPUT_SUMMARY,
        OUTPUT_LEAKAGE_AUDIT,
        OUTPUT_CLASS_BALANCE,
        OUTPUT_FEATURE_GROUP_SUMMARY,
        OUTPUT_JSON,
        OUTPUT_REPORT,
    ]:
        lines.append(str(p))

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    np.random.seed(RANDOM_SEED)

    print("=" * 100)
    print("STAGE 2B BUILD SACU MODELING MATRICES")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input features: {INPUT_FEATURES}")
    print("-" * 100)

    print("Loading features...")
    df = read_csv_required(INPUT_FEATURES)

    print("Normalizing split column...")
    if SPLIT_COLUMN not in df.columns:
        if "split" not in df.columns:
            raise ValueError("Neither split nor normalized_split found in features.")
        df[SPLIT_COLUMN] = df["split"].map(normalize_split)
    else:
        df[SPLIT_COLUMN] = df[SPLIT_COLUMN].map(normalize_split)

    print("Running leakage audit and selecting features...")
    leakage_df, feature_cols = build_leakage_audit(df)

    if not feature_cols:
        raise ValueError("No valid feature columns selected.")

    feature_groups = build_feature_groups(feature_cols)

    print("Splitting train/test...")
    train_df, test_df = split_train_test(df)

    y_train = build_label_vector(train_df)
    y_test = build_label_vector(test_df)

    print("Fitting train-only imputation/scaling...")
    X_train, X_test, preprocessing_params = impute_and_scale_train_test(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    print("Building metadata tables...")
    train_meta = build_metadata(train_df)
    test_meta = build_metadata(test_df)

    overlap_info = check_patient_overlap(train_meta, test_meta)

    print("Building summaries...")
    class_balance_df = build_class_balance(y_train, y_test)
    feature_group_df = build_feature_group_summary(feature_groups)

    summary_df = build_summary(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_cols=feature_cols,
        feature_groups=feature_groups,
        overlap_info=overlap_info,
    )

    feature_columns_df = pd.DataFrame({
        "feature_name": feature_cols,
        "feature_family": [get_feature_family(c) for c in feature_cols],
    })

    print("Saving matrices and audit files...")
    X_train.to_csv(OUTPUT_X_TRAIN, index=False, encoding="utf-8-sig")
    y_train.to_frame("target").to_csv(OUTPUT_Y_TRAIN, index=False, encoding="utf-8-sig")
    X_test.to_csv(OUTPUT_X_TEST, index=False, encoding="utf-8-sig")
    y_test.to_frame("target").to_csv(OUTPUT_Y_TEST, index=False, encoding="utf-8-sig")

    train_meta.to_csv(OUTPUT_META_TRAIN, index=False, encoding="utf-8-sig")
    test_meta.to_csv(OUTPUT_META_TEST, index=False, encoding="utf-8-sig")

    feature_columns_df.to_csv(OUTPUT_FEATURE_COLUMNS, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")
    leakage_df.to_csv(OUTPUT_LEAKAGE_AUDIT, index=False, encoding="utf-8-sig")
    class_balance_df.to_csv(OUTPUT_CLASS_BALANCE, index=False, encoding="utf-8-sig")
    feature_group_df.to_csv(OUTPUT_FEATURE_GROUP_SUMMARY, index=False, encoding="utf-8-sig")

    save_json_outputs(
        summary_df=summary_df,
        feature_groups=feature_groups,
        preprocessing_params=preprocessing_params,
        overlap_info=overlap_info,
    )

    write_report(
        summary_df=summary_df,
        leakage_df=leakage_df,
        class_balance_df=class_balance_df,
        feature_group_df=feature_group_df,
        overlap_info=overlap_info,
    )

    print()
    print("STAGE 2B COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"X_train:                 {OUTPUT_X_TRAIN}")
    print(f"y_train:                 {OUTPUT_Y_TRAIN}")
    print(f"X_test:                  {OUTPUT_X_TEST}")
    print(f"y_test:                  {OUTPUT_Y_TEST}")
    print(f"Train metadata:          {OUTPUT_META_TRAIN}")
    print(f"Test metadata:           {OUTPUT_META_TEST}")
    print(f"Feature columns:         {OUTPUT_FEATURE_COLUMNS}")
    print(f"Feature groups:          {OUTPUT_FEATURE_GROUPS}")
    print(f"Preprocessing params:    {OUTPUT_PREPROCESSING_PARAMS}")
    print(f"Summary:                 {OUTPUT_SUMMARY}")
    print(f"Leakage audit:           {OUTPUT_LEAKAGE_AUDIT}")
    print(f"Class balance:           {OUTPUT_CLASS_BALANCE}")
    print(f"Feature group summary:   {OUTPUT_FEATURE_GROUP_SUMMARY}")
    print(f"JSON summary:            {OUTPUT_JSON}")
    print(f"Text report:             {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()