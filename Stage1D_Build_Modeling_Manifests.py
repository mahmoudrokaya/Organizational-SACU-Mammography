r"""
Stage1D_Build_Modeling_Manifests.py

Purpose
-------
Build clean modeling manifests after Stage1C.

This stage creates dataset- and task-specific CSV manifests for downstream
modeling while preventing leakage and over-claiming.

It creates manifests for:
1. Complete four-view modeling
2. Available-view modeling
3. Same-exam temporal-spatial modeling
4. Labeled VinDr-Mammo modeling
5. External INbreast validation subset
6. CBIS-DDSM available-view subset
7. Complete four-view labeled VinDr train/test split
8. Complete four-view external-style evaluation manifest

Key design principles
---------------------
- Do not treat same-exam multi-view ordering as real longitudinal follow-up.
- Do not include incomplete four-view records in complete-four-view manifests.
- Do not include unlabeled datasets in supervised labeled manifests.
- Use patient/study/exam-level records only.
- Preserve original split labels when available.
- Save transparent inclusion/exclusion reasons for reproducibility.

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Temporal_Modeling_Cohort.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Same_Exam_Multiview_Sequences.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Recommended_Temporal_Use.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B_v2_Global_Multiview_Exam_Records.csv

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\manifests\\Stage1D_*.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1D_Modeling_Manifest_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage1D_Modeling_Manifest_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1D_Build_Modeling_Manifests.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"
MANIFEST_DIR = PROJECT_ROOT / "manifests"

MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_MODELING_COHORT = RESULTS_TABLE_DIR / "Stage1C_Temporal_Modeling_Cohort.csv"
INPUT_SAME_EXAM_SEQUENCES = RESULTS_TABLE_DIR / "Stage1C_Same_Exam_Multiview_Sequences.csv"
INPUT_RECOMMENDED_TEMPORAL_USE = RESULTS_TABLE_DIR / "Stage1C_Recommended_Temporal_Use.csv"
INPUT_EXAM_RECORDS = RESULTS_TABLE_DIR / "Stage1B_v2_Global_Multiview_Exam_Records.csv"

OUTPUT_COMPLETE_FOUR_VIEW = MANIFEST_DIR / "Stage1D_Complete_Four_View_Modeling_Manifest.csv"
OUTPUT_AVAILABLE_VIEW = MANIFEST_DIR / "Stage1D_Available_View_Modeling_Manifest.csv"
OUTPUT_TEMPORAL_SPATIAL = MANIFEST_DIR / "Stage1D_Same_Exam_Temporal_Spatial_Manifest.csv"
OUTPUT_VINDR_LABELED = MANIFEST_DIR / "Stage1D_VinDr_Labeled_Modeling_Manifest.csv"
OUTPUT_VINDR_COMPLETE_LABELED = MANIFEST_DIR / "Stage1D_VinDr_Complete_Four_View_Labeled_Manifest.csv"
OUTPUT_INBREAST_EXTERNAL = MANIFEST_DIR / "Stage1D_INbreast_External_Validation_Manifest.csv"
OUTPUT_CBIS_AVAILABLE = MANIFEST_DIR / "Stage1D_CBIS_Available_View_Manifest.csv"
OUTPUT_EXTERNAL_STYLE_COMPLETE = MANIFEST_DIR / "Stage1D_External_Style_Complete_View_Manifest.csv"
OUTPUT_EXCLUSIONS = MANIFEST_DIR / "Stage1D_Manifest_Exclusions.csv"

OUTPUT_SUMMARY = RESULTS_TABLE_DIR / "Stage1D_Modeling_Manifest_Summary.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage1D_Modeling_Manifest_Summary.json"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage1D_Modeling_Manifest_Report.txt"


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


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value

    v = safe_str(value).lower()

    return v in {"true", "1", "yes", "y"}


def is_labeled(value) -> bool:
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass

    text = safe_str(value)

    if text == "":
        return False

    if text.lower() in {"nan", "none", "unknown"}:
        return False

    return True


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def normalize_split(value: str) -> str:
    v = safe_str(value).lower()

    if v in {"train", "training"}:
        return "training"

    if v in {"test", "testing"}:
        return "test"

    if v in {"val", "valid", "validation"}:
        return "validation"

    if v == "external":
        return "external"

    if v == "":
        return "unknown"

    return v


def ensure_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    d = df.copy()

    for col in columns:
        if col not in d.columns:
            d[col] = ""

    return d


def make_standard_manifest_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a consistent subset/order of columns for all record-level manifests.
    """
    standard_cols = [
        "record_id",
        "dataset",
        "patient_id",
        "study_id",
        "exam_id",
        "split",
        "normalized_split",
        "exam_date",
        "modeling_role",
        "task_name",
        "task_family",
        "manifest_name",
        "inclusion_reason",
        "exclusion_reason",
        "complete_four_view",
        "excluded_from_complete_four_view",
        "same_exam_multiview_available",
        "has_real_longitudinal_history",
        "temporal_input_type",
        "available_views",
        "missing_views",
        "exam_label",
        "exam_label_text",
        "breast_density",
        "lcc_processed_path",
        "lmlo_processed_path",
        "rcc_processed_path",
        "rmlo_processed_path",
        "lcc_image_id",
        "lmlo_image_id",
        "rcc_image_id",
        "rmlo_image_id",
        "lcc_birads",
        "lmlo_birads",
        "rcc_birads",
        "rmlo_birads",
    ]

    d = ensure_columns(df, standard_cols)

    return d[standard_cols].copy()


def add_task_columns(
    df: pd.DataFrame,
    manifest_name: str,
    task_name: str,
    task_family: str,
    inclusion_reason: str,
    exclusion_reason: str = "",
) -> pd.DataFrame:
    d = df.copy()

    d["manifest_name"] = manifest_name
    d["task_name"] = task_name
    d["task_family"] = task_family
    d["inclusion_reason"] = inclusion_reason
    d["exclusion_reason"] = exclusion_reason

    return d


# =============================================================================
# Load and Merge Inputs
# =============================================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    modeling_df = read_csv_required(INPUT_MODELING_COHORT)
    sequence_df = read_csv_required(INPUT_SAME_EXAM_SEQUENCES)
    recommended_df = read_csv_required(INPUT_RECOMMENDED_TEMPORAL_USE)
    exam_df = read_csv_required(INPUT_EXAM_RECORDS)

    return modeling_df, sequence_df, recommended_df, exam_df


def merge_modeling_with_exam_records(modeling_df: pd.DataFrame, exam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Stage1C modeling cohort contains task/temporal role columns.
    Stage1B exam records contain image paths and view-level details.
    Merge them by record_id.
    """
    m = modeling_df.copy()
    e = exam_df.copy()

    m["record_id"] = m["record_id"].astype(str)
    e["record_id"] = e["record_id"].astype(str)

    keep_exam_cols = [
        "record_id",
        "lcc_processed_path",
        "lmlo_processed_path",
        "rcc_processed_path",
        "rmlo_processed_path",
        "lcc_source_path",
        "lmlo_source_path",
        "rcc_source_path",
        "rmlo_source_path",
        "lcc_image_id",
        "lmlo_image_id",
        "rcc_image_id",
        "rmlo_image_id",
        "lcc_birads",
        "lmlo_birads",
        "rcc_birads",
        "rmlo_birads",
        "lcc_label",
        "lmlo_label",
        "rcc_label",
        "rmlo_label",
        "exam_birads_summary",
        "exam_finding_categories",
        "exam_finding_birads",
        "exam_n_findings",
        "exam_has_finding_annotation",
        "exam_finding_boxes_json",
    ]

    e = ensure_columns(e, keep_exam_cols)

    merged = m.merge(
        e[keep_exam_cols],
        on="record_id",
        how="left",
        suffixes=("", "_exam")
    )

    merged["normalized_split"] = merged["split"].map(normalize_split)

    return merged


# =============================================================================
# Build Manifests
# =============================================================================

def build_complete_four_view_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["complete_four_view"].map(bool_value))
        & (~df["excluded_from_complete_four_view"].map(bool_value))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_Complete_Four_View_Modeling_Manifest",
        task_name="complete_four_view_modeling",
        task_family="complete_multiview",
        inclusion_reason="Record has all four standard views and is not excluded by Stage1B1 audit.",
    )

    return make_standard_manifest_columns(d)


def build_available_view_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        df["same_exam_multiview_available"].map(bool_value)
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_Available_View_Modeling_Manifest",
        task_name="available_view_modeling",
        task_family="available_multiview",
        inclusion_reason="Record has at least one usable same-exam multi-view configuration.",
    )

    return make_standard_manifest_columns(d)


def build_vindr_labeled_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["dataset"] == "VinDr-Mammo")
        & (df["exam_label"].apply(is_labeled))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_VinDr_Labeled_Modeling_Manifest",
        task_name="vindr_labeled_modeling",
        task_family="supervised_labeled",
        inclusion_reason="VinDr-Mammo record has official breast-level BI-RADS-derived label.",
    )

    return make_standard_manifest_columns(d)


def build_vindr_complete_labeled_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["dataset"] == "VinDr-Mammo")
        & (df["exam_label"].apply(is_labeled))
        & (df["complete_four_view"].map(bool_value))
        & (~df["excluded_from_complete_four_view"].map(bool_value))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_VinDr_Complete_Four_View_Labeled_Manifest",
        task_name="vindr_complete_four_view_labeled_modeling",
        task_family="supervised_complete_multiview",
        inclusion_reason="VinDr-Mammo record has official label and complete four-view availability.",
    )

    return make_standard_manifest_columns(d)


def build_inbreast_external_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["dataset"] == "INbreast")
        & (df["same_exam_multiview_available"].map(bool_value))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_INbreast_External_Validation_Manifest",
        task_name="inbreast_external_available_view_validation",
        task_family="external_validation",
        inclusion_reason="INbreast record has usable external available-view multi-view data.",
    )

    return make_standard_manifest_columns(d)


def build_cbis_available_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["dataset"] == "CBIS-DDSM")
        & (df["same_exam_multiview_available"].map(bool_value))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_CBIS_Available_View_Manifest",
        task_name="cbis_available_view_analysis",
        task_family="lesion_centered_available_view",
        inclusion_reason="CBIS-DDSM record has usable available-view data; supervised labels require later CBIS label merge.",
    )

    return make_standard_manifest_columns(d)


def build_external_style_complete_manifest(df: pd.DataFrame) -> pd.DataFrame:
    d = df[
        (df["complete_four_view"].map(bool_value))
        & (~df["excluded_from_complete_four_view"].map(bool_value))
        & (df["dataset"].isin(["INbreast", "VinDr-Mammo"]))
    ].copy()

    d = add_task_columns(
        d,
        manifest_name="Stage1D_External_Style_Complete_View_Manifest",
        task_name="external_style_complete_view_evaluation",
        task_family="external_complete_multiview",
        inclusion_reason="External-style dataset record with complete four-view availability.",
    )

    return make_standard_manifest_columns(d)


def build_temporal_spatial_manifest(sequence_df: pd.DataFrame) -> pd.DataFrame:
    d = sequence_df.copy()

    d["normalized_split"] = d["split"].map(normalize_split)

    d["manifest_name"] = "Stage1D_Same_Exam_Temporal_Spatial_Manifest"
    d["task_name"] = "same_exam_temporal_spatial_modeling"
    d["task_family"] = "same_exam_temporal_spatial"
    d["inclusion_reason"] = (
        "Same-exam multi-view sequence; this is temporal-spatial ordering, not real longitudinal follow-up."
    )
    d["exclusion_reason"] = ""

    standard_cols = [
        "sequence_id",
        "dataset",
        "patient_id",
        "study_id",
        "exam_id",
        "split",
        "normalized_split",
        "sequence_type",
        "task_name",
        "task_family",
        "manifest_name",
        "inclusion_reason",
        "exclusion_reason",
        "is_real_longitudinal",
        "is_artificial_temporal_sequence",
        "n_available_views",
        "available_views",
        "missing_views",
        "completeness",
        "complete_four_view",
        "complete_unilateral_multiview",
        "complete_bilateral",
        "exam_label",
        "exam_label_text",
        "breast_density",
        "exam_birads_summary",
        "exam_n_findings",
        "excluded_from_complete_four_view",
        "temporal_note",
        "sequence_items_json",
    ]

    d = ensure_columns(d, standard_cols)

    return d[standard_cols].copy()


def build_exclusion_manifest(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for row in df.itertuples(index=False):
        r = row._asdict()

        record_id = safe_str(r.get("record_id", ""))
        dataset = safe_str(r.get("dataset", ""))

        complete_four = bool_value(r.get("complete_four_view", False))
        excluded = bool_value(r.get("excluded_from_complete_four_view", False))
        labeled = is_labeled(r.get("exam_label", None))
        available = bool_value(r.get("same_exam_multiview_available", False))

        if not complete_four or excluded:
            rows.append({
                "record_id": record_id,
                "dataset": dataset,
                "patient_id": safe_str(r.get("patient_id", "")),
                "exam_id": safe_str(r.get("exam_id", "")),
                "split": safe_str(r.get("split", "")),
                "excluded_from_manifest": "complete_four_view_manifest",
                "reason": safe_str(r.get("missing_views", "")) or "Incomplete or audit-excluded complete-four-view record.",
            })

        if dataset != "VinDr-Mammo" or not labeled:
            rows.append({
                "record_id": record_id,
                "dataset": dataset,
                "patient_id": safe_str(r.get("patient_id", "")),
                "exam_id": safe_str(r.get("exam_id", "")),
                "split": safe_str(r.get("split", "")),
                "excluded_from_manifest": "vindr_labeled_manifest",
                "reason": "Record is not a labeled VinDr-Mammo breast-level record.",
            })

        if not available:
            rows.append({
                "record_id": record_id,
                "dataset": dataset,
                "patient_id": safe_str(r.get("patient_id", "")),
                "exam_id": safe_str(r.get("exam_id", "")),
                "split": safe_str(r.get("split", "")),
                "excluded_from_manifest": "available_view_manifest",
                "reason": "Record lacks usable same-exam multi-view availability.",
            })

    return pd.DataFrame(rows)


# =============================================================================
# Summary and Report
# =============================================================================

def manifest_stats(name: str, df: pd.DataFrame) -> Dict[str, object]:
    if df.empty:
        return {
            "manifest": name,
            "records": 0,
            "datasets": "",
            "training": 0,
            "test": 0,
            "validation": 0,
            "external": 0,
            "unknown_split": 0,
            "labeled_records": 0,
            "positive_labels": 0,
            "negative_labels": 0,
            "complete_four_view_records": 0,
        }

    split_col = "normalized_split" if "normalized_split" in df.columns else "split"

    label_col = "exam_label" if "exam_label" in df.columns else None

    return {
        "manifest": name,
        "records": int(len(df)),
        "datasets": "|".join(sorted(set(df["dataset"].astype(str).tolist()))) if "dataset" in df.columns else "",
        "training": int((df[split_col] == "training").sum()) if split_col in df.columns else 0,
        "test": int((df[split_col] == "test").sum()) if split_col in df.columns else 0,
        "validation": int((df[split_col] == "validation").sum()) if split_col in df.columns else 0,
        "external": int((df[split_col] == "external").sum()) if split_col in df.columns else 0,
        "unknown_split": int((df[split_col] == "unknown").sum()) if split_col in df.columns else 0,
        "labeled_records": int(df[label_col].apply(is_labeled).sum()) if label_col else 0,
        "positive_labels": int((pd.to_numeric(df[label_col], errors="coerce") == 1).sum()) if label_col else 0,
        "negative_labels": int((pd.to_numeric(df[label_col], errors="coerce") == 0).sum()) if label_col else 0,
        "complete_four_view_records": int(df["complete_four_view"].map(bool_value).sum()) if "complete_four_view" in df.columns else 0,
    }


def build_summary(manifests: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []

    for name, df in manifests.items():
        rows.append(manifest_stats(name, df))

    return pd.DataFrame(rows)


def save_json(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "manifest_dir": str(MANIFEST_DIR),
        "inputs": {
            "modeling_cohort": str(INPUT_MODELING_COHORT),
            "same_exam_sequences": str(INPUT_SAME_EXAM_SEQUENCES),
            "recommended_temporal_use": str(INPUT_RECOMMENDED_TEMPORAL_USE),
            "exam_records": str(INPUT_EXAM_RECORDS),
        },
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(summary_df: pd.DataFrame, manifests: Dict[str, pd.DataFrame]) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1D MODELING MANIFEST REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Manifest directory: {MANIFEST_DIR}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("MANIFEST DETAILS")
    lines.append("-" * 100)

    for name, df in manifests.items():
        lines.append("")
        lines.append(name)
        lines.append("-" * 60)
        lines.append(f"Records: {len(df):,}")
        if not df.empty and "dataset" in df.columns:
            dist = df["dataset"].value_counts().reset_index()
            dist.columns = ["dataset", "records"]
            lines.append(dist.to_string(index=False))
        else:
            lines.append("No records.")

    lines.append("")
    lines.append("TEMPORAL CLAIM CONTROL")
    lines.append("-" * 100)
    lines.append(
        "The same-exam temporal-spatial manifest contains concurrently acquired multi-view "
        "mammography records only. It must not be described as real longitudinal follow-up."
    )
    lines.append(
        "No supervised labeled manifest includes unlabeled CBIS-DDSM or INbreast records at this stage."
    )

    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append("-" * 100)

    for p in [
        OUTPUT_COMPLETE_FOUR_VIEW,
        OUTPUT_AVAILABLE_VIEW,
        OUTPUT_TEMPORAL_SPATIAL,
        OUTPUT_VINDR_LABELED,
        OUTPUT_VINDR_COMPLETE_LABELED,
        OUTPUT_INBREAST_EXTERNAL,
        OUTPUT_CBIS_AVAILABLE,
        OUTPUT_EXTERNAL_STYLE_COMPLETE,
        OUTPUT_EXCLUSIONS,
        OUTPUT_SUMMARY,
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
    print("=" * 100)
    print("STAGE 1D BUILD MODELING MANIFESTS")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Manifest directory: {MANIFEST_DIR}")
    print("-" * 100)

    print("Loading inputs...")
    modeling_df, sequence_df, recommended_df, exam_df = load_inputs()

    print("Merging modeling cohort with exam paths...")
    merged_df = merge_modeling_with_exam_records(modeling_df, exam_df)

    print("Building manifests...")
    complete_four_view_df = build_complete_four_view_manifest(merged_df)
    available_view_df = build_available_view_manifest(merged_df)
    temporal_spatial_df = build_temporal_spatial_manifest(sequence_df)
    vindr_labeled_df = build_vindr_labeled_manifest(merged_df)
    vindr_complete_labeled_df = build_vindr_complete_labeled_manifest(merged_df)
    inbreast_external_df = build_inbreast_external_manifest(merged_df)
    cbis_available_df = build_cbis_available_manifest(merged_df)
    external_style_complete_df = build_external_style_complete_manifest(merged_df)
    exclusions_df = build_exclusion_manifest(merged_df)

    manifests = {
        "complete_four_view": complete_four_view_df,
        "available_view": available_view_df,
        "same_exam_temporal_spatial": temporal_spatial_df,
        "vindr_labeled": vindr_labeled_df,
        "vindr_complete_labeled": vindr_complete_labeled_df,
        "inbreast_external": inbreast_external_df,
        "cbis_available": cbis_available_df,
        "external_style_complete": external_style_complete_df,
        "manifest_exclusions": exclusions_df,
    }

    print("Building summary...")
    summary_df = build_summary(manifests)

    print("Saving manifests...")
    complete_four_view_df.to_csv(OUTPUT_COMPLETE_FOUR_VIEW, index=False, encoding="utf-8-sig")
    available_view_df.to_csv(OUTPUT_AVAILABLE_VIEW, index=False, encoding="utf-8-sig")
    temporal_spatial_df.to_csv(OUTPUT_TEMPORAL_SPATIAL, index=False, encoding="utf-8-sig")
    vindr_labeled_df.to_csv(OUTPUT_VINDR_LABELED, index=False, encoding="utf-8-sig")
    vindr_complete_labeled_df.to_csv(OUTPUT_VINDR_COMPLETE_LABELED, index=False, encoding="utf-8-sig")
    inbreast_external_df.to_csv(OUTPUT_INBREAST_EXTERNAL, index=False, encoding="utf-8-sig")
    cbis_available_df.to_csv(OUTPUT_CBIS_AVAILABLE, index=False, encoding="utf-8-sig")
    external_style_complete_df.to_csv(OUTPUT_EXTERNAL_STYLE_COMPLETE, index=False, encoding="utf-8-sig")
    exclusions_df.to_csv(OUTPUT_EXCLUSIONS, index=False, encoding="utf-8-sig")

    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    save_json(summary_df)
    write_report(summary_df, manifests)

    print()
    print("STAGE 1D COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Complete four-view manifest:       {OUTPUT_COMPLETE_FOUR_VIEW}")
    print(f"Available-view manifest:           {OUTPUT_AVAILABLE_VIEW}")
    print(f"Temporal-spatial manifest:         {OUTPUT_TEMPORAL_SPATIAL}")
    print(f"VinDr labeled manifest:            {OUTPUT_VINDR_LABELED}")
    print(f"VinDr complete labeled manifest:   {OUTPUT_VINDR_COMPLETE_LABELED}")
    print(f"INbreast external manifest:        {OUTPUT_INBREAST_EXTERNAL}")
    print(f"CBIS available-view manifest:      {OUTPUT_CBIS_AVAILABLE}")
    print(f"External-style complete manifest:  {OUTPUT_EXTERNAL_STYLE_COMPLETE}")
    print(f"Exclusions manifest:               {OUTPUT_EXCLUSIONS}")
    print(f"Summary:                           {OUTPUT_SUMMARY}")
    print(f"JSON summary:                      {OUTPUT_JSON}")
    print(f"Text report:                       {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()