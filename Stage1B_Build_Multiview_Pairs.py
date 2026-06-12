r"""
Stage1B_Build_Multiview_Pairs.py

Purpose
-------
Build structured multi-view mammography records from the Stage 1A processed image index.

This script:
1. Loads Stage1A_Preprocessed_Image_Index.csv.
2. Re-parses patient ID, laterality, view, split, and lesion type from CBIS-DDSM case_folder.
3. Builds CC/MLO pairs per dataset, patient, breast laterality, and split.
4. Creates complete and partial multi-view records.
5. Preserves all processed paths needed by the dataset loader.
6. Saves:
   - multiview_exam_records.csv for each dataset
   - global multiview records
   - summary tables
   - report text

Important
---------
For CBIS-DDSM, patient_id is re-derived from case_folder, not from DICOM PatientID,
because DICOM PatientID may be inconsistent with the case-level CBIS naming.

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1B_Build_Multiview_Pairs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

INPUT_INDEX_CSV = PROJECT_ROOT / "results" / "tables" / "Stage1A_Preprocessed_Image_Index.csv"

OUTPUT_ROOT = PROJECT_ROOT / "preprocessing" / "multiview_pairs"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GLOBAL_CSV = OUTPUT_TABLE_DIR / "Stage1B_Global_Multiview_Exam_Records.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "Stage1B_Multiview_Pairing_Summary.csv"
OUTPUT_DUPLICATES_CSV = OUTPUT_TABLE_DIR / "Stage1B_Potential_Duplicate_View_Records.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1B_Multiview_Pairing_Summary.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1B_Multiview_Pairing_Report.txt"

DATASETS = ["CBIS-DDSM", "INbreast", "VinDr-Mammo"]


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


def extract_patient_from_case(case_folder: str, fallback: str = "") -> str:
    case_folder = safe_str(case_folder)
    m = re.search(r"(P_\d+)", case_folder, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    fallback = safe_str(fallback)
    if fallback:
        return fallback

    return "UnknownPatient"


def extract_laterality_from_case(case_folder: str, fallback: str = "") -> str:
    text = safe_str(case_folder).upper()

    if "LEFT" in text:
        return "LEFT"
    if "RIGHT" in text:
        return "RIGHT"

    fallback = safe_str(fallback).upper()
    if fallback in {"LEFT", "RIGHT"}:
        return fallback
    if fallback == "L":
        return "LEFT"
    if fallback == "R":
        return "RIGHT"

    return ""


def extract_view_from_case(case_folder: str, fallback: str = "") -> str:
    text = safe_str(case_folder).upper()

    if "_CC" in text or text.endswith("CC"):
        return "CC"
    if "_MLO" in text or text.endswith("MLO"):
        return "MLO"

    fallback = safe_str(fallback).upper()
    if fallback in {"CC", "MLO"}:
        return fallback

    return ""


def extract_split_from_case(case_folder: str, fallback: str = "") -> str:
    text = safe_str(case_folder).upper()

    if "TRAINING" in text:
        return "Training"
    if "TEST" in text:
        return "Test"

    fallback = safe_str(fallback)
    if fallback:
        return fallback

    return "UnknownSplit"


def extract_lesion_type_from_case(case_folder: str, fallback: str = "") -> str:
    text = safe_str(case_folder).upper()

    if text.startswith("CALC"):
        return "Calc"
    if text.startswith("MASS"):
        return "Mass"

    fallback = safe_str(fallback)
    if fallback:
        return fallback

    return "UnknownLesion"


def choose_primary_view_record(view_df: pd.DataFrame) -> pd.Series:
    """
    Select one representative image for a view.

    Preference:
    1. Largest crop area
    2. Largest original image size
    3. First row
    """
    temp = view_df.copy()

    if "crop_w" in temp.columns and "crop_h" in temp.columns:
        temp["crop_area"] = pd.to_numeric(temp["crop_w"], errors="coerce").fillna(0) * \
                            pd.to_numeric(temp["crop_h"], errors="coerce").fillna(0)
    else:
        temp["crop_area"] = 0

    if "original_size_bytes" not in temp.columns:
        temp["original_size_bytes"] = 0

    temp["original_size_bytes"] = pd.to_numeric(
        temp["original_size_bytes"],
        errors="coerce"
    ).fillna(0)

    temp = temp.sort_values(
        ["crop_area", "original_size_bytes"],
        ascending=[False, False]
    )

    return temp.iloc[0]


def normalize_stage1a_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    required = ["dataset", "processed_path", "case_folder"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Stage 1A index is missing required columns: {missing}")

    for col in [
        "patient_id", "laterality", "view", "split", "lesion_type",
        "case_folder", "processed_path", "source_path"
    ]:
        if col not in df.columns:
            df[col] = ""

    normalized_rows = []

    for row in df.itertuples(index=False):
        row_dict = row._asdict()

        dataset = safe_str(row_dict.get("dataset", ""))
        case_folder = safe_str(row_dict.get("case_folder", ""))

        patient_id = safe_str(row_dict.get("patient_id", ""))
        laterality = safe_str(row_dict.get("laterality", ""))
        view = safe_str(row_dict.get("view", ""))
        split = safe_str(row_dict.get("split", ""))
        lesion_type = safe_str(row_dict.get("lesion_type", ""))

        if dataset == "CBIS-DDSM":
            patient_id = extract_patient_from_case(case_folder, fallback=patient_id)
            laterality = extract_laterality_from_case(case_folder, fallback=laterality)
            view = extract_view_from_case(case_folder, fallback=view)
            split = extract_split_from_case(case_folder, fallback=split)
            lesion_type = extract_lesion_type_from_case(case_folder, fallback=lesion_type)

        else:
            patient_id = patient_id if patient_id else extract_patient_from_case(case_folder)
            laterality = extract_laterality_from_case(case_folder, fallback=laterality)
            view = extract_view_from_case(case_folder, fallback=view)
            split = split if split else "UnknownSplit"
            lesion_type = lesion_type if lesion_type else "UnknownLesion"

        row_dict["patient_id_clean"] = patient_id
        row_dict["laterality_clean"] = laterality
        row_dict["view_clean"] = view
        row_dict["split_clean"] = split
        row_dict["lesion_type_clean"] = lesion_type

        normalized_rows.append(row_dict)

    out = pd.DataFrame(normalized_rows)

    # Keep only rows with processed images.
    out["processed_path"] = out["processed_path"].fillna("").astype(str)
    out = out[out["processed_path"].str.len() > 0].copy()

    return out


# =============================================================================
# Build Multi-View Records
# =============================================================================

def build_multiview_records_for_dataset(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    d = df[df["dataset"] == dataset_name].copy()

    if d.empty:
        return pd.DataFrame()

    # Multi-view requires known patient, laterality, and at least one known view.
    d = d[
        d["patient_id_clean"].notna()
        & d["laterality_clean"].notna()
        & d["view_clean"].isin(["CC", "MLO"])
    ].copy()

    d = d[d["laterality_clean"].isin(["LEFT", "RIGHT"])].copy()

    if d.empty:
        return pd.DataFrame()

    group_cols = [
        "dataset",
        "patient_id_clean",
        "split_clean",
        "laterality_clean",
        "lesion_type_clean",
    ]

    records: List[Dict[str, object]] = []

    for key, g in d.groupby(group_cols, dropna=False):
        dataset, patient_id, split, laterality, lesion_type = key

        cc_df = g[g["view_clean"] == "CC"].copy()
        mlo_df = g[g["view_clean"] == "MLO"].copy()

        has_cc = not cc_df.empty
        has_mlo = not mlo_df.empty

        cc_record = choose_primary_view_record(cc_df) if has_cc else None
        mlo_record = choose_primary_view_record(mlo_df) if has_mlo else None

        record_id = f"{dataset}__{patient_id}__{split}__{laterality}__{lesion_type}"

        records.append({
            "record_id": record_id,
            "dataset": dataset,
            "patient_id": patient_id,
            "split": split,
            "laterality": laterality,
            "lesion_type": lesion_type,

            "has_cc": bool(has_cc),
            "has_mlo": bool(has_mlo),
            "complete_multiview": bool(has_cc and has_mlo),

            "cc_processed_path": safe_str(cc_record["processed_path"]) if has_cc else "",
            "mlo_processed_path": safe_str(mlo_record["processed_path"]) if has_mlo else "",

            "cc_source_path": safe_str(cc_record.get("source_path", "")) if has_cc else "",
            "mlo_source_path": safe_str(mlo_record.get("source_path", "")) if has_mlo else "",

            "cc_case_folder": safe_str(cc_record.get("case_folder", "")) if has_cc else "",
            "mlo_case_folder": safe_str(mlo_record.get("case_folder", "")) if has_mlo else "",

            "cc_study_instance_uid": safe_str(cc_record.get("study_instance_uid", "")) if has_cc else "",
            "mlo_study_instance_uid": safe_str(mlo_record.get("study_instance_uid", "")) if has_mlo else "",

            "cc_series_instance_uid": safe_str(cc_record.get("series_instance_uid", "")) if has_cc else "",
            "mlo_series_instance_uid": safe_str(mlo_record.get("series_instance_uid", "")) if has_mlo else "",

            "cc_sop_instance_uid": safe_str(cc_record.get("sop_instance_uid", "")) if has_cc else "",
            "mlo_sop_instance_uid": safe_str(mlo_record.get("sop_instance_uid", "")) if has_mlo else "",

            "cc_study_date": safe_str(cc_record.get("study_date", "")) if has_cc else "",
            "mlo_study_date": safe_str(mlo_record.get("study_date", "")) if has_mlo else "",

            "n_cc_candidates": int(len(cc_df)),
            "n_mlo_candidates": int(len(mlo_df)),
            "n_total_candidates": int(len(g)),
        })

    return pd.DataFrame(records)


def find_potential_duplicate_views(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "dataset",
        "patient_id_clean",
        "split_clean",
        "laterality_clean",
        "lesion_type_clean",
        "view_clean",
    ]

    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    d = df[
        df["view_clean"].isin(["CC", "MLO"])
        & df["laterality_clean"].isin(["LEFT", "RIGHT"])
    ].copy()

    group_cols = required_cols

    counts = (
        d.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_records")
    )

    dup_groups = counts[counts["n_records"] > 1].copy()

    if dup_groups.empty:
        return pd.DataFrame()

    merged = d.merge(dup_groups[group_cols], on=group_cols, how="inner")
    return merged.sort_values(group_cols)


# =============================================================================
# Summary and Reports
# =============================================================================

def build_summary(global_records: pd.DataFrame, normalized_index: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for dataset_name in DATASETS:
        rec = global_records[global_records["dataset"] == dataset_name].copy()
        idx = normalized_index[normalized_index["dataset"] == dataset_name].copy()

        rows.append({
            "dataset": dataset_name,
            "input_processed_images": int(len(idx)),
            "multiview_records": int(len(rec)),
            "complete_multiview_records": int(rec["complete_multiview"].sum()) if not rec.empty else 0,
            "partial_multiview_records": int((~rec["complete_multiview"]).sum()) if not rec.empty else 0,
            "unique_patients": int(rec["patient_id"].nunique()) if not rec.empty else 0,
            "left_records": int((rec["laterality"] == "LEFT").sum()) if not rec.empty else 0,
            "right_records": int((rec["laterality"] == "RIGHT").sum()) if not rec.empty else 0,
            "training_records": int((rec["split"] == "Training").sum()) if not rec.empty else 0,
            "test_records": int((rec["split"] == "Test").sum()) if not rec.empty else 0,
            "unknown_split_records": int((rec["split"] == "UnknownSplit").sum()) if not rec.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "input_processed_images": int(len(normalized_index)),
        "multiview_records": int(len(global_records)),
        "complete_multiview_records": int(global_records["complete_multiview"].sum()) if not global_records.empty else 0,
        "partial_multiview_records": int((~global_records["complete_multiview"]).sum()) if not global_records.empty else 0,
        "unique_patients": int(global_records["patient_id"].nunique()) if not global_records.empty else 0,
        "left_records": int((global_records["laterality"] == "LEFT").sum()) if not global_records.empty else 0,
        "right_records": int((global_records["laterality"] == "RIGHT").sum()) if not global_records.empty else 0,
        "training_records": int((global_records["split"] == "Training").sum()) if not global_records.empty else 0,
        "test_records": int((global_records["split"] == "Test").sum()) if not global_records.empty else 0,
        "unknown_split_records": int((global_records["split"] == "UnknownSplit").sum()) if not global_records.empty else 0,
    })

    return pd.DataFrame(rows)


def save_dataset_specific_outputs(global_records: pd.DataFrame) -> None:
    for dataset_name in DATASETS:
        dataset_dir = OUTPUT_ROOT / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        out_csv = dataset_dir / "multiview_exam_records.csv"

        dataset_records = global_records[global_records["dataset"] == dataset_name].copy()
        dataset_records.to_csv(out_csv, index=False, encoding="utf-8-sig")


def save_json_summary(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "input_index_csv": str(INPUT_INDEX_CSV),
        "output_root": str(OUTPUT_ROOT),
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    summary_df: pd.DataFrame,
    global_records: pd.DataFrame,
    duplicates_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("STAGE 1B BUILD MULTI-VIEW PAIRS REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input index: {INPUT_INDEX_CSV}")
    lines.append(f"Output root: {OUTPUT_ROOT}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    if not global_records.empty:
        lines.append("COMPLETE MULTI-VIEW RECORD DISTRIBUTION")
        lines.append("-" * 100)
        dist = (
            global_records.groupby(["dataset", "split", "lesion_type", "laterality"])["complete_multiview"]
            .agg(["count", "sum"])
            .reset_index()
            .rename(columns={"count": "records", "sum": "complete_records"})
        )
        lines.append(dist.to_string(index=False))
        lines.append("")

    lines.append("POTENTIAL DUPLICATE VIEW RECORDS")
    lines.append("-" * 100)
    if duplicates_df.empty:
        lines.append("No potential duplicate view records detected.")
    else:
        lines.append(f"Rows involved in duplicate view groups: {len(duplicates_df):,}")
        lines.append(duplicates_df.head(30).to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_GLOBAL_CSV))
    lines.append(str(OUTPUT_SUMMARY_CSV))
    lines.append(str(OUTPUT_DUPLICATES_CSV))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))
    for dataset_name in DATASETS:
        lines.append(str(OUTPUT_ROOT / dataset_name / "multiview_exam_records.csv"))
    lines.append("")

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1B BUILD MULTI-VIEW PAIRS")
    print("=" * 100)
    print(f"Input index: {INPUT_INDEX_CSV}")
    print(f"Output root: {OUTPUT_ROOT}")
    print("-" * 100)

    if not INPUT_INDEX_CSV.exists():
        raise FileNotFoundError(f"Input index not found: {INPUT_INDEX_CSV}")

    index_df = pd.read_csv(INPUT_INDEX_CSV)
    normalized_index = normalize_stage1a_index(index_df)

    all_records: List[pd.DataFrame] = []

    for dataset_name in DATASETS:
        print(f"Building multi-view records for: {dataset_name}")
        dataset_records = build_multiview_records_for_dataset(normalized_index, dataset_name)
        all_records.append(dataset_records)

    if all_records:
        global_records = pd.concat(all_records, ignore_index=True)
    else:
        global_records = pd.DataFrame()

    duplicates_df = find_potential_duplicate_views(normalized_index)
    summary_df = build_summary(global_records, normalized_index)

    save_dataset_specific_outputs(global_records)

    global_records.to_csv(OUTPUT_GLOBAL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    duplicates_df.to_csv(OUTPUT_DUPLICATES_CSV, index=False, encoding="utf-8-sig")

    save_json_summary(summary_df)
    write_report(summary_df, global_records, duplicates_df)

    print()
    print("STAGE 1B COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Global records:      {OUTPUT_GLOBAL_CSV}")
    print(f"Summary:             {OUTPUT_SUMMARY_CSV}")
    print(f"Duplicates:          {OUTPUT_DUPLICATES_CSV}")
    print(f"JSON summary:        {OUTPUT_JSON}")
    print(f"Text report:         {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()