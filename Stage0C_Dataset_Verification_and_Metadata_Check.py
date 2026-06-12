r"""
Stage0C_Dataset_Verification_and_Metadata_Check.py

Purpose
-------
Verify the downloaded mammography datasets before preprocessing.

This script checks:
1. Dataset folder existence
2. File counts and sizes
3. DICOM availability
4. PNG availability
5. CSV metadata availability
6. CBIS-DDSM case-folder structure
7. Patient ID extraction from folder names
8. View extraction: CC / MLO
9. Laterality extraction: LEFT / RIGHT
10. Mass / Calcification split
11. Train / Test split
12. Potential multi-view availability
13. Potential bilateral availability
14. Potential duplicate patient-view records
15. Missing expected datasets
16. Readability of CSV metadata files

Input Root
----------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\datasets

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage0C_Dataset_Verification_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage0C_CBIS_DDSM_Case_Index.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage0C_Metadata_File_Check.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage0C_Multiview_Bilateral_Availability.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage0C_Dataset_Verification_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage0C_Dataset_Verification_and_Metadata_Check.py
"""

from __future__ import annotations

import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
DATASET_ROOT = PROJECT_ROOT / "datasets"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

EXPECTED_DATASETS = [
    "CBIS-DDSM",
    "INbreast",
    "VinDr-Mammo",
]

OUTPUT_SUMMARY_CSV = RESULTS_TABLE_DIR / "Stage0C_Dataset_Verification_Summary.csv"
OUTPUT_CBIS_INDEX_CSV = RESULTS_TABLE_DIR / "Stage0C_CBIS_DDSM_Case_Index.csv"
OUTPUT_METADATA_CHECK_CSV = RESULTS_TABLE_DIR / "Stage0C_Metadata_File_Check.csv"
OUTPUT_AVAILABILITY_CSV = RESULTS_TABLE_DIR / "Stage0C_Multiview_Bilateral_Availability.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage0C_Dataset_Verification_Summary.json"
OUTPUT_REPORT_TXT = RESULTS_REPORT_DIR / "Stage0C_Dataset_Verification_Report.txt"


# =============================================================================
# Helpers
# =============================================================================

def format_size(num_bytes: int) -> str:
    """Return human-readable file size."""
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def safe_stat_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def count_files_by_extension(root: Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    if not root.exists():
        return counts

    for item in root.rglob("*"):
        if item.is_file():
            ext = item.suffix.lower().strip()
            if ext == "":
                ext = "[no_extension]"
            counts[ext] = counts.get(ext, 0) + 1

    return counts


def folder_size(root: Path) -> int:
    if not root.exists():
        return 0

    total = 0
    for item in root.rglob("*"):
        if item.is_file():
            total += safe_stat_size(item)
    return total


def count_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_dir())


def count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def find_csv_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.csv"))


def find_dicom_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.dcm"))


def find_png_files(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted(root.rglob("*.png"))


def read_csv_safely(path: Path) -> Tuple[bool, int, int, str]:
    try:
        df = pd.read_csv(path)
        return True, int(df.shape[0]), int(df.shape[1]), ""
    except Exception as e:
        return False, 0, 0, str(e)


def extract_cbis_info_from_name(name: str) -> Dict[str, Optional[str]]:
    """
    Extract information from CBIS-DDSM folder names like:
    Calc-Test_P_00038_LEFT_CC_1
    Mass-Training_P_00001_RIGHT_MLO
    """

    info = {
        "raw_case_name": name,
        "lesion_type": None,
        "split": None,
        "patient_id": None,
        "laterality": None,
        "view": None,
        "abnormality_index": None,
    }

    lesion_match = re.search(r"^(Calc|Mass)", name, flags=re.IGNORECASE)
    split_match = re.search(r"(Training|Test)", name, flags=re.IGNORECASE)
    patient_match = re.search(r"(P_\d+)", name, flags=re.IGNORECASE)
    laterality_match = re.search(r"(LEFT|RIGHT)", name, flags=re.IGNORECASE)
    view_match = re.search(r"(CC|MLO)", name, flags=re.IGNORECASE)

    if lesion_match:
        info["lesion_type"] = lesion_match.group(1).capitalize()

    if split_match:
        info["split"] = split_match.group(1).capitalize()

    if patient_match:
        info["patient_id"] = patient_match.group(1).upper()

    if laterality_match:
        info["laterality"] = laterality_match.group(1).upper()

    if view_match:
        info["view"] = view_match.group(1).upper()

    # abnormality index is usually the last numeric token after view
    # e.g., Calc-Test_P_00038_LEFT_CC_1 -> 1
    parts = name.split("_")
    if len(parts) > 0 and parts[-1].isdigit():
        info["abnormality_index"] = parts[-1]

    return info


def first_existing_child(root: Path, candidates: List[str]) -> Optional[Path]:
    for candidate in candidates:
        p = root / candidate
        if p.exists():
            return p
    return None


# =============================================================================
# Dataset-Level Verification
# =============================================================================

def verify_dataset_folders() -> pd.DataFrame:
    rows = []

    for dataset_name in EXPECTED_DATASETS:
        dataset_path = DATASET_ROOT / dataset_name

        ext_counts = count_files_by_extension(dataset_path)
        total_size = folder_size(dataset_path)

        rows.append({
            "dataset": dataset_name,
            "path": str(dataset_path),
            "exists": dataset_path.exists(),
            "folders": count_dirs(dataset_path),
            "files": count_files(dataset_path),
            "size_bytes": total_size,
            "size_readable": format_size(total_size),
            "dcm_files": ext_counts.get(".dcm", 0),
            "png_files": ext_counts.get(".png", 0),
            "csv_files": ext_counts.get(".csv", 0),
            "xml_files": ext_counts.get(".xml", 0),
            "roi_files": ext_counts.get(".roi", 0),
            "zip_files": ext_counts.get(".zip", 0),
            "tcia_files": ext_counts.get(".tcia", 0),
            "status": "OK" if dataset_path.exists() and count_files(dataset_path) > 0 else "MISSING_OR_EMPTY",
        })

    return pd.DataFrame(rows)


def verify_metadata_files() -> pd.DataFrame:
    rows = []

    for dataset_name in EXPECTED_DATASETS:
        dataset_path = DATASET_ROOT / dataset_name
        csv_files = find_csv_files(dataset_path)

        if not csv_files:
            rows.append({
                "dataset": dataset_name,
                "metadata_file": "",
                "exists": False,
                "readable": False,
                "rows": 0,
                "columns": 0,
                "error": "No CSV metadata files found",
            })
            continue

        for csv_file in csv_files:
            readable, n_rows, n_cols, err = read_csv_safely(csv_file)
            rows.append({
                "dataset": dataset_name,
                "metadata_file": str(csv_file),
                "exists": csv_file.exists(),
                "readable": readable,
                "rows": n_rows,
                "columns": n_cols,
                "error": err,
            })

    return pd.DataFrame(rows)


# =============================================================================
# CBIS-DDSM Case Index
# =============================================================================

def build_cbis_case_index() -> pd.DataFrame:
    cbis_root = DATASET_ROOT / "CBIS-DDSM"

    # Common layout from TCIA/NBIA download:
    # CBIS-DDSM/cbis_ddsm/<case_folder>/<subfolders>/*.dcm
    cbis_data_root = first_existing_child(
        cbis_root,
        ["cbis_ddsm", "CBIS-DDSM", "manifest-166xxxx", "manifest"]
    )

    if cbis_data_root is None:
        cbis_data_root = cbis_root

    rows = []

    if not cbis_data_root.exists():
        return pd.DataFrame(rows)

    for case_dir in sorted([p for p in cbis_data_root.iterdir() if p.is_dir()]):
        case_info = extract_cbis_info_from_name(case_dir.name)

        dcm_files = find_dicom_files(case_dir)
        png_files = find_png_files(case_dir)
        xml_files = sorted(case_dir.rglob("*.xml"))
        roi_files = sorted(case_dir.rglob("*.roi"))

        largest_dcm = ""
        largest_dcm_size = 0

        if dcm_files:
            largest = max(dcm_files, key=lambda p: safe_stat_size(p))
            largest_dcm = str(largest)
            largest_dcm_size = safe_stat_size(largest)

        rows.append({
            "dataset": "CBIS-DDSM",
            "case_folder": case_dir.name,
            "case_path": str(case_dir),
            "lesion_type": case_info["lesion_type"],
            "split": case_info["split"],
            "patient_id": case_info["patient_id"],
            "laterality": case_info["laterality"],
            "view": case_info["view"],
            "abnormality_index": case_info["abnormality_index"],
            "dcm_count": len(dcm_files),
            "png_count": len(png_files),
            "xml_count": len(xml_files),
            "roi_count": len(roi_files),
            "largest_dcm": largest_dcm,
            "largest_dcm_size_bytes": largest_dcm_size,
            "largest_dcm_size_readable": format_size(largest_dcm_size),
            "has_dicom": len(dcm_files) > 0,
            "has_png": len(png_files) > 0,
            "has_annotation": (len(xml_files) + len(roi_files)) > 0,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Multi-view / Bilateral Availability
# =============================================================================

def compute_multiview_bilateral_availability(cbis_index: pd.DataFrame) -> pd.DataFrame:
    if cbis_index.empty:
        return pd.DataFrame()

    required_cols = ["patient_id", "laterality", "view"]
    for col in required_cols:
        if col not in cbis_index.columns:
            return pd.DataFrame()

    df = cbis_index.copy()

    df = df[
        df["patient_id"].notna()
        & df["laterality"].notna()
        & df["view"].notna()
    ].copy()

    if df.empty:
        return pd.DataFrame()

    rows = []

    for patient_id, g in df.groupby("patient_id"):
        views = set(
            f"{str(row.laterality).upper()}_{str(row.view).upper()}"
            for row in g.itertuples()
        )

        has_left_cc = "LEFT_CC" in views
        has_left_mlo = "LEFT_MLO" in views
        has_right_cc = "RIGHT_CC" in views
        has_right_mlo = "RIGHT_MLO" in views

        left_multiview = has_left_cc and has_left_mlo
        right_multiview = has_right_cc and has_right_mlo

        cc_bilateral = has_left_cc and has_right_cc
        mlo_bilateral = has_left_mlo and has_right_mlo

        complete_four_view = (
            has_left_cc
            and has_left_mlo
            and has_right_cc
            and has_right_mlo
        )

        rows.append({
            "patient_id": patient_id,
            "n_case_folders": int(len(g)),
            "has_left_cc": has_left_cc,
            "has_left_mlo": has_left_mlo,
            "has_right_cc": has_right_cc,
            "has_right_mlo": has_right_mlo,
            "left_multiview_available": left_multiview,
            "right_multiview_available": right_multiview,
            "any_multiview_available": left_multiview or right_multiview,
            "cc_bilateral_available": cc_bilateral,
            "mlo_bilateral_available": mlo_bilateral,
            "any_bilateral_available": cc_bilateral or mlo_bilateral,
            "complete_four_view_available": complete_four_view,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Report
# =============================================================================

def write_report(
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cbis_index_df: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("STAGE 0C DATASET VERIFICATION AND METADATA CHECK")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Dataset root: {DATASET_ROOT}")
    lines.append("")

    lines.append("DATASET SUMMARY")
    lines.append("-" * 100)
    if not summary_df.empty:
        lines.append(summary_df.to_string(index=False))
    else:
        lines.append("No dataset summary available.")
    lines.append("")

    lines.append("METADATA FILE CHECK")
    lines.append("-" * 100)
    if not metadata_df.empty:
        lines.append(metadata_df.to_string(index=False))
    else:
        lines.append("No metadata files found.")
    lines.append("")

    lines.append("CBIS-DDSM CASE INDEX SUMMARY")
    lines.append("-" * 100)
    if not cbis_index_df.empty:
        lines.append(f"Case folders indexed: {len(cbis_index_df):,}")
        lines.append(f"Unique patients: {cbis_index_df['patient_id'].nunique(dropna=True):,}")
        lines.append(f"DICOM-positive case folders: {int(cbis_index_df['has_dicom'].sum()):,}")
        lines.append(f"Annotation-positive case folders: {int(cbis_index_df['has_annotation'].sum()):,}")

        if "split" in cbis_index_df.columns:
            lines.append("")
            lines.append("Split distribution:")
            lines.append(cbis_index_df["split"].value_counts(dropna=False).to_string())

        if "lesion_type" in cbis_index_df.columns:
            lines.append("")
            lines.append("Lesion type distribution:")
            lines.append(cbis_index_df["lesion_type"].value_counts(dropna=False).to_string())

        if "view" in cbis_index_df.columns:
            lines.append("")
            lines.append("View distribution:")
            lines.append(cbis_index_df["view"].value_counts(dropna=False).to_string())

        if "laterality" in cbis_index_df.columns:
            lines.append("")
            lines.append("Laterality distribution:")
            lines.append(cbis_index_df["laterality"].value_counts(dropna=False).to_string())
    else:
        lines.append("No CBIS-DDSM case folders indexed.")
    lines.append("")

    lines.append("MULTI-VIEW AND BILATERAL AVAILABILITY")
    lines.append("-" * 100)
    if not availability_df.empty:
        n_patients = len(availability_df)
        lines.append(f"Patients analyzed: {n_patients:,}")
        lines.append(
            f"Any multi-view available: "
            f"{int(availability_df['any_multiview_available'].sum()):,}"
        )
        lines.append(
            f"Any bilateral available: "
            f"{int(availability_df['any_bilateral_available'].sum()):,}"
        )
        lines.append(
            f"Complete four-view available: "
            f"{int(availability_df['complete_four_view_available'].sum()):,}"
        )
    else:
        lines.append("No availability information computed.")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_SUMMARY_CSV))
    lines.append(str(OUTPUT_METADATA_CHECK_CSV))
    lines.append(str(OUTPUT_CBIS_INDEX_CSV))
    lines.append(str(OUTPUT_AVAILABILITY_CSV))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))
    lines.append("")

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_summary(
    summary_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    cbis_index_df: pd.DataFrame,
    availability_df: pd.DataFrame,
) -> None:
    data = {
        "generated": str(datetime.now()),
        "dataset_root": str(DATASET_ROOT),
        "summary": summary_df.to_dict(orient="records"),
        "metadata_files": metadata_df.to_dict(orient="records"),
        "cbis_ddsm": {
            "case_folders": int(len(cbis_index_df)) if not cbis_index_df.empty else 0,
            "unique_patients": int(cbis_index_df["patient_id"].nunique(dropna=True))
            if not cbis_index_df.empty and "patient_id" in cbis_index_df.columns else 0,
            "dicom_positive_case_folders": int(cbis_index_df["has_dicom"].sum())
            if not cbis_index_df.empty and "has_dicom" in cbis_index_df.columns else 0,
            "annotation_positive_case_folders": int(cbis_index_df["has_annotation"].sum())
            if not cbis_index_df.empty and "has_annotation" in cbis_index_df.columns else 0,
        },
        "availability": {
            "patients_analyzed": int(len(availability_df)) if not availability_df.empty else 0,
            "any_multiview_available": int(availability_df["any_multiview_available"].sum())
            if not availability_df.empty else 0,
            "any_bilateral_available": int(availability_df["any_bilateral_available"].sum())
            if not availability_df.empty else 0,
            "complete_four_view_available": int(availability_df["complete_four_view_available"].sum())
            if not availability_df.empty else 0,
        }
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 0C DATASET VERIFICATION AND METADATA CHECK")
    print("=" * 100)
    print(f"Dataset root: {DATASET_ROOT}")

    summary_df = verify_dataset_folders()
    metadata_df = verify_metadata_files()
    cbis_index_df = build_cbis_case_index()
    availability_df = compute_multiview_bilateral_availability(cbis_index_df)

    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    metadata_df.to_csv(OUTPUT_METADATA_CHECK_CSV, index=False, encoding="utf-8-sig")
    cbis_index_df.to_csv(OUTPUT_CBIS_INDEX_CSV, index=False, encoding="utf-8-sig")
    availability_df.to_csv(OUTPUT_AVAILABILITY_CSV, index=False, encoding="utf-8-sig")

    save_json_summary(
        summary_df=summary_df,
        metadata_df=metadata_df,
        cbis_index_df=cbis_index_df,
        availability_df=availability_df,
    )

    write_report(
        summary_df=summary_df,
        metadata_df=metadata_df,
        cbis_index_df=cbis_index_df,
        availability_df=availability_df,
    )

    print()
    print("STAGE 0C COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Dataset summary:      {OUTPUT_SUMMARY_CSV}")
    print(f"Metadata check:       {OUTPUT_METADATA_CHECK_CSV}")
    print(f"CBIS case index:      {OUTPUT_CBIS_INDEX_CSV}")
    print(f"Availability table:   {OUTPUT_AVAILABILITY_CSV}")
    print(f"JSON summary:         {OUTPUT_JSON}")
    print(f"Text report:          {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()