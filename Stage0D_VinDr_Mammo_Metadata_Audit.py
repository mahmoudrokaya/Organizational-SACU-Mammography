r"""
Stage0D_VinDr_Mammo_Metadata_Audit.py

Purpose
-------
Explore and audit the VinDr-Mammo dataset folder to identify why metadata
needed for multi-view, bilateral, and labeled evaluation is missing.

This script checks:
1. Folder structure
2. Image counts
3. Case-folder counts
4. Number of images per case
5. Presence of metadata files: CSV, XLSX, JSON, TXT, TSV
6. Presence of compressed archives: ZIP, TAR, GZ, 7Z
7. Whether metadata may be hidden inside archives
8. PNG filename structure
9. Case-folder naming structure
10. Whether Stage1A and Stage1A2 outputs contain usable VinDr metadata
11. Required fields for valid mammography experiments:
    - image_id
    - study_id / exam_id
    - laterality
    - view / view_position
    - BI-RADS / label
    - breast density
12. Final readiness decision

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage0D_VinDr_Mammo_Metadata_Audit.py
"""

from __future__ import annotations

import json
import zipfile
import tarfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
VINDR_ROOT = PROJECT_ROOT / "datasets" / "VinDr-Mammo"

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

STAGE1A_INDEX = RESULTS_TABLE_DIR / "Stage1A_Preprocessed_Image_Index.csv"
STAGE1A2_V2_METADATA = RESULTS_TABLE_DIR / "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv"

OUTPUT_STRUCTURE_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Folder_Structure_Audit.csv"
OUTPUT_FILETYPE_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Filetype_Counts.csv"
OUTPUT_IMAGE_CASE_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Image_Case_Audit.csv"
OUTPUT_METADATA_FILES_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Metadata_File_Candidates.csv"
OUTPUT_ARCHIVE_FILES_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Archive_File_Audit.csv"
OUTPUT_CSV_PROFILES_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Metadata_CSV_Profiles.csv"
OUTPUT_REQUIRED_FIELDS_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Required_Field_Audit.csv"
OUTPUT_STAGE_OUTPUT_CSV = RESULTS_TABLE_DIR / "Stage0D_VinDr_Stage_Output_Audit.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage0D_VinDr_Mammo_Metadata_Audit.json"
OUTPUT_REPORT_TXT = RESULTS_REPORT_DIR / "Stage0D_VinDr_Mammo_Metadata_Audit_Report.txt"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom", ".tif", ".tiff"}
METADATA_EXTENSIONS = {".csv", ".tsv", ".xlsx", ".xls", ".json", ".txt", ".xml", ".parquet", ".feather"}
ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".7z", ".rar"}

REQUIRED_CONCEPTS = {
    "image_id": ["image_id", "imageid", "image", "filename", "file_name", "file"],
    "study_id": ["study_id", "studyid", "study", "exam_id", "examid"],
    "laterality": ["laterality", "breast", "side", "left", "right"],
    "view": ["view", "view_position", "viewposition", "projection", "cc", "mlo"],
    "label": ["label", "target", "class", "birads", "bi-rads", "breast_birads", "finding"],
    "density": ["density", "breast_density", "acr"],
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


def file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return 0


def human_size(num_bytes: int) -> str:
    if num_bytes >= 1024 ** 3:
        return f"{num_bytes / (1024 ** 3):.2f} GB"
    if num_bytes >= 1024 ** 2:
        return f"{num_bytes / (1024 ** 2):.2f} MB"
    if num_bytes >= 1024:
        return f"{num_bytes / 1024:.2f} KB"
    return f"{num_bytes} B"


def compact_path(path: Path, keep_last: int = 6) -> str:
    parts = path.parts
    return str(Path(*parts[-keep_last:])) if len(parts) > keep_last else str(path)


def read_csv_flexible(path: Path) -> Tuple[pd.DataFrame, str, str]:
    attempts = [
        ("comma", {"sep": ","}),
        ("semicolon", {"sep": ";"}),
        ("tab", {"sep": "\t"}),
        ("pipe", {"sep": "|"}),
    ]

    for mode, kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df, mode, ""
        except Exception as e:
            last_error = str(e)

    try:
        df = pd.read_csv(path, header=None)
        return df, "single_column_or_raw", ""
    except Exception as e:
        return pd.DataFrame(), "failed", str(e)


def detect_columns(columns: List[str]) -> Dict[str, List[str]]:
    detected = {}

    normalized = {
        c: c.lower().replace(" ", "").replace("_", "").replace("-", "")
        for c in columns
    }

    for concept, keywords in REQUIRED_CONCEPTS.items():
        hits = []

        for col, clean in normalized.items():
            for key in keywords:
                key_clean = key.lower().replace(" ", "").replace("_", "").replace("-", "")
                if key_clean in clean or clean in key_clean:
                    hits.append(col)

        detected[concept] = sorted(set(hits))

    return detected


def contains_required_terms_in_text(text: str) -> Dict[str, bool]:
    text_upper = safe_str(text).upper()

    return {
        "contains_laterality_terms": any(x in text_upper for x in ["LEFT", "RIGHT", "_L_", "_R_", ",L,", ",R,", ";L;", ";R;"]),
        "contains_view_terms": any(x in text_upper for x in ["CC", "MLO", "VIEW", "PROJECTION"]),
        "contains_label_terms": any(x in text_upper for x in ["BI-RADS", "BIRADS", "MALIGNANT", "BENIGN", "NORMAL", "LABEL"]),
        "contains_density_terms": any(x in text_upper for x in ["DENSITY", "ACR"]),
    }


# =============================================================================
# Folder and File Audits
# =============================================================================

def audit_folder_structure() -> pd.DataFrame:
    rows = []

    if not VINDR_ROOT.exists():
        return pd.DataFrame([{
            "path": str(VINDR_ROOT),
            "relative_path": "",
            "is_dir": False,
            "depth": 0,
            "file_count_direct": 0,
            "folder_count_direct": 0,
            "size_bytes_direct": 0,
            "size_readable_direct": "0 B",
            "note": "VinDr-Mammo root folder does not exist",
        }])

    for path in [VINDR_ROOT] + [p for p in VINDR_ROOT.rglob("*") if p.is_dir()]:
        files = [x for x in path.iterdir() if x.is_file()]
        folders = [x for x in path.iterdir() if x.is_dir()]
        size_direct = sum(file_size(f) for f in files)

        rel = path.relative_to(VINDR_ROOT) if path != VINDR_ROOT else Path(".")

        rows.append({
            "path": str(path),
            "relative_path": str(rel),
            "is_dir": True,
            "depth": len(rel.parts) if str(rel) != "." else 0,
            "file_count_direct": len(files),
            "folder_count_direct": len(folders),
            "size_bytes_direct": size_direct,
            "size_readable_direct": human_size(size_direct),
            "note": "",
        })

    return pd.DataFrame(rows)


def audit_filetypes() -> pd.DataFrame:
    counts: Dict[str, Dict[str, int]] = {}

    if not VINDR_ROOT.exists():
        return pd.DataFrame()

    for path in VINDR_ROOT.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()
        if ext == "":
            ext = "[no_extension]"

        if ext not in counts:
            counts[ext] = {
                "file_count": 0,
                "total_size_bytes": 0,
            }

        counts[ext]["file_count"] += 1
        counts[ext]["total_size_bytes"] += file_size(path)

    rows = []
    for ext, info in sorted(counts.items(), key=lambda x: x[1]["file_count"], reverse=True):
        rows.append({
            "extension": ext,
            "file_count": info["file_count"],
            "total_size_bytes": info["total_size_bytes"],
            "total_size_readable": human_size(info["total_size_bytes"]),
            "category": (
                "image" if ext in IMAGE_EXTENSIONS else
                "metadata" if ext in METADATA_EXTENSIONS else
                "archive" if ext in ARCHIVE_EXTENSIONS else
                "other"
            )
        })

    return pd.DataFrame(rows)


def audit_image_cases() -> pd.DataFrame:
    rows = []

    if not VINDR_ROOT.exists():
        return pd.DataFrame()

    image_paths = [
        p for p in VINDR_ROOT.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    case_map: Dict[str, List[Path]] = {}

    for img in image_paths:
        case_id = img.parent.name
        case_map.setdefault(case_id, []).append(img)

    for case_id, imgs in sorted(case_map.items()):
        stems = [p.stem for p in imgs]
        exts = sorted(set(p.suffix.lower() for p in imgs))

        rows.append({
            "case_id": case_id,
            "image_count": len(imgs),
            "extensions": "|".join(exts),
            "sample_image_1": str(imgs[0]) if len(imgs) > 0 else "",
            "sample_image_2": str(imgs[1]) if len(imgs) > 1 else "",
            "sample_image_3": str(imgs[2]) if len(imgs) > 2 else "",
            "sample_image_4": str(imgs[3]) if len(imgs) > 3 else "",
            "stem_sample": "|".join(stems[:4]),
            "all_stems_hex_like": all(len(s) >= 16 and all(c in "0123456789abcdefABCDEF" for c in s) for s in stems),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Metadata and Archive Audit
# =============================================================================

def audit_metadata_candidates() -> pd.DataFrame:
    rows = []

    if not VINDR_ROOT.exists():
        return pd.DataFrame()

    for path in VINDR_ROOT.rglob("*"):
        if not path.is_file():
            continue

        ext = path.suffix.lower()

        if ext not in METADATA_EXTENSIONS:
            continue

        row = {
            "path": str(path),
            "relative_path": str(path.relative_to(VINDR_ROOT)),
            "extension": ext,
            "size_bytes": file_size(path),
            "size_readable": human_size(file_size(path)),
            "name": path.name,
            "likely_metadata": True,
            "notes": "",
        }

        rows.append(row)

    return pd.DataFrame(rows)


def audit_archives() -> pd.DataFrame:
    rows = []

    if not VINDR_ROOT.exists():
        return pd.DataFrame()

    for path in VINDR_ROOT.rglob("*"):
        if not path.is_file():
            continue

        suffixes = "".join(path.suffixes).lower()
        ext = path.suffix.lower()

        is_archive = ext in ARCHIVE_EXTENSIONS or any(suffixes.endswith(x) for x in [".tar.gz", ".tgz"])

        if not is_archive:
            continue

        internal_files = []
        internal_metadata = []
        error = ""

        try:
            if zipfile.is_zipfile(path):
                with zipfile.ZipFile(path, "r") as z:
                    internal_files = z.namelist()[:200]
                    internal_metadata = [
                        x for x in z.namelist()
                        if Path(x).suffix.lower() in METADATA_EXTENSIONS
                    ][:200]

            elif tarfile.is_tarfile(path):
                with tarfile.open(path, "r:*") as t:
                    names = t.getnames()
                    internal_files = names[:200]
                    internal_metadata = [
                        x for x in names
                        if Path(x).suffix.lower() in METADATA_EXTENSIONS
                    ][:200]

        except Exception as e:
            error = str(e)

        rows.append({
            "path": str(path),
            "relative_path": str(path.relative_to(VINDR_ROOT)),
            "extension": ext,
            "size_bytes": file_size(path),
            "size_readable": human_size(file_size(path)),
            "internal_file_sample": " | ".join(internal_files[:30]),
            "internal_metadata_candidates": " | ".join(internal_metadata[:30]),
            "internal_metadata_count_sampled": len(internal_metadata),
            "error": error,
        })

    return pd.DataFrame(rows)


def profile_metadata_csvs(metadata_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if metadata_df.empty:
        return pd.DataFrame()

    csv_paths = metadata_df[
        metadata_df["extension"].isin([".csv", ".tsv", ".txt"])
    ]["path"].tolist()

    for p in csv_paths:
        path = Path(p)
        df, mode, error = read_csv_flexible(path)

        if df.empty:
            rows.append({
                "path": str(path),
                "parse_mode": mode,
                "rows": 0,
                "columns": 0,
                "column_names": "",
                "detected_image_id_cols": "",
                "detected_study_id_cols": "",
                "detected_laterality_cols": "",
                "detected_view_cols": "",
                "detected_label_cols": "",
                "detected_density_cols": "",
                "contains_laterality_terms": False,
                "contains_view_terms": False,
                "contains_label_terms": False,
                "contains_density_terms": False,
                "first_rows_sample": "",
                "error": error,
            })
            continue

        columns = [str(c) for c in df.columns]
        detected = detect_columns(columns)

        sample_text = " ".join(
            df.head(20).astype(str).fillna("").values.reshape(-1).tolist()
        )

        term_flags = contains_required_terms_in_text(sample_text)

        rows.append({
            "path": str(path),
            "parse_mode": mode,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "column_names": " | ".join(columns[:80]),
            "detected_image_id_cols": " | ".join(detected["image_id"]),
            "detected_study_id_cols": " | ".join(detected["study_id"]),
            "detected_laterality_cols": " | ".join(detected["laterality"]),
            "detected_view_cols": " | ".join(detected["view"]),
            "detected_label_cols": " | ".join(detected["label"]),
            "detected_density_cols": " | ".join(detected["density"]),
            **term_flags,
            "first_rows_sample": sample_text[:3000],
            "error": "",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Required Field and Stage Output Audit
# =============================================================================

def audit_required_fields(csv_profile_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if csv_profile_df.empty:
        for concept in REQUIRED_CONCEPTS:
            rows.append({
                "required_concept": concept,
                "available": False,
                "candidate_files": "",
                "candidate_columns": "",
                "status": "MISSING",
                "notes": "No metadata CSV/TSV/TXT files found or readable.",
            })
        return pd.DataFrame(rows)

    concept_col_map = {
        "image_id": "detected_image_id_cols",
        "study_id": "detected_study_id_cols",
        "laterality": "detected_laterality_cols",
        "view": "detected_view_cols",
        "label": "detected_label_cols",
        "density": "detected_density_cols",
    }

    for concept, col_name in concept_col_map.items():
        candidates = csv_profile_df[
            csv_profile_df[col_name].astype(str).str.len() > 0
        ].copy()

        available = not candidates.empty

        rows.append({
            "required_concept": concept,
            "available": bool(available),
            "candidate_files": " | ".join(candidates["path"].head(10).tolist()) if available else "",
            "candidate_columns": " | ".join(candidates[col_name].head(10).tolist()) if available else "",
            "status": "AVAILABLE" if available else "MISSING",
            "notes": "",
        })

    return pd.DataFrame(rows)


def audit_stage_outputs() -> pd.DataFrame:
    rows = []

    # Stage1A processed index
    if STAGE1A_INDEX.exists():
        try:
            df = pd.read_csv(STAGE1A_INDEX)
            d = df[df["dataset"] == "VinDr-Mammo"].copy()

            rows.append({
                "source": "Stage1A_Preprocessed_Image_Index.csv",
                "exists": True,
                "records": int(len(d)),
                "known_patient_id": int(d["patient_id"].notna().sum()) if "patient_id" in d.columns else 0,
                "known_laterality": int(d["laterality"].notna().sum()) if "laterality" in d.columns else 0,
                "known_view": int(d["view"].notna().sum()) if "view" in d.columns else 0,
                "known_label": 0,
                "notes": "Stage1A is image preprocessing; metadata fields may remain unknown.",
            })
        except Exception as e:
            rows.append({
                "source": "Stage1A_Preprocessed_Image_Index.csv",
                "exists": True,
                "records": 0,
                "known_patient_id": 0,
                "known_laterality": 0,
                "known_view": 0,
                "known_label": 0,
                "notes": f"Could not read file: {e}",
            })
    else:
        rows.append({
            "source": "Stage1A_Preprocessed_Image_Index.csv",
            "exists": False,
            "records": 0,
            "known_patient_id": 0,
            "known_laterality": 0,
            "known_view": 0,
            "known_label": 0,
            "notes": "Missing Stage1A output.",
        })

    # Stage1A2 v2 harmonized metadata
    if STAGE1A2_V2_METADATA.exists():
        try:
            df = pd.read_csv(STAGE1A2_V2_METADATA)
            d = df[df["dataset"] == "VinDr-Mammo"].copy()

            rows.append({
                "source": "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv",
                "exists": True,
                "records": int(len(d)),
                "known_patient_id": int(d["patient_id"].astype(str).str.len().gt(0).sum()) if "patient_id" in d.columns else 0,
                "known_laterality": int(d["laterality"].astype(str).str.len().gt(0).sum()) if "laterality" in d.columns else 0,
                "known_view": int(d["view"].astype(str).str.len().gt(0).sum()) if "view" in d.columns else 0,
                "known_label": int(d["label"].notna().sum()) if "label" in d.columns else 0,
                "notes": "If view/laterality/label are zero, VinDr cannot be used for multi-view/bilateral/labeled evaluation.",
            })
        except Exception as e:
            rows.append({
                "source": "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv",
                "exists": True,
                "records": 0,
                "known_patient_id": 0,
                "known_laterality": 0,
                "known_view": 0,
                "known_label": 0,
                "notes": f"Could not read file: {e}",
            })
    else:
        rows.append({
            "source": "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv",
            "exists": False,
            "records": 0,
            "known_patient_id": 0,
            "known_laterality": 0,
            "known_view": 0,
            "known_label": 0,
            "notes": "Missing Stage1A2 v2 output.",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Final Diagnosis
# =============================================================================

def final_diagnosis(
    filetype_df: pd.DataFrame,
    image_case_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    required_df: pd.DataFrame,
    stage_output_df: pd.DataFrame,
) -> Dict[str, object]:

    png_count = 0
    if not filetype_df.empty and ".png" in set(filetype_df["extension"]):
        png_count = int(filetype_df[filetype_df["extension"] == ".png"]["file_count"].iloc[0])

    case_count = int(len(image_case_df)) if not image_case_df.empty else 0

    metadata_count = int(len(metadata_df)) if not metadata_df.empty else 0
    archive_count = int(len(archive_df)) if not archive_df.empty else 0

    missing_concepts = []
    if not required_df.empty:
        missing_concepts = required_df[
            required_df["status"] == "MISSING"
        ]["required_concept"].tolist()

    ready_for_multiview = not any(x in missing_concepts for x in ["image_id", "laterality", "view"])
    ready_for_labeled_eval = not any(x in missing_concepts for x in ["image_id", "label"])
    ready_for_bilateral = not any(x in missing_concepts for x in ["image_id", "laterality", "view"])

    diagnosis = {
        "png_count": png_count,
        "case_folder_count": case_count,
        "metadata_file_count": metadata_count,
        "archive_file_count": archive_count,
        "missing_required_concepts": missing_concepts,
        "ready_for_multiview": ready_for_multiview,
        "ready_for_bilateral": ready_for_bilateral,
        "ready_for_labeled_evaluation": ready_for_labeled_eval,
        "final_status": "",
        "recommended_action": "",
    }

    if png_count > 0 and metadata_count == 0 and archive_count == 0:
        diagnosis["final_status"] = "IMAGES_ONLY_NO_METADATA"
        diagnosis["recommended_action"] = (
            "Download/add the official VinDr-Mammo metadata files. The current folder appears to contain "
            "only PNG images grouped by case folders. Do not infer view/laterality by image order."
        )

    elif metadata_count > 0 and missing_concepts:
        diagnosis["final_status"] = "METADATA_PRESENT_BUT_INCOMPLETE"
        diagnosis["recommended_action"] = (
            "Inspect candidate metadata files and map their columns to image_id, laterality, view, and label. "
            "If these columns are absent, download the full official annotation package."
        )

    elif archive_count > 0 and metadata_count == 0:
        diagnosis["final_status"] = "ARCHIVES_PRESENT_CHECK_EXTRACTION"
        diagnosis["recommended_action"] = (
            "Inspect/extract archives because metadata may be inside compressed files."
        )

    elif ready_for_multiview and ready_for_labeled_eval:
        diagnosis["final_status"] = "READY"
        diagnosis["recommended_action"] = (
            "VinDr-Mammo appears ready for harmonized multi-view/labeled processing."
        )

    else:
        diagnosis["final_status"] = "UNKNOWN_NEEDS_MANUAL_CHECK"
        diagnosis["recommended_action"] = (
            "Review the audit tables manually."
        )

    return diagnosis


# =============================================================================
# Report
# =============================================================================

def write_report(
    filetype_df: pd.DataFrame,
    image_case_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    archive_df: pd.DataFrame,
    csv_profile_df: pd.DataFrame,
    required_df: pd.DataFrame,
    stage_output_df: pd.DataFrame,
    diagnosis: Dict[str, object],
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 0D VINDR-MAMMO METADATA AUDIT REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"VinDr root: {VINDR_ROOT}")
    lines.append("")

    lines.append("FINAL DIAGNOSIS")
    lines.append("-" * 100)
    for k, v in diagnosis.items():
        lines.append(f"{k}: {v}")
    lines.append("")

    lines.append("FILETYPE SUMMARY")
    lines.append("-" * 100)
    lines.append(filetype_df.to_string(index=False) if not filetype_df.empty else "No filetype data.")
    lines.append("")

    lines.append("IMAGE CASE SUMMARY")
    lines.append("-" * 100)
    if not image_case_df.empty:
        dist = image_case_df["image_count"].value_counts().sort_index()
        lines.append("Images per case distribution:")
        lines.append(dist.to_string())
        lines.append("")
        lines.append("Sample cases:")
        lines.append(image_case_df.head(20).to_string(index=False))
    else:
        lines.append("No image case data.")
    lines.append("")

    lines.append("METADATA CANDIDATE FILES")
    lines.append("-" * 100)
    lines.append(metadata_df.to_string(index=False) if not metadata_df.empty else "No metadata candidate files found.")
    lines.append("")

    lines.append("ARCHIVE FILES")
    lines.append("-" * 100)
    lines.append(archive_df.to_string(index=False) if not archive_df.empty else "No archive files found.")
    lines.append("")

    lines.append("CSV PROFILE")
    lines.append("-" * 100)
    lines.append(csv_profile_df.to_string(index=False) if not csv_profile_df.empty else "No readable metadata CSV/TSV/TXT profiles.")
    lines.append("")

    lines.append("REQUIRED FIELD AUDIT")
    lines.append("-" * 100)
    lines.append(required_df.to_string(index=False) if not required_df.empty else "No required field audit.")
    lines.append("")

    lines.append("STAGE OUTPUT AUDIT")
    lines.append("-" * 100)
    lines.append(stage_output_df.to_string(index=False) if not stage_output_df.empty else "No stage output audit.")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_STRUCTURE_CSV,
        OUTPUT_FILETYPE_CSV,
        OUTPUT_IMAGE_CASE_CSV,
        OUTPUT_METADATA_FILES_CSV,
        OUTPUT_ARCHIVE_FILES_CSV,
        OUTPUT_CSV_PROFILES_CSV,
        OUTPUT_REQUIRED_FIELDS_CSV,
        OUTPUT_STAGE_OUTPUT_CSV,
        OUTPUT_JSON,
        OUTPUT_REPORT_TXT,
    ]:
        lines.append(str(p))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json(diagnosis: Dict[str, object]) -> None:
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated": str(datetime.now()),
            "project_root": str(PROJECT_ROOT),
            "vindr_root": str(VINDR_ROOT),
            "diagnosis": diagnosis,
        }, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 0D VINDR-MAMMO METADATA AUDIT")
    print("=" * 100)
    print(f"VinDr root: {VINDR_ROOT}")
    print("-" * 100)

    print("Auditing folder structure...")
    structure_df = audit_folder_structure()

    print("Auditing file types...")
    filetype_df = audit_filetypes()

    print("Auditing image cases...")
    image_case_df = audit_image_cases()

    print("Finding metadata candidate files...")
    metadata_df = audit_metadata_candidates()

    print("Auditing archives...")
    archive_df = audit_archives()

    print("Profiling metadata CSV/TSV/TXT files...")
    csv_profile_df = profile_metadata_csvs(metadata_df)

    print("Auditing required fields...")
    required_df = audit_required_fields(csv_profile_df)

    print("Auditing previous stage outputs...")
    stage_output_df = audit_stage_outputs()

    print("Preparing final diagnosis...")
    diagnosis = final_diagnosis(
        filetype_df=filetype_df,
        image_case_df=image_case_df,
        metadata_df=metadata_df,
        archive_df=archive_df,
        required_df=required_df,
        stage_output_df=stage_output_df,
    )

    structure_df.to_csv(OUTPUT_STRUCTURE_CSV, index=False, encoding="utf-8-sig")
    filetype_df.to_csv(OUTPUT_FILETYPE_CSV, index=False, encoding="utf-8-sig")
    image_case_df.to_csv(OUTPUT_IMAGE_CASE_CSV, index=False, encoding="utf-8-sig")
    metadata_df.to_csv(OUTPUT_METADATA_FILES_CSV, index=False, encoding="utf-8-sig")
    archive_df.to_csv(OUTPUT_ARCHIVE_FILES_CSV, index=False, encoding="utf-8-sig")
    csv_profile_df.to_csv(OUTPUT_CSV_PROFILES_CSV, index=False, encoding="utf-8-sig")
    required_df.to_csv(OUTPUT_REQUIRED_FIELDS_CSV, index=False, encoding="utf-8-sig")
    stage_output_df.to_csv(OUTPUT_STAGE_OUTPUT_CSV, index=False, encoding="utf-8-sig")

    save_json(diagnosis)
    write_report(
        filetype_df=filetype_df,
        image_case_df=image_case_df,
        metadata_df=metadata_df,
        archive_df=archive_df,
        csv_profile_df=csv_profile_df,
        required_df=required_df,
        stage_output_df=stage_output_df,
        diagnosis=diagnosis,
    )

    print()
    print("STAGE 0D COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Final status:          {diagnosis.get('final_status')}")
    print(f"Recommended action:    {diagnosis.get('recommended_action')}")
    print(f"Structure audit:       {OUTPUT_STRUCTURE_CSV}")
    print(f"Filetype counts:       {OUTPUT_FILETYPE_CSV}")
    print(f"Image case audit:      {OUTPUT_IMAGE_CASE_CSV}")
    print(f"Metadata candidates:   {OUTPUT_METADATA_FILES_CSV}")
    print(f"Archive audit:         {OUTPUT_ARCHIVE_FILES_CSV}")
    print(f"CSV profiles:          {OUTPUT_CSV_PROFILES_CSV}")
    print(f"Required fields:       {OUTPUT_REQUIRED_FIELDS_CSV}")
    print(f"Stage output audit:    {OUTPUT_STAGE_OUTPUT_CSV}")
    print(f"JSON summary:          {OUTPUT_JSON}")
    print(f"Text report:           {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()