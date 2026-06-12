r"""
Stage1A2_Metadata_Harmonization_INbreast_VinDr_v2.py

Purpose
-------
Harmonize INbreast and VinDr-Mammo metadata after image preprocessing.

This version fixes VinDr-Mammo label propagation by merging:
1. breast-level_annotations.csv
2. finding_annotations.csv
3. metadata.csv

Outputs
-------
D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments\results\tables\Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv
D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments\results\tables\Stage1A2_v2_Harmonization_Summary.csv
D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments\results\reports\Stage1A2_v2_Metadata_Harmonization_Report.txt
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
DATASET_ROOT = PROJECT_ROOT / "datasets"

INPUT_STAGE1A_INDEX = PROJECT_ROOT / "results" / "tables" / "Stage1A_Preprocessed_Image_Index.csv"

INBREAST_ROOT = DATASET_ROOT / "INbreast"
VINDR_ROOT = DATASET_ROOT / "VinDr-Mammo"

OUTPUT_HARMONIZED_ROOT = PROJECT_ROOT / "preprocessing" / "metadata_harmonized"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_HARMONIZED_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GLOBAL_CSV = OUTPUT_TABLE_DIR / "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "Stage1A2_v2_Harmonization_Summary.csv"
OUTPUT_UNRESOLVED_CSV = OUTPUT_TABLE_DIR / "Stage1A2_v2_Unresolved_Metadata_Records.csv"
OUTPUT_NONSTANDARD_CSV = OUTPUT_TABLE_DIR / "Stage1A2_v2_Nonstandard_View_Records.csv"
OUTPUT_VINDR_FINDINGS_CSV = OUTPUT_TABLE_DIR / "Stage1A2_v2_VinDr_Finding_Level_Summary.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1A2_v2_Harmonization_Summary.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1A2_v2_Metadata_Harmonization_Report.txt"


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


def file_stem(path_text: str) -> str:
    try:
        return Path(safe_str(path_text)).stem
    except Exception:
        text = safe_str(path_text).replace("\\", "/").split("/")[-1]
        return re.sub(r"\.[A-Za-z0-9]+$", "", text)


def normalize_laterality(value: str) -> str:
    v = safe_str(value).upper()

    if v in {"L", "LEFT", "L."}:
        return "LEFT"
    if v in {"R", "RIGHT", "R."}:
        return "RIGHT"

    if "LEFT" in v:
        return "LEFT"
    if "RIGHT" in v:
        return "RIGHT"

    return ""


def normalize_view(value: str) -> str:
    v = safe_str(value).upper()

    if v == "CC":
        return "CC"
    if v in {"MLO", "ML"}:
        return "MLO"
    if v == "FB":
        return "FB"

    if "MLO" in v:
        return "MLO"
    if re.search(r"(^|[_\-\s])ML([_\-\s]|$)", v):
        return "MLO"
    if re.search(r"(^|[_\-\s])CC([_\-\s]|$)", v):
        return "CC"
    if re.search(r"(^|[_\-\s])FB([_\-\s]|$)", v):
        return "FB"

    return ""


def normalize_birads_text(value) -> str:
    v = safe_str(value).upper().replace(" ", "")

    if v in {"", "NAN", "NONE", "REMOVED"}:
        return "Unknown"

    match = re.search(r"BI-?RADS(\d+)", v)
    if match:
        return f"BI-RADS {match.group(1)}"

    nums = re.findall(r"\d+", v)
    if nums:
        return f"BI-RADS {nums[0]}"

    return safe_str(value)


def birads_to_binary_label(value) -> Optional[int]:
    """
    Conservative label mapping:
    BI-RADS 1, 2, 3 -> 0
    BI-RADS 4, 5, 6 -> 1
    """
    text = normalize_birads_text(value)

    nums = re.findall(r"\d+", text)
    if not nums:
        return None

    b = int(nums[0])

    if b in {1, 2, 3}:
        return 0
    if b in {4, 5, 6}:
        return 1

    return None


def binary_label_text(label: Optional[int]) -> str:
    if label == 1:
        return "Malignant_or_Suspicious"
    if label == 0:
        return "Benign_or_Nonmalignant"
    return "Unknown"


def density_to_text(value) -> str:
    return safe_str(value)


# =============================================================================
# INbreast Harmonization
# =============================================================================

def parse_inbreast_filename(path_text: str) -> Dict[str, str]:
    stem = file_stem(path_text)

    out = {
        "file_id": "",
        "exam_hash": "",
        "laterality": "",
        "view": "",
        "patient_id": "",
        "exam_id": "",
        "filename_parse_status": "FAILED",
    }

    pattern = re.compile(
        r"^(?P<file_id>\d+)_(?P<exam_hash>[A-Za-z0-9]+)_MG_(?P<lat>[LR])_(?P<view>CC|ML|MLO|FB)_ANON$",
        flags=re.IGNORECASE,
    )

    m = pattern.match(stem)

    if not m:
        return out

    file_id = m.group("file_id")
    exam_hash = m.group("exam_hash")
    lat = normalize_laterality(m.group("lat"))
    view = normalize_view(m.group("view"))

    out["file_id"] = file_id
    out["exam_hash"] = exam_hash
    out["laterality"] = lat
    out["view"] = view
    out["patient_id"] = f"INB_{exam_hash}"
    out["exam_id"] = f"INB_{exam_hash}"
    out["filename_parse_status"] = "OK" if lat and view else "PARTIAL"

    return out


def read_inbreast_csv() -> pd.DataFrame:
    csv_candidates = sorted(INBREAST_ROOT.rglob("INbreast.csv"))

    if not csv_candidates:
        csv_candidates = sorted(INBREAST_ROOT.rglob("*.csv"))

    if not csv_candidates:
        return pd.DataFrame()

    csv_path = csv_candidates[0]

    try:
        df = pd.read_csv(csv_path, sep=";")
    except Exception:
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()

    df.columns = [safe_str(c) for c in df.columns]
    df["__metadata_file__"] = str(csv_path)

    file_col = None
    for c in df.columns:
        if c.lower().replace(" ", "") in {"filename", "file"}:
            file_col = c
            break

    if file_col:
        df["file_id_key"] = df[file_col].astype(str).str.strip()
    else:
        df["file_id_key"] = ""

    return df


def build_inbreast_lookup(meta: pd.DataFrame) -> Dict[str, Dict[str, object]]:
    lookup = {}

    if meta.empty or "file_id_key" not in meta.columns:
        return lookup

    for row in meta.itertuples(index=False):
        d = row._asdict()
        key = safe_str(d.get("file_id_key", ""))
        if key:
            lookup[key] = d

    return lookup


def inbreast_status(patient_id: str, laterality: str, view: str) -> str:
    if patient_id and laterality and view in {"CC", "MLO"}:
        return "OK"
    if patient_id and laterality and view == "FB":
        return "OK_NONSTANDARD_VIEW"
    return "PARTIAL"


def harmonize_inbreast(stage1a_df: pd.DataFrame) -> pd.DataFrame:
    d = stage1a_df[stage1a_df["dataset"] == "INbreast"].copy()

    meta = read_inbreast_csv()
    lookup = build_inbreast_lookup(meta)

    rows = []

    for row in d.itertuples(index=False):
        r = row._asdict()

        source_path = safe_str(r.get("source_path", ""))
        processed_path = safe_str(r.get("processed_path", ""))

        parsed = parse_inbreast_filename(source_path)
        meta_row = lookup.get(parsed["file_id"], {})

        laterality = normalize_laterality(meta_row.get("Laterality", "")) or parsed["laterality"]
        view = normalize_view(meta_row.get("View", "")) or parsed["view"]

        birads = safe_str(meta_row.get("Bi-Rads", ""))
        label = birads_to_binary_label(birads)

        patient_id = parsed["patient_id"]
        exam_id = parsed["exam_id"]

        status = inbreast_status(patient_id, laterality, view)

        notes = ""
        if status == "OK_NONSTANDARD_VIEW":
            notes = "FB is retained as a valid non-standard view but excluded from CC/MLO pairing."

        rows.append({
            "dataset": "INbreast",
            "patient_id": patient_id,
            "patient_original_id": safe_str(meta_row.get("Patient ID", parsed["exam_hash"])),
            "study_id": exam_id,
            "series_id": "",
            "exam_id": exam_id,
            "image_id": parsed["file_id"],
            "file_id": parsed["file_id"],
            "exam_hash": parsed["exam_hash"],
            "split": "External",
            "laterality": laterality,
            "view": view,
            "view_position": view,
            "label": label,
            "label_text": binary_label_text(label),
            "breast_birads": normalize_birads_text(birads),
            "breast_density": "",
            "finding_birads": "",
            "finding_categories": "",
            "n_findings": 0,
            "has_finding_annotation": False,
            "finding_boxes_json": "[]",
            "lesion_type": "UnknownLesion",
            "source_path": source_path,
            "processed_path": processed_path,
            "case_folder": safe_str(r.get("case_folder", "")),
            "study_date": safe_str(meta_row.get("Acquisition date", r.get("study_date", ""))),
            "patient_age": safe_str(meta_row.get("Patient age", "")),
            "height": "",
            "width": "",
            "manufacturer": safe_str(r.get("manufacturer", "")),
            "metadata_status": status,
            "metadata_source": "INbreast.csv + filename",
            "notes": notes,
        })

    return pd.DataFrame(rows)


# =============================================================================
# VinDr Harmonization
# =============================================================================

def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def read_vindr_tables() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    breast = read_csv_required(VINDR_ROOT / "breast-level_annotations.csv")
    finding = read_csv_required(VINDR_ROOT / "finding_annotations.csv")
    metadata = read_csv_required(VINDR_ROOT / "metadata.csv")

    breast.columns = [safe_str(c) for c in breast.columns]
    finding.columns = [safe_str(c) for c in finding.columns]
    metadata.columns = [safe_str(c) for c in metadata.columns]

    return breast, finding, metadata


def normalize_vindr_breast_table(breast: pd.DataFrame) -> pd.DataFrame:
    df = breast.copy()

    df["image_id"] = df["image_id"].astype(str).str.strip()
    df["study_id"] = df["study_id"].astype(str).str.strip()
    df["series_id"] = df["series_id"].astype(str).str.strip()

    df["laterality"] = df["laterality"].map(normalize_laterality)
    df["view"] = df["view_position"].map(normalize_view)
    df["view_position"] = df["view"]

    df["breast_birads"] = df["breast_birads"].map(normalize_birads_text)
    df["label"] = df["breast_birads"].map(birads_to_binary_label)
    df["label_text"] = df["label"].map(binary_label_text)

    df["breast_density"] = df["breast_density"].map(density_to_text)

    return df


def normalize_vindr_metadata_table(metadata: pd.DataFrame) -> pd.DataFrame:
    df = metadata.copy()

    # VinDr metadata uses DICOM-style column names.
    rename_map = {
        "SOP Instance UID": "image_id",
        "Series Instance UID": "series_id",
        "Patient's Age": "patient_age",
        "View Position": "metadata_view_position",
        "Image Laterality": "metadata_laterality",
        "Rows": "metadata_rows",
        "Columns": "metadata_columns",
        "Manufacturer": "manufacturer",
        "Manufacturer's Model Name": "manufacturer_model",
        "Photometric Interpretation": "photometric_interpretation",
        "Imager Pixel Spacing": "imager_pixel_spacing",
        "Pixel Spacing": "pixel_spacing",
    }

    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    if "image_id" in df.columns:
        df["image_id"] = df["image_id"].astype(str).str.strip()

    if "metadata_laterality" in df.columns:
        df["metadata_laterality"] = df["metadata_laterality"].map(normalize_laterality)

    if "metadata_view_position" in df.columns:
        df["metadata_view_position"] = df["metadata_view_position"].map(normalize_view)

    return df


def build_vindr_finding_summary(finding: pd.DataFrame) -> pd.DataFrame:
    if finding.empty:
        return pd.DataFrame(columns=[
            "image_id", "n_findings", "finding_categories",
            "finding_birads", "has_finding_annotation", "finding_boxes_json"
        ])

    df = finding.copy()

    df["image_id"] = df["image_id"].astype(str).str.strip()

    if "finding_categories" not in df.columns:
        df["finding_categories"] = ""

    if "finding_birads" not in df.columns:
        df["finding_birads"] = ""

    box_cols = ["xmin", "ymin", "xmax", "ymax"]
    for c in box_cols:
        if c not in df.columns:
            df[c] = None

    rows = []

    for image_id, g in df.groupby("image_id"):
        categories = sorted(set(safe_str(x) for x in g["finding_categories"].tolist() if safe_str(x)))
        birads = sorted(set(normalize_birads_text(x) for x in g["finding_birads"].tolist() if safe_str(x)))

        boxes = []
        for rr in g.itertuples(index=False):
            d = rr._asdict()
            boxes.append({
                "xmin": d.get("xmin"),
                "ymin": d.get("ymin"),
                "xmax": d.get("xmax"),
                "ymax": d.get("ymax"),
                "finding_categories": safe_str(d.get("finding_categories", "")),
                "finding_birads": normalize_birads_text(d.get("finding_birads", "")),
            })

        rows.append({
            "image_id": image_id,
            "n_findings": int(len(g)),
            "finding_categories": " | ".join(categories),
            "finding_birads": " | ".join(birads),
            "has_finding_annotation": True,
            "finding_boxes_json": json.dumps(boxes, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def parse_vindr_from_path(source_path: str) -> Dict[str, str]:
    p = Path(safe_str(source_path))
    image_id = p.stem
    case_folder = p.parent.name

    return {
        "image_id": image_id,
        "case_folder": case_folder,
        "fallback_patient_id": f"VINDR_{case_folder}",
        "fallback_exam_id": f"VINDR_{case_folder}",
    }


def harmonize_vindr(stage1a_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = stage1a_df[stage1a_df["dataset"] == "VinDr-Mammo"].copy()

    breast, finding, metadata = read_vindr_tables()

    breast = normalize_vindr_breast_table(breast)
    metadata = normalize_vindr_metadata_table(metadata)
    finding_summary = build_vindr_finding_summary(finding)

    finding_summary.to_csv(OUTPUT_VINDR_FINDINGS_CSV, index=False, encoding="utf-8-sig")

    merged = breast.merge(
        metadata,
        on="image_id",
        how="left",
        suffixes=("", "_metadata")
    )

    merged = merged.merge(
        finding_summary,
        on="image_id",
        how="left"
    )

    merged["n_findings"] = merged["n_findings"].fillna(0).astype(int)
    merged["finding_categories"] = merged["finding_categories"].fillna("")
    merged["finding_birads"] = merged["finding_birads"].fillna("")
    merged["has_finding_annotation"] = merged["has_finding_annotation"].fillna(False)
    merged["finding_boxes_json"] = merged["finding_boxes_json"].fillna("[]")

    lookup = {
        safe_str(row["image_id"]): row
        for _, row in merged.iterrows()
    }

    rows = []
    unresolved = []

    for row in d.itertuples(index=False):
        r = row._asdict()

        source_path = safe_str(r.get("source_path", ""))
        processed_path = safe_str(r.get("processed_path", ""))

        parsed = parse_vindr_from_path(source_path)
        image_id = parsed["image_id"]

        m = lookup.get(image_id)

        if m is None:
            status = "PARTIAL"
            unresolved.append({
                "image_id": image_id,
                "source_path": source_path,
                "reason": "Image ID not found in VinDr breast-level metadata."
            })

            rows.append({
                "dataset": "VinDr-Mammo",
                "patient_id": parsed["fallback_patient_id"],
                "patient_original_id": parsed["case_folder"],
                "study_id": "",
                "series_id": "",
                "exam_id": parsed["fallback_exam_id"],
                "image_id": image_id,
                "file_id": image_id,
                "exam_hash": parsed["case_folder"],
                "split": "External",
                "laterality": "",
                "view": "",
                "view_position": "",
                "label": None,
                "label_text": "Unknown",
                "breast_birads": "Unknown",
                "breast_density": "",
                "finding_birads": "",
                "finding_categories": "",
                "n_findings": 0,
                "has_finding_annotation": False,
                "finding_boxes_json": "[]",
                "lesion_type": "UnknownLesion",
                "source_path": source_path,
                "processed_path": processed_path,
                "case_folder": parsed["case_folder"],
                "study_date": "",
                "patient_age": "",
                "height": "",
                "width": "",
                "manufacturer": "",
                "metadata_status": status,
                "metadata_source": "VinDr metadata not matched",
                "notes": "Image ID not found in breast-level_annotations.csv",
            })
            continue

        study_id = safe_str(m.get("study_id", ""))
        series_id = safe_str(m.get("series_id", ""))

        laterality = safe_str(m.get("laterality", ""))
        view = safe_str(m.get("view", ""))
        label = m.get("label", None)

        status = "OK" if study_id and image_id and laterality and view and pd.notna(label) else "PARTIAL"

        patient_id = f"VINDR_{study_id}"
        exam_id = f"VINDR_{study_id}"

        rows.append({
            "dataset": "VinDr-Mammo",
            "patient_id": patient_id,
            "patient_original_id": study_id,
            "study_id": study_id,
            "series_id": series_id,
            "exam_id": exam_id,
            "image_id": image_id,
            "file_id": image_id,
            "exam_hash": parsed["case_folder"],
            "split": safe_str(m.get("split", "External")),
            "laterality": laterality,
            "view": view,
            "view_position": view,
            "label": int(label) if pd.notna(label) else None,
            "label_text": safe_str(m.get("label_text", binary_label_text(label))),
            "breast_birads": safe_str(m.get("breast_birads", "Unknown")),
            "breast_density": safe_str(m.get("breast_density", "")),
            "finding_birads": safe_str(m.get("finding_birads", "")),
            "finding_categories": safe_str(m.get("finding_categories", "")),
            "n_findings": int(m.get("n_findings", 0)),
            "has_finding_annotation": bool(m.get("has_finding_annotation", False)),
            "finding_boxes_json": safe_str(m.get("finding_boxes_json", "[]")),
            "lesion_type": safe_str(m.get("finding_categories", "")) if safe_str(m.get("finding_categories", "")) else "NoFindingOrBreastLevelOnly",
            "source_path": source_path,
            "processed_path": processed_path,
            "case_folder": parsed["case_folder"],
            "study_date": "",
            "patient_age": safe_str(m.get("patient_age", "")),
            "height": safe_str(m.get("height", m.get("metadata_rows", ""))),
            "width": safe_str(m.get("width", m.get("metadata_columns", ""))),
            "manufacturer": safe_str(m.get("manufacturer", "")),
            "metadata_status": status,
            "metadata_source": "VinDr breast-level + finding-level + DICOM metadata",
            "notes": "",
        })

    return pd.DataFrame(rows), pd.DataFrame(unresolved)


# =============================================================================
# Summary and Report
# =============================================================================

def build_summary(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in ["INbreast", "VinDr-Mammo"]:
        d = harmonized_df[harmonized_df["dataset"] == dataset].copy()

        rows.append({
            "dataset": dataset,
            "records": int(len(d)),
            "metadata_ok": int((d["metadata_status"] == "OK").sum()) if not d.empty else 0,
            "metadata_ok_nonstandard_view": int((d["metadata_status"] == "OK_NONSTANDARD_VIEW").sum()) if not d.empty else 0,
            "metadata_partial": int((d["metadata_status"] == "PARTIAL").sum()) if not d.empty else 0,
            "unique_patients": int(d["patient_id"].nunique()) if not d.empty else 0,
            "unique_exams": int(d["exam_id"].nunique()) if not d.empty else 0,
            "known_view": int(d["view"].astype(str).str.len().gt(0).sum()) if not d.empty else 0,
            "cc_images": int((d["view"] == "CC").sum()) if not d.empty else 0,
            "mlo_images": int((d["view"] == "MLO").sum()) if not d.empty else 0,
            "fb_images": int((d["view"] == "FB").sum()) if not d.empty else 0,
            "known_laterality": int(d["laterality"].astype(str).str.len().gt(0).sum()) if not d.empty else 0,
            "left_images": int((d["laterality"] == "LEFT").sum()) if not d.empty else 0,
            "right_images": int((d["laterality"] == "RIGHT").sum()) if not d.empty else 0,
            "known_label": int(d["label"].notna().sum()) if not d.empty else 0,
            "label_0": int((d["label"] == 0).sum()) if not d.empty else 0,
            "label_1": int((d["label"] == 1).sum()) if not d.empty else 0,
            "known_density": int(d["breast_density"].astype(str).str.len().gt(0).sum()) if not d.empty else 0,
            "images_with_findings": int((d["n_findings"] > 0).sum()) if not d.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "records": int(len(harmonized_df)),
        "metadata_ok": int((harmonized_df["metadata_status"] == "OK").sum()),
        "metadata_ok_nonstandard_view": int((harmonized_df["metadata_status"] == "OK_NONSTANDARD_VIEW").sum()),
        "metadata_partial": int((harmonized_df["metadata_status"] == "PARTIAL").sum()),
        "unique_patients": int(harmonized_df["patient_id"].nunique()),
        "unique_exams": int(harmonized_df["exam_id"].nunique()),
        "known_view": int(harmonized_df["view"].astype(str).str.len().gt(0).sum()),
        "cc_images": int((harmonized_df["view"] == "CC").sum()),
        "mlo_images": int((harmonized_df["view"] == "MLO").sum()),
        "fb_images": int((harmonized_df["view"] == "FB").sum()),
        "known_laterality": int(harmonized_df["laterality"].astype(str).str.len().gt(0).sum()),
        "left_images": int((harmonized_df["laterality"] == "LEFT").sum()),
        "right_images": int((harmonized_df["laterality"] == "RIGHT").sum()),
        "known_label": int(harmonized_df["label"].notna().sum()),
        "label_0": int((harmonized_df["label"] == 0).sum()),
        "label_1": int((harmonized_df["label"] == 1).sum()),
        "known_density": int(harmonized_df["breast_density"].astype(str).str.len().gt(0).sum()),
        "images_with_findings": int((harmonized_df["n_findings"] > 0).sum()),
    })

    return pd.DataFrame(rows)


def write_dataset_outputs(harmonized_df: pd.DataFrame) -> None:
    for dataset in ["INbreast", "VinDr-Mammo"]:
        out_dir = OUTPUT_HARMONIZED_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        d = harmonized_df[harmonized_df["dataset"] == dataset].copy()
        d.to_csv(out_dir / "harmonized_metadata.csv", index=False, encoding="utf-8-sig")


def write_report(summary_df: pd.DataFrame, unresolved_df: pd.DataFrame, nonstandard_df: pd.DataFrame) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1A2 v2 METADATA HARMONIZATION REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("NONSTANDARD VIEW RECORDS")
    lines.append("-" * 100)
    lines.append(nonstandard_df.to_string(index=False) if not nonstandard_df.empty else "No nonstandard records.")
    lines.append("")

    lines.append("UNRESOLVED / PARTIAL METADATA SAMPLE")
    lines.append("-" * 100)
    lines.append(unresolved_df.head(80).to_string(index=False) if not unresolved_df.empty else "No unresolved records.")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_GLOBAL_CSV))
    lines.append(str(OUTPUT_SUMMARY_CSV))
    lines.append(str(OUTPUT_UNRESOLVED_CSV))
    lines.append(str(OUTPUT_NONSTANDARD_CSV))
    lines.append(str(OUTPUT_VINDR_FINDINGS_CSV))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "input_stage1a_index": str(INPUT_STAGE1A_INDEX),
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1A2 v2 METADATA HARMONIZATION FOR INBREAST AND VINDR-MAMMO")
    print("=" * 100)
    print(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    print("-" * 100)

    if not INPUT_STAGE1A_INDEX.exists():
        raise FileNotFoundError(f"Stage1A index not found: {INPUT_STAGE1A_INDEX}")

    stage1a_df = pd.read_csv(INPUT_STAGE1A_INDEX)

    print("Harmonizing INbreast...")
    inbreast_df = harmonize_inbreast(stage1a_df)

    print("Harmonizing VinDr-Mammo with labels, density, and findings...")
    vindr_df, vindr_unresolved_df = harmonize_vindr(stage1a_df)

    harmonized_df = pd.concat([inbreast_df, vindr_df], ignore_index=True)

    unresolved_df = harmonized_df[harmonized_df["metadata_status"] == "PARTIAL"].copy()
    nonstandard_df = harmonized_df[harmonized_df["metadata_status"] == "OK_NONSTANDARD_VIEW"].copy()

    summary_df = build_summary(harmonized_df)

    write_dataset_outputs(harmonized_df)

    harmonized_df.to_csv(OUTPUT_GLOBAL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    unresolved_df.to_csv(OUTPUT_UNRESOLVED_CSV, index=False, encoding="utf-8-sig")
    nonstandard_df.to_csv(OUTPUT_NONSTANDARD_CSV, index=False, encoding="utf-8-sig")

    save_json(summary_df)
    write_report(summary_df, unresolved_df, nonstandard_df)

    print()
    print("STAGE 1A2 v2 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Harmonized metadata:       {OUTPUT_GLOBAL_CSV}")
    print(f"Summary:                   {OUTPUT_SUMMARY_CSV}")
    print(f"Unresolved records:        {OUTPUT_UNRESOLVED_CSV}")
    print(f"Nonstandard view records:  {OUTPUT_NONSTANDARD_CSV}")
    print(f"VinDr finding summary:     {OUTPUT_VINDR_FINDINGS_CSV}")
    print(f"JSON summary:              {OUTPUT_JSON}")
    print(f"Text report:               {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()