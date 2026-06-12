r"""
Stage1A2_Metadata_Harmonization_INbreast_VinDr.py

Purpose
-------
Create harmonized metadata records for INbreast and VinDr-Mammo so that
Stage1B can build multi-view CC/MLO pairs for all datasets.

Why this script is needed
-------------------------
Stage1A successfully preprocesses INbreast and VinDr-Mammo images, but Stage1B
cannot form multi-view records unless patient_id, laterality, view, split, label,
and processed_path are consistently available.

This script:
1. Loads Stage1A_Preprocessed_Image_Index.csv.
2. Extracts INbreast metadata from:
   - DICOM metadata already stored in Stage1A index when available
   - image filename/path patterns
   - INbreast.csv when possible
3. Extracts VinDr-Mammo metadata from:
   - processed/source filename patterns
   - folder structure
   - optional CSV metadata if later added
4. Creates a harmonized metadata table with unified columns.
5. Saves dataset-specific harmonized records for INbreast and VinDr-Mammo.
6. Saves a combined harmonized metadata file for Stage1B_v2.

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1A2_Metadata_Harmonization_INbreast_VinDr.py

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\preprocessing\\metadata_harmonized\\INbreast\\harmonized_metadata.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\preprocessing\\metadata_harmonized\\VinDr-Mammo\\harmonized_metadata.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1A2_Harmonized_Metadata_INbreast_VinDr.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1A2_Harmonization_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage1A2_Metadata_Harmonization_Report.txt
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

OUTPUT_HARMONIZED_ROOT = PROJECT_ROOT / "preprocessing" / "metadata_harmonized"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_HARMONIZED_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GLOBAL_CSV = OUTPUT_TABLE_DIR / "Stage1A2_Harmonized_Metadata_INbreast_VinDr.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "Stage1A2_Harmonization_Summary.csv"
OUTPUT_UNRESOLVED_CSV = OUTPUT_TABLE_DIR / "Stage1A2_Unresolved_Metadata_Records.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1A2_Harmonization_Summary.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1A2_Metadata_Harmonization_Report.txt"

TARGET_DATASETS = ["INbreast", "VinDr-Mammo"]


# =============================================================================
# Utility Functions
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


def normalize_laterality(value: str) -> str:
    v = safe_str(value).upper()

    if v in {"L", "LEFT"}:
        return "LEFT"
    if v in {"R", "RIGHT"}:
        return "RIGHT"

    if "LEFT" in v or re.search(r"(^|[_\-\s])L([_\-\s]|$)", v):
        return "LEFT"
    if "RIGHT" in v or re.search(r"(^|[_\-\s])R([_\-\s]|$)", v):
        return "RIGHT"

    return ""


def normalize_view(value: str) -> str:
    v = safe_str(value).upper()

    if v in {"CC", "MLO"}:
        return v

    if re.search(r"(^|[_\-\s])CC([_\-\s]|$)", v):
        return "CC"

    if re.search(r"(^|[_\-\s])MLO([_\-\s]|$)", v):
        return "MLO"

    return ""


def normalize_label(value: str) -> str:
    v = safe_str(value).upper()

    if v in {"1", "MALIGNANT", "MALIGNANCY", "CANCER", "POSITIVE"}:
        return "Malignant"

    if v in {"0", "BENIGN", "NORMAL", "NEGATIVE"}:
        return "Benign"

    if "MALIGNANT" in v or "CANCER" in v:
        return "Malignant"

    if "BENIGN" in v or "NORMAL" in v:
        return "Benign"

    return ""


def binary_label(label_text: str) -> Optional[int]:
    label = normalize_label(label_text)
    if label == "Malignant":
        return 1
    if label == "Benign":
        return 0
    return None


def infer_split_from_path(path_text: str) -> str:
    p = safe_str(path_text).upper()

    if "TRAIN" in p or "TRAINING" in p:
        return "Training"
    if "TEST" in p or "VALID" in p or "VAL" in p:
        return "Test"

    return "UnknownSplit"


def clean_id(text: str) -> str:
    text = safe_str(text)
    text = text.replace("\\", "/")
    text = text.strip("/").split("/")[-1]
    text = re.sub(r"\.[A-Za-z0-9]+$", "", text)
    return text


def path_tokens(path_text: str) -> List[str]:
    p = safe_str(path_text)
    p = p.replace("\\", "/")
    tokens = []
    for part in p.split("/"):
        for sub in re.split(r"[_\-\s]+", part):
            sub = sub.strip()
            if sub:
                tokens.append(sub)
    return tokens


def find_first_matching_token(tokens: List[str], pattern: str) -> str:
    rgx = re.compile(pattern, flags=re.IGNORECASE)
    for token in tokens:
        if rgx.search(token):
            return token
    return ""


# =============================================================================
# INbreast Metadata Handling
# =============================================================================

def load_inbreast_csv() -> pd.DataFrame:
    """
    INbreast.csv is sometimes semicolon-separated or stored as one column.
    This function tries several parsing strategies.
    """
    candidates = sorted((DATASET_ROOT / "INbreast").rglob("*.csv"))

    if not candidates:
        return pd.DataFrame()

    csv_path = candidates[0]

    attempts = [
        {"sep": ","},
        {"sep": ";"},
        {"sep": "\t"},
    ]

    for kwargs in attempts:
        try:
            df = pd.read_csv(csv_path, **kwargs)
            if df.shape[1] > 1:
                df["__metadata_file__"] = str(csv_path)
                return df
        except Exception:
            pass

    # Last resort: read as raw lines and split by semicolon if possible.
    try:
        raw = pd.read_csv(csv_path, header=None)
        if raw.shape[1] == 1:
            expanded = raw[0].astype(str).str.split(";", expand=True)
            expanded["__metadata_file__"] = str(csv_path)
            return expanded
    except Exception:
        pass

    return pd.DataFrame()


def infer_inbreast_patient_id(row: pd.Series) -> str:
    source_path = safe_str(row.get("source_path", ""))
    processed_path = safe_str(row.get("processed_path", ""))
    dicom_pid = safe_str(row.get("patient_id_dicom", ""))
    case_folder = safe_str(row.get("case_folder", ""))

    if dicom_pid:
        return f"INB_{dicom_pid}"

    tokens = path_tokens(source_path + " " + processed_path + " " + case_folder)

    # INbreast file identifiers are often numeric, e.g. 20586908.
    candidate = find_first_matching_token(tokens, r"^\d{6,}$")
    if candidate:
        return f"INB_{candidate}"

    stem = clean_id(source_path or processed_path)
    return f"INB_{stem}" if stem else "INB_UnknownPatient"


def infer_inbreast_view(row: pd.Series) -> str:
    # Try Stage1A/DICOM field first.
    for key in ["view", "view_position"]:
        v = normalize_view(safe_str(row.get(key, "")))
        if v:
            return v

    text = " ".join([
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ])
    return normalize_view(text)


def infer_inbreast_laterality(row: pd.Series) -> str:
    for key in ["laterality", "image_laterality"]:
        v = normalize_laterality(safe_str(row.get(key, "")))
        if v:
            return v

    text = " ".join([
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ])
    return normalize_laterality(text)


def infer_inbreast_label(row: pd.Series, inbreast_meta: pd.DataFrame) -> Tuple[str, Optional[int]]:
    """
    Conservative label extraction.
    If metadata columns cannot be interpreted, leave label unknown.
    """
    text_fields = [
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ]

    combined = " ".join(text_fields)
    label = normalize_label(combined)
    if label:
        return label, binary_label(label)

    # Try metadata table by matching numeric image ID in path.
    source_text = safe_str(row.get("source_path", ""))
    tokens = path_tokens(source_text)
    numeric_tokens = [t for t in tokens if re.fullmatch(r"\d{6,}", t)]

    if not inbreast_meta.empty and numeric_tokens:
        meta_text = inbreast_meta.astype(str)
        for token in numeric_tokens:
            mask = meta_text.apply(lambda col: col.str.contains(token, case=False, na=False)).any(axis=1)
            matched = inbreast_meta[mask]
            if not matched.empty:
                row_text = " ".join(matched.iloc[0].astype(str).tolist())
                label = normalize_label(row_text)
                if label:
                    return label, binary_label(label)

    return "Unknown", None


def harmonize_inbreast(stage1a_df: pd.DataFrame) -> pd.DataFrame:
    d = stage1a_df[stage1a_df["dataset"] == "INbreast"].copy()
    inbreast_meta = load_inbreast_csv()

    rows: List[Dict[str, object]] = []

    for r in d.itertuples(index=False):
        row = pd.Series(r._asdict())

        patient_id = infer_inbreast_patient_id(row)
        view = infer_inbreast_view(row)
        laterality = infer_inbreast_laterality(row)
        label_text, label_binary = infer_inbreast_label(row, inbreast_meta)
        split = "External"

        exam_id = patient_id

        rows.append({
            "dataset": "INbreast",
            "patient_id": patient_id,
            "exam_id": exam_id,
            "split": split,
            "laterality": laterality,
            "view": view,
            "label_text": label_text,
            "label": label_binary,
            "lesion_type": "UnknownLesion",
            "source_path": safe_str(row.get("source_path", "")),
            "processed_path": safe_str(row.get("processed_path", "")),
            "case_folder": safe_str(row.get("case_folder", "")),
            "study_date": safe_str(row.get("study_date", "")),
            "study_instance_uid": safe_str(row.get("study_instance_uid", "")),
            "series_instance_uid": safe_str(row.get("series_instance_uid", "")),
            "sop_instance_uid": safe_str(row.get("sop_instance_uid", "")),
            "metadata_status": "OK" if patient_id and view and laterality else "PARTIAL",
        })

    return pd.DataFrame(rows)


# =============================================================================
# VinDr-Mammo Metadata Handling
# =============================================================================

def find_vindr_metadata_files() -> List[Path]:
    root = DATASET_ROOT / "VinDr-Mammo"
    if not root.exists():
        return []
    return sorted(list(root.rglob("*.csv")))


def load_vindr_metadata() -> pd.DataFrame:
    """
    VinDr metadata may be absent in the current folder.
    If CSV files exist later, this function loads and combines them.
    """
    files = find_vindr_metadata_files()
    frames = []

    for f in files:
        try:
            df = pd.read_csv(f)
            df["__metadata_file__"] = str(f)
            frames.append(df)
        except Exception:
            pass

    if frames:
        return pd.concat(frames, ignore_index=True)

    return pd.DataFrame()


def infer_vindr_patient_id(row: pd.Series, vindr_meta: pd.DataFrame) -> str:
    source_path = safe_str(row.get("source_path", ""))
    processed_path = safe_str(row.get("processed_path", ""))
    case_folder = safe_str(row.get("case_folder", ""))

    tokens = path_tokens(source_path + " " + processed_path + " " + case_folder)

    # VinDr filenames are often image IDs. Use parent folder or stem.
    stem = clean_id(source_path or processed_path)
    parent = Path(source_path).parent.name if source_path else ""

    # Prefer folder if it looks like a study/patient folder.
    if parent and parent.lower() not in {"images", "png", "vindr-mammo"}:
        return f"VINDR_{parent}"

    if stem:
        return f"VINDR_{stem}"

    return "VINDR_UnknownPatient"


def infer_vindr_view(row: pd.Series, vindr_meta: pd.DataFrame) -> str:
    text = " ".join([
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ])

    v = normalize_view(text)
    if v:
        return v

    # Try metadata match by image_id/stem if metadata exists.
    if not vindr_meta.empty:
        stem = clean_id(safe_str(row.get("source_path", "")))
        meta_cols = {c.lower(): c for c in vindr_meta.columns}

        id_cols = [c for c in vindr_meta.columns if c.lower() in {"image_id", "imageid", "file_name", "filename"}]
        view_cols = [c for c in vindr_meta.columns if "view" in c.lower()]

        if id_cols and view_cols and stem:
            id_col = id_cols[0]
            view_col = view_cols[0]
            matched = vindr_meta[vindr_meta[id_col].astype(str).str.contains(stem, case=False, na=False)]
            if not matched.empty:
                return normalize_view(str(matched.iloc[0][view_col]))

    return ""


def infer_vindr_laterality(row: pd.Series, vindr_meta: pd.DataFrame) -> str:
    text = " ".join([
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ])

    v = normalize_laterality(text)
    if v:
        return v

    if not vindr_meta.empty:
        stem = clean_id(safe_str(row.get("source_path", "")))

        id_cols = [c for c in vindr_meta.columns if c.lower() in {"image_id", "imageid", "file_name", "filename"}]
        lat_cols = [c for c in vindr_meta.columns if any(k in c.lower() for k in ["laterality", "breast", "side"])]

        if id_cols and lat_cols and stem:
            id_col = id_cols[0]
            lat_col = lat_cols[0]
            matched = vindr_meta[vindr_meta[id_col].astype(str).str.contains(stem, case=False, na=False)]
            if not matched.empty:
                return normalize_laterality(str(matched.iloc[0][lat_col]))

    return ""


def infer_vindr_label(row: pd.Series, vindr_meta: pd.DataFrame) -> Tuple[str, Optional[int]]:
    text = " ".join([
        safe_str(row.get("source_path", "")),
        safe_str(row.get("processed_path", "")),
        safe_str(row.get("case_folder", "")),
    ])

    label = normalize_label(text)
    if label:
        return label, binary_label(label)

    if not vindr_meta.empty:
        stem = clean_id(safe_str(row.get("source_path", "")))

        id_cols = [c for c in vindr_meta.columns if c.lower() in {"image_id", "imageid", "file_name", "filename"}]
        label_cols = [
            c for c in vindr_meta.columns
            if any(k in c.lower() for k in ["label", "birads", "finding", "breast_birads"])
        ]

        if id_cols and label_cols and stem:
            id_col = id_cols[0]
            matched = vindr_meta[vindr_meta[id_col].astype(str).str.contains(stem, case=False, na=False)]
            if not matched.empty:
                row_text = " ".join(str(matched.iloc[0][c]) for c in label_cols)
                label = normalize_label(row_text)
                if label:
                    return label, binary_label(label)

                # BI-RADS handling: usually 1/2/3 benign or probably benign, 4/5 malignant/suspicious.
                birads_text = row_text.upper()
                if any(x in birads_text for x in ["BI-RADS 4", "BIRADS 4", "BI-RADS 5", "BIRADS 5"]):
                    return "Malignant", 1
                if any(x in birads_text for x in ["BI-RADS 1", "BI-RADS 2", "BI-RADS 3", "BIRADS 1", "BIRADS 2", "BIRADS 3"]):
                    return "Benign", 0

    return "Unknown", None


def harmonize_vindr(stage1a_df: pd.DataFrame) -> pd.DataFrame:
    d = stage1a_df[stage1a_df["dataset"] == "VinDr-Mammo"].copy()
    vindr_meta = load_vindr_metadata()

    rows: List[Dict[str, object]] = []

    for r in d.itertuples(index=False):
        row = pd.Series(r._asdict())

        patient_id = infer_vindr_patient_id(row, vindr_meta)
        view = infer_vindr_view(row, vindr_meta)
        laterality = infer_vindr_laterality(row, vindr_meta)
        label_text, label_binary = infer_vindr_label(row, vindr_meta)

        split = infer_split_from_path(
            safe_str(row.get("source_path", "")) + " " + safe_str(row.get("processed_path", ""))
        )
        if split == "UnknownSplit":
            split = "External"

        exam_id = patient_id

        rows.append({
            "dataset": "VinDr-Mammo",
            "patient_id": patient_id,
            "exam_id": exam_id,
            "split": split,
            "laterality": laterality,
            "view": view,
            "label_text": label_text,
            "label": label_binary,
            "lesion_type": "UnknownLesion",
            "source_path": safe_str(row.get("source_path", "")),
            "processed_path": safe_str(row.get("processed_path", "")),
            "case_folder": safe_str(row.get("case_folder", "")),
            "study_date": safe_str(row.get("study_date", "")),
            "study_instance_uid": safe_str(row.get("study_instance_uid", "")),
            "series_instance_uid": safe_str(row.get("series_instance_uid", "")),
            "sop_instance_uid": safe_str(row.get("sop_instance_uid", "")),
            "metadata_status": "OK" if patient_id and view and laterality else "PARTIAL",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Summary
# =============================================================================

def build_summary(harmonized_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in TARGET_DATASETS:
        d = harmonized_df[harmonized_df["dataset"] == dataset].copy()

        rows.append({
            "dataset": dataset,
            "records": int(len(d)),
            "metadata_ok": int((d["metadata_status"] == "OK").sum()) if not d.empty else 0,
            "metadata_partial": int((d["metadata_status"] == "PARTIAL").sum()) if not d.empty else 0,
            "unique_patients": int(d["patient_id"].nunique()) if not d.empty else 0,
            "known_view": int(d["view"].astype(str).str.len().gt(0).sum()) if not d.empty else 0,
            "cc_images": int((d["view"] == "CC").sum()) if not d.empty else 0,
            "mlo_images": int((d["view"] == "MLO").sum()) if not d.empty else 0,
            "known_laterality": int(d["laterality"].astype(str).str.len().gt(0).sum()) if not d.empty else 0,
            "left_images": int((d["laterality"] == "LEFT").sum()) if not d.empty else 0,
            "right_images": int((d["laterality"] == "RIGHT").sum()) if not d.empty else 0,
            "known_label": int(d["label"].notna().sum()) if not d.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "records": int(len(harmonized_df)),
        "metadata_ok": int((harmonized_df["metadata_status"] == "OK").sum()) if not harmonized_df.empty else 0,
        "metadata_partial": int((harmonized_df["metadata_status"] == "PARTIAL").sum()) if not harmonized_df.empty else 0,
        "unique_patients": int(harmonized_df["patient_id"].nunique()) if not harmonized_df.empty else 0,
        "known_view": int(harmonized_df["view"].astype(str).str.len().gt(0).sum()) if not harmonized_df.empty else 0,
        "cc_images": int((harmonized_df["view"] == "CC").sum()) if not harmonized_df.empty else 0,
        "mlo_images": int((harmonized_df["view"] == "MLO").sum()) if not harmonized_df.empty else 0,
        "known_laterality": int(harmonized_df["laterality"].astype(str).str.len().gt(0).sum()) if not harmonized_df.empty else 0,
        "left_images": int((harmonized_df["laterality"] == "LEFT").sum()) if not harmonized_df.empty else 0,
        "right_images": int((harmonized_df["laterality"] == "RIGHT").sum()) if not harmonized_df.empty else 0,
        "known_label": int(harmonized_df["label"].notna().sum()) if not harmonized_df.empty else 0,
    })

    return pd.DataFrame(rows)


def write_dataset_outputs(harmonized_df: pd.DataFrame) -> None:
    for dataset in TARGET_DATASETS:
        out_dir = OUTPUT_HARMONIZED_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        d = harmonized_df[harmonized_df["dataset"] == dataset].copy()
        d.to_csv(out_dir / "harmonized_metadata.csv", index=False, encoding="utf-8-sig")


def write_report(summary_df: pd.DataFrame, unresolved_df: pd.DataFrame) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1A2 METADATA HARMONIZATION REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("UNRESOLVED / PARTIAL METADATA SAMPLE")
    lines.append("-" * 100)
    if unresolved_df.empty:
        lines.append("No unresolved records.")
    else:
        lines.append(unresolved_df.head(50).to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_GLOBAL_CSV))
    lines.append(str(OUTPUT_SUMMARY_CSV))
    lines.append(str(OUTPUT_UNRESOLVED_CSV))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))
    lines.append(str(OUTPUT_HARMONIZED_ROOT / "INbreast" / "harmonized_metadata.csv"))
    lines.append(str(OUTPUT_HARMONIZED_ROOT / "VinDr-Mammo" / "harmonized_metadata.csv"))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_summary(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "input_stage1a_index": str(INPUT_STAGE1A_INDEX),
        "output_harmonized_root": str(OUTPUT_HARMONIZED_ROOT),
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1A2 METADATA HARMONIZATION FOR INBREAST AND VINDR-MAMMO")
    print("=" * 100)
    print(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    print("-" * 100)

    if not INPUT_STAGE1A_INDEX.exists():
        raise FileNotFoundError(f"Stage1A index not found: {INPUT_STAGE1A_INDEX}")

    stage1a_df = pd.read_csv(INPUT_STAGE1A_INDEX)

    print("Harmonizing INbreast...")
    inbreast_df = harmonize_inbreast(stage1a_df)

    print("Harmonizing VinDr-Mammo...")
    vindr_df = harmonize_vindr(stage1a_df)

    harmonized_df = pd.concat([inbreast_df, vindr_df], ignore_index=True)

    unresolved_df = harmonized_df[
        (harmonized_df["metadata_status"] != "OK")
        | (harmonized_df["view"].astype(str).str.len() == 0)
        | (harmonized_df["laterality"].astype(str).str.len() == 0)
    ].copy()

    summary_df = build_summary(harmonized_df)

    write_dataset_outputs(harmonized_df)

    harmonized_df.to_csv(OUTPUT_GLOBAL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    unresolved_df.to_csv(OUTPUT_UNRESOLVED_CSV, index=False, encoding="utf-8-sig")

    save_json_summary(summary_df)
    write_report(summary_df, unresolved_df)

    print()
    print("STAGE 1A2 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Harmonized metadata: {OUTPUT_GLOBAL_CSV}")
    print(f"Summary:             {OUTPUT_SUMMARY_CSV}")
    print(f"Unresolved records:  {OUTPUT_UNRESOLVED_CSV}")
    print(f"JSON summary:        {OUTPUT_JSON}")
    print(f"Text report:         {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()