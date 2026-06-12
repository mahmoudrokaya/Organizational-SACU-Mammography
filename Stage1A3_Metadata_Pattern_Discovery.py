r"""
Stage1A3_Metadata_Pattern_Discovery.py

Purpose
-------
Discover actual filename, folder, and metadata patterns for INbreast and
VinDr-Mammo after Stage1A preprocessing.

Why this script is needed
-------------------------
Stage1A2 did not fully harmonize INbreast and VinDr-Mammo because the real
naming patterns were not completely captured. This script inspects source paths,
processed paths, stems, parent folders, DICOM metadata columns stored in
Stage1A index, and available metadata CSV files.

It saves pattern tables that will be used to write a correct:
Stage1A2_Metadata_Harmonization_INbreast_VinDr_v2.py

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1A3_Metadata_Pattern_Discovery.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
DATASET_ROOT = PROJECT_ROOT / "datasets"

INPUT_STAGE1A_INDEX = PROJECT_ROOT / "results" / "tables" / "Stage1A_Preprocessed_Image_Index.csv"

OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_SAMPLE_CSV = OUTPUT_TABLE_DIR / "Stage1A3_Metadata_Pattern_Samples.csv"
OUTPUT_TOKEN_FREQ_CSV = OUTPUT_TABLE_DIR / "Stage1A3_Token_Frequency.csv"
OUTPUT_PATTERN_COUNTS_CSV = OUTPUT_TABLE_DIR / "Stage1A3_Filename_Pattern_Counts.csv"
OUTPUT_DICOM_FIELD_CSV = OUTPUT_TABLE_DIR / "Stage1A3_DICOM_Field_Availability.csv"
OUTPUT_METADATA_CSV_PROFILE = OUTPUT_TABLE_DIR / "Stage1A3_Metadata_CSV_Profile.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1A3_Metadata_Pattern_Discovery.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1A3_Metadata_Pattern_Discovery_Report.txt"

TARGET_DATASETS = ["INbreast", "VinDr-Mammo"]
SAMPLE_PER_DATASET = 300


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


def split_path(path_text: str) -> List[str]:
    return [p for p in safe_str(path_text).replace("\\", "/").split("/") if p]


def file_stem(path_text: str) -> str:
    try:
        return Path(safe_str(path_text)).stem
    except Exception:
        text = safe_str(path_text).replace("\\", "/").split("/")[-1]
        return re.sub(r"\.[A-Za-z0-9]+$", "", text)


def parent_name(path_text: str, level: int = 1) -> str:
    try:
        p = Path(safe_str(path_text))
        for _ in range(level):
            p = p.parent
        return p.name
    except Exception:
        return ""


def tokenize(text: str) -> List[str]:
    text = safe_str(text)
    text = text.replace("\\", "/")
    raw = re.split(r"[/_\-\s\.]+", text)
    return [x.strip() for x in raw if x.strip()]


def normalize_token(token: str) -> str:
    t = safe_str(token).upper()
    if t in {"L", "LEFT"}:
        return "LEFT_TOKEN"
    if t in {"R", "RIGHT"}:
        return "RIGHT_TOKEN"
    if t in {"CC"}:
        return "CC_TOKEN"
    if t in {"MLO", "ML", "LMLO", "RMLO"}:
        return "MLO_OR_ML_TOKEN"
    if t in {"MG", "MAMMO", "MAMMOGRAPHY"}:
        return "MG_TOKEN"
    if t in {"ANON", "ANONYMOUS"}:
        return "ANON_TOKEN"
    if re.fullmatch(r"\d{4,}", t):
        return "LONG_NUMBER"
    if re.fullmatch(r"[0-9A-F]{8,}", t):
        return "HEX_LIKE"
    if re.fullmatch(r"[A-Za-z0-9]{12,}", token):
        return "ALNUM_LONG"
    return t


def infer_pattern_from_stem(stem: str) -> str:
    tokens = tokenize(stem)
    normalized = [normalize_token(t) for t in tokens]
    return "_".join(normalized)


def detect_laterality_candidates(text: str) -> str:
    tokens = [t.upper() for t in tokenize(text)]
    hits = []

    for t in tokens:
        if t in {"L", "LEFT"}:
            hits.append("LEFT")
        elif t in {"R", "RIGHT"}:
            hits.append("RIGHT")

    if "LEFT" in safe_str(text).upper():
        hits.append("LEFT_TEXT")
    if "RIGHT" in safe_str(text).upper():
        hits.append("RIGHT_TEXT")

    return "|".join(sorted(set(hits)))


def detect_view_candidates(text: str) -> str:
    tokens = [t.upper() for t in tokenize(text)]
    hits = []

    for t in tokens:
        if t == "CC":
            hits.append("CC")
        elif t in {"MLO", "ML"}:
            hits.append(t)

    return "|".join(sorted(set(hits)))


def detect_possible_label_tokens(text: str) -> str:
    t = safe_str(text).upper()
    hits = []

    for word in [
        "MALIGNANT", "BENIGN", "NORMAL", "CANCER",
        "BI-RADS", "BIRADS", "SUSPICIOUS", "MASS", "CALC"
    ]:
        if word in t:
            hits.append(word)

    return "|".join(hits)


def compact_path(path_text: str, keep_last: int = 5) -> str:
    parts = split_path(path_text)
    return "/".join(parts[-keep_last:])


# =============================================================================
# Stage1A Index Pattern Discovery
# =============================================================================

def build_sample_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in TARGET_DATASETS:
        d = df[df["dataset"] == dataset].copy()

        if d.empty:
            continue

        if len(d) > SAMPLE_PER_DATASET:
            d = d.sample(SAMPLE_PER_DATASET, random_state=42)

        for row in d.itertuples(index=False):
            r = row._asdict()

            source_path = safe_str(r.get("source_path", ""))
            processed_path = safe_str(r.get("processed_path", ""))
            case_folder = safe_str(r.get("case_folder", ""))

            source_stem = file_stem(source_path)
            processed_stem = file_stem(processed_path)

            combined = " ".join([
                source_path,
                processed_path,
                case_folder,
                source_stem,
                processed_stem,
            ])

            rows.append({
                "dataset": dataset,
                "source_path_compact": compact_path(source_path),
                "processed_path_compact": compact_path(processed_path),
                "case_folder": case_folder,
                "source_stem": source_stem,
                "processed_stem": processed_stem,
                "source_parent_1": parent_name(source_path, 1),
                "source_parent_2": parent_name(source_path, 2),
                "source_parent_3": parent_name(source_path, 3),
                "processed_parent_1": parent_name(processed_path, 1),
                "processed_parent_2": parent_name(processed_path, 2),
                "filename_pattern": infer_pattern_from_stem(source_stem),
                "laterality_candidates": detect_laterality_candidates(combined),
                "view_candidates": detect_view_candidates(combined),
                "label_candidates": detect_possible_label_tokens(combined),
                "stage1a_patient_id": safe_str(r.get("patient_id", "")),
                "stage1a_laterality": safe_str(r.get("laterality", "")),
                "stage1a_view": safe_str(r.get("view", "")),
                "study_date": safe_str(r.get("study_date", "")),
                "study_instance_uid": safe_str(r.get("study_instance_uid", "")),
                "series_instance_uid": safe_str(r.get("series_instance_uid", "")),
                "sop_instance_uid": safe_str(r.get("sop_instance_uid", "")),
            })

    return pd.DataFrame(rows)


def build_token_frequency(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in TARGET_DATASETS:
        d = df[df["dataset"] == dataset].copy()

        token_counts: Dict[str, int] = {}
        norm_counts: Dict[str, int] = {}

        for row in d.itertuples(index=False):
            r = row._asdict()
            text = " ".join([
                safe_str(r.get("source_path", "")),
                safe_str(r.get("processed_path", "")),
                safe_str(r.get("case_folder", "")),
            ])

            for token in tokenize(text):
                token_upper = token.upper()
                token_counts[token_upper] = token_counts.get(token_upper, 0) + 1

                norm = normalize_token(token)
                norm_counts[norm] = norm_counts.get(norm, 0) + 1

        for token, count in sorted(token_counts.items(), key=lambda x: x[1], reverse=True)[:300]:
            rows.append({
                "dataset": dataset,
                "token_type": "raw",
                "token": token,
                "count": count,
            })

        for token, count in sorted(norm_counts.items(), key=lambda x: x[1], reverse=True)[:300]:
            rows.append({
                "dataset": dataset,
                "token_type": "normalized",
                "token": token,
                "count": count,
            })

    return pd.DataFrame(rows)


def build_pattern_counts(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in TARGET_DATASETS:
        d = df[df["dataset"] == dataset].copy()

        pattern_counts: Dict[str, int] = {}

        for row in d.itertuples(index=False):
            r = row._asdict()
            stem = file_stem(safe_str(r.get("source_path", "")))
            pattern = infer_pattern_from_stem(stem)
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

        for pattern, count in sorted(pattern_counts.items(), key=lambda x: x[1], reverse=True):
            rows.append({
                "dataset": dataset,
                "filename_pattern": pattern,
                "count": count,
            })

    return pd.DataFrame(rows)


def build_dicom_field_availability(df: pd.DataFrame) -> pd.DataFrame:
    candidate_fields = [
        "patient_id",
        "view",
        "laterality",
        "study_date",
        "study_instance_uid",
        "series_instance_uid",
        "sop_instance_uid",
        "modality",
        "manufacturer",
        "case_folder",
        "source_path",
        "processed_path",
    ]

    rows = []

    for dataset in TARGET_DATASETS:
        d = df[df["dataset"] == dataset].copy()

        for field in candidate_fields:
            if field not in d.columns:
                rows.append({
                    "dataset": dataset,
                    "field": field,
                    "exists": False,
                    "nonempty_count": 0,
                    "unique_nonempty": 0,
                    "sample_values": "",
                })
                continue

            s = d[field].astype(str).fillna("").map(lambda x: x.strip())
            nonempty = s[s != ""]

            samples = nonempty.drop_duplicates().head(10).tolist()

            rows.append({
                "dataset": dataset,
                "field": field,
                "exists": True,
                "nonempty_count": int(len(nonempty)),
                "unique_nonempty": int(nonempty.nunique()),
                "sample_values": " | ".join(samples),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Metadata CSV Profile
# =============================================================================

def read_csv_flexible(path: Path) -> Tuple[pd.DataFrame, str]:
    attempts = [
        ("comma", {"sep": ","}),
        ("semicolon", {"sep": ";"}),
        ("tab", {"sep": "\t"}),
        ("pipe", {"sep": "|"}),
    ]

    for name, kwargs in attempts:
        try:
            df = pd.read_csv(path, **kwargs)
            if df.shape[1] > 1:
                return df, name
        except Exception:
            pass

    try:
        raw = pd.read_csv(path, header=None)
        return raw, "raw_single_column"
    except Exception:
        return pd.DataFrame(), "failed"


def profile_metadata_csvs() -> pd.DataFrame:
    rows = []

    for dataset in TARGET_DATASETS:
        root = DATASET_ROOT / dataset
        csv_files = sorted(root.rglob("*.csv")) if root.exists() else []

        if not csv_files:
            rows.append({
                "dataset": dataset,
                "csv_path": "",
                "parse_mode": "",
                "rows": 0,
                "columns": 0,
                "column_names": "",
                "first_row": "",
                "contains_laterality_terms": False,
                "contains_view_terms": False,
                "contains_label_terms": False,
            })
            continue

        for csv_path in csv_files:
            df, mode = read_csv_flexible(csv_path)

            if df.empty:
                rows.append({
                    "dataset": dataset,
                    "csv_path": str(csv_path),
                    "parse_mode": mode,
                    "rows": 0,
                    "columns": 0,
                    "column_names": "",
                    "first_row": "",
                    "contains_laterality_terms": False,
                    "contains_view_terms": False,
                    "contains_label_terms": False,
                })
                continue

            text_blob = " ".join(df.astype(str).head(200).fillna("").values.reshape(-1).tolist()).upper()

            rows.append({
                "dataset": dataset,
                "csv_path": str(csv_path),
                "parse_mode": mode,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "column_names": " | ".join([str(c) for c in df.columns[:50]]),
                "first_row": " | ".join(df.iloc[0].astype(str).tolist()[:50]) if len(df) else "",
                "contains_laterality_terms": any(x in text_blob for x in ["LEFT", "RIGHT", "_L_", "_R_", ";L;", ";R;"]),
                "contains_view_terms": any(x in text_blob for x in ["CC", "MLO", " ML ", "_ML_", ";ML;", ";MLO;"]),
                "contains_label_terms": any(x in text_blob for x in ["BENIGN", "MALIGNANT", "BI-RADS", "BIRADS", "NORMAL"]),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Report
# =============================================================================

def write_report(
    sample_df: pd.DataFrame,
    token_df: pd.DataFrame,
    pattern_df: pd.DataFrame,
    dicom_field_df: pd.DataFrame,
    csv_profile_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1A3 METADATA PATTERN DISCOVERY REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    lines.append("")

    lines.append("FILENAME PATTERN COUNTS")
    lines.append("-" * 100)
    if pattern_df.empty:
        lines.append("No patterns found.")
    else:
        lines.append(pattern_df.head(80).to_string(index=False))
    lines.append("")

    lines.append("DICOM / INDEX FIELD AVAILABILITY")
    lines.append("-" * 100)
    if dicom_field_df.empty:
        lines.append("No field availability data.")
    else:
        lines.append(dicom_field_df.to_string(index=False))
    lines.append("")

    lines.append("METADATA CSV PROFILE")
    lines.append("-" * 100)
    if csv_profile_df.empty:
        lines.append("No CSV metadata profiles.")
    else:
        lines.append(csv_profile_df.to_string(index=False))
    lines.append("")

    lines.append("SAMPLE RECORDS")
    lines.append("-" * 100)
    if sample_df.empty:
        lines.append("No sample records.")
    else:
        lines.append(sample_df.head(100).to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_SAMPLE_CSV))
    lines.append(str(OUTPUT_TOKEN_FREQ_CSV))
    lines.append(str(OUTPUT_PATTERN_COUNTS_CSV))
    lines.append(str(OUTPUT_DICOM_FIELD_CSV))
    lines.append(str(OUTPUT_METADATA_CSV_PROFILE))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_summary(
    pattern_df: pd.DataFrame,
    dicom_field_df: pd.DataFrame,
    csv_profile_df: pd.DataFrame,
) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "input_stage1a_index": str(INPUT_STAGE1A_INDEX),
        "top_filename_patterns": pattern_df.head(50).to_dict(orient="records"),
        "dicom_field_availability": dicom_field_df.to_dict(orient="records"),
        "metadata_csv_profile": csv_profile_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1A3 METADATA PATTERN DISCOVERY")
    print("=" * 100)
    print(f"Input Stage1A index: {INPUT_STAGE1A_INDEX}")
    print("-" * 100)

    if not INPUT_STAGE1A_INDEX.exists():
        raise FileNotFoundError(f"Stage1A index not found: {INPUT_STAGE1A_INDEX}")

    df = pd.read_csv(INPUT_STAGE1A_INDEX)

    print("Building sample table...")
    sample_df = build_sample_table(df)

    print("Building token frequency table...")
    token_df = build_token_frequency(df)

    print("Building filename pattern counts...")
    pattern_df = build_pattern_counts(df)

    print("Checking DICOM/index field availability...")
    dicom_field_df = build_dicom_field_availability(df)

    print("Profiling metadata CSV files...")
    csv_profile_df = profile_metadata_csvs()

    sample_df.to_csv(OUTPUT_SAMPLE_CSV, index=False, encoding="utf-8-sig")
    token_df.to_csv(OUTPUT_TOKEN_FREQ_CSV, index=False, encoding="utf-8-sig")
    pattern_df.to_csv(OUTPUT_PATTERN_COUNTS_CSV, index=False, encoding="utf-8-sig")
    dicom_field_df.to_csv(OUTPUT_DICOM_FIELD_CSV, index=False, encoding="utf-8-sig")
    csv_profile_df.to_csv(OUTPUT_METADATA_CSV_PROFILE, index=False, encoding="utf-8-sig")

    save_json_summary(pattern_df, dicom_field_df, csv_profile_df)
    write_report(
        sample_df=sample_df,
        token_df=token_df,
        pattern_df=pattern_df,
        dicom_field_df=dicom_field_df,
        csv_profile_df=csv_profile_df,
    )

    print()
    print("STAGE 1A3 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Sample table:          {OUTPUT_SAMPLE_CSV}")
    print(f"Token frequencies:     {OUTPUT_TOKEN_FREQ_CSV}")
    print(f"Pattern counts:        {OUTPUT_PATTERN_COUNTS_CSV}")
    print(f"DICOM field check:     {OUTPUT_DICOM_FIELD_CSV}")
    print(f"Metadata CSV profile:  {OUTPUT_METADATA_CSV_PROFILE}")
    print(f"JSON summary:          {OUTPUT_JSON}")
    print(f"Text report:           {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()