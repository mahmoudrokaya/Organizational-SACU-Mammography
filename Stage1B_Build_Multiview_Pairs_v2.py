r"""
Stage1B_Build_Multiview_Pairs_v2.py

Purpose
-------
Build complete CC/MLO multi-view mammography records for:

1. CBIS-DDSM
   - Uses Stage1A_Preprocessed_Image_Index.csv
   - Re-parses patient/laterality/view/split/lesion type from CBIS case folders

2. INbreast
   - Uses Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv
   - Keeps only standard CC/MLO views
   - Excludes FB non-standard view from CC/MLO pairing

3. VinDr-Mammo
   - Uses Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv
   - Uses official study_id, image_id, laterality, view, BI-RADS label, density, split
   - Builds complete four-view exam-level records

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\preprocessing\\multiview_pairs_v2\\<dataset>\\multiview_exam_records.csv

D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B_v2_Global_Multiview_Exam_Records.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B_v2_Multiview_Pairing_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage1B_v2_Multiview_Pairing_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1B_Build_Multiview_Pairs_v2.py
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

STAGE1A_INDEX = PROJECT_ROOT / "results" / "tables" / "Stage1A_Preprocessed_Image_Index.csv"
STAGE1A2_HARMONIZED = PROJECT_ROOT / "results" / "tables" / "Stage1A2_v2_Harmonized_Metadata_INbreast_VinDr.csv"

OUTPUT_ROOT = PROJECT_ROOT / "preprocessing" / "multiview_pairs_v2"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_GLOBAL_CSV = OUTPUT_TABLE_DIR / "Stage1B_v2_Global_Multiview_Exam_Records.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "Stage1B_v2_Multiview_Pairing_Summary.csv"
OUTPUT_VIEW_LEVEL_CSV = OUTPUT_TABLE_DIR / "Stage1B_v2_View_Level_Input_Records.csv"
OUTPUT_DUPLICATES_CSV = OUTPUT_TABLE_DIR / "Stage1B_v2_Duplicate_View_Candidates.csv"
OUTPUT_INCOMPLETE_CSV = OUTPUT_TABLE_DIR / "Stage1B_v2_Incomplete_Multiview_Records.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1B_v2_Multiview_Pairing_Summary.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1B_v2_Multiview_Pairing_Report.txt"

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


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def normalize_laterality(value: str) -> str:
    v = safe_str(value).upper()
    if v in {"L", "LEFT"}:
        return "LEFT"
    if v in {"R", "RIGHT"}:
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
    if "MLO" in v:
        return "MLO"
    if re.search(r"(^|[_\-\s])ML([_\-\s]|$)", v):
        return "MLO"
    if re.search(r"(^|[_\-\s])CC([_\-\s]|$)", v):
        return "CC"
    return ""


def parse_cbis_case_folder(case_folder: str, fallback_patient: str = "") -> Dict[str, str]:
    text = safe_str(case_folder)

    patient_match = re.search(r"(P_\d+)", text, flags=re.IGNORECASE)
    patient_id = patient_match.group(1).upper() if patient_match else safe_str(fallback_patient)

    laterality = ""
    if "LEFT" in text.upper():
        laterality = "LEFT"
    elif "RIGHT" in text.upper():
        laterality = "RIGHT"

    view = normalize_view(text)

    split = "UnknownSplit"
    if "TRAINING" in text.upper():
        split = "Training"
    elif "TEST" in text.upper():
        split = "Test"

    lesion_type = "UnknownLesion"
    if text.upper().startswith("CALC"):
        lesion_type = "Calc"
    elif text.upper().startswith("MASS"):
        lesion_type = "Mass"

    return {
        "patient_id": patient_id if patient_id else "UnknownPatient",
        "exam_id": patient_id if patient_id else "UnknownExam",
        "study_id": "",
        "series_id": "",
        "laterality": laterality,
        "view": view,
        "split": split,
        "lesion_type": lesion_type,
    }


def choose_primary_record(df: pd.DataFrame) -> pd.Series:
    """
    If duplicate images exist for the same exam/laterality/view, choose the most reliable one.

    Priority:
    1. Larger crop area if available
    2. Larger original file size if available
    3. First row
    """
    d = df.copy()

    if "crop_w" in d.columns and "crop_h" in d.columns:
        d["crop_area"] = (
            pd.to_numeric(d["crop_w"], errors="coerce").fillna(0)
            * pd.to_numeric(d["crop_h"], errors="coerce").fillna(0)
        )
    else:
        d["crop_area"] = 0

    if "original_size_bytes" in d.columns:
        d["original_size_bytes_num"] = pd.to_numeric(
            d["original_size_bytes"],
            errors="coerce"
        ).fillna(0)
    else:
        d["original_size_bytes_num"] = 0

    d = d.sort_values(
        ["crop_area", "original_size_bytes_num"],
        ascending=[False, False]
    )

    return d.iloc[0]


# =============================================================================
# Load and Normalize Inputs
# =============================================================================

def load_stage1a() -> pd.DataFrame:
    if not STAGE1A_INDEX.exists():
        raise FileNotFoundError(f"Missing Stage1A index: {STAGE1A_INDEX}")
    return pd.read_csv(STAGE1A_INDEX)


def load_stage1a2() -> pd.DataFrame:
    if not STAGE1A2_HARMONIZED.exists():
        raise FileNotFoundError(f"Missing Stage1A2 harmonized metadata: {STAGE1A2_HARMONIZED}")
    return pd.read_csv(STAGE1A2_HARMONIZED)


def build_cbis_view_level(stage1a_df: pd.DataFrame) -> pd.DataFrame:
    d = stage1a_df[stage1a_df["dataset"] == "CBIS-DDSM"].copy()

    rows = []

    for row in d.itertuples(index=False):
        r = row._asdict()

        case_folder = safe_str(r.get("case_folder", ""))
        parsed = parse_cbis_case_folder(
            case_folder=case_folder,
            fallback_patient=safe_str(r.get("patient_id", ""))
        )

        view = parsed["view"]
        laterality = parsed["laterality"]

        if view not in {"CC", "MLO"}:
            continue

        if laterality not in {"LEFT", "RIGHT"}:
            continue

        patient_id = parsed["patient_id"]
        exam_id = f"CBIS_{patient_id}_{parsed['split']}_{parsed['lesion_type']}"

        rows.append({
            "dataset": "CBIS-DDSM",
            "patient_id": f"CBIS_{patient_id}",
            "patient_original_id": patient_id,
            "study_id": "",
            "series_id": safe_str(r.get("series_instance_uid", "")),
            "exam_id": exam_id,
            "image_id": safe_str(r.get("sop_instance_uid", "")) or Path(safe_str(r.get("source_path", ""))).stem,
            "file_id": Path(safe_str(r.get("source_path", ""))).stem,
            "split": parsed["split"],
            "laterality": laterality,
            "view": view,
            "view_position": view,
            "label": None,
            "label_text": "Unknown",
            "breast_birads": "",
            "breast_density": "",
            "finding_birads": "",
            "finding_categories": "",
            "n_findings": 0,
            "has_finding_annotation": False,
            "finding_boxes_json": "[]",
            "lesion_type": parsed["lesion_type"],
            "source_path": safe_str(r.get("source_path", "")),
            "processed_path": safe_str(r.get("processed_path", "")),
            "case_folder": case_folder,
            "study_date": safe_str(r.get("study_date", "")),
            "patient_age": "",
            "height": safe_str(r.get("target_height", "")),
            "width": safe_str(r.get("target_width", "")),
            "manufacturer": safe_str(r.get("manufacturer", "")),
            "metadata_status": "OK",
            "metadata_source": "CBIS-DDSM case folder + Stage1A",
            "notes": "",
            "crop_w": r.get("crop_w", ""),
            "crop_h": r.get("crop_h", ""),
            "original_size_bytes": r.get("original_size_bytes", ""),
        })

    return pd.DataFrame(rows)


def build_harmonized_view_level(stage1a2_df: pd.DataFrame) -> pd.DataFrame:
    d = stage1a2_df[stage1a2_df["dataset"].isin(["INbreast", "VinDr-Mammo"])].copy()

    # Keep only standard CC/MLO views for multiview construction.
    d["view"] = d["view"].map(normalize_view)
    d["laterality"] = d["laterality"].map(normalize_laterality)

    d = d[d["view"].isin(["CC", "MLO"])].copy()
    d = d[d["laterality"].isin(["LEFT", "RIGHT"])].copy()
    d = d[d["processed_path"].astype(str).str.len() > 0].copy()

    required_cols = [
        "dataset", "patient_id", "patient_original_id", "study_id", "series_id",
        "exam_id", "image_id", "file_id", "split", "laterality", "view",
        "view_position", "label", "label_text", "breast_birads",
        "breast_density", "finding_birads", "finding_categories",
        "n_findings", "has_finding_annotation", "finding_boxes_json",
        "lesion_type", "source_path", "processed_path", "case_folder",
        "study_date", "patient_age", "height", "width", "manufacturer",
        "metadata_status", "metadata_source", "notes"
    ]

    for col in required_cols:
        if col not in d.columns:
            d[col] = ""

    return d[required_cols].copy()


def build_all_view_level_records() -> pd.DataFrame:
    stage1a_df = load_stage1a()
    stage1a2_df = load_stage1a2()

    cbis_view = build_cbis_view_level(stage1a_df)
    harmonized_view = build_harmonized_view_level(stage1a2_df)

    all_view = pd.concat([cbis_view, harmonized_view], ignore_index=True)

    all_view["view_key"] = all_view["laterality"] + "_" + all_view["view"]

    return all_view


# =============================================================================
# Build Multiview Exam Records
# =============================================================================

def aggregate_exam_label(g: pd.DataFrame) -> Dict[str, object]:
    labels = pd.to_numeric(g["label"], errors="coerce").dropna().astype(int).tolist()

    if not labels:
        return {
            "exam_label": None,
            "exam_label_text": "Unknown",
            "n_labeled_views": 0,
            "n_positive_views": 0,
        }

    n_pos = int(sum(1 for x in labels if x == 1))
    exam_label = 1 if n_pos > 0 else 0

    return {
        "exam_label": exam_label,
        "exam_label_text": "Malignant_or_Suspicious" if exam_label == 1 else "Benign_or_Nonmalignant",
        "n_labeled_views": len(labels),
        "n_positive_views": n_pos,
    }


def aggregate_density(g: pd.DataFrame) -> str:
    vals = [
        safe_str(x)
        for x in g.get("breast_density", pd.Series(dtype=str)).tolist()
        if safe_str(x)
    ]
    if not vals:
        return ""
    return sorted(set(vals))[0]


def aggregate_birads(g: pd.DataFrame) -> str:
    vals = [
        safe_str(x)
        for x in g.get("breast_birads", pd.Series(dtype=str)).tolist()
        if safe_str(x)
    ]
    if not vals:
        return ""
    return " | ".join(sorted(set(vals)))


def aggregate_findings(g: pd.DataFrame) -> Dict[str, object]:
    categories = []
    birads = []
    boxes = []

    total_findings = 0

    for row in g.itertuples(index=False):
        r = row._asdict()

        cat = safe_str(r.get("finding_categories", ""))
        fb = safe_str(r.get("finding_birads", ""))

        if cat:
            categories.extend([x.strip() for x in cat.split("|") if x.strip()])
        if fb:
            birads.extend([x.strip() for x in fb.split("|") if x.strip()])

        total_findings += safe_int(r.get("n_findings", 0))

        box_json = safe_str(r.get("finding_boxes_json", "[]"))
        try:
            parsed = json.loads(box_json)
            if isinstance(parsed, list):
                for b in parsed:
                    b["image_id"] = safe_str(r.get("image_id", ""))
                    b["view_key"] = safe_str(r.get("view_key", ""))
                    boxes.append(b)
        except Exception:
            pass

    return {
        "exam_finding_categories": " | ".join(sorted(set(categories))),
        "exam_finding_birads": " | ".join(sorted(set(birads))),
        "exam_n_findings": int(total_findings),
        "exam_has_finding_annotation": bool(total_findings > 0),
        "exam_finding_boxes_json": json.dumps(boxes, ensure_ascii=False),
    }


def record_for_view(g: pd.DataFrame, laterality: str, view: str) -> Optional[pd.Series]:
    sub = g[(g["laterality"] == laterality) & (g["view"] == view)].copy()
    if sub.empty:
        return None
    return choose_primary_record(sub)


def build_exam_records(view_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    group_cols = ["dataset", "exam_id"]

    for (dataset, exam_id), g in view_df.groupby(group_cols, dropna=False):
        if safe_str(exam_id) == "":
            continue

        lcc = record_for_view(g, "LEFT", "CC")
        lmlo = record_for_view(g, "LEFT", "MLO")
        rcc = record_for_view(g, "RIGHT", "CC")
        rmlo = record_for_view(g, "RIGHT", "MLO")

        has_lcc = lcc is not None
        has_lmlo = lmlo is not None
        has_rcc = rcc is not None
        has_rmlo = rmlo is not None

        complete_left_multiview = has_lcc and has_lmlo
        complete_right_multiview = has_rcc and has_rmlo
        complete_bilateral_cc = has_lcc and has_rcc
        complete_bilateral_mlo = has_lmlo and has_rmlo
        complete_four_view = has_lcc and has_lmlo and has_rcc and has_rmlo

        label_info = aggregate_exam_label(g)
        finding_info = aggregate_findings(g)

        any_row = g.iloc[0]

        def get_from(row: Optional[pd.Series], field: str) -> str:
            if row is None:
                return ""
            return safe_str(row.get(field, ""))

        record_id = f"{dataset}__{safe_str(exam_id)}"

        rows.append({
            "record_id": record_id,
            "dataset": dataset,
            "patient_id": safe_str(any_row.get("patient_id", "")),
            "patient_original_id": safe_str(any_row.get("patient_original_id", "")),
            "study_id": safe_str(any_row.get("study_id", "")),
            "exam_id": safe_str(exam_id),
            "split": safe_str(any_row.get("split", "UnknownSplit")),

            "has_lcc": has_lcc,
            "has_lmlo": has_lmlo,
            "has_rcc": has_rcc,
            "has_rmlo": has_rmlo,

            "complete_left_multiview": complete_left_multiview,
            "complete_right_multiview": complete_right_multiview,
            "complete_any_unilateral_multiview": complete_left_multiview or complete_right_multiview,
            "complete_bilateral_cc": complete_bilateral_cc,
            "complete_bilateral_mlo": complete_bilateral_mlo,
            "complete_any_bilateral": complete_bilateral_cc or complete_bilateral_mlo,
            "complete_four_view": complete_four_view,

            "lcc_processed_path": get_from(lcc, "processed_path"),
            "lmlo_processed_path": get_from(lmlo, "processed_path"),
            "rcc_processed_path": get_from(rcc, "processed_path"),
            "rmlo_processed_path": get_from(rmlo, "processed_path"),

            "lcc_source_path": get_from(lcc, "source_path"),
            "lmlo_source_path": get_from(lmlo, "source_path"),
            "rcc_source_path": get_from(rcc, "source_path"),
            "rmlo_source_path": get_from(rmlo, "source_path"),

            "lcc_image_id": get_from(lcc, "image_id"),
            "lmlo_image_id": get_from(lmlo, "image_id"),
            "rcc_image_id": get_from(rcc, "image_id"),
            "rmlo_image_id": get_from(rmlo, "image_id"),

            "lcc_birads": get_from(lcc, "breast_birads"),
            "lmlo_birads": get_from(lmlo, "breast_birads"),
            "rcc_birads": get_from(rcc, "breast_birads"),
            "rmlo_birads": get_from(rmlo, "breast_birads"),

            "lcc_label": get_from(lcc, "label"),
            "lmlo_label": get_from(lmlo, "label"),
            "rcc_label": get_from(rcc, "label"),
            "rmlo_label": get_from(rmlo, "label"),

            "exam_label": label_info["exam_label"],
            "exam_label_text": label_info["exam_label_text"],
            "n_labeled_views": label_info["n_labeled_views"],
            "n_positive_views": label_info["n_positive_views"],

            "breast_density": aggregate_density(g),
            "exam_birads_summary": aggregate_birads(g),

            "exam_finding_categories": finding_info["exam_finding_categories"],
            "exam_finding_birads": finding_info["exam_finding_birads"],
            "exam_n_findings": finding_info["exam_n_findings"],
            "exam_has_finding_annotation": finding_info["exam_has_finding_annotation"],
            "exam_finding_boxes_json": finding_info["exam_finding_boxes_json"],

            "n_view_records": int(len(g)),
            "n_unique_views": int(g["view_key"].nunique()),
            "n_lcc_candidates": int(len(g[(g["laterality"] == "LEFT") & (g["view"] == "CC")])),
            "n_lmlo_candidates": int(len(g[(g["laterality"] == "LEFT") & (g["view"] == "MLO")])),
            "n_rcc_candidates": int(len(g[(g["laterality"] == "RIGHT") & (g["view"] == "CC")])),
            "n_rmlo_candidates": int(len(g[(g["laterality"] == "RIGHT") & (g["view"] == "MLO")])),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Audits
# =============================================================================

def find_duplicate_views(view_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["dataset", "exam_id", "laterality", "view"]

    counts = (
        view_df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_records")
    )

    dup_keys = counts[counts["n_records"] > 1]

    if dup_keys.empty:
        return pd.DataFrame()

    return view_df.merge(
        dup_keys[group_cols],
        on=group_cols,
        how="inner"
    ).sort_values(group_cols)


def find_incomplete_records(exam_df: pd.DataFrame) -> pd.DataFrame:
    if exam_df.empty:
        return pd.DataFrame()

    return exam_df[
        ~exam_df["complete_four_view"]
    ].copy()


def build_summary(exam_df: pd.DataFrame, view_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset in DATASETS:
        e = exam_df[exam_df["dataset"] == dataset].copy()
        v = view_df[view_df["dataset"] == dataset].copy()

        rows.append({
            "dataset": dataset,
            "view_level_records": int(len(v)),
            "exam_records": int(len(e)),
            "unique_patients": int(e["patient_id"].nunique()) if not e.empty else 0,
            "complete_four_view": int(e["complete_four_view"].sum()) if not e.empty else 0,
            "complete_any_unilateral_multiview": int(e["complete_any_unilateral_multiview"].sum()) if not e.empty else 0,
            "complete_any_bilateral": int(e["complete_any_bilateral"].sum()) if not e.empty else 0,
            "incomplete_records": int((~e["complete_four_view"]).sum()) if not e.empty else 0,
            "known_exam_labels": int(e["exam_label"].notna().sum()) if not e.empty else 0,
            "positive_exam_labels": int((e["exam_label"] == 1).sum()) if not e.empty else 0,
            "negative_exam_labels": int((e["exam_label"] == 0).sum()) if not e.empty else 0,
            "training_records": int((e["split"] == "training").sum() + (e["split"] == "Training").sum()) if not e.empty else 0,
            "test_records": int((e["split"] == "test").sum() + (e["split"] == "Test").sum()) if not e.empty else 0,
            "external_records": int((e["split"] == "External").sum()) if not e.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "view_level_records": int(len(view_df)),
        "exam_records": int(len(exam_df)),
        "unique_patients": int(exam_df["patient_id"].nunique()) if not exam_df.empty else 0,
        "complete_four_view": int(exam_df["complete_four_view"].sum()) if not exam_df.empty else 0,
        "complete_any_unilateral_multiview": int(exam_df["complete_any_unilateral_multiview"].sum()) if not exam_df.empty else 0,
        "complete_any_bilateral": int(exam_df["complete_any_bilateral"].sum()) if not exam_df.empty else 0,
        "incomplete_records": int((~exam_df["complete_four_view"]).sum()) if not exam_df.empty else 0,
        "known_exam_labels": int(exam_df["exam_label"].notna().sum()) if not exam_df.empty else 0,
        "positive_exam_labels": int((exam_df["exam_label"] == 1).sum()) if not exam_df.empty else 0,
        "negative_exam_labels": int((exam_df["exam_label"] == 0).sum()) if not exam_df.empty else 0,
        "training_records": int((exam_df["split"] == "training").sum() + (exam_df["split"] == "Training").sum()) if not exam_df.empty else 0,
        "test_records": int((exam_df["split"] == "test").sum() + (exam_df["split"] == "Test").sum()) if not exam_df.empty else 0,
        "external_records": int((exam_df["split"] == "External").sum()) if not exam_df.empty else 0,
    })

    return pd.DataFrame(rows)


# =============================================================================
# Output
# =============================================================================

def save_dataset_specific_outputs(exam_df: pd.DataFrame) -> None:
    for dataset in DATASETS:
        out_dir = OUTPUT_ROOT / dataset
        out_dir.mkdir(parents=True, exist_ok=True)

        d = exam_df[exam_df["dataset"] == dataset].copy()
        d.to_csv(out_dir / "multiview_exam_records.csv", index=False, encoding="utf-8-sig")


def save_json(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "stage1a_index": str(STAGE1A_INDEX),
        "stage1a2_harmonized": str(STAGE1A2_HARMONIZED),
        "output_root": str(OUTPUT_ROOT),
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    summary_df: pd.DataFrame,
    exam_df: pd.DataFrame,
    duplicate_df: pd.DataFrame,
    incomplete_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1B v2 MULTI-VIEW PAIRING REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Stage1A input: {STAGE1A_INDEX}")
    lines.append(f"Stage1A2 input: {STAGE1A2_HARMONIZED}")
    lines.append(f"Output root: {OUTPUT_ROOT}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("DATASET DISTRIBUTION")
    lines.append("-" * 100)
    if not exam_df.empty:
        cols = ["dataset", "split", "complete_four_view", "exam_label"]
        dist = (
            exam_df.groupby(cols, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values(["dataset", "split", "complete_four_view", "exam_label"])
        )
        lines.append(dist.to_string(index=False))
    else:
        lines.append("No exam records.")
    lines.append("")

    lines.append("DUPLICATE VIEW CANDIDATES")
    lines.append("-" * 100)
    if duplicate_df.empty:
        lines.append("No duplicate view candidates.")
    else:
        lines.append(f"Duplicate rows: {len(duplicate_df):,}")
        lines.append(duplicate_df.head(50).to_string(index=False))
    lines.append("")

    lines.append("INCOMPLETE MULTI-VIEW RECORDS")
    lines.append("-" * 100)
    if incomplete_df.empty:
        lines.append("No incomplete records.")
    else:
        lines.append(f"Incomplete records: {len(incomplete_df):,}")
        lines.append(incomplete_df.head(50).to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_GLOBAL_CSV,
        OUTPUT_SUMMARY_CSV,
        OUTPUT_VIEW_LEVEL_CSV,
        OUTPUT_DUPLICATES_CSV,
        OUTPUT_INCOMPLETE_CSV,
        OUTPUT_JSON,
        OUTPUT_REPORT_TXT,
    ]:
        lines.append(str(p))

    for dataset in DATASETS:
        lines.append(str(OUTPUT_ROOT / dataset / "multiview_exam_records.csv"))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1B v2 BUILD MULTI-VIEW PAIRS")
    print("=" * 100)
    print(f"Stage1A index:       {STAGE1A_INDEX}")
    print(f"Stage1A2 harmonized: {STAGE1A2_HARMONIZED}")
    print(f"Output root:         {OUTPUT_ROOT}")
    print("-" * 100)

    print("Building view-level records...")
    view_df = build_all_view_level_records()

    print("Building exam-level multi-view records...")
    exam_df = build_exam_records(view_df)

    print("Auditing duplicates and incomplete records...")
    duplicate_df = find_duplicate_views(view_df)
    incomplete_df = find_incomplete_records(exam_df)

    print("Building summary...")
    summary_df = build_summary(exam_df, view_df)

    print("Saving outputs...")
    save_dataset_specific_outputs(exam_df)

    view_df.to_csv(OUTPUT_VIEW_LEVEL_CSV, index=False, encoding="utf-8-sig")
    exam_df.to_csv(OUTPUT_GLOBAL_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    duplicate_df.to_csv(OUTPUT_DUPLICATES_CSV, index=False, encoding="utf-8-sig")
    incomplete_df.to_csv(OUTPUT_INCOMPLETE_CSV, index=False, encoding="utf-8-sig")

    save_json(summary_df)
    write_report(
        summary_df=summary_df,
        exam_df=exam_df,
        duplicate_df=duplicate_df,
        incomplete_df=incomplete_df,
    )

    print()
    print("STAGE 1B v2 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"View-level records:   {OUTPUT_VIEW_LEVEL_CSV}")
    print(f"Global exam records:  {OUTPUT_GLOBAL_CSV}")
    print(f"Summary:              {OUTPUT_SUMMARY_CSV}")
    print(f"Duplicate candidates: {OUTPUT_DUPLICATES_CSV}")
    print(f"Incomplete records:   {OUTPUT_INCOMPLETE_CSV}")
    print(f"JSON summary:         {OUTPUT_JSON}")
    print(f"Text report:          {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()