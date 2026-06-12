r"""
Stage1C_Build_Longitudinal_Temporal_Cohorts.py

Purpose
-------
Build transparent temporal/longitudinal cohort tables after Stage1B_v2 and
Stage1B1 auditing.

This script separates three different concepts that must not be confused:

1. Real longitudinal sequences
   - Multiple exams for the same patient across different study dates.
   - Only valid when a dataset has reliable patient-level repeated visits and dates.

2. Same-exam multi-view spatial sequences
   - LCC, LMLO, RCC, RMLO from the same exam.
   - These are spatial/multiview sequences, not true longitudinal follow-up.

3. Single-time external validation records
   - One exam per patient/study, often with complete four-view data.
   - These are useful for external evaluation but not for longitudinal claims.

Why this stage is needed
------------------------
Reviewer comments questioned whether temporal/longitudinal modeling used real
follow-up data or artificially generated sequences. This script produces explicit
tables that document real temporal availability, missing temporal history, and
how temporal-spatial inputs will be handled.

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B_v2_Global_Multiview_Exam_Records.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B_v2_View_Level_Input_Records.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1B1_Recommended_Exclusions.csv

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Exam_Temporal_Availability.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Longitudinal_Sequences.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Same_Exam_Multiview_Sequences.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Temporal_Modeling_Cohort.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Recommended_Temporal_Use.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage1C_Temporal_Cohort_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage1C_Longitudinal_Temporal_Cohort_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1C_Build_Longitudinal_Temporal_Cohorts.py
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

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

INPUT_EXAM_RECORDS = RESULTS_TABLE_DIR / "Stage1B_v2_Global_Multiview_Exam_Records.csv"
INPUT_VIEW_RECORDS = RESULTS_TABLE_DIR / "Stage1B_v2_View_Level_Input_Records.csv"
INPUT_EXCLUSIONS = RESULTS_TABLE_DIR / "Stage1B1_Recommended_Exclusions.csv"

OUTPUT_TEMPORAL_AVAILABILITY = RESULTS_TABLE_DIR / "Stage1C_Exam_Temporal_Availability.csv"
OUTPUT_LONGITUDINAL_SEQUENCES = RESULTS_TABLE_DIR / "Stage1C_Longitudinal_Sequences.csv"
OUTPUT_SAME_EXAM_SEQUENCES = RESULTS_TABLE_DIR / "Stage1C_Same_Exam_Multiview_Sequences.csv"
OUTPUT_TEMPORAL_MODELING_COHORT = RESULTS_TABLE_DIR / "Stage1C_Temporal_Modeling_Cohort.csv"
OUTPUT_RECOMMENDED_USE = RESULTS_TABLE_DIR / "Stage1C_Recommended_Temporal_Use.csv"
OUTPUT_SUMMARY = RESULTS_TABLE_DIR / "Stage1C_Temporal_Cohort_Summary.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage1C_Temporal_Cohort_Summary.json"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage1C_Longitudinal_Temporal_Cohort_Report.txt"

RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

VIEW_ORDER = ["LCC", "LMLO", "RCC", "RMLO"]


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


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def read_csv_optional(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def parse_date_like(value) -> str:
    """
    Return a normalized YYYY-MM-DD-style string when possible.
    Otherwise return empty string.
    """
    raw = safe_str(value)

    if raw == "":
        return ""

    # Common DICOM date: YYYYMMDD or YYYYMMDD.0
    raw_clean = raw.replace(".0", "").strip()

    if re.fullmatch(r"\d{8}", raw_clean):
        return f"{raw_clean[0:4]}-{raw_clean[4:6]}-{raw_clean[6:8]}"

    # Try pandas datetime.
    try:
        dt = pd.to_datetime(raw, errors="coerce")
        if pd.isna(dt):
            return ""
        return str(dt.date())
    except Exception:
        return ""


def extract_exam_date(row: pd.Series) -> str:
    for col in ["study_date", "exam_date", "acquisition_date", "date"]:
        if col in row.index:
            date_text = parse_date_like(row.get(col, ""))
            if date_text:
                return date_text
    return ""


def temporal_days_between(d1: str, d2: str) -> Optional[int]:
    if not d1 or not d2:
        return None

    try:
        a = pd.to_datetime(d1)
        b = pd.to_datetime(d2)
        return int((b - a).days)
    except Exception:
        return None


def split_view_tokens(row: pd.Series) -> Dict[str, str]:
    return {
        "LCC": safe_str(row.get("lcc_processed_path", "")),
        "LMLO": safe_str(row.get("lmlo_processed_path", "")),
        "RCC": safe_str(row.get("rcc_processed_path", "")),
        "RMLO": safe_str(row.get("rmlo_processed_path", "")),
    }


def get_missing_views(row: pd.Series) -> str:
    missing = []

    if not bool_value(row.get("has_lcc", False)):
        missing.append("LCC")
    if not bool_value(row.get("has_lmlo", False)):
        missing.append("LMLO")
    if not bool_value(row.get("has_rcc", False)):
        missing.append("RCC")
    if not bool_value(row.get("has_rmlo", False)):
        missing.append("RMLO")

    return "|".join(missing)


def get_available_views(row: pd.Series) -> str:
    available = []

    if bool_value(row.get("has_lcc", False)):
        available.append("LCC")
    if bool_value(row.get("has_lmlo", False)):
        available.append("LMLO")
    if bool_value(row.get("has_rcc", False)):
        available.append("RCC")
    if bool_value(row.get("has_rmlo", False)):
        available.append("RMLO")

    return "|".join(available)


def make_list_text(values: List[str]) -> str:
    return "|".join([safe_str(x) for x in values if safe_str(x)])


# =============================================================================
# Preprocessing Inputs
# =============================================================================

def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exam_df = read_csv_required(INPUT_EXAM_RECORDS)
    view_df = read_csv_required(INPUT_VIEW_RECORDS)
    exclusions_df = read_csv_optional(INPUT_EXCLUSIONS)

    return exam_df, view_df, exclusions_df


def prepare_exam_records(exam_df: pd.DataFrame, exclusions_df: pd.DataFrame) -> pd.DataFrame:
    d = exam_df.copy()

    d["exam_date"] = d.apply(extract_exam_date, axis=1)
    d["available_views"] = d.apply(get_available_views, axis=1)
    d["missing_views"] = d.apply(get_missing_views, axis=1)

    d["complete_four_view_bool"] = d["complete_four_view"].map(bool_value)
    d["complete_unilateral_bool"] = d["complete_any_unilateral_multiview"].map(bool_value)
    d["complete_bilateral_bool"] = d["complete_any_bilateral"].map(bool_value)

    d["has_exam_label"] = d["exam_label"].notna()

    d["temporal_exclusion_reason"] = ""
    d["excluded_from_complete_four_view"] = False

    if not exclusions_df.empty and "record_id" in exclusions_df.columns:
        exclusion_ids = set(exclusions_df["record_id"].astype(str).tolist())
        d["excluded_from_complete_four_view"] = d["record_id"].astype(str).isin(exclusion_ids)
        d.loc[d["excluded_from_complete_four_view"], "temporal_exclusion_reason"] = (
            "Recommended exclusion from complete-four-view analysis by Stage1B1 audit."
        )

    return d


# =============================================================================
# Temporal Availability
# =============================================================================

def build_temporal_availability(exam_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, dataset_df in exam_df.groupby("dataset", dropna=False):
        for patient_id, g in dataset_df.groupby("patient_id", dropna=False):
            g = g.copy()

            unique_exam_ids = sorted(set(g["exam_id"].astype(str).tolist()))
            dates = sorted(set(x for x in g["exam_date"].astype(str).tolist() if x))

            n_exams = len(unique_exam_ids)
            n_dates = len(dates)

            has_real_longitudinal_sequence = n_dates >= 2
            has_multiple_exam_records = n_exams >= 2

            if has_real_longitudinal_sequence:
                temporal_status = "REAL_LONGITUDINAL_AVAILABLE"
            elif has_multiple_exam_records:
                temporal_status = "MULTIPLE_EXAMS_NO_RELIABLE_DATE"
            else:
                temporal_status = "SINGLE_TIMEPOINT"

            rows.append({
                "dataset": safe_str(dataset),
                "patient_id": safe_str(patient_id),
                "n_exam_records": int(n_exams),
                "n_distinct_exam_dates": int(n_dates),
                "exam_ids": make_list_text(unique_exam_ids),
                "exam_dates": make_list_text(dates),
                "has_real_longitudinal_sequence": bool(has_real_longitudinal_sequence),
                "has_multiple_exam_records": bool(has_multiple_exam_records),
                "temporal_status": temporal_status,
                "complete_four_view_exams": int(g["complete_four_view_bool"].sum()),
                "available_view_exams": int(len(g)),
                "labeled_exams": int(g["has_exam_label"].sum()),
                "positive_exams": int((g["exam_label"] == 1).sum()) if "exam_label" in g.columns else 0,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Longitudinal Sequences
# =============================================================================

def build_longitudinal_sequences(exam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build real longitudinal sequences only when a patient has multiple distinct
    exam dates. Otherwise no artificial longitudinal sequence is generated.
    """
    rows = []

    for dataset, dataset_df in exam_df.groupby("dataset", dropna=False):
        for patient_id, g in dataset_df.groupby("patient_id", dropna=False):
            g = g.copy()

            g = g[g["exam_date"].astype(str).str.len() > 0].copy()

            if g.empty:
                continue

            g = g.sort_values(["exam_date", "exam_id"])

            unique_dates = sorted(set(g["exam_date"].astype(str).tolist()))

            if len(unique_dates) < 2:
                continue

            sequence_id = f"{safe_str(dataset)}__{safe_str(patient_id)}__longitudinal"

            sequence_items = []

            prev_date = None
            intervals = []

            for idx, row in enumerate(g.itertuples(index=False)):
                r = row._asdict()

                current_date = safe_str(r.get("exam_date", ""))

                interval = None
                if prev_date is not None:
                    interval = temporal_days_between(prev_date, current_date)
                    if interval is not None:
                        intervals.append(interval)

                prev_date = current_date

                sequence_items.append({
                    "sequence_index": idx,
                    "exam_id": safe_str(r.get("exam_id", "")),
                    "exam_date": current_date,
                    "available_views": safe_str(r.get("available_views", "")),
                    "missing_views": safe_str(r.get("missing_views", "")),
                    "complete_four_view": bool_value(r.get("complete_four_view_bool", False)),
                    "exam_label": r.get("exam_label", None),
                    "exam_birads_summary": safe_str(r.get("exam_birads_summary", "")),
                    "breast_density": safe_str(r.get("breast_density", "")),
                })

            rows.append({
                "sequence_id": sequence_id,
                "dataset": safe_str(dataset),
                "patient_id": safe_str(patient_id),
                "sequence_type": "REAL_LONGITUDINAL",
                "n_exams_in_sequence": int(len(g)),
                "n_distinct_dates": int(len(unique_dates)),
                "first_exam_date": unique_dates[0],
                "last_exam_date": unique_dates[-1],
                "temporal_span_days": temporal_days_between(unique_dates[0], unique_dates[-1]),
                "intervals_days": make_list_text([str(x) for x in intervals]),
                "contains_complete_four_view_exam": bool(g["complete_four_view_bool"].any()),
                "all_exams_complete_four_view": bool(g["complete_four_view_bool"].all()),
                "n_labeled_exams": int(g["has_exam_label"].sum()),
                "n_positive_exams": int((g["exam_label"] == 1).sum()) if "exam_label" in g.columns else 0,
                "sequence_items_json": json.dumps(sequence_items, ensure_ascii=False),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Same-Exam Multi-View Sequences
# =============================================================================

def build_same_exam_multiview_sequences(exam_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build spatial sequence records from same-exam views. These are not real
    temporal sequences. They are marked as SAME_EXAM_MULTIVIEW_SPATIAL.
    """
    rows = []

    for row in exam_df.itertuples(index=False):
        r = pd.Series(row._asdict())

        view_paths = split_view_tokens(r)
        available_views = [v for v in VIEW_ORDER if safe_str(view_paths.get(v, ""))]

        sequence_items = []

        for idx, view_name in enumerate(VIEW_ORDER):
            path = safe_str(view_paths.get(view_name, ""))

            image_id_col = {
                "LCC": "lcc_image_id",
                "LMLO": "lmlo_image_id",
                "RCC": "rcc_image_id",
                "RMLO": "rmlo_image_id",
            }.get(view_name, "")

            label_col = {
                "LCC": "lcc_label",
                "LMLO": "lmlo_label",
                "RCC": "rcc_label",
                "RMLO": "rmlo_label",
            }.get(view_name, "")

            birads_col = {
                "LCC": "lcc_birads",
                "LMLO": "lmlo_birads",
                "RCC": "rcc_birads",
                "RMLO": "rmlo_birads",
            }.get(view_name, "")

            sequence_items.append({
                "sequence_index": idx,
                "view": view_name,
                "available": bool(path),
                "processed_path": path,
                "image_id": safe_str(r.get(image_id_col, "")),
                "view_label": safe_str(r.get(label_col, "")),
                "view_birads": safe_str(r.get(birads_col, "")),
            })

        if len(available_views) == 4:
            completeness = "COMPLETE_FOUR_VIEW"
        elif len(available_views) >= 2:
            completeness = "PARTIAL_MULTIVIEW"
        else:
            completeness = "SINGLE_OR_LIMITED_VIEW"

        sequence_id = f"{safe_str(r.get('dataset', ''))}__{safe_str(r.get('exam_id', ''))}__same_exam_multiview"

        rows.append({
            "sequence_id": sequence_id,
            "dataset": safe_str(r.get("dataset", "")),
            "patient_id": safe_str(r.get("patient_id", "")),
            "study_id": safe_str(r.get("study_id", "")),
            "exam_id": safe_str(r.get("exam_id", "")),
            "split": safe_str(r.get("split", "")),
            "sequence_type": "SAME_EXAM_MULTIVIEW_SPATIAL",
            "is_real_longitudinal": False,
            "is_artificial_temporal_sequence": False,
            "n_available_views": int(len(available_views)),
            "available_views": make_list_text(available_views),
            "missing_views": safe_str(r.get("missing_views", "")),
            "completeness": completeness,
            "complete_four_view": bool_value(r.get("complete_four_view_bool", False)),
            "complete_unilateral_multiview": bool_value(r.get("complete_unilateral_bool", False)),
            "complete_bilateral": bool_value(r.get("complete_bilateral_bool", False)),
            "exam_label": r.get("exam_label", None),
            "exam_label_text": safe_str(r.get("exam_label_text", "")),
            "breast_density": safe_str(r.get("breast_density", "")),
            "exam_birads_summary": safe_str(r.get("exam_birads_summary", "")),
            "exam_n_findings": safe_int(r.get("exam_n_findings", 0)),
            "excluded_from_complete_four_view": bool_value(r.get("excluded_from_complete_four_view", False)),
            "temporal_note": (
                "This is a same-exam multi-view spatial sequence and must not be reported as real longitudinal follow-up."
            ),
            "sequence_items_json": json.dumps(sequence_items, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Temporal Modeling Cohort
# =============================================================================

def determine_modeling_role(row: pd.Series) -> Tuple[str, str]:
    dataset = safe_str(row.get("dataset", ""))

    complete_four = bool_value(row.get("complete_four_view_bool", False))
    excluded = bool_value(row.get("excluded_from_complete_four_view", False))
    has_label = bool_value(row.get("has_exam_label", False))

    if complete_four and not excluded:
        if dataset == "VinDr-Mammo" and has_label:
            return "LABELED_COMPLETE_FOUR_VIEW_MODELING", "Use for labeled complete-four-view training/evaluation."
        if dataset == "INbreast":
            return "EXTERNAL_COMPLETE_FOUR_VIEW_AVAILABLE_VIEW_VALIDATION", "Use as external complete-view validation subset when labels are not required."
        if dataset == "CBIS-DDSM":
            return "CBIS_COMPLETE_FOUR_VIEW_SUBSET", "Use cautiously; CBIS is lesion-centered and labels must be merged before supervised use."

    if bool_value(row.get("complete_unilateral_bool", False)):
        return "AVAILABLE_UNILATERAL_MULTIVIEW", "Use for available-view unilateral multi-view analysis."

    if bool_value(row.get("complete_bilateral_bool", False)):
        return "AVAILABLE_BILATERAL_SINGLE_PROJECTION", "Use for bilateral asymmetry analysis on available projections."

    return "LIMITED_OR_INCOMPLETE_AVAILABLE_VIEW", "Exclude from complete-view experiments; retain only for audit or limited available-view analysis."


def build_temporal_modeling_cohort(exam_df: pd.DataFrame, longitudinal_df: pd.DataFrame) -> pd.DataFrame:
    longitudinal_patient_ids = set(longitudinal_df["patient_id"].astype(str).tolist()) if not longitudinal_df.empty else set()

    rows = []

    for row in exam_df.itertuples(index=False):
        r = pd.Series(row._asdict())

        role, recommendation = determine_modeling_role(r)

        patient_id = safe_str(r.get("patient_id", ""))

        has_real_longitudinal_history = patient_id in longitudinal_patient_ids

        if has_real_longitudinal_history:
            temporal_input_type = "REAL_LONGITUDINAL_HISTORY_AVAILABLE"
        else:
            temporal_input_type = "NO_REAL_LONGITUDINAL_HISTORY"

        rows.append({
            "record_id": safe_str(r.get("record_id", "")),
            "dataset": safe_str(r.get("dataset", "")),
            "patient_id": patient_id,
            "study_id": safe_str(r.get("study_id", "")),
            "exam_id": safe_str(r.get("exam_id", "")),
            "split": safe_str(r.get("split", "")),
            "exam_date": safe_str(r.get("exam_date", "")),
            "has_real_longitudinal_history": bool(has_real_longitudinal_history),
            "temporal_input_type": temporal_input_type,
            "same_exam_multiview_available": bool_value(r.get("complete_unilateral_bool", False)),
            "complete_four_view": bool_value(r.get("complete_four_view_bool", False)),
            "excluded_from_complete_four_view": bool_value(r.get("excluded_from_complete_four_view", False)),
            "available_views": safe_str(r.get("available_views", "")),
            "missing_views": safe_str(r.get("missing_views", "")),
            "exam_label": r.get("exam_label", None),
            "exam_label_text": safe_str(r.get("exam_label_text", "")),
            "breast_density": safe_str(r.get("breast_density", "")),
            "modeling_role": role,
            "recommendation": recommendation,
        })

    return pd.DataFrame(rows)


# =============================================================================
# Recommended Temporal Use
# =============================================================================

def build_recommended_temporal_use(
    availability_df: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
    same_exam_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset in sorted(set(modeling_df["dataset"].astype(str).tolist())):
        a = availability_df[availability_df["dataset"] == dataset].copy()
        l = longitudinal_df[longitudinal_df["dataset"] == dataset].copy() if not longitudinal_df.empty else pd.DataFrame()
        s = same_exam_df[same_exam_df["dataset"] == dataset].copy()
        m = modeling_df[modeling_df["dataset"] == dataset].copy()

        n_real_longitudinal = int(len(l))
        n_same_exam = int(len(s))
        n_complete_four = int((m["complete_four_view"] & ~m["excluded_from_complete_four_view"]).sum())
        n_labeled = int(m["exam_label"].notna().sum())

        if n_real_longitudinal > 0:
            temporal_claim_status = "REAL_LONGITUDINAL_ALLOWED_FOR_SUBSET"
            manuscript_language = (
                "Real longitudinal analyses are restricted to patients with repeated dated examinations; "
                "same-exam multi-view sequences are reported separately."
            )
        else:
            temporal_claim_status = "NO_REAL_LONGITUDINAL_CLAIM"
            manuscript_language = (
                "No real longitudinal follow-up is available in this processed cohort. "
                "The temporal branch should be described as same-exam temporal-spatial ordering of views, "
                "not as real disease-course longitudinal modeling."
            )

        rows.append({
            "dataset": dataset,
            "patients_or_studies": int(a["patient_id"].nunique()) if not a.empty else 0,
            "real_longitudinal_sequences": n_real_longitudinal,
            "same_exam_multiview_sequences": n_same_exam,
            "complete_four_view_sequences": n_complete_four,
            "labeled_exam_records": n_labeled,
            "temporal_claim_status": temporal_claim_status,
            "recommended_manuscript_language": manuscript_language,
        })

    rows.append({
        "dataset": "ALL",
        "patients_or_studies": int(modeling_df["patient_id"].nunique()) if not modeling_df.empty else 0,
        "real_longitudinal_sequences": int(len(longitudinal_df)),
        "same_exam_multiview_sequences": int(len(same_exam_df)),
        "complete_four_view_sequences": int((modeling_df["complete_four_view"] & ~modeling_df["excluded_from_complete_four_view"]).sum()),
        "labeled_exam_records": int(modeling_df["exam_label"].notna().sum()),
        "temporal_claim_status": "USE_DATASET_SPECIFIC_LANGUAGE",
        "recommended_manuscript_language": (
            "Separate real longitudinal availability from same-exam multi-view temporal-spatial ordering. "
            "Do not claim real longitudinal disease-course modeling unless sequences contain repeated dated examinations."
        ),
    })

    return pd.DataFrame(rows)


# =============================================================================
# Summary and Report
# =============================================================================

def build_summary(
    availability_df: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
    same_exam_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    datasets = sorted(set(modeling_df["dataset"].astype(str).tolist()))

    for dataset in datasets:
        a = availability_df[availability_df["dataset"] == dataset].copy()
        l = longitudinal_df[longitudinal_df["dataset"] == dataset].copy() if not longitudinal_df.empty else pd.DataFrame()
        s = same_exam_df[same_exam_df["dataset"] == dataset].copy()
        m = modeling_df[modeling_df["dataset"] == dataset].copy()

        rows.append({
            "dataset": dataset,
            "patients": int(a["patient_id"].nunique()) if not a.empty else 0,
            "exam_records": int(len(m)),
            "patients_with_multiple_exam_records": int((a["n_exam_records"] >= 2).sum()) if not a.empty else 0,
            "patients_with_real_longitudinal_dates": int((a["has_real_longitudinal_sequence"] == True).sum()) if not a.empty else 0,
            "real_longitudinal_sequences": int(len(l)),
            "same_exam_multiview_sequences": int(len(s)),
            "complete_four_view_sequences_usable": int((m["complete_four_view"] & ~m["excluded_from_complete_four_view"]).sum()) if not m.empty else 0,
            "available_unilateral_sequences": int(m["same_exam_multiview_available"].sum()) if not m.empty else 0,
            "labeled_records": int(m["exam_label"].notna().sum()) if not m.empty else 0,
            "positive_labeled_records": int((m["exam_label"] == 1).sum()) if not m.empty else 0,
            "negative_labeled_records": int((m["exam_label"] == 0).sum()) if not m.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "patients": int(modeling_df["patient_id"].nunique()) if not modeling_df.empty else 0,
        "exam_records": int(len(modeling_df)),
        "patients_with_multiple_exam_records": int((availability_df["n_exam_records"] >= 2).sum()) if not availability_df.empty else 0,
        "patients_with_real_longitudinal_dates": int((availability_df["has_real_longitudinal_sequence"] == True).sum()) if not availability_df.empty else 0,
        "real_longitudinal_sequences": int(len(longitudinal_df)),
        "same_exam_multiview_sequences": int(len(same_exam_df)),
        "complete_four_view_sequences_usable": int((modeling_df["complete_four_view"] & ~modeling_df["excluded_from_complete_four_view"]).sum()) if not modeling_df.empty else 0,
        "available_unilateral_sequences": int(modeling_df["same_exam_multiview_available"].sum()) if not modeling_df.empty else 0,
        "labeled_records": int(modeling_df["exam_label"].notna().sum()) if not modeling_df.empty else 0,
        "positive_labeled_records": int((modeling_df["exam_label"] == 1).sum()) if not modeling_df.empty else 0,
        "negative_labeled_records": int((modeling_df["exam_label"] == 0).sum()) if not modeling_df.empty else 0,
    })

    return pd.DataFrame(rows)


def save_json(summary_df: pd.DataFrame, recommended_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "exam_records": str(INPUT_EXAM_RECORDS),
            "view_records": str(INPUT_VIEW_RECORDS),
            "recommended_exclusions": str(INPUT_EXCLUSIONS),
        },
        "summary": summary_df.to_dict(orient="records"),
        "recommended_temporal_use": recommended_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    summary_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    longitudinal_df: pd.DataFrame,
    same_exam_df: pd.DataFrame,
    modeling_df: pd.DataFrame,
    recommended_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1C LONGITUDINAL AND TEMPORAL COHORT REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input exam records: {INPUT_EXAM_RECORDS}")
    lines.append(f"Input view records: {INPUT_VIEW_RECORDS}")
    lines.append(f"Input exclusions: {INPUT_EXCLUSIONS}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("RECOMMENDED TEMPORAL USE")
    lines.append("-" * 100)
    lines.append(recommended_df.to_string(index=False))
    lines.append("")

    lines.append("TEMPORAL AVAILABILITY STATUS DISTRIBUTION")
    lines.append("-" * 100)
    if not availability_df.empty:
        dist = (
            availability_df.groupby(["dataset", "temporal_status"])
            .size()
            .reset_index(name="patients_or_studies")
            .sort_values(["dataset", "temporal_status"])
        )
        lines.append(dist.to_string(index=False))
    else:
        lines.append("No temporal availability rows.")
    lines.append("")

    lines.append("REAL LONGITUDINAL SEQUENCES")
    lines.append("-" * 100)
    if longitudinal_df.empty:
        lines.append("No real longitudinal repeated-date sequences were identified.")
    else:
        lines.append(longitudinal_df.head(50).to_string(index=False))
    lines.append("")

    lines.append("SAME-EXAM MULTIVIEW SEQUENCES SAMPLE")
    lines.append("-" * 100)
    if same_exam_df.empty:
        lines.append("No same-exam multi-view sequences.")
    else:
        cols = [
            "sequence_id",
            "dataset",
            "exam_id",
            "sequence_type",
            "n_available_views",
            "available_views",
            "missing_views",
            "completeness",
            "complete_four_view",
            "exam_label",
            "temporal_note",
        ]
        available_cols = [c for c in cols if c in same_exam_df.columns]
        lines.append(same_exam_df[available_cols].head(50).to_string(index=False))
    lines.append("")

    lines.append("MODELING COHORT ROLE DISTRIBUTION")
    lines.append("-" * 100)
    if not modeling_df.empty:
        dist = (
            modeling_df.groupby(["dataset", "modeling_role"])
            .size()
            .reset_index(name="records")
            .sort_values(["dataset", "records"], ascending=[True, False])
        )
        lines.append(dist.to_string(index=False))
    else:
        lines.append("No modeling cohort rows.")
    lines.append("")

    lines.append("PUBLICATION-READY CLARIFICATION")
    lines.append("-" * 100)
    lines.append(
        "This stage explicitly separates real longitudinal follow-up from same-exam "
        "multi-view spatial ordering. If no repeated dated examinations are available "
        "for a dataset, the manuscript must not claim real longitudinal disease-course "
        "modeling for that dataset. Instead, the temporal branch should be described as "
        "temporal-spatial organization of concurrently acquired mammographic views."
    )
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_TEMPORAL_AVAILABILITY,
        OUTPUT_LONGITUDINAL_SEQUENCES,
        OUTPUT_SAME_EXAM_SEQUENCES,
        OUTPUT_TEMPORAL_MODELING_COHORT,
        OUTPUT_RECOMMENDED_USE,
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
    print("STAGE 1C BUILD LONGITUDINAL AND TEMPORAL COHORTS")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input exam records: {INPUT_EXAM_RECORDS}")
    print(f"Input view records: {INPUT_VIEW_RECORDS}")
    print(f"Input exclusions: {INPUT_EXCLUSIONS}")
    print("-" * 100)

    print("Loading inputs...")
    exam_df, view_df, exclusions_df = load_inputs()

    print("Preparing exam records...")
    prepared_exam_df = prepare_exam_records(exam_df, exclusions_df)

    print("Building temporal availability table...")
    availability_df = build_temporal_availability(prepared_exam_df)

    print("Building real longitudinal sequence table...")
    longitudinal_df = build_longitudinal_sequences(prepared_exam_df)

    print("Building same-exam multi-view sequence table...")
    same_exam_df = build_same_exam_multiview_sequences(prepared_exam_df)

    print("Building temporal modeling cohort...")
    modeling_df = build_temporal_modeling_cohort(prepared_exam_df, longitudinal_df)

    print("Building recommended temporal use table...")
    recommended_df = build_recommended_temporal_use(
        availability_df=availability_df,
        longitudinal_df=longitudinal_df,
        same_exam_df=same_exam_df,
        modeling_df=modeling_df,
    )

    print("Building summary...")
    summary_df = build_summary(
        availability_df=availability_df,
        longitudinal_df=longitudinal_df,
        same_exam_df=same_exam_df,
        modeling_df=modeling_df,
    )

    print("Saving outputs...")
    availability_df.to_csv(OUTPUT_TEMPORAL_AVAILABILITY, index=False, encoding="utf-8-sig")
    longitudinal_df.to_csv(OUTPUT_LONGITUDINAL_SEQUENCES, index=False, encoding="utf-8-sig")
    same_exam_df.to_csv(OUTPUT_SAME_EXAM_SEQUENCES, index=False, encoding="utf-8-sig")
    modeling_df.to_csv(OUTPUT_TEMPORAL_MODELING_COHORT, index=False, encoding="utf-8-sig")
    recommended_df.to_csv(OUTPUT_RECOMMENDED_USE, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    save_json(summary_df, recommended_df)

    write_report(
        summary_df=summary_df,
        availability_df=availability_df,
        longitudinal_df=longitudinal_df,
        same_exam_df=same_exam_df,
        modeling_df=modeling_df,
        recommended_df=recommended_df,
    )

    print()
    print("STAGE 1C COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Temporal availability:       {OUTPUT_TEMPORAL_AVAILABILITY}")
    print(f"Real longitudinal sequences: {OUTPUT_LONGITUDINAL_SEQUENCES}")
    print(f"Same-exam sequences:         {OUTPUT_SAME_EXAM_SEQUENCES}")
    print(f"Temporal modeling cohort:    {OUTPUT_TEMPORAL_MODELING_COHORT}")
    print(f"Recommended temporal use:    {OUTPUT_RECOMMENDED_USE}")
    print(f"Summary:                     {OUTPUT_SUMMARY}")
    print(f"JSON summary:                {OUTPUT_JSON}")
    print(f"Text report:                 {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()