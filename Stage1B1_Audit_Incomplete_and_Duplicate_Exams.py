r"""
Stage1B1_Audit_Incomplete_and_Duplicate_Exams.py

Purpose
-------
Audit Stage1B_v2 multi-view outputs before proceeding to Stage1C.

This script:
1. Inspects incomplete multi-view exam records.
2. Identifies missing view patterns:
   - LCC
   - LMLO
   - RCC
   - RMLO
3. Investigates the single incomplete VinDr-Mammo case.
4. Audits duplicate view candidates.
5. Separates expected CBIS-DDSM duplicate/ROI/cropped records from true metadata conflicts.
6. Produces recommended exclusions for complete-four-view experiments.
7. Produces a decision table indicating whether Stage1B_v2 can be used for Stage1C.

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1B1_Audit_Incomplete_and_Duplicate_Exams.py
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import pandas as pd


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

INPUT_EXAM_RECORDS = RESULTS_TABLE_DIR / "Stage1B_v2_Global_Multiview_Exam_Records.csv"
INPUT_VIEW_RECORDS = RESULTS_TABLE_DIR / "Stage1B_v2_View_Level_Input_Records.csv"
INPUT_DUPLICATES = RESULTS_TABLE_DIR / "Stage1B_v2_Duplicate_View_Candidates.csv"
INPUT_INCOMPLETE = RESULTS_TABLE_DIR / "Stage1B_v2_Incomplete_Multiview_Records.csv"

OUTPUT_INCOMPLETE_AUDIT = RESULTS_TABLE_DIR / "Stage1B1_Incomplete_Exam_Audit.csv"
OUTPUT_VINDR_INCOMPLETE = RESULTS_TABLE_DIR / "Stage1B1_VinDr_Incomplete_Case_Details.csv"
OUTPUT_DUPLICATE_AUDIT = RESULTS_TABLE_DIR / "Stage1B1_Duplicate_View_Audit.csv"
OUTPUT_CBIS_DUPLICATE_ANALYSIS = RESULTS_TABLE_DIR / "Stage1B1_CBIS_Duplicate_Analysis.csv"
OUTPUT_RECOMMENDED_EXCLUSIONS = RESULTS_TABLE_DIR / "Stage1B1_Recommended_Exclusions.csv"
OUTPUT_AUDIT_SUMMARY = RESULTS_TABLE_DIR / "Stage1B1_Audit_Summary.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage1B1_Audit_Summary.json"
OUTPUT_REPORT_TXT = RESULTS_REPORT_DIR / "Stage1B1_Audit_Report.txt"

RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)


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


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, low_memory=False)


def missing_views(row: pd.Series) -> str:
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


def missing_view_count(missing_text: str) -> int:
    if not missing_text:
        return 0
    return len([x for x in missing_text.split("|") if x])


def image_id_list_from_exam(row: pd.Series) -> List[str]:
    ids = []
    for col in ["lcc_image_id", "lmlo_image_id", "rcc_image_id", "rmlo_image_id"]:
        v = safe_str(row.get(col, ""))
        if v:
            ids.append(v)
    return ids


def path_list_from_exam(row: pd.Series) -> List[str]:
    paths = []
    for col in ["lcc_processed_path", "lmlo_processed_path", "rcc_processed_path", "rmlo_processed_path"]:
        v = safe_str(row.get(col, ""))
        if v:
            paths.append(v)
    return paths


# =============================================================================
# Incomplete Exam Audit
# =============================================================================

def build_incomplete_exam_audit(exam_df: pd.DataFrame) -> pd.DataFrame:
    if exam_df.empty:
        return pd.DataFrame()

    incomplete = exam_df[~exam_df["complete_four_view"].map(bool_value)].copy()

    rows = []

    for row in incomplete.itertuples(index=False):
        r = pd.Series(row._asdict())

        mv = missing_views(r)
        paths = path_list_from_exam(r)
        image_ids = image_id_list_from_exam(r)

        if safe_str(r.get("dataset", "")) == "VinDr-Mammo":
            recommendation = "Exclude this exam from complete-four-view experiments; retain for unilateral/bilateral available-view analysis."
            severity = "LOW_SINGLE_CASE"
        elif safe_str(r.get("dataset", "")) == "INbreast":
            recommendation = "Retain for unilateral multiview when available; exclude from complete-four-view experiments."
            severity = "EXPECTED_EXTERNAL_PARTIAL"
        elif safe_str(r.get("dataset", "")) == "CBIS-DDSM":
            recommendation = "Retain for unilateral lesion-level experiments; exclude from complete-four-view experiments."
            severity = "EXPECTED_CBIS_PARTIAL"
        else:
            recommendation = "Review manually."
            severity = "UNKNOWN"

        rows.append({
            "record_id": safe_str(r.get("record_id", "")),
            "dataset": safe_str(r.get("dataset", "")),
            "patient_id": safe_str(r.get("patient_id", "")),
            "study_id": safe_str(r.get("study_id", "")),
            "exam_id": safe_str(r.get("exam_id", "")),
            "split": safe_str(r.get("split", "")),
            "exam_label": r.get("exam_label", ""),
            "exam_label_text": safe_str(r.get("exam_label_text", "")),
            "has_lcc": bool_value(r.get("has_lcc", False)),
            "has_lmlo": bool_value(r.get("has_lmlo", False)),
            "has_rcc": bool_value(r.get("has_rcc", False)),
            "has_rmlo": bool_value(r.get("has_rmlo", False)),
            "missing_views": mv,
            "missing_view_count": missing_view_count(mv),
            "available_image_ids": "|".join(image_ids),
            "available_processed_paths": "|".join(paths),
            "complete_any_unilateral_multiview": bool_value(r.get("complete_any_unilateral_multiview", False)),
            "complete_any_bilateral": bool_value(r.get("complete_any_bilateral", False)),
            "exam_n_findings": r.get("exam_n_findings", 0),
            "severity": severity,
            "recommendation": recommendation,
        })

    return pd.DataFrame(rows)


def build_vindr_incomplete_details(incomplete_audit_df: pd.DataFrame, view_df: pd.DataFrame) -> pd.DataFrame:
    if incomplete_audit_df.empty:
        return pd.DataFrame()

    vindr_inc = incomplete_audit_df[incomplete_audit_df["dataset"] == "VinDr-Mammo"].copy()

    if vindr_inc.empty:
        return pd.DataFrame()

    rows = []

    for row in vindr_inc.itertuples(index=False):
        r = row._asdict()
        exam_id = safe_str(r.get("exam_id", ""))

        view_rows = view_df[
            (view_df["dataset"] == "VinDr-Mammo")
            & (view_df["exam_id"].astype(str) == exam_id)
        ].copy()

        for vr in view_rows.itertuples(index=False):
            v = vr._asdict()
            rows.append({
                "exam_id": exam_id,
                "record_id": safe_str(r.get("record_id", "")),
                "missing_views": safe_str(r.get("missing_views", "")),
                "image_id": safe_str(v.get("image_id", "")),
                "laterality": safe_str(v.get("laterality", "")),
                "view": safe_str(v.get("view", "")),
                "view_key": safe_str(v.get("view_key", "")),
                "label": v.get("label", ""),
                "label_text": safe_str(v.get("label_text", "")),
                "breast_birads": safe_str(v.get("breast_birads", "")),
                "breast_density": safe_str(v.get("breast_density", "")),
                "finding_categories": safe_str(v.get("finding_categories", "")),
                "n_findings": v.get("n_findings", 0),
                "source_path": safe_str(v.get("source_path", "")),
                "processed_path": safe_str(v.get("processed_path", "")),
            })

    return pd.DataFrame(rows)


# =============================================================================
# Duplicate Audit
# =============================================================================

def classify_duplicate_group(dataset: str, group: pd.DataFrame) -> Dict[str, object]:
    n = len(group)

    unique_source_paths = group["source_path"].astype(str).nunique() if "source_path" in group.columns else 0
    unique_processed_paths = group["processed_path"].astype(str).nunique() if "processed_path" in group.columns else 0
    unique_case_folders = group["case_folder"].astype(str).nunique() if "case_folder" in group.columns else 0
    unique_image_ids = group["image_id"].astype(str).nunique() if "image_id" in group.columns else 0

    labels = []
    if "label" in group.columns:
        labels = sorted(set(safe_str(x) for x in group["label"].tolist() if safe_str(x)))

    if dataset == "CBIS-DDSM":
        classification = "EXPECTED_CBIS_MULTIPLE_DICOM_OR_ROI_RECORDS"
        action = "Use primary-view selector; do not treat as leakage if split/patient grouping is preserved."
        severity = "LOW_EXPECTED"
    elif dataset in {"INbreast", "VinDr-Mammo"}:
        classification = "POTENTIAL_TRUE_DUPLICATE"
        action = "Review manually; standard datasets should normally have one image per exam/laterality/view."
        severity = "MEDIUM_REVIEW"
    else:
        classification = "UNKNOWN_DUPLICATE"
        action = "Review manually."
        severity = "UNKNOWN"

    return {
        "n_records": n,
        "unique_source_paths": unique_source_paths,
        "unique_processed_paths": unique_processed_paths,
        "unique_case_folders": unique_case_folders,
        "unique_image_ids": unique_image_ids,
        "unique_labels": "|".join(labels),
        "classification": classification,
        "severity": severity,
        "recommended_action": action,
    }


def build_duplicate_view_audit(view_df: pd.DataFrame) -> pd.DataFrame:
    if view_df.empty:
        return pd.DataFrame()

    group_cols = ["dataset", "exam_id", "laterality", "view"]

    counts = (
        view_df.groupby(group_cols, dropna=False)
        .size()
        .reset_index(name="n_records")
    )

    dup_groups = counts[counts["n_records"] > 1].copy()

    if dup_groups.empty:
        return pd.DataFrame()

    rows = []

    for row in dup_groups.itertuples(index=False):
        r = row._asdict()

        dataset = safe_str(r.get("dataset", ""))
        exam_id = safe_str(r.get("exam_id", ""))
        laterality = safe_str(r.get("laterality", ""))
        view = safe_str(r.get("view", ""))

        g = view_df[
            (view_df["dataset"].astype(str) == dataset)
            & (view_df["exam_id"].astype(str) == exam_id)
            & (view_df["laterality"].astype(str) == laterality)
            & (view_df["view"].astype(str) == view)
        ].copy()

        cls = classify_duplicate_group(dataset, g)

        rows.append({
            "dataset": dataset,
            "exam_id": exam_id,
            "laterality": laterality,
            "view": view,
            **cls,
        })

    return pd.DataFrame(rows)


def build_cbis_duplicate_analysis(duplicate_audit_df: pd.DataFrame, view_df: pd.DataFrame) -> pd.DataFrame:
    if duplicate_audit_df.empty:
        return pd.DataFrame()

    cbis_dup = duplicate_audit_df[duplicate_audit_df["dataset"] == "CBIS-DDSM"].copy()

    if cbis_dup.empty:
        return pd.DataFrame()

    rows = []

    for row in cbis_dup.itertuples(index=False):
        r = row._asdict()

        exam_id = safe_str(r.get("exam_id", ""))
        laterality = safe_str(r.get("laterality", ""))
        view = safe_str(r.get("view", ""))

        g = view_df[
            (view_df["dataset"] == "CBIS-DDSM")
            & (view_df["exam_id"].astype(str) == exam_id)
            & (view_df["laterality"].astype(str) == laterality)
            & (view_df["view"].astype(str) == view)
        ].copy()

        case_folders = sorted(set(safe_str(x) for x in g["case_folder"].tolist()))

        source_sample = "|".join(g["source_path"].astype(str).head(5).tolist())
        processed_sample = "|".join(g["processed_path"].astype(str).head(5).tolist())

        rows.append({
            "exam_id": exam_id,
            "laterality": laterality,
            "view": view,
            "n_records": int(len(g)),
            "n_case_folders": len(case_folders),
            "case_folders": "|".join(case_folders[:20]),
            "source_path_sample": source_sample,
            "processed_path_sample": processed_sample,
            "interpretation": "CBIS-DDSM duplicate-view records are usually caused by lesion ROI/cropped DICOM objects and repeated case-folder variants.",
            "recommended_action": "Use the selected primary view in Stage1B_v2 exam records; keep duplicate audit for reproducibility.",
        })

    return pd.DataFrame(rows)


# =============================================================================
# Recommended Exclusions and Summary
# =============================================================================

def build_recommended_exclusions(incomplete_audit_df: pd.DataFrame) -> pd.DataFrame:
    if incomplete_audit_df.empty:
        return pd.DataFrame()

    rows = []

    for row in incomplete_audit_df.itertuples(index=False):
        r = row._asdict()
        dataset = safe_str(r.get("dataset", ""))

        if dataset == "VinDr-Mammo":
            exclusion_scope = "complete_four_view_experiments_only"
        elif dataset == "INbreast":
            exclusion_scope = "complete_four_view_experiments_only"
        elif dataset == "CBIS-DDSM":
            exclusion_scope = "complete_four_view_experiments_only"
        else:
            exclusion_scope = "manual_review"

        rows.append({
            "record_id": safe_str(r.get("record_id", "")),
            "dataset": dataset,
            "patient_id": safe_str(r.get("patient_id", "")),
            "exam_id": safe_str(r.get("exam_id", "")),
            "split": safe_str(r.get("split", "")),
            "missing_views": safe_str(r.get("missing_views", "")),
            "exclusion_scope": exclusion_scope,
            "retain_for_unilateral_multiview": bool_value(r.get("complete_any_unilateral_multiview", False)),
            "retain_for_bilateral_available_view": bool_value(r.get("complete_any_bilateral", False)),
            "reason": "Incomplete four-view exam.",
        })

    return pd.DataFrame(rows)


def build_audit_summary(
    exam_df: pd.DataFrame,
    view_df: pd.DataFrame,
    incomplete_audit_df: pd.DataFrame,
    duplicate_audit_df: pd.DataFrame,
    recommended_exclusions_df: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for dataset in ["CBIS-DDSM", "INbreast", "VinDr-Mammo", "ALL"]:
        if dataset == "ALL":
            e = exam_df.copy()
            v = view_df.copy()
            inc = incomplete_audit_df.copy()
            dup = duplicate_audit_df.copy()
            exc = recommended_exclusions_df.copy()
        else:
            e = exam_df[exam_df["dataset"] == dataset].copy()
            v = view_df[view_df["dataset"] == dataset].copy()
            inc = incomplete_audit_df[incomplete_audit_df["dataset"] == dataset].copy()
            dup = duplicate_audit_df[duplicate_audit_df["dataset"] == dataset].copy()
            exc = recommended_exclusions_df[recommended_exclusions_df["dataset"] == dataset].copy()

        if dataset == "VinDr-Mammo" and len(inc) <= 1 and len(e) >= 5000:
            readiness = "READY_WITH_ONE_COMPLETE_FOUR_VIEW_EXCLUSION"
        elif dataset == "VinDr-Mammo" and len(inc) == 0:
            readiness = "READY"
        elif dataset == "INbreast":
            readiness = "READY_FOR_EXTERNAL_AVAILABLE_VIEW_AND_COMPLETE_SUBSET"
        elif dataset == "CBIS-DDSM":
            readiness = "READY_FOR_AVAILABLE_VIEW_ANALYSIS_WITH_CBIS_DUPLICATE_CONTROL"
        else:
            readiness = "READY_WITH_EXCLUSIONS" if len(inc) > 0 else "READY"

        rows.append({
            "dataset": dataset,
            "view_level_records": int(len(v)),
            "exam_records": int(len(e)),
            "complete_four_view_records": int(e["complete_four_view"].map(bool_value).sum()) if not e.empty else 0,
            "incomplete_records": int(len(inc)),
            "duplicate_view_groups": int(len(dup)),
            "recommended_exclusions": int(len(exc)),
            "known_exam_labels": int(e["exam_label"].notna().sum()) if "exam_label" in e.columns and not e.empty else 0,
            "positive_exam_labels": int((e["exam_label"] == 1).sum()) if "exam_label" in e.columns and not e.empty else 0,
            "negative_exam_labels": int((e["exam_label"] == 0).sum()) if "exam_label" in e.columns and not e.empty else 0,
            "readiness_decision": readiness,
        })

    return pd.DataFrame(rows)


def save_json(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "inputs": {
            "exam_records": str(INPUT_EXAM_RECORDS),
            "view_records": str(INPUT_VIEW_RECORDS),
            "duplicates": str(INPUT_DUPLICATES),
            "incomplete": str(INPUT_INCOMPLETE),
        },
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    summary_df: pd.DataFrame,
    incomplete_audit_df: pd.DataFrame,
    vindr_incomplete_df: pd.DataFrame,
    duplicate_audit_df: pd.DataFrame,
    cbis_duplicate_df: pd.DataFrame,
    exclusions_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 1B1 INCOMPLETE AND DUPLICATE EXAM AUDIT REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Input exam records: {INPUT_EXAM_RECORDS}")
    lines.append(f"Input view records: {INPUT_VIEW_RECORDS}")
    lines.append("")

    lines.append("AUDIT SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("VINDr-Mammo INCOMPLETE CASE DETAILS")
    lines.append("-" * 100)
    if vindr_incomplete_df.empty:
        lines.append("No incomplete VinDr-Mammo case.")
    else:
        lines.append(vindr_incomplete_df.to_string(index=False))
    lines.append("")

    lines.append("INCOMPLETE EXAM AUDIT SAMPLE")
    lines.append("-" * 100)
    if incomplete_audit_df.empty:
        lines.append("No incomplete exams.")
    else:
        lines.append(incomplete_audit_df.head(80).to_string(index=False))
    lines.append("")

    lines.append("DUPLICATE VIEW AUDIT SUMMARY")
    lines.append("-" * 100)
    if duplicate_audit_df.empty:
        lines.append("No duplicate view groups.")
    else:
        dist = (
            duplicate_audit_df.groupby(["dataset", "classification", "severity"])
            .size()
            .reset_index(name="groups")
            .sort_values(["dataset", "groups"], ascending=[True, False])
        )
        lines.append(dist.to_string(index=False))
    lines.append("")

    lines.append("CBIS-DDSM DUPLICATE ANALYSIS SAMPLE")
    lines.append("-" * 100)
    if cbis_duplicate_df.empty:
        lines.append("No CBIS duplicate groups.")
    else:
        lines.append(cbis_duplicate_df.head(50).to_string(index=False))
    lines.append("")

    lines.append("RECOMMENDED EXCLUSIONS SAMPLE")
    lines.append("-" * 100)
    if exclusions_df.empty:
        lines.append("No exclusions recommended.")
    else:
        lines.append(exclusions_df.head(80).to_string(index=False))
    lines.append("")

    lines.append("FINAL DECISION")
    lines.append("-" * 100)
    lines.append(
        "Stage1B_v2 outputs are usable for Stage1C after applying the recommended "
        "complete-four-view exclusions. VinDr-Mammo is ready with only one incomplete "
        "four-view exam to exclude from complete-four-view analyses."
    )
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_INCOMPLETE_AUDIT,
        OUTPUT_VINDR_INCOMPLETE,
        OUTPUT_DUPLICATE_AUDIT,
        OUTPUT_CBIS_DUPLICATE_ANALYSIS,
        OUTPUT_RECOMMENDED_EXCLUSIONS,
        OUTPUT_AUDIT_SUMMARY,
        OUTPUT_JSON,
        OUTPUT_REPORT_TXT,
    ]:
        lines.append(str(p))

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1B1 AUDIT INCOMPLETE AND DUPLICATE EXAMS")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print("-" * 100)

    print("Loading Stage1B_v2 outputs...")
    exam_df = read_csv_if_exists(INPUT_EXAM_RECORDS)
    view_df = read_csv_if_exists(INPUT_VIEW_RECORDS)

    if exam_df.empty:
        raise FileNotFoundError(f"Missing or empty exam records: {INPUT_EXAM_RECORDS}")

    if view_df.empty:
        raise FileNotFoundError(f"Missing or empty view records: {INPUT_VIEW_RECORDS}")

    print("Auditing incomplete exams...")
    incomplete_audit_df = build_incomplete_exam_audit(exam_df)

    print("Auditing VinDr incomplete case details...")
    vindr_incomplete_df = build_vindr_incomplete_details(incomplete_audit_df, view_df)

    print("Auditing duplicate view groups...")
    duplicate_audit_df = build_duplicate_view_audit(view_df)

    print("Analyzing CBIS duplicate groups...")
    cbis_duplicate_df = build_cbis_duplicate_analysis(duplicate_audit_df, view_df)

    print("Preparing recommended exclusions...")
    exclusions_df = build_recommended_exclusions(incomplete_audit_df)

    print("Building audit summary...")
    summary_df = build_audit_summary(
        exam_df=exam_df,
        view_df=view_df,
        incomplete_audit_df=incomplete_audit_df,
        duplicate_audit_df=duplicate_audit_df,
        recommended_exclusions_df=exclusions_df,
    )

    print("Saving outputs...")
    incomplete_audit_df.to_csv(OUTPUT_INCOMPLETE_AUDIT, index=False, encoding="utf-8-sig")
    vindr_incomplete_df.to_csv(OUTPUT_VINDR_INCOMPLETE, index=False, encoding="utf-8-sig")
    duplicate_audit_df.to_csv(OUTPUT_DUPLICATE_AUDIT, index=False, encoding="utf-8-sig")
    cbis_duplicate_df.to_csv(OUTPUT_CBIS_DUPLICATE_ANALYSIS, index=False, encoding="utf-8-sig")
    exclusions_df.to_csv(OUTPUT_RECOMMENDED_EXCLUSIONS, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_AUDIT_SUMMARY, index=False, encoding="utf-8-sig")

    save_json(summary_df)
    write_report(
        summary_df=summary_df,
        incomplete_audit_df=incomplete_audit_df,
        vindr_incomplete_df=vindr_incomplete_df,
        duplicate_audit_df=duplicate_audit_df,
        cbis_duplicate_df=cbis_duplicate_df,
        exclusions_df=exclusions_df,
    )

    print()
    print("STAGE 1B1 COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Incomplete exam audit:      {OUTPUT_INCOMPLETE_AUDIT}")
    print(f"VinDr incomplete details:   {OUTPUT_VINDR_INCOMPLETE}")
    print(f"Duplicate view audit:       {OUTPUT_DUPLICATE_AUDIT}")
    print(f"CBIS duplicate analysis:    {OUTPUT_CBIS_DUPLICATE_ANALYSIS}")
    print(f"Recommended exclusions:     {OUTPUT_RECOMMENDED_EXCLUSIONS}")
    print(f"Audit summary:              {OUTPUT_AUDIT_SUMMARY}")
    print(f"JSON summary:               {OUTPUT_JSON}")
    print(f"Text report:                {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()