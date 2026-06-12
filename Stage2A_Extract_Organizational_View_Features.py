r"""
Stage2A_Extract_Organizational_View_Features.py

Purpose
-------
Extract lightweight organizational view features consistent with the SACU method.

This stage implements the descriptor-construction layer described in the Methods:
1. Local regional descriptors from each mammographic view.
2. Multi-view CC/MLO descriptors.
3. Bilateral left-right asymmetry descriptors.
4. Same-exam temporal-spatial ordering descriptors.
5. Diagnostic complexity cues for adaptive SACU activation.
6. Resource-aware compactness indicators.

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\manifests\\Stage1D_VinDr_Complete_Four_View_Labeled_Manifest.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\manifests\\Stage1D_Complete_Four_View_Modeling_Manifest.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\manifests\\Stage1D_Available_View_Modeling_Manifest.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\manifests\\Stage1D_Same_Exam_Temporal_Spatial_Manifest.csv

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\features\\Stage2A_Organizational_View_Features.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\features\\Stage2A_VinDr_Complete_Four_View_Features.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\features\\Stage2A_Feature_Dictionary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2A_Feature_Extraction_Summary.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage2A_Feature_Extraction_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage2A_Extract_Organizational_View_Features.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

MANIFEST_DIR = PROJECT_ROOT / "manifests"
FEATURE_DIR = PROJECT_ROOT / "features"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

FEATURE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_VINDR_COMPLETE = MANIFEST_DIR / "Stage1D_VinDr_Complete_Four_View_Labeled_Manifest.csv"
INPUT_COMPLETE_FOUR = MANIFEST_DIR / "Stage1D_Complete_Four_View_Modeling_Manifest.csv"
INPUT_AVAILABLE_VIEW = MANIFEST_DIR / "Stage1D_Available_View_Modeling_Manifest.csv"
INPUT_TEMPORAL_SPATIAL = MANIFEST_DIR / "Stage1D_Same_Exam_Temporal_Spatial_Manifest.csv"

OUTPUT_ALL_FEATURES = FEATURE_DIR / "Stage2A_Organizational_View_Features.csv"
OUTPUT_VINDR_COMPLETE_FEATURES = FEATURE_DIR / "Stage2A_VinDr_Complete_Four_View_Features.csv"
OUTPUT_FEATURE_DICT = FEATURE_DIR / "Stage2A_Feature_Dictionary.csv"
OUTPUT_SUMMARY = RESULTS_TABLE_DIR / "Stage2A_Feature_Extraction_Summary.csv"
OUTPUT_JSON = RESULTS_TABLE_DIR / "Stage2A_Feature_Extraction_Summary.json"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage2A_Feature_Extraction_Report.txt"

# Image processing settings
RESIZE_FOR_FEATURES = (512, 512)
GRID_ROWS = 4
GRID_COLS = 4
EPS = 1e-8

VIEW_KEYS = ["lcc", "lmlo", "rcc", "rmlo"]
VIEW_LABELS = {
    "lcc": "LEFT_CC",
    "lmlo": "LEFT_MLO",
    "rcc": "RIGHT_CC",
    "rmlo": "RIGHT_MLO",
}

RANDOM_SEED = 42


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


def safe_float(value, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except Exception:
        return default


def bool_value(value) -> bool:
    if isinstance(value, bool):
        return value
    return safe_str(value).lower() in {"true", "1", "yes", "y"}


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def path_exists(path_text: str) -> bool:
    p = Path(safe_str(path_text))
    return p.exists() and p.is_file()


def parse_numeric_label(value) -> Optional[int]:
    try:
        if pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def density_to_numeric(value: str) -> float:
    v = safe_str(value).upper()
    if "A" in v:
        return 1.0
    if "B" in v:
        return 2.0
    if "C" in v:
        return 3.0
    if "D" in v:
        return 4.0
    return 0.0


def birads_to_numeric(value: str) -> float:
    text = safe_str(value).upper()
    for k in ["6", "5", "4", "3", "2", "1", "0"]:
        if k in text:
            return float(k)
    return 0.0


# =============================================================================
# Image Loading and Local Regional Descriptors
# =============================================================================

def load_grayscale_image(path_text: str) -> Optional[np.ndarray]:
    path = Path(safe_str(path_text))

    if not path.exists():
        return None

    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

    if img is None:
        return None

    img = cv2.resize(img, RESIZE_FOR_FEATURES, interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0

    return img


def get_breast_mask(img: np.ndarray) -> np.ndarray:
    """
    Simple tissue mask for processed mammograms.
    This is not lesion segmentation. It defines the visible breast/tissue area
    for regional descriptors.
    """
    if img is None:
        return np.zeros(RESIZE_FOR_FEATURES, dtype=np.uint8)

    blur = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = max(0.03, float(np.percentile(blur, 10)))
    mask = (blur > thresh).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def entropy_hist(values: np.ndarray, bins: int = 32) -> float:
    if values.size == 0:
        return 0.0
    hist, _ = np.histogram(values, bins=bins, range=(0.0, 1.0), density=False)
    prob = hist.astype(np.float64)
    prob = prob / (prob.sum() + EPS)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob + EPS)).sum())


def edge_density(region: np.ndarray) -> float:
    if region.size == 0:
        return 0.0
    r = (region * 255).astype(np.uint8)
    edges = cv2.Canny(r, 40, 120)
    return float(edges.mean() / 255.0)


def gradient_stats(region: np.ndarray) -> Tuple[float, float]:
    if region.size == 0:
        return 0.0, 0.0

    gx = cv2.Sobel(region, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(region, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)

    return float(np.mean(mag)), float(np.std(mag))


def local_region_descriptors(img: Optional[np.ndarray], prefix: str) -> Dict[str, float]:
    """
    Extract shallow local descriptors from a 4x4 anatomical grid.

    These descriptors represent localized tissue appearance, consistent with
    the local descriptor construction described in the SACU methods.
    """
    features: Dict[str, float] = {}

    if img is None:
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                base = f"{prefix}_grid{r}{c}"
                for name in [
                    "mean", "std", "p10", "p50", "p90", "entropy",
                    "edge_density", "grad_mean", "grad_std", "tissue_ratio"
                ]:
                    features[f"{base}_{name}"] = np.nan
        for name in [
            "global_mean", "global_std", "global_entropy",
            "global_edge_density", "global_grad_mean", "global_grad_std",
            "tissue_ratio", "mass_center_x", "mass_center_y"
        ]:
            features[f"{prefix}_{name}"] = np.nan
        return features

    mask = get_breast_mask(img)
    h, w = img.shape

    tissue_values = img[mask > 0]
    if tissue_values.size == 0:
        tissue_values = img.reshape(-1)

    features[f"{prefix}_global_mean"] = float(np.mean(tissue_values))
    features[f"{prefix}_global_std"] = float(np.std(tissue_values))
    features[f"{prefix}_global_entropy"] = entropy_hist(tissue_values)
    features[f"{prefix}_global_edge_density"] = edge_density(img)
    gm, gs = gradient_stats(img)
    features[f"{prefix}_global_grad_mean"] = gm
    features[f"{prefix}_global_grad_std"] = gs
    features[f"{prefix}_tissue_ratio"] = float(mask.mean())

    ys, xs = np.where(mask > 0)
    if len(xs) > 0:
        features[f"{prefix}_mass_center_x"] = float(xs.mean() / max(w - 1, 1))
        features[f"{prefix}_mass_center_y"] = float(ys.mean() / max(h - 1, 1))
    else:
        features[f"{prefix}_mass_center_x"] = 0.5
        features[f"{prefix}_mass_center_y"] = 0.5

    cell_h = h // GRID_ROWS
    cell_w = w // GRID_COLS

    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            y0 = r * cell_h
            y1 = h if r == GRID_ROWS - 1 else (r + 1) * cell_h
            x0 = c * cell_w
            x1 = w if c == GRID_COLS - 1 else (c + 1) * cell_w

            region = img[y0:y1, x0:x1]
            region_mask = mask[y0:y1, x0:x1]
            vals = region[region_mask > 0]

            if vals.size == 0:
                vals = region.reshape(-1)

            base = f"{prefix}_grid{r}{c}"

            features[f"{base}_mean"] = float(np.mean(vals))
            features[f"{base}_std"] = float(np.std(vals))
            features[f"{base}_p10"] = float(np.percentile(vals, 10))
            features[f"{base}_p50"] = float(np.percentile(vals, 50))
            features[f"{base}_p90"] = float(np.percentile(vals, 90))
            features[f"{base}_entropy"] = entropy_hist(vals)
            features[f"{base}_edge_density"] = edge_density(region)
            g_mean, g_std = gradient_stats(region)
            features[f"{base}_grad_mean"] = g_mean
            features[f"{base}_grad_std"] = g_std
            features[f"{base}_tissue_ratio"] = float(region_mask.mean())

    return features


def extract_single_view_features(path_text: str, prefix: str) -> Dict[str, float]:
    img = load_grayscale_image(path_text)
    return local_region_descriptors(img, prefix)


# =============================================================================
# Organizational Derived Features
# =============================================================================

def pairwise_abs_diff(features: Dict[str, float], prefix_a: str, prefix_b: str, out_prefix: str) -> Dict[str, float]:
    """
    Compute difference descriptors between two views based on shared descriptor suffixes.
    """
    out: Dict[str, float] = {}

    keys_a = [k for k in features if k.startswith(prefix_a + "_")]

    for ka in keys_a:
        suffix = ka[len(prefix_a) + 1:]
        kb = f"{prefix_b}_{suffix}"
        if kb not in features:
            continue

        va = safe_float(features.get(ka, np.nan), np.nan)
        vb = safe_float(features.get(kb, np.nan), np.nan)

        if np.isnan(va) or np.isnan(vb):
            out[f"{out_prefix}_{suffix}_absdiff"] = np.nan
            out[f"{out_prefix}_{suffix}_signeddiff"] = np.nan
        else:
            out[f"{out_prefix}_{suffix}_absdiff"] = abs(va - vb)
            out[f"{out_prefix}_{suffix}_signeddiff"] = va - vb

    return out


def aggregate_diff_stats(diff_features: Dict[str, float], prefix: str) -> Dict[str, float]:
    vals = []

    for k, v in diff_features.items():
        if k.startswith(prefix) and k.endswith("_absdiff"):
            try:
                if not pd.isna(v):
                    vals.append(float(v))
            except Exception:
                pass

    if not vals:
        return {
            f"{prefix}_mean_absdiff": np.nan,
            f"{prefix}_max_absdiff": np.nan,
            f"{prefix}_std_absdiff": np.nan,
        }

    arr = np.asarray(vals, dtype=np.float32)

    return {
        f"{prefix}_mean_absdiff": float(np.mean(arr)),
        f"{prefix}_max_absdiff": float(np.max(arr)),
        f"{prefix}_std_absdiff": float(np.std(arr)),
    }


def multiview_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Multi-view branch:
    - left CC/MLO consistency
    - right CC/MLO consistency
    """
    out: Dict[str, float] = {}

    left_mv = pairwise_abs_diff(base_features, "lcc", "lmlo", "mv_left_cc_mlo")
    right_mv = pairwise_abs_diff(base_features, "rcc", "rmlo", "mv_right_cc_mlo")

    out.update(left_mv)
    out.update(right_mv)
    out.update(aggregate_diff_stats(left_mv, "mv_left_cc_mlo"))
    out.update(aggregate_diff_stats(right_mv, "mv_right_cc_mlo"))

    left_mean = out.get("mv_left_cc_mlo_mean_absdiff", np.nan)
    right_mean = out.get("mv_right_cc_mlo_mean_absdiff", np.nan)

    if not pd.isna(left_mean) and not pd.isna(right_mean):
        out["mv_global_mean_absdiff"] = float((left_mean + right_mean) / 2.0)
        out["mv_left_right_gap"] = float(abs(left_mean - right_mean))
    else:
        out["mv_global_mean_absdiff"] = np.nan
        out["mv_left_right_gap"] = np.nan

    return out


def bilateral_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Bilateral branch:
    - left-right asymmetry in CC
    - left-right asymmetry in MLO
    """
    out: Dict[str, float] = {}

    cc_bi = pairwise_abs_diff(base_features, "lcc", "rcc", "bi_cc_left_right")
    mlo_bi = pairwise_abs_diff(base_features, "lmlo", "rmlo", "bi_mlo_left_right")

    out.update(cc_bi)
    out.update(mlo_bi)
    out.update(aggregate_diff_stats(cc_bi, "bi_cc_left_right"))
    out.update(aggregate_diff_stats(mlo_bi, "bi_mlo_left_right"))

    cc_mean = out.get("bi_cc_left_right_mean_absdiff", np.nan)
    mlo_mean = out.get("bi_mlo_left_right_mean_absdiff", np.nan)

    if not pd.isna(cc_mean) and not pd.isna(mlo_mean):
        out["bi_global_mean_asymmetry"] = float((cc_mean + mlo_mean) / 2.0)
        out["bi_projection_gap"] = float(abs(cc_mean - mlo_mean))
    else:
        out["bi_global_mean_asymmetry"] = np.nan
        out["bi_projection_gap"] = np.nan

    return out


def temporal_spatial_features(base_features: Dict[str, float]) -> Dict[str, float]:
    """
    Same-exam temporal-spatial branch.

    This is NOT real longitudinal follow-up.
    It encodes the ordered same-exam view sequence:
    LCC -> LMLO -> RCC -> RMLO
    """
    out: Dict[str, float] = {}

    ordered_pairs = [
        ("lcc", "lmlo", "ts_lcc_to_lmlo"),
        ("lmlo", "rcc", "ts_lmlo_to_rcc"),
        ("rcc", "rmlo", "ts_rcc_to_rmlo"),
    ]

    pair_means = []

    for a, b, prefix in ordered_pairs:
        diff = pairwise_abs_diff(base_features, a, b, prefix)
        out.update(diff)
        stats = aggregate_diff_stats(diff, prefix)
        out.update(stats)

        val = stats.get(f"{prefix}_mean_absdiff", np.nan)
        if not pd.isna(val):
            pair_means.append(float(val))

    if pair_means:
        out["ts_sequence_mean_step_change"] = float(np.mean(pair_means))
        out["ts_sequence_max_step_change"] = float(np.max(pair_means))
        out["ts_sequence_std_step_change"] = float(np.std(pair_means))
    else:
        out["ts_sequence_mean_step_change"] = np.nan
        out["ts_sequence_max_step_change"] = np.nan
        out["ts_sequence_std_step_change"] = np.nan

    out["ts_real_longitudinal_flag"] = 0.0
    out["ts_same_exam_spatial_ordering_flag"] = 1.0

    return out


def metadata_features(row: pd.Series) -> Dict[str, float]:
    out: Dict[str, float] = {}

    out["meta_breast_density_numeric"] = density_to_numeric(row.get("breast_density", ""))

    for view in VIEW_KEYS:
        birads_col = f"{view}_birads"
        label_col = f"{view}_label"

        out[f"meta_{view}_birads_numeric"] = birads_to_numeric(row.get(birads_col, ""))
        label = parse_numeric_label(row.get(label_col, None))
        out[f"meta_{view}_label"] = float(label) if label is not None else np.nan

    birads_values = [
        out[f"meta_{view}_birads_numeric"]
        for view in VIEW_KEYS
        if out[f"meta_{view}_birads_numeric"] > 0
    ]

    if birads_values:
        out["meta_exam_birads_max"] = float(max(birads_values))
        out["meta_exam_birads_mean"] = float(np.mean(birads_values))
    else:
        out["meta_exam_birads_max"] = 0.0
        out["meta_exam_birads_mean"] = 0.0

    out["meta_exam_label"] = float(parse_numeric_label(row.get("exam_label", np.nan)) or 0)
    out["meta_has_label"] = 1.0 if parse_numeric_label(row.get("exam_label", np.nan)) is not None else 0.0

    return out


def complexity_and_resource_features(all_features: Dict[str, float], row: pd.Series) -> Dict[str, float]:
    """
    Adaptive SACU cues:
    - Diagnostic complexity estimated from asymmetry, multiview inconsistency,
      entropy, density, and finding load.
    - Resource proxy estimated from number of available views.
    """
    out: Dict[str, float] = {}

    mv = safe_float(all_features.get("mv_global_mean_absdiff", 0.0), 0.0)
    bi = safe_float(all_features.get("bi_global_mean_asymmetry", 0.0), 0.0)
    ts = safe_float(all_features.get("ts_sequence_mean_step_change", 0.0), 0.0)
    density = safe_float(all_features.get("meta_breast_density_numeric", 0.0), 0.0)

    global_entropies = []
    for view in VIEW_KEYS:
        v = all_features.get(f"{view}_global_entropy", np.nan)
        if not pd.isna(v):
            global_entropies.append(float(v))

    entropy_mean = float(np.mean(global_entropies)) if global_entropies else 0.0

    n_available_views = 0
    for view in VIEW_KEYS:
        p = safe_str(row.get(f"{view}_processed_path", ""))
        if p:
            n_available_views += 1

    n_findings = safe_float(row.get("exam_n_findings", 0.0), 0.0)

    raw_complexity = (
        0.25 * mv
        + 0.30 * bi
        + 0.15 * ts
        + 0.10 * entropy_mean
        + 0.10 * (density / 4.0 if density > 0 else 0.0)
        + 0.10 * min(n_findings / 5.0, 1.0)
    )

    complexity = float(1.0 / (1.0 + math.exp(-5.0 * (raw_complexity - 0.25))))

    resource_score = float(n_available_views / 4.0)

    # Adaptive active-agent estimate.
    # This is not the final learned agent selector; it is a deterministic feature cue
    # for later SACU training and ablation.
    min_agents = 3
    max_agents = 12
    active_agents_est = min_agents + int(round((max_agents - min_agents) * complexity * max(resource_score, 0.25)))

    out["sacu_input_complexity_score"] = complexity
    out["sacu_resource_view_availability_score"] = resource_score
    out["sacu_estimated_active_agents"] = float(active_agents_est)
    out["sacu_entropy_mean"] = entropy_mean
    out["sacu_multiview_inconsistency"] = mv
    out["sacu_bilateral_asymmetry"] = bi
    out["sacu_temporal_spatial_step_change"] = ts

    return out


# =============================================================================
# Feature Extraction for One Exam
# =============================================================================

def extract_exam_features(row: pd.Series) -> Dict[str, object]:
    start = time.perf_counter()

    record: Dict[str, object] = {}

    # Identity / manifest fields
    identity_cols = [
        "record_id", "dataset", "patient_id", "study_id", "exam_id",
        "split", "normalized_split", "task_name", "task_family",
        "modeling_role", "available_views", "missing_views",
        "exam_label", "exam_label_text", "breast_density",
    ]

    for col in identity_cols:
        record[col] = row.get(col, "")

    # Local view descriptors
    base_features: Dict[str, float] = {}

    for view in VIEW_KEYS:
        path_col = f"{view}_processed_path"
        path = safe_str(row.get(path_col, ""))

        record[f"{view}_processed_path"] = path
        record[f"{view}_available"] = 1 if path_exists(path) else 0

        feats = extract_single_view_features(path, view)
        base_features.update(feats)

    # Organizational branches
    derived: Dict[str, float] = {}
    derived.update(base_features)
    derived.update(multiview_features(base_features))
    derived.update(bilateral_features(base_features))
    derived.update(temporal_spatial_features(base_features))
    derived.update(metadata_features(row))

    sacu_feats = complexity_and_resource_features(derived, row)
    derived.update(sacu_feats)

    # Add features to record
    record.update(derived)

    elapsed = time.perf_counter() - start
    record["feature_extraction_time_sec"] = elapsed
    record["feature_extraction_status"] = "OK"

    return record


# =============================================================================
# Feature Dictionary and Summary
# =============================================================================

def build_feature_dictionary(feature_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    identity_cols = {
        "record_id", "dataset", "patient_id", "study_id", "exam_id",
        "split", "normalized_split", "task_name", "task_family",
        "modeling_role", "available_views", "missing_views",
        "exam_label", "exam_label_text", "breast_density",
        "feature_extraction_status",
    }

    for col in feature_df.columns:
        if col in identity_cols:
            family = "identity_or_label"
        elif col.startswith(("lcc_", "lmlo_", "rcc_", "rmlo_")):
            family = "local_regional_descriptor"
        elif col.startswith("mv_"):
            family = "multi_view_reasoning"
        elif col.startswith("bi_"):
            family = "bilateral_asymmetry"
        elif col.startswith("ts_"):
            family = "same_exam_temporal_spatial"
        elif col.startswith("meta_"):
            family = "clinical_metadata"
        elif col.startswith("sacu_"):
            family = "adaptive_sacu_control"
        else:
            family = "other"

        rows.append({
            "feature_name": col,
            "feature_family": family,
            "description": describe_feature(col, family),
        })

    return pd.DataFrame(rows)


def describe_feature(name: str, family: str) -> str:
    if family == "local_regional_descriptor":
        return "Localized per-view shallow descriptor from anatomical grid or global tissue region."
    if family == "multi_view_reasoning":
        return "CC/MLO complementary-view consistency or difference descriptor."
    if family == "bilateral_asymmetry":
        return "Left-right asymmetry descriptor between anatomically corresponding views."
    if family == "same_exam_temporal_spatial":
        return "Ordered same-exam view-sequence descriptor; not real longitudinal follow-up."
    if family == "clinical_metadata":
        return "Metadata-derived descriptor such as BI-RADS label or density."
    if family == "adaptive_sacu_control":
        return "Complexity/resource cue used for adaptive SACU agent activation."
    if family == "identity_or_label":
        return "Record identity, split, label, or manifest metadata."
    return "Auxiliary feature."


def build_summary(feature_df: pd.DataFrame, feature_dict: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for dataset, d in feature_df.groupby("dataset", dropna=False):
        rows.append({
            "dataset": dataset,
            "records": int(len(d)),
            "labels_known": int(d["exam_label"].notna().sum()) if "exam_label" in d.columns else 0,
            "positive_labels": int((pd.to_numeric(d["exam_label"], errors="coerce") == 1).sum()) if "exam_label" in d.columns else 0,
            "negative_labels": int((pd.to_numeric(d["exam_label"], errors="coerce") == 0).sum()) if "exam_label" in d.columns else 0,
            "lcc_available": int(d["lcc_available"].sum()) if "lcc_available" in d.columns else 0,
            "lmlo_available": int(d["lmlo_available"].sum()) if "lmlo_available" in d.columns else 0,
            "rcc_available": int(d["rcc_available"].sum()) if "rcc_available" in d.columns else 0,
            "rmlo_available": int(d["rmlo_available"].sum()) if "rmlo_available" in d.columns else 0,
            "mean_complexity": float(d["sacu_input_complexity_score"].mean()) if "sacu_input_complexity_score" in d.columns else np.nan,
            "mean_estimated_active_agents": float(d["sacu_estimated_active_agents"].mean()) if "sacu_estimated_active_agents" in d.columns else np.nan,
            "mean_extraction_time_sec": float(d["feature_extraction_time_sec"].mean()) if "feature_extraction_time_sec" in d.columns else np.nan,
        })

    rows.append({
        "dataset": "ALL",
        "records": int(len(feature_df)),
        "labels_known": int(feature_df["exam_label"].notna().sum()) if "exam_label" in feature_df.columns else 0,
        "positive_labels": int((pd.to_numeric(feature_df["exam_label"], errors="coerce") == 1).sum()) if "exam_label" in feature_df.columns else 0,
        "negative_labels": int((pd.to_numeric(feature_df["exam_label"], errors="coerce") == 0).sum()) if "exam_label" in feature_df.columns else 0,
        "lcc_available": int(feature_df["lcc_available"].sum()) if "lcc_available" in feature_df.columns else 0,
        "lmlo_available": int(feature_df["lmlo_available"].sum()) if "lmlo_available" in feature_df.columns else 0,
        "rcc_available": int(feature_df["rcc_available"].sum()) if "rcc_available" in feature_df.columns else 0,
        "rmlo_available": int(feature_df["rmlo_available"].sum()) if "rmlo_available" in feature_df.columns else 0,
        "mean_complexity": float(feature_df["sacu_input_complexity_score"].mean()) if "sacu_input_complexity_score" in feature_df.columns else np.nan,
        "mean_estimated_active_agents": float(feature_df["sacu_estimated_active_agents"].mean()) if "sacu_estimated_active_agents" in feature_df.columns else np.nan,
        "mean_extraction_time_sec": float(feature_df["feature_extraction_time_sec"].mean()) if "feature_extraction_time_sec" in feature_df.columns else np.nan,
    })

    rows.append({
        "dataset": "FEATURE_DICTIONARY",
        "records": int(len(feature_dict)),
        "labels_known": "",
        "positive_labels": "",
        "negative_labels": "",
        "lcc_available": "",
        "lmlo_available": "",
        "rcc_available": "",
        "rmlo_available": "",
        "mean_complexity": "",
        "mean_estimated_active_agents": "",
        "mean_extraction_time_sec": "",
    })

    return pd.DataFrame(rows)


def save_json_summary(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "method_alignment": {
            "local_descriptors": "4x4 anatomical-grid shallow descriptors per view",
            "multi_view_branch": "CC/MLO descriptor differences within each breast",
            "bilateral_branch": "left-right descriptor differences for CC and MLO",
            "temporal_branch": "same-exam temporal-spatial ordering only; no real longitudinal claim",
            "adaptive_sacu": "complexity and resource cues for later active-agent selection",
        },
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(summary_df: pd.DataFrame, feature_dict: pd.DataFrame) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2A ORGANIZATIONAL VIEW FEATURE EXTRACTION REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("METHOD CONSISTENCY")
    lines.append("-" * 100)
    lines.append("Implemented local 4x4 regional descriptors for each available mammographic view.")
    lines.append("Implemented multi-view CC/MLO difference descriptors.")
    lines.append("Implemented bilateral left-right asymmetry descriptors.")
    lines.append("Implemented same-exam temporal-spatial ordered descriptors.")
    lines.append("No real longitudinal feature was created because Stage1C found no repeated dated longitudinal sequences.")
    lines.append("Implemented SACU-compatible complexity/resource cues for adaptive agent activation.")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    lines.append(summary_df.to_string(index=False))
    lines.append("")

    lines.append("FEATURE FAMILIES")
    lines.append("-" * 100)
    fam = feature_dict["feature_family"].value_counts().reset_index()
    fam.columns = ["feature_family", "count"]
    lines.append(fam.to_string(index=False))
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_ALL_FEATURES,
        OUTPUT_VINDR_COMPLETE_FEATURES,
        OUTPUT_FEATURE_DICT,
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
    warnings.filterwarnings("ignore")
    np.random.seed(RANDOM_SEED)

    print("=" * 100)
    print("STAGE 2A EXTRACT ORGANIZATIONAL VIEW FEATURES")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input manifest: {INPUT_VINDR_COMPLETE}")
    print("-" * 100)

    manifest = read_csv_required(INPUT_VINDR_COMPLETE)

    print(f"Records to process: {len(manifest):,}")

    records = []

    for row in tqdm(manifest.itertuples(index=False), total=len(manifest), desc="Extracting features"):
        r = pd.Series(row._asdict())
        try:
            rec = extract_exam_features(r)
            records.append(rec)
        except Exception as e:
            error_rec = {
                "record_id": safe_str(r.get("record_id", "")),
                "dataset": safe_str(r.get("dataset", "")),
                "patient_id": safe_str(r.get("patient_id", "")),
                "exam_id": safe_str(r.get("exam_id", "")),
                "exam_label": r.get("exam_label", np.nan),
                "feature_extraction_status": f"FAILED: {e}",
            }
            records.append(error_rec)

    feature_df = pd.DataFrame(records)

    print("Building feature dictionary...")
    feature_dict = build_feature_dictionary(feature_df)

    print("Building summary...")
    summary_df = build_summary(feature_df, feature_dict)

    print("Saving outputs...")
    feature_df.to_csv(OUTPUT_ALL_FEATURES, index=False, encoding="utf-8-sig")
    feature_df.to_csv(OUTPUT_VINDR_COMPLETE_FEATURES, index=False, encoding="utf-8-sig")
    feature_dict.to_csv(OUTPUT_FEATURE_DICT, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, encoding="utf-8-sig")

    save_json_summary(summary_df)
    write_report(summary_df, feature_dict)

    print()
    print("STAGE 2A COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"All features:             {OUTPUT_ALL_FEATURES}")
    print(f"VinDr complete features:  {OUTPUT_VINDR_COMPLETE_FEATURES}")
    print(f"Feature dictionary:       {OUTPUT_FEATURE_DICT}")
    print(f"Summary:                  {OUTPUT_SUMMARY}")
    print(f"JSON summary:             {OUTPUT_JSON}")
    print(f"Text report:              {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()