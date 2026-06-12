r"""
Stage1A_Preprocess_DICOM_and_Images.py

Purpose
-------
Preprocess downloaded mammography images for the organizational shallow-agent
mammography pipeline.

This script:
1. Reads CBIS-DDSM DICOM images.
2. Reads INbreast DICOM images.
3. Reads VinDr-Mammo PNG images.
4. Applies mammography-safe preprocessing:
   - grayscale conversion
   - robust intensity normalization
   - optional photometric inversion correction
   - foreground breast-region cropping
   - resizing to a fixed resolution
5. Saves standardized PNG images.
6. Saves a preprocessing image index CSV.
7. Saves dataset-level preprocessing summaries.
8. Saves a text report for reproducibility.

Input
-----
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\datasets

Output
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\preprocessing\\processed_images
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage1A_Preprocess_DICOM_and_Images.py

Notes
-----
Install required packages first:

pip install pydicom opencv-python pandas numpy tqdm
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
    PYDICOM_AVAILABLE = True
except Exception:
    PYDICOM_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
DATASET_ROOT = PROJECT_ROOT / "datasets"

OUTPUT_IMAGE_ROOT = PROJECT_ROOT / "preprocessing" / "processed_images"
OUTPUT_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
OUTPUT_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

OUTPUT_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_REPORT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_INDEX_CSV = OUTPUT_TABLE_DIR / "Stage1A_Preprocessed_Image_Index.csv"
OUTPUT_SUMMARY_CSV = OUTPUT_TABLE_DIR / "Stage1A_Preprocessing_Summary.csv"
OUTPUT_FAILURES_CSV = OUTPUT_TABLE_DIR / "Stage1A_Preprocessing_Failures.csv"
OUTPUT_JSON = OUTPUT_TABLE_DIR / "Stage1A_Preprocessing_Summary.json"
OUTPUT_REPORT_TXT = OUTPUT_REPORT_DIR / "Stage1A_Preprocessing_Report.txt"

TARGET_HEIGHT = 1024
TARGET_WIDTH = 768

SAVE_FORMAT = ".png"
PNG_COMPRESSION = 3

# If True, existing processed files will not be rewritten.
SKIP_EXISTING = True

DATASETS = ["CBIS-DDSM", "INbreast", "VinDr-Mammo"]


# =============================================================================
# Utility Functions
# =============================================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_file_size(path: Path) -> int:
    try:
        return int(path.stat().st_size)
    except Exception:
        return 0


def normalize_uint8(img: np.ndarray) -> np.ndarray:
    """
    Robust percentile-based normalization to uint8.
    """
    img = img.astype(np.float32)

    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros_like(img, dtype=np.uint8)

    values = img[finite]

    lo = np.percentile(values, 0.5)
    hi = np.percentile(values, 99.5)

    if hi <= lo:
        lo = float(values.min())
        hi = float(values.max())

    if hi <= lo:
        return np.zeros_like(img, dtype=np.uint8)

    img = np.clip(img, lo, hi)
    img = (img - lo) / (hi - lo)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)

    return img


def maybe_invert_mammogram(img: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Heuristic inversion correction.
    Mammograms usually have a dark background and bright breast tissue.
    If the image has a very bright border/background, invert it.
    """
    if img.ndim != 2:
        return img, False

    h, w = img.shape
    border_width = max(5, min(h, w) // 40)

    top = img[:border_width, :]
    bottom = img[-border_width:, :]
    left = img[:, :border_width]
    right = img[:, -border_width:]

    border_pixels = np.concatenate([
        top.reshape(-1),
        bottom.reshape(-1),
        left.reshape(-1),
        right.reshape(-1),
    ])

    border_median = float(np.median(border_pixels))
    center = img[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_median = float(np.median(center)) if center.size else float(np.median(img))

    if border_median > 180 and border_median > center_median:
        return 255 - img, True

    return img, False


def crop_foreground_breast(img: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    """
    Crop non-background region using thresholding and connected components.

    Returns
    -------
    cropped_img
    bbox: x, y, w, h in original image coordinates
    """
    if img.ndim != 2:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold can fail for very dark images; combine with low threshold.
    _, otsu_mask = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    low_mask = (blur > max(5, int(np.percentile(blur, 5)))).astype(np.uint8) * 255

    mask = cv2.bitwise_and(otsu_mask, low_mask)

    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    if num_labels <= 1:
        h, w = gray.shape
        return img, (0, 0, w, h)

    # Ignore background label 0.
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = int(np.argmax(areas) + 1)

    x = int(stats[largest_label, cv2.CC_STAT_LEFT])
    y = int(stats[largest_label, cv2.CC_STAT_TOP])
    w = int(stats[largest_label, cv2.CC_STAT_WIDTH])
    h = int(stats[largest_label, cv2.CC_STAT_HEIGHT])

    # Add margin.
    img_h, img_w = gray.shape
    margin_x = int(0.03 * w)
    margin_y = int(0.03 * h)

    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(img_w, x + w + margin_x)
    y2 = min(img_h, y + h + margin_y)

    cropped = img[y1:y2, x1:x2]

    return cropped, (x1, y1, x2 - x1, y2 - y1)


def resize_with_padding(
    img: np.ndarray,
    target_size: Tuple[int, int],
) -> Tuple[np.ndarray, float, int, int]:
    """
    Resize while preserving aspect ratio, then pad to target size.

    target_size: height, width
    returns: output_image, scale, pad_top, pad_left
    """
    target_h, target_w = target_size

    h, w = img.shape[:2]

    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w), dtype=np.uint8), 1.0, 0, 0

    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_h, target_w), dtype=np.uint8)

    pad_top = (target_h - new_h) // 2
    pad_left = (target_w - new_w) // 2

    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    return canvas, scale, pad_top, pad_left


def read_dicom_image(path: Path) -> Tuple[np.ndarray, Dict[str, str]]:
    """
    Read DICOM image and return pixel array and selected metadata.
    """
    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is not installed. Run: pip install pydicom")

    ds = pydicom.dcmread(str(path), force=True)

    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
    except Exception:
        arr = ds.pixel_array

    if arr.ndim == 3:
        arr = arr[0]

    metadata = {
        "patient_id_dicom": str(getattr(ds, "PatientID", "")),
        "study_instance_uid": str(getattr(ds, "StudyInstanceUID", "")),
        "series_instance_uid": str(getattr(ds, "SeriesInstanceUID", "")),
        "sop_instance_uid": str(getattr(ds, "SOPInstanceUID", "")),
        "modality": str(getattr(ds, "Modality", "")),
        "photometric_interpretation": str(getattr(ds, "PhotometricInterpretation", "")),
        "rows": str(getattr(ds, "Rows", "")),
        "columns": str(getattr(ds, "Columns", "")),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
        "body_part_examined": str(getattr(ds, "BodyPartExamined", "")),
        "view_position": str(getattr(ds, "ViewPosition", "")),
        "image_laterality": str(getattr(ds, "ImageLaterality", "")),
        "laterality": str(getattr(ds, "Laterality", "")),
        "study_date": str(getattr(ds, "StudyDate", "")),
    }

    # Apply MONOCHROME1 correction where needed.
    photo = metadata["photometric_interpretation"].upper().strip()
    dicom_inverted = False
    if photo == "MONOCHROME1":
        arr = np.max(arr) - arr
        dicom_inverted = True

    metadata["dicom_inverted"] = str(dicom_inverted)

    return arr, metadata


def read_png_image(path: Path) -> Tuple[np.ndarray, Dict[str, str]]:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Could not read PNG image: {path}")

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    metadata = {
        "patient_id_dicom": "",
        "study_instance_uid": "",
        "series_instance_uid": "",
        "sop_instance_uid": "",
        "modality": "PNG",
        "photometric_interpretation": "",
        "rows": str(img.shape[0]),
        "columns": str(img.shape[1]),
        "manufacturer": "",
        "body_part_examined": "",
        "view_position": "",
        "image_laterality": "",
        "laterality": "",
        "study_date": "",
        "dicom_inverted": "False",
    }

    return img, metadata


def infer_dataset_from_path(path: Path) -> str:
    parts = [p.lower() for p in path.parts]

    if "cbis-ddsm".lower() in parts:
        return "CBIS-DDSM"
    if "inbreast".lower() in parts:
        return "INbreast"
    if "vindr-mammo".lower() in parts:
        return "VinDr-Mammo"

    return "Unknown"


def infer_case_folder(path: Path, dataset_name: str) -> str:
    """
    For CBIS-DDSM, use folder containing Calc/Mass name.
    For others, use closest useful parent.
    """
    if dataset_name == "CBIS-DDSM":
        for parent in path.parents:
            name = parent.name
            if name.startswith(("Calc-", "Mass-")):
                return name

    if dataset_name == "INbreast":
        return path.parent.name

    if dataset_name == "VinDr-Mammo":
        return path.parent.name

    return path.parent.name


def infer_patient_id_from_case(case_folder: str, metadata: Dict[str, str]) -> str:
    if metadata.get("patient_id_dicom"):
        return metadata.get("patient_id_dicom", "")

    # CBIS case: Calc-Test_P_00038_LEFT_CC_1
    import re
    m = re.search(r"(P_\d+)", case_folder)
    if m:
        return m.group(1)

    return ""


def infer_view_from_case_or_metadata(case_folder: str, metadata: Dict[str, str]) -> str:
    meta_view = metadata.get("view_position", "").upper().strip()
    if meta_view in {"CC", "MLO"}:
        return meta_view

    name = case_folder.upper()
    if "_CC" in name or name.endswith("CC"):
        return "CC"
    if "_MLO" in name or name.endswith("MLO"):
        return "MLO"

    return ""


def infer_laterality_from_case_or_metadata(case_folder: str, metadata: Dict[str, str]) -> str:
    for key in ["image_laterality", "laterality"]:
        value = metadata.get(key, "").upper().strip()
        if value in {"L", "LEFT"}:
            return "LEFT"
        if value in {"R", "RIGHT"}:
            return "RIGHT"

    name = case_folder.upper()
    if "LEFT" in name:
        return "LEFT"
    if "RIGHT" in name:
        return "RIGHT"

    return ""


def infer_split_from_case(case_folder: str) -> str:
    name = case_folder.upper()
    if "TRAINING" in name:
        return "Training"
    if "TEST" in name:
        return "Test"
    return ""


def infer_lesion_type_from_case(case_folder: str) -> str:
    name = case_folder.upper()
    if name.startswith("CALC"):
        return "Calc"
    if name.startswith("MASS"):
        return "Mass"
    return ""


def make_output_path(
    dataset_name: str,
    source_path: Path,
    case_folder: str,
    patient_id: str,
    laterality: str,
    view: str,
) -> Path:
    safe_patient = patient_id.replace("\\", "_").replace("/", "_").replace(" ", "_")
    if safe_patient == "":
        safe_patient = "UnknownPatient"

    safe_case = case_folder.replace("\\", "_").replace("/", "_").replace(" ", "_")
    stem = source_path.stem

    out_dir = OUTPUT_IMAGE_ROOT / dataset_name / safe_patient
    ensure_dir(out_dir)

    prefix_parts = [
        safe_patient,
        laterality if laterality else "UNKLAT",
        view if view else "UNKVIEW",
        safe_case,
    ]
    prefix = "__".join(prefix_parts)

    return out_dir / f"{prefix}__{stem}{SAVE_FORMAT}"


# =============================================================================
# File Discovery
# =============================================================================

def discover_input_images() -> List[Path]:
    files: List[Path] = []

    # CBIS and INbreast DICOM
    for dataset_name in ["CBIS-DDSM", "INbreast"]:
        root = DATASET_ROOT / dataset_name
        if root.exists():
            files.extend(sorted(root.rglob("*.dcm")))

    # VinDr PNG
    vindr_root = DATASET_ROOT / "VinDr-Mammo"
    if vindr_root.exists():
        files.extend(sorted(vindr_root.rglob("*.png")))

    return files


# =============================================================================
# Processing
# =============================================================================

def preprocess_one_image(path: Path) -> Dict[str, object]:
    dataset_name = infer_dataset_from_path(path)
    case_folder = infer_case_folder(path, dataset_name)

    original_size_bytes = safe_file_size(path)
    suffix = path.suffix.lower()

    if suffix == ".dcm":
        arr, metadata = read_dicom_image(path)
        input_type = "DICOM"
    elif suffix == ".png":
        arr, metadata = read_png_image(path)
        input_type = "PNG"
    else:
        raise ValueError(f"Unsupported image type: {suffix}")

    original_shape = list(arr.shape)

    img_uint8 = normalize_uint8(arr)
    img_uint8, heuristic_inverted = maybe_invert_mammogram(img_uint8)
    cropped, bbox = crop_foreground_breast(img_uint8)

    final_img, scale, pad_top, pad_left = resize_with_padding(
        cropped,
        target_size=(TARGET_HEIGHT, TARGET_WIDTH),
    )

    patient_id = infer_patient_id_from_case(case_folder, metadata)
    view = infer_view_from_case_or_metadata(case_folder, metadata)
    laterality = infer_laterality_from_case_or_metadata(case_folder, metadata)
    split = infer_split_from_case(case_folder)
    lesion_type = infer_lesion_type_from_case(case_folder)

    output_path = make_output_path(
        dataset_name=dataset_name,
        source_path=path,
        case_folder=case_folder,
        patient_id=patient_id,
        laterality=laterality,
        view=view,
    )

    skipped_existing = False

    if output_path.exists() and SKIP_EXISTING:
        skipped_existing = True
    else:
        cv2.imwrite(
            str(output_path),
            final_img,
            [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION],
        )

    row = {
        "dataset": dataset_name,
        "input_type": input_type,
        "source_path": str(path),
        "processed_path": str(output_path),
        "case_folder": case_folder,
        "patient_id": patient_id,
        "split": split,
        "lesion_type": lesion_type,
        "laterality": laterality,
        "view": view,
        "original_size_bytes": original_size_bytes,
        "original_shape": json.dumps(original_shape),
        "target_height": TARGET_HEIGHT,
        "target_width": TARGET_WIDTH,
        "crop_x": bbox[0],
        "crop_y": bbox[1],
        "crop_w": bbox[2],
        "crop_h": bbox[3],
        "resize_scale": scale,
        "pad_top": pad_top,
        "pad_left": pad_left,
        "dicom_inverted": metadata.get("dicom_inverted", ""),
        "heuristic_inverted": heuristic_inverted,
        "modality": metadata.get("modality", ""),
        "study_instance_uid": metadata.get("study_instance_uid", ""),
        "series_instance_uid": metadata.get("series_instance_uid", ""),
        "sop_instance_uid": metadata.get("sop_instance_uid", ""),
        "study_date": metadata.get("study_date", ""),
        "manufacturer": metadata.get("manufacturer", ""),
        "status": "SKIPPED_EXISTING" if skipped_existing else "PROCESSED",
    }

    return row


def process_all_images() -> Tuple[pd.DataFrame, pd.DataFrame]:
    input_files = discover_input_images()

    rows: List[Dict[str, object]] = []
    failures: List[Dict[str, object]] = []

    for path in tqdm(input_files, desc="Preprocessing images", unit="img"):
        try:
            row = preprocess_one_image(path)
            rows.append(row)

        except Exception as e:
            failures.append({
                "source_path": str(path),
                "dataset": infer_dataset_from_path(path),
                "error": str(e),
                "traceback": traceback.format_exc(),
            })

    return pd.DataFrame(rows), pd.DataFrame(failures)


# =============================================================================
# Summary and Report
# =============================================================================

def build_summary(index_df: pd.DataFrame, failures_df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    if index_df.empty:
        for dataset_name in DATASETS:
            rows.append({
                "dataset": dataset_name,
                "processed_or_skipped": 0,
                "processed_new": 0,
                "skipped_existing": 0,
                "failures": int((failures_df["dataset"] == dataset_name).sum()) if not failures_df.empty else 0,
                "unique_patients": 0,
                "cc_images": 0,
                "mlo_images": 0,
                "left_images": 0,
                "right_images": 0,
            })
        return pd.DataFrame(rows)

    for dataset_name in DATASETS:
        g = index_df[index_df["dataset"] == dataset_name].copy()

        if failures_df.empty:
            fail_count = 0
        else:
            fail_count = int((failures_df["dataset"] == dataset_name).sum())

        rows.append({
            "dataset": dataset_name,
            "processed_or_skipped": int(len(g)),
            "processed_new": int((g["status"] == "PROCESSED").sum()) if not g.empty else 0,
            "skipped_existing": int((g["status"] == "SKIPPED_EXISTING").sum()) if not g.empty else 0,
            "failures": fail_count,
            "unique_patients": int(g["patient_id"].nunique()) if not g.empty and "patient_id" in g.columns else 0,
            "cc_images": int((g["view"] == "CC").sum()) if not g.empty else 0,
            "mlo_images": int((g["view"] == "MLO").sum()) if not g.empty else 0,
            "left_images": int((g["laterality"] == "LEFT").sum()) if not g.empty else 0,
            "right_images": int((g["laterality"] == "RIGHT").sum()) if not g.empty else 0,
        })

    rows.append({
        "dataset": "ALL",
        "processed_or_skipped": int(len(index_df)),
        "processed_new": int((index_df["status"] == "PROCESSED").sum()),
        "skipped_existing": int((index_df["status"] == "SKIPPED_EXISTING").sum()),
        "failures": int(len(failures_df)),
        "unique_patients": int(index_df["patient_id"].nunique()) if "patient_id" in index_df.columns else 0,
        "cc_images": int((index_df["view"] == "CC").sum()) if "view" in index_df.columns else 0,
        "mlo_images": int((index_df["view"] == "MLO").sum()) if "view" in index_df.columns else 0,
        "left_images": int((index_df["laterality"] == "LEFT").sum()) if "laterality" in index_df.columns else 0,
        "right_images": int((index_df["laterality"] == "RIGHT").sum()) if "laterality" in index_df.columns else 0,
    })

    return pd.DataFrame(rows)


def write_report(
    index_df: pd.DataFrame,
    failures_df: pd.DataFrame,
    summary_df: pd.DataFrame,
) -> None:
    lines: List[str] = []

    lines.append("=" * 100)
    lines.append("STAGE 1A DICOM AND IMAGE PREPROCESSING REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append(f"Dataset root: {DATASET_ROOT}")
    lines.append(f"Processed image root: {OUTPUT_IMAGE_ROOT}")
    lines.append(f"Target size: {TARGET_HEIGHT} x {TARGET_WIDTH}")
    lines.append("")

    lines.append("SUMMARY")
    lines.append("-" * 100)
    if not summary_df.empty:
        lines.append(summary_df.to_string(index=False))
    else:
        lines.append("No images processed.")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    lines.append(str(OUTPUT_INDEX_CSV))
    lines.append(str(OUTPUT_SUMMARY_CSV))
    lines.append(str(OUTPUT_FAILURES_CSV))
    lines.append(str(OUTPUT_JSON))
    lines.append(str(OUTPUT_REPORT_TXT))
    lines.append("")

    if not failures_df.empty:
        lines.append("FAILURE SAMPLE")
        lines.append("-" * 100)
        lines.append(failures_df.head(20).to_string(index=False))
        lines.append("")

    with open(OUTPUT_REPORT_TXT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_json_summary(summary_df: pd.DataFrame) -> None:
    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "processed_image_root": str(OUTPUT_IMAGE_ROOT),
        "target_height": TARGET_HEIGHT,
        "target_width": TARGET_WIDTH,
        "summary": summary_df.to_dict(orient="records"),
    }

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 100)
    print("STAGE 1A PREPROCESS DICOM AND IMAGES")
    print("=" * 100)
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output image root: {OUTPUT_IMAGE_ROOT}")
    print(f"Target size: {TARGET_HEIGHT} x {TARGET_WIDTH}")
    print("-" * 100)

    if not PYDICOM_AVAILABLE:
        raise ImportError("pydicom is required. Install with: pip install pydicom")

    index_df, failures_df = process_all_images()
    summary_df = build_summary(index_df, failures_df)

    index_df.to_csv(OUTPUT_INDEX_CSV, index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    failures_df.to_csv(OUTPUT_FAILURES_CSV, index=False, encoding="utf-8-sig")

    save_json_summary(summary_df)
    write_report(index_df, failures_df, summary_df)

    print()
    print("STAGE 1A COMPLETED")
    print("-" * 100)
    print(f"Preprocessed image index: {OUTPUT_INDEX_CSV}")
    print(f"Preprocessing summary:    {OUTPUT_SUMMARY_CSV}")
    print(f"Failures file:            {OUTPUT_FAILURES_CSV}")
    print(f"JSON summary:             {OUTPUT_JSON}")
    print(f"Text report:              {OUTPUT_REPORT_TXT}")
    print("=" * 100)


if __name__ == "__main__":
    main()