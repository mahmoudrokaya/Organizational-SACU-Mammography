r"""
Stage2D2B_Clean_Exact_SACU_Predictions_for_Stage2E.py

Purpose
-------
Clean Stage2D2 exported predictions and keep only valid probability models for Stage2E.

Output
------
Stage2D2B_Stage2E_Ready_Clean_Predictions.csv
"""

from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")
TABLE_DIR = PROJECT_ROOT / "results" / "tables"
REPORT_DIR = PROJECT_ROOT / "results" / "reports"

INPUT_FILE = TABLE_DIR / "Stage2D2_Stage2E_Ready_Long_Predictions.csv"
OUTPUT_FILE = TABLE_DIR / "Stage2D2B_Stage2E_Ready_Clean_Predictions.csv"
REPORT_FILE = REPORT_DIR / "Stage2D2B_Clean_Exact_SACU_Predictions_Report.txt"

KEEP_MODELS = [
    "LearnedShallowMetaFusion_score",
    "AdaptiveSACUWeightFusion_score",
    "FixedEqualWeightFusion_score",
    "LocalRegionalAgent_score",
    "MultiViewAgent_score",
    "BilateralAgent_score",
    "TemporalSpatialAgent_score",
    "MetadataAgent_score",
    "AdaptiveControlAgent_score",
]

RENAME_MODELS = {
    "LearnedShallowMetaFusion_score": "SACU_LearnedShallowMetaFusion",
    "AdaptiveSACUWeightFusion_score": "SACU_AdaptiveWeightFusion",
    "FixedEqualWeightFusion_score": "SACU_FixedEqualWeightFusion",
    "LocalRegionalAgent_score": "LocalRegionalAgent",
    "MultiViewAgent_score": "MultiViewAgent",
    "BilateralAgent_score": "BilateralAgent",
    "TemporalSpatialAgent_score": "TemporalSpatialAgent",
    "MetadataAgent_score": "MetadataAgent",
    "AdaptiveControlAgent_score": "AdaptiveControlAgent",
}

def main():
    df = pd.read_csv(INPUT_FILE)

    clean = df[df["model"].isin(KEEP_MODELS)].copy()
    clean["model"] = clean["model"].replace(RENAME_MODELS)

    clean["y_true"] = pd.to_numeric(clean["y_true"], errors="coerce").astype(int)
    clean["y_score"] = pd.to_numeric(clean["y_score"], errors="coerce").clip(0, 1)

    clean = clean.dropna(subset=["y_true", "y_score"])

    clean.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    lines = []
    lines.append("=" * 100)
    lines.append("STAGE 2D2B CLEAN EXACT SACU PREDICTIONS FOR STAGE 2E")
    lines.append("=" * 100)
    lines.append(f"Input file: {INPUT_FILE}")
    lines.append(f"Output file: {OUTPUT_FILE}")
    lines.append("")
    lines.append("Models retained:")
    lines.append(clean["model"].value_counts().to_string())
    lines.append("")
    lines.append("Models removed:")
    removed = sorted(set(df["model"].unique()) - set(KEEP_MODELS))
    for m in removed:
        lines.append(f"- {m}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("=" * 100)
    print("STAGE 2D2B COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Clean Stage2E predictions: {OUTPUT_FILE}")
    print(f"Report: {REPORT_FILE}")
    print("=" * 100)

if __name__ == "__main__":
    main()

