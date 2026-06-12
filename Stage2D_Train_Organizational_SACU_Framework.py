r"""
Stage2D_Train_Organizational_SACU_Framework.py

Purpose
-------
Train the proposed Organizational SACU framework using the leakage-safe
Stage2B matrices.

This stage implements the paper's proposed method after the baseline stage.

Method consistency
------------------
The implementation follows the Methods structure:

1. Local Regional Agent
   - Uses local regional descriptors.

2. Multi-View Agent
   - Uses CC/MLO integration features.

3. Bilateral Agent
   - Uses left-right asymmetry features.

4. Same-Exam Temporal-Spatial Agent
   - Uses same-exam ordered view-sequence descriptors.
   - Does not claim real longitudinal disease-course modeling.

5. Metadata Agent
   - Uses non-leaking clinical metadata features.

6. Adaptive Control Agent
   - Uses SACU complexity/resource cues.

7. Organizational Fusion
   - Each agent outputs a malignancy probability.
   - Agent reliability and case-level confidence are estimated.
   - Adaptive influence weights are computed per case.
   - Final prediction is the weighted organizational fusion of agent outputs.

8. Required ablation
   - Fixed equal-weight fusion
   - Adaptive SACU-weight fusion
   - Leave-one-agent-out ablation

Inputs
------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_train.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_X_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_y_test.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_Feature_Groups.json
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_train_metadata.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\matrices\\Stage2B_test_metadata.csv

Outputs
-------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Agent_Level_Performance.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Agent_Predictions.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Adaptive_Weights.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Organizational_Fusion_Performance.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Ablation_Study.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_Confusion_Matrices.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_ROC_Curves.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\tables\\Stage2D_PR_Curves.csv
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\models\\Stage2D_Organizational_SACU_Model.joblib
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\results\\reports\\Stage2D_Organizational_SACU_Report.txt

Recommended save location
-------------------------
D:\\47\\472\\New-Papers\\Frontires_Atlam2-2026\\Experiments\\scripts\\Stage2D_Train_Organizational_SACU_Framework.py
"""

from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
    brier_score_loss,
)


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path(r"D:\47\472\New-Papers\Frontires_Atlam2-2026\Experiments")

MATRIX_DIR = PROJECT_ROOT / "matrices"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_TABLE_DIR = PROJECT_ROOT / "results" / "tables"
RESULTS_REPORT_DIR = PROJECT_ROOT / "results" / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_TABLE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_REPORT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_X_TRAIN = MATRIX_DIR / "Stage2B_X_train.csv"
INPUT_Y_TRAIN = MATRIX_DIR / "Stage2B_y_train.csv"
INPUT_X_TEST = MATRIX_DIR / "Stage2B_X_test.csv"
INPUT_Y_TEST = MATRIX_DIR / "Stage2B_y_test.csv"
INPUT_FEATURE_GROUPS = MATRIX_DIR / "Stage2B_Feature_Groups.json"
INPUT_TRAIN_META = MATRIX_DIR / "Stage2B_train_metadata.csv"
INPUT_TEST_META = MATRIX_DIR / "Stage2B_test_metadata.csv"

OUTPUT_AGENT_PERFORMANCE = RESULTS_TABLE_DIR / "Stage2D_Agent_Level_Performance.csv"
OUTPUT_AGENT_PREDICTIONS = RESULTS_TABLE_DIR / "Stage2D_Agent_Predictions.csv"
OUTPUT_ADAPTIVE_WEIGHTS = RESULTS_TABLE_DIR / "Stage2D_Adaptive_Weights.csv"
OUTPUT_FUSION_PERFORMANCE = RESULTS_TABLE_DIR / "Stage2D_Organizational_Fusion_Performance.csv"
OUTPUT_ABLATION = RESULTS_TABLE_DIR / "Stage2D_Ablation_Study.csv"
OUTPUT_CONFUSION = RESULTS_TABLE_DIR / "Stage2D_Confusion_Matrices.csv"
OUTPUT_ROC = RESULTS_TABLE_DIR / "Stage2D_ROC_Curves.csv"
OUTPUT_PR = RESULTS_TABLE_DIR / "Stage2D_PR_Curves.csv"
OUTPUT_CALIBRATION = RESULTS_TABLE_DIR / "Stage2D_Calibration_Curves.csv"
OUTPUT_STATISTICAL = RESULTS_TABLE_DIR / "Stage2D_Statistical_Comparison.csv"
OUTPUT_SUMMARY_JSON = RESULTS_TABLE_DIR / "Stage2D_Organizational_SACU_Summary.json"
OUTPUT_MODEL = MODEL_DIR / "Stage2D_Organizational_SACU_Model.joblib"
OUTPUT_REPORT = RESULTS_REPORT_DIR / "Stage2D_Organizational_SACU_Report.txt"

RANDOM_SEED = 42
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 2000
EPS = 1e-8

POS_LABEL = 1
NEG_LABEL = 0


# =============================================================================
# Agent Definition
# =============================================================================

AGENT_GROUP_MAP = {
    "LocalRegionalAgent": ["local_regional_descriptor"],
    "MultiViewAgent": ["multi_view_reasoning"],
    "BilateralAgent": ["bilateral_asymmetry"],
    "TemporalSpatialAgent": ["same_exam_temporal_spatial"],
    "MetadataAgent": ["clinical_metadata"],
    "AdaptiveControlAgent": ["adaptive_sacu_control"],
}

AGENT_ROLE_DESCRIPTION = {
    "LocalRegionalAgent": "Processes localized anatomical regional descriptors from each mammographic view.",
    "MultiViewAgent": "Models CC/MLO complementary information within each breast.",
    "BilateralAgent": "Models left-right breast asymmetry.",
    "TemporalSpatialAgent": "Models same-exam ordered view-sequence descriptors; not real longitudinal follow-up.",
    "MetadataAgent": "Uses non-leaking clinical metadata covariates.",
    "AdaptiveControlAgent": "Uses SACU complexity/resource cues to support adaptive organizational coordination.",
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


def read_csv_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required input not found: {path}")
    return pd.read_csv(path, low_memory=False)


def load_json_required(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_y(df: pd.DataFrame) -> np.ndarray:
    if "target" in df.columns:
        return df["target"].astype(int).values
    if df.shape[1] == 1:
        return df.iloc[:, 0].astype(int).values
    raise ValueError("Could not identify target column.")


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def normalize_rows(mat: np.ndarray) -> np.ndarray:
    denom = mat.sum(axis=1, keepdims=True)
    denom = np.where(denom <= EPS, 1.0, denom)
    return mat / denom


def threshold_predictions(scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (scores >= threshold).astype(int)


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def np_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def select_existing_columns(X: pd.DataFrame, columns: List[str]) -> List[str]:
    return [c for c in columns if c in X.columns]


def build_feature_groups_for_agents(feature_groups: Dict[str, List[str]], X: pd.DataFrame) -> Dict[str, List[str]]:
    agent_features = {}

    for agent, groups in AGENT_GROUP_MAP.items():
        cols = []
        for group in groups:
            cols.extend(feature_groups.get(group, []))
        cols = select_existing_columns(X, cols)
        agent_features[agent] = cols

    return agent_features


# =============================================================================
# Models and Prediction
# =============================================================================

def build_agent_model() -> GradientBoostingClassifier:
    """
    GradientBoosting is used because Stage2C identified it as the strongest
    shallow baseline, and it remains consistent with the shallow-agent design.
    """
    return GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=2,
        random_state=RANDOM_SEED,
    )


def train_agent_models(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    agent_features: Dict[str, List[str]],
) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, Dict[str, float]]:

    models = {}
    train_scores = pd.DataFrame(index=X_train.index)
    test_scores = pd.DataFrame(index=X_test.index)
    train_times = {}

    for agent, cols in agent_features.items():
        if not cols:
            print(f"Skipping {agent}: no features.")
            continue

        print(f"Training {agent} with {len(cols)} features...")
        model = build_agent_model()

        start = time.perf_counter()
        model.fit(X_train[cols], y_train)
        train_time = time.perf_counter() - start

        train_prob = model.predict_proba(X_train[cols])[:, 1]
        test_prob = model.predict_proba(X_test[cols])[:, 1]

        models[agent] = model
        train_scores[agent] = train_prob
        test_scores[agent] = test_prob
        train_times[agent] = float(train_time)

    return models, train_scores, test_scores, train_times


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    y_pred = threshold_predictions(scores, threshold)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    out = {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "sensitivity_recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity_from_confusion(tn, fp),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "pr_auc": average_precision_score(y_true, scores) if len(np.unique(y_true)) == 2 else np.nan,
        "brier_score": brier_score_loss(y_true, scores),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return {k: np_float(v) if isinstance(v, (np.floating, float, int, np.integer)) else v for k, v in out.items()}


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric_name: str,
    threshold: float = 0.5,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    vals = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_b = y_true[idx]
        s_b = scores[idx]

        if len(np.unique(y_b)) < 2:
            continue

        try:
            vals.append(compute_metrics(y_b, s_b, threshold)[metric_name])
        except Exception:
            continue

    if not vals:
        return np.nan, np.nan

    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def find_best_threshold_youden(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    youden = tpr - fpr
    idx = int(np.argmax(youden))
    thr = float(thresholds[idx])
    if np.isinf(thr) or np.isnan(thr):
        return 0.5
    return thr


def evaluate_scores(
    name: str,
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float = 0.5,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    row = {
        "model_or_agent": name,
        **compute_metrics(y_true, scores, threshold),
    }

    for metric in ["roc_auc", "pr_auc", "balanced_accuracy", "sensitivity_recall", "specificity", "f1"]:
        lo, hi = bootstrap_ci(y_true, scores, metric, threshold)
        row[f"{metric}_ci95_low"] = lo
        row[f"{metric}_ci95_high"] = hi

    best_thr = find_best_threshold_youden(y_true, scores)
    best_metrics = compute_metrics(y_true, scores, best_thr)

    row["youden_threshold"] = best_thr
    row["youden_balanced_accuracy"] = best_metrics["balanced_accuracy"]
    row["youden_sensitivity"] = best_metrics["sensitivity_recall"]
    row["youden_specificity"] = best_metrics["specificity"]
    row["youden_f1"] = best_metrics["f1"]

    if extra:
        row.update(extra)

    return row


def build_confusion_row(name: str, y_true: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, object]:
    y_pred = threshold_predictions(scores, threshold)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "model_or_agent": name,
        "threshold": threshold,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_curve_tables(name: str, y_true: np.ndarray, scores: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    fpr, tpr, roc_thr = roc_curve(y_true, scores)
    precision, recall, pr_thr = precision_recall_curve(y_true, scores)

    roc_df = pd.DataFrame({
        "model_or_agent": name,
        "fpr": fpr,
        "tpr_sensitivity": tpr,
        "threshold": roc_thr,
    })

    pr_thresholds = list(pr_thr) + [np.nan]

    pr_df = pd.DataFrame({
        "model_or_agent": name,
        "precision": precision,
        "recall_sensitivity": recall,
        "threshold": pr_thresholds,
    })

    return roc_df, pr_df


def build_calibration_table(name: str, y_true: np.ndarray, scores: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({
        "y_true": y_true,
        "score": scores,
    })

    df["bin"] = pd.cut(
        df["score"],
        bins=np.linspace(0.0, 1.0, n_bins + 1),
        include_lowest=True,
        labels=False,
    )

    rows = []

    for b, g in df.groupby("bin", dropna=False):
        if pd.isna(b):
            continue

        rows.append({
            "model_or_agent": name,
            "bin": int(b),
            "n": int(len(g)),
            "mean_predicted_probability": float(g["score"].mean()),
            "observed_positive_rate": float(g["y_true"].mean()),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Organizational Fusion
# =============================================================================

def agent_reliability_from_train(y_train: np.ndarray, train_scores: pd.DataFrame) -> Dict[str, float]:
    """
    Compute global agent reliability from training ROC-AUC and PR-AUC.
    """
    rel = {}

    for agent in train_scores.columns:
        s = train_scores[agent].values

        try:
            auc = roc_auc_score(y_train, s)
        except Exception:
            auc = 0.5

        try:
            ap = average_precision_score(y_train, s)
        except Exception:
            ap = np.mean(y_train)

        # Convert to positive reliability.
        # AUC below 0.5 should not receive negative influence.
        rel_score = max(auc - 0.5, 0.0) + 0.25 * max(ap - np.mean(y_train), 0.0)

        rel[agent] = float(rel_score + 0.05)

    return rel


def case_confidence(scores: np.ndarray) -> np.ndarray:
    """
    Confidence is distance from uncertainty.
    """
    return np.abs(scores - 0.5) * 2.0


def compute_adaptive_weights(
    test_scores: pd.DataFrame,
    train_reliability: Dict[str, float],
    meta_test: pd.DataFrame,
) -> pd.DataFrame:
    """
    Adaptive SACU weighting:
    per-case weight = global reliability * case confidence * role activation prior.

    Role activation prior reflects the SACU mechanism:
    - higher complexity slightly increases multi-view, bilateral, and temporal-spatial influence
    - metadata remains low but stable
    - adaptive control contributes to coordination but is not dominant
    """
    agents = list(test_scores.columns)

    weight_raw = pd.DataFrame(index=test_scores.index)

    # Complexity proxy from metadata if available; otherwise use score disagreement.
    score_matrix = test_scores[agents].values
    disagreement = np.std(score_matrix, axis=1)

    # Normalize disagreement to [0,1].
    if disagreement.max() > disagreement.min():
        complexity = (disagreement - disagreement.min()) / (disagreement.max() - disagreement.min())
    else:
        complexity = np.zeros_like(disagreement)

    for agent in agents:
        s = test_scores[agent].values
        conf = case_confidence(s)
        reliability = train_reliability.get(agent, 0.05)

        if agent in {"MultiViewAgent", "BilateralAgent", "TemporalSpatialAgent"}:
            role_prior = 1.0 + 0.50 * complexity
        elif agent == "AdaptiveControlAgent":
            role_prior = 0.8 + 0.35 * complexity
        elif agent == "MetadataAgent":
            role_prior = 0.75
        else:
            role_prior = 1.0

        weight_raw[agent] = reliability * (0.25 + 0.75 * conf) * role_prior

    weights = pd.DataFrame(
        normalize_rows(weight_raw.values),
        columns=agents,
        index=test_scores.index,
    )

    weights["case_complexity_from_agent_disagreement"] = complexity

    return weights


def fuse_scores_equal(test_scores: pd.DataFrame) -> np.ndarray:
    return test_scores.mean(axis=1).values


def fuse_scores_adaptive(test_scores: pd.DataFrame, weights: pd.DataFrame) -> np.ndarray:
    agent_cols = list(test_scores.columns)
    return np.sum(test_scores[agent_cols].values * weights[agent_cols].values, axis=1)


def train_meta_fusion(
    train_scores: pd.DataFrame,
    y_train: np.ndarray,
    test_scores: pd.DataFrame,
) -> Tuple[np.ndarray, object]:
    """
    Optional shallow learned organizational fusion for comparison.
    This is not the primary adaptive coordinator, but a learned shallow fusion baseline.
    """
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        class_weight="balanced",
        solver="liblinear",
        max_iter=5000,
        random_state=RANDOM_SEED,
    )

    model.fit(train_scores.values, y_train)
    fused = model.predict_proba(test_scores.values)[:, 1]

    return fused, model


# =============================================================================
# Ablations
# =============================================================================

def build_ablation_scores(
    y_test: np.ndarray,
    test_scores: pd.DataFrame,
    adaptive_weights: pd.DataFrame,
    equal_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    learned_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []

    fusion_variants = {
        "FixedEqualWeightFusion": equal_scores,
        "AdaptiveSACUWeightFusion": adaptive_scores,
        "LearnedShallowMetaFusion": learned_scores,
    }

    for name, scores in fusion_variants.items():
        rows.append({
            "ablation": name,
            "ablation_type": "coordination_mechanism",
            **compute_metrics(y_test, scores, 0.5),
            "youden_threshold": find_best_threshold_youden(y_test, scores),
        })

    agents = list(test_scores.columns)

    for removed_agent in agents:
        keep_agents = [a for a in agents if a != removed_agent]

        if not keep_agents:
            continue

        sub_scores = test_scores[keep_agents].copy()
        sub_weights = adaptive_weights[keep_agents].copy()
        sub_weights = pd.DataFrame(
            normalize_rows(sub_weights.values),
            columns=keep_agents,
            index=sub_weights.index,
        )

        fused = np.sum(sub_scores.values * sub_weights.values, axis=1)

        rows.append({
            "ablation": f"Remove_{removed_agent}",
            "ablation_type": "leave_one_agent_out",
            "removed_agent": removed_agent,
            **compute_metrics(y_test, fused, 0.5),
            "youden_threshold": find_best_threshold_youden(y_test, fused),
        })

    # Single-agent variants
    for agent in agents:
        rows.append({
            "ablation": f"Only_{agent}",
            "ablation_type": "single_agent",
            "included_agent": agent,
            **compute_metrics(y_test, test_scores[agent].values, 0.5),
            "youden_threshold": find_best_threshold_youden(y_test, test_scores[agent].values),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Statistical Comparison
# =============================================================================

def permutation_test_auc_difference(
    y_true: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = N_PERMUTATIONS,
    seed: int = RANDOM_SEED,
) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)

    observed = roc_auc_score(y_true, scores_a) - roc_auc_score(y_true, scores_b)

    diffs = []

    for _ in range(n_permutations):
        swap = rng.random(len(y_true)) < 0.5

        perm_a = scores_a.copy()
        perm_b = scores_b.copy()

        perm_a[swap], perm_b[swap] = perm_b[swap], perm_a[swap]

        try:
            diffs.append(roc_auc_score(y_true, perm_a) - roc_auc_score(y_true, perm_b))
        except Exception:
            continue

    if not diffs:
        return float(observed), np.nan

    diffs = np.asarray(diffs)
    p_value = float(np.mean(np.abs(diffs) >= abs(observed)))

    return float(observed), p_value


def build_statistical_comparison(y_test: np.ndarray, score_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []

    names = list(score_map.keys())

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]

            diff, p = permutation_test_auc_difference(
                y_true=y_test,
                scores_a=score_map[a],
                scores_b=score_map[b],
            )

            rows.append({
                "model_a": a,
                "model_b": b,
                "roc_auc_difference_a_minus_b": diff,
                "paired_permutation_p_value": p,
                "test": "paired_permutation_test_on_roc_auc",
                "n_permutations": N_PERMUTATIONS,
            })

    return pd.DataFrame(rows)


# =============================================================================
# Output Builders
# =============================================================================

def build_agent_performance(
    y_test: np.ndarray,
    test_scores: pd.DataFrame,
    agent_features: Dict[str, List[str]],
    train_times: Dict[str, float],
) -> pd.DataFrame:
    rows = []

    for agent in test_scores.columns:
        rows.append(evaluate_scores(
            name=agent,
            y_true=y_test,
            scores=test_scores[agent].values,
            threshold=0.5,
            extra={
                "agent_role": AGENT_ROLE_DESCRIPTION.get(agent, ""),
                "n_features": len(agent_features.get(agent, [])),
                "train_time_sec": train_times.get(agent, np.nan),
            },
        ))

    return pd.DataFrame(rows)


def build_fusion_performance(
    y_test: np.ndarray,
    equal_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    learned_scores: np.ndarray,
) -> pd.DataFrame:
    rows = []

    rows.append(evaluate_scores(
        "FixedEqualWeightFusion",
        y_test,
        equal_scores,
        0.5,
        {"fusion_type": "fixed_equal_weights"},
    ))

    rows.append(evaluate_scores(
        "AdaptiveSACUWeightFusion",
        y_test,
        adaptive_scores,
        0.5,
        {"fusion_type": "case_level_adaptive_weights"},
    ))

    rows.append(evaluate_scores(
        "LearnedShallowMetaFusion",
        y_test,
        learned_scores,
        0.5,
        {"fusion_type": "shallow_logistic_meta_fusion"},
    ))

    return pd.DataFrame(rows)


def build_prediction_table(
    y_test: np.ndarray,
    test_meta: pd.DataFrame,
    test_scores: pd.DataFrame,
    equal_scores: np.ndarray,
    adaptive_scores: np.ndarray,
    learned_scores: np.ndarray,
) -> pd.DataFrame:
    out = test_meta.copy()
    out["target"] = y_test

    for agent in test_scores.columns:
        out[f"{agent}_score"] = test_scores[agent].values

    out["FixedEqualWeightFusion_score"] = equal_scores
    out["AdaptiveSACUWeightFusion_score"] = adaptive_scores
    out["LearnedShallowMetaFusion_score"] = learned_scores

    out["AdaptiveSACUWeightFusion_pred_0p5"] = threshold_predictions(adaptive_scores, 0.5)

    return out


def build_weight_table(test_meta: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    out = test_meta.copy().reset_index(drop=True)
    w = weights.reset_index(drop=True)

    return pd.concat([out, w], axis=1)


def build_confusion_matrices(y_test: np.ndarray, score_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    for name, scores in score_map.items():
        rows.append(build_confusion_row(name, y_test, scores, threshold=0.5))
    return pd.DataFrame(rows)


def build_curves(y_test: np.ndarray, score_map: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    roc_frames = []
    pr_frames = []

    for name, scores in score_map.items():
        roc_df, pr_df = build_curve_tables(name, y_test, scores)
        roc_frames.append(roc_df)
        pr_frames.append(pr_df)

    return pd.concat(roc_frames, ignore_index=True), pd.concat(pr_frames, ignore_index=True)


def build_calibration(y_test: np.ndarray, score_map: Dict[str, np.ndarray]) -> pd.DataFrame:
    frames = []
    for name, scores in score_map.items():
        frames.append(build_calibration_table(name, y_test, scores))
    return pd.concat(frames, ignore_index=True)


# =============================================================================
# Save and Report
# =============================================================================

def save_json_summary(
    agent_perf: pd.DataFrame,
    fusion_perf: pd.DataFrame,
    ablation_df: pd.DataFrame,
    agent_features: Dict[str, List[str]],
) -> None:
    best_fusion_row = fusion_perf.sort_values("roc_auc", ascending=False).iloc[0]

    data = {
        "generated": str(datetime.now()),
        "project_root": str(PROJECT_ROOT),
        "method_consistency": {
            "local_agent": "local regional descriptors",
            "multiview_agent": "CC/MLO integration descriptors",
            "bilateral_agent": "left-right asymmetry descriptors",
            "temporal_spatial_agent": "same-exam temporal-spatial descriptors only",
            "metadata_agent": "non-leaking metadata features",
            "adaptive_control_agent": "SACU complexity/resource cues",
            "fusion": "case-level adaptive influence weights",
            "ablation": "fixed equal weights vs adaptive weights and leave-one-agent-out",
            "longitudinal_claim": "No real longitudinal modeling introduced",
        },
        "agent_feature_counts": {k: len(v) for k, v in agent_features.items()},
        "best_fusion": best_fusion_row.to_dict(),
        "agent_performance": agent_perf.to_dict(orient="records"),
        "fusion_performance": fusion_perf.to_dict(orient="records"),
        "ablation": ablation_df.to_dict(orient="records"),
    }

    with open(OUTPUT_SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def write_report(
    agent_perf: pd.DataFrame,
    fusion_perf: pd.DataFrame,
    ablation_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    statistical_df: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    agent_features: Dict[str, List[str]],
) -> None:
    lines = []

    lines.append("=" * 100)
    lines.append("STAGE 2D ORGANIZATIONAL SACU FRAMEWORK REPORT")
    lines.append("=" * 100)
    lines.append(f"Generated: {datetime.now()}")
    lines.append(f"Project root: {PROJECT_ROOT}")
    lines.append("")

    lines.append("METHOD CONSISTENCY")
    lines.append("-" * 100)
    lines.append("Implemented SACU as a population of shallow role-conditioned agents.")
    lines.append("Agent roles correspond to local regional, multi-view, bilateral, same-exam temporal-spatial, metadata, and adaptive-control pathways.")
    lines.append("Each agent uses GradientBoosting as a shallow learner, consistent with Stage2C baseline evidence.")
    lines.append("Organizational fusion uses agent probabilities and adaptive case-level influence weights.")
    lines.append("The adaptive coordination ablation compares fixed equal weights against adaptive SACU weights.")
    lines.append("No real longitudinal disease-course feature or claim is introduced.")
    lines.append("")

    lines.append("DATA SHAPE")
    lines.append("-" * 100)
    lines.append(f"X_train: {X_train.shape}")
    lines.append(f"X_test: {X_test.shape}")
    lines.append(f"Train positives: {int((y_train == 1).sum())}")
    lines.append(f"Train negatives: {int((y_train == 0).sum())}")
    lines.append(f"Test positives: {int((y_test == 1).sum())}")
    lines.append(f"Test negatives: {int((y_test == 0).sum())}")
    lines.append("")

    lines.append("AGENT FEATURE COUNTS")
    lines.append("-" * 100)
    for agent, cols in agent_features.items():
        lines.append(f"{agent}: {len(cols)}")
    lines.append("")

    lines.append("AGENT-LEVEL PERFORMANCE")
    lines.append("-" * 100)
    display_cols = [
        "model_or_agent",
        "n_features",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]
    lines.append(agent_perf[display_cols].to_string(index=False))
    lines.append("")

    lines.append("ORGANIZATIONAL FUSION PERFORMANCE")
    lines.append("-" * 100)
    fusion_cols = [
        "model_or_agent",
        "fusion_type",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity_recall",
        "specificity",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "youden_threshold",
        "youden_balanced_accuracy",
        "youden_sensitivity",
        "youden_specificity",
    ]
    lines.append(fusion_perf[fusion_cols].to_string(index=False))
    lines.append("")

    lines.append("ABLATION STUDY")
    lines.append("-" * 100)
    ablation_cols = [
        "ablation",
        "ablation_type",
        "accuracy",
        "balanced_accuracy",
        "sensitivity_recall",
        "specificity",
        "f1",
        "roc_auc",
        "pr_auc",
    ]
    lines.append(ablation_df[ablation_cols].to_string(index=False))
    lines.append("")

    lines.append("CONFUSION MATRICES")
    lines.append("-" * 100)
    lines.append(confusion_df.to_string(index=False))
    lines.append("")

    lines.append("STATISTICAL COMPARISON")
    lines.append("-" * 100)
    lines.append(statistical_df.to_string(index=False) if not statistical_df.empty else "No statistical comparison.")
    lines.append("")

    lines.append("OUTPUT FILES")
    lines.append("-" * 100)
    for p in [
        OUTPUT_AGENT_PERFORMANCE,
        OUTPUT_AGENT_PREDICTIONS,
        OUTPUT_ADAPTIVE_WEIGHTS,
        OUTPUT_FUSION_PERFORMANCE,
        OUTPUT_ABLATION,
        OUTPUT_CONFUSION,
        OUTPUT_ROC,
        OUTPUT_PR,
        OUTPUT_CALIBRATION,
        OUTPUT_STATISTICAL,
        OUTPUT_SUMMARY_JSON,
        OUTPUT_MODEL,
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
    print("STAGE 2D TRAIN ORGANIZATIONAL SACU FRAMEWORK")
    print("=" * 100)
    print(f"Project root: {PROJECT_ROOT}")
    print("-" * 100)

    print("Loading matrices...")
    X_train = read_csv_required(INPUT_X_TRAIN)
    y_train_df = read_csv_required(INPUT_Y_TRAIN)
    X_test = read_csv_required(INPUT_X_TEST)
    y_test_df = read_csv_required(INPUT_Y_TEST)
    train_meta = read_csv_required(INPUT_TRAIN_META)
    test_meta = read_csv_required(INPUT_TEST_META)
    feature_groups = load_json_required(INPUT_FEATURE_GROUPS)

    y_train = get_y(y_train_df)
    y_test = get_y(y_test_df)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"Train positives: {int((y_train == 1).sum())}, negatives: {int((y_train == 0).sum())}")
    print(f"Test positives:  {int((y_test == 1).sum())}, negatives: {int((y_test == 0).sum())}")
    print("-" * 100)

    print("Preparing SACU agent feature groups...")
    agent_features = build_feature_groups_for_agents(feature_groups, X_train)

    for agent, cols in agent_features.items():
        print(f"{agent}: {len(cols)} features")

    print("-" * 100)

    print("Training role-conditioned shallow SACU agents...")
    agent_models, train_agent_scores, test_agent_scores, train_times = train_agent_models(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        agent_features=agent_features,
    )

    print("Computing agent reliability and adaptive weights...")
    reliability = agent_reliability_from_train(y_train, train_agent_scores)

    adaptive_weights = compute_adaptive_weights(
        test_scores=test_agent_scores,
        train_reliability=reliability,
        meta_test=test_meta,
    )

    print("Computing organizational fusion outputs...")
    equal_fusion_scores = fuse_scores_equal(test_agent_scores)
    adaptive_fusion_scores = fuse_scores_adaptive(test_agent_scores, adaptive_weights)
    learned_fusion_scores, learned_fusion_model = train_meta_fusion(
        train_scores=train_agent_scores,
        y_train=y_train,
        test_scores=test_agent_scores,
    )

    print("Evaluating agents and fusion variants...")
    agent_perf = build_agent_performance(
        y_test=y_test,
        test_scores=test_agent_scores,
        agent_features=agent_features,
        train_times=train_times,
    )

    fusion_perf = build_fusion_performance(
        y_test=y_test,
        equal_scores=equal_fusion_scores,
        adaptive_scores=adaptive_fusion_scores,
        learned_scores=learned_fusion_scores,
    )

    ablation_df = build_ablation_scores(
        y_test=y_test,
        test_scores=test_agent_scores,
        adaptive_weights=adaptive_weights,
        equal_scores=equal_fusion_scores,
        adaptive_scores=adaptive_fusion_scores,
        learned_scores=learned_fusion_scores,
    )

    score_map = {}
    for agent in test_agent_scores.columns:
        score_map[agent] = test_agent_scores[agent].values

    score_map["FixedEqualWeightFusion"] = equal_fusion_scores
    score_map["AdaptiveSACUWeightFusion"] = adaptive_fusion_scores
    score_map["LearnedShallowMetaFusion"] = learned_fusion_scores

    confusion_df = build_confusion_matrices(y_test, score_map)
    roc_df, pr_df = build_curves(y_test, score_map)
    calibration_df = build_calibration(y_test, score_map)
    statistical_df = build_statistical_comparison(y_test, score_map)

    predictions_df = build_prediction_table(
        y_test=y_test,
        test_meta=test_meta,
        test_scores=test_agent_scores,
        equal_scores=equal_fusion_scores,
        adaptive_scores=adaptive_fusion_scores,
        learned_scores=learned_fusion_scores,
    )

    weights_df = build_weight_table(test_meta, adaptive_weights)

    print("Saving outputs...")
    agent_perf.to_csv(OUTPUT_AGENT_PERFORMANCE, index=False, encoding="utf-8-sig")
    predictions_df.to_csv(OUTPUT_AGENT_PREDICTIONS, index=False, encoding="utf-8-sig")
    weights_df.to_csv(OUTPUT_ADAPTIVE_WEIGHTS, index=False, encoding="utf-8-sig")
    fusion_perf.to_csv(OUTPUT_FUSION_PERFORMANCE, index=False, encoding="utf-8-sig")
    ablation_df.to_csv(OUTPUT_ABLATION, index=False, encoding="utf-8-sig")
    confusion_df.to_csv(OUTPUT_CONFUSION, index=False, encoding="utf-8-sig")
    roc_df.to_csv(OUTPUT_ROC, index=False, encoding="utf-8-sig")
    pr_df.to_csv(OUTPUT_PR, index=False, encoding="utf-8-sig")
    calibration_df.to_csv(OUTPUT_CALIBRATION, index=False, encoding="utf-8-sig")
    statistical_df.to_csv(OUTPUT_STATISTICAL, index=False, encoding="utf-8-sig")

    joblib.dump({
        "agent_models": agent_models,
        "learned_fusion_model": learned_fusion_model,
        "agent_features": agent_features,
        "agent_reliability": reliability,
        "agent_role_description": AGENT_ROLE_DESCRIPTION,
        "method_note": "Same-exam temporal-spatial SACU branch; no real longitudinal disease-course modeling.",
    }, OUTPUT_MODEL)

    save_json_summary(
        agent_perf=agent_perf,
        fusion_perf=fusion_perf,
        ablation_df=ablation_df,
        agent_features=agent_features,
    )

    write_report(
        agent_perf=agent_perf,
        fusion_perf=fusion_perf,
        ablation_df=ablation_df,
        confusion_df=confusion_df,
        statistical_df=statistical_df,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        agent_features=agent_features,
    )

    print()
    print("STAGE 2D COMPLETED SUCCESSFULLY")
    print("-" * 100)
    print(f"Agent performance:       {OUTPUT_AGENT_PERFORMANCE}")
    print(f"Agent predictions:       {OUTPUT_AGENT_PREDICTIONS}")
    print(f"Adaptive weights:        {OUTPUT_ADAPTIVE_WEIGHTS}")
    print(f"Fusion performance:      {OUTPUT_FUSION_PERFORMANCE}")
    print(f"Ablation study:          {OUTPUT_ABLATION}")
    print(f"Confusion matrices:      {OUTPUT_CONFUSION}")
    print(f"ROC curves:              {OUTPUT_ROC}")
    print(f"PR curves:               {OUTPUT_PR}")
    print(f"Calibration curves:      {OUTPUT_CALIBRATION}")
    print(f"Statistical comparison:  {OUTPUT_STATISTICAL}")
    print(f"Model:                   {OUTPUT_MODEL}")
    print(f"JSON summary:            {OUTPUT_SUMMARY_JSON}")
    print(f"Text report:             {OUTPUT_REPORT}")
    print("=" * 100)


if __name__ == "__main__":
    main()