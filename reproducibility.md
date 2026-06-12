# Reproducibility Guide

## Organizational-SACU-Mammography

This document provides detailed instructions for reproducing the experiments implemented in the Organizational-SACU-Mammography repository.

The primary objective of this repository is to enable transparent and reproducible evaluation of an organizational shallow-agent architecture for mammography-based breast cancer detection. The repository therefore includes the complete experimental pipeline, configuration files, statistical evaluation procedures, and supporting analyses used throughout the study.

---

# Reproducibility Principles

The project follows several reproducibility principles:

1. **Patient-level separation**

   * No patient appears in more than one data partition.

2. **Deterministic pipeline execution**

   * Fixed random seeds are used whenever possible.

3. **Complete workflow availability**

   * All major experimental stages are included.

4. **Transparent evaluation**

   * Performance metrics are generated from saved predictions.

5. **Independent verification**

   * Statistical analyses can be rerun independently of model training.

---

# System Requirements

## Recommended Hardware

### Minimum Configuration

* 16 GB RAM
* Quad-core CPU
* 50 GB free storage

### Recommended Configuration

* 32 GB RAM or higher
* 8-core CPU or higher
* SSD storage
* 100 GB or more available space

The SACU framework was intentionally designed to remain computationally lightweight compared with extremely deep mammography architectures.

---

# Software Requirements

## Conda Installation

```bash
conda env create -f environment.yml
conda activate organizational-sacu-mammography
```

## Pip Installation

```bash
pip install -r requirements.txt
```

---

# Random Seeds

To maximize reproducibility, all experiments should use fixed random seeds.

Recommended seed:

```python
RANDOM_SEED = 42
```

Apply the same seed consistently to:

* NumPy
* Scikit-learn
* Python random module

Example:

```python
import random
import numpy as np

random.seed(42)
np.random.seed(42)
```

---

# Data Reproducibility

## Dataset Acquisition

The repository does not redistribute mammography datasets.

Users must independently obtain the datasets from their original providers.

Refer to:

```text
docs/dataset_preparation.md
```

for preparation instructions.

---

## Directory Structure

After preparation, the directory structure should resemble:

```text
data/
├── raw/
├── interim/
├── processed/
└── manifests/
```

---

# Full Pipeline Reproduction

The complete workflow may be executed sequentially.

---

## Stage 0

Dataset verification:

```bash
python scripts/00_verify_datasets.py
```

Expected outputs:

```text
Dataset audit reports
Metadata verification reports
Quality-control summaries
```

---

## Stage 1

Representation construction:

```bash
python scripts/01_preprocess_dicom.py
python scripts/02_harmonize_metadata.py
python scripts/03_build_multiview_pairs.py
python scripts/04_audit_exam_integrity.py
python scripts/05_build_temporal_cohorts.py
python scripts/06_build_modeling_manifests.py
```

Expected outputs:

```text
Patient manifests
Temporal cohorts
Multi-view examination structures
Harmonized metadata
```

---

## Stage 2

Organizational modeling:

```bash
python scripts/07_extract_organizational_features.py
python scripts/08_build_sacu_matrices.py
python scripts/09_train_baselines.py
python scripts/10_train_sacu_framework.py
```

Expected outputs:

```text
Feature matrices
Baseline predictions
SACU predictions
Performance summaries
```

---

# Reproducing Individual Experiments

The repository supports reproducing individual analyses independently.

---

## Clinical Operating-Point Analysis

```bash
python scripts/12_clinical_threshold_analysis.py
```

Generates:

* Threshold comparisons
* ROC operating points
* Precision–Recall operating points
* Clinical confusion matrices

---

## Organizational Ablation Analysis

```bash
python scripts/13_role_ablation_analysis.py
```

Generates:

* Pathway-removal experiments
* Contribution-loss analyses
* Component-importance summaries

---

## Statistical Validation

```bash
python scripts/14_bootstrap_model_comparison.py
```

Generates:

* Bootstrap confidence intervals
* Paired model comparisons
* Operating-point confidence intervals

---

## Organizational Interpretability Analysis

```bash
python scripts/15_pathway_interpretability_analysis.py
```

Generates:

* Pathway contribution summaries
* Dominant-pathway analyses
* Adaptive-weight distributions
* Pathway-correctness associations

---

## Coordination Mechanism Validation

```bash
python scripts/16_coordination_mechanism_ablation.py
```

Generates:

* Coordination-ablation experiments
* Fixed versus adaptive coordination comparisons
* Resource-limited pathway analyses
* Adaptive-weight entropy statistics

---

# Expected Outputs

The pipeline generates outputs under:

```text
results/
├── tables/
├── figures/
└── reports/
```

Typical outputs include:

* Performance tables
* ROC curves
* Precision–Recall curves
* Ablation summaries
* Statistical confidence intervals
* Interpretability analyses
* Coordination-validation reports

---

# Statistical Reproducibility

The repository uses bootstrap-based uncertainty estimation.

Recommended configuration:

| Parameter             | Value |
| --------------------- | ----- |
| Bootstrap repetitions | 2000  |
| Confidence level      | 95%   |
| Random seed           | 42    |

These settings were selected to provide stable confidence interval estimates while maintaining reasonable computational cost.

---

# Reproducibility Verification Checklist

Before comparing results, verify:

* Dataset preparation completed successfully.
* Patient-level partitions are preserved.
* Random seeds are fixed.
* Configuration files are unchanged.
* Software dependencies match the repository environment.
* Statistical analyses use the recommended bootstrap settings.

---

# Sources of Variation

Small differences may arise because of:

* Operating system differences
* Numerical library versions
* Parallel processing behavior
* Floating-point arithmetic differences

Such variations should typically have negligible impact on the reported conclusions.

---

# Reproducing Manuscript Figures and Tables

The repository includes scripts used to generate the principal figures and tables reported in the study.

After running the full pipeline, users can regenerate:

* Performance comparison tables
* Ablation-analysis tables
* Statistical confidence interval tables
* Pathway-contribution summaries
* Coordination-ablation summaries
* ROC and Precision–Recall figures

Generated outputs are stored under:

```text
results/tables/
results/figures/
```

---

# Reporting Reproducibility Issues

If unexpected discrepancies are observed:

1. Verify dataset preparation.
2. Confirm software versions.
3. Confirm random-seed settings.
4. Verify patient-level partitioning.
5. Re-run the affected stage independently.

Because all major stages are modular, individual analyses can be reproduced without rerunning the entire pipeline.

---

# Reproducible Research Statement

The Organizational-SACU-Mammography repository was developed to support transparent and reproducible research in mammography-based breast cancer detection. The complete workflow, including data preparation, organizational modeling, statistical validation, interpretability analysis, and adaptive-coordination evaluation, is provided to facilitate independent verification and future extension of the framework.
