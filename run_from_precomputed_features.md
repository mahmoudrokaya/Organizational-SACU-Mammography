# Running from Precomputed Features

This document describes how to reproduce all modeling results without reprocessing mammography images.

---

# Purpose

Many users may only have access to:

```text
feature_matrix.csv
```

or

```text
train_matrix.csv
validation_matrix.csv
test_matrix.csv
```

In this scenario, image preprocessing and feature extraction can be skipped.

---

# Required Inputs

Place feature matrices in:

```text
data/processed/
```

Example:

```text
data/processed/
├── train_matrix.csv
├── validation_matrix.csv
├── test_matrix.csv
```

---

# Run Baselines

```bash
python scripts/run_baselines.py
```

---

# Run SACU Training

```bash
python scripts/run_sacu_training.py
```

---

# Run Evaluation

```bash
python scripts/run_evaluation.py
```

---

# Generated Outputs

```text
results/tables/
results/figures/
results/reports/
```

---

# Recommended Use Cases

Use precomputed features when:

* Reproducing manuscript results.
* Comparing alternative classifiers.
* Running reviewer-requested analyses.
* Performing statistical validation.

---

# Not Reproduced

The following stages are skipped:

* DICOM loading.
* Orientation correction.
* Breast cropping.
* Intensity normalization.
* Multi-view construction.
* Bilateral construction.
* Temporal sequence construction.

These stages are assumed to have already been executed.
