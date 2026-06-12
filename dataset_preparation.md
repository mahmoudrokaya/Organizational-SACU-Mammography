# Dataset Preparation Guide

## Organizational-SACU-Mammography

This document describes the dataset preparation process required to reproduce the experiments presented in the Organizational-SACU-Mammography framework.

The repository does **not** redistribute any mammography datasets. Users must obtain access to the original datasets directly from their official providers and prepare them according to the procedures described below.

---

# Supported Datasets

The framework was designed to operate on screening mammography examinations that contain:

* Cranio-Caudal (CC) views
* Mediolateral Oblique (MLO) views
* Breast laterality information
* Examination dates
* Patient identifiers
* Diagnostic labels

The implementation supports datasets that provide sufficient metadata to reconstruct multi-view, bilateral, and temporal mammography relationships.

---

# Required Dataset Characteristics

Each mammography examination should ideally contain:

| Requirement        | Description                              |
| ------------------ | ---------------------------------------- |
| Patient Identifier | Unique patient ID                        |
| Study Identifier   | Unique examination ID                    |
| Examination Date   | Required for temporal analysis           |
| Laterality         | Left or Right breast                     |
| View Position      | CC or MLO                                |
| Mammography Image  | DICOM image                              |
| Diagnostic Label   | Benign, malignant, or equivalent mapping |

The temporal pathway requires multiple examinations for the same patient acquired at different time points.

Patients without prior examinations can still be included because the framework uses pathway-aware masking when temporal information is unavailable.

---

# Directory Structure

Create the following directory structure:

```text
data/
│
├── raw/
│   ├── dataset_1/
│   ├── dataset_2/
│   └── ...
│
├── interim/
│
├── processed/
│
└── manifests/
```

Raw downloaded datasets should be stored under:

```text
data/raw/
```

No manual modifications should be performed on the original DICOM files.

---

# Step 1: Dataset Verification

Run:

```bash
python scripts/00_verify_datasets.py
```

This stage performs:

* File integrity checks
* Missing file detection
* Label validation
* Metadata consistency verification

Expected outputs:

```text
results/reports/dataset_verification/
```

---

# Step 2: DICOM Preprocessing

Run:

```bash
python scripts/01_preprocess_dicom.py
```

This stage performs:

* DICOM loading
* Photometric correction
* Orientation normalization
* Intensity normalization
* Image quality validation

Generated outputs are stored in:

```text
data/interim/
```

---

# Step 3: Metadata Harmonization

Run:

```bash
python scripts/02_harmonize_metadata.py
```

This stage standardizes:

* Patient identifiers
* Study identifiers
* Examination dates
* View labels
* Laterality labels
* Diagnostic labels

Output:

```text
data/interim/harmonized_metadata.csv
```

---

# Step 4: Multi-View Pair Construction

Run:

```bash
python scripts/03_build_multiview_pairs.py
```

This stage associates:

* Left CC
* Left MLO
* Right CC
* Right MLO

into complete mammography examinations.

Output:

```text
data/interim/multiview_pairs.csv
```

---

# Step 5: Examination Integrity Audit

Run:

```bash
python scripts/04_audit_exam_integrity.py
```

This stage identifies:

* Missing views
* Duplicate examinations
* Incomplete studies
* Metadata inconsistencies

Output:

```text
data/interim/exam_audit_report.csv
```

---

# Step 6: Temporal Cohort Construction

Run:

```bash
python scripts/05_build_temporal_cohorts.py
```

This stage:

* Sorts examinations chronologically
* Links prior studies
* Builds temporal sequences

Output:

```text
data/interim/temporal_cohorts.csv
```

---

# Step 7: Modeling Manifest Generation

Run:

```bash
python scripts/06_build_modeling_manifests.py
```

This stage creates:

* Training manifests
* Validation manifests
* Testing manifests

The framework uses patient-level partitioning to avoid information leakage.

Output:

```text
data/manifests/
```

---

# Label Preparation

The framework expects a binary classification target.

Example mapping:

| Original Label | Binary Target |
| -------------- | ------------- |
| Benign         | 0             |
| Malignant      | 1             |

If the dataset uses alternative label definitions, modify:

```text
src/organizational_sacu_mammography/metadata/label_mapping.py
```

accordingly.

---

# Temporal Information

The Temporal-Spatial pathway uses historical examinations when available.

Patients may belong to one of two categories:

### Temporal Cases

At least one previous examination exists.

Example:

```text
Patient A
 ├── 2018 Exam
 ├── 2020 Exam
 └── 2022 Exam
```

### Non-Temporal Cases

Only a single examination exists.

Example:

```text
Patient B
 └── 2022 Exam
```

The framework automatically masks unavailable temporal information.

No synthetic temporal data are generated.

---

# Missing Views

Some datasets may contain incomplete examinations.

Example:

```text
Left CC
Left MLO
Right CC
```

with missing:

```text
Right MLO
```

Such cases are handled during the integrity audit stage.

The framework records missing information and applies pathway-aware masking where appropriate.

---

# Recommended Data Splitting Strategy

Patient-level partitioning is strongly recommended.

Example:

```text
Training     70%
Validation   15%
Testing      15%
```

Important:

Patients must not appear in more than one partition.

This prevents data leakage across training and evaluation cohorts.

---

# Expected Pipeline Outputs

After successful preparation, the following artifacts should be available:

```text
data/manifests/train_manifest.csv
data/manifests/validation_manifest.csv
data/manifests/test_manifest.csv
```

These manifests serve as the starting point for:

```bash
python scripts/07_extract_organizational_features.py
```

and the subsequent organizational modeling stages.

---

# Reproducibility Notes

For reproducibility:

* Preserve original DICOM files unchanged.
* Maintain patient-level partitions.
* Use consistent random seeds.
* Record software versions.
* Store generated manifests.
* Document any label mapping modifications.

The preparation workflow was designed to ensure that all downstream organizational analyses operate on standardized and reproducible mammography representations.
