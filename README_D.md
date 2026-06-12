# Data Directory

This directory contains all datasets, manifests, and intermediate representations used by the Organizational SACU Mammography framework.

## Directory Structure

```text
data/
├── raw/
├── interim/
├── processed/
└── manifests/
```

## Data Flow

```text
Raw Mammography Data
        ↓
Metadata Harmonization
        ↓
Multi-view Pairing
        ↓
Bilateral Alignment
        ↓
Temporal Sequence Construction
        ↓
Feature Extraction
        ↓
Processed Modeling Matrices
```

## Dataset Requirements

The repository is designed for full-field digital mammography datasets containing:

* Patient identifiers
* Study identifiers
* Image identifiers
* View information (CC/MLO)
* Laterality information (Left/Right)
* Acquisition dates
* Diagnostic labels

Supported cohorts may include:

* VinDr-Mammo
* Private institutional cohorts
* Multi-center screening datasets

## Expected Views

Each examination may contain:

| View | Description                 |
| ---- | --------------------------- |
| LCC  | Left Cranio-Caudal          |
| LMLO | Left Medio-Lateral Oblique  |
| RCC  | Right Cranio-Caudal         |
| RMLO | Right Medio-Lateral Oblique |

## Labels

The SACU framework supports:

* Binary malignancy prediction
* Benign vs malignant classification
* Multi-class diagnostic categorization

Label mappings should be documented in:

```text
data/manifests/sample_label_mapping.csv
```

## Reproducibility

All processed datasets must be generated through the repository pipeline and should not be manually modified.

The raw data directory should remain immutable after ingestion.

## Privacy

Patient-identifiable information must be removed prior to repository usage.

No protected health information (PHI) should be stored within this repository.
