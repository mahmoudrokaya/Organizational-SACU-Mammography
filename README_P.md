# Processed Data

This directory contains modeling-ready datasets used by baseline models and Organizational SACU agents.

## Characteristics

Processed datasets should:

* Contain harmonized feature representations
* Exclude protected health information
* Support reproducible training
* Be suitable for direct model ingestion

## Typical Files

```text
processed/
├── feature_matrix.csv
├── train_matrix.csv
├── validation_matrix.csv
├── test_matrix.csv
├── pathway_feature_matrix.csv
└── metadata_features.csv
```

## Feature Categories

The SACU framework organizes features into:

### Local-Regional Features

Breast-region descriptors extracted from individual views.

### Multi-View Features

Relationships between CC and MLO views.

### Bilateral Features

Left-right symmetry and asymmetry measures.

### Temporal Features

Longitudinal progression indicators.

### Metadata Features

Clinical and acquisition metadata.

### Adaptive-Control Features

Pathway availability and reliability indicators.

## Modeling

Processed matrices should be considered final analytical inputs for:

* Baseline classifiers
* SACU pathway agents
* Coordination modules
* Organizational fusion models
