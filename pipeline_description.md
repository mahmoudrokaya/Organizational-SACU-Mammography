# Pipeline Description

## Organizational-SACU-Mammography

This document describes the complete experimental workflow implemented in the Organizational-SACU-Mammography repository. The pipeline is designed to transform raw mammography examinations into structured organizational representations that support cooperative diagnostic reasoning through Shallow Adaptive Cooperative Units (SACUs).

The workflow is organized into sequential stages that progressively transform raw imaging data into interpretable diagnostic decisions.

---

# Pipeline Overview

```text
Raw Mammography Data
        │
        ▼
Stage 0: Dataset Verification
        │
        ▼
Stage 1: Representation Construction
        │
        ▼
Stage 2A: Organizational Feature Extraction
        │
        ▼
Stage 2B: SACU Modeling Matrices
        │
        ▼
Stage 2C: Baseline Models
        │
        ▼
Stage 2D: Organizational SACU Framework
        │
        ▼
Stage 2E–2J: Evaluation and Analysis
```

---

# Stage 0 — Dataset Verification

## Purpose

Ensure that all datasets are internally consistent before any modeling begins.

## Components

### Stage0C Dataset Verification

Verifies:

* Dataset integrity
* File availability
* Label consistency
* Examination completeness

### Stage0D VinDr Audit

Performs:

* Metadata validation
* Examination auditing
* Dataset consistency checks
* Cohort verification

## Outputs

* Dataset audit reports
* Quality-control summaries
* Metadata verification tables

---

# Stage 1 — Mammography Representation Construction

This stage converts raw mammography examinations into structured organizational representations.

---

## Stage1A — DICOM Preprocessing

### Purpose

Standardize mammography images.

### Operations

* DICOM loading
* Photometric correction
* Orientation normalization
* Intensity normalization
* Quality verification

### Outputs

* Standardized mammography images

---

## Stage1A2 — Metadata Harmonization

### Purpose

Create a unified metadata representation.

### Operations

* Harmonize patient identifiers
* Standardize examination metadata
* Normalize labels
* Verify examination dates

### Outputs

* Harmonized metadata tables

---

## Stage1B — Multi-View Pairing

### Purpose

Construct clinically meaningful mammography examinations.

### Operations

* Match CC and MLO projections
* Associate views with breast laterality
* Create complete examination structures

### Outputs

* Multi-view examination pairs

---

## Stage1B1 — Examination Integrity Audit

### Purpose

Identify incomplete or duplicated examinations.

### Operations

* Detect missing views
* Detect duplicated studies
* Verify examination completeness

### Outputs

* Audit reports
* Clean examination manifests

---

## Stage1C — Temporal Cohort Construction

### Purpose

Build longitudinal screening trajectories.

### Operations

* Sort examinations chronologically
* Associate prior examinations
* Create temporal sequences

### Outputs

* Temporal screening cohorts

---

## Stage1D — Modeling Manifest Generation

### Purpose

Prepare model-ready cohort definitions.

### Operations

* Create patient-level partitions
* Generate training manifests
* Generate validation manifests
* Generate testing manifests

### Outputs

* Modeling manifests
* Patient-level split definitions

---

# Stage 2 — Organizational Modeling

This stage implements the SACU framework.

---

## Stage2A — Organizational Feature Extraction

### Purpose

Extract pathway-specific descriptors.

### Pathways

#### Local-Regional Pathway

Captures:

* Local tissue structure
* Regional density characteristics
* Local image statistics

#### Multi-View Pathway

Captures:

* CC/MLO consistency
* Projection agreement
* Cross-view relationships

#### Bilateral Pathway

Captures:

* Left-right symmetry
* Anatomical correspondence
* Bilateral differences

#### Temporal-Spatial Pathway

Captures:

* Longitudinal change
* Progression patterns
* Historical examination differences

#### Metadata Pathway

Captures:

* Patient information
* Examination context
* Structured metadata descriptors

### Outputs

* Organizational feature matrices

---

## Stage2B — SACU Modeling Matrices

### Purpose

Construct pathway-specific learning matrices.

### Operations

* Feature aggregation
* Pathway partitioning
* Matrix generation
* Training-ready formatting

### Outputs

* SACU modeling matrices

---

## Stage2C — Baseline Models

### Purpose

Establish conventional reference performance.

### Models

Typical baseline models include:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Extra Trees
* AdaBoost

### Outputs

* Baseline predictions
* Performance benchmarks

---

## Stage2D — Organizational SACU Framework

### Purpose

Train the cooperative organizational architecture.

### SACU Components

#### LocalRegionalAgent

Analyzes local tissue patterns.

#### MultiViewAgent

Analyzes multi-view consistency.

#### BilateralAgent

Analyzes anatomical asymmetry.

#### TemporalSpatialAgent

Analyzes longitudinal progression.

#### MetadataAgent

Analyzes contextual metadata.

#### AdaptiveControlAgent

Coordinates pathway interactions.

### Coordination Mechanism

The final prediction is generated through organizational cooperation among pathway-specific agents.

### Outputs

* SACU predictions
* Organizational performance metrics

---

# Evaluation and Analysis

The remaining stages investigate different aspects of model behavior.

---

## Stage2E — Clinical Operating-Point Analysis

### Purpose

Evaluate threshold-dependent behavior.

### Analyses

* Sensitivity-specificity trade-offs
* Clinical operating points
* Confusion matrices
* ROC analysis
* Precision–Recall analysis

---

## Stage2F — Organizational Ablation Analysis

### Purpose

Quantify pathway contributions.

### Analyses

* Agent removal experiments
* Contribution loss analysis
* Pathway importance assessment

---

## Stage2G — Statistical Validation

### Purpose

Estimate uncertainty and robustness.

### Analyses

* Bootstrap confidence intervals
* Pairwise model comparison
* Operating-point confidence intervals

---

## Stage2I — Organizational Interpretability Analysis

### Purpose

Understand pathway-level reasoning.

### Analyses

* Pathway contribution assessment
* Dominant-pathway identification
* Adaptive-weight analysis
* Pathway correctness association

---

## Stage2J — Adaptive Coordination Validation

### Purpose

Validate organizational adaptivity.

### Analyses

* Coordination-mechanism ablation
* Fixed versus adaptive fusion
* Resource-limited pathway evaluation
* Pathway-entropy analysis
* Case-level variability assessment

---

# Organizational Design Philosophy

The central hypothesis of this repository is that diagnostic reasoning can be modeled as a cooperative organizational process.

Instead of relying on a single highly complex model, the framework distributes responsibility among specialized pathways that analyze complementary aspects of mammographic evidence.

The organizational coordinator then integrates pathway-specific evidence into a unified diagnostic decision.

This design aims to provide:

* Competitive diagnostic performance
* Reduced computational complexity
* Improved interpretability
* Enhanced modularity
* Better alignment with clinical reasoning processes

---

# Reproducibility

The repository provides:

* Complete experimental scripts
* Configuration files
* Statistical evaluation procedures
* Supporting documentation
* Reproducible pipeline definitions

Users can reproduce individual stages independently or execute the entire pipeline from raw data preparation through final evaluation.

Refer to the `reproducibility/` directory for detailed execution instructions and environment setup procedures.
