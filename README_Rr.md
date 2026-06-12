# Reports Directory

This directory contains experiment reports, audit summaries, and reproducibility documentation.

## Purpose

Reports provide detailed documentation of:

* Experimental configuration
* Training procedures
* Evaluation outcomes
* Statistical analyses
* Reproducibility information

## Typical Contents

```text
reports/
├── preprocessing_report.csv
├── metadata_audit.csv
├── representation_report.csv
├── feature_extraction_report.csv
├── baseline_training_report.csv
├── sacu_training_report.csv
├── threshold_analysis_report.csv
├── bootstrap_report.csv
├── interpretability_report.csv
├── coordination_ablation_report.csv
└── reproducibility_report.csv
```

## Recommended Reports

### Dataset Audit Report

Documents:

* Sample counts
* Missing values
* Label distributions
* View availability

### Feature Audit Report

Documents:

* Feature-group composition
* Missingness
* Variance analysis

### Modeling Report

Documents:

* Baseline results
* SACU pathway results
* Fusion performance

### Evaluation Report

Documents:

* ROC-AUC
* PR-AUC
* Sensitivity
* Specificity
* MCC
* Confidence intervals

### Organizational Analysis Report

Documents:

* Pathway influence
* Adaptive weighting behavior
* Coordination effects
* Ablation findings

## Reproducibility

Every report should record:

* Repository version
* Random seed
* Configuration files used
* Execution timestamp

## Publication Support

The reports directory serves as the primary source for:

* Manuscript tables
* Manuscript figures
* Supplementary material
* Reviewer-response documentation
