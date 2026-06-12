# Tables Directory

This directory contains manuscript-ready and audit-ready tables generated during experimentation.

## Purpose

Tables provide structured summaries of:

* Dataset properties
* Feature characteristics
* Model performance
* Statistical analyses
* Ablation studies

## Typical Contents

```text
tables/
├── dataset_summary.csv
├── cohort_statistics.csv
├── feature_matrix_summary.csv
├── baseline_performance.csv
├── sacu_agent_performance.csv
├── fusion_performance.csv
├── threshold_analysis.csv
├── bootstrap_confidence_intervals.csv
├── pathway_ablation.csv
├── coordination_ablation.csv
└── interpretability_summary.csv
```

## Recommended Manuscript Tables

### Table 1

Dataset characteristics.

### Table 2

Feature-group composition.

### Table 3

Baseline classifier performance.

### Table 4

Individual SACU pathway performance.

### Table 5

Organizational fusion performance.

### Table 6

Threshold analysis.

### Table 7

Bootstrap confidence intervals.

### Table 8

Pathway ablation analysis.

### Table 9

Coordination ablation analysis.

### Table 10

Interpretability and pathway contribution analysis.

## Formatting

Tables should:

* Use consistent decimal precision.
* Include sample sizes where applicable.
* Clearly identify evaluation split.
* Include confidence intervals when available.

## Export Formats

Preferred formats:

* CSV
* XLSX

CSV should remain the primary archival format.
