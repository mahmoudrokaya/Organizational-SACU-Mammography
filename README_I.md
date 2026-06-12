# Interim Data

This directory contains intermediate representations generated during preprocessing and cohort construction.

## Purpose

Interim datasets support:

* Metadata harmonization
* Multi-view pairing
* Bilateral alignment
* Temporal sequence generation
* Quality-control auditing

## Typical Outputs

Examples include:

```text
interim/
├── harmonized_metadata.csv
├── patient_mapping.csv
├── multiview_pairs.csv
├── bilateral_alignment.csv
├── temporal_sequences.csv
└── pathway_masks.csv
```

## Reproducibility

All files inside this directory should be generated automatically by repository scripts.

Manual editing is discouraged.

## Quality Assurance

Before creating processed modeling matrices:

* Missing views should be audited
* Duplicate patients should be removed
* Temporal ordering should be verified
* Label consistency should be confirmed
