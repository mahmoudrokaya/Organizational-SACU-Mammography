# Data Availability

## Organizational-SACU-Mammography

This document describes the data availability policy for the Organizational-SACU-Mammography repository.

The repository provides the complete implementation of the Organizational SACU framework, including data preparation scripts, modeling pipelines, evaluation procedures, and reproducibility resources. However, the repository does **not** redistribute any mammography images or patient-level medical records.

---

# Data Availability Statement

The datasets used in this research are subject to their respective licensing agreements, usage restrictions, and ethical requirements.

Consequently:

* Raw mammography images are not distributed through this repository.
* Patient-level metadata are not distributed through this repository.
* Protected health information (PHI) is not distributed through this repository.
* Derived data that could potentially identify individual patients are not distributed through this repository.

The repository only contains:

* Source code
* Configuration files
* Documentation
* Manifest templates
* Reproducibility resources
* Publication-related assets

---

# Dataset Access

Researchers wishing to reproduce the experiments must independently obtain access to the original datasets from their official providers.

Users are responsible for:

* Complying with all licensing agreements.
* Complying with all institutional review requirements.
* Complying with all applicable privacy regulations.
* Complying with all dataset-specific usage policies.

---

# Supported Data Structure

After obtaining the datasets, users should prepare them according to:

```text
docs/dataset_preparation.md
```

The repository expects a standardized mammography organization consisting of:

* Patient identifiers
* Examination identifiers
* Examination dates
* Laterality information
* View-position information
* Diagnostic labels
* Mammography DICOM images

The exact internal dataset structure may vary depending on the source dataset.

---

# Included Data Resources

The repository includes several resources that support reproducibility without redistributing the original data.

## Manifest Templates

Located in:

```text
data/manifests/
```

These templates illustrate the expected schema used by the pipeline.

---

## Configuration Files

Located in:

```text
configs/
```

These files describe the processing and modeling settings used by the framework.

---

## Example Metadata Schemas

The repository may include example metadata structures and field definitions that allow users to adapt their datasets to the expected format.

No real patient information is included.

---

# Excluded Data Resources

The following resources are intentionally excluded from the repository.

## Raw Mammography Images

Examples:

```text
*.dcm
*.dicom
```

These files remain under the control of the original data providers.

---

## Patient Records

Examples:

```text
Patient identifiers
Clinical notes
Protected health information
```

These data are not distributed.

---

## Intermediate Dataset Copies

Examples:

```text
Preprocessed image archives
Derived image collections
Merged patient databases
```

These resources are not included.

---

## Trained Models Derived from Restricted Data

Some researchers may choose to generate trained models using their locally acquired datasets.

Because licensing requirements may vary across datasets, trained model files are not distributed as part of the primary repository release.

---

# Reproducing the Experiments

To reproduce the experiments:

1. Obtain the datasets directly from their official providers.
2. Prepare the datasets according to:

```text
docs/dataset_preparation.md
```

3. Verify the dataset structure:

```bash
python scripts/00_verify_datasets.py
```

4. Execute the remaining stages of the pipeline.

---

# Data Security and Privacy

The Organizational-SACU-Mammography repository was designed with medical-data privacy considerations in mind.

The repository therefore:

* Does not contain patient-identifiable information.
* Does not contain clinical records.
* Does not contain protected health information.
* Does not contain raw mammography datasets.

Users must ensure that their local handling of medical datasets complies with:

* Institutional policies
* National regulations
* Dataset licensing agreements
* Ethical review requirements

---

# Sharing Derived Results

Researchers using this repository are encouraged to share:

* Source code improvements
* Configuration files
* Evaluation scripts
* Statistical analyses
* Aggregate performance results

Researchers should avoid publicly sharing:

* Raw medical images
* Patient-level metadata
* Protected health information
* Data products prohibited by dataset licenses

---

# Reproducibility Resources

The repository includes multiple resources to facilitate independent verification:

```text
docs/overview.md
docs/pipeline_description.md
docs/dataset_preparation.md
docs/reproducibility.md
docs/expected_outputs.md
```

Together, these resources provide a complete description of the experimental workflow while respecting dataset licensing and patient privacy requirements.

---

# Contact and Questions

Questions regarding dataset access should be directed to the original dataset providers.

Questions regarding the repository implementation, reproducibility procedures, or organizational SACU framework may be submitted through the repository issue tracker.

---

# Summary

The Organizational-SACU-Mammography repository provides a fully reproducible implementation of the organizational shallow-agent framework while respecting medical-data privacy, licensing restrictions, and ethical considerations. The repository contains all code, documentation, and reproducibility resources necessary to recreate the experimental workflow once users obtain the required datasets through their official distribution channels.
