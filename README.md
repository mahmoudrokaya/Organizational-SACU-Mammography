# Organizational-SACU-Mammography

**Organizational Shallow-Agent Architecture for Multi-View, Bilateral, and Temporal Mammography Analysis**

This repository contains the complete implementation of an organizational shallow-agent framework for mammography-based breast cancer detection. The proposed architecture integrates multiple clinically relevant diagnostic perspectives, including:

* Multi-view mammographic analysis (CC and MLO projections)
* Bilateral anatomical symmetry assessment
* Temporal progression analysis across screening examinations
* Organizational coordination through Shallow Adaptive Cooperative Units (SACUs)
* Clinical operating-point optimization
* Statistical validation and uncertainty estimation
* Organizational interpretability analysis
* Coordination-mechanism ablation studies

The framework investigates whether structured organizational intelligence can provide a computationally efficient alternative to extremely deep neural architectures while preserving clinically meaningful diagnostic reasoning.

---

## Repository Overview

The project follows a structured experimental pipeline consisting of dataset preparation, organizational representation construction, cooperative modeling, evaluation, interpretability analysis, and adaptive-coordination validation.

### Stage 0 – Dataset Verification

* Dataset integrity validation
* Metadata auditing
* VinDr-Mammo consistency verification

### Stage 1 – Mammography Representation Construction

* DICOM preprocessing
* Metadata harmonization
* Multi-view pairing
* Bilateral organization
* Temporal cohort construction
* Modeling manifest generation

### Stage 2 – Organizational Modeling and Evaluation

* Organizational feature extraction
* SACU modeling matrix construction
* Baseline model development
* Organizational SACU framework training
* Clinical operating-point analysis
* Component ablation analysis
* Statistical confidence interval estimation
* Interpretability analysis
* Coordination-mechanism validation

---

## Organizational Framework

The proposed framework decomposes mammography reasoning into specialized cooperative pathways.

| Pathway               | Clinical Role                   |
| --------------------- | ------------------------------- |
| Local-Regional        | Local tissue-pattern analysis   |
| Multi-View            | CC/MLO consistency analysis     |
| Bilateral             | Anatomical asymmetry assessment |
| Temporal-Spatial      | Longitudinal change analysis    |
| Metadata              | Clinical context integration    |
| Adaptive Coordination | Organizational decision fusion  |

Rather than relying on a single monolithic model, diagnostic decisions emerge through cooperation among specialized pathways.

---

## Repository Structure

```text
configs/                 Configuration files
data/                    Dataset templates and manifests
docs/                    Documentation
notebooks/               Exploratory notebooks
paper_assets/            Figures and tables used in the manuscript
reproducibility/         Reproducibility instructions
scripts/                 End-to-end experimental pipeline
src/                     Core implementation
tests/                   Validation tests
```

---

## Installation

### Conda

```bash
conda env create -f environment.yml
conda activate organizational-sacu-mammography
```

### Pip

```bash
pip install -r requirements.txt
```

---

## Data Availability

This repository does not redistribute any mammography datasets.

Users must obtain the datasets directly from their original providers and prepare them according to the instructions provided in:

```text
docs/dataset_preparation.md
```

---

## Running the Pipeline

Example execution order:

```bash
python scripts/00_verify_datasets.py
python scripts/01_preprocess_dicom.py
python scripts/02_harmonize_metadata.py
python scripts/03_build_multiview_pairs.py
python scripts/04_audit_exam_integrity.py
python scripts/05_build_temporal_cohorts.py
python scripts/06_build_modeling_manifests.py
python scripts/07_extract_organizational_features.py
python scripts/08_build_sacu_matrices.py
python scripts/09_train_baselines.py
python scripts/10_train_sacu_framework.py
```

Additional scripts are available for threshold optimization, ablation studies, statistical validation, interpretability analysis, and adaptive-coordination evaluation.

---

## Reproducibility

The repository includes:

* Configuration files
* Experimental scripts
* Statistical evaluation procedures
* Bootstrap confidence interval estimation
* Organizational interpretability analysis
* Coordination-mechanism ablation experiments

Detailed instructions are provided in:

```text
reproducibility/
```

---

## Research Objectives

The primary objectives of this repository are:

1. To investigate organizational intelligence for mammography analysis.
2. To integrate multi-view, bilateral, and temporal diagnostic evidence within a unified framework.
3. To evaluate whether cooperative shallow-agent architectures can provide competitive diagnostic performance with substantially lower computational complexity than extremely deep neural networks.
4. To improve transparency through pathway-level organizational interpretability.

---

## Citation

If you use this repository in academic work, please cite:

```text
Mahmoud Rokaya.
Organizational-SACU-Mammography:
Organizational Shallow-Agent Architecture for Multi-View,
Bilateral, and Temporal Mammography Analysis.
GitHub Repository.
```

Citation metadata are available in:

```text
CITATION.cff
```

---

## License

This project is released under the MIT License.

See:

```text
LICENSE
```
