# Running the Full Organizational SACU Pipeline

This document describes how to reproduce the complete experimental workflow from raw mammography images to final manuscript-ready results.

---

# Overview

The Organizational SACU framework consists of the following stages:

```text
Raw Mammography Images
        ↓
Metadata Harmonization
        ↓
Image Preprocessing
        ↓
Multi-View Representation Construction
        ↓
Bilateral Representation Construction
        ↓
Temporal Representation Construction
        ↓
Feature Extraction
        ↓
Feature Matrix Assembly
        ↓
Baseline Model Training
        ↓
SACU Agent Training
        ↓
Coordination and Fusion
        ↓
Evaluation and Statistical Analysis
        ↓
Figures, Tables, and Reports
```

---

# Step 1: Environment Setup

Clone the repository:

```bash
git clone https://github.com/<repository>/Organizational-SACU-Mammography.git
cd Organizational-SACU-Mammography
```

Create the environment:

```bash
conda env create -f environment.yml
conda activate organizational-sacu-mammography
```

or

```bash
pip install -r requirements.txt
```

---

# Step 2: Prepare Raw Data

Place mammography datasets under:

```text
data/raw/
```

Expected structure:

```text
data/raw/
├── patient_0001/
├── patient_0002/
├── ...
```

---

# Step 3: Metadata Harmonization

Execute metadata preparation:

```bash
python scripts/run_metadata_harmonization.py
```

Outputs:

```text
data/interim/
```

---

# Step 4: Representation Construction

Execute:

```bash
python scripts/run_representation.py
```

Outputs:

```text
multiview_pairs.csv
bilateral_alignment.csv
temporal_sequences.csv
pathway_masks.csv
```

---

# Step 5: Feature Extraction

Execute:

```bash
python scripts/run_feature_extraction.py
```

Outputs:

```text
outputs/features/
```

---

# Step 6: Modeling Matrix Construction

Execute:

```bash
python scripts/run_feature_matrix_builder.py
```

Outputs:

```text
data/processed/
```

---

# Step 7: Baseline Models

Execute:

```bash
python scripts/run_baselines.py
```

Outputs:

```text
results/tables/baseline_performance.csv
```

---

# Step 8: SACU Training

Execute:

```bash
python scripts/run_sacu_training.py
```

Outputs:

```text
results/tables/sacu_agent_performance.csv
results/tables/fusion_performance.csv
```

---

# Step 9: Evaluation

Execute:

```bash
python scripts/run_evaluation.py
```

Outputs:

```text
threshold analysis
bootstrap confidence intervals
ablation studies
interpretability analyses
coordination analyses
```

---

# Step 10: Manuscript Assets

Execute:

```bash
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

Outputs:

```text
results/tables/
results/figures/
results/reports/
```

---

# Expected Runtime

Approximate runtime:

| Stage                       | Time          |
| --------------------------- | ------------- |
| Metadata harmonization      | Minutes       |
| Representation construction | Minutes       |
| Feature extraction          | Minutes–Hours |
| Baselines                   | Minutes       |
| SACU training               | Minutes       |
| Evaluation                  | Minutes       |

Runtime depends on dataset size and hardware.
