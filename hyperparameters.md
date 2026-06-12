# Hyperparameters

This document records the principal hyperparameters used throughout the Organizational SACU framework.

---

# Baseline Models

## Logistic Regression

```yaml
solver: liblinear
max_iter: 1000
class_weight: balanced
```

## Random Forest

```yaml
n_estimators: 500
max_depth: None
min_samples_split: 2
class_weight: balanced
random_state: 42
```

## XGBoost

```yaml
n_estimators: 500
learning_rate: 0.05
max_depth: 6
subsample: 0.8
colsample_bytree: 0.8
random_state: 42
```

---

# SACU Agents

All SACU agents use shallow-learning architectures.

Examples include:

* Logistic Regression
* Random Forest
* Extra Trees
* Gradient Boosting

depending on configuration.

---

# Coordination Module

Default settings:

```yaml
adaptive_weighting: true
reliability_weighting: true
confidence_weighting: true
```

---

# Fusion Layer

```yaml
fusion_model: logistic_regression
fusion_probability_threshold: 0.50
```

---

# Threshold Analysis

```yaml
default_threshold: 0.50
high_sensitivity_target: 0.90
high_specificity_target: 0.90
```

---

# Bootstrap Analysis

```yaml
n_bootstrap: 2000
confidence_level: 0.95
random_seed: 42
```

---

# Reproducibility

```yaml
random_seed: 42
```

All manuscript experiments should use identical random seeds unless explicitly stated otherwise.
