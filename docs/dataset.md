# Dataset Inspection Report

## Dataset Overview
- **Filename**: `data/raw/Financial Distress.csv`
- **Total Records (Rows)**: 3,672
- **Total Attributes (Columns)**: 86
- **Missing Values**: 0 (0.00%)
- **Duplicate Rows**: 0 (0.00%)

## Column Classifications & Categorization

| Column Category | Columns | Description |
| :--- | :--- | :--- |
| **Identifiers** | `Company` | Integer ID (1 to 422). Identifies 422 distinct companies. |
| **Temporal** | `Time` | Time period indicator (1 to 14). Average time periods per company: ~8.7. |
| **Target Variable** | `Financial Distress` | Continuous distress metric float. Kaggle benchmark threshold: `Financial Distress <= -0.5` represents financial distress (`1`), while `> -0.5` represents healthy (`0`). |
| **Financial Features** | `x1` to `x79`, `x81` | Continuous financial ratios and metrics. |
| **Categorical/Discrete** | `x80`, `x82`, `x83` | Discrete counts / categorical indicators (`x80`: 37 unique values, `x82`: 49 unique values, `x83`: 71 unique values). |
| **Lagged / Leaking Feature** | `x81` | Verified as `Financial Distress` at `Time t-1`. Highly correlated (`r = 0.443`). Must be handled carefully during modeling. |

## Target Class Distribution
- **Raw Metric Summary**: Min = -8.6317, 25% = 0.1723, Median = 0.5838, 75% = 1.3518, Max = 128.4000.
- **Binarized Target Distribution (`Financial Distress <= -0.5`)**:
  - Healthy (Class 0): 3,536 instances (96.30%)
  - Distressed (Class 1): 136 instances (3.70%)
- **Imbalance Ratio**: ~26:1. Class imbalance techniques (e.g. `scale_pos_weight` in XGBoost, PR-AUC, F1 metrics) are mandatory.
- **Affected Companies**: 136 out of 422 companies experience distress during the observed timeline.
