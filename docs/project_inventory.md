# Project Inventory & Architecture Strategy

## 1. Files & Workspace Directory Inventory
- `data/raw/Financial Distress.csv`: Verified Kaggle Financial Distress prediction dataset (3,672 rows x 86 columns).
- `scripts/inspect_data.py`: Executable script for statistical inspection of dataset integrity, schema, distributions, missingness, and feature correlations.
- `docs/project_inventory.md`: Current project inventory and architecture blueprint.
- `docs/dataset.md`: Comprehensive documentation of the financial distress dataset structure and feature categorizations.

## 2. Current Project State
- Workspace initialized on macOS.
- Financial dataset unzipped to `data/raw/Financial Distress.csv` and archive zip cleaned up per user instructions.
- Verification and empirical inspection complete (Phase 0 and Phase 1).

## 3. Proposed End-to-End System Architecture

```
Financial Dataset (Kaggle Financial Distress)
             │
             ▼
    Data Preprocessing Pipeline
   (Handling lag features, scaling)
             │
             ▼
   XGBoost Financial Risk Model
             │
             ▼
  Financial Risk Score (0-100)
             │
             ├────────────────────────────────┐
             │                                │
             ▼                                ▼
   FinBERT Sentiment Analysis         News Data Stream
   (Positive, Negative, Neutral)       (Company headlines)
             │                                │
             ▼                                │
   Daily Sentiment Volatility                 │
             │                                │
             ▼                                │
  90-Day Sequence Generator ◄─────────────────┘
             │
             ▼
    PyTorch LSTM Model
             │
             ▼
     News Risk Score
             │
             ▼
 Risk Fusion Engine (Baseline & Adaptive)
             │
             ▼
    Final Multimodal Risk Score
             │
             ▼
  FastAPI Backend ──► Next.js Frontend Dashboard
```

## 4. Key Implementation Safeguards & Distinction
- **Financial Risk vs. Supplier Disruption**: The financial distress model predicts financial distress based on financial ratios. It does **not** assume financial distress equals supplier disruption.
- **Data Leakage Prevention**: `x81` is verified to be `Financial Distress_{t-1}` (lagged target). The preprocessing engine treats lag features strictly avoiding future lookahead.
