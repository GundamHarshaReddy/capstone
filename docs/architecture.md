# System Architecture Specification

## Overview
**Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News** is organized as a decoupled, multi-tier research architecture separating data engineering, ML modeling, REST API services, and the interactive frontend.

```
+-----------------------------------------------------------------------------------------+
|                                    NEXT.JS FRONTEND                                     |
|           (Dashboard | Company Directory | Risk Sandbox | Performance & Ablation)       |
+-----------------------------------------------------------------------------------------+
                                             │
                                   HTTP / REST API (Port 8000)
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                     FASTAPI BACKEND                                     |
|             (/api/statistics, /api/companies, /api/predict/final, /api/model-metrics)   |
+-----------------------------------------------------------------------------------------+
                                             │
               ┌─────────────────────────────┼─────────────────────────────┐
               │                             │                             │
               ▼                             ▼                             ▼
+-----------------------------+ +-----------------------------+ +-----------------------------+
|    Financial Risk Service   | |   FinBERT Sentiment Service | |   PyTorch 90-Day LSTM       |
| (XGBoost 82/83 Features)    | | (ProsusAI/finbert Softmax)  | | (Sequences: [batch, 90, 6]) |
+-----------------------------+ +-----------------------------+ +-----------------------------+
               │                             │                             │
               └─────────────────────────────┼─────────────────────────────┘
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                                  RISK FUSION ENGINE                                     |
|    - Baseline Fixed Fusion: 0.60 * Financial Risk + 0.40 * News Risk                    |
|    - Data-Dependent Adaptive Heuristic: Dynamic weight re-allocation based on coverage  |
+-----------------------------------------------------------------------------------------+
                                             │
                                             ▼
+-----------------------------------------------------------------------------------------+
|                               EXPLAINABILITY & CITATIONS                                |
|    - Tree Gain Importance & SHAP Ranking                                                |
|    - Non-Causal Risk Factor Attributions                                               |
+-----------------------------------------------------------------------------------------+
```

## Subsystem Details
1. **Financial Risk Service (`backend/services/financial_risk.py`)**:
   - Loads trained XGBoost models (`models/xgboost_with_x81.pkl` and `models/xgboost_without_x81.pkl`).
   - Maps continuous probability to $[0, 100]$ standardized risk scale.
2. **FinBERT Sentiment Service (`backend/services/news_sentiment.py`)**:
   - Hugging Face `ProsusAI/finbert` tokenization and sequence classification.
   - Optimized with `torch.set_num_threads(1)` for high-throughput execution.
3. **Temporal News LSTM Service (`backend/services/news_lstm.py`)**:
   - 2-layer PyTorch LSTM evaluating $(N, 90, 6)$ sequences.
   - Evaluates forward risk probability on chronological corporate sequences.
4. **Risk Fusion Service (`backend/services/risk_fusion.py`)**:
   - Baseline 60/40 weighted average and coverage-adaptive dynamic weighting.
5. **Model Explainability Service (`backend/services/explainability.py`)**:
   - Strictly enforces non-causal language ("features contributing to model predictions").
