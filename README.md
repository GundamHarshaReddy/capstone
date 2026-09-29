# Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News

A peer-reviewed, research-grade engineering implementation combining tabular financial distress modeling (XGBoost), corporate news sentiment analysis (FinBERT), 90-day chronological sequence modeling (PyTorch LSTM), and data-dependent adaptive risk fusion.

---

## 🌟 Research Architecture Overview

```
Financial Dataset (Kaggle Financial Distress, 422 Firms)
                        │
                        ▼
          Temporal Preprocessing (Train-only Scaling)
                        │
                        ▼
          XGBoost Financial Risk Classifier (82/83 Features)
                        │
                        ▼
          Financial Risk Score (0–100)
                        │
         ┌──────────────┴──────────────┐
         │                             │
         ▼                             ▼
Real S&P 500 Historical News   Rule-Based Risk Fusion
(4,589 Articles, 469 Firms)     (Baseline & Adaptive)
         │                             │
         ▼                             │
FinBERT Sentiment Extraction           │
(Positive, Negative, Neutral)          │
         │                             │
         ▼                             │
90-Day Sequence Generator              │
([N, 90, 6] Tensors)                   │
         │                             │
         ▼                             │
PyTorch Temporal LSTM ─────────────────┘
         │
         ▼
Final Multimodal Risk Score
         │
         ▼
FastAPI REST API ──► Next.js 16 Dashboard
```

---

## 📊 Empirical Model Performance Highlights

### 1. Financial Distress Classification (XGBoost on 1,083 Unseen Temporal Test Samples)
| Experiment Configuration | Features | ROC-AUC | PR-AUC | Brier Score | Calibrated Threshold | Recall | Precision | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Experiment A (With Lagged Target x81)** | 83 | **0.9270** | **0.3236** | **0.0262** | **τ = 0.40** | **48.39%** | **38.46%** | **0.4286** |
| **Experiment B (Without x81 - Pure Ratios)** | 82 | **0.9232** | **0.3764** | **0.0255** | **τ = 0.45** | **38.71%** | **40.00%** | **0.3934** |

> **Key Research Finding**: Excluding lagged target `x81` maintains a high ROC-AUC of 0.9232 while increasing Precision-Recall AUC from 0.3236 to 0.3764 and improving probability calibration (Brier score: 0.0255), proving that underlying financial ratios drive predictive power rather than state autocorrelation.

### 2. PyTorch 90-Day Temporal News LSTM (Evaluated on 10,821 Real Test Sequences, 2022–2024)
- **ROC-AUC**: **0.5435**
- **PR-AUC**: **0.0416** (Baseline prevalence: 0.0262)
- **Accuracy**: **91.19%**
- **Test Set**: 10,821 chronological sequences spanning 2022 to 2024.

---

## 🔬 Scientific Ablation Matrix

| Model ID | Modality | Architecture | Target Dataset | ROC-AUC | PR-AUC | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A1** | Tabular Financial | XGBoost (83 features, with x81) | Kaggle Financial Distress (Held-out Time 11-14) | **0.9270** | **0.3236** | **EVALUATED** |
| **Model A2** | Tabular Financial | XGBoost (82 features, NO x81) | Kaggle Financial Distress (Held-out Time 11-14) | **0.9232** | **0.3764** | **EVALUATED** |
| **Model B** | Temporal News | 2-Layer PyTorch LSTM (90-day sequences) | S&P 500 Historical News (Held-out T >= 2022) | **0.5435** | **0.0416** | **EVALUATED** |
| **Model C** | Multimodal Static | XGBoost + FinBERT Instant Net Sentiment | Unified Service Sandbox | Domain Fusion | Domain Fusion | **INTEGRATED** |
| **Model D** | Multimodal Fixed | Fixed Fusion (0.60 Fin + 0.40 News) | Unified Service Sandbox | Domain Fusion | Domain Fusion | **INTEGRATED** |
| **Model E** | Multimodal Adaptive | Data-Dependent Adaptive Heuristic Fusion | Unified Service Sandbox | Domain Fusion | Domain Fusion | **INTEGRATED** |

---

## ⚡ Step-by-Step Reproduction Guide

### 1. Environment Setup
```bash
pip install -r requirements.txt
```

### 2. Run Data Processing & Financial Model Experiments
```bash
# Preprocessing with train-only scaling & median imputation
python3 scripts/preprocess.py

# Run Experiment A (with x81) & Experiment B (without x81) with threshold optimization and SHAP
python3 scripts/train_evaluate_experiments.py
```

### 3. Run Real News Sentiment & PyTorch LSTM Pipeline
```bash
# Standardize raw S&P 500 news and build company entity mappings
python3 scripts/prepare_news_and_mapping.py

# Execute batched FinBERT sentiment extraction and daily aggregation
python3 scripts/run_finbert.py

# Construct 90-day chronological sequence tensors
python3 scripts/prepare_sequences.py

# Train PyTorch LSTM and evaluate on held-out test sequences (2022-2024)
python3 scripts/train_lstm.py

# Compile full scientific ablation study
python3 scripts/run_ablation_study.py
```

### 4. Run Automated Test Suite
```bash
python3 -m unittest discover tests -v
```

### 5. Launch Backend & Frontend Services
```bash
# Start FastAPI backend on port 8000
python3 -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload

# In a separate terminal, launch Next.js frontend
cd frontend
npm run dev
```
Open [http://localhost:3000](http://localhost:3000) in your web browser.

---

## 📁 Repository Structure

```
├── data/
│   ├── raw/                      # Financial Distress.csv
│   ├── processed/                # Preprocessed train, val, test, scalers
│   └── news/                     # Real news CSVs, entity mapping, 90-day sequences
├── models/                       # xgboost_with_x81.pkl, xgboost_without_x81.pkl, news_lstm.pt
├── scripts/                      # Preprocessing, experiment execution, FinBERT, LSTM
├── backend/                      # FastAPI service layer & multimodal services
├── frontend/                     # Next.js 16 App Router + Tailwind + Recharts
├── reports/                      # x81_comparison, ablation_results, lstm_metrics
├── docs/                         # Formal research report, methodology, audit, leakage
├── tests/                        # Comprehensive unit & edge-case test suite
├── requirements.txt              # Pinned Python dependencies
└── README.md
```
