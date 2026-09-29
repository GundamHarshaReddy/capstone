# Final Research Implementation & Scientific Roadmap

## 1. Current Architecture
The current system implements a multi-tier pipeline:
- **Tabular Tier**: XGBoost classifier trained on 83 financial indicators (`x1`–`x83`).
- **Textual Sentiment Tier**: Hugging Face FinBERT (`ProsusAI/finbert`) for probability extraction ($p_{\text{pos}}, p_{\text{neg}}, p_{\text{neu}}$).
- **Temporal Sequence Tier**: 2-layer PyTorch LSTM designed to process 90-day sentiment sequences $(N, 90, 5)$.
- **Fusion Tier**: Fixed-weight baseline ($0.60 \times \text{Fin} + 0.40 \times \text{News}$) and a heuristic adaptive weighting rule.
- **Application Tier**: FastAPI REST backend and Next.js 16 frontend with Recharts visualization.

## 2. Current Datasets
- **Financial Distress Dataset (`data/raw/Financial Distress.csv`)**: 3,672 observations across 422 companies and 14 time periods. Source: Kaggle (S. Ebrahimi, Tehran Stock Exchange).
- **News Dataset (`data/news/`)**: Currently contains only a schema template (`sample_news_schema.csv`). No real historical news articles are linked to the anonymized Iranian companies.

## 3. Current Targets
- **Financial Distress Target**: Binary indicator $Y \in \{0, 1\}$ defined as $\mathbb{I}(\text{Financial Distress} \le -0.5)$. Positive class rate: 3.70% (136 instances).
- **Supplier Disruption Target**: Absent. The current dataset contains financial distress, not physical supply chain disruption events.

## 4. Existing Valid Components
- XGBoost financial distress classification with class imbalance handling (`scale_pos_weight = 23.66`).
- Temporal train/test splitting (`Time <= 10` for train, `Time 11..14` for test).
- Evaluation pipeline computing ROC-AUC, PR-AUC, Confusion Matrix, Precision, Recall, F1.
- Full FastAPI backend, Next.js frontend, and unit test harness.

## 5. Invalid / Unvalidated Components Requiring Correction
1. **Global Imputation Before Split**: `preprocess.py` currently executes `df.fillna(df.median())` globally. Learned parameters must be fitted solely on training data.
2. **Lagged Target Feature (`x81`)**: Feature `x81` is $Y_{t-1}$. While temporally valid, it acts as an auto-regressive shortcut. A mandatory comparative experiment (with vs. without `x81`) is required.
3. **Arbitrary Classification Threshold**: Threshold $0.5$ on imbalanced data leads to high false negatives (FN=20 vs TP=11). A proper internal validation set (`Time 9–10`) must be used for optimal threshold selection.
4. **Unvalidated News & LSTM**: The LSTM and FinBERT pipelines lack real news tied to verifiable targets. If companies cannot be de-anonymized, the research scope must be rigorously defined as Financial Distress Risk, and any news integration must be grounded in verified public data or transparently scoped.
5. **Heuristic Fusion Labeling**: Adaptive fusion must be explicitly labeled as a rule-based domain heuristic, not a "learned" model.

## 6. Data Requirements
- For news modeling: Historical financial news with verifiable company identities, timestamps, and target alignment.
- For disruption modeling: Documented supply disruption ground truth. If unavailable, scope strictly to financial distress risk.

## 7. Experimental Plan
- **Experiment 1A**: Train & evaluate XGBoost with `x81`.
- **Experiment 1B**: Train & evaluate XGBoost without `x81`.
- **Threshold Optimization**: Train on `Time <= 8`, tune threshold on `Time 9..10`, freeze and evaluate once on `Time 11..14`.
- **SHAP & Feature Importance**: Compute SHAP tree explainers for both experiments.
- **Multimodal Evaluation**: Rigorous evaluation of models A–E on consistent test populations.

## 8. Final Architecture
```
Financial Data (422 Companies, 82 Features without leakage)
                       │
                       ▼
          Temporal Split & Imputation (Train-only)
                       │
                       ▼
          XGBoost Financial Risk Classifier
                       │
                       ▼
        Financial Distress Probability & Score (0–100)
                       │
                       ├────────────────────────┐
                       ▼                        ▼
       FinBERT Sentiment Engine         Rule-Based Fusion Engine
       (Text-level Sentiment Risk)       (Adaptive / Baseline)
                       │                        │
                       ▼                        ▼
           Temporal LSTM Model       Final Multimodal Risk
                       │                        │
                       └────────────────────────┘
                                    │
                                    ▼
                         FastAPI + Next.js App
```

## 9. Reproducibility Plan
- All scripts executed deterministically with explicit seeds (`random_state=42`).
- Preprocessing and model artifacts saved into `data/processed/` and `models/`.
- All metrics saved in structured JSON reports under `reports/`.
