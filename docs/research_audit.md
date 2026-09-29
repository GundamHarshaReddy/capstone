# Independent ML Research & Engineering Audit Report

**Project Title**: Dynamic Multimodal Financial Risk & Supplier Disruption Risk Assessment  
**Auditor Role**: Senior ML Research Auditor & Research Software Engineer  
**Audit Date**: September 29, 2026  
**Target Repository**: `capstone`  

---

## 1. Executive Summary

This independent research audit evaluated the data integrity, machine learning methodology, code implementation, experimental validity, and research claims of the **Dynamic Multimodal Financial Risk & Supplier Disruption Risk Assessment** system.

The codebase consists of a complete, working software system including an XGBoost financial risk model, FinBERT news sentiment service, PyTorch 90-day temporal LSTM architecture, risk fusion engines (baseline and adaptive), FastAPI REST backend, Next.js interactive dashboard, and automated integration test suites.

**Key Findings**:
1. **Financial Model Validity**: The XGBoost financial distress classifier is empirically trained and evaluated on 1,083 unseen temporal test observations (Time periods 11–14). It achieves a verified **ROC-AUC of 0.9270** and **Accuracy of 96.68%** (F1-score: 0.3793, Recall: 35.48%, Precision: 40.74%).
2. **Data Leakage Safeguards**: Temporal splitting (`Time <= 10` train, `Time > 10` test) and train-set-only scaling prevent future lookahead data leakage.
3. **News & Multimodal Data Absence**: No real historical news dataset is present in `data/news/`. The FinBERT sentiment analyzer, 90-day sequence generator, and PyTorch LSTM are fully functional code modules, but operate without empirical news data.
4. **Target Label Scope**: The dataset strictly represents **financial distress** (Kaggle benchmark). No ground-truth supplier disruption labels exist. The implementation correctly avoids mislabeling financial distress as supplier disruption.
5. **Adaptive Risk Fusion Nature**: The adaptive risk fusion module is a **manually engineered heuristic algorithm**, not a supervised learned model.

---

## 2. Dataset Validity

- **Raw Source File**: `data/raw/Financial Distress.csv`
- **Total Observations**: 3,672 rows
- **Feature Space**: 86 columns (1 Company ID, 1 Time Period, 1 continuous target `Financial Distress`, 83 features `x1` to `x83`).
- **Company Count**: 422 distinct companies (Company ID 1 to 422).
- **Time Periods**: 14 distinct time periods (Time 1 to 14). Mean periods per company: ~8.7.
- **Target Construction**: Binary target $Y = 1$ if `Financial Distress <= -0.5`, else $Y = 0$.
- **Class Distribution**: 
  - Healthy ($Y=0$): 3,536 observations (96.30%)
  - Distressed ($Y=1$): 136 observations (3.70%)
  - Class Imbalance Ratio: ~26:1.
- **Temporal Splitting**: Train set uses `Time <= 10` (2,589 samples, 105 distress cases). Test set uses `Time 11..14` (1,083 samples, 31 distress cases).
- **Lagged Target Analysis (`x81`)**: Feature `x81` is verified to be `Financial Distress_{t-1}` ($r = 0.443$). In panel time-series prediction at time $t$, $Y_{t-1}$ is known at $t-1$ and is legitimate for auto-regressive forecasting, but acts as a strong shortcut feature.

---

## 3. XGBoost Validation

- **Script Path**: `scripts/train_xgboost.py` & `scripts/evaluate_xgboost.py`
- **Saved Model**: `models/xgboost_financial_risk.pkl`
- **Class Imbalance Management**: `scale_pos_weight = 23.66` (ratio of healthy to distressed samples in training set).
- **Hyperparameters**: `n_estimators=150`, `max_depth=4`, `learning_rate=0.05`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="logloss"`.
- **Reproduced Empirical Metrics (Test Set: 1,083 samples)**:
  - **Accuracy**: 0.9668 (96.68%)
  - **Precision**: 0.4074 (40.74%)
  - **Recall**: 0.3548 (35.48%)
  - **F1-Score**: 0.3793 (0.3793)
  - **ROC-AUC**: 0.9270 (92.70%)
  - **PR-AUC**: 0.3236 (32.36%)
  - **Confusion Matrix**: `TN = 1036, FP = 16, FN = 20, TP = 11`

**Assessment**: All XGBoost metrics reproduce 100% deterministically. The high ROC-AUC (0.9270) confirms strong risk ranking capability. Low recall (35.48%) at the default 0.5 classification threshold indicates that 20 out of 31 test distress events were missed, suggesting optimal threshold tuning (e.g. threshold = 0.35) would improve recall.

---

## 4. News Data Validation

- **Directory Path**: `data/news/`
- **Raw News CSV Present**: **NO**. `data/news/raw_news.csv` is absent in the workspace.
- **News Article Count**: 0 real articles.
- **Companies with News**: 0.
- **News Pipelines**: `scripts/process_news.py`, `scripts/run_finbert.py`, and `scripts/prepare_sequences.py` are functional code pipelines, but no real news dataset exists in the workspace.
- **Assessment**: The news processing pipeline is structurally complete as a software template, but **has zero empirical data validation**.

---

## 5. FinBERT Validation

- **Service Path**: `backend/services/news_sentiment.py`
- **Model Architecture**: Hugging Face `ProsusAI/finbert` (`AutoTokenizer` and `AutoModelForSequenceClassification`).
- **Outputs**: $[p_{\text{pos}}, p_{\text{neg}}, p_{\text{neu}}]$.
- **Sentiment Risk Formula**:
  $$\text{net\_sentiment} = p_{\text{pos}} - p_{\text{neg}}$$
  $$\text{sentiment\_risk\_score} = \text{clip}\left(\frac{1.0 - \text{net\_sentiment}}{2.0} \times 100, 0, 100\right)$$
- **Execution Status**: FinBERT code and inference pipeline are verified and functional via unit tests, but 0 real news articles were processed from the dataset.

---

## 6. LSTM Validation

- **Script Path**: `scripts/train_lstm.py` & `scripts/evaluate_lstm.py`
- **Saved Model**: `models/news_lstm.pt`
- **Input Dimension**: Sequence tensor of shape `(batch_size, 90, 5)`.
- **Features**: `[daily_sentiment_risk, mean_negative, mean_positive, negative_news_ratio, article_count]`.
- **Architecture**: 2-layer PyTorch LSTM (`hidden_dim=32`, `dropout=0.2`), FC dense layer, Sigmoid activation.
- **Supervised Training/Evaluation**: **NOT EVALUATED**.
- **Reason**: No ground-truth temporal news disruption target labels exist in the raw dataset.
- **Assessment**: The LSTM network architecture and forward-pass inference engine are fully verified, but supervised performance claims cannot be made. The project documentation correctly labels this component as **NOT EVALUATED**.

---

## 7. Fusion Validation

- **Service Path**: `backend/services/risk_fusion.py`
- **Baseline Fusion**: Fixed weight combination ($\text{Final Risk} = 0.60 \cdot \text{Fin Risk} + 0.40 \cdot \text{News Risk}$).
- **Adaptive Fusion Formulation**:
  $$\alpha = \min\left(1.0, \frac{\text{article\_count}}{10.0}\right)$$
  $$w_{\text{news}} = \text{clip}\left(0.40 \cdot \alpha \cdot (1.0 + 0.2 \cdot \text{volatility}), 0.0, 0.70\right)$$
  $$w_{\text{fin}} = 1.0 - w_{\text{news}}$$
  $$\text{Final Risk} = w_{\text{fin}} \cdot \text{Financial Risk} + w_{\text{news}} \cdot \text{News Risk}$$
- **Nature of Fusion**: Manually engineered domain heuristic function. It is **NOT** a learned neural network or supervised model.
- **Empirical Evaluation**: Unvalidated against joint ground truth due to absence of combined disruption targets.

---

## 8. Ablation Validation

| Model ID | Configuration | Target Dataset | Status | Accuracy | F1-Score | ROC-AUC | PR-AUC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A** | Financial Model Only (XGBoost) | Kaggle Financial Distress (Time 11–14) | **EVALUATED** | **96.68%** | **0.3793** | **0.9270** | **0.3236** |
| **Model B** | Financial + Current News (FinBERT) | None (Missing raw news) | NOT EVALUATED | N/A | N/A | N/A | N/A |
| **Model C** | Financial + 90d LSTM (Baseline Fusion) | None (Missing news labels) | NOT EVALUATED | N/A | N/A | N/A | N/A |
| **Model D** | Financial + 90d LSTM (Adaptive Fusion) | None (Missing news labels) | NOT EVALUATED | N/A | N/A | N/A | N/A |

**Assessment**: The ablation study strictly adheres to scientific integrity by marking Models B, C, D as NOT EVALUATED rather than fabricating false comparative metrics.

---

## 9. Leakage Audit

1. **Temporal Lookahead Leakage**: **PREVENTED**. Split cleanly by `Time <= 10` (train) and `Time > 10` (test).
2. **Scaler Leakage**: **PREVENTED**. `StandardScaler` fitted exclusively on training set.
3. **Lagged Target Feature (`x81`)**: `x81` is `Financial Distress_{t-1}`. Valid for auto-regressive panel prediction at time $t$, but presents a strong statistical shortcut.
4. **Target Column Separation**: Raw target `Financial Distress` and binary `target` strictly excluded from input feature matrix $X$.
5. **Preprocessing Imputation Leakage**: Minor vector — `fillna(df.median())` was computed globally prior to splitting, though raw dataset missing count is 0.

---

## 10. Reproducibility Audit

All model training, dataset preprocessing, metric evaluation, and integration test commands run with **100% deterministic reproducibility**:

```bash
python3 scripts/preprocess.py                # Exited 0
python3 scripts/train_xgboost.py             # Exited 0
python3 scripts/evaluate_xgboost.py          # Exited 0
python3 scripts/run_ablation_study.py        # Exited 0
python3 tests/test_pipeline.py -v            # Exited 0
npm run build (in frontend/)                 # Compiled in 2.2s (0 errors)
```

---

## 11. Research Claims Audit

| Claim | Source | Evidence | Valid? | Required Correction |
| :--- | :--- | :--- | :--- | :--- |
| **"XGBoost financial distress model achieves 0.9270 ROC-AUC"** | `README.md`, Reports | Test set evaluation on 1,083 samples | **VALID** | None. |
| **"Financial distress targets represent supplier disruption"** | Conceptual | Dataset represents Kaggle financial distress ratios | **INVALID** | Clearly distinguish financial risk from supplier disruption. |
| **"Multimodal news sentiment and temporal LSTM models are empirically validated"** | Conceptual | No raw news dataset present in workspace | **INVALID** | Explicitly state news pipeline is an unvalidated structural template. |
| **"Adaptive risk fusion is a learned model"** | Services | Code uses manually defined heuristic weighting | **INVALID** | Clarify that adaptive fusion is a domain heuristic function. |

---

## 12. Critical Problems

1. **[CRITICAL] Missing Real News Dataset**: `data/news/raw_news.csv` does not exist. FinBERT, news sequence generator, and PyTorch LSTM run without empirical data.
2. **[HIGH] Lack of Supplier Disruption Ground Truth**: Kaggle dataset labels financial distress, not physical supplier disruption.
3. **[MEDIUM] Default Classification Threshold Inefficiency**: At threshold = 0.5, XGBoost recall is 35.48% (FN=20 out of 31).
4. **[LOW] Minor Pre-Split Imputation**: `fillna` executed on full DataFrame before train/test split.

---

## 13. Recommended Corrections

1. **Explicit Dataset Labeling**: Maintain clear separation between "Financial Distress Risk" and "Supplier Disruption Risk" in all academic writeups.
2. **Document Unvalidated Modules**: Ensure documentation explicitly marks FinBERT and LSTM sequence pipelines as *architectural framework components pending empirical news data ingestion*.
3. **Decision Threshold Tuning**: Optimize XGBoost classification threshold (e.g. $\tau = 0.35$) to improve Recall and F1-score on distressed companies.
4. **Heuristic Clarification**: Document adaptive risk fusion explicitly as a rule-based domain heuristic rather than a learned neural model.

---

## 14. Final Research Readiness

### GO / NO-GO DETERMINATION

### **GO (CONDITIONAL FOR FINANCIAL RISK ASSESSMENT)**
### **NO-GO (FOR CLAIMING FULL MULTIMODAL SUPPLIER DISRUPTION VALIDATION)**

**Justification**:
- **Supported by Evidence**: The XGBoost financial distress prediction model, temporal splitting methodology, feature importance explainability, FastAPI backend, Next.js dashboard, and system architecture are **fully valid, empirically tested, and 100% reproducible**.
- **Not Supported by Evidence**: Claims of empirical validation for FinBERT news sentiment, 90-day LSTM temporal modeling, or supplier disruption prediction are **not supported by data** due to the absence of raw news datasets and disruption ground-truth labels.

---

*End of Independent Audit Report. Awaiting instructions before modifying any implementation files.*
