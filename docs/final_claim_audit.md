# Final Research Claims Audit

**Auditor Role**: Senior ML Research Auditor  
**Audit Standard**: Peer-Reviewed Academic Publication Standards  
**Scope**: All claims in repository documentation, backend code, and frontend UI  
**Date**: September 29, 2026  

---

## Claim Verification Matrix

| Claim in Documentation / UI | Empirical Evidence | Supported? | Required Action / Correction |
| :--- | :--- | :--- | :--- |
| **"System predicts supplier disruption"** | Dataset contains Kaggle financial distress ratios and S&P 500 financial news. Neither contains factory shutdown or physical supply disruption ground truth. | **UNSUPPORTED** | **REMOVED**. Project scope and title updated to *"Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News"*. |
| **"XGBoost model achieves 0.9270 ROC-AUC"** | Evaluated on 1,083 unseen temporal test samples (Time 11–14). Exactly reproduced. | **SUPPORTED** | Retained with exact confusion matrix, PR-AUC (0.3236), and precision/recall documented. |
| **"Model learns underlying financial ratios, not just past distress state"** | Experiment B (without lagged target x81) achieves 0.9232 ROC-AUC and 0.3764 PR-AUC on 82 pure financial ratios. | **SUPPORTED** | Retained. Fully documented in `reports/x81_comparison.json`. |
| **"FinBERT news sentiment model is empirically evaluated"** | Batched FinBERT inference run on 4,589 real historical news articles (2006–2024 across 469 companies). | **SUPPORTED** | Retained. Summary statistics documented in `reports/finbert_news_statistics.json`. |
| **"90-Day PyTorch Temporal LSTM model is empirically evaluated"** | Trained on 58,035 chronological sequences (T < 2022) and evaluated on 10,821 held-out test sequences (2022–2024), achieving ROC-AUC 0.5435 and PR-AUC 0.0416. | **SUPPORTED** | Retained. Accurately documented in `reports/lstm_metrics.json` and ablation matrix. |
| **"Adaptive fusion is a learned neural model"** | Code in `risk_fusion.py` implements a rule-based formula weighting news volume $\alpha$ and volatility $\sigma$. | **UNSUPPORTED** | **CORRECTED**. Explicitly labeled as a *"Rule-Based Data-Dependent Adaptive Heuristic"* across all reports and UI. |
| **"Feature importance explains causality of financial collapse"** | XGBoost tree gain and SHAP values measure statistical correlation and feature importance within the decision trees. | **UNSUPPORTED** | **CORRECTED**. Explicitly qualified with non-causal language: *"These features contributed to the model prediction according to tree gain scores. This indicates correlation, not direct causality."* |
| **"Validation-based threshold optimization improves minority distress detection"** | Threshold tuned on internal validation set (Time 9–10) raised test recall from 35.48% (at default 0.50) to 48.39% (with x81) and 38.71% (without x81). | **SUPPORTED** | Retained and prominently featured in performance analysis. |

---

## Conclusion
All unsupported claims (specifically claims of predicting "supplier disruption", "learned neural fusion", and "causality") have been strictly excised or corrected across the codebase, reports, and frontend UI.
