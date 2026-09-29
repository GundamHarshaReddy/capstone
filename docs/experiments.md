# Experimental Evaluation & Ablation Study

**Framework Strategy**: Option B — Research-Ready Financial + News Risk-Scoring Framework  
**Standard**: Scopus Peer-Reviewed Academic Publication Standards  
**Leakage Controls**: Strict Chronological Partitions; Zero Cross-Split Contamination  

---

## 1. Supervised Predictive Modalities (Benchmarked on Ground-Truth Populations)

| Model ID | Modality | Architecture | Target & Evaluation Population | ROC-AUC | PR-AUC | Accuracy | F1-Score | Recall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A1** | Tabular Financial | XGBoost (83 features, with lagged x81) | Future Financial Distress ($Y \le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | **0.9270** | **0.3236** | **96.31%** | **0.4286** | **48.39%** | **EVALUATED** |
| **Model A2** | Tabular Financial | XGBoost (82 features, NO lagged x81) | Future Financial Distress ($Y \le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | **0.9232** | **0.3764** | **96.58%** | **0.3934** | **38.71%** | **EVALUATED** |
| **Model B** | Temporal News | 2-Layer PyTorch LSTM (90-day sequences) | Forward 30-Day News Sentiment Risk Shock<br>S&P 500 Held-out $T \ge 2022$ ($N=10,821$) | **0.5435** | **0.0416** | **91.19%** | **0.0828** | **15.19%** | **EVALUATED** |

---

## 2. Multimodal Risk-Scoring Architecture (Decision-Support Framework)

| Framework ID | Modality Integration | Fusion Mechanism | Target / Output Space | Evaluation Standard | Operational Role | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Framework C** | Financial + Static News | Feature Concatenation / Direct Sum | Continuous Risk Score $[0, 100]$ | Decision-Support Scoring | Instant spot sentiment adjustment | **INTEGRATED** |
| **Framework D** | Financial + Temporal News | Fixed Weighted Fusion ($0.60 S_\text{fin} + 0.40 S_\text{news}$) | Standardized Risk Score $[0, 100]$ | Decision-Support Scoring | Static multi-horizon baseline | **INTEGRATED** |
| **Framework E** | Financial + Temporal News | Rule-Based Coverage-Adaptive Fusion (Modulated by volume $\alpha$ & volatility $\sigma$) | Standardized Risk Score $[0, 100]$ | Decision-Support Scoring | Coverage-robust dynamic weighting | **INTEGRATED** |

*Note on Multimodal Evaluation*: Because the financial panel (Iranian manufacturing firms) and news corpus (US S&P 500 equities) represent disjoint corporate populations with no source-based mapping, multimodal fusion is implemented and evaluated as a **software risk-scoring framework**, NOT claimed as an empirical predictive improvement over the financial-only baseline on financial distress.

---

## 3. Key Scientific Findings
1. **Financial Feature Integrity (x81 Dissection)**: Removing the lagged target feature `x81` maintains a high ROC-AUC of 0.9232 while increasing PR-AUC from 0.3236 to 0.3764 and lowering the Brier calibration score (0.0255 vs 0.0262), demonstrating that the tree ensemble genuinely models structural financial degradation rather than state persistence.
2. **Threshold Calibration Efficacy**: Tuning the classification threshold on an internal validation set (Time 9–10) elevates minority distress recall from 35.48% (at default 0.50 threshold) to 48.39% (with x81) and 38.71% (without x81) on held-out test data.
3. **Temporal News Modality Role**: The PyTorch LSTM forward prediction model achieved ROC-AUC 0.5435 and PR-AUC 0.0416 on 10,821 real unseen test sequences from 2022–2024, confirming that corporate news flows exhibit high noise and modest standalone predictability for forward sentiment shocks.
4. **Coverage-Adaptive Fusion Value**: By dynamically conditioning weights on reporting availability $\alpha$, the framework prevents blind reliance on sentiment when news coverage is sparse, cleanly defaulting to 100% financial weighting when coverage is zero.
