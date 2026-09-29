# Final Peer-Review Verification & Scientific Methodological Report

**Project Title**: Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News  
**Standard**: Scopus-Indexed Peer-Reviewed Journal Publication Criteria  
**Role**: Senior ML Research Auditor, Principal Data Scientist & Reproducibility Engineer  
**Date**: September 29, 2026  
**Final Decision**: **OPTION B — RESEARCH READY AS FINANCIAL + NEWS RISK-SCORING FRAMEWORK**  

---

## 1. Research Question

### Original Inception vs. Final Methodological Formulation
The research inquiry under investigation is:
> *Can historical financial indicators and temporal financial-news information jointly improve prediction of future corporate financial distress compared with financial-only and news-only baselines?*

### Methodological Resolution
To answer this research question in a supervised empirical setting, a single common ground-truth target must link company-date observations across both modalities. As proven in Section 3 and Section 4 of this report, the tabular financial dataset (Kaggle Financial Distress panel) and the textual news dataset (S&P 500 Financial News corpus) represent disjoint corporate populations with zero legitimate cross-walk key. 

Furthermore, searching legitimate public sources (SEC EDGAR, LoPucki Bankruptcy Research Database, CrisisNet) confirms that public S&P 500 corporations experience near-zero bankruptcy events ($<0.5\%$) across 2006–2024, making supervised distress modeling on S&P 500 news entities statistically unviable without fabricating synthetic data.

In accordance with strict scientific research integrity:
1. We do **not** manufacture a common target merely to force a multimodal prediction claim.
2. We do **not** manufacture a company mapping between Iranian panel firms and US public equities.
3. We do **not** claim that multimodal fusion achieves superior predictive ROC-AUC/PR-AUC on financial distress over the financial baseline.

Instead, the research question is answered through **Option B**:
- The **Tabular Financial Modality** is independently and rigorously evaluated as a supervised predictor of corporate financial distress.
- The **Temporal News Modality** is independently and rigorously evaluated as an auxiliary temporal forecaster of corporate news sentiment risk shocks.
- The **Multimodal Fusion** is formulated and evaluated as an **Integrated Software Risk-Scoring Decision-Support Framework**, dynamically modulating structural accounting risk with situational market sentiment volatility.

---

## 2. Dataset Sources

### 2.1 Tabular Financial Dataset
* **Benchmark**: Kaggle Financial Distress Prediction Dataset (Ebrahimi, 2019).
* **Underlying Corporate Universe**: Listed manufacturing companies on the Tehran Stock Exchange.
* **Volume**: 3,672 company-period panel observations across 422 unique companies.
* **Temporal Horizon**: 14 discrete sequential time periods ($\text{Time} \in [1, 14]$).
* **Feature Dimensionality**: 83 numerical features ($x1$ to $x83$):
  - 82 continuous financial ratios (liquidity, solvency, leverage, asset turnover, profitability).
  - 1 categorical feature ($x80$, industrial sector classification).
  - 1 autoregressive lagged target feature ($x81 = Y_{t-1}$).
* **Ground-Truth Target**: Continuous financial distress score (`Financial Distress`), converted via standard literature threshold into a binary distress indicator ($Y_{\text{fin}} = \mathbb{I}(\text{Financial Distress} \le -0.50)$). Total positive instances: 136 (3.70% prevalence; 26:1 class imbalance).

### 2.2 Textual News Dataset
* **Benchmark**: `KrossKinetic/SP500-Financial-News-Articles-Time-Series` (hosted on Hugging Face).
* **Underlying Corporate Universe**: Standard & Poor's 500 index public equities (United States).
* **Volume**: 4,589 historical corporate news articles across 469 unique public companies.
* **Temporal Horizon**: December 2006 to April 2024 (exact calendar timestamps).
* **Feature Dimensionality**: Headline text, full article narrative, publication timestamp, source link, and ticker symbol.
* **Ground-Truth Target in Raw Data**: Unstructured news text corpus.

---

## 3. Dataset Compatibility Audit

| Dimension | Tabular Financial Dataset | Textual News Dataset | Compatibility Finding |
| :--- | :--- | :--- | :--- |
| **Geographic Jurisdiction** | Iran (Tehran Stock Exchange) | United States (NYSE / NASDAQ) | **Total Discrepancy**: Completely disjoint corporate populations. |
| **Company Identifiers** | Anonymized integer IDs (`1` to `422`) | US equity ticker symbols (`AAPL`, `MMM`) | **Zero Overlap**: Kaggle provides no cross-walk key. |
| **Temporal Frequency** | Discrete accounting periods ($1 \dots 14$) | Exact daily dates ($2006 \dots 2024$) | **Unanchored**: No calendar date alignment possible. |
| **Reporting Standards** | Iranian National Accounting Standards | US GAAP / SEC EDGAR | **Disparate Accounting Principles**. |
| **Supervised Target** | Financial Distress ($\le -0.50$) | None in raw text; derived sentiment shock | **Disjoint Supervisory Signals**. |

---

## 4. Company Mapping Audit

An exhaustive audit of the repository revealed that an earlier helper script (`prepare_news_and_mapping.py`) sorted unique S&P 500 tickers alphabetically and assigned sequential integers `1` to `470` (`A -> 1`, `AAL -> 2`, `AAPL -> 3`). 

When merged with the Kaggle dataset, this produced spurious pairings where Apple Inc. (`AAPL`) news was assigned to Kaggle Company ID `3` (an anonymized Iranian manufacturing firm).

### Audit Determination
Assigning sequential numerical IDs to S&P 500 tickers to match Kaggle IDs creates **fabricated entity cross-walks**. 
- Firm 3 in the financial dataset is NOT Apple Inc.
- Firm 1 in the financial dataset is NOT Agilent Technologies.

**Mandatory Scientific Conclusion**:
> **The two datasets cannot be directly joined for supervised multimodal financial-distress prediction.**

All artificial joins have been eliminated. In `company_mapping.csv` and backend routes, S&P 500 entities and Kaggle panel firms are decoupled. The backend cleanly sets news weight to zero ($w_{\text{fin}} = 1.0$) for anonymized panel firms when news coverage is unavailable.

---

## 5. Common Target Feasibility Analysis

We evaluated whether an alternative public dataset could supply a genuine common target linking financial ratios and news for the S&P 500 news corpus:
1. **S&P 500 Bankruptcy Sparsity**: Constituents of the S&P 500 are mega-cap solvent corporations. Out of 469 companies in the news corpus between 2006 and 2024, fewer than 0.5% (e.g., Lehman Brothers in 2008, Silicon Valley Bank in 2023) experienced Chapter 7/11 bankruptcy. Over 99.5% of firms experienced zero distress events. Supervised distress prediction on this population is severely label-starved ($N_{\text{distress}} \approx 2\text{--}3$ across 18 years).
2. **SEC EDGAR Panel Construction**: Building an 18-year quarterly balance sheet panel for 469 firms requires licensed commercial platforms (WRDS Compustat / Capital IQ). Scraping ad-hoc financial ratios or synthesizing bankruptcy labels would violate scientific integrity.

### Dual-Track Target Protocol
In accordance with Option B, two distinct ground-truth targets are maintained:
* **Track 1 Target ($Y_{\text{fin}}$)**: Corporate Financial Distress at forward period $t+1$ ($Y = \mathbb{I}(\text{Financial Distress} \le -0.50)$), evaluated on Kaggle panel.
* **Track 2 Target ($Y_{\text{news}}$)**: Forward 30-Day News Sentiment Risk Shock ($\ge 1$ negative article and mean sentiment risk $\ge 60.0$ in $[T+1, T+30]$), evaluated on S&P 500 news sequences.

---

## 6. Temporal Design & Information-Timestamp Protocol

To prevent lookahead data leakage, all models and sequences adhere to the **Strict Information-Timestamp Rule**:

```
PAST INFORMATION ONLY                PREDICTION TIMESTAMP T                FUTURE OUTCOME ONLY
───────────────────────────────────►          ▼          ─────────────────────────────────────►
[t <= T]                                      │          [t > T]
• Financial ratios at or before T             │          • Financial distress at T+1
• News published at or before T (T-89 to T)   │          • News sentiment shocks in [T+1, T+30]
• No future variables allowed                 │          • Never accessible during prediction
```

### Partitioning Protocols
1. **Financial Modality Chronological Splits**:
   - Sub-Training: $\text{Time} \in [1, 8]$ ($N=2,095$; 87 distress cases).
   - Internal Validation: $\text{Time} \in [9, 10]$ ($N=494$; 18 distress cases). Thresholds and hyper-parameters frozen here.
   - Final Held-Out Test: $\text{Time} \in [11, 14]$ ($N=1,083$; 31 distress cases). Fully unseen.
2. **News Modality Chronological Splits**:
   - Training Sequences: $T < \text{2022-01-01}$ ($N=58,035$ rolling 90-day sequences).
   - Final Held-Out Test Sequences: $T \ge \text{2022-01-01}$ ($N=10,821$ rolling 90-day sequences spanning 2022–2024).

---

## 7. Financial Model Evaluation & The x81 Experiment

The tabular financial model is an XGBoost Classifier (tree depth 4, learning rate 0.05, 200 estimators). We conducted an empirical experiment isolating the impact of the lagged distress feature `x81` ($Y_{t-1}$):

### Comparative Empirical Results (Held-Out Test Set: Time 11–14, N=1,083, 31 Distressed)

| Evaluation Metric | Experiment A1 (With x81) | Experiment A2 (Without x81 - Pure Ratios) | Difference / Impact | Scientific Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **Input Features** | 83 features | 82 features | -1 feature (`x81`) | Isolates autonomous ratio predictive power |
| **ROC-AUC** | **0.9270** | **0.9232** | -0.0038 (-0.41%) | Pure ratios retain 99.6% of ranking capability |
| **PR-AUC (Precision-Recall)** | 0.3236 | **0.3764** | **+0.0528 (+16.3%)** | **Higher PR-AUC without x81** |
| **Brier Calibration Score** | 0.0262 | **0.0255** | **-0.0007 (Better)** | Pure ratios produce superior probability calibration |
| **Validation-Tuned Threshold ($\tau$)** | $\tau = 0.40$ | $\tau = 0.45$ | +0.05 | Calibrated on Time 9–10 F1 score |
| **Accuracy (at Optimized $\tau$)** | 96.31% | **96.58%** | +0.27% | High overall accuracy |
| **Precision (at Optimized $\tau$)** | 38.46% | **40.00%** | +1.54% | Calibrated precision boundary |
| **Recall (at Optimized $\tau$)** | **48.39%** | 38.71% | -9.68% | Substantial gain over default 0.50 threshold (35.48%) |
| **F1-Score (at Optimized $\tau$)** | **0.4286** | 0.3934 | -0.0352 | F1 maximized at calibrated threshold |

### Explanation of the x81 Result
1. **Autoregressive State Shortcut**: Feature `x81` is $Y_{t-1}$. In financial panel data, distress exhibits state persistence. Including `x81` provides an autoregressive shortcut, raising recall to 48.39%.
2. **Why x81 is NOT Leakage**: Because $Y_{t-1}$ is chronologically observed at period $t-1 < t$, it is temporally available for forecasting.
3. **Pure Financial Ratio Autonomous Power**: In Experiment A2, when forced to rely purely on 82 accounting ratios (liquidity, debt-to-assets, turnover, operating margins) without `x81`, the model achieves **0.9232 ROC-AUC** and a **higher PR-AUC (0.3764 vs. 0.3236)** with lower Brier calibration error. This demonstrates that the tree ensemble learns fundamental balance-sheet degradation rather than merely repeating past distress status.

---

## 8. News Sentiment Modality (FinBERT)

* **Model**: Pretrained `ProsusAI/finbert` (Financial BERT).
* **Execution**: Batched inference across 4,589 historical S&P 500 news articles (2006–2024 across 469 companies).
* **Empirical Sentiment Distribution**:
  - Positive Articles: 817 (17.80%)
  - Negative Articles: 896 (19.52%)
  - Neutral Articles: 2,876 (62.67%)
* **Risk Score Formulation**:
  $$\text{Net Sentiment} = p_{\text{positive}} - p_{\text{negative}} \in [-1.0, +1.0]$$
  $$\text{Sentiment Risk Score} = \text{clip}\left(\frac{1.0 - \text{Net Sentiment}}{2.0} \times 100, 0, 100\right)$$

---

## 9. Temporal Sequence LSTM

* **Architecture**: 2-Layer PyTorch LSTM (input dimension 6, hidden dimension 32, recurrent dropout 0.2, linear projection to Sigmoid).
* **Input Tensor**: 90-day sequence tensor $\mathbf{X}_i(T) \in \mathbb{R}^{90 \times 6}$ capturing daily sentiment risk, mean negative probability, mean positive probability, negative news ratio, article count, and a binary news availability indicator.
* **Supervisory Target**: Forward 30-Day News Sentiment Risk Shock ($Y_{\text{news}}$).
* **Evaluation on Unseen Test Sequences ($T \ge 2022$, $N=10,821$)**:
  - **ROC-AUC**: **0.5435**
  - **PR-AUC**: **0.0416**
  - **Accuracy (at $\tau=0.50$)**: 91.19%
  - **Precision**: 0.0573
  - **Recall**: 15.19%
  - **F1-Score**: 0.0828
* **Corrected Scientific Interpretation**:
  The news sequence LSTM is evaluated strictly as an **auxiliary sentiment forecasting model**. Its modest ROC-AUC (0.5435) reflects high noise and reporting sparsity in public corporate news flows. It is **never** presented as an empirically validated standalone predictor of corporate balance-sheet distress.

---

## 10. Multimodal Fusion Mechanism

### 10.1 Baseline Fixed Fusion
$$\text{Final Risk}_{\text{baseline}} = 0.60 \times S_{\text{fin}} + 0.40 \times S_{\text{news}}$$

### 10.2 Rule-Based Coverage-Adaptive Fusion
Recognizing that public news coverage is highly heterogeneous across corporate entities, the adaptive fusion dynamically scales weights based on reporting availability factor $\alpha$ and sentiment volatility $\sigma$:
$$\alpha = \min\left(1.0, \frac{\text{article\_count}}{10.0}\right)$$
$$w_{\text{news}} = \text{clip}\left(0.40 \times \alpha \times (1.0 + 0.2 \times \sigma), 0.0, 0.70\right)$$
$$w_{\text{fin}} = 1.0 - w_{\text{news}}$$
$$\text{Final Multimodal Risk} = w_{\text{fin}} \times S_{\text{fin}} + w_{\text{news}} \times S_{\text{news}}$$

### Strict Labeling Standard
The adaptive fusion is explicitly labeled as a **Rule-Based Coverage-Adaptive Fusion Heuristic**. It is **not** called a learned neural model. When a company has zero news coverage ($\alpha = 0$), $w_{\text{news}} = 0$ and the final score defaults cleanly to 100% financial risk ($w_{\text{fin}} = 1.0$).

---

## 11. Comprehensive Ablation Study Matrix

| Model / Framework ID | Modality | Architecture | Target Dataset & Population | ROC-AUC | PR-AUC | Accuracy | F1-Score | Recall | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Model A1** | Tabular Financial | XGBoost (83 features, with x81) | Future Financial Distress ($Y \le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | **0.9270** | 0.3236 | 96.31% | **0.4286** | **48.39%** | **EVALUATED** |
| **Model A2** | Tabular Financial | XGBoost (82 features, NO x81 - Pure Ratios) | Future Financial Distress ($Y \le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | 0.9232 | **0.3764** | **96.58%** | 0.3934 | 38.71% | **EVALUATED** |
| **Model B** | Temporal News | 2-Layer PyTorch LSTM (90-day sequences) | Forward 30-Day News Sentiment Shock<br>S&P 500 Held-out $T \ge 2022$ ($N=10,821$) | 0.5435 | 0.0416 | 91.19% | 0.0828 | 15.19% | **EVALUATED** |
| **Framework C** | Multimodal Static | XGBoost + FinBERT Instant Sentiment Risk | Unified Risk Service Sandbox | Decision-Support | Decision-Support | N/A | N/A | N/A | **INTEGRATED** |
| **Framework D** | Multimodal Temporal | Fixed Weighted Fusion ($0.60 S_{\text{fin}} + 0.40 S_{\text{news}}$) | Unified Risk Service Sandbox | Decision-Support | Decision-Support | N/A | N/A | N/A | **INTEGRATED** |
| **Framework E** | Multimodal Adaptive | Rule-Based Coverage-Adaptive Fusion | Unified Risk Service Sandbox | Decision-Support | Decision-Support | N/A | N/A | N/A | **INTEGRATED** |

---

## 12. Leakage Audit

1. **No Lookahead Contamination**:
   - Financial dataset: Training partitions ($T \le 10$) and test partitions ($T \in [11, 14]$) are strictly separated chronologically. Imputer medians and `StandardScaler` parameters are fit solely on training data.
   - News dataset: Sequences strictly observe the lookback window $[T-89, T]$; forward shock targets are computed on $[T+1, T+30]$. Training sequences ($T < 2022$) and test sequences ($T \ge 2022$) have zero timestamp overlap.
2. **Threshold Tuning Leakage Avoidance**:
   - Optimal thresholds ($\tau = 0.40$ for A1, $\tau = 0.45$ for A2) were selected on internal validation periods (Time 9–10) and frozen before test set inference.

---

## 13. Error Analysis

### Financial Model Error Analysis (Test Set, N=1,083, 31 Distressed)
* **At Default Threshold ($\tau = 0.50$)**:
  - True Positives: 11 / 31 (Recall: 35.48%)
  - False Negatives: 20 (Severe minority under-detection due to 26:1 class imbalance)
  - False Positives: 18 (Precision: 37.93%)
* **At Validation-Optimized Threshold ($\tau = 0.40$, With x81)**:
  - True Positives: 15 / 31 (Recall: 48.39%, +36.4% relative improvement)
  - False Negatives: 16 (Reduced by 20%)
  - False Positives: 24 (Precision: 38.46%)
* **At Validation-Optimized Threshold ($\tau = 0.45$, Without x81)**:
  - True Positives: 12 / 31 (Recall: 38.71%)
  - False Negatives: 19
  - False Positives: 18 (Precision: 40.00%)

### News LSTM Error Analysis (Test Set, N=10,821)
* Due to reporting sparsity, false positives occur when positive macro news temporarily obscures company-specific risk, and false negatives occur during sudden unannounced market shocks. News provides qualitative situational context rather than deterministic insolvency signals.

---

## 14. Statistical Analysis & Limitations of Single Temporal Split

* **Evaluation Standard**: Both benchmarks use realistic forward-looking chronological holdout sets, reflecting actual deployment conditions.
* **Limitation**: Due to discrete panel indexing in the Kaggle dataset ($T=1 \dots 14$) and historical news availability, evaluation was performed on a single chronological test partition (Time 11–14 for financial; 2022–2024 for news). We explicitly report this limitation rather than inventing synthetic cross-validation folds that violate temporal ordering.

---

## 15. Research Limitations

1. **Disjoint Corporate Populations**: The financial dataset (Iranian manufacturing panel) and the news corpus (US S&P 500 equities) cannot be directly joined into a single supervised dataset.
2. **News Reporting Heterogeneity**: Non-mega-cap firms experience significant periods without public news coverage. The framework handles this via coverage-adaptive weighting, but textual signal density remains uneven.
3. **No Physical Disruption Ground Truth**: The models predict financial distress and sentiment shocks, not physical factory shutdowns.

---

## 16. Reproducibility Protocol

All experimental pipelines are automated, deterministic, and self-contained:
1. `python3 scripts/preprocess.py`: Generates leak-free chronological splits.
2. `python3 scripts/train_evaluate_experiments.py`: Trains XGBoost models with and without x81, evaluates test metrics and Brier scores.
3. `python3 scripts/run_finbert.py`: Performs batched FinBERT sentiment inference.
4. `python3 scripts/prepare_sequences.py`: Generates 90-day sequence tensors.
5. `python3 scripts/train_lstm.py`: Trains and evaluates the PyTorch LSTM on forward sentiment shocks.
6. `python3 scripts/run_ablation_study.py`: Produces structured ablation reports and markdown summaries.
7. `python3 -m unittest discover tests -v`: Runs automated 6-point verification suite (100% passing).
8. `cd frontend && npm run build`: Next.js 16 build passes with 0 errors.

---

## 17. Valid Research Contributions

1. **Lagged Target Feature Dissection**: A rigorous empirical demonstration of the autoregressive role of feature `x81`, proving that excluding `x81` yields a pure financial ratio model retaining 0.9232 ROC-AUC with superior PR-AUC (0.3764) and lower Brier calibration error (0.0255).
2. **Validation-Based Threshold Optimization**: Demonstration that internal validation threshold tuning on Time 9–10 increases test distress recall from 35.48% to 48.39% (with x81) and 38.71% (without x81) under 26:1 class imbalance.
3. **Temporal Sequence Modeling of Financial News**: Construction of 68,856 90-day sentiment sequences using FinBERT and an empirical evaluation of a 2-layer PyTorch LSTM predicting forward news sentiment shocks.
4. **Coverage-Adaptive Risk-Scoring Framework**: A software risk-scoring architecture that dynamically adjusts weights between fundamental financial distress probabilities and sentiment volatility based on news reporting availability $\alpha$.

---

## 18. Supported Claims Summary

- [x] XGBoost predicts financial distress with 0.9270 ROC-AUC (with x81) and 0.9232 ROC-AUC (without x81) on 1,083 unseen test samples.
- [x] Pure financial ratios without x81 improve PR-AUC from 0.3236 to 0.3764 and lower Brier score from 0.0262 to 0.0255.
- [x] Internal validation threshold optimization increases distressed class recall to 48.39% (with x81) and 38.71% (without x81).
- [x] FinBERT classifies financial sentiment across 4,589 historical news articles.
- [x] PyTorch LSTM predicts forward news sentiment risk shocks with 0.5435 ROC-AUC on 10,821 test sequences.
- [x] Rule-based coverage-adaptive fusion dynamically reallocates weights based on news availability $\alpha$ and volatility $\sigma$.

---

## 19. Unsupported Claims Audit (Excised & Prohibited)

- [x] **REMOVED**: Claims that multimodal fusion improves financial distress prediction over financial-only baselines.
- [x] **REMOVED**: Claims of predicting physical supplier disruption / factory floor shutdown.
- [x] **REMOVED**: Claims that Company IDs 1–422 map to S&P 500 tickers.
- [x] **REMOVED**: Claims that the news LSTM predicts financial distress.
- [x] **REMOVED**: Claims that adaptive fusion is a learned neural network.
- [x] **REMOVED**: Claims of causal feature attribution.

---

## 20. Final Readiness Decision

### **OPTION B: RESEARCH READY AS FINANCIAL + NEWS RISK-SCORING FRAMEWORK**

**Scientific Justification**:  
The repository adheres to rigorous peer-reviewed research standards. All models are trained and evaluated on authentic historical data under strict chronological splits without data leakage. Because the tabular accounting dataset and textual news corpus cannot share a common supervised target without fabricating corporate identities, the research is framed with complete transparency as an **Integrated Multimodal Risk-Scoring Framework**. Every claim is empirically supported, mathematically defensible, and fully reproducible.
