# Rigorous Scientific Methodology & Mathematical Formulations

## 1. Research Scope & Problem Definition
**Title**: Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News  
**Objective**: To assess corporate financial risk and probability of financial distress through a multimodal paradigm combining tabular financial ratio analytics (XGBoost), corporate news sentiment NLP (FinBERT), and chronological sequence momentum (PyTorch LSTM).

---

## 2. Tabular Financial Distress Modeling (Modality 1)

### Dataset
- 3,672 panel observations across 422 companies over 14 discrete time periods (Time 1 to 14).
- Binary classification target: $Y = \mathbb{I}(\text{Financial Distress} \le -0.5)$. Positive distress class prevalence: 3.70% (136 instances).

### Preprocessing & Leakage Controls
- **Temporal Partitioning**:
  - Full Training Set: $\text{Time} \in [1, 10]$ ($N=2,589$, $N_{\text{distress}}=105$).
  - Sub-Training Set: $\text{Time} \in [1, 8]$ ($N=2,095$, $N_{\text{distress}}=87$).
  - Internal Validation Set: $\text{Time} \in [9, 10]$ ($N=494$, $N_{\text{distress}}=18$).
  - Held-out Evaluation Set: $\text{Time} \in [11, 14]$ ($N=1,083$, $N_{\text{distress}}=31$).
- **Imputation & Normalization**: Median vectors and `StandardScaler` transformations are strictly fitted exclusively on training partitions.

### The Lagged Target Experiment (`x81`)
Feature `x81` is verified to be $Y_{t-1}$ ($r = 0.443$). Two distinct scientific experiments were evaluated:
- **Experiment A (With x81)**: 83 features. Evaluates predictive capacity when previous-period distress state is observed.
- **Experiment B (Without x81)**: 82 features. Isolates the autonomous predictive capability of fundamental financial ratios without reliance on state autocorrelation.

### Classification Threshold Calibration
To resolve minority class under-detection under severe imbalance (~26:1), decision boundaries were calibrated using candidate thresholds $\tau \in [0.10, 0.60]$ on the internal validation set ($\text{Time} \in [9, 10]$). The optimal threshold maximizing validation F1-score ($\tau = 0.40$ with x81, $\tau = 0.45$ without x81) was frozen prior to test evaluation.

---

## 3. Financial News Sentiment & Sequence Modeling (Modality 2)

### Corporate News Corpus
- 4,589 historical financial news articles spanning December 2006 to April 2024 across 469 S&P 500 corporate entities.

### FinBERT Sentiment Extraction
Pretrained `ProsusAI/finbert` computes softmax class probabilities:
$$\mathbf{p}_{\text{sentiment}} = [p_{\text{positive}}, p_{\text{negative}}, p_{\text{neutral}}]$$
$$\text{Net Sentiment} = p_{\text{positive}} - p_{\text{negative}} \in [-1.0, +1.0]$$
$$\text{Sentiment Risk Score} = \text{clip}\left(\frac{1.0 - \text{Net Sentiment}}{2.0} \times 100, 0, 100\right)$$

### Temporal Sequence Generation
For firm $i$ on prediction date $T$, a 90-day historical tensor $\mathbf{X}_i(T) \in \mathbb{R}^{90 \times 6}$ is constructed using daily features:
1. `daily_sentiment_risk`
2. `mean_negative`
3. `mean_positive`
4. `negative_news_ratio`
5. `article_count`
6. `news_available` (explicit reporting indicator)

### PyTorch LSTM Architecture
- Input dimension: 6
- 2 LSTM layers with hidden dimension 32 and recurrent dropout 0.2
- Linear projection layer followed by Sigmoid activation
- Supervisory Target: Forward 30-day sentiment risk shock $Y_i(T) \in \{0, 1\}$
- Chronological Split: Train on sequences with $T < \text{2022-01-01}$ ($N=58,035$); evaluate on held-out sequences with $T \ge \text{2022-01-01}$ ($N=10,821$).

---

## 4. Multimodal Risk Fusion Engine

### Baseline Fixed-Weight Fusion
$$\text{Final Risk}_{\text{baseline}} = 0.60 \times \text{Financial Risk} + 0.40 \times \text{News Risk}$$

### Data-Dependent Adaptive Heuristic Fusion
Recognizing that news reporting volume is heterogeneous across firms, an adaptive heuristic reallocates weights based on reporting availability factor $\alpha$ and sentiment volatility $\sigma$:
$$\alpha = \min\left(1.0, \frac{\text{article\_count}}{10.0}\right)$$
$$w_{\text{news}} = \text{clip}\left(0.40 \times \alpha \times (1.0 + 0.2 \times \sigma), 0.0, 0.70\right)$$
$$w_{\text{fin}} = 1.0 - w_{\text{news}}$$
$$\text{Final Multimodal Risk} = w_{\text{fin}} \times \text{Financial Risk} + w_{\text{news}} \times \text{News Risk}$$
When a firm has zero news coverage ($\alpha = 0$), $w_{\text{news}} = 0$ and the final risk cleanly defaults to 100% financial risk.
