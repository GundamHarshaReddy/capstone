# Target Definition, Temporal Horizon & Information-Timestamp Protocol

**Project**: Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News  
**Standard**: Peer-Reviewed Academic Research Integrity (Scopus Standards)  
**Date**: September 29, 2026  

---

## 1. Dual-Track Target Definitions

Because the tabular accounting dataset (Kaggle Financial Distress) and the textual news dataset (S&P 500 News Corpus) cannot be directly joined into a single supervised dataset without fabricating corporate identities (see `docs/unified_dataset_strategy.md`), the project maintains two distinct, mathematically precise target definitions.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DUAL-TRACK EVALUATION PROTOCOL                         │
├──────────────────────────────────────┬──────────────────────────────────────────┤
│ Track 1: Tabular Financial Modality  │ Track 2: Temporal News Modality          │
│ Ground-Truth Target:                 │ Ground-Truth Target:                     │
│ Future Corporate Financial Distress  │ Forward 30-Day News Sentiment Risk Shock │
│ Y_fin in {0, 1}                      │ Y_news in {0, 1}                         │
└──────────────────────────────────────┴──────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│               MULTIMODAL COVERAGE-ADAPTIVE RISK-SCORING FRAMEWORK                │
│ Decision-Support Risk Score: S_final = w_fin * S_fin + w_news * S_news          │
│ (Formulated as a decision-support system; NOT claimed as a joint predictor)     │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Modality 1: Corporate Financial Distress Target ($Y_{\text{fin}}$)

### 2.1 Target Formulation
* **Target Symbol**: $Y_{\text{fin}}(i, t+1) \in \{0, 1\}$
* **Underlying Dataset**: Kaggle Financial Distress Panel (Ebrahimi, 2019; Tehran Stock Exchange manufacturing panel).
* **Mathematical Definition**:
  $$Y_{\text{fin}}(i, t+1) = \begin{cases} 1 & \text{if } \text{Financial Distress}_{i, t+1} \le -0.50 \\ 0 & \text{if } \text{Financial Distress}_{i, t+1} > -0.50 \end{cases}$$
* **Threshold Justification**: A continuous score $\le -0.50$ represents severe insolvency, bankruptcy petition, or capital restructuring under established financial distress literature (Ebrahimi et al., 2019).

### 2.2 Prediction Horizon & Class Distribution
* **Prediction Horizon**: 1 discrete forward reporting period ($t+1$).
* **Positive Class ($Y=1$, Distressed)**: Firm experiences financial distress in period $t+1$. Total across dataset: 136 instances (3.70%).
* **Negative Class ($Y=0$, Healthy)**: Firm maintains normal solvency in period $t+1$. Total across dataset: 3,536 instances (96.30%).
* **Class Imbalance**: Approximately 26:1.
* **Labeling Source**: Official stock exchange delisting, insolvency filings, and debt default records compiled in the Ebrahimi benchmark.

### 2.3 Chronological Temporal Partitions
To eliminate lookahead data leakage, samples are partitioned strictly by their discrete temporal index $T$:
* **Sub-Training Set**: $\text{Time} \in [1, 8]$ ($N=2,095$; 87 distressed cases).
* **Internal Validation Set**: $\text{Time} \in [9, 10]$ ($N=494$; 18 distressed cases). Used exclusively for tuning classification thresholds and hyperparameters.
* **Held-Out Final Test Set**: $\text{Time} \in [11, 14]$ ($N=1,083$; 31 distressed cases). Frozen until final model evaluation.

---

## 3. Modality 2: Forward News Sentiment Risk Shock Target ($Y_{\text{news}}$)

### 3.1 Target Formulation
* **Target Symbol**: $Y_{\text{news}}(i, T) \in \{0, 1\}$
* **Underlying Dataset**: S&P 500 Financial News Corpus (KrossKinetic, 2024).
* **Mathematical Definition**:
  $$Y_{\text{news}}(i, T) = \mathbb{I}\left( \sum_{k=T+1}^{T+30} \text{negative\_articles}_{i, k} \ge 1 \quad \text{AND} \quad \overline{\text{sentiment\_risk}}_{i, [T+1, T+30]} \ge 60.0 \right)$$
* **Justification**: A corporate news sentiment risk shock is defined as an acute influx of negative reporting over the subsequent month ($T+1$ to $T+30$) accompanied by an elevated mean FinBERT risk score ($\ge 60.0$ on a $0\text{--}100$ scale).

### 3.2 Prediction Horizon & Class Distribution
* **Input Lookback Window**: 90 consecutive calendar days preceding prediction timestamp $T$ ($[T-89, T]$).
* **Prediction Horizon**: Forward 30 calendar days ($[T+1, T+30]$).
* **Positive Class ($Y=1$, Sentiment Risk Shock)**: Firm suffers negative news shock during the forward 30-day window.
* **Negative Class ($Y=0$, Stable/Neutral)**: Firm maintains normal, neutral, or positive news sentiment flow.
* **Labeling Source**: Automated NLP sentiment labeling via pretrained domain-specific transformer `ProsusAI/finbert`.

### 3.3 Chronological Temporal Partitions
* **Training Sequences**: All prediction dates $T < \text{2022-01-01}$ ($N=58,035$ rolling 90-day sequences).
* **Held-Out Final Test Sequences**: All prediction dates $T \ge \text{2022-01-01}$ ($N=10,821$ sequences spanning 2022–2024).
* **Temporal Cutoff**: Zero overlap between training sequence timestamps and evaluation test sequence timestamps.

---

## 4. Strict Information-Timestamp Rule

To ensure absolute methodological validity for peer review, all features, sequences, and targets adhere strictly to the **Information-Timestamp Rule**:

```
PAST INFORMATION ONLY                PREDICTION TIMESTAMP T                FUTURE OUTCOME ONLY
───────────────────────────────────►          ▼          ─────────────────────────────────────►
[t <= T]                                      │          [t > T]
• Financial ratios at or before T             │          • Financial distress at T+1
• News published at or before T (T-89 to T)   │          • News sentiment shocks in [T+1, T+30]
• No future variables allowed                 │          • Never accessible during prediction
```

### 4.1 Strict Invariants Enforced
1. **No Future News in Past Predictions**: News features at timestamp $T$ use only articles published on or before $T$. No article published on $T+1$ or later is accessible.
2. **No Future Financial Ratios**: Financial features at time $T$ reflect balance-sheet disclosures available at period $T$.
3. **No Target Leakage in Features**: The target occurs strictly AFTER timestamp $T$.
4. **Clarification on Lagged Feature `x81`**:
   - In the financial dataset, feature `x81` represents $Y_{t-1}$ (distress status in the preceding period).
   - Because $Y_{t-1}$ was observed at period $t-1 \le t$, it does not violate chronological availability.
   - However, because it acts as an autoregressive state persistence shortcut, we conduct a full comparative ablation (**Experiment A1 with x81 vs. Experiment A2 without x81**) to verify that the model learns fundamental financial ratios and does not merely repeat past distress status.

---

## 5. Methodological Boundary on Multimodal Claims

1. **No Supervised Common-Target Claim**: The system does NOT claim that combining XGBoost and the news LSTM achieves superior predictive accuracy on financial distress compared to financial-only models, because the two modalities operate on separate empirical populations.
2. **Software Risk-Scoring Framework**: Multimodal fusion is evaluated and presented as a **Decision-Support Risk-Scoring Architecture**. It dynamically reconciles structural accounting risk with real-time market sentiment volatility via rule-based coverage-adaptive weighting.
3. **Separation of Concerns**: The financial model is a validated predictive model of financial distress; the news LSTM is an auxiliary sentiment sequence forecaster; and the fusion engine is a risk-scoring software framework.
