# Unified Dataset Strategy & Entity Resolution Audit

**Project**: Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News  
**Standard**: Scopus Peer-Reviewed Academic Research Integrity  
**Date**: September 29, 2026  
**Auditor**: Senior ML Research Auditor & Methodology Lead  

---

## 1. Executive Summary

This document presents a rigorous methodological audit of the two empirical datasets utilized in this research project:
1. **Financial Panel Dataset**: Kaggle Financial Distress Prediction (Ebrahimi, 2019).
2. **Textual News Dataset**: S&P 500 Financial News Articles Time Series (KrossKinetic, 2024).

The central methodological inquiry addressed here is:
> *Can historical financial indicators and temporal financial-news information be legitimately joined at the firm-date level into a single supervised common-target multimodal dataset for financial distress prediction without fabricating data or entities?*

Following an exhaustive audit of raw data schemas, company identifiers, temporal resolution, and open public repositories, **this audit concludes that the two datasets cannot be directly joined for supervised multimodal financial-distress prediction**. 

Consequently, to maintain strict scientific integrity and avoid research misconduct, this project adopts **Option B: Research-Ready Financial + News Risk-Scoring Framework**, evaluating individual predictive modalities on their respective ground-truth benchmarks and integrating them into a coverage-adaptive risk-scoring decision-support architecture.

---

## 2. Dataset Schema & Characteristics Audit

### 2.1 Tabular Financial Dataset
* **Source**: Kaggle Financial Distress Prediction Dataset (derived from research on Iranian listed companies by Ebrahimi et al., Tehran Stock Exchange).
* **File Location**: `data/raw/Financial Distress.csv`
* **Observations**: 3,672 company-period panel records.
* **Entities**: 422 unique companies.
* **Entity Identifiers**: Anonymized integer IDs ranging from `1` to `422`.
* **Temporal Dimension**: Discrete integer time steps ranging from `Time = 1` to `Time = 14` (unanchored to calendar dates; approximately annual financial reporting periods).
* **Feature Space**: 83 numerical features ($x1$ through $x83$):
  - 82 continuous financial ratios (liquidity, leverage, profitability, turnover, operating cash flow).
  - 1 categorical industry/company characteristic feature ($x80$).
  - 1 lagged financial distress target feature ($x81 = Y_{t-1}$).
* **Ground-Truth Target**: Continuous financial distress score (`Financial Distress`). Transformed via standard literature threshold into binary distress indicator:
  $$Y_{\text{fin}} = \mathbb{I}(\text{Financial Distress} \le -0.5)$$
  Prevalence: 136 distressed records (3.70% positive class rate across entire panel; 31 distressed records in held-out test periods 11–14).

### 2.2 Textual Financial News Dataset
* **Source**: `KrossKinetic/SP500-Financial-News-Articles-Time-Series` (hosted on Hugging Face).
* **File Location**: `data/news/sp500_news_raw.jsonl` / `data/news/raw_news.csv`
* **Observations**: 4,589 historical corporate news articles.
* **Entities**: 469 public corporations included in the Standard & Poor's 500 index.
* **Entity Identifiers**: US equity ticker symbols (`AAPL`, `MSFT`, `MMM`, `JPM`, etc.) and canonical company names (`Apple Inc.`, `Microsoft`, etc.).
* **Temporal Dimension**: Continuous calendar timestamps with exact publication dates spanning December 2006 to April 2024.
* **Textual Variables**: Article headline title, full body text, publisher source URL, publication date.
* **Ground-Truth Target in Raw Data**: None (raw unstructured text corpus).

---

## 3. Company Join & Entity Resolution Audit

### 3.1 Investigation of Potential Cross-Walk
We examined whether any legitimate, empirical, or source-based cross-walk exists between the 422 Kaggle companies and the 469 S&P 500 news entities.

| Dimension | Kaggle Financial Dataset | S&P 500 News Dataset | Compatibility Finding |
| :--- | :--- | :--- | :--- |
| **Geographic Jurisdiction** | Iran (Tehran Stock Exchange) | United States (NYSE / NASDAQ) | **Total Mismatch**: Completely disjoint corporate universes. |
| **Identifier Scheme** | Anonymous integers (`1` to `422`) | Ticker symbols (`AAPL`, `MMM`) | **No Mapping Key**: Kaggle provides zero metadata mapping integers to real firms. |
| **Temporal Frequency** | Discrete accounting periods ($1 \dots 14$) | Exact daily dates ($2006 \dots 2024$) | **Unanchored**: No calendar alignment possible. |
| **Reporting Standards** | Iranian National Accounting Standards | US GAAP / SEC EDGAR | **Structural Discrepancy**: Different disclosure and audit conventions. |

### 3.2 Evaluation of Artificial Index Assignment
In preliminary iterations of this workspace, a helper script (`prepare_news_and_mapping.py`) assigned sequential integers `1` to `470` to the sorted unique S&P 500 ticker symbols (`A -> 1`, `AAL -> 2`, `AAPL -> 3`, ...). 

This resulted in Kaggle Company ID `3` (an anonymized Iranian manufacturing firm) being matched with news articles for Apple Inc. (`AAPL`).

**Audit Verdict on Artificial Indexing**:
Assigning sequential numerical IDs to S&P 500 tickers to match Kaggle IDs creates **spurious entity pairings**. Treating Apple Inc. news as the text modality for an anonymized manufacturing firm constitutes **data fabrication and pseudoscientific leakage**. 

In strict adherence to peer-review research standards:
> **The two datasets cannot be directly joined for supervised multimodal financial-distress prediction.**

Any claim that a multimodal model trained on such an artificial join improves financial distress prediction is rejected and strictly prohibited.

---

## 4. Alternative Unified Public Dataset Search

In accordance with Step 3 of the peer-review directive, we investigated legitimate public sources for an alternative unified dataset that simultaneously contains:
1. Public company identifier (e.g., CIK, Ticker, LEI).
2. Continuous calendar timestamps.
3. Audited financial ratios / balance sheet fundamentals.
4. Validated corporate financial distress / bankruptcy labels ($Y$).
5. Dense, continuous historical news article coverage for the exact same firms.

### 4.1 Candidate Sources Reviewed

#### A. SEC EDGAR XBRL Data + Florida–UCLA–LoPucki Bankruptcy Research Database (BRD)
* **Description**: SEC 10-K/10-Q financial filings paired with LoPucki Chapter 7/11 bankruptcy filings.
* **Findings for S&P 500 Entities**:
  - The S&P 500 index consists almost exclusively of highly capitalized, solvent corporations.
  - Between 2006 and 2024, corporate bankruptcies among active S&P 500 firms are exceptionally rare ($< 0.5\%$, e.g., Lehman Brothers in 2008, CIT Group in 2009, Silicon Valley Bank in 2023).
  - Out of 469 companies in the news corpus, more than 99.5% experienced zero distress events over the entire 18-year period.
  - Training a supervised machine learning model on this population would face extreme label starvation ($N_{\text{distress}} \approx 2\text{--}3$ across 18 years), rendering supervised predictive modeling statistically unviable without artificial synthetic oversampling.

#### B. CrisisNet (Hugging Face / Energy Sector 2015–2025)
* **Description**: A multi-modal dataset covering 40 energy sector firms.
* **Findings**: Covers only 40 specialized energy firms over a restricted window; lacks generalizability to broader industrial supply chains and does not overlap with the broader S&P 500 news corpus.

#### C. Kaggle US Public Companies Bankruptcy Dataset (1999–2018)
* **Description**: Accounting ratios from 8,262 US public firms with SEC bankruptcy labels.
* **Findings**: Does not contain aligned historical news article text. The available news dataset (`KrossKinetic`) contains only 4,589 articles total across 469 firms over 18 years (an average of only ~0.5 articles per company per year). Over 60% of trading days have zero news coverage for any given firm, creating severe text sparsity.

### 4.2 Feasibility Conclusion
Acquiring an authoritative panel combining quarterly SEC accounting filings, continuous news text, and balanced bankruptcy events across hundreds of public firms requires proprietary, commercial databases (e.g., WRDS Compustat + Capital IQ + RavenPack / Bloomberg News Archive with academic institutional licenses).

Synthesizing balance-sheet ratios via ad-hoc web scraping or inventing distressed labels to force a common target would violate core scientific research ethics:
> *"Do not use fabricated or manually invented financial observations."*  
> *"Do not manufacture a common target merely to make the multimodal experiment possible."*

---

## 5. Strategic Research Framing Decision: Option B

Because a legitimate common-target dataset cannot be constructed without fabrication, this research adopts:

### **OPTION B: RESEARCH READY AS FINANCIAL + NEWS RISK-SCORING FRAMEWORK**

Under Option B:
1. **Scientific Integrity**: No artificial company cross-walk is created. The Kaggle financial panel and the S&P 500 news corpus are kept distinct and evaluated on their genuine respective targets and populations.
2. **Tabular Financial Modality**: Rigorously evaluated as a supervised predictive classifier of future financial distress ($Y_{\text{fin}}$) under strict chronological splits, with comprehensive ablation of lagged target feature `x81` and validation-based threshold optimization.
3. **Temporal News Modality**: Rigorously evaluated as an auxiliary temporal sentiment forecasting model predicting forward 30-day negative sentiment risk shocks ($Y_{\text{news}}$) using 90-day FinBERT sequences in a 2-layer PyTorch LSTM.
4. **Multimodal Integration**: Formulated and demonstrated as a **Software Risk-Scoring Decision-Support Architecture**. The framework dynamically balances structural accounting risk with situational news sentiment volatility through a rule-based coverage-adaptive heuristic.
5. **Defensible Research Claims**: The paper makes **no claim** that multimodal fusion achieves superior predictive ROC-AUC or PR-AUC on financial distress over the financial baseline. Instead, the contribution is framed around the architectural design, leakage prevention protocol, and empirical evaluation of individual modalities under realistic data sparsity.
