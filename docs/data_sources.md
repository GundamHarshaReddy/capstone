# Scientific Data Sources & Integrity Documentation

## 1. Primary Financial Ratio Dataset: Kaggle Financial Distress (Ebrahimi, 2019)
- **Dataset Name**: Financial Distress Prediction Dataset
- **URL**: https://www.kaggle.com/datasets/shebrahimi/financial-distress
- **Provider**: Saqib Ebrahimi (Tehran Stock Exchange Study)
- **Access Date**: Verified in workspace
- **License / Terms**: Open Public Domain (CC0 / Kaggle Open Dataset)
- **Number of Records**: 3,672 observations
- **Number of Companies**: 422 unique companies
- **Time Periods**: 14 distinct periods (Time 1 to 14)
- **Target Variable**: Continuous distress value where $\le -0.5$ denotes financial distress ($Y=1$, 136 instances, 3.70%), and $> -0.5$ denotes healthy ($Y=0$, 3,536 instances, 96.30%).
- **Variables**: 83 financial indicators (`x1` to `x83`).
  - Feature `x81` is verified as $Y_{t-1}$ (lagged target from previous period).
  - Evaluated in dual configurations: Experiment A (with `x81`) and Experiment B (without `x81`).
- **Anonymization Constraint**: Company identities in this dataset are strictly anonymized integers (`1` to `422`). There is no public key linking these integers to named corporate entities.

---

## 2. Primary Real Historical Financial News Dataset: S&P 500 Financial News Time Series
- **Dataset Name**: SP500-Financial-News-Articles-Time-Series
- **URL**: https://huggingface.co/datasets/KrossKinetic/SP500-Financial-News-Articles-Time-Series
- **Provider**: KrossKinetic (Open Academic Repository)
- **Access Date**: September 29, 2026
- **License / Terms**: Open Access Research Dataset
- **Number of Records**: 4,589 real historical news articles
- **Number of Companies**: 469 distinct public companies / tickers (e.g., 3M, Abbott, AbbVie, Adobe, AMD, Apple, Microsoft)
- **Date Range**: December 4, 2006 to April 20, 2024
- **Fields Provided**:
  - `symbol`: Ticker symbol (e.g., MMM, AAPL, AMZN)
  - `company`: Official corporate name (e.g., 3M, Apple Inc.)
  - `Publishdate`: Publication date (YYYY-MM-DD)
  - `Title`: Full news article headline
  - `Text`: Complete financial article body text
  - `links`: Source URL citation

---

## 3. Pretrained FinBERT Sentiment Model
- **Model Name**: `ProsusAI/finbert`
- **URL**: https://huggingface.co/ProsusAI/finbert
- **Provider**: Prosus AI (Araci, 2019)
- **Architecture**: Domain-adapted BERT for financial text sentiment classification
- **Classes**: `[positive, negative, neutral]`

---

## 4. Scientific Alignment & Non-Fabrication Protocol
1. **No Fake Joins**: Because the Kaggle financial dataset contains anonymized IDs (1..422) and the financial news dataset contains identified tickers (`MMM`, `AAPL`, etc.), we do **not** manufacture a fictitious mapping between them. Doing so would violate scientific integrity.
2. **Dual-Track Empirical Evaluation**:
   - **Track 1**: Supervised tabular financial distress modeling on 3,672 records (XGBoost with vs. without `x81`, validation threshold tuning).
   - **Track 2**: Empirical FinBERT sentiment inference and 90-day PyTorch LSTM sequence modeling on 4,589 real news articles across 469 companies.
   - **Track 3**: Multimodal risk fusion engine dynamically weighting financial ratios and sentiment momentum.
3. **Disruption Target Guardrail**: Neither dataset contains physical supplier disruption ground-truth events (e.g., factory fires, maritime port strikes). Therefore, in strict compliance with peer-reviewed publication standards, the project title and research claims are scoped as **"Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News"**.
