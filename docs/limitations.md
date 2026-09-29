# Scientific Limitations & Scope of Research

## 1. Disruption Target Scope
The primary tabular dataset represents **corporate financial distress** (measured via Altman-style financial accounting ratios on the Tehran Stock Exchange). It does **not** contain factory shutdowns, maritime port bottlenecks, or component supply disruptions. Conflating financial distress with supply disruption would be methodologically invalid. Consequently, this study scopes its claims strictly to **Financial Distress and Market Sentiment Risk Assessment**.

## 2. Anonymized Financial Identities vs. Named Corporate Entities
The Kaggle Financial Distress benchmark dataset indexes firms by anonymized integer IDs (`1` to `422`). To preserve scientific integrity, we do not fabricate an artificial cross-walk between these integer IDs and real company names. Instead, the tabular and textual modalities are evaluated on their respective benchmark populations under strict temporal separation, and unified through the multimodal service layer.

## 3. Standalone Predictive Ceiling of Sentiment Sequences
The PyTorch LSTM trained on 90-day sentiment sequences achieved a test ROC-AUC of 0.5435 and PR-AUC of 0.0416 on 10,821 chronological sequences from 2022 to 2024. This confirms that news sentiment alone contains low signal-to-noise ratio for predicting long-horizon forward distress events, functioning primarily as an orthogonal situational moderator rather than a replacement for financial accounting indicators.

## 4. Heuristic Adaptive Fusion Nature
The adaptive risk fusion mechanism dynamically modulates news weighting based on reporting article volume $\alpha$ and sentiment volatility $\sigma$. While mathematically sound and practically robust, it is a **rule-based domain heuristic** rather than a parameter-optimized neural network.

## 5. Non-Causal Nature of Explanations
Feature importances derived from XGBoost gain and SHAP values reflect tree decision path correlations within the training distribution. They must not be construed as prescriptive causal interventions.
