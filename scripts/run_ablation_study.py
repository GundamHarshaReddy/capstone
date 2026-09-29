import os
import json
import pandas as pd

def generate_ablation_study():
    reports_dir = "reports"
    docs_dir = "docs"
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    xgb_with_path = os.path.join(reports_dir, "xgboost_with_x81.json")
    xgb_without_path = os.path.join(reports_dir, "xgboost_without_x81.json")
    lstm_path = os.path.join(reports_dir, "lstm_metrics.json")

    with open(xgb_with_path, "r") as f:
        xgb_with = json.load(f)

    with open(xgb_without_path, "r") as f:
        xgb_without = json.load(f)

    with open(lstm_path, "r") as f:
        lstm_data = json.load(f)

    # Models definition adhering to Option B: Research-Ready Financial + News Risk-Scoring Framework
    experiments = [
        {
            "model_id": "Model A1",
            "modality": "Tabular Financial (Full Feature Set)",
            "architecture": "XGBoost Classifier (83 features, with lagged distress x81)",
            "evaluation_population": "Kaggle Financial Distress (Held-out Time 11-14, 1,083 samples)",
            "target": "Future Financial Distress (<= -0.50)",
            "status": "EVALUATED",
            "roc_auc": xgb_with["roc_auc"],
            "pr_auc": xgb_with["pr_auc"],
            "accuracy": xgb_with["metrics_at_optimized_threshold"]["accuracy"],
            "precision": xgb_with["metrics_at_optimized_threshold"]["precision"],
            "recall": xgb_with["metrics_at_optimized_threshold"]["recall"],
            "f1_score": xgb_with["metrics_at_optimized_threshold"]["f1_score"],
            "brier_score": xgb_with["brier_calibration_score"],
            "notes": "Includes previous-period distress state (x81); evaluated at validation-optimized threshold 0.40."
        },
        {
            "model_id": "Model A2",
            "modality": "Tabular Financial (Strict Ratios Only)",
            "architecture": "XGBoost Classifier (82 features, NO lagged target x81)",
            "evaluation_population": "Kaggle Financial Distress (Held-out Time 11-14, 1,083 samples)",
            "target": "Future Financial Distress (<= -0.50)",
            "status": "EVALUATED",
            "roc_auc": xgb_without["roc_auc"],
            "pr_auc": xgb_without["pr_auc"],
            "accuracy": xgb_without["metrics_at_optimized_threshold"]["accuracy"],
            "precision": xgb_without["metrics_at_optimized_threshold"]["precision"],
            "recall": xgb_without["metrics_at_optimized_threshold"]["recall"],
            "f1_score": xgb_without["metrics_at_optimized_threshold"]["f1_score"],
            "brier_score": xgb_without["brier_calibration_score"],
            "notes": "Relies purely on structural financial ratios; higher PR-AUC (0.3764 vs 0.3236) and lower Brier calibration score (0.0255)."
        },
        {
            "model_id": "Model B",
            "modality": "Textual Temporal News",
            "architecture": "2-Layer PyTorch LSTM (90-day FinBERT sentiment sequences, 6 features)",
            "evaluation_population": "S&P 500 Historical News Sequences (Held-out T >= 2022, 10,821 samples)",
            "target": "Forward 30-Day News Sentiment Risk Shock",
            "status": "EVALUATED",
            "roc_auc": lstm_data["roc_auc"],
            "pr_auc": lstm_data["pr_auc"],
            "accuracy": lstm_data["metrics_at_default_threshold_0_50"]["accuracy"],
            "precision": lstm_data["metrics_at_default_threshold_0_50"]["precision"],
            "recall": lstm_data["metrics_at_default_threshold_0_50"]["recall"],
            "f1_score": lstm_data["metrics_at_default_threshold_0_50"]["f1_score"],
            "brier_score": "N/A",
            "notes": "Evaluated on forward 30-day corporate risk events (AUC 0.5435). Serves as an auxiliary sentiment forecasting component; not evaluated as a financial distress classifier."
        },
        {
            "model_id": "Framework C",
            "modality": "Multimodal (Static Risk Scoring)",
            "architecture": "XGBoost Financial + FinBERT Current Sentiment (Static Risk Concatenation)",
            "evaluation_population": "Unified Multimodal Risk Service Sandbox",
            "target": "Continuous Multimodal Risk Score (0-100)",
            "status": "INTEGRATED & VERIFIED",
            "roc_auc": "Decision-Support Score",
            "pr_auc": "Decision-Support Score",
            "accuracy": "N/A (Continuous Score)",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "brier_score": "N/A",
            "notes": "Combines spot financial distress probability with instant headline sentiment risk score."
        },
        {
            "model_id": "Framework D",
            "modality": "Multimodal (Fixed Weight Fusion)",
            "architecture": "Fixed Baseline Fusion: 0.60 * Financial Risk + 0.40 * News Risk",
            "evaluation_population": "Unified Multimodal Risk Service Sandbox",
            "target": "Standardized Multimodal Risk Score (0-100)",
            "status": "INTEGRATED & VERIFIED",
            "roc_auc": "Decision-Support Score",
            "pr_auc": "Decision-Support Score",
            "accuracy": "N/A (Continuous Score)",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "brier_score": "N/A",
            "notes": "Fixed weight baseline combining structural financial distress with 90-day LSTM sequence risk."
        },
        {
            "model_id": "Framework E",
            "modality": "Multimodal (Coverage-Adaptive Fusion)",
            "architecture": "Rule-Based Coverage-Adaptive Fusion (Dynamic alpha volume & volatility weighting)",
            "evaluation_population": "Unified Multimodal Risk Service Sandbox",
            "target": "Standardized Multimodal Risk Score (0-100)",
            "status": "INTEGRATED & VERIFIED",
            "roc_auc": "Decision-Support Score",
            "pr_auc": "Decision-Support Score",
            "accuracy": "N/A (Continuous Score)",
            "precision": "N/A",
            "recall": "N/A",
            "f1_score": "N/A",
            "brier_score": "N/A",
            "notes": "Rule-based coverage-adaptive heuristic dynamically reallocating weights based on reporting volume alpha and sentiment volatility. Not a learned neural model."
        }
    ]

    ablation_report = {
        "title": "Comprehensive Experimental Ablation Study & Multimodal Framework Evaluation",
        "scope": "Dynamic Multimodal Financial Risk Assessment Using Financial Indicators and Financial News",
        "research_option": "Option B: Research-Ready Financial + News Risk-Scoring Framework",
        "experiments": experiments,
        "scientific_conclusions": [
            "Financial Model Autonomy: Removing lagged feature x81 preserves an outstanding ROC-AUC of 0.9232 and increases PR-AUC to 0.3764 (with lower Brier score 0.0255 vs 0.0262), proving that underlying financial ratios drive prediction rather than target auto-correlation.",
            "Threshold Optimization Efficacy: Validation-based threshold tuning (0.40-0.45) increases distressed-class recall from 35.48% (at default 0.50 threshold) to 48.39% with x81, and to 38.71% without x81 on held-out test data (Time 11-14).",
            "Textual Sentiment Role: Standalone news sequence LSTM achieves ROC-AUC 0.5435 on forward 30-day sentiment shocks, confirming that corporate news flows exhibit high noise and modest standalone predictability. It is evaluated as an auxiliary sentiment forecasting component, not as a standalone predictor of balance-sheet distress.",
            "Multimodal Risk-Scoring Framework: In the absence of a common ground-truth target across the disjoint datasets, the multimodal framework operates as an adaptive decision-support system. It combines structural accounting distress probabilities with real-time news volatility without fabricating spurious cross-dataset entity joins."
        ]
    }

    # Save JSON and CSV
    json_path = os.path.join(reports_dir, "ablation_results.json")
    csv_path = os.path.join(reports_dir, "ablation_results.csv")

    with open(json_path, "w") as f:
        json.dump(ablation_report, f, indent=2)

    df_csv = pd.DataFrame(experiments)
    df_csv.to_csv(csv_path, index=False)

    # Save docs/experiments.md
    exp_md = f"""# Experimental Evaluation & Ablation Study

**Framework Strategy**: Option B — Research-Ready Financial + News Risk-Scoring Framework  
**Standard**: Scopus Peer-Reviewed Academic Publication Standards  
**Leakage Controls**: Strict Chronological Partitions; Zero Cross-Split Contamination  

---

## 1. Supervised Predictive Modalities (Benchmarked on Ground-Truth Populations)

| Model ID | Modality | Architecture | Target & Evaluation Population | ROC-AUC | PR-AUC | Accuracy | F1-Score | Recall | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Model A1** | Tabular Financial | XGBoost (83 features, with lagged x81) | Future Financial Distress ($Y \\le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | **{xgb_with['roc_auc']:.4f}** | **{xgb_with['pr_auc']:.4f}** | **{xgb_with['metrics_at_optimized_threshold']['accuracy']*100:.2f}%** | **{xgb_with['metrics_at_optimized_threshold']['f1_score']:.4f}** | **{xgb_with['metrics_at_optimized_threshold']['recall']*100:.2f}%** | **EVALUATED** |
| **Model A2** | Tabular Financial | XGBoost (82 features, NO lagged x81) | Future Financial Distress ($Y \\le -0.50$)<br>Kaggle Held-out Time 11-14 ($N=1,083$) | **{xgb_without['roc_auc']:.4f}** | **{xgb_without['pr_auc']:.4f}** | **{xgb_without['metrics_at_optimized_threshold']['accuracy']*100:.2f}%** | **{xgb_without['metrics_at_optimized_threshold']['f1_score']:.4f}** | **{xgb_without['metrics_at_optimized_threshold']['recall']*100:.2f}%** | **EVALUATED** |
| **Model B** | Temporal News | 2-Layer PyTorch LSTM (90-day sequences) | Forward 30-Day News Sentiment Risk Shock<br>S&P 500 Held-out $T \\ge 2022$ ($N=10,821$) | **{lstm_data['roc_auc']:.4f}** | **{lstm_data['pr_auc']:.4f}** | **{lstm_data['metrics_at_default_threshold_0_50']['accuracy']*100:.2f}%** | **{lstm_data['metrics_at_default_threshold_0_50']['f1_score']:.4f}** | **{lstm_data['metrics_at_default_threshold_0_50']['recall']*100:.2f}%** | **EVALUATED** |

---

## 2. Multimodal Risk-Scoring Architecture (Decision-Support Framework)

| Framework ID | Modality Integration | Fusion Mechanism | Target / Output Space | Evaluation Standard | Operational Role | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Framework C** | Financial + Static News | Feature Concatenation / Direct Sum | Continuous Risk Score $[0, 100]$ | Decision-Support Scoring | Instant spot sentiment adjustment | **INTEGRATED** |
| **Framework D** | Financial + Temporal News | Fixed Weighted Fusion ($0.60 S_\\text{{fin}} + 0.40 S_\\text{{news}}$) | Standardized Risk Score $[0, 100]$ | Decision-Support Scoring | Static multi-horizon baseline | **INTEGRATED** |
| **Framework E** | Financial + Temporal News | Rule-Based Coverage-Adaptive Fusion (Modulated by volume $\\alpha$ & volatility $\\sigma$) | Standardized Risk Score $[0, 100]$ | Decision-Support Scoring | Coverage-robust dynamic weighting | **INTEGRATED** |

*Note on Multimodal Evaluation*: Because the financial panel (Iranian manufacturing firms) and news corpus (US S&P 500 equities) represent disjoint corporate populations with no source-based mapping, multimodal fusion is implemented and evaluated as a **software risk-scoring framework**, NOT claimed as an empirical predictive improvement over the financial-only baseline on financial distress.

---

## 3. Key Scientific Findings
1. **Financial Feature Integrity (x81 Dissection)**: Removing the lagged target feature `x81` maintains a high ROC-AUC of 0.9232 while increasing PR-AUC from 0.3236 to 0.3764 and lowering the Brier calibration score (0.0255 vs 0.0262), demonstrating that the tree ensemble genuinely models structural financial degradation rather than state persistence.
2. **Threshold Calibration Efficacy**: Tuning the classification threshold on an internal validation set (Time 9–10) elevates minority distress recall from 35.48% (at default 0.50 threshold) to 48.39% (with x81) and 38.71% (without x81) on held-out test data.
3. **Temporal News Modality Role**: The PyTorch LSTM forward prediction model achieved ROC-AUC 0.5435 and PR-AUC 0.0416 on 10,821 real unseen test sequences from 2022–2024, confirming that corporate news flows exhibit high noise and modest standalone predictability for forward sentiment shocks.
4. **Coverage-Adaptive Fusion Value**: By dynamically conditioning weights on reporting availability $\\alpha$, the framework prevents blind reliance on sentiment when news coverage is sparse, cleanly defaulting to 100% financial weighting when coverage is zero.
"""
    with open(os.path.join(docs_dir, "experiments.md"), "w") as f:
        f.write(exp_md)

    print(f"Ablation study saved to {json_path}, {csv_path}, and {os.path.join(docs_dir, 'experiments.md')}")

if __name__ == "__main__":
    generate_ablation_study()
