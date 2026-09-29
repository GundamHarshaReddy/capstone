import os
import json
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve
)

def evaluate_xgboost():
    test_path = "data/processed/test.csv"
    model_path = "models/xgboost_financial_risk.pkl"
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    print(f"Loading test dataset from {test_path}...")
    test_df = pd.read_csv(test_path)
    feature_cols = joblib.load("data/processed/feature_names.pkl")
    model = joblib.load(model_path)

    X_test = test_df[feature_cols]
    y_test = test_df['target']

    # Predict probabilities and binary class
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Compute metrics
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, zero_division=0)
    rec = recall_score(y_test, preds, zero_division=0)
    f1 = f1_score(y_test, preds, zero_division=0)
    roc_auc = roc_auc_score(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    cm = confusion_matrix(y_test, preds).tolist()

    # Curves for visualization
    fpr, tpr, roc_thresh = roc_curve(y_test, probs)
    precision_pts, recall_pts, pr_thresh = precision_recall_curve(y_test, probs)

    # Feature Importance (top 10)
    importances = model.feature_importances_
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    top_features = [{"feature": f, "importance": float(imp)} for f, imp in feat_imp[:15]]

    metrics = {
        "model_name": "XGBoost Financial Risk Model",
        "dataset_split": "Temporal Test (Time 11-14)",
        "total_test_samples": int(len(y_test)),
        "distressed_test_samples": int(y_test.sum()),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "confusion_matrix": cm,
        "top_features": top_features,
        "roc_curve": {
            "fpr": [float(x) for x in fpr[::max(1, len(fpr)//50)]],
            "tpr": [float(x) for x in tpr[::max(1, len(tpr)//50)]]
        },
        "pr_curve": {
            "precision": [float(x) for x in precision_pts[::max(1, len(precision_pts)//50)]],
            "recall": [float(x) for x in recall_pts[::max(1, len(recall_pts)//50)]]
        }
    }

    report_path = os.path.join(report_dir, "xgboost_metrics.json")
    with open(report_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("=== XGBOOST MODEL EVALUATION ===")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"Confusion Matrix: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"Metrics report saved to {report_path}")

if __name__ == "__main__":
    evaluate_xgboost()
