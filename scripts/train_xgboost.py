import os
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier

def train_xgboost():
    train_path = "data/processed/train.csv"
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    print(f"Loading training data from {train_path}...")
    train_df = pd.read_csv(train_path)

    feature_cols = joblib.load("data/processed/feature_names.pkl")
    X_train = train_df[feature_cols]
    y_train = train_df['target']

    # Calculate class imbalance scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"Train samples: {len(X_train)} (Healthy: {neg_count}, Distressed: {pos_count})")
    print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

    # Train XGBoost Classifier
    model = XGBClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss"
    )

    print("Training XGBoost financial risk classifier...")
    model.fit(X_train, y_train)

    model_path = os.path.join(model_dir, "xgboost_financial_risk.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    train_xgboost()
