import os
import joblib
import pandas as pd
import numpy as np

class FinancialRiskService:
    def __init__(self, model_path="models/xgboost_financial_risk.pkl", feature_names_path="data/processed/feature_names.pkl"):
        self.model_path = model_path
        self.feature_names_path = feature_names_path
        self.model = None
        self.feature_names = None
        self._load_artifacts()

    def _load_artifacts(self):
        if os.path.exists(self.model_path) and os.path.exists(self.feature_names_path):
            self.model = joblib.load(self.model_path)
            self.feature_names = joblib.load(self.feature_names_path)
            print("FinancialRiskService: XGBoost model & feature names loaded successfully.")
        else:
            print("FinancialRiskService Warning: Model or feature names file missing.")

    def calculate_risk_score(self, probability: float) -> float:
        """
        Converts financial distress probability [0.0, 1.0] into a 0-100 financial risk score.
        Formula: score = min(100.0, max(0.0, probability * 100.0))
        """
        score = probability * 100.0
        return round(float(np.clip(score, 0.0, 100.0)), 2)

    def categorize_risk(self, score: float) -> str:
        if score < 30.0:
            return "Low Financial Risk"
        elif score < 60.0:
            return "Medium Financial Risk"
        else:
            return "High Financial Risk"

    def predict(self, feature_values: dict) -> dict:
        if self.model is None or self.feature_names is None:
            raise ValueError("Financial risk model is not loaded.")

        # Ensure all required features are present
        input_data = []
        for feat in self.feature_names:
            val = feature_values.get(feat, 0.0)
            input_data.append(val)

        df_input = pd.DataFrame([input_data], columns=self.feature_names)
        prob = float(self.model.predict_proba(df_input)[0, 1])
        score = self.calculate_risk_score(prob)
        category = self.categorize_risk(score)

        # Calculate feature contributions using feature importances
        importances = self.model.feature_importances_
        feature_contributions = []
        for feat, val, imp in zip(self.feature_names, input_data, importances):
            if imp > 0.001:
                feature_contributions.append({
                    "feature": feat,
                    "value": float(val),
                    "importance": float(imp)
                })

        feature_contributions = sorted(feature_contributions, key=lambda x: x["importance"], reverse=True)[:5]

        return {
            "distress_probability": round(prob, 4),
            "financial_risk_score": score,
            "risk_category": category,
            "top_contributing_features": feature_contributions
        }

if __name__ == "__main__":
    service = FinancialRiskService()
    # Test sample prediction
    dummy_input = {f: 1.0 for f in service.feature_names} if service.feature_names else {}
    res = service.predict(dummy_input)
    print("Test Prediction Output:", res)
