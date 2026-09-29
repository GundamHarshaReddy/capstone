import os
import joblib
import pandas as pd
import numpy as np

class ModelExplainabilityService:
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

    def explain_financial_prediction(self, feature_values: dict) -> dict:
        """
        Provides non-causal feature contribution analysis for the XGBoost model predictions.
        Identifies key features contributing towards elevated or reduced risk.
        """
        if self.model is None or self.feature_names is None:
            return {"status": "Model not available for explainability"}

        importances = self.model.feature_importances_
        contributions = []

        for feat, imp in zip(self.feature_names, importances):
            if imp > 0.005:
                val = float(feature_values.get(feat, 0.0))
                contributions.append({
                    "feature_name": feat,
                    "importance_score": float(imp),
                    "feature_value": val,
                    "impact_direction": "elevating_risk" if val > 0 else "mitigating_risk"
                })

        contributions = sorted(contributions, key=lambda x: x["importance_score"], reverse=True)

        top_positive = [c for c in contributions if c["impact_direction"] == "elevating_risk"][:5]
        top_mitigating = [c for c in contributions if c["impact_direction"] == "mitigating_risk"][:5]

        return {
            "disclaimer": "These features contributed to the model prediction according to XGBoost tree gain scores. This analysis indicates correlation, not direct causality.",
            "top_risk_contributors": top_positive,
            "top_mitigating_factors": top_mitigating,
            "all_important_features": contributions[:10]
        }

    def explain_news_sentiment(self, article_sentiments: list) -> dict:
        """
        Explains news risk factors based on article volume, recent negative sentiment, and volatility.
        """
        if not article_sentiments:
            return {
                "article_count": 0,
                "sentiment_trend": "No recent news articles recorded",
                "sentiment_volatility": 0.0,
                "recent_negative_headlines": []
            }

        scores = [a.get("sentiment_risk_score", 50.0) for a in article_sentiments]
        negs = [a for a in article_sentiments if a.get("dominant_sentiment") == "negative"]

        volatility = float(np.std(scores)) if len(scores) > 1 else 0.0
        avg_score = float(np.mean(scores))

        trend = "Stable"
        if len(scores) >= 2:
            diff = scores[-1] - scores[0]
            if diff > 10:
                trend = "Deteriorating Sentiment (Increasing Risk)"
            elif diff < -10:
                trend = "Improving Sentiment (Decreasing Risk)"

        return {
            "article_count": len(article_sentiments),
            "average_sentiment_risk": round(avg_score, 2),
            "sentiment_trend": trend,
            "sentiment_volatility": round(volatility, 2),
            "recent_negative_headlines": [a.get("headline", "") for a in negs[:5]]
        }

if __name__ == "__main__":
    exp = ModelExplainabilityService()
    res = exp.explain_financial_prediction({"x36": 2.5, "x46": -1.0, "x12": 0.5})
    print("Explainability Output:", res)
