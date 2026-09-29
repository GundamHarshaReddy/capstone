import numpy as np

class RiskFusionService:
    def __init__(self):
        pass

    def baseline_fusion(self, financial_risk: float, news_risk: float) -> dict:
        """
        Baseline Risk Fusion: Fixed 60/40 weighted average.
        Formula: Final Risk = 0.60 * Financial Risk + 0.40 * News Risk
        """
        f_risk = float(financial_risk)
        n_risk = float(news_risk)
        final_score = 0.60 * f_risk + 0.40 * n_risk
        final_score = round(float(np.clip(final_score, 0.0, 100.0)), 2)

        return {
            "fusion_type": "Baseline Fixed Weighted Fusion",
            "weights": {"financial_weight": 0.60, "news_weight": 0.40},
            "financial_risk_score": f_risk,
            "news_risk_score": n_risk,
            "final_multimodal_risk_score": final_score,
            "risk_category": self._categorize_final_risk(final_score)
        }

    def adaptive_fusion(
        self,
        financial_risk: float,
        news_risk: float,
        article_count: int = 0,
        sentiment_volatility: float = 0.0,
        financial_confidence: float = 0.90
    ) -> dict:
        """
        Data-Dependent Adaptive Risk Fusion:
        Dynamically adjusts news weight based on news availability (article_count) and sentiment volatility.

        Mathematical Formulation:
        1. Availability factor: alpha = min(1.0, article_count / 10.0)
        2. Dynamic news weight: w_news = np.clip(0.40 * alpha * (1.0 + 0.2 * sentiment_volatility), 0.0, 0.70)
        3. Dynamic financial weight: w_fin = 1.0 - w_news
        4. Final Multimodal Risk = w_fin * Financial Risk + w_news * News Risk
        """
        f_risk = float(financial_risk)
        n_risk = float(news_risk)

        alpha = min(1.0, article_count / 10.0) if article_count > 0 else 0.0
        w_news = float(np.clip(0.40 * alpha * (1.0 + 0.2 * float(sentiment_volatility)), 0.0, 0.70))
        w_fin = 1.0 - w_news

        final_score = w_fin * f_risk + w_news * n_risk
        final_score = round(float(np.clip(final_score, 0.0, 100.0)), 2)

        return {
            "fusion_type": "Rule-Based Coverage-Adaptive Fusion",
            "weights": {"financial_weight": round(w_fin, 4), "news_weight": round(w_news, 4)},
            "availability_factor_alpha": round(alpha, 2),
            "financial_risk_score": f_risk,
            "news_risk_score": n_risk,
            "final_multimodal_risk_score": final_score,
            "risk_category": self._categorize_final_risk(final_score)
        }

    def _categorize_final_risk(self, score: float) -> str:
        if score < 30.0:
            return "Low Risk"
        elif score < 60.0:
            return "Medium Risk"
        else:
            return "High Risk"

if __name__ == "__main__":
    fusion = RiskFusionService()
    base = fusion.baseline_fusion(75.0, 40.0)
    adapt = fusion.adaptive_fusion(75.0, 40.0, article_count=5, sentiment_volatility=1.5)
    print("Baseline Fusion Output:", base)
    print("Adaptive Fusion Output:", adapt)
