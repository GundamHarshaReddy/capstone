import unittest
os_env = __import__("os")
os_env.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath("."))
from backend.services.financial_risk import FinancialRiskService
from backend.services.news_sentiment import FinBERTSentimentService
from backend.services.news_lstm import NewsLSTMService
from backend.services.risk_fusion import RiskFusionService
from backend.services.explainability import ModelExplainabilityService

class TestRiskAssessmentPipeline(unittest.TestCase):

    def setUp(self):
        self.financial_service = FinancialRiskService()
        self.sentiment_service = FinBERTSentimentService()
        self.lstm_service = NewsLSTMService()
        self.fusion_service = RiskFusionService()
        self.explain_service = ModelExplainabilityService()

    def test_01_dataset_integrity(self):
        raw_path = "data/raw/Financial Distress.csv"
        processed_path = "data/processed/processed_financial_data.csv"
        train_path = "data/processed/train.csv"
        test_path = "data/processed/test.csv"
        news_raw_path = "data/news/raw_news.csv"
        news_proc_path = "data/news/processed_news.csv"

        self.assertTrue(os.path.exists(raw_path), "Raw Financial Distress CSV missing")
        self.assertTrue(os.path.exists(processed_path), "Processed dataset CSV missing")
        self.assertTrue(os.path.exists(train_path), "Train CSV missing")
        self.assertTrue(os.path.exists(test_path), "Test CSV missing")
        self.assertTrue(os.path.exists(news_raw_path), "Raw News CSV missing")
        self.assertTrue(os.path.exists(news_proc_path), "Processed News CSV missing")

        df = pd.read_csv(processed_path)
        self.assertEqual(len(df), 3672)
        self.assertIn("target", df.columns)
        self.assertIn("Company", df.columns)
        self.assertIn("Time", df.columns)

        news_df = pd.read_csv(news_proc_path)
        self.assertGreater(len(news_df), 4000)
        self.assertIn("p_positive", news_df.columns)
        self.assertIn("p_negative", news_df.columns)
        self.assertIn("p_neutral", news_df.columns)

    def test_02_xgboost_financial_prediction_and_missing_features(self):
        self.assertIsNotNone(self.financial_service.model, "XGBoost model failed to load")
        self.assertIsNotNone(self.financial_service.feature_names, "Feature names missing")

        # Standard sample prediction
        sample_features = {f: 1.0 for f in self.financial_service.feature_names}
        res = self.financial_service.predict(sample_features)
        self.assertIn("distress_probability", res)
        self.assertIn("financial_risk_score", res)
        self.assertTrue(0.0 <= res["financial_risk_score"] <= 100.0)

        # Edge case: completely empty dictionary (should default missing features gracefully)
        empty_res = self.financial_service.predict({})
        self.assertIn("financial_risk_score", empty_res)
        self.assertTrue(0.0 <= empty_res["financial_risk_score"] <= 100.0)

    def test_03_finbert_sentiment_and_edge_cases(self):
        # Normal positive text
        pos_res = self.sentiment_service.analyze_text("Company announces major revenue growth and record profits.")
        self.assertIn("positive", pos_res)
        self.assertGreater(pos_res["positive"], 0.5)

        # Extreme distress text
        neg_res = self.sentiment_service.analyze_text("Company files for bankruptcy following catastrophic default and liquidity crisis.")
        self.assertGreater(neg_res["negative"], 0.5)
        self.assertGreater(neg_res["sentiment_risk_score"], 60.0)

        # Edge case: empty string
        empty_res = self.sentiment_service.analyze_text("")
        self.assertEqual(empty_res["neutral"], 1.0)
        self.assertEqual(empty_res["sentiment_risk_score"], 30.0)

        # Edge case: None input
        none_res = self.sentiment_service.analyze_text(None)
        self.assertEqual(none_res["dominant_sentiment"], "neutral")

    def test_04_pytorch_lstm_sequence_inference_and_shapes(self):
        # 90-day sequence with 6 features
        sample_sequence = np.random.randn(90, 6).astype(np.float32)
        res = self.lstm_service.predict_sequence(sample_sequence)
        self.assertIn("news_risk_score", res)
        self.assertEqual(res["temporal_window_days"], 90)
        self.assertTrue(0.0 <= res["news_risk_score"] <= 100.0)

        # Edge case: 5 features provided (auto-padded to 6)
        sample_5f = np.random.randn(90, 5).astype(np.float32)
        res_5f = self.lstm_service.predict_sequence(sample_5f)
        self.assertIn("news_risk_score", res_5f)
        self.assertTrue(0.0 <= res_5f["news_risk_score"] <= 100.0)

    def test_05_risk_fusion_algorithms_and_edge_cases(self):
        # Baseline fixed fusion: 0.60 * 70 + 0.40 * 40 = 58.0
        base = self.fusion_service.baseline_fusion(70.0, 40.0)
        self.assertEqual(base["final_multimodal_risk_score"], 58.0)

        # Edge case: Company with NO news (article_count = 0)
        # Should dynamically assign 100% weight to financial risk
        adapt_zero = self.fusion_service.adaptive_fusion(75.0, 30.0, article_count=0, sentiment_volatility=0.0)
        self.assertEqual(adapt_zero["weights"]["financial_weight"], 1.0)
        self.assertEqual(adapt_zero["weights"]["news_weight"], 0.0)
        self.assertEqual(adapt_zero["final_multimodal_risk_score"], 75.0)

        # Normal adaptive case: 10 articles, high volatility
        adapt_active = self.fusion_service.adaptive_fusion(75.0, 30.0, article_count=10, sentiment_volatility=1.5)
        self.assertLess(adapt_active["weights"]["financial_weight"], 1.0)
        self.assertGreater(adapt_active["weights"]["news_weight"], 0.0)

    def test_06_explainability_non_causal_contract(self):
        sample_features = {f: 1.0 for f in (self.financial_service.feature_names or [])}
        res = self.explain_service.explain_financial_prediction(sample_features)
        self.assertIn("disclaimer", res)
        # Verify non-causal wording
        self.assertIn("correlation, not direct causality", res["disclaimer"])
        self.assertIn("contributed to the model prediction", res["disclaimer"])

        # Edge case: Empty news history explanation
        news_exp = self.explain_service.explain_news_sentiment([])
        self.assertEqual(news_exp["article_count"], 0)

if __name__ == "__main__":
    unittest.main()
