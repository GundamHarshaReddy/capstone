import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

class FinBERTSentimentService:
    def __init__(self, model_name="ProsusAI/finbert"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._is_loaded = False

    def _ensure_loaded(self):
        if self._is_loaded:
            return
        try:
            import torch
            torch.set_num_threads(1)
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            print(f"Lazy loading FinBERT model ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self.model.eval()
            self._is_loaded = True
            print("FinBERT model loaded successfully.")
        except Exception as e:
            print(f"Error lazy loading FinBERT model: {e}")
            self.tokenizer = None
            self.model = None
            self._is_loaded = False

    def analyze_text(self, text: str) -> dict:
        if not text or not isinstance(text, str):
            return {
                "positive": 0.0,
                "negative": 0.0,
                "neutral": 1.0,
                "sentiment_risk_score": 30.0,
                "dominant_sentiment": "neutral"
            }

        self._ensure_loaded()

        if self.model is None or self.tokenizer is None:
            return self._heuristic_fallback(text)

        import torch
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0].tolist()

        p_pos, p_neg, p_neu = probs[0], probs[1], probs[2]
        net_sentiment = p_pos - p_neg
        risk_score = float(np.clip((1.0 - net_sentiment) / 2.0 * 100.0, 0.0, 100.0))

        labels = ["positive", "negative", "neutral"]
        dominant = labels[int(np.argmax(probs))]

        return {
            "positive": round(p_pos, 4),
            "negative": round(p_neg, 4),
            "neutral": round(p_neu, 4),
            "net_sentiment": round(net_sentiment, 4),
            "sentiment_risk_score": round(risk_score, 2),
            "dominant_sentiment": dominant
        }

    def _heuristic_fallback(self, text: str) -> dict:
        lower = text.lower()
        neg_words = ["distress", "bankrupt", "loss", "decline", "default", "lawsuit", "debt", "crisis"]
        pos_words = ["growth", "profit", "surge", "gain", "revenue", "expansion", "success"]

        neg_count = sum(1 for w in neg_words if w in lower)
        pos_count = sum(1 for w in pos_words if w in lower)

        if neg_count > pos_count:
            return {"positive": 0.1, "negative": 0.8, "neutral": 0.1, "sentiment_risk_score": 85.0, "dominant_sentiment": "negative"}
        elif pos_count > neg_count:
            return {"positive": 0.8, "negative": 0.1, "neutral": 0.1, "sentiment_risk_score": 15.0, "dominant_sentiment": "positive"}
        else:
            return {"positive": 0.2, "negative": 0.2, "neutral": 0.6, "sentiment_risk_score": 50.0, "dominant_sentiment": "neutral"}
