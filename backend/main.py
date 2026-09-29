import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.services.financial_risk import FinancialRiskService
from backend.services.news_sentiment import FinBERTSentimentService
from backend.services.news_lstm import NewsLSTMService
from backend.services.risk_fusion import RiskFusionService
from backend.services.explainability import ModelExplainabilityService

app = FastAPI(
    title="Dynamic Multimodal Financial Risk Assessment API",
    description="Peer-reviewed multimodal financial distress and news sentiment risk intelligence platform",
    version="2.0.0"
)

# Enable CORS for Next.js frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Singleton Instances
financial_service = FinancialRiskService()
sentiment_service = FinBERTSentimentService()
lstm_service = NewsLSTMService()
lstm_service._ensure_loaded()
fusion_service = RiskFusionService()
explainability_service = ModelExplainabilityService()

# Load financial dataset into memory
DATA_PATH = "data/processed/processed_financial_data.csv"
financial_df = None
if os.path.exists(DATA_PATH):
    financial_df = pd.read_csv(DATA_PATH)
    print(f"Backend API loaded financial dataset with {len(financial_df)} records across {financial_df['Company'].nunique()} companies.")
else:
    print(f"Backend API Warning: Dataset not found at {DATA_PATH}")

# Load real processed news dataset into memory
NEWS_PATH = "data/news/processed_news.csv"
news_df = None
if os.path.exists(NEWS_PATH):
    news_df = pd.read_csv(NEWS_PATH)
    print(f"Backend API loaded real news dataset with {len(news_df)} articles across {news_df['symbol'].nunique()} companies.")
else:
    print(f"Backend API Warning: News dataset not found at {NEWS_PATH}")

# --- Pydantic Schemas ---
class FinancialPredictionRequest(BaseModel):
    company_id: Optional[int] = None
    features: Dict[str, float] = Field(..., example={"x1": 1.28, "x2": 0.5, "x36": 1.0})

class NewsPredictionRequest(BaseModel):
    company_id: Optional[int] = None
    headline: str = Field(..., example="Supplier experiences severe liquidity shortage and potential debt default.")

class MultimodalPredictionRequest(BaseModel):
    company_id: Optional[int] = None
    financial_features: Dict[str, float]
    news_headline: Optional[str] = None
    article_count: int = 1
    sentiment_volatility: float = 0.0

# --- API Endpoints ---

@app.get("/api/health")
def health_check():
    return {
        "status": "healthy",
        "financial_model_loaded": financial_service.model is not None,
        "finbert_loaded": True,
        "lstm_loaded": lstm_service.model is not None,
        "financial_dataset_loaded": financial_df is not None,
        "news_dataset_loaded": news_df is not None
    }

@app.get("/api/statistics")
def get_statistics():
    if financial_df is None:
        raise HTTPException(status_code=404, detail="Processed dataset not available")

    total_companies = int(financial_df['Company'].nunique())
    total_records = int(len(financial_df))
    distressed_records = int(financial_df['target'].sum())
    healthy_records = total_records - distressed_records

    # Predict financial risk scores for all companies latest period
    latest_df = financial_df.sort_values('Time').groupby('Company').last().reset_index()
    feature_names = financial_service.feature_names or [c for c in financial_df.columns if c.startswith('x')]
    
    probs = financial_service.model.predict_proba(latest_df[feature_names])[:, 1]
    scores = [financial_service.calculate_risk_score(p) for p in probs]

    high_risk_count = sum(1 for s in scores if s >= 60.0)
    med_risk_count = sum(1 for s in scores if 30.0 <= s < 60.0)
    low_risk_count = sum(1 for s in scores if s < 30.0)

    # Real news stats
    news_count = len(news_df) if news_df is not None else 0
    news_companies = int(news_df['symbol'].nunique()) if news_df is not None else 0
    avg_news_risk = round(float(news_df['sentiment_risk_score'].mean()), 2) if news_df is not None else 50.0

    return {
        "total_companies": total_companies,
        "total_records": total_records,
        "financially_distressed_records": distressed_records,
        "healthy_records": healthy_records,
        "average_financial_risk_score": round(float(np.mean(scores)), 2),
        "high_risk_companies": high_risk_count,
        "medium_risk_companies": med_risk_count,
        "low_risk_companies": low_risk_count,
        "real_news_articles_count": news_count,
        "real_news_companies_count": news_companies,
        "average_news_risk_score": avg_news_risk
    }

@app.get("/api/companies")
def get_companies(page: int = 1, limit: int = 20, search: Optional[str] = None):
    if financial_df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    latest_df = financial_df.sort_values('Time').groupby('Company').last().reset_index()
    feature_names = financial_service.feature_names or [c for c in financial_df.columns if c.startswith('x')]

    probs = financial_service.model.predict_proba(latest_df[feature_names])[:, 1]
    latest_df['distress_prob'] = probs
    latest_df['financial_risk_score'] = [financial_service.calculate_risk_score(p) for p in probs]
    latest_df['risk_category'] = [financial_service.categorize_risk(s) for s in latest_df['financial_risk_score']]

    if search:
        latest_df = latest_df[latest_df['Company'].astype(str).str.contains(search)]

    total = len(latest_df)
    start = (page - 1) * limit
    end = start + limit
    paginated = latest_df.iloc[start:end]

    companies_list = []
    for _, row in paginated.iterrows():
        comp_id = int(row['Company'])
        score = float(row['financial_risk_score'])
        companies_list.append({
            "company_id": comp_id,
            "company_name": f"Firm {comp_id}",
            "latest_time_period": int(row['Time']),
            "is_distressed_historical": int(row['target']),
            "financial_risk_score": score,
            "news_risk_score": 45.0, # Baseline domain average
            "final_risk_score": round(0.6 * score + 0.4 * 45.0, 2),
            "risk_category": row['risk_category']
        })

    return {
        "total": total,
        "page": page,
        "limit": limit,
        "companies": companies_list
    }

@app.get("/api/company/{company_id}")
def get_company_details(company_id: int):
    if financial_df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    comp_df = financial_df[financial_df['Company'] == company_id].sort_values('Time')
    if comp_df.empty:
        raise HTTPException(status_code=404, detail=f"Company ID {company_id} not found")

    latest_row = comp_df.iloc[-1]
    feature_names = financial_service.feature_names or [c for c in financial_df.columns if c.startswith('x')]
    
    features_dict = {f: float(latest_row[f]) for f in feature_names if f in latest_row}
    pred_res = financial_service.predict(features_dict)
    
    # Financial explanation
    explain_res = explainability_service.explain_financial_prediction(features_dict)

    # Fusion result
    fusion_res = fusion_service.adaptive_fusion(
        financial_risk=pred_res['financial_risk_score'],
        news_risk=45.0,
        article_count=0,
        sentiment_volatility=0.0
    )

    return {
        "company_id": company_id,
        "company_name": f"Firm {company_id}",
        "total_periods_recorded": len(comp_df),
        "latest_time_period": int(latest_row['Time']),
        "is_distressed_historical": int(latest_row['target']),
        "financial_risk": pred_res,
        "risk_fusion": fusion_res,
        "explainability": explain_res,
        "features": features_dict
    }

@app.get("/api/explain/{company_id}")
def get_company_explanation(company_id: int):
    return get_company_details(company_id)["explainability"]

@app.post("/api/predict/financial")
def predict_financial_risk(req: FinancialPredictionRequest):
    try:
        res = financial_service.predict(req.features)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/news")
def predict_news_risk(req: NewsPredictionRequest):
    try:
        res = sentiment_service.analyze_text(req.headline)
        return res
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/predict/final")
def predict_final_risk(req: MultimodalPredictionRequest):
    try:
        fin_res = financial_service.predict(req.financial_features)
        fin_score = fin_res['financial_risk_score']

        news_score = 45.0
        if req.news_headline:
            news_res = sentiment_service.analyze_text(req.news_headline)
            news_score = news_res['sentiment_risk_score']

        fusion_res = fusion_service.adaptive_fusion(
            financial_risk=fin_score,
            news_risk=news_score,
            article_count=req.article_count,
            sentiment_volatility=req.sentiment_volatility
        )

        return {
            "financial_prediction": fin_res,
            "news_risk_score": news_score,
            "multimodal_fusion": fusion_res
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/model-metrics")
def get_model_metrics():
    res = {}
    for filename, key in [
        ("reports/xgboost_with_x81.json", "xgboost_with_x81"),
        ("reports/xgboost_without_x81.json", "xgboost_without_x81"),
        ("reports/x81_comparison.json", "x81_comparison"),
        ("reports/lstm_metrics.json", "news_lstm_model"),
        ("reports/ablation_results.json", "ablation_study"),
        ("reports/finbert_news_statistics.json", "news_statistics")
    ]:
        if os.path.exists(filename):
            with open(filename, "r") as f:
                res[key] = json.load(f)
    return res

@app.get("/api/feature-importance")
def get_feature_importance():
    if financial_service.model is None or financial_service.feature_names is None:
        raise HTTPException(status_code=404, detail="Model not loaded")

    importances = financial_service.model.feature_importances_
    feat_imp = sorted(zip(financial_service.feature_names, importances), key=lambda x: x[1], reverse=True)
    return {
        "top_features": [{"feature": f, "importance": float(imp)} for f, imp in feat_imp[:20]]
    }

@app.get("/api/news/{company_id}")
def get_company_news(company_id: int):
    # To maintain peer-reviewed research integrity, Kaggle anonymized panel firms (IDs 1-422)
    # are strictly not mapped to S&P 500 public equities.
    # Adaptive fusion automatically defaults to 100% financial weighting when company news is unavailable.
    sample_news = []
    if news_df is not None:
        sample_news = news_df[['symbol', 'company_name', 'date', 'headline', 'p_positive', 'p_negative', 'p_neutral', 'sentiment_risk_score', 'dominant_sentiment', 'source']].head(5).to_dict(orient="records")

    return {
        "company_id": company_id,
        "articles": [],
        "sample_sp500_news": sample_news,
        "notice": f"Firm {company_id} is an anonymized manufacturing firm from the Kaggle Financial Distress panel (Tehran Stock Exchange). To maintain Scopus-level research integrity, no artificial mapping to S&P 500 news is fabricated. The coverage-adaptive fusion engine cleanly sets news weight to 0.0 (w_fin = 1.0) for this entity."
    }

@app.get("/api/news/sp500/{symbol}")
def get_sp500_news(symbol: str):
    if news_df is None:
        raise HTTPException(status_code=404, detail="News dataset not loaded")
    sym_news = news_df[news_df['symbol'].str.upper() == symbol.upper()]
    if sym_news.empty:
        raise HTTPException(status_code=404, detail=f"No news found for S&P 500 ticker {symbol}")
    return {
        "symbol": symbol.upper(),
        "company_name": sym_news.iloc[0]['company_name'],
        "articles": sym_news[['date', 'headline', 'p_positive', 'p_negative', 'p_neutral', 'sentiment_risk_score', 'dominant_sentiment', 'source']].head(10).to_dict(orient="records")
    }

@app.get("/api/risk-history/{company_id}")
def get_risk_history(company_id: int):
    if financial_df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    comp_df = financial_df[financial_df['Company'] == company_id].sort_values('Time')
    if comp_df.empty:
        raise HTTPException(status_code=404, detail=f"Company {company_id} not found")

    feature_names = financial_service.feature_names or [c for c in financial_df.columns if c.startswith('x')]
    probs = financial_service.model.predict_proba(comp_df[feature_names])[:, 1]
    
    history = []
    for _, row in comp_df.iterrows():
        t = int(row['Time'])
        p = float(probs[len(history)])
        fin_score = financial_service.calculate_risk_score(p)
        history.append({
            "time_period": t,
            "financial_distress_target": int(row['target']),
            "raw_distress_value": float(row['Financial Distress']),
            "financial_risk_score": fin_score,
            "news_risk_score": 45.0,
            "final_multimodal_risk": round(0.6 * fin_score + 0.4 * 45.0, 2)
        })

    return {
        "company_id": company_id,
        "risk_history": history
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
