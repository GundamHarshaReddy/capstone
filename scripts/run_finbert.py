import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def run_finbert_pipeline(batch_size=64):
    raw_news_path = "data/news/raw_news.csv"
    processed_news_path = "data/news/processed_news.csv"
    daily_agg_path = "data/news/daily_sentiment_aggregated.csv"
    stats_path = "reports/finbert_news_statistics.json"

    if not os.path.exists(raw_news_path):
        raise FileNotFoundError(f"Missing raw news at {raw_news_path}")

    print(f"Loading raw news dataset from {raw_news_path}...")
    df = pd.read_csv(raw_news_path)
    print(f"Loaded {len(df)} news articles across {df['symbol'].nunique()} companies.")

    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Initializing FinBERT (ProsusAI/finbert) on device: {device}...")
    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
    model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert').to(device)
    model.eval()

    # Prepare text: headline is the primary signal
    headlines = df['headline'].fillna('').astype(str).tolist()
    total_articles = len(headlines)

    print(f"Running batched FinBERT sentiment classification for {total_articles} articles (batch size {batch_size})...")
    start_time = time.time()

    all_pos, all_neg, all_neu = [], [], []

    for i in range(0, total_articles, batch_size):
        batch_texts = headlines[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

        # FinBERT labels order: [positive, negative, neutral]
        all_pos.extend(probs[:, 0].tolist())
        all_neg.extend(probs[:, 1].tolist())
        all_neu.extend(probs[:, 2].tolist())

        if (i // batch_size) % 15 == 0 or (i + batch_size) >= total_articles:
            elapsed = time.time() - start_time
            rate = (i + len(batch_texts)) / max(elapsed, 0.001)
            print(f"Processed {i + len(batch_texts)}/{total_articles} articles ({rate:.1f} articles/s)...")

    df['p_positive'] = np.round(all_pos, 4)
    df['p_negative'] = np.round(all_neg, 4)
    df['p_neutral'] = np.round(all_neu, 4)

    # Net sentiment [-1.0, 1.0] and Sentiment Risk Score [0.0, 100.0]
    df['net_sentiment'] = np.round(df['p_positive'] - df['p_negative'], 4)
    df['sentiment_risk_score'] = np.round(np.clip((1.0 - df['net_sentiment']) / 2.0 * 100.0, 0.0, 100.0), 2)

    dominant = []
    for pos, neg, neu in zip(all_pos, all_neg, all_neu):
        if pos >= neg and pos >= neu:
            dominant.append('positive')
        elif neg >= pos and neg >= neu:
            dominant.append('negative')
        else:
            dominant.append('neutral')
    df['dominant_sentiment'] = dominant

    df.to_csv(processed_news_path, index=False)
    print(f"Saved article-level processed sentiments to {processed_news_path}")

    # Daily aggregation per company
    print("Aggregating daily sentiment features...")
    daily_df = df.groupby(['company_id', 'company_name', 'symbol', 'date']).agg(
        article_count=('headline', 'count'),
        mean_positive=('p_positive', 'mean'),
        mean_negative=('p_negative', 'mean'),
        mean_neutral=('p_neutral', 'mean'),
        daily_sentiment_risk=('sentiment_risk_score', 'mean'),
        sentiment_volatility=('sentiment_risk_score', lambda x: float(x.std()) if len(x) > 1 else 0.0),
        negative_news_ratio=('dominant_sentiment', lambda x: float((x == 'negative').mean()))
    ).reset_index()

    daily_df['daily_sentiment_risk'] = np.round(daily_df['daily_sentiment_risk'], 2)
    daily_df['sentiment_volatility'] = np.round(daily_df['sentiment_volatility'], 2)
    daily_df['mean_positive'] = np.round(daily_df['mean_positive'], 4)
    daily_df['mean_negative'] = np.round(daily_df['mean_negative'], 4)
    daily_df['mean_neutral'] = np.round(daily_df['mean_neutral'], 4)
    daily_df['negative_news_ratio'] = np.round(daily_df['negative_news_ratio'], 4)

    daily_df.to_csv(daily_agg_path, index=False)
    print(f"Saved {len(daily_df)} daily aggregated sentiment records to {daily_agg_path}")

    # Compute comprehensive statistical summary
    stats = {
        "dataset_name": "S&P 500 Real Financial News Time Series",
        "total_articles_processed": int(len(df)),
        "unique_companies": int(df['symbol'].nunique()),
        "date_range": {
            "start": str(df['date'].min()),
            "end": str(df['date'].max())
        },
        "average_articles_per_company": round(float(len(df) / df['symbol'].nunique()), 2),
        "sentiment_distribution": {
            "positive_articles": int((df['dominant_sentiment'] == 'positive').sum()),
            "negative_articles": int((df['dominant_sentiment'] == 'negative').sum()),
            "neutral_articles": int((df['dominant_sentiment'] == 'neutral').sum()),
            "positive_ratio": round(float((df['dominant_sentiment'] == 'positive').mean()), 4),
            "negative_ratio": round(float((df['dominant_sentiment'] == 'negative').mean()), 4),
            "neutral_ratio": round(float((df['dominant_sentiment'] == 'neutral').mean()), 4)
        },
        "mean_sentiment_risk_score": round(float(df['sentiment_risk_score'].mean()), 2),
        "std_sentiment_risk_score": round(float(df['sentiment_risk_score'].std()), 2)
    }

    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"FinBERT news processing complete! Statistics saved to {stats_path}")
    print(f"Positive: {stats['sentiment_distribution']['positive_articles']} | Negative: {stats['sentiment_distribution']['negative_articles']} | Neutral: {stats['sentiment_distribution']['neutral_articles']}")

if __name__ == "__main__":
    run_finbert_pipeline()
