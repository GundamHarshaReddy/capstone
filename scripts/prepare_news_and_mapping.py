import json
import os
import pandas as pd

def prepare_news():
    raw_jsonl_path = "data/news/sp500_news_raw.jsonl"
    news_csv_path = "data/news/raw_news.csv"
    mapping_csv_path = "data/news/company_mapping.csv"

    if not os.path.exists(raw_jsonl_path):
        raise FileNotFoundError(f"Missing raw jsonl news file at {raw_jsonl_path}")

    print("Loading raw S&P 500 financial news jsonl...")
    records = []
    with open(raw_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    df_raw = pd.DataFrame(records)
    print(f"Loaded {len(df_raw)} raw article records.")

    # 1. Clean & Standardize dates
    df_raw['date'] = pd.to_datetime(df_raw['Publishdate'], errors='coerce')
    df_raw = df_raw.dropna(subset=['date', 'symbol', 'Title'])
    df_raw['date_str'] = df_raw['date'].dt.strftime('%Y-%m-%d')

    # 2. Entity Resolution: Build canonical company mapping
    unique_companies = df_raw[['symbol', 'company']].drop_duplicates().sort_values('symbol').reset_index(drop=True)
    unique_companies['company_id'] = range(1, len(unique_companies) + 1)
    unique_companies['mapping_method'] = 'Ticker-Symbol Exact Match'
    unique_companies['confidence'] = 1.0

    mapping_df = unique_companies.rename(columns={
        'company': 'canonical_company_name',
        'symbol': 'source_company_symbol'
    })[['source_company_symbol', 'canonical_company_name', 'company_id', 'mapping_method', 'confidence']]

    mapping_df.to_csv(mapping_csv_path, index=False)
    print(f"Saved company entity mapping ({len(mapping_df)} entities) to {mapping_csv_path}")

    # 3. Merge company_id into raw news
    company_id_map = dict(zip(unique_companies['symbol'], unique_companies['company_id']))
    df_raw['company_id'] = df_raw['symbol'].map(company_id_map)

    news_df = pd.DataFrame({
        "company_id": df_raw['company_id'],
        "company_name": df_raw['company'],
        "symbol": df_raw['symbol'],
        "date": df_raw['date_str'],
        "headline": df_raw['Title'],
        "article_text": df_raw['Text'],
        "source": df_raw['links']
    })

    news_df = news_df.sort_values(['company_id', 'date']).reset_index(drop=True)
    news_df.to_csv(news_csv_path, index=False)
    print(f"Saved {len(news_df)} standardized news records to {news_csv_path}")

if __name__ == "__main__":
    prepare_news()
