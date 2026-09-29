import os
import pandas as pd

def process_news():
    news_dir = "data/news"
    os.makedirs(news_dir, exist_ok=True)
    raw_news_path = os.path.join(news_dir, "raw_news.csv")
    processed_news_path = os.path.join(news_dir, "processed_news.csv")
    template_path = os.path.join(news_dir, "sample_news_schema.csv")

    # Create a schema template file for reference
    template_df = pd.DataFrame(columns=["company_id", "date", "headline", "article_text", "source"])
    template_df.to_csv(template_path, index=False)
    print(f"Created news schema template at {template_path}")

    if not os.path.exists(raw_news_path):
        print(f"\n[INFO] Raw news dataset not found at {raw_news_path}.")
        print("The pipeline is configured for real news data ingestion.")
        print("To run FinBERT sentiment on news, place your real news CSV at 'data/news/raw_news.csv' with columns:")
        print("['company_id', 'date', 'headline', 'article_text', 'source']")
        return

    print(f"Processing raw news data from {raw_news_path}...")
    df = pd.read_csv(raw_path)

    # Validate required columns
    required_cols = {"company_id", "date", "headline"}
    if not required_cols.issubset(set(df.columns)):
        raise ValueError(f"Missing required columns in raw news data. Must contain: {required_cols}")

    # Standardize & Clean
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['company_id', 'date', 'headline'])
    df['text_content'] = df['headline'].astype(str) + " " + df.get('article_text', '').fillna('').astype(str)
    df['text_content'] = df['text_content'].str.strip()

    df.to_csv(processed_news_path, index=False)
    print(f"Successfully processed {len(df)} news records into {processed_news_path}")

if __name__ == "__main__":
    process_news()
