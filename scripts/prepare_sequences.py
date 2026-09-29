import os
import pandas as pd
import numpy as np

def generate_90day_sequences():
    daily_path = "data/news/daily_sentiment_aggregated.csv"
    output_npz = "data/news/temporal_sequences.npz"

    if not os.path.exists(daily_path):
        raise FileNotFoundError(f"Missing daily sentiment file at {daily_path}")

    print(f"Loading daily sentiment data from {daily_path}...")
    df = pd.read_csv(daily_path)
    df['date'] = pd.to_datetime(df['date'])

    feature_cols = [
        'daily_sentiment_risk',
        'mean_negative',
        'mean_positive',
        'negative_news_ratio',
        'article_count',
        'news_available'
    ]

    df['news_available'] = 1.0

    symbols = df['symbol'].unique()
    print(f"Constructing temporal sequences across {len(symbols)} companies...")

    seq_len = 90
    forward_horizon = 30

    train_X, train_y, train_meta = [], [], []
    test_X, test_y, test_meta = [], [], []

    # Chronological split cutoff
    split_cutoff = pd.to_datetime("2022-01-01")

    total_valid_sequences = 0

    for sym in symbols:
        comp_df = df[df['symbol'] == sym].sort_values('date').copy()
        if len(comp_df) < 2:
            continue

        min_d = comp_df['date'].min()
        max_d = comp_df['date'].max()

        # Check span: Need at least 90 + 30 = 120 days between first and last recorded date
        span_days = (max_d - min_d).days
        if span_days < (seq_len + forward_horizon):
            continue

        # Reindex to full daily calendar
        full_idx = pd.date_range(min_d, max_d, freq='D')
        full_df = comp_df.set_index('date').reindex(full_idx).reset_index().rename(columns={'index': 'date'})

        # Impute non-news days with neutral baseline values
        full_df['daily_sentiment_risk'] = full_df['daily_sentiment_risk'].fillna(50.0)
        full_df['mean_negative'] = full_df['mean_negative'].fillna(0.0)
        full_df['mean_positive'] = full_df['mean_positive'].fillna(0.0)
        full_df['negative_news_ratio'] = full_df['negative_news_ratio'].fillna(0.0)
        full_df['article_count'] = full_df['article_count'].fillna(0.0)
        full_df['news_available'] = full_df['news_available'].fillna(0.0)

        feature_matrix = full_df[feature_cols].values
        dates = full_df['date'].values

        # Sample sequences every 15 days to reduce temporal autocorrelation while preserving diverse horizons
        step_size = 15
        for i in range(seq_len, len(full_df) - forward_horizon, step_size):
            # Lookback: [i - seq_len : i]
            seq_X = feature_matrix[i - seq_len:i]
            pred_date = pd.to_datetime(dates[i - 1])

            # Forward window: [i : i + forward_horizon]
            fwd_df = full_df.iloc[i:i + forward_horizon]
            
            # Forward risk target: Has at least 1 negative news day and average sentiment risk >= 55.0
            has_neg_news = (fwd_df['negative_news_ratio'] > 0.0).any()
            fwd_risk_mean = fwd_df[fwd_df['news_available'] == 1.0]['daily_sentiment_risk'].mean()
            if pd.isna(fwd_risk_mean):
                fwd_risk_mean = 50.0

            target_val = 1 if (has_neg_news and fwd_risk_mean >= 55.0) else 0

            meta = f"{sym}_{pred_date.strftime('%Y-%m-%d')}"

            if pred_date < split_cutoff:
                train_X.append(seq_X)
                train_y.append(target_val)
                train_meta.append(meta)
            else:
                test_X.append(seq_X)
                test_y.append(target_val)
                test_meta.append(meta)

            total_valid_sequences += 1

    train_X = np.array(train_X, dtype=np.float32)
    train_y = np.array(train_y, dtype=np.int64)
    test_X = np.array(test_X, dtype=np.float32)
    test_y = np.array(test_y, dtype=np.int64)

    print(f"Total sequences generated: {total_valid_sequences}")
    print(f"Train split (T < 2022-01-01): {len(train_X)} sequences | Positives: {train_y.sum()} ({train_y.mean():.2%})")
    print(f"Test split (T >= 2022-01-01): {len(test_X)} sequences | Positives: {test_y.sum()} ({test_y.mean():.2%})")

    np.savez_compressed(
        output_npz,
        train_X=train_X,
        train_y=train_y,
        test_X=test_X,
        test_y=test_y,
        feature_cols=np.array(feature_cols)
    )
    print(f"Saved temporal sequence tensors to {output_npz}")

if __name__ == "__main__":
    generate_90day_sequences()
