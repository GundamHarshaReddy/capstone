import os
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def preprocess_financial_data():
    raw_path = "data/raw/Financial Distress.csv"
    output_dir = "data/processed"
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading raw data from {raw_path}...")
    df = pd.read_csv(raw_path)

    # 1. Clean duplicate rows
    initial_len = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_len - len(df)} duplicate rows.")

    # 2. Infinite value handling (replace inf with NaN without filling yet)
    df = df.replace([np.inf, -np.inf], np.nan)

    # 3. Target creation
    # Kaggle standard definition: Financial Distress <= -0.5 is distress (1), > -0.5 is normal (0)
    df['target'] = (df['Financial Distress'] <= -0.5).astype(int)
    print(f"Target distribution in full dataset: Distressed(1)={df['target'].sum()}, Healthy(0)={(df['target']==0).sum()}")

    # 4. Feature identification
    id_cols = ['Company', 'Time']
    target_cols = ['target', 'Financial Distress']
    all_feature_cols = [c for c in df.columns if c not in id_cols + target_cols]
    feature_cols_no_x81 = [c for c in all_feature_cols if c != 'x81']

    print(f"Total features: {len(all_feature_cols)} | Features excluding x81: {len(feature_cols_no_x81)}")

    # 5. Temporal Splits to prevent data leakage:
    # Full Training: Time 1-10
    # Validation Splitting for Threshold Tuning:
    #   Sub-Training: Time 1-8
    #   Internal Validation: Time 9-10
    # Final Held-Out Test: Time 11-14
    train_full_df = df[df['Time'] <= 10].copy()
    train_sub_df = df[df['Time'] <= 8].copy()
    val_df = df[(df['Time'] >= 9) & (df['Time'] <= 10)].copy()
    test_df = df[df['Time'] > 10].copy()

    print(f"Train Full split (Time 1-10): {len(train_full_df)} rows, {train_full_df['target'].sum()} distress")
    print(f"Train Sub split (Time 1-8):   {len(train_sub_df)} rows, {train_sub_df['target'].sum()} distress")
    print(f"Validation split (Time 9-10): {len(val_df)} rows, {val_df['target'].sum()} distress")
    print(f"Final Test split (Time 11-14): {len(test_df)} rows, {test_df['target'].sum()} distress")

    # 6. Learned Imputation (fitted ONLY on training splits)
    median_full = train_full_df[all_feature_cols].median()
    train_full_df[all_feature_cols] = train_full_df[all_feature_cols].fillna(median_full)
    test_df[all_feature_cols] = test_df[all_feature_cols].fillna(median_full)

    median_sub = train_sub_df[all_feature_cols].median()
    train_sub_df[all_feature_cols] = train_sub_df[all_feature_cols].fillna(median_sub)
    val_df[all_feature_cols] = val_df[all_feature_cols].fillna(median_sub)

    # 7. Normalization / Scaler (fitted ONLY on training splits)
    scaler_full = StandardScaler().fit(train_full_df[all_feature_cols])
    scaler_sub = StandardScaler().fit(train_sub_df[all_feature_cols])
    scaler_no_x81 = StandardScaler().fit(train_full_df[feature_cols_no_x81])

    # 8. Save all processed datasets and artifacts
    train_full_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    train_sub_df.to_csv(os.path.join(output_dir, "train_sub.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "processed_financial_data.csv"), index=False)

    joblib.dump(scaler_full, os.path.join(output_dir, "scaler.pkl"))
    joblib.dump(scaler_sub, os.path.join(output_dir, "scaler_sub.pkl"))
    joblib.dump(scaler_no_x81, os.path.join(output_dir, "scaler_no_x81.pkl"))
    joblib.dump(all_feature_cols, os.path.join(output_dir, "feature_names.pkl"))
    joblib.dump(feature_cols_no_x81, os.path.join(output_dir, "feature_names_no_x81.pkl"))

    print("Rigorous scientific preprocessing completed successfully. All artifacts saved.")

if __name__ == "__main__":
    preprocess_financial_data()
