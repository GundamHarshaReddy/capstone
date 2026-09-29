import os
import pandas as pd
import numpy as np

def detailed_inspection():
    csv_path = "data/raw/Financial Distress.csv"
    df = pd.read_csv(csv_path)

    # Binarize target
    df['target'] = (df['Financial Distress'] <= -0.5).astype(int)

    # Check lag relationship for x81
    print("=== CHECKING LAGGED TARGET FEATURE (x81) ===")
    sample_co1 = df[df['Company'] == 1][['Time', 'Financial Distress', 'x81']]
    print(sample_co1)

    # Check categorical features (x80, x82, x83)
    print("\n=== DISCRETE / CATEGORICAL FEATURES ===")
    for col in ['x80', 'x82', 'x83']:
        print(f"{col}: {df[col].nunique()} unique values, min={df[col].min()}, max={df[col].max()}")

    # Check correlation of features with target
    corrs = df.drop(columns=['Company', 'Time']).corr()['Financial Distress'].abs().sort_values(ascending=False)
    print("\n=== TOP CORRELATED FEATURES WITH FINANCIAL DISTRESS ===")
    print(corrs.head(10))

    # Check time periods per company
    time_counts = df.groupby('Company')['Time'].count()
    print("\n=== TIME PERIODS PER COMPANY ===")
    print(time_counts.describe())

if __name__ == "__main__":
    detailed_inspection()
