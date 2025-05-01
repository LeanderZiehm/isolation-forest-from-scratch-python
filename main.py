#!/usr/bin/env python3
# custom_iforest_metaverse.py

import pandas as pd
from isolation_forest import IsolationForest


def load_data(path: str) -> pd.DataFrame:
    """Load the transactions dataset from CSV."""
    return pd.read_csv(path)


def prepare_input_columns(df: pd.DataFrame, feature_cols: list) -> list:
    """
    Transpose the DataFrame into the format expected by the custom IsolationForest:
    a list of feature‐columns, each itself a list of values.
    """
    transposed = [df[col].tolist() for col in feature_cols]

    return transposed


def map_ground_truth(series: pd.Series) -> list:
    """
    Map the original 'anomaly' column into binary labels:
      'low_risk'  → 0 (normal)
      otherwise   → 1 (anomaly)
    """
    return [0 if v == "low_risk" else 1 for v in series]


def main():
    # --- DATA LOADING ---
    df = load_data(
        "datasetExamples/MetaverseFinancial/metaverse_transactions_dataset.csv"
    )

    # --- FEATURE SELECTION ---
    # Drop non‐numeric or non‐predictive columns:
    # drop_cols = ["timestamp", "sending_address", "receiving_address", "anomaly"]
    # feature_cols = [c for c in df.columns if c not in drop_cols]

    # drop all text columns
    # transposed.remove([col for col in transposed if df[col].dtype == "object"])

    # all numeric columns
    feature_cols = df.select_dtypes(include=["number"]).columns.tolist()

    # Prepare input for custom IsolationForest
    X_cols = prepare_input_columns(df, feature_cols)

    # --- MODEL TRAINING ---
    # Instantiate with 100 trees, subsample size 256, fixed seed
    iforest = IsolationForest(num_trees=100, subsample_size=256, random_seed=0)
    iforest.fit(X_cols)

    # --- SCORING ---
    # Returns one anomaly score per row; higher = more anomalous
    scores = iforest.score_samples()

    # --- THRESHOLDING & OUTPUT ---
    anomaly_threshold = 0.9

    # (Optional) prepare the true labels for evaluation
    y_true = map_ground_truth(df["anomaly"])

    print("Index | Score   | Detected  | True Label")
    print("----------------------------------------")
    for idx, score in enumerate(scores):
        detected = "ANOMALY" if score > anomaly_threshold else "normal"
        true_lbl = "anomaly" if y_true[idx] == 1 else "normal"
        print(f"{idx:5d} | {score:7.3f} | {detected:8s} | {true_lbl}")


if __name__ == "__main__":
    main()
