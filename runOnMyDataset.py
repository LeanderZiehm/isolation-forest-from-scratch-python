from isolation_forest import IsolationForest
from utils import find_knee_threshold

# Example usage (if this file is run directly)
if __name__ == "__main__":
    # --- DATA ---
    x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    input_columns = [x1]  # only features
    solution_column = y  # labels (not used in training)

    # Create and fit the model
    iforest = IsolationForest(num_trees=100, subsample_size=256, random_seed=0)
    iforest.fit(input_columns)
    scores = iforest.score_samples()
    automatic_anomaly_threshold = find_knee_threshold(scores)
    print(
        f"Automatic anomaly threshold (knee point): {automatic_anomaly_threshold:.3f}"
    )
    # Print results
    print("Anomaly scores:")
    for row_index, score in enumerate(scores):
        label = "ANOMALY" if score > automatic_anomaly_threshold else "normal"
        print(
            f"Index {row_index:2d} (value={input_columns[0][row_index]:>3}): score={score:.3f} → {label}"
        )
