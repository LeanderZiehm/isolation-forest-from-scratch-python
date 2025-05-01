# EXAMPLE USAGE:

```
from my import IsolationForest


if __name__ == "__main__":
    # --- DATA ---
    x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    input_columns = [x1]  # only features
    solution_column = y  # labels (not used in training)

    # Create and fit the model
    iforest = IsolationForest(num_trees=100, subsample_size=256, random_seed=0)
    iforest.fit(input_columns)

    # Score all data points
    anomaly_threshold = 0.9
    scores = iforest.score_samples()

    # Print results
    print("Anomaly scores:")
    for row_index, score in enumerate(scores):
        label = "ANOMALY" if score > anomaly_threshold else "normal"
        print(
            f"Index {row_index:2d} (value={input_columns[0][row_index]:>3}): score={score:.3f} → {label}"
        )

```
