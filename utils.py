def find_knee_threshold(scores):
    """
    Find the knee point in a sorted list of anomaly scores using the maximum curvature method.

    Parameters:
    -----------
    scores : list of float
        Anomaly scores (not necessarily sorted)

    Returns:
    --------
    float
        Knee threshold value
    """
    # Sort scores in ascending order and keep track of original indices
    sorted_scores = sorted(scores)
    n_points = len(sorted_scores)

    # Coordinates of all points
    x = list(range(n_points))
    y = sorted_scores

    # Line between first and last point
    x1, y1 = 0, y[0]
    x2, y2 = n_points - 1, y[-1]

    # Compute distances to the line
    distances = []
    for i in range(n_points):
        numerator = abs((y2 - y1) * x[i] - (x2 - x1) * y[i] + x2 * y1 - y2 * x1)
        denominator = ((y2 - y1) ** 2 + (x2 - x1) ** 2) ** 0.5
        distance = numerator / denominator
        distances.append(distance)

    # Find index of max distance
    knee_index = distances.index(max(distances))
    knee_threshold = y[knee_index]

    return knee_threshold


def frequency_encode_df(df, categorical_cols):
    # print(df, categorical_cols)

    for col in categorical_cols:
        freq = df[col].value_counts(normalize=True)
        # print(f"{col} frequency: {freq}")
        df[col] = df[col].map(freq)
    return df
