import numpy as np
import matplotlib.pyplot as plt
from kneed import KneeLocator
from utils import find_knee_threshold

# --- assume you already have scores_test from pipeline.decision_function(X_test) ---

# 1) Convert to an “anomaly score” so that higher = more anomalous
#    (decision_function: high = normal, low = anomalous)
anom_scores = -scores_test

# 2) Sort them
idx = np.argsort(anom_scores)
sorted_scores = anom_scores[idx]
x = np.arange(len(sorted_scores))

# 3) Find the elbow (knee) point
kneedle = KneeLocator(
    x,
    sorted_scores,
    curve="convex",  # the curve is convex (upwards)
    direction="increasing",
)
knee_idx = kneedle.knee  # index in the sorted array
knee_score = sorted_scores[knee_idx]
myKneeScore = find_knee_threshold(anom_scores)


print(f"Detected knee at index {knee_idx}, score threshold = {knee_score:.4f}")
print(f"Custom knee score threshold = {myKneeScore:.4f}")

# 4) Plot
plt.figure(figsize=(8, 5))
plt.plot(x, sorted_scores, label="Sorted anomaly score")

# Plot knee found by KneeLocator
plt.vlines(
    knee_idx,
    ymin=sorted_scores.min(),
    ymax=sorted_scores.max(),
    linestyles="--",
    colors="red",
    label=f"KneeLocator @ {knee_score:.4f}",
)
plt.scatter([knee_idx], [knee_score], color="red")

# Plot custom knee threshold
plt.hlines(
    myKneeScore,
    xmin=0,
    xmax=len(sorted_scores) - 1,
    linestyles="--",
    colors="green",
    label=f"Custom Threshold @ {myKneeScore:.4f}",
)

plt.xlabel("Sample rank")
plt.ylabel("Anomaly score (-decision_function)")
plt.title("Elbow Method for ISO Forest Threshold")
plt.legend()
plt.tight_layout()
plt.show()
