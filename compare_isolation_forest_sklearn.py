import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest as SKL_IF
from isolation_forest import IsolationForest  # your implementation

# --- config ---
IMG_DIR = "imgs"
os.makedirs(IMG_DIR, exist_ok=True)
N_SEEDS = 100


def generate_data():
    X = np.concatenate([np.linspace(0, 10, 100), [50, 60, 70]])[:, None]
    return X


def compute_scores(X, seed=0):
    # our IF
    input_columns = [X.flatten().tolist()]
    our_if = IsolationForest(num_trees=100, subsample_size=256, random_seed=seed)
    our_if.fit(input_columns)
    ours = np.array(our_if.score_samples())

    # sklearn IF (negate decision_function so larger ⇒ more anomalous)
    skl_if = SKL_IF(
        n_estimators=100, max_samples=256, contamination="auto", random_state=seed
    )
    skl_if.fit(X)
    skl = -skl_if.decision_function(X)

    return ours, skl


def normalize(v):
    """Min–max to [0,1]."""
    mn, mx = v.min(), v.max()
    return (v - mn) / (mx - mn) if mx > mn else np.zeros_like(v)


def aggregate_scores(X):
    """Run through seeds and build normalized‐scores arrays of shape (N_SEEDS, N_pts)."""
    n_pts = X.shape[0]
    ours_all = np.zeros((N_SEEDS, n_pts))
    skl_all = np.zeros((N_SEEDS, n_pts))

    for i, seed in enumerate(range(N_SEEDS)):
        ours, skl = compute_scores(X, seed)
        ours_all[i] = normalize(ours)
        skl_all[i] = normalize(skl)

    return ours_all, skl_all


def plot_intervals(ours_all, skl_all, save_path):
    """Plot mean±std intervals for both methods."""
    idx = np.arange(ours_all.shape[1])

    # Compute statistics
    ours_mu = ours_all.mean(axis=0)
    ours_std = ours_all.std(axis=0)
    skl_mu = skl_all.mean(axis=0)
    skl_std = skl_all.std(axis=0)

    fig, ax = plt.subplots(figsize=(12, 5))

    # Ours: mean line + shaded ±1σ
    ax.plot(idx, ours_mu, color="C0", label="ours (mean)")
    ax.fill_between(
        idx,
        ours_mu - ours_std,
        ours_mu + ours_std,
        color="C0",
        alpha=0.3,
        label="ours (±1σ)",
    )

    # sklearn: same
    ax.plot(idx, skl_mu, color="C1", label="sklearn (mean)")
    ax.fill_between(
        idx,
        skl_mu - skl_std,
        skl_mu + skl_std,
        color="C1",
        alpha=0.3,
        label="sklearn (±1σ)",
    )

    ax.set_title(
        "Anomaly‐score variability over 100 seeds\nshaded = ±1 standard deviation"
    )
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Normalized anomaly score")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Interval plot saved to {save_path}")


if __name__ == "__main__":
    X = generate_data()
    ours_all, skl_all = aggregate_scores(X)
    plot_intervals(ours_all, skl_all, os.path.join(IMG_DIR, "intervals.png"))
