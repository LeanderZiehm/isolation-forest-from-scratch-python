import os
import matplotlib.pyplot as plt
import imageio
from isolation_forest import IsolationForest

# -------------------------------
# 1. INPUT DATA
# -------------------------------
x0 = [1, 2, 3, 4, 10]
y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
input_columns = [x0]
solution_column = y  # Not used in training

# -------------------------------
# 2. FIT ISOLATION FOREST
# -------------------------------
iforest = IsolationForest(num_trees=1, subsample_size=256, random_seed=0)
iforest.fit(input_columns)
one_tree = iforest.forest[0]


# -------------------------------
# 3. EXTRACT SPLIT POINTS
# -------------------------------
def extract_splits(tree, splits=None):
    if splits is None:
        splits = []
    if not tree.get("is_leaf", False):
        splits.append(tree["split_value"])
        extract_splits(tree["left"], splits)
        extract_splits(tree["right"], splits)
    return splits


split_points = extract_splits(one_tree)


# -------------------------------
# 4. PLOT ISOLATION PROGRESS
# -------------------------------
def plot_isolation_progress(x_vals, split_points, out_dir="frames"):
    os.makedirs(out_dir, exist_ok=True)

    for i in range(1, len(split_points) + 1):
        plt.figure(figsize=(10, 2.5))
        plt.scatter(
            x_vals,
            [0] * len(x_vals),
            color="blue",
            zorder=5,
            label="Data Points",
            s=500,
        )

        for j in range(i):
            # plt.axvline(x=split_points[j], color="black", linestyle="-", ymax=0.6)
            plt.axvline(
                x=split_points[j],
                color="black",
                linestyle="-",
                linewidth=10,
                ymin=0,
                ymax=1.0,
                zorder=1,
            )

        plt.ylim(-1, 1)
        plt.xlim(min(x_vals) - 1, max(x_vals) + 1)
        # plt.xlabel("x[0]", fontsize=22)
        plt.title(f"Split {i}", fontsize=22)
        plt.yticks([])
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # plt.title(f"Isolation Progress - {i} split(s)", fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/frame_{i:03d}.png")
        plt.close()


plot_isolation_progress(x0, split_points)


# -------------------------------
# 5. MAKE GIF
# -------------------------------
def make_gif(frame_folder="frames", output="isolation.gif"):
    frames = sorted(f for f in os.listdir(frame_folder) if f.endswith(".png"))
    images = [imageio.imread(os.path.join(frame_folder, f)) for f in frames]
    imageio.mimsave(output, images, duration=0.5)


make_gif()
