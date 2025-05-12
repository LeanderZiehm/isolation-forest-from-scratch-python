import random
import math


class IsolationForest:
    def __init__(
        self, num_trees=100, subsample_size=256, max_depth=None, random_seed=None
    ):
        """
        Initialize the Isolation Forest algorithm.

        Parameters:
        -----------
        num_trees : int
            Number of trees in the forest (m)
        subsample_size : int
            Size of subsample to build each tree (ψ)
        max_depth : int, optional
            Maximum depth of trees. If None, calculated as log2(subsample_size)
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.num_trees = num_trees
        self.subsample_size = subsample_size
        self.max_depth = (
            max_depth
            if max_depth is not None
            else int(math.ceil(math.log2(subsample_size)))
        )
        self.forest = None

        if random_seed is not None:
            random.seed(random_seed)

    def fit(self, input_columns):
        """
        Build the isolation forest from the input data.

        Parameters:
        -----------
        input_columns : list of lists
            Each inner list represents a feature column
        """
        self.input_columns = input_columns
        self.forest = self._build_forest()
        return self

    def _build_iTree(self, indexes, current_depth):
        """Build a single isolation tree."""
        node = {"indexes": indexes, "item_count": len(indexes), "depth": current_depth}

        # Stopping conditions
        if current_depth >= self.max_depth or len(indexes) <= 1:
            node["is_leaf"] = True
            node["item_value"] = self.input_columns[0][indexes[0]]
            return node

        # Random split
        feat_idx = random.randrange(len(self.input_columns))
        values = [self.input_columns[feat_idx][i] for i in indexes]
        min_v, max_v = min(values), max(values)

        if min_v == max_v:
            node["is_leaf"] = True
            return node

        split_v = random.uniform(min_v, max_v)
        left_idxs, right_idxs = [], []

        for i in indexes:
            if self.input_columns[feat_idx][i] < split_v:
                left_idxs.append(i)
            else:
                right_idxs.append(i)

        node.update(
            {
                "is_leaf": False,
                "input_column_index": feat_idx,
                "split_value": split_v,
                "left": self._build_iTree(left_idxs, current_depth + 1),
                "right": self._build_iTree(right_idxs, current_depth + 1),
            }
        )

        return node

    def _build_forest(self):
        """Build the entire isolation forest."""
        forest = []
        N = len(self.input_columns[0])

        for _ in range(self.num_trees):
            # Subsample indexes (with replacement if N>ψ)
            if N > self.subsample_size:
                samp = [random.randrange(N) for _ in range(self.subsample_size)]
            else:
                samp = list(range(N))

            tree = self._build_iTree(samp, 0)
            forest.append(tree)

        return forest

    def _path_length(self, row_index, tree):
        """Calculate the path length for a data point in a tree."""
        if tree.get("is_leaf", False):
            # add adjustment: average external path length
            return tree["depth"] + self._c(tree["item_count"])

        selected_feature_index = tree["input_column_index"]
        selected_feature = self.input_columns[selected_feature_index]

        if selected_feature[row_index] < tree["split_value"]:
            return self._path_length(row_index, tree["left"])
        else:
            return self._path_length(row_index, tree["right"])

    def _c(self, n):
        """Average path length of unsuccessful search in binary tree."""
        if n <= 1:
            return 0

        def H(h):
            return math.log(h) + 0.5772156649  # Euler's constant

        return 2 * H(n - 1) - 2 * (n - 1) / n

    def score_samples(self, row_indices=None):
        """
        Calculate anomaly scores for the given indices or all data points.

        Parameters:
        -----------
        row_indices : list or None
            List of row indices to score. If None, scores all data points.

        Returns:
        --------
        list
            Anomaly scores for the specified rows
        """
        if self.forest is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if row_indices is None:
            row_indices = range(len(self.input_columns[0]))

        scores = []
        for idx in row_indices:
            scores.append(self._anomaly_score(idx))

        return scores

    def _anomaly_score(self, row_index):
        """Calculate the anomaly score for a single data point."""
        n = self.subsample_size

        def Eh(row_index):
            sum_path_lengths = sum(
                self._path_length(row_index, tree) for tree in self.forest
            )
            trees_count = len(self.forest)
            return sum_path_lengths / trees_count

        return 2 ** (-Eh(row_index) / self._c(n))

    def predict(self, threshold=0.5, row_indices=None):
        """
        Predict anomalies based on threshold.

        Parameters:
        -----------
        threshold : float
            Threshold for anomaly detection (default: 0.5)
        row_indices : list or None
            List of row indices to predict. If None, predicts for all data points.

        Returns:
        --------
        list
            1 for anomalies, 0 for normal instances
        """
        scores = self.score_samples(row_indices)
        return [1 if score > threshold else 0 for score in scores]


if __name__ == "__main__":
    # Example usage (if this file is run directly)
    x1 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 100]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    input_columns = [x1]  # only features
    solution_column = y  # labels (not used in training)

    # Create and fit the model
    iforest = IsolationForest(num_trees=100, subsample_size=256, random_seed=0)
    iforest.fit(input_columns)
    scores = iforest.score_samples()
    print(scores)
