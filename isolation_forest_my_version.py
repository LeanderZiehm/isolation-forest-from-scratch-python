import random
import math
from typing import List, Optional, Dict, Any, Sequence, Union


class IsolationForest:
    def __init__(
        self,
        num_trees: int = 100,
        subsample_size: int = 256,
        max_depth: Optional[int] = None,
        random_seed: Optional[int] = None,
    ) -> None:
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
        self.num_trees: int = num_trees
        self.subsample_size: int = subsample_size
        self.max_depth: int = (
            max_depth
            if max_depth is not None
            else int(math.ceil(math.log2(subsample_size)))
        )
        self.forest: Optional[List[Dict[str, Any]]] = None
        self.table: Optional[Sequence[Sequence[float]]] = None

        if random_seed is not None:
            random.seed(random_seed)

    def fit(self, table: Sequence[Sequence[float]]) -> "IsolationForest":
        """
        Build the isolation forest from the input data.

        Parameters:
        -----------
        table : Sequence of feature columns, each a sequence of floats
        """
        self.table = table
        self.forest = self._build_forest()
        return self

    def _build_forest(self) -> List[Dict[str, Any]]:
        """Build the entire isolation forest."""
        forest: List[Dict[str, Any]] = []
        row_count = len(self.table[0]) if self.table else 0

        for _ in range(self.num_trees):
            # adjust sample if fewer rows than subsample_size
            if row_count < self.subsample_size:
                samp = list(range(row_count))
            else:
                samp = [random.randrange(row_count) for _ in range(self.subsample_size)]

            tree = self._build_iTree(samp, 0)
            forest.append(tree)

        return forest

    def _build_iTree(self, indexes: List[int], current_depth: int) -> Dict[str, Any]:
        """Build a single isolation tree."""
        node: Dict[str, Any] = {
            "indexes": indexes,
            "item_count": len(indexes),
            "depth": current_depth,
        }

        # Stopping conditions
        if current_depth >= self.max_depth or len(indexes) <= 1:
            node["is_leaf"] = True
            # store value for singleton leaf
            node["item_value"] = self.table[0][indexes[0]] if indexes else None
            return node

        # Random split
        feat_idx = random.randrange(len(self.table))  # type: ignore
        values = [self.table[feat_idx][i] for i in indexes]  # type: ignore
        min_v, max_v = min(values), max(values)

        if min_v == max_v:
            node["is_leaf"] = True
            return node

        split_v = random.uniform(min_v, max_v)
        left_idxs: List[int] = []
        right_idxs: List[int] = []

        for i in indexes:
            if self.table[feat_idx][i] < split_v:  # type: ignore
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

    def _path_length(self, row_index: int, tree: Dict[str, Any]) -> float:
        """Calculate the path length for a data point in a tree."""
        if tree.get("is_leaf", False):
            return tree["depth"] + self._c(tree.get("item_count", 0))

        feat_idx = tree["input_column_index"]
        if self.table and self.table[feat_idx][row_index] < tree["split_value"]:  # type: ignore
            return self._path_length(row_index, tree["left"])
        return self._path_length(row_index, tree["right"])

    def _c(self, n: int) -> float:
        """Average path length of unsuccessful search in binary tree."""
        if n <= 1:
            return 0.0

        euler_const = 0.5772156649
        return 2.0 * (math.log(n - 1) + euler_const) - (2.0 * (n - 1) / n)

    def score_samples(self, row_indices: Optional[List[int]] = None) -> List[float]:
        """
        Calculate anomaly scores for the given indices or all data points.

        Returns
        -------
        List of anomaly scores
        """
        if self.forest is None or self.table is None:
            raise ValueError("Model not fitted. Call fit() first.")

        if row_indices is None:
            row_indices = list(range(len(self.table[0])))  # type: ignore

        return [self._anomaly_score(idx) for idx in row_indices]

    def _anomaly_score(self, row_index: int) -> float:
        """Calculate the anomaly score for a single data point."""
        if self.forest is None:
            raise ValueError("Forest not built.")

        n = self.subsample_size
        avg_path = sum(
            self._path_length(row_index, tree) for tree in self.forest
        ) / len(self.forest)
        return 2 ** (-avg_path / self._c(n))

    def predict(
        self, threshold: float = 0.5, row_indices: Optional[List[int]] = None
    ) -> List[int]:
        """
        Predict anomalies based on threshold.

        Returns
        -------
        List[int]
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
