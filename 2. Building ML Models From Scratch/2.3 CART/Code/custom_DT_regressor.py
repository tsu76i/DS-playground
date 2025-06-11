import numpy as np
from typing import Dict
from numpy.typing import NDArray
from node import Node


class CustomDecisionTreeRegressor:
    """
    A class representing a decision tree regressor.

    Attributes:
        max_depth (int or None): Maximum depth of the tree. None for unlimited depth.
        metric (str): Splitting criterion, either "variance" or "mse".
        root (dict): Root node of the decision tree.
        feature_names (list[str] or None): List of feature names. None if not provided.
        min_variance (float): Minimum variance to continue splitting.
        min_samples_split (int): Minimum number of samples required to split a node.
    """

    def __init__(self, max_depth: int = None, metric: str = 'variance',
                 min_variance: float = 1e-7, min_sample_split: int = 2):
        """
        Initialise a CustomDecisionTreeRegressor instance.

        Args:
            max_depth (int or None): Maximum depth of the tree. None for unlimited depth.
            metric (str): Splitting criterion, either "variance" or "mse".
            min_variance (float): Minimum variance to continue splitting.
            min_sample_split (int): Minimum number of samples required to split a node.
        """
        self.max_depth = max_depth
        self.metric = metric
        self.root = None
        self.feature_names = None
        self.min_variance = min_variance
        self.min_sample_split = min_sample_split

    def variance(self, y: NDArray[np.float64]) -> float:
        """
        Calculate the variance.

        Args:
            y (NDArray[np.float64]): Array of values.

        Returns:
            float: Variance value.
        """
        return np.var(y) if len(y) > 0 else 0

    def mse(self, y: NDArray[np.float64]) -> float:
        """
        Calculate the mean squared error.

        Args:
            y (NDArray[np.float64]): Array of values.

        Returns:
            float: Mean squared error value.
        """
        return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

    def information_gain(self, y: NDArray[np.float64], y_left: NDArray[np.float64], y_right: NDArray[np.float64]) -> float:
        """
        Compute the information gain of a split.

        Args:
            y (NDArray[np.float64]): Values of the parent node.
            y_left (NDArray[np.float64]): Values of the left child node.
            y_right (NDArray[np.float64]): Values of the right child node.

        Returns:
            float: Information gain from the split.
        """
        if self.metric == "variance":
            parent_metric = self.variance(y)
            left_metric = self.variance(y_left)
            right_metric = self.variance(y_right)
        else:  # metric == "mse"
            parent_metric = self.mse(y)
            left_metric = self.mse(y_left)
            right_metric = self.mse(y_right)

        weighted_metric = (
            len(y_left) / len(y) * left_metric
            + len(y_right) / len(y) * right_metric
        )
        return parent_metric - weighted_metric

    def best_split(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> Dict[int, np.float64]:
        """
        Find the best feature and threshold to split the dataset.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.float64]): Target values as float.

        Returns:
            Dict[int, np.float64]: Best split details with keys 'feature_index' and 'threshold'.
        """
        best_info_gain = float("-inf")
        best_split: Dict = None
        n_features: int = X.shape[1]

        for feature in range(n_features):
            thresholds: NDArray[np.float64] = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask: NDArray[np.bool] = X[:, feature] <= threshold
                right_mask: NDArray[np.bool] = X[:, feature] > threshold

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                info_gain: float = self.information_gain(
                    y, y[left_mask], y[right_mask])

                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        "feature_index": feature,
                        "threshold": threshold,
                    }

        return best_split

    def build_tree(self, X: NDArray[np.float64], y: NDArray[np.float64], depth: int = 0) -> Node:
        """
        Build the decision tree recursively.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.float64]): Target variables as float.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the decision tree.
        """
        if (
            len(y) < self.min_sample_split or
            (self.max_depth is not None and depth == self.max_depth) or
            np.var(y) <= self.min_variance
        ):
            return Node(type="leaf", value=round(float(np.mean(y)), 4))

        split = self.best_split(X, y)
        if not split:
            return Node(type="leaf", value=round(float(np.mean(y)), 4))

        # Split the data
        left_mask: NDArray[np.bool] = X[:,
                                        split["feature_index"]] <= split["threshold"]
        right_mask: NDArray[np.bool] = X[:,
                                         split["feature_index"]] > split["threshold"]

        # Recursively build the left and right subtrees
        left_tree: Node = self.build_tree(
            X[left_mask], y[left_mask], depth + 1)
        right_tree: Node = self.build_tree(
            X[right_mask], y[right_mask], depth + 1)

        # Store the feature index directly for easier traversal
        feature_index: int = split["feature_index"]
        return Node(type="node", feature=feature_index, threshold=split["threshold"], left=left_tree, right=right_tree)

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64], feature_names: NDArray[np.str_] = None) -> None:
        """
        Fit the decision tree regressor to the given data.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.float64]): Target values.
            feature_names (NDArray[np.str_], optional): Names of the features. Defaults to None.
        """
        self.feature_names = feature_names
        self.root = self.build_tree(X, y)

    def traverse_tree(self, x: NDArray[np.float64], node: Node) -> float:
        """
        Traverse the decision tree to make a prediction for a single sample.

        Args:
            x (NDArray[np.float64]): Single sample.
            node (Node): Current node.

        Returns:
            float: Predicted value.
        """
        if node.type == "leaf":
            return node.value

        feature_index = node.feature
        if x[feature_index] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X: NDArray[np.float64]) -> float | NDArray[np.float64]:
        """
        Predict values for the given dataset.

        Args:
            X (NDArray[np.float64]): Input features.

        Returns:
            float or NDArray[np.float64]: Predicted value(s).
        """
        if len(X.shape) == 1:
            return self.traverse_tree(X, self.root)
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def print_tree(self, node: Node = None, depth: int = 0, prefix: str = "Root: ") -> None:
        """
        Print the tree structure in a readable format.

        Args:
            node (Node, optional): Current node. Defaults to the root node.
            depth (int, optional): Current depth. Defaults to 0.
            prefix (str, optional): Prefix for the current node. Defaults to "Root: ".
        """
        if node is None:
            node = self.root

        if node.type == "leaf":
            print("  " * depth + prefix + f"Predict -> {node.value:.4f}")
        else:
            feature_name = self.feature_names[node.feature
                                              ] if self.feature_names is not None else f"Feature_{node.feature}"
            print("  " * depth + prefix +
                  f"{feature_name} <= {node.threshold:.4f}")
            self.print_tree(node.left, depth + 1, "├─ True: ")
            self.print_tree(node.right, depth + 1, "└─ False: ")
