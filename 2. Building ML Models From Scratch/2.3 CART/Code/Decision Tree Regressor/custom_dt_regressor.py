import numpy as np
import pandas as pd
from typing import Dict
from numpy.typing import NDArray
from node import Node


class CustomDecisionTreeRegressor:
    """
    A class representing a decision tree regressor.

    Attributes:
        max_depth: Maximum depth of the tree. None for unlimited depth.
        metric: Splitting criterion, either 'variance' or 'mse'.
        root: Root node of the decision tree.
        feature_names: List of feature names. None if not provided.
        min_variance: Minimum variance to continue splitting.
        min_samples_split: Minimum number of samples required to split a node.
    """

    def __init__(self, max_depth: int = None, metric: str = 'variance',
                 min_variance: float = 1e-7, min_sample_split: int = 2) -> None:
        """
        Initialise the CustomDecisionTreeRegressor instance.

        Args:
            max_depth: Maximum depth of the tree. None for unlimited depth.
            metric: Splitting criterion, either 'variance' or 'mse'.
            min_variance: Minimum variance to continue splitting.
            min_sample_split: Minimum number of samples required to split a node.
        """
        self.max_depth = max_depth
        self.metric = metric
        self.root = None
        self.feature_names = None
        self.min_variance = min_variance
        self.min_sample_split = min_sample_split

    def variance(self, y: pd.Series) -> float:
        """
        Calculate the variance.

        Args:
            y: Series of values.

        Returns:
            float: Variance value.
        """
        return np.var(y) if len(y) > 0 else 0

    def mse(self, y: pd.Series) -> float:
        """
        Calculate the mean squared error.

        Args:
            y: Series of values.

        Returns:
            Mean squared error value.
        """
        return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

    def information_gain(self, y: pd.Series, y_left: pd.Series, y_right: pd.Series) -> float:
        """
        Compute the information gain of a split.

        Args:
            y: Values of the parent node.
            y_left: Values of the left child node.
            y_right: Values of the right child node.

        Returns:
            float: Information gain from the split.
        """
        if self.metric == 'variance':
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

    def best_split(self, X: pd.DataFrame, y: pd.Series) -> Dict[int, np.float64]:
        """
        Find the best feature and threshold to split the dataset.

        Args:
            X: Input features.
            y: Target values as float.

        Returns:
            Best split details with keys 'feature_index' and 'threshold'.
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
                        'feature_index': feature,
                        'threshold': threshold,
                    }

        return best_split

    def build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> Node:
        """
        Build the decision tree recursively.

        Args:
            X: Input features.
            y : Target variables as float.
            depth: Current depth of the tree.

        Returns:
            Root node of the decision tree.
        """

        # Convert DataFrames to NumPy arrays
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()
        if hasattr(y, 'to_numpy'):
            y = y.to_numpy().flatten()  # Ensure 1D array

        if (
            len(y) < self.min_sample_split or
            (self.max_depth is not None and depth == self.max_depth) or
            np.var(y) <= self.min_variance
        ):
            return Node(type='leaf', value=round(float(np.mean(y)), 4))

        split = self.best_split(X, y)
        if not split:
            return Node(type='leaf', value=round(float(np.mean(y)), 4))

        # Split the data
        left_mask: NDArray[np.bool] = X[:,
                                        split['feature_index']] <= split['threshold']
        right_mask: NDArray[np.bool] = X[:,
                                         split['feature_index']] > split['threshold']

        # Recursively build the left and right subtrees
        left_tree: Node = self.build_tree(
            X[left_mask], y[left_mask], depth + 1)
        right_tree: Node = self.build_tree(
            X[right_mask], y[right_mask], depth + 1)

        # Store the feature index directly for easier traversal
        feature_index: int = split['feature_index']
        return Node(type='node', feature=feature_index, threshold=split['threshold'], left=left_tree, right=right_tree)

    def fit(self, X: pd.DataFrame, y: pd.Series, feature_names: NDArray[np.str_] = None) -> None:
        """
        Fit the decision tree regressor to the given data.

        Args:
            X: Input features.
            y: Target values.
            feature_names: Names of the features. Defaults to None.
        """
        self.feature_names = feature_names
        self.root = self.build_tree(X, y)

    def traverse_tree(self, x: pd.DataFrame, node: Node) -> float:
        """
        Traverse the decision tree to make a prediction for a single sample.

        Args:
            x: Single sample.
            node: Current node.

        Returns:
            Predicted value.
        """
        if node.type == 'leaf':
            return node.value

        feature_index = node.feature
        if x[feature_index] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X: pd.DataFrame) -> float | NDArray[np.float64]:
        """
        Predict values for the given dataset.

        Args:
            X: Input features.

        Returns:
            Predicted value(s).
        """
        # Convert DataFrames to NumPy arrays
        if hasattr(X, 'to_numpy'):
            X = X.to_numpy()

        if len(X.shape) == 1:
            return self.traverse_tree(X, self.root)
        return np.array([self.traverse_tree(x, self.root) for x in X])

    def print_tree(self, node: Node = None, depth: int = 0, prefix: str = 'Root: ') -> None:
        """
        Print the tree structure in a readable format.

        Args:
            node: Current node. Defaults to the root node.
            depth: Current depth. Defaults to 0.
            prefix: Prefix for the current node. Defaults to 'Root: '.
        """
        if node is None:
            node = self.root

        if node.type == 'leaf':
            print('  ' * depth + prefix + f'Predict -> {node.value:.4f}')
        else:
            feature_name = self.feature_names[node.feature
                                              ] if self.feature_names is not None else f'Feature_{node.feature}'
            print('  ' * depth + prefix +
                  f'{feature_name} <= {node.threshold:.4f}')
            self.print_tree(node.left, depth + 1, '├─ True: ')
            self.print_tree(node.right, depth + 1, '└─ False: ')
