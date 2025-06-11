import numpy as np
from typing import Dict
from numpy.typing import NDArray
from node import Node


class CustomDecisionTreeClassifier:
    """
    A class representing a decision tree classifier.

    Args:
        max_depth (int or None): Maximum depth of the tree. None for unlimited depth.
        metric (str): Splitting criterion, either "gini" or "entropy".

    Attributes:
        max_depth (int or None): Maximum depth of the tree.
        metric (str): Splitting criterion.
        root (Node or None): Root node of the decision tree, initialised as None.
        feature_names (list[str] or None): Names of the features, set during fitting.
        class_names (List[str] or None): Names of the classes, set during fitting.
    """

    def __init__(self, max_depth: int = None, metric: str = 'gini'):
        self.max_depth = max_depth
        self.metric = metric
        self.root = None
        self.feature_names = None
        self.class_names = None

    def gini(self, y: NDArray[np.int64]) -> float:
        """
        Calculate the Gini impurity.

        Args:
            y (NDArray[np.int64]): Array of labels.

        Returns:
            float: Gini impurity.
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        return 1 - np.sum(proportions ** 2)

    def entropy(self, y: NDArray[np.int64]) -> float:
        """
        Calculate the entropy.

        Args:
            y (NDArray[np.int64]): Array of labels.

        Returns:
            float: Entropy value.
        """
        if len(y) == 0:
            return 0
        proportions = np.bincount(y) / len(y)
        proportions = proportions[proportions > 0]  # Avoid log(0)
        return -np.sum(proportions * np.log2(proportions))

    def information_gain(self, y: NDArray[np.int64], y_left: NDArray[np.int64], y_right: NDArray[np.int64]) -> float:
        """
        Compute the information gain of a split.

        Args:
            y (NDArray[np.int64]): Labels of the parent node.
            y_left (NDArray[np.int64]): Labels of the left child node.
            y_right (NDArray[np.int64]): Labels of the right child node.

        Returns:
            float: Information gain from the split.
        """
        if self.metric == "gini":
            parent_metric = self.gini(y)
            left_metric = self.gini(y_left)
            right_metric = self.gini(y_right)
        else:  # metric == "entropy"
            parent_metric = self.entropy(y)
            left_metric = self.entropy(y_left)
            right_metric = self.entropy(y_right)

        weighted_metric: float = (
            len(y_left) / len(y) * left_metric
            + len(y_right) / len(y) * right_metric
        )
        return parent_metric - weighted_metric

    def best_split(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> dict:
        """
        Find the best feature and threshold to split the dataset.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.int64]): Labels as int.

        Returns:
            dict: Best split details with keys 'feature_index' and 'threshold'.
        """
        best_info_gain = float("-inf")
        best_split: Dict = None
        n_features: int = X.shape[1]

        # Iterate over all features.
        for feature in range(n_features):
            # Iterate over all unique thresholds for each feature.
            thresholds: NDArray[np.float64] = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split the data into left and right subsets based on the threshold.
                left_mask: NDArray[np.bool] = X[:, feature] <= threshold
                right_mask: NDArray[np.bool] = X[:, feature] > threshold

                # Skip invalid splits.
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                # Compute IG.
                info_gain: float = self.information_gain(
                    y, y[left_mask], y[right_mask])

                # Update `best_info_gain` if `info_gain` > `best_info_gain`.
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        "feature_index": feature,
                        "threshold": threshold,
                    }

        return best_split

    def build_tree(self, X: NDArray[np.float64], y: NDArray[np.int64], depth: int = 0) -> Node:
        """
        Build the decision tree recursively.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.int64]): Labels.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the decision tree.
        """
        # Stop recursion if all labels are identical or max depth is reached
        if len(set(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            return Node(type="leaf", value=np.argmax(np.bincount(y)))

        # Find the best split
        split: Dict = self.best_split(X, y)
        if not split:
            return Node(type="leaf", value=np.argmax(np.bincount(y)))

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

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int64],
            class_names: NDArray[np.str_], feature_names: NDArray[np.str_] = None) -> None:
        """
        Fit the decision tree model to the given data.

        Args:
            X (NDArray[np.float64]): Input features.
            y (NDArray[np.int64]): Labels.
            class_names (NDArray[np.str_]): Names of the labels.
            feature_names (NDArray[np.str_], optional): Names of the features. Defaults to None.
        """
        self.feature_names = feature_names
        self.class_names = class_names
        self.root = self.build_tree(X, y)

    def traverse_tree(self, x: NDArray[np.float64], node: Node) -> int:
        """
        Traverse the decision tree to make a prediction for a single sample.

        Args:
            x (NDArray[np.float64]): Single sample.
            node (Node): Current node.

        Returns:
            int: Predicted label.
        """
        if node.type == "leaf":
            return node.value

        # node.feature = feature index
        feature_index: int = node.feature
        if x[feature_index] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)

    def predict(self, X: NDArray[np.float64]) -> int | NDArray[np.int64]:
        """
        Predict labels for the given dataset.

        Args:
            X (NDArray[np.float64]): Input features.

        Returns:
            int or NDArray[np.int64]: Predicted label(s).
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
            print("  " * depth + prefix +
                  f"Predict -> {self.class_names[node.value]}")
        else:
            feature_name = self.feature_names[
                node.feature] if self.feature_names is not None else f"Feature_{node.feature}"
            print("  " * depth + prefix +
                  f"{feature_name} <= {node.threshold:.4f}")
            if node.left:
                self.print_tree(node.left, depth + 1, "├─ True: ")
            if node.right:
                self.print_tree(node.right, depth + 1, "└─ False: ")