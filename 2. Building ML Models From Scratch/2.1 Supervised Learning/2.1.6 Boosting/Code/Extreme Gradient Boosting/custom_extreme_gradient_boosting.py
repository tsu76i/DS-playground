import numpy as np
from typing import Tuple, List, Dict, Any
from numpy.typing import NDArray


class CustomXGBoost:
    """
    Custom Extreme Gradient Boosting for regression with decision trees.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        lambda_: float = 1.0,
        gamma: float = 0.0,
    ) -> None:
        """
        Initialise the CustomXGBoost regressor with specified hyperparameters.

        Parameters:
            n_estimators: Number of boosting rounds (trees).
            learning_rate: Shrinkage factor for each tree's contribution.
            max_depth: Maximum depth of each regression tree.
            min_samples_leaf: Minimum number of samples required in a leaf node.
            lambda_: L2 regularisation parameter for leaf weights.
            gamma: Minimum loss reduction required to make a split.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.lambda_ = lambda_
        self.gamma = gamma
        self.initial_prediction = None
        self.models: List[Dict[str, Any] | float] = []

    def _compute_gradients_and_hessians(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Compute gradients and Hessians for squared error loss.

        Args:
            y_true: True target values, shape (n_samples,).
            y_pred: Predicted values, shape (n_samples,).

        Returns:
            A tuple with gradients and Hessians, both of shape (n_samples,).
        """
        gradients = y_pred - y_true
        hessians = np.ones_like(y_true)
        return gradients, hessians

    def _best_split(
        self,
        X: NDArray[np.float64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
    ) -> Tuple[int, float, float]:
        """
        Find the best split for a node in the XGBoost tree.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            gradients: Gradients for each sample, shape (n_samples,).
            hessians: Hessians for each sample, shape (n_samples,).
        Returns:
            A tuple with best feature index, best threshold, and best gain.
        """
        best_feature, best_threshold, best_gain = None, None, -np.inf
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask
                if (
                    left_mask.sum() < self.min_samples_leaf
                    or right_mask.sum() < self.min_samples_leaf
                ):
                    continue
                G_L, H_L = gradients[left_mask].sum(), hessians[left_mask].sum()
                G_R, H_R = gradients[right_mask].sum(), hessians[right_mask].sum()
                gain = (
                    0.5
                    * (
                        G_L**2 / (H_L + self.lambda_)
                        + G_R**2 / (H_R + self.lambda_)
                        - (G_L + G_R) ** 2 / (H_L + H_R + self.lambda_)
                    )
                    - self.gamma
                )
                if gain > best_gain:
                    best_feature, best_threshold, best_gain = feature, threshold, gain
        return best_feature, best_threshold, best_gain

    def _build_tree(
        self,
        X: NDArray[np.float64],
        gradients: NDArray[np.float64],
        hessians: NDArray[np.float64],
        depth: int = 0,
    ) -> Dict[str, Any] | float:
        """
        Recursively build a decision tree for XGBoost.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            gradients: Gradients for each sample, shape (n_samples,).
            hessians: Hessians for each sample, shape (n_samples,).
            depth: Current depth of the tree. Defaults to 0.

        Returns:
            Tree structure as nested dictionaries, or a float for leaf value.
        """
        if depth >= self.max_depth or len(gradients) <= self.min_samples_leaf:
            return -gradients.sum() / (hessians.sum() + self.lambda_)
        feature, threshold, gain = self._best_split(X, gradients, hessians)
        if feature is None or gain <= 0:
            return -gradients.sum() / (hessians.sum() + self.lambda_)
        left_mask = X[:, feature] < threshold
        right_mask = ~left_mask
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(
                X[left_mask], gradients[left_mask], hessians[left_mask], depth + 1
            ),
            "right": self._build_tree(
                X[right_mask], gradients[right_mask], hessians[right_mask], depth + 1
            ),
        }

    def _predict_tree(
        self, tree: Dict[str, Any] | float, x: NDArray[np.float64]
    ) -> float:
        """
        Predict the output for a single sample using a decision tree.

        Args:
            tree: Tree structure or leaf value.
            x: Feature vector, shape (n_features,).

        Returns:
            Predicted value for the sample.
        """
        while isinstance(tree, dict):
            if x[tree["feature"]] < tree["threshold"]:
                tree = tree["left"]
            else:
                tree = tree["right"]

        return tree

    def _predict_tree_batch(
        self, tree: Dict[str, Any] | float, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Predict outputs for a batch of samples using a decision tree.

        Args:
            tree: Tree structure or leaf value.
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        return np.array([self._predict_tree(tree, x) for x in X])

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Fit an XGBoost-like model for regression.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
        """
        self.models = []
        y_pred = np.full_like(y, np.mean(y), dtype=float)
        self.initial_prediction = np.mean(y)
        for _ in range(self.n_estimators):
            gradients, hessians = self._compute_gradients_and_hessians(y, y_pred)
            tree = self._build_tree(X, gradients, hessians)
            update = self._predict_tree_batch(tree, X)
            y_pred += self.learning_rate * update
            self.models.append(tree)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict outputs for a batch of samples using the fitted model.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values, shape (n_samples,).
        """
        y_pred = np.full(X.shape[0], self.initial_prediction, dtype=float)
        for tree in self.models:
            y_pred += self.learning_rate * self._predict_tree_batch(tree, X)
        return y_pred
