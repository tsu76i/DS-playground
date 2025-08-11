import numpy as np
from numpy.typing import NDArray


class CustomSGBRegressor:
    """
    Custom Stochastic Gradient Boosting for regression with decision trees.
    """

    def __init__(
        self,
        n_estimators: int = 3,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        sub_samples: float = 0.8,  # Added
        sub_features: float = 0.8,  # Added
        random_state: int = None,  # Added
    ):
        """
        Initialises the CustomSGBRegressor.

        Args:
            n_estimators: Number of boosting rounds.
            learning_rate: Learning rate (shrinkage).
            max_depth: Maximum depth of each tree.
            min_samples_leaf: Minimum samples per leaf.
            sub_samples: Friction of row subsampling. Default = 0.8.
            sub_features: Friction of feature subsampling. Default = 0.8.
            random_state: Seed for the random number generator to ensure reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.models: list[dict | float] = []
        self.initial_prediction: float = 0.0
        self.sub_samples = sub_samples  # Added
        self.sub_features = sub_features  # Added
        self.random_state = random_state  # Added

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Fit the gradient boosting regressor to the data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
        """
        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.models = []
        self.initial_prediction = float(np.mean(y))
        predictions = np.full_like(y, self.initial_prediction, dtype=float)
        for _ in range(self.n_estimators):
            residuals = y - predictions
            # Stochastic row subsampling
            n_samples = int(self.sub_samples * X.shape[0])
            sample_indices = np.random.choice(X.shape[0], n_samples, replace=False)
            X_sub, residuals_sub = X[sample_indices], residuals[sample_indices]
            tree = self._build_tree(
                # Build trees with subsamples
                X_sub,
                residuals_sub,
                self.max_depth,
                self.min_samples_leaf,
            )
            self.models.append(tree)
            # Predictions on the entire dataset
            update = self._predict_tree_batch(tree, X)
            predictions += self.learning_rate * update

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict target values for given feature matrix.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values.
        """
        y_pred = np.full(X.shape[0], self.initial_prediction, dtype=np.float64)
        for tree in self.models:
            y_pred += self.learning_rate * self._predict_tree_batch(tree, X)
        return y_pred

    def _variance(self, y: NDArray[np.float64]) -> float:
        """
        Calculate the variance of the target values.

        Args:
            y: Target values.

        Returns:
            float: Variance of y.
        """
        return np.var(y)

    def _split_dataset(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        feature_index: int,
        threshold: float,
    ) -> tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        """
        Splits the dataset based on a feature and threshold.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target values, shape (n_samples,).
            feature_index: Index of the feature to split on.
            threshold: Threshold value for the split.

        Returns:
            Tuple containing X_left, y_left, X_right, y_right after the split.
        """
        left_mask = X[:, feature_index] < threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _best_split(
        self, X: NDArray[np.float64], y: NDArray[np.float64], min_samples_leaf: int
    ) -> tuple[int | None, float | None]:
        """
        Find the best feature and threshold to split the dataset, minimising weighted variance.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target values.
            min_samples_leaf: Minimum number of samples required at a leaf node.

        Returns:
            Tuple of (best_feature, best_threshold). Returns (None, None) if no valid split is found.
        """
        m, n = X.shape
        # Stochastic feature subsampling
        n_features = int(self.sub_features * n)
        feature_indices = np.random.choice(n, n_features, replace=False)

        best_feature, best_threshold, best_var = None, None, float("inf")
        for feature in feature_indices:
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                _, y_left, _, y_right = self._split_dataset(X, y, feature, threshold)
                if len(y_left) < min_samples_leaf or len(y_right) < min_samples_leaf:
                    continue
                var_left = self._variance(y_left)
                var_right = self._variance(y_right)
                var_split = (len(y_left) * var_left + len(y_right) * var_right) / m
                if var_split < best_var:
                    best_feature = feature
                    best_threshold = threshold
                    best_var = var_split
        return best_feature, best_threshold

    def _build_tree(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        max_depth: int,
        min_samples_leaf: int,
        depth: int = 0,
    ) -> dict | float:
        """
        Recursively build a regression tree.

        Args:
            X: Feature matrix.
            y: Target values.
            max_depth: Maximum depth of the tree.
            min_samples_leaf: Minimum samples required at a leaf node.
            depth: Current depth of the tree (default is 0).

        Returns:
            Tree as a nested dictionary, or a float if a leaf node.
        """
        if depth >= max_depth or len(y) <= min_samples_leaf:
            return float(np.mean(y))
        feature, threshold = self._best_split(X, y, min_samples_leaf)
        if feature is None:
            return float(np.mean(y))
        X_left, y_left, X_right, y_right = self._split_dataset(X, y, feature, threshold)
        return {
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(
                X_left, y_left, max_depth, min_samples_leaf, depth + 1
            ),
            "right": self._build_tree(
                X_right, y_right, max_depth, min_samples_leaf, depth + 1
            ),
        }

    def _predict_tree(self, tree: dict | float, x: NDArray[np.float64]) -> float:
        """
        Predict the target value for a single sample using the regression tree.

        Args:
            tree: The regression tree or a leaf value.
            x: Feature vector for a single sample.

        Returns:
            float: Predicted value.
        """
        while isinstance(tree, dict):
            if x[tree["feature"]] < tree["threshold"]:
                tree = tree["left"]
            else:
                tree = tree["right"]
        return float(tree)

    def _predict_tree_batch(
        self, tree: dict | float, X: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Predict target values for a batch of samples using the regression tree.

        Args:
            tree: The regression tree or a leaf value.
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predicted values for all samples.
        """
        return np.array([self._predict_tree(tree, x) for x in X], dtype=np.float64)
