import numpy as np
import pandas as pd
from numpy.typing import NDArray
from joblib import Parallel, delayed


class CustomRandomForestRegressor:
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 15,
        min_samples_leaf: int = 1,
        min_samples_split=2,
        metric: str = "variance",
        max_features: int | None = None,
        random_state: int | None = None,
        n_jobs: int = -1,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.metric = metric
        self.max_features = max_features
        self.random_state = random_state
        self.forest = None
        self.n_jobs = n_jobs

    def _variance(self, y: pd.Series) -> float:
        """
        Calculate the variance.

        Args:
            y: Series of values.

        Returns:
            Variance value.
        """
        return np.var(y) if len(y) > 0 else 0

    def _mse(self, y: pd.Series) -> float:
        """
        Calculate the mean squared error.

        Args:
            y: Series of values.

        Returns:
            Mean squared error value.
        """
        return np.mean((y - np.mean(y)) ** 2) if len(y) > 0 else 0

    def _information_gain(
        self, y: pd.Series, y_left: pd.Series, y_right: pd.Series
    ) -> float:
        """
        Compute the information gain of a split.

        Args:
            y: Values of the parent node.
            y_left: Values of the left child node.
            y_right: Values of the right child node.

        Returns:
            Information gain from the split.
        """
        if self.metric == "variance":
            parent_metric = self._variance(y)
            left_metric = self._variance(y_left)
            right_metric = self._variance(y_right)
        else:  # metric == "mse"
            parent_metric = self._mse(y)
            left_metric = self._mse(y_left)
            right_metric = self._mse(y_right)

        weighted_metric: float = (
            len(y_left) / len(y) * left_metric + len(y_right) / len(y) * right_metric
        )
        return parent_metric - weighted_metric

    def _bootstrap_sample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_samples: int | None = None,
        random_state: int | None = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """
        Generate a bootstrap sample from the dataset.

        Args:
            X: Input features.
            y: Target labels.
            n_samples: Samples to draw (default: dataset size).
            random_state: Random seed.

        Returns:
            Bootstrapped (X, y) tuple.
        """
        rng = np.random.RandomState(random_state)
        if n_samples is None:
            n_samples = len(X)
        indices = rng.randint(0, len(X), size=n_samples)
        return X.iloc[indices], y.iloc[indices]

    def _best_split(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> dict:
        """
        Find the best split for a dataset.

        Args:
            X: Input features (DataFrame of shape [n_samples, total_n_features]).
            y: Labels (Series of shape [n_samples]).
            metric: Splitting criterion, either "gini" or "entropy". Defaults to 'gini'.
            feature_names: List of feature names. If None, indices are used. Defaults to None.
            max_features: Number of features to consider at each split. None(logs(total_n_features)) or int(<=total_n_features). Defaults to None.
        Returns:
            Dictionary containing the best split with keys:
                - 'feature_index' : Index of the feature used for the split.
                - 'feature_name': Name or index of the feature.
                - 'threshold' : Threshold value for the split.
        """

        best_info_gain = float("-inf")
        best_split = None
        total_n_features = X.shape[1]

        if isinstance(self.max_features, int):  # if max_features is int
            selected_n_features = (
                self.max_features
                if self.max_features <= total_n_features
                else total_n_features
            )
        else:  # Default = log2(total_n_features)
            # selected_n_features = int(np.log2(total_n_features))
            selected_n_features = int(np.log2(total_n_features))

        selected_features_idx = np.random.choice(
            a=total_n_features, size=selected_n_features, replace=False
        )

        # Iterate over randomly selected features.
        for feature in selected_features_idx:
            # Iterate over all unique thresholds for each random feature.
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                # Split the data into left and right subsets based on the threshold.
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold

                # Skip invalid splits.
                if (
                    sum(left_mask) < self.min_samples_leaf
                    or sum(right_mask) < self.min_samples_leaf
                ):
                    continue

                # Compute IG.
                info_gain = self._information_gain(y, y[left_mask], y[right_mask])

                # Update `best_info_gain` if `info_gain` > `best_info_gain`.
                if info_gain > best_info_gain:
                    best_info_gain = info_gain
                    best_split = {
                        "feature_index": feature,
                        "feature_name": self.feature_names[feature]
                        if self.feature_names is not None
                        else feature,
                        "threshold": threshold,
                    }

        return best_split

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int = 0) -> dict:
        """
        Recursively build a decision tree.

        Args:
            X: Input features.
            y: Target labels.
            depth: Current tree depth.

        Returns:
            Tree structure dictionary.
        """

        # Convert to numpy arrays
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)
        y_np = (
            y.to_numpy().flatten() if hasattr(y, "to_numpy") else np.array(y).flatten()
        )

        # Stopping conditions
        if len(np.unique(y_np)) == 1 or (
            self.max_depth is not None and depth == self.max_depth
        ):
            return {"type": "leaf", "value": np.mean(y)}

        if len(y) < self.min_samples_leaf:
            return {"type": "leaf", "value": np.mean(y)}

        # Find best split
        split = self._best_split(X_np, y_np)
        if not split:
            return {"type": "leaf", "value": np.mean(y)}

        # Apply split
        feature_idx = split["feature_index"]
        left_mask = X_np[:, feature_idx] <= split["threshold"]
        right_mask = X_np[:, feature_idx] > split["threshold"]

        # Recursive tree building
        left_tree = self._build_tree(
            X.iloc[left_mask] if hasattr(X, "iloc") else X[left_mask],
            y.iloc[left_mask] if hasattr(y, "iloc") else y[left_mask],
            depth + 1,
        )
        right_tree = self._build_tree(
            X.iloc[right_mask] if hasattr(X, "iloc") else X[right_mask],
            y.iloc[right_mask] if hasattr(y, "iloc") else y[right_mask],
            depth + 1,
        )

        return {
            "type": "node",
            "feature": split["feature_name"],
            "threshold": split["threshold"],
            "left": left_tree,
            "right": right_tree,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train the random forest on input data.

        Args:
            X: Training features.
            y: Training labels.
        """
        # Store feature names
        if hasattr(X, "columns"):
            self.feature_names = X.columns.tolist()

        # Set random seeds for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)
        seeds = np.random.randint(0, 10000, size=self.n_estimators)

        # Build trees in parallel
        self.forest = Parallel(n_jobs=self.n_jobs)(
            delayed(self._build_single_tree)(X, y, seed) for seed in seeds
        )

    def _build_single_tree(self, X: pd.DataFrame, y: pd.Series, seed: int) -> dict:
        """
        Build a single decision tree with bootstrap sampling.
        """
        X_boot, y_boot = self._bootstrap_sample(X, y, random_state=seed)
        return self._build_tree(X_boot, y_boot)

    def _traverse_tree(self, x: np.ndarray, tree: dict) -> float:
        """
        Traverse a tree to make a prediction for a single sample.

        Args:
            x: Input sample (1D array).
            tree: Decision tree structure.

        Returns:
            Predicted label.
        """
        if tree["type"] == "leaf":
            return tree["value"]

        # Resolve feature index
        if self.feature_names is not None:
            feature_index = self.feature_names.index(tree["feature"])
        else:
            feature_index = tree["feature"]  # Assume integer index

        if x[feature_index] <= tree["threshold"]:
            return self._traverse_tree(x, tree["left"])
        else:
            return self._traverse_tree(x, tree["right"])

    def predict(self, X: pd.DataFrame | NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict labels for input data using majority voting.

        Args:
            X: Input features (DataFrame or array)

        Returns:
            Predicted labels (1D array)
        """
        if self.forest is None:
            raise RuntimeError("Model not trained. Call fit() first.")

        # Convert to numpy array
        X_np = X.to_numpy() if hasattr(X, "to_numpy") else np.array(X)

        # Single sample case
        if len(X_np.shape) == 1:
            return np.mean([self._traverse_tree(X_np, tree) for tree in self.forest])

        # Batch predictions
        all_preds = [
            [self._traverse_tree(x, tree) for x in X_np] for tree in self.forest
        ]
        all_means = np.mean(all_preds, axis=0)
        return all_means
