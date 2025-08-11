import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy.typing import NDArray


class CustomCatBoost:
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 2,
    ) -> None:
        """
        Initialise the CustomCatBoost model with hyperparameters.

        Args:
            n_estimators: Number of boosting iterations.
            learning_rate: Learning rate for updates.
            max_depth: Maximum depth of each tree.
            min_samples_split: Minimum samples required to split a node.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []

    def ordered_target_encode(
        self, df: pd.DataFrame, col: str, target: str
    ) -> pd.Series:
        """
        Vectorised ordered target encoding for a single categorical column.

        Args:
            df: Input DataFrame containing the data.
            col: Name of the categorical column to encode.
            target: Name of the target column.

        Returns:
            Encoded column with ordered target encoding.
        """
        global_mean = df[target].mean()
        cumsum = df.groupby(col)[target].cumsum() - df[target]
        cumcnt = df.groupby(col).cumcount()
        enc = cumsum / cumcnt.replace(0, np.nan)
        enc.fillna(global_mean, inplace=True)
        return enc

    def apply_ordered_target_encoding(
        self, df: pd.DataFrame, cat_cols: list[str], target: str
    ) -> pd.DataFrame:
        """
        Apply ordered target encoding to all categorical columns in a DataFrame.

        Args:
            df: Input DataFrame containing the data.
            cat_cols: List of categorical column names to encode.
            target: Name of the target column.

        Returns:
            DataFrame with encoded categorical columns.
        """
        df_enc = df.copy()
        for col in cat_cols:
            df_enc[col] = self.ordered_target_encode(df_enc, col, target)
        return df_enc

    def find_best_split(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> tuple[int | None, float | None, NDArray[np.bool_], NDArray[np.bool_]]:
        """
        Find the best feature and split value to minimise the weighted variance of the target.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).

        Returns:
                best_feature: Index of the best feature to split on.
                best_split: Value of the best split.
                best_left: Boolean mask for samples going to the left child.
                best_right: Boolean mask for samples going to the right child.
        """
        n_samples, n_features = X.shape
        best_feature, best_split, best_score, best_left, best_right = (
            None,
            None,
            np.inf,
            None,
            None,
        )
        for feature_idx in range(n_features):
            values = np.unique(X[:, feature_idx])
            if len(values) == 1:
                continue
            sorted_vals = np.sort(values)
            splits = (sorted_vals[:-1] + sorted_vals[1:]) / 2
            for split_val in splits:
                left_idx = X[:, feature_idx] <= split_val
                right_idx = ~left_idx
                if not left_idx.any() or not right_idx.any():
                    continue
                score = (
                    np.var(y[left_idx]) * left_idx.sum()
                    + np.var(y[right_idx]) * right_idx.sum()
                )
                if score < best_score:
                    best_feature = feature_idx
                    best_split = split_val
                    best_score = score
                    best_left = left_idx
                    best_right = right_idx
        return best_feature, best_split, best_left, best_right

    def build_tree(
        self, X: NDArray[np.float64], y: NDArray[np.float64], depth: int = 0
    ) -> dict | float:
        """
        Recursively build a decision tree for regression.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target values of shape (n_samples,).
            depth: Current depth of the tree. Defaults to 0.

        Returns:
            A decision tree node represented as a dictionary or a leaf value (float).
        """
        n_samples = X.shape[0]
        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or np.all(y == y[0])
        ):
            return np.mean(y)
        best_feature, best_split, best_left, best_right = self.find_best_split(X, y)
        if best_feature is None:
            return np.mean(y)
        return {
            "feature": best_feature,
            "split": best_split,
            "left": self.build_tree(X[best_left], y[best_left], depth + 1),
            "right": self.build_tree(X[best_right], y[best_right], depth + 1),
        }

    def predict_tree(self, node: dict | float, row: NDArray[np.float64]) -> float:
        """
        Predict the target value for a single sample using the decision tree.

        Args:
            node: Decision tree node or leaf value.
            row: Feature values of the sample.

        Returns:
            Predicted target value.
        """
        while isinstance(node, dict):
            if row[node["feature"]] <= node["split"]:
                node = node["left"]
            else:
                node = node["right"]
        return node

    def fit(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        n_classes: int,
        cat_cols: list[str] | None = None,
        df: pd.DataFrame | None = None,
        target_col: str | None = None,
    ) -> None:
        """
        Fit the gradient boosting model for multi-class classification.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
            n_classes: Number of classes.
            cat_cols: List of categorical columns for ordered target encoding. Defaults to None.
            df: DataFrame containing original data for encoding. Defaults to None.
            target_col: Target column name in DataFrame. Defaults to None.
        """
        if cat_cols is not None and df is not None and target_col is not None:
            df_enc = self.apply_ordered_target_encoding(df, cat_cols, target_col)
            for col in cat_cols:
                if col in df_enc.columns:
                    X[:, df.columns.get_loc(col)] = df_enc[col].values

        N = X.shape[0]
        F = np.zeros((N, n_classes), dtype=np.float64)
        y_onehot = np.eye(n_classes)[y]
        self.trees = []
        for _ in tqdm(range(self.n_estimators)):
            trees_m = []
            P = np.exp(F - F.max(axis=1, keepdims=True))
            P /= P.sum(axis=1, keepdims=True)
            for k in range(n_classes):
                residual = y_onehot[:, k] - P[:, k]
                tree = self.build_tree(X, residual)
                update = np.array([self.predict_tree(tree, row) for row in X])
                F[:, k] += self.learning_rate * update
                trees_m.append(tree)
            self.trees.append(trees_m)

    def predict(
        self, X: NDArray[np.float64], n_classes: int
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Predict class labels and probabilities using the fitted gradient boosting model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            n_classes: Number of classes.

        Returns:
                predicted_labels: Array of predicted class labels.
                predicted_probabilities: Array of predicted class probabilities.
        """
        N = X.shape[0]
        F = np.zeros((N, n_classes), dtype=np.float64)
        for trees_m in self.trees:
            for k, tree in enumerate(trees_m):
                update = np.array([self.predict_tree(tree, row) for row in X])
                F[:, k] += self.learning_rate * update
        P = np.exp(F - F.max(axis=1, keepdims=True))
        P /= P.sum(axis=1, keepdims=True)
        return np.argmax(P, axis=1), P

    def fit_predict(
        self,
        X: NDArray[np.float64],
        y: NDArray[np.int64],
        n_classes: int,
        cat_cols: list[str] | None = None,
        df: pd.DataFrame | None = None,
        target_col: str | None = None,
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        """
        Fit the model and predict class labels and probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Target labels of shape (n_samples,).
            n_classes: Number of classes.
            cat_cols: List of categorical columns for ordered target encoding. Defaults to None.
            df: DataFrame containing original data for encoding. Defaults to None.
            target_col: Target column name in DataFrame. Defaults to None.

        Returns:
                predicted_labels: Array of predicted class labels.
                predicted_probabilities: Array of predicted class probabilities.
        """
        self.fit(X, y, n_classes, cat_cols, df, target_col)
        return self.predict(X, n_classes)
