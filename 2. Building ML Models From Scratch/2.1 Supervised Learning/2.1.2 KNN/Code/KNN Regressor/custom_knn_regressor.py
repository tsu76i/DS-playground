import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


class CustomKNNRegressor:
    """
    A simple K-Nearest Neighbours (KNN) regressor.
    """

    def __init__(self, k: int = 3) -> None:
        """
        Initialise the model with k, the number of neighbours.

        Args:
            k: The number of nearest neighbours to consider for classification. default = 3.
        """
        self.k = k

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Fit the training data.

        Args:
            X_train: Training features, pd.DataFrame with rows (samples) and columns (features).
            y_train: Training labels, pd.Series of labels corresponding to X_train.
        """
        self.X_train = X_train.values
        self.y_train = y_train.values

    def predict(self, X_test: pd.DataFrame) -> NDArray[np.float64]:
        """
        Predict the labels for X_test.

        Args:
            X_test: Test features, a 2D array with rows (samples) and columns (features).

        Returns:
            Predicted target values for each test sample.
        """

        # Ensure input is 2D
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # Compute distances
        # ! Use scipy library for faster computation of Euclidean distance
        distances = cdist(X_test, self.X_train, metric="euclidean")

        # Find indices of k nearest neighbours for each test sample
        idx = np.argpartition(distances, kth=self.k - 1, axis=1)[:, : self.k]

        # Gather the corresponding y_train values and compute their mean
        predictions = np.mean(self.y_train[idx], axis=1)
        return predictions
