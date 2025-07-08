import numpy as np
import pandas as pd
from typing import Tuple
from numpy.typing import NDArray
from scipy.spatial.distance import cdist


class CustomKNNClassifier:
    """
    A simple K-Nearest Neighbours (KNN) classifier.
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

    def majority_vote(self, labels_row: NDArray[np.str_ | np.int64]) -> str | int:
        """
        Determines the most frequent label in an array of labels.

        Args:
            labels_row: A 1D array containing the labels of the k nearest neighbours.

        Returns:
            The most frequent label in the input array.
        """
        unique_labels, counts = np.unique(labels_row, return_counts=True)
        return unique_labels[np.argmax(counts)]  # Most frequent labels

    def predict(
        self, X_test: pd.DataFrame
    ) -> Tuple[str, NDArray[np.int64], NDArray[np.str_]] | NDArray[np.str_]:
        """
        Predicts the labels for the given test data.

        Args:
            Test features, either a single sample (1D array) or multiple samples (2D array).

        Returns:
            If X_test is a single sample (1D array):
                - The most frequent label among k nearest neighbours.
                - Indices of the k nearest neighbours.
                - Labels of the k nearest neighbours.

            If X_test is multiple samples (2D array):
                Predicted labels for all test samples.
        """
        is_single_sample = X_test.ndim == 1

        # Reshape if X is a single sample
        X_test = X_test.reshape(1, -1) if is_single_sample else X_test

        # Calculate Euclidean distances
        distances = cdist(X_test, self.X_train, metric="euclidean")

        # Identify the indices of k-neighbours
        k_neighbours_idx = np.argpartition(distances, kth=self.k - 1, axis=1)[
            :, : self.k
        ]

        # Identify the labels of k-neighbours
        k_neighbours_labels = self.y_train[k_neighbours_idx]

        if is_single_sample:
            most_common = self.majority_vote(k_neighbours_labels.flatten())
            return (
                most_common,
                k_neighbours_idx.flatten(),
                k_neighbours_labels.flatten(),
            )

        predictions = np.array(
            [self.majority_vote(labels) for labels in k_neighbours_labels]
        )
        return predictions
