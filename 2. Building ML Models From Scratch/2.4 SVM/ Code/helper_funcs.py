from typing import Tuple
import numpy as np
from numpy.typing import NDArray


class HelperFuncs:
    def train_test_split(X: NDArray, y: NDArray, test_size: float = 0.2,
                         random_state: int = None) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Split arrays or matrices into random train and test subsets.

        Args:
            X (NDArray): Input features, a 2D array with rows (samples) and columns (features).
            y (NDArray): Target values/labels, a 1D array with rows (samples).
            test_size (float): Proportion of the dataset to include in the test split. Must be between 0.0 and 1.0. default = 0.2
            random_state (int): Seed for the random number generator to ensure reproducible results. default = None

        Returns:
            Tuple[NDArray, NDArray, NDArray, NDArray]:
            A tuple containing:
                - X_train (NDArray): Training set features.
                - X_test (NDArray): Testing set features.
                - y_train (NDArray): Training set target values.
                - y_test (NDArray): Testing set target values.
        """
        # Set a random seed if it exists
        if random_state:
            np.random.seed(random_state)

        # Create a list of numbers from 0 to len(X)
        indices = np.arange(len(X))

        # Shuffle the indices
        np.random.shuffle(indices)

        # Define the size of our test data from len(X)
        test_size = int(test_size * len(X))

        # Generate indices for test and train data
        test_indices: NDArray[np.int64] = indices[:test_size]
        train_indices: NDArray[np.int64] = indices[test_size:]

        # Return: X_train, X_test, y_train, y_test
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
