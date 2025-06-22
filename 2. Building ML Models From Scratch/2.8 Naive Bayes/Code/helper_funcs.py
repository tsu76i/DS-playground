import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import Tuple, List


class HelperFuncs:
    def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                         random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split arrays or matrices into random train and test subsets.

        Args:
            X: Input features, DataFrame with rows (samples) and columns (features).
            y: Target values/labels, Series with rows (samples).
            test_size: Proportion of the dataset to include in the test split. Must be between 0.0 and 1.0. default = 0.2
            random_state: Seed for the random number generator to ensure reproducible results. default = None

        Returns:
            A tuple containing:
                - X_train: Training set features.
                - X_test: Testing set features.
                - y_train: Training set target values.
                - y_test: Testing set target values.
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
        return X.iloc[train_indices], X.iloc[test_indices], y.iloc[train_indices], y.iloc[test_indices]

    def clean_text(text: str) -> str:
        """
        Clean and preprocess email text for spam detection.

        Args:
            Raw email text.

        Returns:
            Cleaned and preprocessed text.
        """
        return text.lower()
