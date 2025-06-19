import numpy as np
from numpy.typing import NDArray
import pandas as pd
from typing import Tuple


class HelperFuncs:
    def remove_outliers_iqr(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove outliers from all numerical columns in the dataframe using the IQR method.

        Args:
            df (pd.DataFrame): The input dataframe.

        Returns:
            pd.DataFrame: DataFrame with outliers removed from all numerical columns.
        """
        numerical_columns = df.select_dtypes(
            include=['float64', 'int64']).columns
        for column in numerical_columns:
            Q1 = df[column].quantile(0.25)  # First quartile (25th percentile)
            Q3 = df[column].quantile(0.75)  # Third quartile (75th percentile)
            IQR = Q3 - Q1  # Interquartile range

            lower_bound = Q1 - 1.5 * IQR  # Lower bound for outliers
            upper_bound = Q3 + 1.5 * IQR  # Upper bound for outliers

            # Filter out rows with outliers
            df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

        return df

    def ordinal_encode(data: NDArray[np.str_], levels: NDArray[np.str_]) -> NDArray[np.int64]:
        """
        Perform ordinal encoding for a categorical feature.

        Args:
            data (NDArray[np.str_]): List of categorical values.
            levels (NDArray[np.str_]): Ordered list of unique levels in the desired order.

        Returns:
            NDArray[np.int64]: Ordinally encoded array.
        """
        level_map = {level: i for i, level in enumerate(levels)}
        return np.array([level_map[val] for val in data])

    def standardise(data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Standardise the dataset.

        Args:
            data (NDArray[np.float64]): A 2D NumPy array where each column is a feature.

        Returns:
            NDArray[np.float64]: Standardised dataset with mean 0 and variance 1.
        """
        data_mean = np.mean(data, axis=0)
        data_std = np.std(data, axis=0)
        return (data - data_mean) / data_std

    def normalise(data: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Normalise the dataset to range [0, 1].

        Args:
            data (NDArray[np.float64]): A 2D NumPy array where each column is a feature.

        Returns:
            NDArray[np.float64]: Normalised dataset.
        """
        min_val = np.min(data, axis=0)
        max_val = np.max(data, axis=0)
        return (data - min_val) / (max_val - min_val)

    def train_test_split(X: pd.DataFrame, y: pd.Series, test_size: float = 0.2,
                         random_state: int = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split arrays or matrices into random train and test subsets.

        Args:
            X (pd.DataFrame): Input features, a 2D array with rows (samples) and columns (features).
            y (pd.Series): Target values/labels, a 1D array with rows (samples).
            test_size (float): Proportion of the dataset to include in the test split. Must be between 0.0 and 1.0. default = 0.2
            random_state (int): Seed for the random number generator to ensure reproducible results. default = None

        Returns:
            tuple[NDArray, NDArray, NDArray, NDArray]:
            A tuple containing:
                - X_train (pd.DataFrame): Training set features.
                - X_test (pd.DataFrame): Testing set features.
                - y_train (pd.Series): Training set target values.
                - y_test (pd.Series): Testing set target values.
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
