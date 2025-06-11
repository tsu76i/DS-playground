import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.spatial.distance import cdist
from tqdm import tqdm
from helper_funcs import HelperFuncs
from regression_metrics import RegressionMetrics


class CustomKNNRegressor:
    """
    A simple K-Nearest Neighbours (KNN) regressor.
    """

    def __init__(self, k: int = 3) -> None:
        """
        Initialise the model with k, the number of neighbours.

        Args: 
            k (int): The number of nearest neighbours to consider for classification. default = 3.
        """
        self.k = k

    def fit(self, X_train: NDArray[np.float64], y_train: NDArray[np.float64]) -> None:
        """
        Fit the training data.

        Args:
            X_train (NDArray[np.float64]): Training features, a 2D array with rows (samples) and columns (features).
            y_train (NDArray[np.float64]): Training labels, a 1D array of labels corresponding to X_train.
        """
        self.X_train = X_train
        self.y_train = y_train

    def calculate_distance(self, x1: float, x2: float) -> float:
        """
        Calculate the Euclidean distance between two points.

        Args:
            x1 (float): First point.
            x2 (float): Second point.

        Returns:
            float: The Euclidean distance between x1 and x2.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X_test: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict the labels for X_test.

        Args:
            X_test (NDArray[np.float64]): Test features, a 2D array with rows (samples) and columns (features).

        Returns:
            NDArray[np.float64]: Predicted target values for each test sample.
        """

        # Ensure input is 2D
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)

        # Compute distances
        # ! Use scipy library for faster computation of Euclidean distance
        distances = cdist(X_test, self.X_train, metric='euclidean')

        # Find indices of k nearest neighbours for each test sample
        idx = np.argpartition(distances, kth=self.k-1, axis=1)[:, :self.k]

        # Gather the corresponding y_train values and compute their mean
        predictions = np.mean(self.y_train[idx], axis=1)
        return predictions


def main():
    # Import data
    df = pd.read_csv(
        '2. Building ML Models From Scratch/_datasets/diamonds.csv')

    # Data Pre-processing
    df = HelperFuncs.remove_outliers_iqr(df)
    categorical_features = {
        'cut': ['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
        'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
        'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']
    }
    for feature, levels in categorical_features.items():
        df[feature] = HelperFuncs.ordinal_encode(df[feature], levels)

    continuous_features = ['carat', 'x', 'y', 'z',
                           'depth', 'table', 'cut', 'color', 'clarity']
    df[continuous_features] = HelperFuncs.standardise(
        df[continuous_features].values)
    df[continuous_features] = HelperFuncs.normalise(
        df[continuous_features].values)

    # Prepare data
    MSE_list_custom, RMSE_list_custom, MAE_list_custom, R2_list_custom = [], [], [], []
    X = df.drop(columns=['price']).values
    y = df['price'].values
    k_range = range(1, 21)  # Generate k-values from 1 to 20

    # Split data
    X_train, X_test, y_train, y_test = HelperFuncs.train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Train KNN regressor, make predictions and evaluate the model for different k values.
    for k in tqdm(k_range):
        knn_reg = CustomKNNRegressor(k=k)
        knn_reg.fit(X_train, y_train)
        y_pred = knn_reg.predict(X_test)

        metrics = RegressionMetrics(y_test, y_pred)
        mse, rmse, mae, r2 = metrics.evaluate()
        print(f"For k = {k}, MSE: {mse}")
        print(f"For k = {k}, RMSE: {rmse}")
        print(f"For k = {k}, MAE: {mae}")
        print(f"For k = {k}, R2: {r2}")
        print("-----------")

    # Prediction of a sample
    x_single_2d = X_test[0]
    y_pred = knn_reg.predict(x_single_2d)
    y_pred_scalar = y_pred[0]
    print("Predicted value:", y_pred_scalar)


if __name__ == '__main__':
    main()
