import numpy as np
import pandas as pd
from numpy.typing import NDArray


class RegressionMetrics:
    """
    A class to compute and evaluate regression metrics (MSE, RMSE, MAE, and R-Squared).
    """

    def __init__(self, y_true: pd.Series, y_pred: NDArray[np.float64]):
        """
        Initialise the RegressionMetrics instance.

        Args:
            y_true: True values of the dataset.
            y_pred: Predicted values of the dataset.
        """
        self.y_true = y_true
        self.y_pred = y_pred

    def calculate_MSE(self) -> float:
        return np.mean((self.y_true - self.y_pred) ** 2)

    def calculate_RMSE(self) -> float:
        return np.sqrt(np.mean((self.y_true - self.y_pred) ** 2))

    def calculate_MAE(self) -> float:
        return np.mean(np.abs(self.y_true - self.y_pred))

    def calculate_r2(self) -> float:
        ss_total = np.sum((self.y_true - np.mean(self.y_true)) ** 2)
        ss_residual = np.sum((self.y_true - self.y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def evaluate(self) -> tuple[float, float, float, float]:
        """
        Calculate and return evaluation metrics for a regression model, including MSE, RMSE, MAE, and R-squared.

        Returns:
            - mse: Mean Squared Error (MSE), indicating the average of the squared differences between predicted and true values.
            - rmse: Root Mean Squared Error (RMSE), indicating the standard deviation of the residuals.
            - mae: Mean Absolute Error (MAE), representing the average absolute difference between predicted and true values.
            - r2: R-squared (coefficient of determination), showing the proportion of variance in the dependent variable that is predictable from the independent variable(s).
        """
        mse = self.calculate_MSE()
        rmse = self.calculate_RMSE()
        mae = self.calculate_MAE()
        r2 = self.calculate_r2()
        return mse, rmse, mae, r2
