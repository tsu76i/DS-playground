import numpy as np
from numpy.typing import NDArray


class CustomRidgeRegression:
    """
    A simple implementation of Ridge Regression using gradient descent.
    """

    def __init__(self, w: float = 0.0, b: float = 0.0,
                 alpha: float = 0.001, epochs: int = 20000, lambda_: float = 1.0) -> None:
        """
        Initialise the CustomRidgeRegression instance with given hyperparameters.

        Args:
            w: Initial weight (default is 0.0).
            b: Initial bias (default is 0.0).
            alpha: Learning rate for gradient descent (default is 0.001).
            epochs: Number of iterations for gradient descent (default is 20000).
            lambda_: L2 Regularisation parameter (default is 1.0).
        """
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs
        self.lambda_ = lambda_
        self.loss_history = []

    def predict(self, X: float) -> float:
        """
        Predict the output for a given input using the regression line.

        Args:
            X: Input feature value.

        Returns:
            Predicted value.
        """
        return self.w * X + self.b

    def calculate_loss_ridge(self, y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """
        Calculate Ridge Loss function (MSE + L2 penalty).

        Args:
            y: True output values.
            y_pred: Predicted output values.

        Returns:
            MSE + L2 penalty.
        """
        mse = np.mean((y - y_pred) ** 2)
        l2_penalty = self.lambda_ * (self.w ** 2)
        return mse + l2_penalty

    def gradient_descent(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Perform gradient descent to optimise the parameters w and b.

        Args:
            X: Input feature values.
            y: True output values.
        """
        n: int = len(y)
        for _ in range(self.epochs):
            y_pred = np.array([self.predict(x_i) for x_i in X])

            loss = self.calculate_loss_ridge(y, y_pred)
            self.loss_history.append(loss)

            dL_dw = -(2 / n) * np.sum(X * (y - y_pred)) + \
                2 * self.lambda_ * self.w
            dL_db = -(2 / n) * np.sum(y - y_pred)

            self.w -= self.alpha * dL_dw
            self.b -= self.alpha * dL_db

    def train(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Input feature values.
            y: True output values.
        """
        self.gradient_descent(X, y)
        print(
            f"Training completed. Coefficient: {self.w:.5f}, Intercept: {self.b:.5f}")
