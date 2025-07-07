import numpy as np
from numpy.typing import NDArray


class CustomLassoRegression:
    """
    A simple implementation of Lasso Regression using gradient descent.
    """

    def __init__(self, w: float = 0.0, b: float = 0.0,
                 alpha: float = 0.001, epochs: int = 20000, lambda_: float = 1.0) -> None:
        """
        Initialise the CustomLassoRegression instance with given hyperparameters.

        Args:
            w: Initial weight (default is 0.0).
            b: Initial bias (default is 0.0).
            alpha: Learning rate for gradient descent (default is 0.001).
            epochs: Number of iterations for gradient descent (default is 20000).
            lambda_: L1 Regularisation parameter (default is 1.0).
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

    def calculate_loss_lasso(self, y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """
        Calculate Lasso Loss function (MSE + L1 penalty).

        Args:
            y: True output values.
            y_pred: Predicted output values.

        Returns:
            MSE + L1 penalty.
        """
        mse = np.mean((y - y_pred) ** 2)
        l1_penalty = self.lambda_ * np.abs(self.w)
        return mse + l1_penalty

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

            loss = self.calculate_loss_lasso(y, y_pred)
            self.loss_history.append(loss)

            dL_dw = -(2 / n) * np.sum(X * (y - y_pred)) + \
                self.lambda_ * np.sign(self.w)
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
