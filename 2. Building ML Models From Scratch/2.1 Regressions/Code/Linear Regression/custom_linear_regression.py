import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numpy.typing import NDArray


class CustomLinearRegression:
    """
    A simple implementation of Linear Regression using gradient descent.
    """

    def __init__(self, w: float = 0.0, b: float = 0.0,
                 alpha: float = 0.01, epochs: int = 5000) -> None:
        """
        Initialise the CustomLinearRegression instance with given hyperparameters.

        Args:
            w: Initial weight (default is 0.0).
            b: Initial bias (default is 0.0).
            alpha: Learning rate for gradient descent (default is 0.01).
            epochs: Number of iterations for gradient descent (default is 5000).
        """
        self.w = w
        self.b = b
        self.alpha = alpha
        self.epochs = epochs
        self.loss_history = []

    def predict(self, x: float) -> float:
        """
        Predict the output for a given input using the regression line.

        Args:
            x: Input feature value.

        Returns:
            Predicted value.
        """
        return self.w * x + self.b

    def calculate_loss_MSE(self, y: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """
        Calculate the Mean Squared Error (MSE) loss.

        Args:
            y: True output values.
            y_pred: Predicted output values.

        Returns:
            Mean Squared Error.
        """
        return np.mean((y - y_pred) ** 2)

    def gradient_descent(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Perform gradient descent to optimise the regression parameters.

        Args:
            x: Input feature values.
            y: True output values.
        """
        n: int = len(y)
        for _ in tqdm(range(self.epochs)):
            y_pred = np.array([self.predict(x_i) for x_i in x])

            loss = self.calculate_loss_MSE(y, y_pred)
            self.loss_history.append(loss)

            dL_dw = -(2 / n) * np.sum(x * (y - y_pred))
            dL_db = -(2 / n) * np.sum(y - y_pred)

            self.w -= self.alpha * dL_dw
            self.b -= self.alpha * dL_db

    def train(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        """
        Train the model using gradient descent.

        Args:
            x: Input feature values.
            y: True output values.
        """
        self.gradient_descent(x, y)
        print(
            f"Training completed. Coefficient: {self.w:.5f}, Intercept: {self.b:.5f}")

    def plot_loss_history(self) -> None:
        """
        Plot the training loss over epochs.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history[:20], marker='o', color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs (Up to 20)')
        plt.grid(True)
        plt.show()

    def plot_prediction(self, X: NDArray[np.float64], y: NDArray[np.float64], x_test: float,
                        y_pred_single: float) -> None:
        """
        Plot the regression line, training data, and test data point.

        Args:
            X: Input feature values for training data.
            y: Target values for training data.
            x_test: Test input value.
            y_pred_single: Predicted output for the test input.
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(X, y)
        plt.plot(X, self.predict(X), color="red", label="Predicted Values")
        plt.scatter(x_test, y_pred_single, color="orange",
                    edgecolor="black", label="Test Point", zorder=2)
        plt.title('Linear Prediction')
        plt.xlabel("X")
        plt.ylabel("y")
        plt.legend()
        plt.show()
