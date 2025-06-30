import numpy as np
import matplotlib.pyplot as plt
from numpy.typing import NDArray
from tqdm import tqdm


class CustomLogisticRegression:
    """
    Logistic Regression model using gradient descent.
    """

    def __init__(self, W: NDArray[np.float64], alpha: float = 0.01,
                 epochs: int = 5000, threshold: float = 0.5) -> None:
        """
        Initialise the model with weights given hyperparameters.

        Args:
        W: Initial weights, including bias term.
        alpha: Learning rate. Default is 0.01
        epochs: Number of iterations for gradient descent. Default is 5000.
        threshold: Classification threshold for predicting labels. Default is 0.5.
        """
        self.W = W
        self.alpha = alpha
        self.threshold = threshold
        self.epochs = epochs
        self.loss_history = []

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the sigmoid function.

        Args:
            z: Input array, a linear combination of X(features) and W(weights).

        Returns:
            Output after applying the sigmoid function.
        """
        return 1 / (1 + np.exp(-z))

    def hypothesis(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute the hypothesis function for logistic regression.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities.
        """
        return self.sigmoid(np.dot(X, self.W))

    def calculate_loss_BCE(self, y: NDArray[np.int64], y_pred: NDArray[np.float64]) -> float:
        """
        Calculate the Binary Cross-Entropy (BCE) loss.

        Args:
            y: True labels.
            y_pred: Predicted probabilities.

        Returns:
            BCE loss.
        """
        return - (1 / len(y)) * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    def gradient_descent(self, X: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """
        Perform gradient descent to optimize weights.

        Args:
            X: Feature matrix.
            y: True labels.
        """
        for epoch in tqdm(range(self.epochs)):
            y_pred: NDArray[np.float64] = self.hypothesis(X)
            loss: float = self.calculate_loss_BCE(y, y_pred)
            self.loss_history.append(loss)
            dL_dW: NDArray[np.float64] = (1/len(y)) * np.dot(X.T, (y_pred - y))
            self.W -= self.alpha * dL_dW

    def train(self, x: NDArray[np.float64], y: NDArray[np.int64]) -> None:
        """
        Train the model using gradient descent.

        Args:
            X: Feature matrix.
            y: True labels.
        """
        self.gradient_descent(x, y)
        print(
            f"Training completed. Optimised weights:\n {self.W}")

    def calculate_accuracy(self, y: NDArray[np.int64], y_pred: NDArray[np.int64]) -> float:
        """
        Calculate the accuracy of the model.

        Args:
            y_pred: Predicted labels.
            y: True labels.

        Returns:
            Accuracy percentage.
        """
        return np.mean(y == y_pred) * 100

    def predict_proba(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Predict probabilities for the input data.

        Args:
            X: Feature matrix.

        Returns:
            Predicted probabilities.
        """
        return self.hypothesis(X)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Predict class labels based on the threshold.

        Args:
            X: Feature matrix.

        Returns:
            Predicted labels (0 or 1).
        """
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def plot_loss_history(self) -> None:
        """
        Plot the loss history for the training process.
        """
        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_history[:20], marker='o', color='b')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Over Epochs (Up to 20)')
        plt.grid(True)
        plt.show()

    def draw_decision_boundary(self, x_1: NDArray[np.float64], x_2: NDArray[np.float64],
                               y: NDArray[np.int64], sample_data: NDArray[np.float64] = None) -> None:
        """
        Visualise the decision boundary of the logistic regression model.

        Args:
            x_1: Feature 1 values.
            x_2: Feature 2 values.
            y: True labels.
            sample_data: A single sample for visualisation. Default is None.
        """
        decision_boundary = -(self.W[0] + self.W[1] * x_1) / self.W[2]
        plt.figure(figsize=(8, 6))
        plt.scatter(x=x_1[y.flatten() == 0], y=x_2[y.flatten()
                    == 0], label='0', color='orange')
        plt.scatter(x=x_1[y.flatten() == 1], y=x_2[y.flatten()
                    == 1], label='1', color='green')
        plt.plot(x_1, decision_boundary, color="red",
                 label="Decision Boundary")

        if sample_data is not None:
            test_x1, test_x2 = sample_data[0, 1], sample_data[0, 2]
            test_label = self.predict(sample_data)
            color = 'blue' if test_label == 0 else 'purple'
            plt.scatter(test_x1, test_x2, label='Test Point',
                        color=color, edgecolor="black", zorder=2, s=100)

        plt.xlabel('Feature 1 (x1)')
        plt.ylabel('Feature 2 (x2)')
        plt.title('Logistic Regression Decision Boundary')
        plt.legend()
        plt.tight_layout()
        plt.show()


