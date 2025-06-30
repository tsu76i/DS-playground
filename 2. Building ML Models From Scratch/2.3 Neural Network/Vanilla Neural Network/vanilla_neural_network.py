import numpy as np
from numpy.typing import NDArray


class VanillaNeuralNetwork:
    """
    A two-layer neural network implementation.

    This neural network consists of:
    - Input layer.
    - One hidden layer with sigmoid activation.
    - Output layer with sigmoid activation.
    - Trained using backpropagation with MSE loss.

    Attributes:
        W1 (NDArray[np.float64]): Weight matrix between input and hidden layer.
        b1 (NDArray[np.float64]): Bias vector for hidden layer.
        W2 (NDArray[np.float64]): Weight matrix between hidden and output layer.
        b2 (NDArray[np.float64]): Bias vector for output layer.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        """
        Initialise weights and biases for the neural network.

        Args:
            input_size: Number of input features.
            hidden_size: Number of neurons in hidden layer.
            output_size: Number of neurons in output layer.

        Initialisation:
            Weights: Random values from standard normal distribution.
            Biases: Initialised to zeros.
        """
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, Z: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute sigmoid activation function.

        Args:
            Z: Input tensor (linear transformation).

        Returns:
            Sigmoid activation of input, range [0, 1].
        """
        return 1 / (1 + np.exp(-Z))

    def sigmoid_derivative(self, a: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute derivative of sigmoid function.

        Note: This function expects the activation output (a), not the raw input (Z).

        Args:
            a: Output from sigmoid activation (a = σ(Z)).

        Returns:
            Derivative of sigmoid: a * (1 - a).
        """

        return a * (1 - a)

    def forward(self, X: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Perform forward propagation through the network.

        Computes:
            hidden_input = X·W1 + b1
            hidden_output = σ(hidden_input)
            output_input = hidden_output·W2 + b2
            prediction = σ(output_input)

        Stores intermediate values for use in backpropagation.

        Args:
            X: Input data, shape (n_samples, input_size)

        Returns:
            Final predictions, shape (n_samples, output_size)
        """
        self.hidden_input = np.dot(X, self.W1) + self.b1
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.W2) + self.b2
        self.prediction = self.sigmoid(self.output_input)
        return self.prediction

    def backward(self, X: NDArray[np.int64], y: NDArray[np.float64], lr: float = 0.01):
        """
        Perform backpropagation and update weights & biases.

        Computes gradients using chain rule and updates parameters:
            1. Calculate output layer error.
            2. Calculate hidden layer error.
            3. Update weights and biases using gradient descent.

        Args:
            X: Input data, shape (n_samples, input_size).
            y: Target values, shape (n_samples, output_size).
            lr: Learning rate for gradient descent.
        """

        # Output layer error
        output_error = y - self.prediction
        output_delta = output_error * self.sigmoid_derivative(self.prediction)

        # Hidden layer error
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * \
            self.sigmoid_derivative(self.hidden_output)

        # Update weights and biases
        self.W2 += self.hidden_output.T.dot(output_delta) * lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True) * lr
        self.W1 += X.T.dot(hidden_delta) * lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * lr

    def mse(self, y: NDArray[np.float64], y_hat: NDArray[np.float64]) -> float:
        """
        Compute mean squared error loss.

        Args:
            y_true: Ground truth values
            y_pred: Predicted values

        Returns:
            Mean squared error
        """

        return np.mean((y - y_hat) ** 2)

    def train(self, X: NDArray[np.int64], y: NDArray[np.int64], epochs: int, lr: float) -> None:
        """
        Train the neural network using backpropagation for a specified number of epochs.

        Performs iterative training by:
        1. Forward propagating inputs to generate predictions.
        2. Calculating loss between predictions and true labels.
        3. Backpropagating errors to compute gradients.
        4. Updating weights and biases using gradient descent.

        Training progress is printed at the start, at 10% interval epochs, and at completion.

        Args:
            X: Input feature matrix of shape (n_samples, input_size).
            y: Target labels of shape (n_samples, output_size).
                Should be one-hot encoded for classification tasks.
            epochs: Number of complete passes through the training data.
            lr: Learning rate (step size) for gradient descent updates.
        """
        for epoch in range(epochs):
            y_pred = self.forward(X)
            self.backward(X, y, lr)
            loss = self.mse(y, y_pred)
            if epoch + 1 == 1 or (epoch + 1) % (epochs//10) == 0 or epoch + 1 == epochs:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")
