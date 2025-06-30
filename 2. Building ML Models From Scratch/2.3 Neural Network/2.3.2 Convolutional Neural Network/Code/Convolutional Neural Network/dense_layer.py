import numpy as np
from numpy.typing import NDArray


class Dense:
    """
    Fully Connected (Dense) Layer implementation.
    """

    def __init__(self, input_size: int, output_size: int) -> None:
        """
        Initialise dense layer parameters.

        Args:
            input_size: Number of input features.
            output_size: Number of output neurons.
        """
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform forward pass: output = input·W + b.

        Args:
            input: Input tensor of shape (batch_size, input_size).

        Returns:
            Output tensor of shape (batch_size, output_size).
        """
        self.input = input
        return np.dot(input, self.weights) + self.biases

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradients during backpropagation.

        Args:
            grad: Gradient from next layer of shape (batch_size, output_size).

        Returns:
            Gradient w.r.t input of shape (batch_size, input_size).
        """
        grad_input = np.dot(grad, self.weights.T)
        self.grad_weights = np.dot(self.input.T, grad)
        self.grad_biases = np.sum(grad, axis=0)
        return grad_input

    def update(self, learning_rate: float) -> None:
        """
        Update layer parameters using gradient descent.

        Args:
            learning_rate: Step size for parameter updates.
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases
