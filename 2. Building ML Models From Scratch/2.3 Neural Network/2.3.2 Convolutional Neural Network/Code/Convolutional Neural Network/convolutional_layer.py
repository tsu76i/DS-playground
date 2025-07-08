import numpy as np
from numpy.typing import NDArray


class Conv2D:
    """
    2D Convolutional Layer implementation.
    """

    def __init__(self, num_filters: int, filter_size: int, input_shape: int) -> None:
        """
        Initialise convolutional layer parameters.

        Args:
            num_filters: Number of filters/kernels in layer
            filter_size: Spatial dimensions of filters (filter_size x filter_size)
            input_shape: Shape of input tensor (height, width, channels)
        """
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.input_shape = input_shape
        self.filters = (
            np.random.randn(num_filters, filter_size, filter_size, input_shape[-1])
            * 0.1
        )
        self.biases = np.zeros(num_filters)

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform forward pass through convolutional layer.

        Computes:
            output[b, i, j, f] = ∑∑∑ (input_region * filter) + bias.

        Args:
            input: Input tensor of shape (batch_size, height, width, in_channels).

        Returns:
            Output feature maps of shape (batch_size, out_height, out_width, num_filters).
        """
        self.input = input
        batch_size, in_h, in_w, in_c = input.shape
        out_h = in_h - self.filter_size + 1
        out_w = in_w - self.filter_size + 1
        self.output = np.zeros((batch_size, out_h, out_w, self.num_filters))

        for i in range(out_h):
            for j in range(out_w):
                region = input[:, i : i + self.filter_size, j : j + self.filter_size, :]
                self.output[:, i, j, :] = (
                    np.tensordot(region, self.filters, axes=([1, 2, 3], [1, 2, 3]))
                    + self.biases
                )
        return self.output

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradients during backpropagation.

        Calculates:
        1. Gradient w.r.t filters.
        2. Gradient w.r.t biases.
        3. Gradient w.r.t input (using full convolution).

        Args:
            grad: Gradient from next layer of shape (batch_size, out_h, out_w, num_filters).

        Returns:
            Gradient w.r.t input of shape (batch_size, in_h, in_w, in_channels).
        """
        batch_size, out_h, out_w, num_filters = grad.shape
        in_h, in_w = self.input.shape[1], self.input.shape[2]

        # Gradient w.r.t filters
        grad_filters = np.zeros_like(self.filters)
        for f in range(num_filters):
            for i in range(out_h):
                for j in range(out_w):
                    region = self.input[
                        :, i : i + self.filter_size, j : j + self.filter_size, :
                    ]
                    grad_filters[f] += np.sum(
                        region * grad[:, i, j, f][:, None, None, None], axis=0
                    )

        # Gradient w.r.t biases
        grad_biases = np.sum(grad, axis=(0, 1, 2))

        # Gradient w.r.t input (vectorised implementation)
        grad_input = np.zeros_like(self.input)
        padded_grad = np.pad(
            grad,
            (
                (0, 0),
                (self.filter_size - 1, self.filter_size - 1),
                (self.filter_size - 1, self.filter_size - 1),
                (0, 0),
            ),
        )
        flipped_filters = np.flip(self.filters, axis=(1, 2)).transpose(1, 2, 0, 3)

        for i in range(in_h):
            for j in range(in_w):
                region = padded_grad[
                    :, i : i + self.filter_size, j : j + self.filter_size, :
                ]
                # Vectorised computation
                grad_input[:, i, j, :] = np.sum(
                    region[:, :, :, :, None] * flipped_filters[None, :, :, :, :],
                    axis=(1, 2, 3),
                )

        self.grad_filters = grad_filters
        self.grad_biases = grad_biases
        return grad_input

    def update(self, learning_rate: float) -> None:
        """
        Update layer parameters using gradient descent.

        Args:
            learning_rate: Step size for parameter updates.
        """
        self.filters -= learning_rate * self.grad_filters
        self.biases -= learning_rate * self.grad_biases
