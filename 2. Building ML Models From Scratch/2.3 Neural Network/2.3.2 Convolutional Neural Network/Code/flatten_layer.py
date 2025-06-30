import numpy as np
from numpy.typing import NDArray


class Flatten:
    """
    Flatten Layer implementation.
    """

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Flatten input tensor to 2D matrix.

        Args:
            input: Input tensor of shape (batch_size, *spatial_dims, channels).

        Returns:
            Flattened output of shape (batch_size, features).
        """
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Reshape gradient to original input shape.

        Args:
            grad: Gradient from next layer of shape (batch_size, features).

        Returns:
            Gradient w.r.t input of original shape (batch_size, *spatial_dims, channels).
        """
        return grad.reshape(self.input_shape)
