import numpy as np
from numpy.typing import NDArray


class ReLU:
    """
    Rectified Linear Unit (ReLU) Activation Layer.
    """

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply element-wise ReLU activation: output = max(0, input).

        Args:
            input: Input tensor of any shape.

        Returns:
            Activated output of same shape as input.
        """
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute gradient of ReLU: grad * (input > 0).

        Args:
            grad: Gradient from next layer.

        Returns:
            Gradient w.r.t input of same shape.
        """
        return grad * (self.input > 0)
