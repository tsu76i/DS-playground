import numpy as np
from numpy.typing import NDArray


class MaxPooling:
    """
    Max Pooling Layer implementation.
    """

    def __init__(self, pool_size: int) -> None:
        """
        Initialise max pooling layer.

        Args:
            pool_size: Window size for pooling (pool_size x pool_size).
        """
        self.pool_size = pool_size

    def forward(self, input: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform forward pass through max pooling layer.

        Args:
            input: Input tensor of shape (batch_size, height, width, channels).

        Returns:
            Downsampled output of shape (batch_size, out_h, out_w, channels).
        """
        self.input = input
        batch_size, h, w, c = input.shape
        out_h = h // self.pool_size
        out_w = w // self.pool_size
        output = np.zeros((batch_size, out_h, out_w, c))
        self.mask = np.zeros_like(input)

        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i*self.pool_size, (i+1)*self.pool_size
                w_start, w_end = j*self.pool_size, (j+1)*self.pool_size
                region = input[:, h_start:h_end, w_start:w_end, :]
                output[:, i, j, :] = np.max(region, axis=(1, 2))

                # Create mask for backpropagation
                mask_region = (region == output[:, i, j, :][:, None, None, :])
                self.mask[:, h_start:h_end, w_start:w_end, :] = mask_region
        return output

    def backward(self, grad: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Distribute gradients during backpropagation.

        Args:
            grad: Gradient from next layer of shape (batch_size, out_h, out_w, channels).

        Returns:
            Gradient w.r.t input of shape (batch_size, height, width, channels).
        """
        grad_input = np.zeros_like(self.input)
        batch_size, out_h, out_w, c = grad.shape

        for i in range(out_h):
            for j in range(out_w):
                h_start, h_end = i*self.pool_size, (i+1)*self.pool_size
                w_start, w_end = j*self.pool_size, (j+1)*self.pool_size

                # Distribute gradient to max positions
                grad_region = grad[:, i:i+1, j:j+1, :]
                grad_region = np.repeat(grad_region, self.pool_size, axis=1)
                grad_region = np.repeat(grad_region, self.pool_size, axis=2)

                mask_region = self.mask[:, h_start:h_end, w_start:w_end, :]
                grad_input[:, h_start:h_end, w_start:w_end,
                           :] += grad_region * mask_region

        return grad_input
