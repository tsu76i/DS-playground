import numpy as np
from numpy.typing import NDArray
from convolutional_layer import Conv2D
from relu import ReLU
from max_pooling_layer import MaxPooling
from flatten_layer import Flatten
from dense_layer import Dense


class ConvNet:
    """
    Convolutional Neural Network architecture for image classification.
    """

    def __init__(self):
        """
        Initialise CNN layers with fixed architecture.
        """
        self.conv1 = Conv2D(num_filters=8, filter_size=3, input_shape=(28, 28, 1))
        self.relu1 = ReLU()
        self.pool1 = MaxPooling(pool_size=2)
        self.flatten = Flatten()
        self.dense1 = Dense(input_size=13 * 13 * 8, output_size=128)
        self.relu2 = ReLU()
        self.dense2 = Dense(input_size=128, output_size=10)
        self.layers = [
            self.conv1,
            self.relu1,
            self.pool1,
            self.flatten,
            self.dense1,
            self.relu2,
            self.dense2,
        ]

    def forward(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Perform forward pass through all layers.

        Args:
            x: Input image tensor of shape (batch_size, 28, 28, 1).

        Returns:
            Output logits of shape (batch_size, 10).
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: NDArray[np.float64]) -> None:
        """
        Backpropagate gradients through all layers.

        Args:
            grad: Gradient from loss function.
        """
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_params(self, learning_rate: float) -> None:
        """
        Update all trainable parameters in the network.

        Args:
            learning_rate: Step size for parameter updates.
        """
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(learning_rate)
