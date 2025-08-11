import numpy as np
from numpy.typing import NDArray


class HelperFuncs:
    def preprocess_data(
        images: NDArray[np.int64], labels: NDArray[np.int64]
    ) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
        """
        Preprocess image and label data for neural network training.

        Performs:
        1. Image normalisation (0-255 -> 0.0-1.0).
        2. Channel dimension addition.
        3. One-hot encoding of labels.

        Args:
            images: Input image array of shape (n_samples, height, width).
            labels: Integer label array of shape (n_samples,).

        Returns:
            Tuple containing:
            - Processed images: shape (n_samples, height, width, 1).
            - One-hot encoded labels: shape (n_samples, num_classes).
        """
        images = images.astype(float) / 255.0
        images = np.expand_dims(images, axis=-1)  # Add channel dimension
        num_classes = len(np.unique(labels))
        labels = np.eye(num_classes)[labels]
        return images, labels

    def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Compute softmax activation for classification.

        Args:
            x: Input logits of shape (batch_size, num_classes).

        Returns:
            Probability distribution over classes of shape (batch_size, num_classes).
        """
        exp = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def cross_entropy_loss(
        y_pred: NDArray[np.float64], y_true: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Compute categorical cross-entropy loss.

        Args:
            y_pred: Predicted probabilities of shape (batch_size, num_classes).
            y_true: Ground truth one-hot labels of shape (batch_size, num_classes).

        Returns:
            Scalar loss value.
        """
        m = y_true.shape[0]
        log_probs = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)])
        return np.sum(log_probs) / m

    def accuracy(
        y_pred: NDArray[np.int64], y_true: NDArray[np.int64]
    ) -> NDArray[np.float64]:
        """
        Compute classification accuracy.

        Args:
            y_pred: Predicted class probabilities of shape (batch_size, num_classes).
            y_true: Ground truth one-hot labels of shape (batch_size, num_classes).

        Returns:
            Accuracy score between 0.0 and 1.0.
        """
        return np.mean(np.argmax(y_pred, axis=1) == np.argmax(y_true, axis=1))
