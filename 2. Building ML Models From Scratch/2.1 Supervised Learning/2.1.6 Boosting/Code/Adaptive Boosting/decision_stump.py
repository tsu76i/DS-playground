import numpy as np
from numpy.typing import NDArray
from typing import Optional


class DecisionStump:
    """
    A simple decision stump (one-level decision tree) used as a weak learner.

    Attributes:
        polarity: The direction of the inequality for the split.
        feature_index: The index of the feature used for splitting.
        threshold: The threshold value for the split.
        alpha: The weight of this stump in the ensemble.
    """

    def __init__(self) -> None:
        """
        Initialise the decision stump with default values.
        """
        self.polarity: int = 1
        self.feature_index: Optional[int] = None
        self.threshold: Optional[float] = None
        self.alpha: Optional[float] = None

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        """
        Predicts class labels for samples in X using the decision stump.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels (+1 or -1) of shape (n_samples,).
        """
        n_samples = X.shape[0]
        feature_column = X[:, self.feature_index]
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            predictions[feature_column < self.threshold] = -1
        else:
            predictions[feature_column > self.threshold] = -1
        return predictions
