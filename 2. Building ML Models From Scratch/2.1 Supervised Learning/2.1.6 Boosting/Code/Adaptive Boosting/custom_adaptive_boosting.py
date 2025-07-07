import numpy as np
from numpy.typing import NDArray
from decision_stump import DecisionStump


class CustomAdaBoost:
    """
    AdaBoost ensemble classifier using decision stumps.

    Attributes:
        n_weak_learners: Number of weak learners (decision stumps) to use.
        classifiers: List of fitted decision stumps.
    """

    def __init__(self, n_weak_learners: int = 5) -> None:
        """
        Initialise the AdaBoost classifier.

        Args:
            n_weak_learners: Number of weak learners (decision stumps) to use. Defaults to 5.
        """
        self.n_weak_learners = n_weak_learners
        self.classifiers = []

    def fit(self, X: NDArray[np.float64], y: NDArray[np.int8]) -> None:
        """
        Fit the AdaBoost classifier on the training data.

        Args:
            X: Training feature matrix of shape (n_samples, n_features).
            y: Training labels (+1 or -1) of shape (n_samples,).
        """
        n_samples, n_features = X.shape
        # Initialise weights to 1/N
        sample_weights = np.full(n_samples, 1 / n_samples)
        self.classifiers = []

        for _ in range(self.n_weak_learners):
            stump = DecisionStump()
            min_error = float('inf')

            # Find the best decision stump
            for feature_index in range(n_features):
                feature_column = X[:, feature_index]
                thresholds = np.unique(feature_column)
                for threshold in thresholds:
                    polarity = 1
                    predictions = np.ones(n_samples)
                    predictions[feature_column < threshold] = -1

                    # Calculate weighted error
                    error = np.sum(sample_weights[y != predictions])

                    # If error > 0.5, flip polarity
                    if error > 0.5:
                        error = 1 - error
                        polarity = -1

                    if error < min_error:
                        stump.polarity = polarity
                        stump.threshold = threshold
                        stump.feature_index = feature_index
                        min_error = error

            # Compute alpha (learner weight)
            c = 1e-10  # to avoid division by zero
            stump.alpha = 0.5 * np.log((1.0 - min_error + c) / (min_error + c))

            # Update weights
            predictions = stump.predict(X)
            sample_weights *= np.exp(-stump.alpha * y * predictions)
            sample_weights /= np.sum(sample_weights)  # Normalise

            self.classifiers.append(stump)

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int8]:
        """
        Predict class labels for samples in X using the trained AdaBoost ensemble.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted class labels (+1 or -1) of shape (n_samples,).
        """
        weighted_preds = [clf.alpha *
                          clf.predict(X) for clf in self.classifiers]
        y_pred = np.sum(weighted_preds, axis=0)
        return np.sign(y_pred)
