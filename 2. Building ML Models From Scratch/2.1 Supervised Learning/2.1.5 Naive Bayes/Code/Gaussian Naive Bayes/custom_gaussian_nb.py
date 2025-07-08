import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class CustomGaussianNB:
    """
    Gaussian Naive Bayes classifier implementation.

    Attributes:
        epsilon (float): Smoothing parameter for variance
        priors_ (Dict[str, float]): Prior probabilities per class
        means_ (pd.DataFrame): Feature means per class
        variances_ (pd.DataFrame): Feature variances per class
        classes_ (List[str]): Unique class labels
        feature_names_ (pd.Index): Feature names from training data
    """

    def __init__(self, epsilon: float = 1e-9) -> None:
        """
        Initialise Gaussian Naive Bayes classifier.

        Args:
            epsilon: Smoothing parameter for variance (default = 1e-9)
        """
        self.epsilon = epsilon
        self.priors_ = None
        self.means_ = None
        self.variances_ = None
        self.classes_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train Gaussian Naive Bayes model.

        Args:
            X: Feature matrix.
            y: Target class labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_ = X.columns
        self.classes_ = y.unique().tolist()
        self.priors_ = self._calculate_priors(y)
        self.means_, self.variances_ = self._calculate_params(X, y)

    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict class labels for input samples.

        Args:
            X: Feature matrix to predict.

        Returns:
            Predicted class labels.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.feature_names_)
        else:
            X = X.reindex(columns=self.feature_names_, fill_value=0)

        predictions = []
        for i in range(len(X)):
            sample = X.iloc[i]
            log_posteriors = self._calculate_log_posteriors(sample)
            predictions.append(max(log_posteriors, key=log_posteriors.get))
        return predictions

    def _calculate_priors(self, y: pd.Series) -> Dict[str, float]:
        """
        Calculate prior probabilities for each class.

        Args:
            y: Target class labels.

        Returns:
            Prior probabilities for each class.
        """
        return y.value_counts(normalize=True).to_dict()

    def _calculate_params(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compute Gaussian parameters (mean and variance) per class.

        Args:
            X: Feature matrix.
            y: Target class labels.

        Returns:
            Tuple of (means, variances) DataFrames.
        """
        means = X.groupby(y).mean()
        variances = X.groupby(y).var(ddof=0) + self.epsilon
        return means, variances

    def _calculate_log_posteriors(self, sample: pd.Series) -> Dict[str, float]:
        """
        Calculate log-posterior probabilities for a single sample.

        Args:
            sample: Feature vector of a single sample

        Returns:
            Dictionary of log-posterior probabilities per class
        """
        log_posteriors = {}

        for cls in self.classes_:
            # Start with log prior
            log_posterior = np.log(self.priors_[cls])

            # Vectorised log-likelihood calculation
            mean_vec = self.means_.loc[cls].values
            var_vec = self.variances_.loc[cls].values
            x_vec = sample.values

            # Gaussian log PDF: -1/2*[log(2πσ²) + (x-μ)²/σ²]
            log_pdf = (
                -1
                / 2
                * (np.log(2 * np.pi * var_vec) + ((x_vec - mean_vec) ** 2) / var_vec)
            )
            log_posterior += np.sum(log_pdf)

            log_posteriors[cls] = log_posterior

        return log_posteriors
