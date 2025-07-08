import numpy as np
import pandas as pd
from typing import List, Dict


class CustomMultinomialNB:
    """
    Multinomial Naive Bayes classifier implementation with optimised vector operations.

    Attributes:
        alpha (float): Smoothing parameter (default = 1.0)
        priors_ (Dict[str, float]): Prior probabilities per class
        likelihoods_ (pd.DataFrame): Likelihood probabilities (shape: [n_classes, n_features])
        log_likelihoods_ (np.ndarray): Precomputed log-likelihoods (shape: [n_classes, n_features])
        classes_ (List[str]): Unique class labels
        feature_names_ (pd.Index): Feature names from training data
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialise Multinomial Naive Bayes classifier.

        Args:
            alpha: Smoothing parameter for Laplace smoothing (default = 1.0).
        """
        self.alpha = alpha
        self.priors_ = None
        self.likelihoods_ = None
        self.log_likelihoods_ = None
        self.classes_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train Multinomial Naive Bayes model.

        Args:
            X: Document-term matrix (documents x features).
            y: Target class labels.
        """

        # Convert input to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.feature_names_ = X.columns
        self.classes_ = y.unique().tolist()
        self.priors_ = {cls: 1 / len(self.classes_) for cls in self.classes_}
        self.likelihoods_ = self._calculate_likelihoods(X, y)
        self.log_likelihoods_ = np.log(self.likelihoods_.values)

    def predict(self, X: pd.DataFrame) -> List[str]:
        """
        Predict class labels for documents in X.

        Args:
            X: Document-term matrix to predict.

        Returns:
            Predicted class labels.
        """
        # Align features with training data
        X_aligned = X.reindex(columns=self.feature_names_, fill_value=0)

        # Precompute log priors
        log_priors = np.array([np.log(self.priors_[c]) for c in self.classes_])

        # Vectorised prediction (calculating posteriors here)
        word_contributions = X_aligned @ self.log_likelihoods_.T
        log_posteriors = log_priors + word_contributions
        max_indices = np.argmax(log_posteriors, axis=1)

        return [self.classes_[idx] for idx in max_indices]

    def _calculate_priors(self, y: pd.Series) -> Dict[str, float]:
        """
        Calculate prior probabilities for each class in the target variable.

        Args:
            y: Target variable containing class labels (strings).

        Returns:
            Prior probabilities for each class.
        """
        return y.value_counts(normalize=True).to_dict()

    def _calculate_likelihoods(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compute feature likelihoods (P(x_i|y)) with Laplace smoothing.

        Returns:
            Likelihood DataFrame (classes x features)
        """
        class_totals = X.groupby(y).sum()
        total_words_per_class = class_totals.sum(axis=1)
        vocab_size = len(X.columns)

        numerator = class_totals + self.alpha
        denominator = (
            total_words_per_class.values[:, np.newaxis] + vocab_size * self.alpha
        )
        return numerator / denominator
