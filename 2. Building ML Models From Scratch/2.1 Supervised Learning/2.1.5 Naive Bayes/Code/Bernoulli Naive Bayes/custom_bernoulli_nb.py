import numpy as np
import pandas as pd


class CustomBernoulliNB:
    """
    Bernoulli Naive Bayes classifier for boolean features.

    This implementation handles boolean features after label encoding.
    Uses Laplace smoothing to handle underflow.

    Attributes:
        alpha (float): Smoothing parameter (default=1.0).
        priors_ (dict[int, float]): Class prior probabilities.
        likelihoods_ (dict[str, dict[int, dict[int, float]]]): Feature likelihood probabilities.
        classes_ (NDArray[np.int64]): Unique class labels.
        feature_names_ (list[str]): Feature names from training data.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialise Bernoulli Naive Bayes classifier.

        Args:
            alpha: Smoothing parameter for Laplace smoothing (default=1.0).
        """
        self.alpha = alpha
        self.priors_ = None
        self.likelihoods_ = None
        self.classes_ = None
        self.feature_names_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Fit the model to the training data.

        Args:
            X: Training data (integer).
            y: Target values (class labels in integer).

        Computes:
            - Class prior probabilities (priors_).
            - Feature likelihood probabilities (likelihoods_).
        """
        # Validate binary features
        if not X.isin([0, 1]).all().all():
            raise ValueError("All features must be binary (0/1)")

        self.classes_ = np.unique(y)
        self.feature_names_ = X.columns.to_list()
        self.priors_ = self._calculate_priors(y)
        self.likelihoods_ = self._calculate_likelihoods(X, y)

    def predict(self, X: pd.DataFrame) -> list[int]:
        """
        Predict class labels using Categorical Naive Bayes.

        Args:
            X: Feature matrix.

        Returns:
            Predicted class labels.

        Raises:
            ValueError: If model hasn't been fitted.
        """
        if self.priors_ is None or self.likelihoods_ is None:
            raise ValueError("Model not fitted. Call .fit() first.")

        predictions = []
        for row in X.itertuples(index=False):
            log_posteriors = self._calculate_posteriors(row._asdict())
            predictions.append(max(log_posteriors, key=log_posteriors.get))
        return predictions

    def _calculate_priors(self, y: pd.Series) -> dict[int, float]:
        """
        Calculate prior probabilities for each class in the target variable.

        Args:
            y: Target variable containing class labels (0/1).

        Returns:
            Prior probabilities for each class.
        """
        return y.value_counts(normalize=True).to_dict()

    def _calculate_likelihoods(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, dict[int, dict[int, float]]]:
        """
        Calculate conditional probabilities for feature values given each class.

        Args:
            X: Feature matrix (DataFrame with binary columns)
            y: Target variable (Series of binary class labels)

        Returns:
            Nested dictionary with structure:
            {feature_name: {class_label: {feature_value: probability}}}
        """
        likelihoods = {}

        for feature in self.feature_names_:
            likelihoods[feature] = {}

            for class_label in self.classes_:
                c = int(class_label)
                class_mask = y == c
                class_subset = X.loc[class_mask, feature]
                total_in_class = class_mask.sum()  # Number of samples in class

                # Count occurrences of 1s (0s will be total - count_1)
                count_1 = class_subset.sum()
                count_0 = total_in_class - count_1

                # Apply Laplace smoothing for binary features
                # Denominator: total_in_class + 2 * alpha (for two possible values)
                prob_1 = (count_1 + self.alpha) / (total_in_class + 2 * self.alpha)
                prob_0 = (count_0 + self.alpha) / (total_in_class + 2 * self.alpha)

                # Store probabilities for both values
                likelihoods[feature][c] = {
                    0: round(float(prob_0), 4),
                    1: round(float(prob_1), 4),
                }

        return likelihoods

    def _calculate_posteriors(self, x: dict[str, int]) -> dict[int, float]:
        """
        Calculate log-posterior probabilities for all classes given a sample.
        Args:
            x: Input sample as dictionary {feature: value}.

        Returns:
            Dictionary mapping each class to its log-posterior probability.
        """
        log_posteriors = {}
        for c in self.classes_:
            log_proba = np.log(self.priors_[c])  # Log of prior
            for feature in self.feature_names_:  # Sum of the likelihood for x given c
                category = x[feature]
                # Avoid log(0) if the feature does not exist
                proba = self.likelihoods_[feature][c].get(category, 1e-9)
                log_proba += np.log(proba)
            log_posteriors[c] = round(log_proba, 4)
        return log_posteriors  # log-posterior probabilities for all classes
