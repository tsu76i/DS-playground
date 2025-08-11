import numpy as np
import pandas as pd


class CustomCategoricalNB:
    """
    Categorical Naive Bayes classifier for discrete features.

    This implementation handles categorical features directly without requiring label encoding.
    Uses Laplace smoothing to handle unseen feature values during prediction.

    Attributes:
        alpha (float): Smoothing parameter (default=1.0).
        priors_ (dict[str, float]): Class prior probabilities.
        likelihoods_ (dict[str, dict[str, dict[str, float]]]): Feature likelihood probabilities.
        classes_ (NDArray[np.str_]): Unique class labels.
        feature_names_ (list[str]): Feature names from training data.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialise the Categorical Naive Bayes classifier.

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
            X: Training data (categorical).
            y: Target values (class labels).

        Computes:
            - Class prior probabilities (priors_).
            - Feature likelihood probabilities (likelihoods_).
        """
        self.classes_ = np.unique(y)
        self.feature_names_ = X.columns.to_list()
        self.priors_ = self._calculate_priors(y)
        self.likelihoods_ = self._calculate_likelihoods(X, y)

    def predict(self, X: pd.DataFrame) -> list[str]:
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

    def _calculate_priors(self, y: pd.Series) -> dict[str, float]:
        """
        Calculate prior probabilities for each class in the target variable.

        Args:
            y: Target variable containing class labels (strings).

        Returns:
            Prior probabilities for each class.
        """
        return y.value_counts(normalize=True).to_dict()

    def _calculate_likelihoods(
        self, X: pd.DataFrame, y: pd.Series
    ) -> dict[str, dict[str, dict[str, float]]]:
        """
        Calculate conditional probabilities for feature values given each class.

        Args:
            X: Feature matrix (DataFrame with categorical columns)
            y: Target variable (Series of class labels)

        Returns:
            Nested dictionary with structure:
            {feature_name: {class_label: {feature_value: probability}}}
        """
        likelihoods = {}
        for feature in self.feature_names_:  # For each column of X
            likelihoods[feature] = {}

            # Unique feature values in each column
            unique_features = X[feature].unique()

            for c in self.classes_:  # Unique target values of y
                class_subset = X[y == c]
                total = len(class_subset)  # Count(C)

                # Count frequencies (e.g., {'Sunny':3, 'Rain':2} for class 'No')
                value_counts = class_subset[feature].value_counts()

                # All features values are included, even if missing in subset
                value_counts = value_counts.reindex(unique_features, fill_value=0)

                probas = round(
                    (value_counts + self.alpha)
                    / (total + len(unique_features) + self.alpha),
                    4,
                )

                likelihoods[feature][c] = probas.to_dict()

        return likelihoods

    def _calculate_posteriors(self, x: dict[str, str]) -> dict[str, float]:
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
