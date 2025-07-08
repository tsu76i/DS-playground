import numpy as np
from numpy.typing import NDArray
from custom_lasso_regression import CustomLassoRegression


class HelperFuncs:
    def standardise(X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Standardise the input array by removing the mean and scaling to unit variance.

        Args:
            X: Input array of numerical features.

        Returns:
            Standardised array with zero mean and unit variance.
        """
        X_mean = np.mean(X)
        X_std = np.std(X)
        return (X - X_mean) / X_std

    def cross_validate_lasso(
        X: NDArray[np.float64],
        y: NDArray[np.float64],
        lambdas: list,
        k: int = 5,
        alpha: float = 0.001,
        epochs: int = 20000,
        random_state: int = 42,
    ) -> dict:
        """
        Perform k-fold cross-validation for CustomLassoRegression over a grid of lambda values.

        Args:
            X: Feature array (1D or 2D).
            y: Target array.
            lambdas: List of lambda values to evaluate.
            k: Number of folds (default 5).
            alpha: Learning rate for gradient descent.
            epochs: Number of training epochs.
            random_state: Seed for reproducibility.

        Returns:
            Dictionary mapping lambda to average validation MSE.
        """
        np.random.seed(random_state)
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        fold_sizes = np.full(k, len(y) // k, dtype=int)  # Split equally
        fold_sizes[: len(y) % k] += 1  # Distibute remainder
        current = 0
        folds = []
        for fold_size in fold_sizes:  # Split randomised indices into k folds
            start, stop = current, current + fold_size
            # folds contain k folds of randomised indices
            folds.append(indices[start:stop])
            current = stop

        lambda_mse = {}
        for lambda_ in lambdas:
            mse_scores = []
            for i in range(k):
                val_idx = folds[i]  # i-th fold = validation
                # All other folds = training
                train_idx = np.hstack([folds[j] for j in range(k) if j != i])
                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]

                model = CustomLassoRegression(
                    w=0, b=0, alpha=alpha, epochs=epochs, lambda_=lambda_
                )
                model.train(X_train, y_train)
                y_pred = model.predict(X_val)
                mse = np.mean((y_val - y_pred) ** 2)
                mse_scores.append(mse)
            lambda_mse[lambda_] = float(np.mean(mse_scores).round(4))
        return lambda_mse
