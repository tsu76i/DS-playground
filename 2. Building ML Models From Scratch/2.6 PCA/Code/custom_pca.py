import numpy as np
from numpy.typing import NDArray


class CustomPCA:
    """
    Custom implementation of Principal Component Analysis (PCA) for dimensionality reduction.

    Attributes:
        n_components (int): Number of principal components to retain.
        components (NDArray[np.float64]): Principal axes in feature space.
        mean (NDArray[np.float64]): Per-feature mean, estimated from the training set.
        eigenvalues (NDArray[np.float64]): Eigenvalues corresponding to the selected components.
        all_eigenvalues (NDArray[np.float64]): All eigenvalues from the covariance matrix.
    """

    def __init__(self, n_components: int):
        """
        Initialise the CustomPCA object.

        Args:
            n_components (int): Number of principal components to retain.
        """
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.all_eigenvalues = None

    def fit(self, X_standardised: NDArray[np.float64]) -> None:
        """
        Fit the PCA model to the data.

        Computes the covariance matrix, performs eigen decomposition,
        and selects the top n_components principal components.

        Args:
            X_standardised (NDArray[np.float64]): Standardised input features of shape (n_samples, n_features).
        """
        # Calculate the covariance matrix
        cov_matrix = np.cov(X_standardised, rowvar=False)
        # Perform eigen decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # For explained variance
        self.all_eigenvalues = eigenvalues

        # Sort eigenvalues and eigenvectors in descending order
        sorted_idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues = eigenvalues[sorted_idx][:self.n_components]
        self.components = eigenvectors[:, sorted_idx][:, :self.n_components]

    def transform(self, X_standardised: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Project the data onto the principal components.

        Args:
            X_standardised (NDArray[np.float64]): Standardised input features of shape (n_samples, n_features).

        Returns:
            NDArray[np.float64]: Transformed data of shape (n_samples, n_components).
        """
        if self.components is None:
            raise ValueError("Fit the PCA model before transforming data.")
        return np.dot(X_standardised, self.components)

    @property
    def explained_variance_ratio(self) -> NDArray[np.float64]:
        """
        The proportion of variance explained by each principal component.

        Returns:
            NDArray[np.float64]: Explained variance ratio for each component.
        """
        if self.eigenvalues is None:
            raise ValueError(
                "Fit the model before accessing explained_variance_ratio.")
        return self.eigenvalues / np.sum(self.all_eigenvalues)

    @property
    def cumulative_explained_variance(self) -> NDArray[np.float64]:
        """
        The cumulative sum of explained variance ratios.

        Returns:
            NDArray[np.float64]: Cumulative explained variance ratio.
        """
        if self.eigenvalues is None:
            raise ValueError(
                "Fit the model before accessing cumulative_explained_variance.")
        return np.cumsum(self.explained_variance_ratio)
