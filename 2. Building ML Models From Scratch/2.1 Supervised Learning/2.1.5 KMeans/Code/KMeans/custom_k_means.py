import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Optional
HistoryType = List[Dict[str, NDArray[np.float64] | NDArray[np.int64]]]


class CustomKMeans:
    """
    Custom implementation of the K-Means clustering algorithm with vectorised operations.

    Attributes:
        - n_clusters: Number of clusters.
        - max_iters: Maximum number of iterations.
        - tol: Convergence tolerance for centroid shifts.
        - random_state: Seed for random number generator.
        - disp_conv: Whether to display convergence messages.
        - centroids: Final cluster centroids.
        - labels: Cluster labels for input data.
        - history: History of centroids and labels during fitting.
        - n_iter_: Number of iterations performed.
    """

    def __init__(self, n_clusters: int, max_iters: int = 100, tol: float = 1e-4, random_state: int = 42, disp_conv=False):
        """
        Initialise the K-Means model with specified parameters.

        Args:
            - n_clusters: Number of clusters.
            - max_iters: Maximum number of iterations. Default is 100.
            - tol: Convergence tolerance for centroid shifts. Default is 1e-4.
            - random_state: Seed for random number generator. Default is 42.
            - disp_conv: Whether to display convergence messages. Default is False.
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.centroids: Optional[NDArray[np.float64]] = None
        self.labels: Optional[NDArray[np.int64]] = None
        self.history: Optional[HistoryType] = None
        self.disp_conv = disp_conv
        self.n_iter_ = 0

    def _calculate_euclidean(self, X: NDArray[np.float64], centroids: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Calculate the distance between each data point and cluster centroid.

        Args:
            - X: Data points, shape (n_samples, n_features).
            - centroids: Current centroids, shape (n_clusters, n_features).

        Returns:
            - Distance matrix, shape (n_samples, n_clusters).
        """
        return np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

    def _initialise_centroids(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Randomly initialise cluster centroids using the data points.

        Args:
            - X: Data points, shape (n_samples, n_features).

        Returns:
            - Initialised centroids, shape (n_clusters, n_features).
        """
        return self.rng.choice(X, size=self.n_clusters, replace=False, axis=0)

    def _update_centroids(self, X: NDArray[np.float64], labels: NDArray[np.int64]) -> NDArray[np.float64]:
        """
        Update the centroids based on the mean of data points in each cluster.

        Args:
            - X: Data points, shape (n_samples, n_features).
            - labels: Cluster assignments, shape (n_samples,).

        Returns:
            - Updated centroids, shape (n_clusters, n_features).
        """
        new_centroids = np.empty_like(self.centroids)
        for i in range(self.n_clusters):
            cluster_points = X[labels == i]
            # Avoid failure when clusters become empty
            if len(cluster_points) > 0:
                new_centroids[i] = cluster_points.mean(axis=0)
            else:
                # Maintain previous position
                new_centroids[i] = self.centroids[i]
        return new_centroids

    def _is_converged(self, old: NDArray[np.float64], new: NDArray[np.float64]) -> bool:
        """
        Check if centroids have converged based on the specified tolerance.

        Args:
            - old: Centroids from the previous iteration, shape (n_clusters, n_features).
            - new: Current centroids, shape (n_clusters, n_features).

        Returns:
            - True if the centroids have converged, False otherwise.
        """
        return np.all(np.linalg.norm(new - old, axis=1) < self.tol)

    def fit(self, X: NDArray[np.float64]):
        """
        Fit the K-Means model to the data.

        Args:
            - X: Data points, shape (n_samples, n_features).

        Returns:
            - Fitted KMeans instance.
        """
        self.centroids = self._initialise_centroids(X)
        self.history = [{'centroids': self.centroids.copy(), 'labels': None}]

        for self.n_iter_ in range(1, self.max_iters + 1):
            distances = self._calculate_euclidean(X, self.centroids)
            labels = np.argmin(distances, axis=1)
            new_centroids = self._update_centroids(X, labels)

            # Store both centroids AND labels at each iteration
            self.history.append({
                'centroids': new_centroids.copy(),
                'labels': labels.copy()
            })

            if self._is_converged(self.centroids, new_centroids):
                if self.disp_conv:
                    print(f'Converged after {self.n_iter_} iterations.')
                break

            self.centroids = new_centroids
        self.labels = labels
        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.int64]:
        """
        Assign cluster labels to new data points.

        Args:
            - X: Data points to cluster, shape (n_samples, n_features).

        Returns:
            - Assigned cluster labels, shape (n_samples,).
        """
        if self.centroids is None:
            raise ValueError('Model not fitted. Call fit() first.')
        return np.argmin(self._calculate_euclidean(X, self.centroids), axis=1)
