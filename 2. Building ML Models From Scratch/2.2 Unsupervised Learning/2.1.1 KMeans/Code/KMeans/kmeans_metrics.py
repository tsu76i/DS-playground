import numpy as np
from numpy.typing import NDArray
from typing import List, Dict, Tuple
HistoryType = List[Dict[str, NDArray[np.float64] | NDArray[np.int64]]]


class KMeansMetrics:
    def __init__(self, X: NDArray[np.float64], labels: NDArray[np.int64], centroids: NDArray[np.float64]) -> None:
        """
        Initialise the ClassificationMetrics instance.

        Args:
            X: Data points, shape (n_samples, n_features).
            labels: Cluster assignments, shape (n_samples,).
            centroids: Cluster centers, shape (k, n_features).
        """
        self.X = X
        self.labels = labels
        self.centroids = centroids

    def calculate_wcss(self) -> float:
        """
        Calculate Within-Cluster Sum of Squares (WCSS).

        Returns:
            Within-cluster sum of squares (WCSS).
        """
        wcss = 0
        for i in range(len(self.centroids)):
            # Get all points assigned to cluster i
            cluster_points = self.X[self.labels == i]
            if len(cluster_points) > 0:
                # Sum of squared distances from points to their centroid
                wcss += np.sum((cluster_points - self.centroids[i]) ** 2)
        return wcss

    def calculate_bcss(self) -> float:
        """
        Calculate Between-Cluster Sum of Squares (BCSS).

        Returns:
            Between-cluster sum of squares (BCSS).
        """
        overall_mean = np.mean(self.X, axis=0)
        bcss = 0

        for i in range(len(self.centroids)):
            # Number of points in cluster i
            n_i = np.sum(self.labels == i)
            if n_i > 0:
                # Add weighted squared distance from centroid to overall mean
                bcss += n_i * np.sum((self.centroids[i] - overall_mean) ** 2)
        return bcss

    def calculate_total_ss(self) -> float:
        """
        Calculate Total Sum of Squares (TSS).

        Returns:
            Total sum of squares (TSS).
        """
        overall_mean = np.mean(self.X, axis=0)
        return np.sum((self.X - overall_mean) ** 2)

    def evaluate(self) -> Tuple[float, float, float]:
        """
        Evaluate WCSS, BCSS, and TSS metrics.

        Returns:
            WCSS, BCSS, and TSS metrics.
        """
        wcss = self.calculate_wcss()
        bcss = self.calculate_bcss()
        tss = self.calculate_total_ss()
        return wcss, bcss, tss
