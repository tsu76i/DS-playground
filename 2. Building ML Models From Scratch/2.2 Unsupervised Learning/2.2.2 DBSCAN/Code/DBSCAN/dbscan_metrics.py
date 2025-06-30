import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict


class DBSCANMetrics:
    def silhouette_score(X: NDArray[np.float64], labels: List[int]) -> np.float64:
        """
        Compute silhouette score for clustering results (excluding noise points).

        Args:
            X: Input data array.
            labels: Cluster labels (-1 indicates noise).

        Returns:
            Mean silhouette score for non-noise points, or NaN if < 2 valid clusters.
        """
        unique_labels = np.unique(
            [label for label in labels if label != -1])  # without -1
        n_samples = len(X)
        silhouette_vals = np.zeros(n_samples)

        if len(unique_labels) < 2:
            return float('nan')  # Silhouette score undefined for < 2 clusters

        distance_matrix = np.linalg.norm(
            X[:, np.newaxis] - X[np.newaxis, :], axis=2)

        cluster_masks = {cluster: (labels == cluster)
                         for cluster in unique_labels}

        for i in range(n_samples):
            if labels[i] == -1:
                silhouette_vals[i] = 0  # Noise points have 0 silhouette value
                continue

            own_cluster = labels[i]
            is_same_cluster = cluster_masks[own_cluster].copy()
            is_same_cluster[i] = False  # Excluding self

            # Calculate a_i
            if np.any(is_same_cluster):
                a_i = np.mean(distance_matrix[i, is_same_cluster])
            else:
                a_i = 0

            # Calculate b_i
            b_i = np.inf
            for cluster in unique_labels:
                if cluster == own_cluster:  # Skip own cluster
                    continue
                other_mask = cluster_masks[cluster]
                if np.any(other_mask):
                    distance = np.mean(distance_matrix[i, other_mask])
                    if distance < b_i:
                        b_i = distance

            # Calculate silhouette for current point
            silhouette_vals[i] = (b_i - a_i) / \
                max(a_i, b_i) if max(a_i, b_i) > 0 else 0

        valid_points = labels != -1
        if not np.any(valid_points):
            return float('nan')
        return np.mean(silhouette_vals[valid_points]).round(4)

    def noise_ratio_and_cluster_count(labels: List[int]) -> Tuple[float, int]:
        """
        Calculate noise ratio and cluster count for DBSCAN results.

        Args:
            labels: Cluster labels from DBSCAN (-1 = noise)

        Returns:
            Tuple containing:
            - noise_ratio: Proportion of noise points (0.0 to 1.0)
            - n_clusters: Number of clusters (excluding noise)
        """
        labels = np.array(labels)
        is_noise = (labels == -1)
        noise_ratio = float(np.mean(is_noise))

        unique_labels = np.unique(
            [label for label in labels if label != -1])  # without -1
        n_clusters = len(unique_labels)
        return noise_ratio, n_clusters

    def evaluate(X: pd.DataFrame, labels: List[int]) -> Dict[str, float | int]:
        """
        Evaluate DBSCAN clustering results using key metrics.

        Args:
            X: Input data DataFrame.
            labels: Cluster labels from DBSCAN.

        Returns:
            Dictionary containing:
            - silhouette_score: Mean silhouette score
            - noise_ratio: Proportion of noise points
            - n_clusters: Number of clusters
        """
        # Convert to NumPy array if DataFrame
        if not isinstance(X, np.ndarray):
            X = X.values
        silhouette = DBSCANMetrics.silhouette_score(X, labels)
        noise_ratio, n_clusters = DBSCANMetrics.noise_ratio_and_cluster_count(
            labels)
        return {
            'silhouette_score': silhouette,
            'noise_ratio': noise_ratio,
            'n_clusters': n_clusters
        }
