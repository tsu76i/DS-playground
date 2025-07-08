import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class HelperFuncs:
    def compute_k_distances(X: NDArray[np.float64], k: int) -> NDArray[np.float64]:
        """
        Compute distances to the k-th nearest neighbour for each point.

        Args:
            X: Input data, features.
            k: The neighbour's index to compute (k-th nearest neighbour)

        Returns:
            Array of distances to the k-th nearest neighbour for each point.
        """
        n = X.shape[0]
        k_distances = np.zeros(n)
        for i in range(n):
            # Calculate Euclidean distances from point i to all others
            distances = np.linalg.norm(X - X[i], axis=1)
            # Sort distances and select the k-th smallest (exclude self)
            sorted_dist = np.sort(distances)
            k_distances[i] = sorted_dist[k - 1]  # k-th neighbour (index k)
        return k_distances

    def find_optimal_params(X: pd.DataFrame) -> Tuple[float, int]:
        """
        Determine optimal DBSCAN parameters using the k-distance elbow method.
        Also plot a k-distance graph.

        Args:
            X: Input data, features.

        Returns:
            A tuple that contains optimal epsilon and min_pts.
        """
        # Compute k-distances
        X = X.values
        dimensions = X.shape[1]
        min_pts = dimensions * 2
        k_distances = HelperFuncs.compute_k_distances(X, min_pts)
        sorted_k_distances = np.sort(k_distances)

        # Plot k-distance graph
        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(len(sorted_k_distances)), sorted_k_distances, "b-")
        plt.xlabel("Points sorted by k-distance")
        plt.ylabel(f"Distance to {min_pts}-th nearest neighbor")
        plt.title("K-Distance Graph for Parameter Selection")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Find elbow point (heuristic: max curvature)
        slopes = np.diff(sorted_k_distances)
        drop = slopes[:-1] - slopes[1:]
        elbow_idx = np.argmax(drop) + 1  # +1 to adjust index
        epsilon = sorted_k_distances[elbow_idx]
        return epsilon, min_pts

    def visualise_clusters(X: pd.DataFrame) -> None:
        """
        Visualise DBSCAN clustering results showing clusters and noise points.

        Args:
            X: DataFrame containing 'cluster_label' column and at least two features.
        """
        for label in sorted(X["cluster_label"].unique()):
            if label == -1:  # Noise points
                plt.scatter(
                    X.loc[X["cluster_label"] == -1, X.columns[0]],
                    X.loc[X["cluster_label"] == -1, X.columns[1]],
                    s=30,
                    c="black",
                    label="Noise",
                )
            else:  # Cluster points
                plt.scatter(
                    X.loc[X["cluster_label"] == label, X.columns[0]],
                    X.loc[X["cluster_label"] == label, X.columns[1]],
                    s=30,
                    label=f"Cluster {label}",
                )

        plt.title("Clusters with Noise")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()
