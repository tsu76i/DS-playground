import pandas as pd
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import List


class CustomDBSCAN:
    """
    Custom implementation of the DBSCAN clustering algorithm.

    Attributes:
        epsilon (float): The radius of the neighborhood around each point.
        min_pts (int): The minimum number of points required to form a dense region.
        labels_ (NDArray[np.int_]): Cluster labels for each point after fitting.
    """

    def __init__(self, epsilon: float, min_pts: int) -> None:
        """
        Initialise the DBSCAN object.

        Args:
            epsilon (float): Neighborhood radius.
            min_pts (int): Minimum number of points to form a core point.
        """
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.labels_ = None  # Cluster labels will be stored here after fitting

    def _find_neighbours(self, data: NDArray[np.float64], point_idx: int) -> List[int]:
        """
        Find neighbours within epsilon distance of a point.

        Args:
            data: Dataset as a NumPy array of shape (n_samples, n_features).
            point_idx: Index of the target point.

        Returns:
            Indices of neighbours within epsilon distance.
        """
        distances = np.linalg.norm(
            data - data[point_idx], axis=1)  # Vectorised
        # Get indices where distance <= epsilon
        return np.where(distances <= self.epsilon)[0].tolist()

    def _expand_cluster(self, data: NDArray[np.float64], labels: NDArray[np.int16],
                        point_idx: int, neighbours: List[int], cluster_id: int) -> None:
        """
        Expand the cluster from a core point using density reachability.

        Args:
            data: Dataset as a NumPy array.
            labels: Array tracking point states (0=unvisited, -1=noise).
            point_idx: Index of the starting core point.
            neighbours: Initial neighbor indices.
            cluster_id: Current cluster ID to assign.
        """
        labels[point_idx] = cluster_id
        i = 0
        while i < len(neighbours):
            neighbour_idx = neighbours[i]

            if labels[neighbour_idx] == -1:  # Noise -> Border
                labels[neighbour_idx] = cluster_id
            elif labels[neighbour_idx] == 0:  # Unvisited
                labels[neighbour_idx] = cluster_id
                new_neighbours = self._find_neighbours(data, neighbour_idx)
                if len(new_neighbours) >= self.min_pts:  # Core point
                    neighbours += [n for n in new_neighbours if n not in neighbours]
            i += 1

    def fit_predict(self, data: pd.DataFrame) -> NDArray[np.int16]:
        """
        Performs DBSCAN clustering and returns cluster labels.

        Args:
            data: Input data with shape (n_samples, n_features).

        Returns:
            NDArray[np.int16]: Cluster labels (-1 for noise, >0 for cluster IDs).
        """
        data_np = data.values
        n = len(data_np)
        labels = np.zeros(n, dtype=int)  # Initialise with all 0s.
        cluster_id = 0

        for i in tqdm(range(n), desc='Clustering'):
            if labels[i] != 0:  # Skip if already processed
                continue

            neighbours = self._find_neighbours(data_np, i)

            if len(neighbours) < self.min_pts:
                labels[i] = -1  # Mark as noise
            else:
                cluster_id += 1  # New cluster
                self._expand_cluster(data_np, labels, i,
                                     neighbours, cluster_id)
        self.labels_ = labels
        return labels
