from numpy.typing import NDArray
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from typing import List, Dict, Tuple
HistoryType = List[Dict[str, NDArray[np.float64] | NDArray[np.int64]]]


class HelperFuncs:
    def train_test_split(X: NDArray, y: NDArray, test_size: float = 0.2,
                         random_state: int = None) -> Tuple[NDArray, NDArray, NDArray, NDArray]:
        """
        Split arrays or matrices into random train and test subsets.

        Args:
            X: Input features, a 2D array with rows (samples) and columns (features).
            y: Target values/labels, a 1D array with rows (samples).
            test_size: Proportion of the dataset to include in the test split. Must be between 0.0 and 1.0. default = 0.2
            random_state: Seed for the random number generator to ensure reproducible results. default = None

        Returns:
            A tuple containing:
                - X_train: Training set features.
                - X_test: Testing set features.
                - y_train: Training set target values.
                - y_test: Testing set target values.
        """
        # Set a random seed if it exists
        if random_state:
            np.random.seed(random_state)

        # Create a list of numbers from 0 to len(X)
        indices = np.arange(len(X))

        # Shuffle the indices
        np.random.shuffle(indices)

        # Define the size of our test data from len(X)
        test_size = int(test_size * len(X))

        # Generate indices for test and train data
        test_indices: NDArray[np.int64] = indices[:test_size]
        train_indices: NDArray[np.int64] = indices[test_size:]

        # Return: X_train, X_test, y_train, y_test
        return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

    def plot_kmeans_clusters(df: pd.DataFrame, features: List[str], clusters_list: List[NDArray[np.int64]],
                             centroids_list: List[NDArray[np.float64]], palette: List[str]) -> None:
        """
        Plot K-Means clustered data with centroids for given features.

        Args:
            - df: DataFrame containing the dataset.
            - features: List of feature names to plot (e.g., ['sepal', 'petal']).
            - clusters_list: List of cluster label arrays, one for each feature set.
            - centroids_list: List of centroid arrays, one for each feature set.
            - palette: Seaborn colour palette for cluster colours.
        """
        colors = palette.as_hex()
        fig, axes = plt.subplots(
            1, len(features), figsize=(6 * len(features), 6))

        # Handle single subplot case
        if len(features) == 1:
            axes = [axes]

        for i, feature in enumerate(features):
            clusters = clusters_list[i]
            centroids = centroids_list[i]

            # Get the appropriate columns for this feature set
            if i == 0:  # First plot (sepal)
                x_col, y_col = df.columns[0], df.columns[1]
            else:  # Second plot (petal)
                x_col, y_col = df.columns[2], df.columns[3]

            # Create scatter plot with cluster colours
            for cluster_id in range(len(centroids)):
                mask = clusters == cluster_id
                axes[i].scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                                c=[colors[cluster_id]], s=60, alpha=0.7,
                                label=f'Cluster {cluster_id}')

            # Plot centroids
            for j, (x, y) in enumerate(centroids):
                axes[i].scatter(x, y, color=colors[j], marker='*', s=150,
                                edgecolor='black', linewidth=1.5,
                                label=f'Centroid {j}')

            axes[i].set_title(f'{feature.title()} Length vs Width (Clustered)')
            axes[i].set_xlabel(f'{feature.title()} Length (cm)')
            axes[i].set_ylabel(f'{feature.title()} Width (cm)')
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_kmeans_transitions(X: NDArray[np.float64], history: HistoryType, title: str,
                                palette: List[str], steps: int) -> None:
        """
        Visualise the progression of K-Means clustering through iterations.

        Args:
            - X: Data points, shape (n_samples, n_features).
            - history: List of dictionaries storing centroids and labels at each iteration.
            - title: Title for the plot.
            - palette: Seaborn colour palette for cluster colours.
            - steps: Number of steps/iterations to visualise.
        """
        colors = palette.as_hex()
        total_iters = len(history)
        indices = np.linspace(0, total_iters - 1, steps, dtype=int)

        fig, axes = plt.subplots(1, steps, figsize=(4 * steps, 5))

        for ax, idx in zip(axes, indices):
            data = history[idx]
            centroids = data['centroids']
            labels = data['labels']

            # When n_iter_ = 0 (before assigning points to clusters)
            if labels is None:
                ax.scatter(X[:, 0], X[:, 1], color='gray', s=40, alpha=0.6)

            # When n_iter_ > 0
            else:
                for j, color in enumerate(colors):
                    points = X[labels == j]
                    ax.scatter(points[:, 0], points[:, 1], color=color,
                               s=40, alpha=0.8, label=f'Cluster {j}')

            for j, (x, y) in enumerate(centroids):
                ax.scatter(x, y, color=colors[j], marker='*', s=150,
                           edgecolor='black', linewidth=1.5, label=f'Centroid {j}')

            ax.set_title(f'Iteration {idx}')
            ax.set_xlabel('Length (cm)')
            ax.set_ylabel('Width (cm)')
            ax.set_xlim(X[:, 0].min() - 0.5, X[:, 0].max() + 0.5)
            ax.set_ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    def plot_elbow_curve(k_values: List[int], wcss_values: List[float]) -> None:
        """
        Plot the elbow curve to determine optimal k.

        Args:
            k_values: List of k values.
            wcss_values: Corresponding WCSS values.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, wcss_values, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, alpha=0.3)

        # Highlight the potential elbow point
        if len(wcss_values) > 2:

            # Second derivative
            d_2 = np.diff(wcss_values, 2)

            # Elbow is at the index with maximum positive second derivative
            # +2 because diff reduces length twice
            elbow_idx = np.argmax(d_2) + 2
            if elbow_idx < len(k_values):
                plt.axvline(x=k_values[elbow_idx], color='red', linestyle='--',
                            label=f'Potential Elbow at k={k_values[elbow_idx]}')
                plt.legend()
            else:
                print('Elbow point could not be reliably determined.')
        else:
            print('Insufficient data points to determine an elbow point.')
        plt.show()

    def plot_evaluation_metrics(k_values: List[int], wcss_list: List[float], bcss_list: List[float], tss_list: List[float]) -> None:
        """
        Plot WCSS, BCSS, and TSS metrics against k values.

        Args:
            k_values: List of k values used for KMeans clustering.
            wcss_list: Within-Cluster Sum of Squares values.
            bcss_list: Between-Cluster Sum of Squares values.
            tss_list: Total Sum of Squares values.
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 3 subplots in 1 row
        metrics_list = wcss_list, bcss_list, tss_list
        metrics_name = ['WCSS', 'BCSS', 'TSS']
        colors = ['blue', 'green', 'red']

        for i, (metric, name) in enumerate(zip(metrics_list, metrics_name)):
            axes[i].plot(k_values, metric, marker='o',
                         label=name, color=colors[i])
            axes[i].set_title(f'{name} vs k')
            axes[i].set_xlabel('Number of Clusters (k)')
            axes[i].set_ylabel(name)
            axes[i].grid(True)
            axes[i].legend()

        plt.tight_layout()
        plt.show()

    def plot_results(X: NDArray[np.float64], labels: List[int], centroids: List[float],
                     title: str, palette: List[str]) -> None:

        # For consistency of cluster colours
        colors = palette.as_hex()
        methods = ['KMeans', 'KMeans++']
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # Iterate over 2 axes and their corresponding cluster results
        for i, ax in enumerate(axes):
            label = labels[i]
            centroid = centroids[i]

            # Scatter data points and centroids
            for cluster in np.unique(label):  # Unique cluster ID
                # Select points in the current cluster
                cluster_mask = (label == cluster)
                cluster_color = colors[cluster]  # Assign a unique colour

                # Data points in the current cluster
                ax.scatter(X[cluster_mask, 0], X[cluster_mask, 1],
                           color=cluster_color, s=30, alpha=0.6, label=f'Cluster {cluster}')

                # Centroid in the current cluster
                ax.scatter(centroid[cluster, 0], centroid[cluster, 1],
                           color=cluster_color, edgecolor='black', marker='*', s=200)

            ax.set_title(f'{methods[i]} Clustering ({title})')
            ax.legend()

        # Adjust layout and display
        plt.tight_layout()
        plt.show()
