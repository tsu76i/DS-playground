import numpy as np
import seaborn as sns
from numpy.typing import NDArray
from typing import List, Dict
from sklearn.datasets import make_blobs, make_circles
from custom_k_means import CustomKMeans
from custom_k_means_p2 import CustomKMeansP2
from helper_funcs import HelperFuncs
HistoryType = List[Dict[str, NDArray[np.float64] | NDArray[np.int64]]]


def main():
    k = 4
    palette = sns.color_palette('bright', n_colors=k)

    # 1. Normal dataset
    X1, _ = make_blobs(n_samples=200, centers=4, cluster_std=[
        1.0, 2.5, 0.5, 2.8], random_state=42)
    # 2. Various Cluster Size and Densities
    X2, _ = make_blobs(n_samples=[100, 200, 800], centers=[[0, 0], [10, 10], [
        5, -5]], cluster_std=[0.5, 2.5, 0.8], random_state=42)
    # 3. Outliers
    X3, _ = make_blobs(n_samples=200, centers=3,
                       cluster_std=1.0, random_state=42)
    rng = np.random.RandomState(46)
    X3 = np.vstack([X3, rng.uniform(
        low=-10, high=10, size=(40, 2))])  # Add outliers
    # 4. Non-Spherical
    X4, _ = make_circles(
        n_samples=500, factor=0.8, noise=0.1, random_state=55)
    datasets = [X1, X2, X3, X4]
    datasets_types = [
        'Normal', 'Various Cluster Size and Densities', 'Outliers', 'Non-Spherical']

    for i, dataset in enumerate(datasets):
        print("*"*20)
        print(f"{i+1}. {datasets_types[i]}")
        # Instantiate and fit model (Custom KMeans)
        kmeans = CustomKMeans(n_clusters=k, disp_conv=True)
        kmeans.fit(dataset)

        # Instantiate and fit model (Custom KMeansP2)
        kmeans_p2 = CustomKMeansP2(n_clusters=k, disp_conv=True)
        kmeans_p2.fit(dataset)

        # Store results to lists
        labels_list = [kmeans.labels, kmeans_p2.labels]
        centroids_list = [kmeans.centroids, kmeans_p2.centroids]

        # Visualise
        HelperFuncs.plot_results(
            dataset, labels_list, centroids_list, 'Normal Dataset', palette)


if __name__ == '__main__':
    main()
