import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_iris
from numpy.typing import NDArray
from helper_funcs import HelperFuncs
from kmeans_metrics import KMeansMetrics
from custom_k_means import CustomKMeans

HistoryType = list[dict[str, NDArray[np.float64] | NDArray[np.int64]]]


def main():
    k = 3
    random_seed = 42
    # Load data
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["species"] = iris.target

    # For sepal features
    X_sepal = df.iloc[:, :2].values
    kmeans_sepal = CustomKMeans(n_clusters=k, random_state=random_seed)
    kmeans_sepal.fit(X_sepal)
    clusters_sepal = kmeans_sepal.labels
    centroids_sepal = kmeans_sepal.centroids
    sepal_history = kmeans_sepal.history

    # For petal features
    X_petal = df.iloc[:, 2:4].values
    kmeans_petal = CustomKMeans(n_clusters=k, random_state=random_seed)
    kmeans_petal.fit(X_petal)
    clusters_petal = kmeans_petal.labels
    centroids_petal = kmeans_petal.centroids
    petal_history = kmeans_petal.history

    # Use with your plotting function
    palette = sns.color_palette("bright", n_colors=k)
    features = ["sepal", "petal"]
    clusters_list = [clusters_sepal, clusters_petal]
    centroids_list = [centroids_sepal, centroids_petal]

    HelperFuncs.plot_kmeans_clusters(
        df, features, clusters_list, centroids_list, palette
    )

    # Plot transitions for sepal
    HelperFuncs.plot_kmeans_transitions(
        X_sepal, sepal_history, "KMeans Progress (Sepal Features)", palette, steps=4
    )

    # Plot transitions for petal
    HelperFuncs.plot_kmeans_transitions(
        X_petal, petal_history, "KMeans Progress (Petal Features)", palette, steps=4
    )

    k_values = list(range(1, 11))  # `k` from 1 to 10
    wcss_list, bcss_list, tss_list = [], [], []

    # For sepal features
    for k in k_values:
        kmeans_sepal = CustomKMeans(
            n_clusters=k, random_state=random_seed, disp_conv=False
        )
        kmeans_sepal.fit(X_sepal)
        clusters_sepal = kmeans_sepal.labels
        centroids_sepal = kmeans_sepal.centroids
        sepal_history = kmeans_sepal.history
        metrics = KMeansMetrics(
            X=X_sepal, labels=clusters_sepal, centroids=centroids_sepal
        )
        wcss, bcss, tss = metrics.evaluate()
        wcss_list.append(wcss)
        bcss_list.append(bcss)
        tss_list.append(tss)

    HelperFuncs.plot_elbow_curve(k_values=k_values, wcss_values=wcss_list)
    HelperFuncs.plot_evaluation_metrics(k_values, wcss_list, bcss_list, tss_list)


if __name__ == "__main__":
    main()
