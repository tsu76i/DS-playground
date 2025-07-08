from sklearn.datasets import make_blobs
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from helper_funcs import HelperFuncs
from custom_dbscan import CustomDBSCAN
from dbscan_metrics import DBSCANMetrics


def main():
    # Load data
    X, y = make_blobs(
        n_samples=500,
        centers=3,
        cluster_std=1.4,
        #   c=y, cmap="viridis",
        n_features=2,
        random_state=42,
    )

    # Normalise X
    scaler = MinMaxScaler()
    X = pd.DataFrame(scaler.fit_transform(X))

    # Find optimal params
    epsilon, min_pts = HelperFuncs.find_optimal_params(X)
    print(f"Optimal parameters: eps = {epsilon:.4f}, min_pts = {min_pts}")

    # Initialise with parameters
    dbscan = CustomDBSCAN(epsilon=epsilon, min_pts=min_pts)

    # Perform clustering
    predicted_labels = dbscan.fit_predict(X)
    X["cluster_label"] = predicted_labels

    # Visualise clusters
    HelperFuncs.visualise_clusters(X)

    # Evaluate
    metrics = DBSCANMetrics.evaluate(X, predicted_labels)
    print("Evaluation Metrics:")
    print(f" - Silhouette Score (Custom): {metrics['silhouette_score']:.3f}")
    print(f" - Noise Ratio (Custom): {metrics['noise_ratio']:.1%}")
    print(f" - Clusters (Custom): {metrics['n_clusters']}")


if __name__ == "__main__":
    main()
